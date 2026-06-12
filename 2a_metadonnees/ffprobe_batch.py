#!/usr/bin/env python3
"""
Enrichissement technique du corpus via ffprobe.
Extrait les métadonnées (durée, résolution, bitrate, fps, audio)
de chaque fichier média et les ajoute au JSONL.

Champs produits : duree, largeur, hauteur, orientation, fps,
video_bitrate, audio_present, fichier_taille.

Pipeline :
  1. Filtrage — messages avec media_chemin non traités
  2. ffprobe — extraction JSON brut (timeout 10s)
  3. Parsing — normalisation en champs JSONL (vidéo / photo / audio)
  4. Écriture — enrichissement JSONL + fiche individuelle

Options CLI : --input, --output, --media-dir, --limit, --overwrite, --ids,
              --start-date, --end-date, --config

Dépendance unique : ffprobe (installé avec ffmpeg).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    write_jsonl,
    mettre_a_jour_fiche,
    filtrer_eligibles,
    SuiviProgression,
)


# ── ffprobe ──────────────────────────────────────────────────────────────────

def executer_ffprobe(file_path: str, timeout: int = 10) -> dict | None:
    """Lance ffprobe en mode JSON et retourne le dict brut.

    Entrée : file_path — chemin vers le fichier média, timeout — secondes avant abandon
    Sortie : dict {"streams": [...], "format": {...}}, ou None si erreur/timeout
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def etiquette_orientation(w: int, h: int) -> str:
    """Retourne l'orientation à partir des dimensions.

    Entrée : w — largeur en pixels, h — hauteur en pixels
    Sortie : "vertical" | "horizontal" | "square"
    """
    if h > w:
        return "vertical"
    if w > h:
        return "horizontal"
    if w == h:
        return "square"


def parse_fps(stream: dict) -> float | None:
    """Extrait le fps depuis r_frame_rate ou avg_frame_rate (format 'num/den')."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = stream.get(key)
        if not raw:
            continue
        parts = raw.split("/")
        if len(parts) == 2:
            try:
                num, den = int(parts[0]), int(parts[1])
                if den > 0:
                    return round(num / den, 2)
            except ValueError:
                continue
    return None


def extraire_metadonnees(probe: dict) -> dict:
    """Parse la sortie ffprobe et retourne les champs à ajouter au JSONL.

    Gère trois cas :
    - Image (codec photo sans audio) → largeur, hauteur, orientation, fichier_taille
    - Vidéo → duree, largeur, hauteur, orientation, video_bitrate, fps, audio_present, fichier_taille
    - Audio seul → duree, audio_present, fichier_taille

    Entrée : probe — dict brut retourné par executer_ffprobe()
    Sortie : dict des champs à ajouter au message (vide si aucun stream exploitable)
    """
    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    if not streams:
        return {}

    video_stream = None
    audio_stream = None
    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = s
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = s

    # Taille du fichier
    fichier_taille = None
    raw_size = fmt.get("size")
    if raw_size is not None:
        try:
            fichier_taille = int(raw_size)
        except (ValueError, TypeError):
            pass

    # ── Photo (pas de vidéo animée, pas d'audio, codec image) ──
    image_codecs = {"mjpeg", "png", "bmp", "tiff", "webp", "gif", "jpegls"}
    is_image = (
        video_stream is not None
        and audio_stream is None
        and video_stream.get("codec_name", "").lower() in image_codecs
    )

    if is_image:
        result = {}
        w = int(video_stream.get("width", 0))
        h = int(video_stream.get("height", 0))
        if w > 0 and h > 0:
            result["largeur"] = w
            result["hauteur"] = h
            result["orientation"] = etiquette_orientation(w, h)
        if fichier_taille is not None:
            result["fichier_taille"] = fichier_taille
        return result

    # ── Vidéo ──
    if video_stream is not None:
        w = int(video_stream.get("width", 0))
        h = int(video_stream.get("height", 0))

        # Durée : préférer format.duration, fallback sur stream
        duree = None
        for src in (fmt.get("duration"), video_stream.get("duration")):
            if src is not None:
                try:
                    duree = round(float(src), 2)
                    break
                except (ValueError, TypeError):
                    pass

        # Bitrate vidéo en kbps
        video_bitrate = None
        raw_br = video_stream.get("bit_rate")
        if raw_br is not None:
            try:
                video_bitrate = round(int(raw_br) / 1000)
            except (ValueError, TypeError):
                pass

        result = {}
        if duree is not None:
            result["duree"] = duree
        if w > 0 and h > 0:
            result["largeur"] = w
            result["hauteur"] = h
            result["orientation"] = etiquette_orientation(w, h)
        if video_bitrate is not None:
            result["video_bitrate"] = video_bitrate
        result["fps"] = parse_fps(video_stream)
        result["audio_present"] = audio_stream is not None
        if fichier_taille is not None:
            result["fichier_taille"] = fichier_taille

        return result

    # ── Audio seul (pas de stream vidéo) ──
    if audio_stream is not None:
        duree = None
        for src in (fmt.get("duration"), audio_stream.get("duration")):
            if src is not None:
                try:
                    duree = round(float(src), 2)
                    break
                except (ValueError, TypeError):
                    pass
        result = {}
        if duree is not None:
            result["duree"] = duree
        result["audio_present"] = True
        if fichier_taille is not None:
            result["fichier_taille"] = fichier_taille
        return result

    return {}


# ── Batch ────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Enrichissement technique via ffprobe",
        has_media_dir=True,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger("ffprobe", cfg=cfg)
    save_every = 50

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else input_path.parent

    # ── Vérification ffprobe ──
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        log.error("ffprobe introuvable. Installez ffmpeg.")
        sys.exit(1)

    # ── Lecture du JSONL ──
    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    messages = read_jsonl(input_path)
    total = len(messages)
    log.info(f"Corpus : {total} messages")

    # ── Filtrage ──
    filtre_ids = set(args.ids) if args.ids else None
    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        champs_a_verifier=["duree", "fichier_taille"],
        overwrite=args.overwrite,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Filtrer davantage : il faut un media_chemin
    eligibles = [i for i in eligibles if messages[i].get("media_chemin")]

    n_eligibles = len(eligibles)
    log.info(f"Messages éligibles : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    tracker = SuiviProgression(n_eligibles, label="ffprobe")
    n_traites = 0
    n_erreurs = 0

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            media_path_rel = msg.get("media_chemin")

            fichier_media = media_dir / media_path_rel
            basename = os.path.basename(media_path_rel)

            if not fichier_media.is_file():
                log.warning(f"msg {mid}\t{media_path_rel}\tfichier manquant")
                n_erreurs += 1
                tracker.avancer(rank, mid, "fichier manquant ✗")
                continue

            # ffprobe
            probe = executer_ffprobe(str(fichier_media))
            if probe is None:
                log.warning(f"msg {mid}\t{media_path_rel}\tffprobe échoué")
                n_erreurs += 1
                tracker.avancer(rank, mid, "ffprobe échoué ✗")
                continue

            meta = extraire_metadonnees(probe)
            if not meta:
                log.warning(f"msg {mid}\t{media_path_rel}\taucun stream exploitable")
                n_erreurs += 1
                tracker.avancer(rank, mid, "aucun stream ✗")
                continue

            # Enrichir le message
            msg.update(meta)
            n_traites += 1

            # Fiche individuelle
            mettre_a_jour_fiche(msg, meta, media_dir / "fiches", overwrite=args.overwrite)

            # Progression
            parties_desc = []
            if "duree" in meta:
                parties_desc.append(f"{meta['duree']}s")
            if "largeur" in meta and "hauteur" in meta:
                parties_desc.append(f"{meta['largeur']}x{meta['hauteur']}")
            if meta.get("audio_present"):
                parties_desc.append("audio")
            tracker.avancer(rank, mid, ", ".join(parties_desc) + " ✓")

            # Sauvegarde intermédiaire tous les save_every messages
            if n_traites % save_every == 0:
                write_jsonl(messages, output_path)
                log.info(f"  Sauvegarde intermédiaire ({n_traites} traités)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.resumer(errors=n_erreurs, skipped=n_eligibles - n_traites - n_erreurs)


if __name__ == "__main__":
    main()
