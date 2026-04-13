#!/usr/bin/env python3
"""
Enrichissement technique du corpus via ffprobe.
Extrait les métadonnées (durée, résolution, codec, bitrate, fps, audio)
de chaque fichier média et les ajoute au JSONL.

Pipeline :
  1. Filtrage — messages avec media_path non traités
  2. ffprobe — extraction JSON brut (timeout 10s)
  3. Parsing — normalisation en champs JSONL (vidéo / photo / audio)
  4. Écriture — enrichissement JSONL + fiche individuelle

Options CLI : --input, --output, --media-dir, --limit, --overwrite, --ids

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
    build_base_parser,
    load_config,
    setup_logging,
    read_jsonl,
    write_jsonl,
    update_fiche,
    filter_eligible,
    ProgressTracker,
)


# ── ffprobe ──────────────────────────────────────────────────────────────────

def run_ffprobe(file_path: str, timeout: int = 10) -> dict | None:
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


def orientation_label(w: int, h: int) -> str:
    """Retourne l'orientation à partir des dimensions.

    Entrée : w — largeur en pixels, h — hauteur en pixels
    Sortie : "vertical" | "horizontal" | "square"
    """
    if h > w:
        return "vertical"
    if w > h:
        return "horizontal"
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


def extract_metadata(probe: dict) -> dict:
    """Parse la sortie ffprobe et retourne les champs à ajouter au JSONL.

    Gère trois cas :
    - Image (codec photo sans audio) → file_size seulement
    - Vidéo → duration, codecs, bitrate, fps, has_audio, file_size
    - Audio seul → duration, audio_codec, file_size

    Entrée : probe — dict brut retourné par run_ffprobe()
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
    file_size = None
    raw_size = fmt.get("size")
    if raw_size is not None:
        try:
            file_size = int(raw_size)
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
        if file_size is not None:
            result["file_size"] = file_size
        return result

    # ── Vidéo ──
    if video_stream is not None:
        w = int(video_stream.get("width", 0))
        h = int(video_stream.get("height", 0))

        # Durée : préférer format.duration, fallback sur stream
        duration = None
        for src in (fmt.get("duration"), video_stream.get("duration")):
            if src is not None:
                try:
                    duration = round(float(src), 2)
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
        if duration is not None:
            result["duration"] = duration
        result["video_codec"] = video_stream.get("codec_name")
        if video_bitrate is not None:
            result["video_bitrate"] = video_bitrate
        result["fps"] = parse_fps(video_stream)
        result["has_audio"] = audio_stream is not None
        result["audio_codec"] = audio_stream.get("codec_name") if audio_stream else None
        if file_size is not None:
            result["file_size"] = file_size

        return result

    # ── Audio seul (pas de stream vidéo) ──
    if audio_stream is not None:
        duration = None
        for src in (fmt.get("duration"), audio_stream.get("duration")):
            if src is not None:
                try:
                    duration = round(float(src), 2)
                    break
                except (ValueError, TypeError):
                    pass
        result = {}
        if duration is not None:
            result["duration"] = duration
        result["has_audio"] = True
        result["audio_codec"] = audio_stream.get("codec_name")
        if file_size is not None:
            result["file_size"] = file_size
        return result

    return {}


# ── Batch ────────────────────────────────────────────────────────────────────

def main():
    parser = build_base_parser(
        "Enrichissement technique via ffprobe",
        has_media_dir=True,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("ffprobe", cfg=cfg)
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
    ids_filter = set(args.ids) if args.ids else None
    eligible = filter_eligible(
        messages,
        ids_filter=ids_filter,
        check_fields=["duration", "file_size"],
        overwrite=args.overwrite,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Filtrer davantage : il faut un media_path
    eligible = [i for i in eligible if messages[i].get("media_path")]

    n_eligible = len(eligible)
    log.info(f"Messages éligibles : {n_eligible}")

    if n_eligible == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    tracker = ProgressTracker(n_eligible, label="ffprobe")
    processed = 0
    errors = 0

    try:
        for rank, idx in enumerate(eligible):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            media_path_rel = msg.get("media_path")

            media_file = media_dir / media_path_rel
            basename = os.path.basename(media_path_rel)

            if not media_file.is_file():
                log.warning(f"msg {mid}\t{media_path_rel}\tfichier manquant")
                errors += 1
                tracker.tick(rank, mid, "fichier manquant ✗")
                continue

            # ffprobe
            probe = run_ffprobe(str(media_file))
            if probe is None:
                log.warning(f"msg {mid}\t{media_path_rel}\tffprobe échoué")
                errors += 1
                tracker.tick(rank, mid, "ffprobe échoué ✗")
                continue

            meta = extract_metadata(probe)
            if not meta:
                log.warning(f"msg {mid}\t{media_path_rel}\taucun stream exploitable")
                errors += 1
                tracker.tick(rank, mid, "aucun stream ✗")
                continue

            # Enrichir le message
            msg.update(meta)
            dims = msg.get("media_dimensions")
            if dims and len(dims) == 2:
                msg["orientation"] = orientation_label(dims[0], dims[1])
                meta["orientation"] = msg["orientation"]
            processed += 1

            # Fiche individuelle
            update_fiche(msg, meta, media_dir / "fiches", overwrite=args.overwrite)

            # Progression
            desc_parts = []
            if "duration" in meta:
                desc_parts.append(f"{meta['duration']}s")
            if dims and len(dims) == 2:
                desc_parts.append(f"{dims[0]}x{dims[1]}")
            if "video_codec" in meta:
                desc_parts.append(meta["video_codec"])
            if meta.get("audio_codec"):
                desc_parts.append(meta["audio_codec"])
            tracker.tick(rank, mid, ", ".join(desc_parts) + " ✓")

            # Sauvegarde intermédiaire tous les save_every messages
            if processed % save_every == 0:
                write_jsonl(messages, output_path)
                log.info(f"  Sauvegarde intermédiaire ({processed} traités)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.summary(errors=errors, skipped=n_eligible - processed - errors)


if __name__ == "__main__":
    main()
