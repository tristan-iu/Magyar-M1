#!/usr/bin/env python3
"""
phash_keyframes.py — Calcule un pHash 64-bit par keyframe vidéo + par photo.

Contraste avec le champ `perceptual_hash` du JSONL (calculé par le scraper sur
une seule frame par message = vignette). Ici on hashe **chaque** keyframe PNG
produite par keyframer.py (~1 frame toutes les 10s), ce qui donne la vraie
granularité nécessaire aux analyses de recyclage visuel (55_phash_recyclage.R).

Lit  : messages_clean.jsonl (NVME) + fichiers PNG dans fiches/keyframes/
Écrit: 4_data_et_viz/phash_keyframes.csv

Schéma CSV (5 colonnes) :
  message_id, media_type, keyframe_index, keyframe_path, phash

- message_id     : int — clé de jointure
- media_type     : "photo" | "video"
- keyframe_index : 0 pour photo, 1..N pour vidéo (déduit du `_kf_NNN` du nom)
- keyframe_path  : chemin relatif depuis paths.corpus_base
- phash          : hex 16 chars (imagehash.phash, hash_size=8)

Idempotence : si le CSV existe, on lit les couples (message_id, keyframe_path)
déjà présents et on les skip. Drapeau --overwrite pour recalcul complet.

Statut : script indépendant, voué à être intégré dans keyframer.py une fois
validé à l'usage.
"""

from __future__ import annotations

import csv
import glob as globmod
import re
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))

from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    filtrer_eligibles,
    SuiviProgression,
)

# Sortie par défaut : <repo>/4_data_et_viz/phash_keyframes.csv
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "4_data_et_viz" / "phash_keyframes.csv"

CSV_HEADER = ["message_id", "media_type", "keyframe_index", "keyframe_path", "phash"]

# Pattern pour extraire l'index numérique du nom de keyframe
# Ex: "robert_magyar_8_kf_042.png" → 42
_KF_INDEX_RE = re.compile(r"_kf_(\d+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def lire_csv_existant(csv_path: Path) -> set[tuple[int, str]]:
    """Retourne l'ensemble des couples (message_id, keyframe_path) déjà hashés.

    Entrée : csv_path — chemin du CSV existant (peut ne pas exister)
    Sortie : set de tuples (message_id:int, keyframe_path:str)
    """
    if not csv_path.is_file():
        return set()
    deja = set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mid = int(row["message_id"])
            except (KeyError, ValueError, TypeError):
                continue
            kfp = row.get("keyframe_path", "")
            if kfp:
                deja.add((mid, kfp))
    return deja


def lister_keyframes_video(
    canal: str, mid: int, keyframes_dir: Path
) -> list[tuple[int, Path]]:
    """Glob les keyframes PNG/JPG d'un message vidéo, retourne [(index, path), ...] trié.

    L'index est extrait du nom de fichier (`_kf_042.png` → 42), pas du rang dans
    le tri — c'est la position temporelle réelle dans la vidéo.
    """
    patterns = [
        str(keyframes_dir / f"{canal}_{mid}_kf_*.png"),
        str(keyframes_dir / f"{canal}_{mid}_kf_*.jpg"),
        str(keyframes_dir / f"{canal}_{mid}_kf_*.jpeg"),
    ]
    fichiers = []
    for pat in patterns:
        fichiers.extend(globmod.glob(pat))

    indexes_paths = []
    for chemin in fichiers:
        m = _KF_INDEX_RE.search(chemin)
        if m:
            indexes_paths.append((int(m.group(1)), Path(chemin)))

    indexes_paths.sort(key=lambda x: x[0])
    return indexes_paths


def calculer_phash_fichier(image_path: Path) -> str | None:
    """Ouvre une image et retourne son pHash 64-bit en hex (16 chars).

    Identique à la logique de telegram_scraper.calculer_phash :
    imagehash.phash(img, hash_size=8) → 64 bits → str hex.
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError as e:
        raise SystemExit(f"Dépendance manquante : {e}. pip install Pillow imagehash")

    try:
        with Image.open(image_path) as img:
            img.load()
            return str(imagehash.phash(img, hash_size=8))
    except Exception:
        return None


def main():
    parser = creer_parser_base(
        "Calcule un pHash 64-bit par keyframe vidéo et par photo.",
        has_input=False,   # input fixé par config.yaml (jsonl_clean)
        has_output=False,  # output = CSV, géré via --csv
    )
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV),
        help=f"Chemin du CSV de sortie (défaut : {DEFAULT_CSV})",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger("phash_keyframes", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])
    jsonl_path = corpus_base / cfg["paths"]["jsonl_clean"]
    csv_path = Path(args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.is_file():
        log.error(f"JSONL introuvable : {jsonl_path}")
        sys.exit(1)
    if not keyframes_dir.is_dir():
        log.warning(f"Dossier keyframes introuvable : {keyframes_dir}")

    # ── Lecture JSONL + filtrage messages éligibles (photo ou vidéo) ──
    messages = read_jsonl(jsonl_path)
    log.info(f"Corpus : {len(messages)} messages")

    filtre_ids = set(args.ids) if args.ids else None
    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video", "photo"],
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )
    log.info(f"Messages éligibles (photo + vidéo) : {len(eligibles)}")

    # ── Idempotence : on lit ce qui est déjà hashé, sauf --overwrite ──
    if args.overwrite and csv_path.is_file():
        csv_path.unlink()
        log.info(f"--overwrite : {csv_path} supprimé")
    deja = lire_csv_existant(csv_path)
    if deja:
        log.info(f"  {len(deja)} hashes déjà présents — seront skippés")

    # ── Ouverture CSV en append (header si fichier neuf) ──
    csv_neuf = not csv_path.is_file()
    f_out = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f_out)
    if csv_neuf:
        writer.writerow(CSV_HEADER)
        f_out.flush()

    suivi = SuiviProgression(total=len(eligibles), label="phash_kf")
    n_ajoutes = 0
    n_skip = 0
    n_erreurs = 0

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id")
            media_type = msg.get("media_type")
            canal = msg.get("canal", "robert_magyar")

            # On collecte les couples (keyframe_index, keyframe_path_abs)
            # à hasher. Pour les photos : un seul couple (0, media_chemin).
            # Pour les vidéos : tous les fichiers _kf_NNN.png trouvés.
            cibles: list[tuple[int, Path, str]] = []  # (index, path_abs, path_rel)

            if media_type == "photo":
                rel = msg.get("media_chemin")
                if not rel:
                    n_erreurs += 1
                    continue
                cibles.append((0, corpus_base / rel, rel))

            elif media_type == "video":
                kfs = lister_keyframes_video(canal, mid, keyframes_dir)
                if not kfs:
                    # Pas de keyframes extraites pour cette vidéo — pas une erreur
                    # (keyframer.py n'a peut-être pas tourné dessus), juste skip.
                    suivi.avancer(rank, mid, "(0 kf)")
                    continue
                for kf_index, kf_path in kfs:
                    rel = str(kf_path.relative_to(corpus_base))
                    cibles.append((kf_index, kf_path, rel))

            # On filtre les cibles déjà dans le CSV (idempotence)
            cibles_nouvelles = [
                (kf_idx, kf_abs, kf_rel)
                for (kf_idx, kf_abs, kf_rel) in cibles
                if (mid, kf_rel) not in deja
            ]

            if not cibles_nouvelles:
                n_skip += len(cibles)
                suivi.avancer(rank, mid, f"(skip, {len(cibles)} déjà)")
                continue

            ajoutes_msg = 0
            for kf_idx, kf_abs, kf_rel in cibles_nouvelles:
                if not kf_abs.is_file():
                    log.warning(f"msg {mid}\t{kf_rel}\tfichier manquant")
                    n_erreurs += 1
                    continue

                phash_hex = calculer_phash_fichier(kf_abs)
                if phash_hex is None:
                    log.warning(f"msg {mid}\t{kf_rel}\tphash échec")
                    n_erreurs += 1
                    continue

                writer.writerow([mid, media_type, kf_idx, kf_rel, phash_hex])
                n_ajoutes += 1
                ajoutes_msg += 1
                deja.add((mid, kf_rel))

            # Flush régulier pour ne pas perdre le travail si interruption
            if ajoutes_msg > 0:
                f_out.flush()

            suivi.avancer(rank, mid, f"(+{ajoutes_msg}/{len(cibles)})")

    finally:
        f_out.close()

    suivi.resumer(errors=n_erreurs, skipped=n_skip)
    log.info(f"CSV : {csv_path} (+{n_ajoutes} lignes ajoutées)")


if __name__ == "__main__":
    main()
