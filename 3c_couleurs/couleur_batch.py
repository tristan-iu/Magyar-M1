#!/usr/bin/env python3
"""
Analyse couleur HSV des keyframes du corpus.

Calcule les statistiques de couleur (moyenne, variance intra/inter-frames,
entropie de Shannon) pour chaque message du corpus — vidéos via leurs
keyframes, photos directement.

L'espace HSV est préféré à RGB : H (teinte pure), S (saturation), V (luminosité)
sont des axes indépendants et interprétables. La saturation et l'entropie
captent la cohérence éditoriale (color-grading, homogénéisation de palette)
indépendamment de la teinte dominante.

Pipeline par message :
  1. Liste les keyframes via glob {channel}_{id}_kf_*.png, ou la photo directe
  2. Pour chaque frame : resize 256px (speed), BGR→HSV, moyennes H/S/V,
     histogramme HSV 3D (16×16×16 bins)
  3. Agrégation message : moyennes des moyennes, écart-type inter-frames
     (cohérence éditoriale), entropie de Shannon sur histogramme agrégé
  4. Deux sorties : CSV scalaires (pour R) + NPZ histogrammes (pour PCA/similarité)

Sorties :
  - 4_data_et_viz/couleurs/couleur_stats.csv   — 1 ligne/message, 7 métriques scalaires
  - 4_data_et_viz/couleurs/couleur_histogrammes.npz — matrice (N, 4096) float32 + message_ids

Options CLI : --limit, --ids, --overwrite, --hist-bins, --resize-dim

Usage :
    python couleur_batch.py
    python couleur_batch.py --limit 50
    python couleur_batch.py --ids 42 138 --overwrite
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    filtrer_eligibles,
    fmt_eta,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COULEUR_DIR = SCRIPT_DIR.parent / "4_data_et_viz" / "couleurs"

# ── Constantes ──────────────────────────────────────────────────────────────
# Histogramme 3D : 16 bins par canal = 4096 dimensions au total.
# 16 bins est un compromis : assez fin pour distinguer palettes, assez grossier
# pour rester robuste au bruit de compression vidéo.
HIST_BINS = 16

# Resize : les stats HSV sont invariantes à la résolution, resize accélère ×10
# sans perte de signal. 256px sur le grand côté conserve suffisamment de détail.
RESIZE_DIM = 256

# Une frame avec moins de N pixels non-noirs est considérée corrompue / transition
# ffmpeg et skippée pour ne pas polluer les moyennes.
MIN_NON_BLACK_PIXELS = 100

# Colonnes CSV de sortie
CSV_FIELDNAMES = [
    "message_id",
    "hsv_h_mean", "hsv_s_mean", "hsv_v_mean",
    "hsv_s_inter", "hsv_v_inter",
    "hsv_entropy",
    "hsv_n_frames",
]


# ── I/O images ──────────────────────────────────────────────────────────────

def lire_image(path: Path) -> np.ndarray | None:
    """Lit une image via buffer binaire (gère les chemins unicode).
    Retourne None si lecture échoue."""
    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def lister_sources(msg: dict, keyframes_dir: Path, corpus_base: Path) -> list[Path]:
    """Retourne la liste des images à analyser pour un message.

    - Vidéos : toutes les keyframes {canal}_{id}_kf_*.png (détection par glob)
    - Photos : l'image directe via media_chemin
    - Texte seul : liste vide
    """
    channel = msg.get("canal", "robert_magyar")
    mid = msg["message_id"]

    # On cherche d'abord les keyframes (vidéo) — pas de champ JSONL pour le compte,
    # on regarde directement sur disque.
    if msg.get("media_type") == "video":
        pattern = f"{channel}_{mid}_kf_"
        kf_files = sorted(
            p for p in keyframes_dir.iterdir()
            if p.name.startswith(pattern) and p.suffix.lower() in (".jpg", ".png")
        )
        if kf_files:
            return kf_files

    # Sinon, si c'est une photo, on prend l'image directe
    if msg.get("media_type") == "photo" and msg.get("media_chemin"):
        photo = corpus_base / msg["media_chemin"]
        if photo.is_file():
            return [photo]

    return []


# ── Calcul HSV ──────────────────────────────────────────────────────────────

def calculer_stats_frame(img: np.ndarray, bins: int, resize_dim: int) -> dict | None:
    """Calcule les statistiques HSV d'une frame.

    Entrée : img BGR (cv2), bins — nb de bins par canal de l'histogramme,
             resize_dim — taille max du grand côté après redimensionnement
    Sortie : dict avec h_mean, s_mean, v_mean, hist (array 3D aplati float32)
             None si frame corrompue (trop peu de pixels non-noirs)
    """
    # On resize le grand côté à resize_dim pour accélérer sans perte statistique
    h, w = img.shape[:2]
    scale = resize_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # On détecte les frames corrompues (presque entièrement noires) en comptant
    # les pixels dont la somme BGR dépasse un seuil bas
    non_black = int((img.sum(axis=2) > 15).sum())
    if non_black < MIN_NON_BLACK_PIXELS:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Moyennes par canal. OpenCV HSV : H ∈ [0,179], S/V ∈ [0,255]
    h_mean = float(hsv[..., 0].mean())
    s_mean = float(hsv[..., 1].mean())
    v_mean = float(hsv[..., 2].mean())

    # Histogramme 3D HSV — proba de chaque cellule couleur
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total

    return {
        "h_mean": h_mean,
        "s_mean": s_mean,
        "v_mean": v_mean,
        "hist": hist,
    }


def agreger_message(stats_frames: list[dict], bins: int) -> dict:
    """Agrège les stats de plusieurs frames en statistiques message-level.

    Entrée : liste de dicts produits par calculer_stats_frame (non-None)
    Sortie : dict final {hsv_*, hist_aggr} prêt à écrire
    """
    h_means = np.array([s["h_mean"] for s in stats_frames])
    s_means = np.array([s["s_mean"] for s in stats_frames])
    v_means = np.array([s["v_mean"] for s in stats_frames])

    # L'entropie se calcule sur l'histogramme agrégé (moyenne des histogrammes
    # normalisés — équivaut à concaténer les pixels de toutes les frames)
    hist_aggr = np.mean([s["hist"] for s in stats_frames], axis=0)
    nonzero = hist_aggr[hist_aggr > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) if nonzero.size else 0.0

    # inter = écart-type entre frames (cohérence éditoriale)
    # Une seule frame (photo) → inter = 0 par convention
    n = len(stats_frames)
    s_inter = float(s_means.std()) if n > 1 else 0.0
    v_inter = float(v_means.std()) if n > 1 else 0.0

    return {
        "hsv_h_mean":   round(float(h_means.mean()), 3),
        "hsv_s_mean":   round(float(s_means.mean()), 3),
        "hsv_v_mean":   round(float(v_means.mean()), 3),
        "hsv_s_inter":  round(s_inter, 3),
        "hsv_v_inter":  round(v_inter, 3),
        "hsv_entropy":  round(entropy, 4),
        "hsv_n_frames": n,
        "_hist_aggr":   hist_aggr.astype(np.float32),
    }


# ── Idempotence CSV ─────────────────────────────────────────────────────────

def charger_csv_existant(csv_path: Path) -> set[int]:
    """Retourne les message_id déjà présents dans le CSV."""
    done = set()
    if not csv_path.is_file():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(int(row["message_id"]))
    return done


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Analyse couleur HSV des keyframes (CSV stats + NPZ histogrammes)",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source (défaut: messages_blasons.jsonl)")
    parser.add_argument(
        "--csv",
        default=str(COULEUR_DIR / "couleur_stats.csv"),
        help="CSV scalaires par message",
    )
    parser.add_argument(
        "--npz",
        default=str(COULEUR_DIR / "couleur_histogrammes.npz"),
        help="NPZ histogrammes HSV 3D aplati par message",
    )
    parser.add_argument("--hist-bins", type=int, default=HIST_BINS,
                        help=f"Bins par canal HSV (défaut: {HIST_BINS})")
    parser.add_argument("--resize-dim", type=int, default=RESIZE_DIM,
                        help=f"Taille max du grand côté avant calcul (défaut: {RESIZE_DIM})")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("couleurs", cfg=cfg)

    corpus_base   = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])

    input_path = (Path(args.input) if args.input
                  else corpus_base / "messages_blasons.jsonl")
    csv_path = Path(args.csv)
    npz_path = Path(args.npz)
    bins = args.hist_bins

    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)
    if not keyframes_dir.is_dir():
        log.error(f"Dossier keyframes introuvable : {keyframes_dir}")
        sys.exit(1)

    resize_dim = args.resize_dim

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    filtre_ids = set(args.ids) if args.ids else None
    eligible_indices = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        overwrite=True,  # idempotence gérée via le CSV, pas via les champs JSONL
        start_date=args.start_date,
        end_date=args.end_date,
    )
    # On ne garde que vidéos (keyframes vérifiées par glob plus tard) ou photos
    eligible_indices = [
        i for i in eligible_indices
        if messages[i].get("media_type") == "video"
        or (messages[i].get("media_type") == "photo"
            and messages[i].get("media_chemin"))
    ]

    # Idempotence CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = charger_csv_existant(csv_path) if not args.overwrite else set()
    if not args.overwrite:
        eligible_indices = [
            i for i in eligible_indices
            if messages[i]["message_id"] not in done
        ]

    if args.limit:
        eligible_indices = eligible_indices[:args.limit]

    n_eligible = len(eligible_indices)
    log.info(f"Messages éligibles : {n_eligible} (déjà fait : {len(done)})")
    log.info(f"Bins : {bins}×{bins}×{bins} = {bins**3} dims | Resize : {args.resize_dim}px")

    if n_eligible == 0:
        log.info("Rien à faire.")
        return

    # Si overwrite, on réécrit le NPZ depuis zéro ; sinon on charge l'existant
    existing_ids = np.array([], dtype=np.int64)
    existing_hists = np.zeros((0, bins**3), dtype=np.float32)
    if not args.overwrite and npz_path.is_file():
        with np.load(npz_path) as data:
            existing_ids = data["message_ids"]
            existing_hists = data["histograms"]
        if existing_hists.shape[1] != bins**3:
            log.error(
                f"NPZ existant a {existing_hists.shape[1]} dims, "
                f"mais --hist-bins demande {bins**3}. Utilisez --overwrite."
            )
            sys.exit(1)

    csv_mode = "w" if args.overwrite else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if args.overwrite or not done:
        csv_writer.writeheader()

    new_ids = []
    new_hists = []

    processed = 0
    errors = 0
    skipped_no_frame = 0
    t0 = time.time()

    try:
        for rank, idx in enumerate(tqdm(eligible_indices, desc="HSV", unit="msg")):
            msg = messages[idx]
            mid = msg["message_id"]
            try:
                sources = lister_sources(msg, keyframes_dir, corpus_base)
                if not sources:
                    skipped_no_frame += 1
                    continue

                frames_stats = []
                for src in sources:
                    img = lire_image(src)
                    if img is None:
                        continue
                    stats = calculer_stats_frame(img, bins, resize_dim)
                    if stats is not None:
                        frames_stats.append(stats)

                if not frames_stats:
                    skipped_no_frame += 1
                    continue

                aggr = agreger_message(frames_stats, bins)

                row = {k: aggr[k] for k in CSV_FIELDNAMES if k != "message_id"}
                row["message_id"] = mid
                csv_writer.writerow(row)
                csv_file.flush()

                new_ids.append(mid)
                new_hists.append(aggr["_hist_aggr"])
                processed += 1

            except Exception as e:
                log.warning(f"msg {mid} : {e}")
                errors += 1

    finally:
        csv_file.close()

        # Sauvegarde NPZ : on concatène existant + nouveau, puis on écrit
        if new_ids:
            all_ids = np.concatenate([existing_ids, np.array(new_ids, dtype=np.int64)])
            all_hists = np.vstack([existing_hists, np.array(new_hists, dtype=np.float32)])
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, message_ids=all_ids, histograms=all_hists)
            log.info(f"NPZ sauvegardé : {npz_path} ({all_hists.shape})")

    elapsed = time.time() - t0
    log.info(
        f"\nTerminé en {fmt_eta(elapsed)} — "
        f"{processed} succès, {skipped_no_frame} sans frame utilisable, "
        f"{errors} erreurs."
    )


if __name__ == "__main__":
    main()
