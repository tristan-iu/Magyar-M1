#!/usr/bin/env python3
"""
Detection des blasons/logos de brigade dans les keyframes du corpus.

Detecte la presence du blason 414 OBr ("ПТАХИ МАДЯРА") en watermark
dans les coins des keyframes via SIFT feature matching.

SIFT est prefere a ORB ici : descripteurs flottants 128d plus discriminants
sur les details fins (trident, texte cyrillique), meilleure invariance
d'echelle (facteur x2.5 entre resolutions 480p et 1080p), et ratio test
de Lowe plus fiable sur distances euclidiennes que sur Hamming.

Pipeline par message :
  1. Extraction des ROI (coins de l'image : haut-droite, bas-droite)
  2. SIFT feature matching de chaque ROI contre les references
  3. Seuil sur le nombre de bons matches (ratio test de Lowe)
  4. CSV per-frame + enrichissement JSONL

References :
  Placer des crops PNG/JPG du blason seul (bien cadre, fond minimal)
  dans le dossier references/. 10 crops par type couvrant les variantes
  de taille, contraste et compression du corpus.

Champs JSONL produits :
  - blason_present  (bool) — blason trouve dans au moins 1 keyframe
  - blason_detecte  (str)  — categorie dominante (ex: "414_obr"), null sinon
  - blason_roi      (str)  — ROI dominante (ex: "epaule"), null sinon

CSV per-frame (1 ligne = 1 keyframe) :
  message_id, keyframe, frame_position,
  blason_present, n_inliers, blason_detecte, blason_roi

Options CLI : --match-threshold, --ratio, --roi-pct, --limit, --ids, --overwrite

Usage :
    python detect_blasons.py
    python detect_blasons.py --limit 10
    python detect_blasons.py --ids 908 1080 1325 --overwrite
    python detect_blasons.py --match-threshold 8 --ratio 0.75
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    build_base_parser,
    load_config,
    setup_logging,
    read_jsonl,
    write_jsonl,
    update_fiche,
    filter_eligible,
    fmt_eta,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Noms de colonnes du CSV de sortie
CSV_FIELDNAMES = [
    "message_id", "keyframe", "frame_position",
    "blason_present", "n_inliers", "blason_detecte", "blason_roi",
]

# Extensions images acceptees
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Coins a scanner (nom, x_start%, y_start%, x_end%, y_end%)
# Les pourcentages definissent la region d'interet dans l'image
DEFAULT_ROIS = [
    ("haut_droite",  0.70, 0.00, 1.00, 0.30),
    ("bas_droite",   0.70, 0.70, 1.00, 1.00),
    ("haut_gauche",  0.00, 0.00, 0.30, 0.30),
    ("bas_gauche",   0.00, 0.70, 0.30, 1.00),
]


def imread_safe(path: str) -> np.ndarray | None:
    """Lit une image via buffer binaire (gere les chemins unicode)."""
    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_frame_index(filename: str) -> int:
    """Extrait l'index numerique d'un nom de keyframe.
    'robert_magyar_123_kf_0005.png' -> 5. Photos directes -> 1."""
    m = re.search(r"_kf_(\d+)", filename)
    return int(m.group(1)) if m else 1


def extract_roi(img: np.ndarray, roi_pcts: tuple) -> np.ndarray:
    """Decoupe une region d'interet d'apres des pourcentages (x0, y0, x1, y1)."""
    h, w = img.shape[:2]
    _, x0_pct, y0_pct, x1_pct, y1_pct = roi_pcts
    x0, y0 = int(w * x0_pct), int(h * y0_pct)
    x1, y1 = int(w * x1_pct), int(h * y1_pct)
    return img[y0:y1, x0:x1]


def load_references(refs_dir: Path, log) -> list[dict]:
    """Charge les images de reference et calcule les descripteurs SIFT.

    Scanne les sous-dossiers de refs_dir : chaque sous-dossier est une
    categorie (ex: 414_obr, 414_mono, pm_SARG). Les images a la racine
    de refs_dir sont classees "inconnu".

    Retourne une liste de dicts {name, categorie, keypoints, descriptors}.
    """
    refs = []
    sift = cv2.SIFT_create()

    # Construire la liste (chemin, categorie) en scannant sous-dossiers + racine
    sources: list[tuple[Path, str]] = []
    for entry in sorted(refs_dir.iterdir()):
        if entry.is_dir():
            for img_path in sorted(entry.iterdir()):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    sources.append((img_path, entry.name))
        elif entry.is_file() and entry.suffix.lower() in IMG_EXTENSIONS:
            sources.append((entry, "inconnu"))

    if not sources:
        log.error(f"Aucune image de reference dans {refs_dir} (ni sous-dossiers)")
        sys.exit(1)

    categories_vues = set()
    for img_path, categorie in sources:
        img = imread_safe(str(img_path))
        if img is None:
            log.warning(f"  [{categorie}] Impossible de lire : {img_path.name}")
            continue

        kp, desc = sift.detectAndCompute(img, None)
        if desc is None or len(kp) < 5:
            log.warning(
                f"  [{categorie}] Trop peu de keypoints : "
                f"{img_path.name} ({len(kp) if kp else 0})"
            )
            continue

        refs.append({
            "name": img_path.stem,
            "categorie": categorie,
            "keypoints": kp,
            "descriptors": desc,
        })
        categories_vues.add(categorie)
        log.info(
            f"  [{categorie}] {img_path.name} — {len(kp)} kp"
        )

    log.info(
        f"{len(refs)} references chargees "
        f"({len(categories_vues)} categories : {', '.join(sorted(categories_vues))})"
    )
    return refs


def match_roi_against_refs(
    roi_img: np.ndarray,
    refs: list[dict],
    sift: cv2.SIFT,
    bf: cv2.BFMatcher,
    ratio: float,
) -> tuple[int, str]:
    """Match une ROI contre toutes les references via SIFT + RANSAC.

    Pipeline :
      1. Ratio test de Lowe — filtre les matches ambigus
      2. Verification geometrique RANSAC (homographie) — seuls les points
         formant une transformation coherente (inliers) sont comptes.
         Les faux positifs texturaux ne passent pas cette etape.

    Retourne (n_inliers, categorie_meilleure_ref).
    n_inliers est le score de confiance pour le seuil de detection.
    """
    kp_roi, desc_roi = sift.detectAndCompute(roi_img, None)
    if desc_roi is None or len(kp_roi) < 4:
        return 0, ""

    best_inliers = 0
    best_cat = ""

    for ref in refs:
        try:
            matches = bf.knnMatch(ref["descriptors"], desc_roi, k=2)
        except cv2.error:
            continue

        # Ratio test de Lowe (len(pair) == 2 : knnMatch peut retourner 1 seul voisin)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)

        if len(good) < 4:
            continue

        # Verification geometrique RANSAC
        src_pts = np.float32(
            [ref["keypoints"][m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp_roi[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        n_inliers = int(mask.sum()) if mask is not None else 0

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_cat = ref["categorie"]

    return best_inliers, best_cat


def load_existing_csv_messages(csv_path: Path) -> set[int]:
    """Charge les message_id deja presents dans le CSV (idempotence)."""
    done = set()
    if not csv_path.is_file():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(int(row["message_id"]))
    return done


def main():
    parser = build_base_parser(
        "Detection des blasons de brigade (ORB feature matching)",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source (defaut depuis config)")
    parser.add_argument("--output", default=None,
                        help="JSONL enrichi (defaut: messages_blasons.jsonl)")
    parser.add_argument(
        "--refs-dir",
        default=str(SCRIPT_DIR / "references"),
        help="Dossier des crops de reference (defaut: blasons/references/)",
    )
    parser.add_argument("--match-threshold", type=int, default=15,
                        help="Nb minimum d'inliers RANSAC pour declarer detection (defaut: 15)")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="Ratio test de Lowe (defaut: 0.75)")
    parser.add_argument("--roi-pct", type=float, default=0.30,
                        help="Taille des ROI en pct de l'image (defaut: 0.30 = 30%%)")
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR / "results" / "blason_detection.csv"),
        help="CSV de sortie per-frame",
    )
    parser.add_argument("--rois", nargs="+", default=["haut_droite", "bas_droite"],
                        choices=["haut_droite", "bas_droite", "haut_gauche", "bas_gauche"],
                        help="Coins a scanner (defaut: haut_droite bas_droite)")
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config) if args.config else load_config()
    log = setup_logging("blasons", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])
    fiches_dir = Path(cfg["paths"]["fiches_dir"])
    save_every = 50

    # Chemins I/O
    input_path = (Path(args.input) if args.input
                  else corpus_base / cfg["paths"]["jsonl_faces"])
    output_path = (Path(args.output) if args.output
                   else corpus_base / "messages_blasons.jsonl")
    csv_path = Path(args.csv)
    refs_dir = Path(args.refs_dir)

    # Construire les ROI actives selon --rois et --roi-pct
    pct = args.roi_pct
    active_rois = []
    for roi_def in DEFAULT_ROIS:
        if roi_def[0] in args.rois:
            active_rois.append(roi_def)
    # Mettre a jour les pourcentages si --roi-pct different de 0.30
    if pct != 0.30:
        updated = []
        for name, x0, y0, x1, y1 in active_rois:
            if "droite" in name:
                x0 = 1.0 - pct
            else:
                x1 = pct
            if "haut" in name:
                y1 = pct
            else:
                y0 = 1.0 - pct
            updated.append((name, x0, y0, x1, y1))
        active_rois = updated

    # Verifications
    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)
    if not refs_dir.is_dir():
        log.error(f"Dossier de references introuvable : {refs_dir}")
        sys.exit(1)

    # Charger references
    log.info("Chargement des references ORB...")
    refs = load_references(refs_dir, log)
    if not refs:
        log.error("Aucune reference utilisable. Ajoutez des crops dans le dossier.")
        sys.exit(1)

    # Initialiser SIFT et BFMatcher (NORM_L2 pour descripteurs flottants SIFT)
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Charger JSONL
    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    # Filtrer messages avec keyframes ou photos
    ids_filter = set(args.ids) if args.ids else None
    eligible_indices = filter_eligible(
        messages,
        ids_filter=ids_filter,
        check_fields=["blason_present"],
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    eligible_indices = [
        i for i in eligible_indices
        if (messages[i].get("keyframes_count") or 0) > 0
        or (messages[i].get("media_type") == "photo"
            and messages[i].get("media_path"))
    ]

    if args.limit:
        eligible_indices = eligible_indices[:args.limit]

    n_eligible = len(eligible_indices)
    log.info(f"Messages eligibles : {n_eligible}")
    log.info(f"Seuil matches : {args.match_threshold} | "
             f"Ratio Lowe : {args.ratio} | "
             f"ROIs : {[r[0] for r in active_rois]}")

    if n_eligible == 0:
        log.info("Rien a faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # CSV : idempotence
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_msgs = load_existing_csv_messages(csv_path) if not args.overwrite else set()
    log.info(f"Messages deja dans le CSV : {len(done_msgs)}")

    csv_mode = "w" if args.overwrite else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if args.overwrite or not done_msgs:
        csv_writer.writeheader()

    processed = 0
    errors = 0
    total_detections = 0
    t0 = time.time()

    try:
        for rank, idx in enumerate(tqdm(eligible_indices,
                                        desc="Blasons", unit="msg")):
            msg = messages[idx]
            mid = msg["message_id"]
            channel = msg.get("channel", "robert_magyar")
            # Lister les keyframes (ou photo directe)
            kf_pattern = f"{channel}_{mid}_kf_"
            kf_files = sorted(
                p for p in keyframes_dir.iterdir()
                if p.name.startswith(kf_pattern)
                and p.suffix in (".jpg", ".png")
            )

            is_photo = False
            if not kf_files:
                media_path = msg.get("media_path", "")
                photo_file = corpus_base / media_path if media_path else None
                if photo_file and photo_file.is_file():
                    kf_files = [photo_file]
                    is_photo = True
                else:
                    log.warning(f"msg {mid} : aucune keyframe ni photo")
                    errors += 1
                    continue

            total_kf = len(kf_files)
            frames_with_blason = 0
            frame_results = []

            for kf_idx, kf_path in enumerate(kf_files):
                kf_name = kf_path.name
                kf_num = 1 if is_photo else parse_frame_index(kf_name)

                img = imread_safe(str(kf_path))
                if img is None:
                    log.warning(f"msg {mid} : impossible de lire {kf_name}")
                    continue

                # Tester chaque ROI — pour les photos on scanne l'image entiere
                # en plus des coins, car le blason peut etre n'importe ou
                rois_a_tester = active_rois
                if is_photo:
                    rois_a_tester = active_rois + [
                        ("image_entiere", 0.0, 0.0, 1.0, 1.0)]

                best_n_inliers = 0
                best_roi_name = ""
                best_cat_name = ""

                for roi_def in rois_a_tester:
                    roi_img = extract_roi(img, roi_def)
                    if roi_img.size == 0:
                        continue

                    n_inliers, cat_name = match_roi_against_refs(
                        roi_img, refs, sift, bf, args.ratio)

                    if n_inliers > best_n_inliers:
                        best_n_inliers = n_inliers
                        best_roi_name = roi_def[0]
                        best_cat_name = cat_name

                # Seuil sur les inliers RANSAC (score de confiance)
                detected = best_n_inliers >= args.match_threshold
                if detected:
                    frames_with_blason += 1

                if total_kf > 1:
                    frame_pos = round(kf_idx / (total_kf - 1), 4)
                else:
                    frame_pos = 0.0

                frame_results.append({
                    "kf_num": kf_num,
                    "frame_position": frame_pos,
                    "blason_present": detected,
                    "n_inliers": best_n_inliers,
                    "blason_detecte": best_cat_name,
                    "blason_roi": best_roi_name,
                })

            if not frame_results:
                continue

            # Ecrire CSV
            if mid not in done_msgs:
                for fr in frame_results:
                    csv_writer.writerow({
                        "message_id": mid,
                        "keyframe": fr["kf_num"],
                        "frame_position": fr["frame_position"],
                        "blason_present": fr["blason_present"],
                        "n_inliers": fr["n_inliers"],
                        "blason_detecte": fr["blason_detecte"],
                        "blason_roi": fr["blason_roi"],
                    })

            # Agregation message
            total_frames = len(frame_results)
            blason_present = frames_with_blason > 0

            # Categorie et ROI dominantes parmi les frames positives
            if blason_present:
                cats = [fr["blason_detecte"] for fr in frame_results
                        if fr["blason_present"] and fr["blason_detecte"]]
                rois = [fr["blason_roi"] for fr in frame_results
                        if fr["blason_present"] and fr["blason_roi"]]
                blason_detecte = max(set(cats), key=cats.count) if cats else None
                blason_roi = max(set(rois), key=rois.count) if rois else None
            else:
                blason_detecte = None
                blason_roi = None

            if blason_present:
                total_detections += 1

            # Enrichir JSONL (3 champs)
            msg["blason_present"] = blason_present
            msg["blason_detecte"] = blason_detecte
            msg["blason_roi"] = blason_roi

            # Enrichir fiche (3 champs)
            fiche_fields = {
                "blason_present": blason_present,
                "blason_detecte": blason_detecte,
                "blason_roi": blason_roi,
            }
            update_fiche(msg, fiche_fields, fiches_dir,
                         overwrite=args.overwrite)

            processed += 1

            label_str = f" [{blason_detecte}]" if blason_present else ""
            max_inliers = max(fr["n_inliers"] for fr in frame_results) if frame_results else 0
            tqdm.write(
                f"  msg {mid}: {frames_with_blason}/{total_frames} frames "
                f"avec blason{label_str} (max inliers: {max_inliers})"
            )

            # Sauvegarde intermediaire
            if processed % save_every == 0:
                write_jsonl(messages, output_path)
                csv_file.flush()
                log.info(f"  Sauvegarde intermediaire ({processed} traites)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        csv_file.close()
        write_jsonl(messages, output_path)

    elapsed = time.time() - t0
    skipped = n_eligible - processed - errors
    log.info(
        f"\nTermine en {fmt_eta(elapsed)} — "
        f"{processed} traites, {total_detections} avec blason, "
        f"{skipped} skippes, {errors} erreurs."
    )
    log.info(f"CSV : {csv_path}")
    log.info(f"JSONL : {output_path}")


if __name__ == "__main__":
    main()
