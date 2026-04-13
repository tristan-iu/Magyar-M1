#!/usr/bin/env python3
"""
Classification zero-shot CLIP sur les keyframes du corpus Magyar.
Produit un CSV avec les scores binaires par concept pour chaque keyframe.

Architecture : 7 classifieurs binaires independants.
Chaque classifieur = softmax(pos_prompt, neg_prompt)[0] → score 0→1.
Les scores sont independants (pas de competition softmax entre concepts).

Pourquoi binaire plutot que softmax multi-classe :
- La softmax a 9-12 labels dilue les probabilites sur des classes proches
  (ex: "drone surveillance" vs "FPV" se cannibalisent mutuellement)
- Chaque concept recoit sa propre comparaison calibree
- Un frame peut scorer haut sur plusieurs concepts simultanement (correct)

Concepts detectes :
  clip_vlog   — personne face camera, style selfie vlog
  clip_aerial — vue aerienne top-down terrain (ISR)
  clip_fpv    — drone FPV avec prop guards circulaires
  clip_stats  — carte stats institutionnelle (fond noir + ВИЯВЛЕНО X)
  clip_screen — filmer un ecran/controleur DJI
  clip_strike — impact/explosion/destruction vue aerienne
  clip_hud    — HUD telemetrie drone (chiffres ALT/vitesse/distance)

CSV produit : message_id, frame_filename, date, phase, clip_vlog, ..., clip_hud

Options CLI : --input, --csv, --batch-size, --limit, --ids, --overwrite,
              --start-date, --end-date, --config
"""

import csv
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    build_base_parser,
    load_config,
    setup_logging,
    read_jsonl,
    phase_label,
    fmt_eta,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Classifieurs binaires ───────────────────────────────────────────────────
# Chaque entree : (nom_court, prompt_positif, prompt_negatif)
# Le score final = softmax([pos, neg])[0] — probabilite du concept positif.
# Labels en anglais : CLIP est entraine sur des captions anglaises.
# Le prompt negatif est choisi pour etre visuellement oppose, pas juste "not X".

BINARY_CLASSIFIERS = [
    (
        "clip_vlog",
        "a soldier or person speaking directly into a handheld camera "
        "in close-up selfie vlog style",
        "aerial drone footage of a battlefield from above without any "
        "person speaking to the camera",
    ),
    (
        "clip_aerial",
        "aerial surveillance drone footage looking straight down at fields "
        "dirt roads and terrain from high altitude",
        "footage filmed at ground level from a person vehicle or low-flying "
        "drone showing a forward or sideways view",
    ),
    (
        "clip_fpv",
        "FPV drone video where the curved plastic propeller guard ring or "
        "frame struts are visible at the edges of the frame flying fast "
        "over trees or vegetation",
        "footage filmed from inside a military vehicle or car driving on a "
        "road or stable overhead surveillance drone looking straight down",
    ),
    (
        "clip_stats",
        "a standalone black background title card with oversized bold text "
        "and numbers centered on screen reporting military statistics, with "
        "a military unit badge — no live video, only the graphic",
        "live video footage of soldiers or terrain with subtitle text "
        "captions overlaid on real scenes, or a person talking to camera",
    ),
    (
        "clip_screen",
        "a hand holding a DJI drone controller or tablet displaying live "
        "drone video on its touchscreen outdoors",
        "direct footage of an outdoor scene or person without any screens "
        "or displays visible",
    ),
    (
        "clip_strike",
        "aerial drone footage showing an explosion smoke rising or a burning "
        "vehicle or structure visible on the ground from above",
        "a hand holding a drone controller showing drone video on its "
        "touchscreen or quiet overhead surveillance without fire or explosion",
    ),
]

# Noms de colonnes pour le CSV
CLIP_COLS = [name for name, _, _ in BINARY_CLASSIFIERS]  # 6 concepts
CSV_FIELDNAMES = ["message_id", "frame_filename", "date", "phase"] + CLIP_COLS


def load_existing_csv(csv_path: Path) -> set[str]:
    """Charge les noms de frames deja traites dans le CSV (idempotence)."""
    done = set()
    if not csv_path.is_file():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["frame_filename"])
    return done


def collect_frames(messages: list[dict], keyframes_dir: Path,
                   ids_filter: set[int] | None, limit: int | None,
                   overwrite: bool, done_frames: set[str],
                   cfg: dict,
                   start_date=None, end_date=None) -> list[dict]:
    """Liste toutes les keyframes eligibles avec leurs metadonnees message."""
    from datetime import datetime
    frames = []
    msg_count = 0
    for msg in messages:
        mid = msg["message_id"]
        if ids_filter and mid not in ids_filter:
            continue
        if start_date or end_date:
            raw = msg.get("date")
            if raw:
                try:
                    msg_date = datetime.fromisoformat(raw[:19]).date()
                except (ValueError, TypeError):
                    msg_date = None
            else:
                msg_date = None
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
        kf_count = msg.get("keyframes_count", 0)
        if not kf_count or kf_count <= 0:
            continue
        if limit and msg_count >= limit:
            break

        channel = msg.get("channel", "robert_magyar")
        msg_date_str = msg.get("date", "")
        phase = phase_label(msg_date_str, cfg) if msg_date_str else ""

        kf_pattern = f"{channel}_{mid}_kf_"
        kf_files = sorted(
            p for p in keyframes_dir.iterdir()
            if p.name.startswith(kf_pattern) and p.suffix in (".jpg", ".png")
        )
        if not kf_files:
            continue

        has_new = False
        for kf_path in kf_files:
            if not overwrite and kf_path.name in done_frames:
                continue
            has_new = True
            frames.append({
                "message_id": mid,
                "frame_filename": kf_path.name,
                "frame_path": kf_path,
                "date": msg_date_str,
                "phase": phase or "",
            })

        if has_new:
            msg_count += 1

    return frames


def classify_batch(images: list[Image.Image], processor, model,
                   device: torch.device) -> list[dict]:
    """Classifie un batch d'images via 7 classifieurs binaires independants.

    Pour chaque concept : softmax([prompt_positif, prompt_negatif])[0].
    Les 7 passes sont regroupees en une seule operation matricielle
    pour minimiser les allers-retours GPU.

    Retourne une liste de dicts {clip_col: score} (1 dict par image).
    """
    n_images = len(images)
    n_classifiers = len(BINARY_CLASSIFIERS)

    # Construire la liste de tous les prompts (positif + negatif pour chaque)
    all_prompts = []
    for _, pos, neg in BINARY_CLASSIFIERS:
        all_prompts.append(pos)
        all_prompts.append(neg)
    # all_prompts : 14 textes (2 × 7 classifieurs)

    with torch.no_grad():
        inputs = processor(
            text=all_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # logits_per_image : (n_images, 14)
        logits = outputs.logits_per_image  # shape: (n_images, 14)

    results = []
    for i in range(n_images):
        row = {}
        for j, (col_name, _, _) in enumerate(BINARY_CLASSIFIERS):
            # Extraire les logits pour ce classifieur (pos=2j, neg=2j+1)
            pair_logits = logits[i, [2 * j, 2 * j + 1]]
            score = float(torch.softmax(pair_logits, dim=0)[0].cpu())
            row[col_name] = round(score, 4)
        results.append(row)

    return results


def main():
    parser = build_base_parser(
        "Classification zero-shot CLIP binaire sur les keyframes",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None, help="JSONL source (defaut depuis config)")
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "clip_classification.csv"),
                        help="CSV de sortie")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = setup_logging("clip", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])

    input_path = (Path(args.input) if args.input
                  else corpus_base / cfg["paths"].get("jsonl_faces",
                                                       cfg["paths"]["jsonl_computervision"]))
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)

    model_name = (cfg.get("models", {}).get("clip", {})
                  .get("model_name", "openai/clip-vit-large-patch14"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Chargement CLIP : {model_name} sur {device}")

    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    log.info(f"Modele charge — {len(BINARY_CLASSIFIERS)} classifieurs binaires")
    for col, pos, _ in BINARY_CLASSIFIERS:
        log.info(f"  {col}: \"{pos[:60]}...\"")

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    done_frames = load_existing_csv(csv_path) if not args.overwrite else set()
    log.info(f"Frames deja traitees (CSV) : {len(done_frames)}")

    ids_filter = set(args.ids) if args.ids else None
    frames = collect_frames(messages, keyframes_dir, ids_filter,
                            args.limit, args.overwrite, done_frames, cfg,
                            start_date=args.start_date, end_date=args.end_date)
    log.info(f"Frames a traiter : {len(frames)}")

    if not frames:
        log.info("Rien a faire.")
        return

    csv_mode = "w" if args.overwrite or not done_frames else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if csv_mode == "w":
        csv_writer.writeheader()

    t0 = time.time()
    n_batches = (len(frames) + args.batch_size - 1) // args.batch_size

    try:
        for batch_idx in tqdm(range(n_batches), desc="CLIP", unit="batch"):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(frames))
            batch_frames = frames[start:end]

            images = []
            valid_indices = []
            for i, fr in enumerate(batch_frames):
                try:
                    img = Image.open(fr["frame_path"]).convert("RGB")
                    images.append(img)
                    valid_indices.append(i)
                except Exception as e:
                    log.warning(f"Erreur lecture {fr['frame_filename']}: {e}")

            if not images:
                continue

            scores = classify_batch(images, processor, model, device)

            for idx, score_dict in zip(valid_indices, scores):
                fr = batch_frames[idx]
                row = {
                    "message_id": fr["message_id"],
                    "frame_filename": fr["frame_filename"],
                    "date": fr["date"],
                    "phase": fr["phase"],
                }
                row.update(score_dict)
                csv_writer.writerow(row)

            if (batch_idx + 1) % 50 == 0:
                csv_file.flush()

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde CSV...")
    finally:
        csv_file.close()

    elapsed = time.time() - t0
    log.info(f"Termine en {fmt_eta(elapsed)} — {len(frames)} frames, CSV : {csv_path}")


if __name__ == "__main__":
    main()
