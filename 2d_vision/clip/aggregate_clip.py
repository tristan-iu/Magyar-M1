#!/usr/bin/env python3
"""
Agregation des scores CLIP binaires vers le JSONL principal.

Pipeline :
  1. Lecture du CSV per-frame produit par clip_classify.py
  2. Agregation par message (moyenne des scores + flag si score > seuil)
  3. Enrichissement JSONL : ajoute clip_vlog, clip_aerial, ..., clip_hud
     (scores moyens 0-1 + flags booléens clip_vlog_flag, etc.)
  4. CSV agregé par message (pour analyses R)

Coexistence avec les anciens champs :
  Les anciens champs clip_scene_* et clip_human_* restent dans le JSONL
  (ils ne sont pas supprimés). Les nouveaux champs ont des noms sans préfixe
  de groupe pour les distinguer.

Options CLI : --csv, --output, --threshold, --config, --no-enrich-jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# Doit correspondre aux BINARY_CLASSIFIERS de clip_classify.py
CLIP_COLS = [
    "clip_vlog",
    "clip_aerial",
    "clip_fpv",
    "clip_stats",
    "clip_screen",
    "clip_strike",
]

# Stratégie d'agrégation par concept :
#   "mean" — présence continue sur la majorité des frames (vlog, aerial, screen)
#   "max"  — événement ponctuel pouvant n'apparaître que sur 1-2 frames (stats, strike)
#   "p75"  — présence dominante sans outlier (fpv : mixte FPV + autres plans)
CLIP_AGG = {
    "clip_vlog":   "mean",
    "clip_aerial": "mean",
    "clip_fpv":    "p75",
    "clip_stats":  "max",   # une seule frame stats suffit
    "clip_screen": "mean",
    "clip_strike": "mean",  # mean évite les faux positifs sur frames isolées
}

# Seuil par défaut pour les flags booléens (> seuil → True)
# 0.60 conservateur ; clip_stats/strike sur max → seuil plus élevé acceptable
DEFAULT_THRESHOLD = 0.60


def aggregate_by_message(df: pd.DataFrame,
                         threshold: float) -> pd.DataFrame:
    """Agrège per-frame → par message.

    Stratégie par concept (voir CLIP_AGG) :
    - mean  : présence continue (vlog, aerial, screen)
    - max   : événement ponctuel (stats, strike) — une seule frame suffit
    - p75   : percentile 75 (fpv — filtre les outliers sans les ignorer)

    Champs produits par concept :
      {col}       — score agrégé selon CLIP_AGG
      {col}_flag  — booléen si score > threshold

    Retourne un DataFrame avec 1 ligne par message_id.
    """
    present_cols = [c for c in CLIP_COLS if c in df.columns]
    if not present_cols:
        raise ValueError("Aucune colonne clip_* trouvée dans le CSV.")

    grp = df.groupby(["message_id", "date", "phase"])

    rows = []
    for (mid, date, phase), sub in grp:
        row = {"message_id": mid, "date": date, "phase": phase}
        for col in present_cols:
            strategy = CLIP_AGG.get(col, "mean")
            if strategy == "mean":
                val = float(sub[col].mean())
            elif strategy == "max":
                val = float(sub[col].max())
            elif strategy == "p75":
                val = float(sub[col].quantile(0.75))
            else:
                val = float(sub[col].mean())
            row[col] = round(val, 4)
        rows.append(row)

    agg = pd.DataFrame(rows)

    # Flags booléens
    for col in present_cols:
        agg[f"{col}_flag"] = agg[col] > threshold

    return agg


def enrich_jsonl(msg_df: pd.DataFrame, cfg: dict,
                 input_path: Path | None = None,
                 output_path: Path | None = None,
                 overwrite: bool = False) -> None:
    """Enrichit le JSONL principal avec les scores et flags CLIP.

    Champs écrits par message (si présents dans msg_df) :
      clip_vlog, clip_aerial, ..., clip_hud  — score moyen (float)
      clip_vlog_flag, ..., clip_hud_flag     — flag booléen (bool)
      clip_vlog_max, ..., clip_hud_max       — score max (float, utile pour strike)

    Idempotent : n'écrase pas les champs existants sauf --overwrite.
    Les anciens champs clip_scene_* et clip_human_* sont préservés.
    """
    corpus_base = Path(cfg["paths"]["corpus_base"])
    jsonl_input = input_path or (
        corpus_base / cfg["paths"].get("jsonl_faces", cfg["paths"]["jsonl_computervision"])
    )
    jsonl_output = output_path or jsonl_input

    if not jsonl_input.is_file():
        print(f"Warning: JSONL introuvable, skip : {jsonl_input}")
        return

    # Colonnes à écrire : score agrégé + flag
    present_cols = [c for c in CLIP_COLS if c in msg_df.columns]
    write_cols = {}
    for col in present_cols:
        write_cols[col]           = col             # score agrégé
        write_cols[f"{col}_flag"] = f"{col}_flag"

    # Index par message_id
    clip_index = {}
    for _, row in msg_df.iterrows():
        mid = int(row["message_id"])
        entry = {}
        for field_name, df_col in write_cols.items():
            if df_col in row.index and pd.notna(row[df_col]):
                val = row[df_col]
                if "_flag" in field_name:
                    entry[field_name] = bool(val)
                else:
                    entry[field_name] = round(float(val), 4)
        clip_index[mid] = entry

    messages = []
    with open(jsonl_input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    enriched = 0
    for msg in messages:
        mid = int(msg.get("message_id", -1))
        if mid in clip_index:
            for k, v in clip_index[mid].items():
                if overwrite or k not in msg:
                    msg[k] = v
                    enriched += 1

    with open(jsonl_output, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    print(f"JSONL enrichi : {enriched} champs sur {len(clip_index)} messages "
          f"→ {jsonl_output.name}")


def main():
    parser = argparse.ArgumentParser(description="Agregation scores CLIP binaires")
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "clip_classification.csv"),
                        help="CSV per-frame produit par clip_classify.py")
    parser.add_argument("--input", default=None,
                        help="JSONL source a enrichir (defaut: depuis config)")
    parser.add_argument("--output", default=None,
                        help="JSONL de sortie (defaut: meme que --input)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Seuil pour les flags booleens (defaut: {DEFAULT_THRESHOLD})")
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Reecrire les champs clip_* existants dans le JSONL")
    parser.add_argument("--no-enrich-jsonl", action="store_true",
                        help="Ne pas ecrire dans le JSONL principal")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Erreur: CSV introuvable : {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Charge : {len(df)} lignes depuis {csv_path.name} "
          f"({df['message_id'].nunique()} messages, "
          f"{len([c for c in CLIP_COLS if c in df.columns])}/7 colonnes CLIP)")

    # Vérification colonnes manquantes
    missing = [c for c in CLIP_COLS if c not in df.columns]
    if missing:
        print(f"Warning: colonnes absentes du CSV (run partiel?) : {missing}")

    # Agrégation
    msg_df = aggregate_by_message(df, args.threshold)
    results_dir = csv_path.parent
    out_csv = results_dir / "clip_by_message.csv"
    msg_df.to_csv(out_csv, index=False)
    print(f"Sauve : {out_csv} ({len(msg_df)} messages)")

    # Résumé des flags
    print(f"\n=== Flags (seuil moyenne > {args.threshold}) ===")
    for col in CLIP_COLS:
        flag_col = f"{col}_flag"
        if flag_col in msg_df.columns:
            n = msg_df[flag_col].sum()
            pct = 100 * n / len(msg_df)
            print(f"  {col:<14}: {int(n):4d} messages ({pct:.1f}%)")

    # Enrichissement JSONL
    if not args.no_enrich_jsonl:
        input_path  = Path(args.input)  if args.input  else None
        output_path = Path(args.output) if args.output else None
        enrich_jsonl(msg_df, cfg,
                     input_path=input_path,
                     output_path=output_path,
                     overwrite=args.overwrite)

    print("\nAgregation terminee.")


if __name__ == "__main__":
    main()
