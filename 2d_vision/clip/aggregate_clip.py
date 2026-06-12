#!/usr/bin/env python3
"""
Agregation des scores CLIP binaires — sortie CSV uniquement.

Pipeline :
  1. Lecture du CSV per-frame produit par clip_classify.py
  2. Agregation par message (mean / max / p75 selon CLIP_AGG)
  3. Ecriture du CSV agrégé clip_by_message.csv (pour analyses R)

Note méthodologique :
  Les classifications CLIP zero-shot se sont révélées peu discriminantes par
  phase sur ce corpus (voir mémoire §limites). Les champs CLIP ne sont plus
  écrits dans le JSONL canonique — le CSV par message reste exploitable comme
  artefact de recherche, mais aucun script en aval ne l'agrège dans le JSONL.

Options CLI : --csv, --threshold, --config
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

# Doit correspondre aux CLASSIFIEURS_BINAIRES de clip_classify.py
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
    "clip_stats":  "max",
    "clip_screen": "mean",
    "clip_strike": "mean",
}

DEFAULT_THRESHOLD = 0.60


def agreger_par_message(df: pd.DataFrame,
                         threshold: float) -> pd.DataFrame:
    """Agrège per-frame → par message.

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

    for col in present_cols:
        agg[f"{col}_flag"] = agg[col] > threshold

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Agregation scores CLIP binaires (CSV uniquement)"
    )
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "clip_classification.csv"),
                        help="CSV per-frame produit par clip_classify.py")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Seuil pour les flags booleens (defaut: {DEFAULT_THRESHOLD})")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    # On charge la config pour cohérence avec le reste du pipeline,
    # même si on n'écrit plus dans le JSONL.
    _ = load_config(args.config) if args.config else load_config()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Erreur: CSV introuvable : {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Charge : {len(df)} lignes depuis {csv_path.name} "
          f"({df['message_id'].nunique()} messages, "
          f"{len([c for c in CLIP_COLS if c in df.columns])}/{len(CLIP_COLS)} colonnes CLIP)")

    missing = [c for c in CLIP_COLS if c not in df.columns]
    if missing:
        print(f"Warning: colonnes absentes du CSV (run partiel?) : {missing}")

    msg_df = agreger_par_message(df, args.threshold)
    out_csv = csv_path.parent / "clip_by_message.csv"
    msg_df.to_csv(out_csv, index=False)
    print(f"Sauve : {out_csv} ({len(msg_df)} messages)")

    print(f"\n=== Flags (seuil moyenne > {args.threshold}) ===")
    for col in CLIP_COLS:
        flag_col = f"{col}_flag"
        if flag_col in msg_df.columns:
            n = msg_df[flag_col].sum()
            pct = 100 * n / len(msg_df)
            print(f"  {col:<14}: {int(n):4d} messages ({pct:.1f}%)")

    print("\nAgregation terminee (CSV uniquement, pas d'écriture JSONL).")


if __name__ == "__main__":
    main()
