#!/usr/bin/env python3
"""
collocations.py — Bigrammes significatifs (G² Dunning + PMI).

Produit, pour chaque source (caption, dialogue) et chaque phase (P1/P2/P3)
+ global, un classement des bigrammes adjacents scorés par :
  - log-likelihood G² de Dunning (1993) — robuste aux faibles effectifs
  - PMI (pointwise mutual information) — complémentaire, biaisé vers le rare

Le corpus Magyar contient des unités lexicales composées (« 414 ОБр »,
« камікадзе дрон », « мадяр птах », « приватбанк моно », « морський дрон »)
que les unigrammes TF-IDF ne peuvent pas capter. Les bigrammes les isolent.

Référence :
  Dunning 1993 — Accurate Methods for the Statistics of Surprise
  and Coincidence, Computational Linguistics 19(1)

Usage :
    python collocations.py --output-dir 4_data_et_viz/lexico
    python collocations.py --output-dir ... --sources dialogue --min-count 10
    python collocations.py --output-dir ... --no-plot

Sorties :
    collocations_{src}.csv            — global (toutes phases)
    collocations_{src}_{P1,P2,P3}.csv — par phase
    fig_collocations_top30_{src}.png  — barplot G² top 30 global
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config  # noqa: E402
from categories_semantiques import PHASE_SHORT  # noqa: E402

# CSV lus dans 4_data_et_viz/lexico/ ; figures écrites un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import pandas as pd
import matplotlib.pyplot as plt


MIN_COUNT_DEFAULT = 5
TOP_N_CSV = 200
TOP_N_PLOT = 30


# ---------------------------------------------------------------------------
# Bigrammes
# ---------------------------------------------------------------------------

def extraire_bigrammes(df):
    """
    Extrait les bigrammes adjacents par message (ordre préservé).
    Retourne (bigram_counts, unigram_counts, N_bigrams).
    """
    bigrams = Counter()
    unigrams = Counter()
    n_bigrams = 0

    for _, group in df.groupby("message_id", sort=False):
        lemmes = group["lemma"].tolist()
        for lem in lemmes:
            unigrams[lem] += 1
        for a, b in zip(lemmes, lemmes[1:]):
            if a == b:
                continue  # bigrammes w-w (bégaiements dialogue) écartés
            bigrams[(a, b)] += 1
            n_bigrams += 1

    return bigrams, unigrams, n_bigrams


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _xlogx(x):
    return x * math.log(x) if x > 0 else 0.0


def log_likelihood_g2(o11, o12, o21, o22):
    """
    Dunning G² pour table 2×2 :
        w2=1   w2=0
    w1=1  o11    o12
    w1=0  o21    o22
    """
    n = o11 + o12 + o21 + o22
    if n == 0:
        return 0.0
    row1 = o11 + o12
    row2 = o21 + o22
    col1 = o11 + o21
    col2 = o12 + o22
    e11 = row1 * col1 / n
    e12 = row1 * col2 / n
    e21 = row2 * col1 / n
    e22 = row2 * col2 / n
    g2 = 0.0
    for o, e in ((o11, e11), (o12, e12), (o21, e21), (o22, e22)):
        if o > 0 and e > 0:
            g2 += o * math.log(o / e)
    return 2.0 * g2


def scorer_bigrammes(bigrams, unigrams, n_bigrams, min_count):
    """
    Calcule G² et PMI pour chaque bigramme fréquent ≥ min_count.
    """
    n_unigrams_total = sum(unigrams.values())
    rows = []
    for (a, b), c_ab in bigrams.items():
        if c_ab < min_count:
            continue
        c_a = unigrams[a]
        c_b = unigrams[b]
        if c_a == 0 or c_b == 0:
            continue

        # Table 2×2 sur bigrammes (unité = paire adjacente)
        o11 = c_ab
        # c_a = nb d'occurrences de a (unigrammes). Nb de bigrammes (a,*) est
        # approximé par c_a (chaque occurrence de a, sauf la dernière du
        # message, produit un bigramme). Différence marginale → suffisant.
        o12 = c_a - o11
        o21 = c_b - o11
        o22 = n_bigrams - o11 - o12 - o21
        if o12 < 0 or o21 < 0 or o22 < 0:
            continue
        g2 = log_likelihood_g2(o11, o12, o21, o22)

        # PMI classique sur probabilités d'unigramme
        p_ab = c_ab / n_bigrams
        p_a = c_a / n_unigrams_total
        p_b = c_b / n_unigrams_total
        pmi = math.log2(p_ab / (p_a * p_b)) if p_a * p_b > 0 else 0.0

        rows.append({
            "w1": a,
            "w2": b,
            "count": c_ab,
            "freq_w1": c_a,
            "freq_w2": c_b,
            "g2": round(g2, 3),
            "pmi": round(pmi, 4),
        })

    return pd.DataFrame(rows).sort_values("g2", ascending=False)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def tracer_top_bigrammes(df, src, fig_dir, top_n=TOP_N_PLOT):
    if df.empty:
        return
    top = df.head(top_n).iloc[::-1]
    labels = [f"{r.w1} {r.w2}" for r in top.itertuples()]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh(labels, top["g2"], color="#2a5a7a")
    ax.set_xlabel("Dunning G² (log-likelihood)")
    ax.set_title(f"Top {top_n} bigrammes — {src}")
    fig.tight_layout()
    out = os.path.join(fig_dir, f"fig_collocations_top{top_n}_{src}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    -> {out}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def traiter(df, src, output_dir, fig_dir, min_count, phase_suffix=None, make_plot=True):
    bigrams, unigrams, n_bigrams = extraire_bigrammes(df)
    scored = scorer_bigrammes(bigrams, unigrams, n_bigrams, min_count)

    label = f"{src}" + (f"_{phase_suffix}" if phase_suffix else "")
    out_csv = os.path.join(output_dir, f"collocations_{label}.csv")
    scored.head(TOP_N_CSV).to_csv(out_csv, index=False)
    print(f"    {len(scored)} bigrammes ≥ {min_count} | "
          f"top-{min(TOP_N_CSV, len(scored))} -> {out_csv}")

    if make_plot and not phase_suffix:
        tracer_top_bigrammes(scored, src, fig_dir)

    return scored


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--sources", nargs="*", default=["caption", "dialogue"])
    parser.add_argument("--min-count", type=int, default=MIN_COUNT_DEFAULT,
                        help=f"Fréquence min d'un bigramme (défaut : {MIN_COUNT_DEFAULT})")
    parser.add_argument("--no-plot", action="store_true",
                        help="Ne pas générer le barplot")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    load_config(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Collocations (G² Dunning + PMI)")
    print(f"  min_count : {args.min_count}\n")

    for src in args.sources:
        csv_path = os.path.join(args.output_dir, f"lemmes_{src}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {src} : {csv_path} introuvable")
            continue

        print(f"── {src} ──")
        df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
        df = df[df["phase"].notna() & (df["phase"] != "")]

        # Global
        print("  [global]")
        traiter(df, src, args.output_dir, fig_dir, args.min_count,
                make_plot=not args.no_plot)

        # Par phase
        for phase_raw in sorted(df["phase"].unique()):
            phase_short = PHASE_SHORT.get(phase_raw)
            if phase_short is None:
                continue
            df_phase = df[df["phase"] == phase_raw]
            print(f"  [{phase_short}]")
            traiter(df_phase, src, args.output_dir, fig_dir, args.min_count,
                    phase_suffix=phase_short, make_plot=False)
        print()

    print("Terminé.")


if __name__ == "__main__":
    main()
