#!/usr/bin/env python3
"""
specificites.py — Spécificités lexicales par phase (modèle hypergéométrique de Lafon).

Mesure de référence en analyse des données textuelles pour la question
« quels mots caractérisent une partie du corpus ? ». Contrairement au TF-IDF
(qui, sur seulement 3 phases agrégées, dégénère en fréquence pondérée) et au
chi2 par paires, la spécificité de Lafon teste directement, pour chaque forme,
si sa fréquence dans une partie s'écarte significativement de ce qu'un tirage
aléatoire produirait — sous loi hypergéométrique.

Modèle (Lafon, 1980, « Sur la variabilité de la fréquence des formes dans un
corpus ») :
  - corpus de T occurrences, partie (phase) de t occurrences
  - forme de fréquence totale F, dont f dans la partie
  - sous H0 (répartition au hasard), le nombre d'occurrences de la forme dans
    la partie suit une loi hypergéométrique H(T, F, t)
  - spécificité POSITIVE (sur-emploi) : -log10 P(X >= f)
  - spécificité NÉGATIVE (sous-emploi) : -log10 P(X <= f), reportée signée
  - une forme dont f ≈ E = F·t/T est « banale » (spécificité ~ 0)

Le score est donc un -log10(probabilité) signé : +4 ≈ sur-emploi à p≈10⁻⁴,
−4 ≈ sous-emploi à p≈10⁻⁴. Seuil de banalité usuel : |score| >= 2 (p <= 0.01).

Ce que ce script lit : lemmes_{src}.csv (produit par lexicometrie.py).
Ce qu'il produit :
  - specificites_{src}.csv : phase, lemma, f, F, t, T, attendu, specificite, signe
  - fig_specificites_{src}.png : top-N spécificités positives par phase (3 panneaux)

Usage :
    python specificites.py --output-dir 4_data_et_viz/lexico
    python specificites.py --output-dir ... --top-n 20 --min-count 5 --sources caption dialogue
"""

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config  # noqa: E402
from categories_semantiques import (  # noqa: E402
    categorize, COLORS, PHASE_SHORT, translate,
)

# CSV de lemmes lus dans 4_data_et_viz/lexico/ ; figures écrites un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

# Plafond du score : les probabilités extrêmes sous-débordent (P→0), -log10
# tend vers l'infini. On plafonne pour l'affichage et l'export. Au plafond,
# plusieurs formes quasi-exclusives à une phase saturent ex aequo (p sous le
# plancher machine) : on les départage alors par fréquence brute f décroissante
# (convention Lexico3/TXM — la forme la plus fréquente parmi les exclusives
# prime), d'où le tri secondaire ["specificite", "f"] partout.
SPEC_CAP = 50.0
# Seuil de banalité : |spécificité| < SEUIL_BANALITE => forme non spécifique.
SEUIL_BANALITE = 2.0
_LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Calcul des spécificités
# ---------------------------------------------------------------------------

def calculer_specificites(csv_path, min_count=5, min_messages=3):
    """
    Calcule la spécificité hypergéométrique de chaque (phase, lemme).

    Les totaux T (corpus) et t (partie) sont comptés sur TOUS les tokens
    retenus après lemmatisation — pas seulement sur le vocabulaire reporté —
    pour rester fidèle au modèle de Lafon. Le filtre min_count/min_messages
    ne restreint QUE les formes effectivement reportées (anti-hapax).

    Entrée : csv_path — lemmes_{src}.csv ; min_count — fréquence corpus min ;
             min_messages — nb de messages distincts min (garde-fou hapax).
    Sortie : DataFrame (phase, lemma, f, F, t, T, attendu, specificite, signe).
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
    df = df[df["phase"].notna() & (df["phase"] != "")]

    T = len(df)                                   # taille du corpus (tokens)
    t_par_phase = df["phase"].value_counts().to_dict()    # taille de chaque partie
    F_par_lemme = df["lemma"].value_counts().to_dict()    # fréquence corpus de chaque forme
    msg_counts = df.groupby("lemma")["message_id"].nunique().to_dict()

    # Vocabulaire reporté : formes assez fréquentes ET pas quasi-hapax.
    vocab = [
        lemma for lemma, F in F_par_lemme.items()
        if F >= min_count and msg_counts.get(lemma, 0) >= min_messages
    ]

    # Comptage f = occurrences de chaque forme dans chaque phase.
    f_table = (df[df["lemma"].isin(vocab)]
               .groupby(["phase", "lemma"]).size())

    phases = sorted(t_par_phase.keys())
    rows = []
    for phase in phases:
        t = t_par_phase[phase]
        for lemma in vocab:
            F = F_par_lemme[lemma]
            f = int(f_table.get((phase, lemma), 0))
            attendu = F * t / T

            # Sur-emploi : -log10 P(X >= f) ; sous-emploi : -log10 P(X <= f).
            # On passe par logsf/logcdf (base e) pour éviter le sous-débordement.
            if f >= attendu:
                logp = hypergeom.logsf(f - 1, T, F, t)   # ln P(X >= f)
                spec = -logp / _LN10
                signe = "+"
            else:
                logp = hypergeom.logcdf(f, T, F, t)       # ln P(X <= f)
                spec = logp / _LN10                       # négatif
                signe = "-"

            spec = float(np.clip(spec, -SPEC_CAP, SPEC_CAP))
            rows.append({
                "phase": phase,
                "lemma": lemma,
                "f": f,
                "F": F,
                "t": t,
                "T": T,
                "attendu": round(attendu, 2),
                "specificite": round(spec, 3),
                "signe": signe,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure : top spécificités positives par phase
# ---------------------------------------------------------------------------

def tracer_specificites(df_spec, src, fig_dir, top_n=20):
    """
    Trois panneaux (une phase chacun) : barres horizontales des top_n
    spécificités positives, colorées par catégorie sémantique.
    """
    phases = sorted(df_spec["phase"].unique())
    fig, axes = plt.subplots(1, len(phases),
                             figsize=(5.5 * len(phases), max(6, top_n * 0.34)),
                             squeeze=False)
    axes = axes[0]

    cat_color = {
        "militaire": COLORS["militaire"]["bar"],
        "finance":   COLORS["finance"]["bar"],
        "associes":  COLORS["associes"]["text"],
    }

    for ax, phase in zip(axes, phases):
        sub = (df_spec[(df_spec["phase"] == phase)
                       & (df_spec["specificite"] >= SEUIL_BANALITE)]
               .sort_values(["specificite", "f"], ascending=[False, False])
               .head(top_n)
               .iloc[::-1])  # inversé pour barh (plus fort en haut)

        if sub.empty:
            ax.text(0.5, 0.5, "aucune forme spécifique",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")
            ax.set_title(PHASE_SHORT.get(phase, phase), fontsize=12,
                         fontweight="bold")
            ax.axis("off")
            continue

        colors = [cat_color[categorize(l)] for l in sub["lemma"]]
        y = np.arange(len(sub))
        ax.barh(y, sub["specificite"], color=colors, alpha=0.9, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{l}\n{translate(l)}" for l in sub["lemma"]], fontsize=7.5)
        ax.set_xlabel("Spécificité (−log₁₀ p, sur-emploi)", fontsize=9)
        ax.set_title(PHASE_SHORT.get(phase, phase), fontsize=12,
                     fontweight="bold")
        ax.axvline(SEUIL_BANALITE, color="#999999", linewidth=0.7,
                   linestyle="--")
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=cat_color["militaire"], label="Militaire"),
        mpatches.Patch(color=cat_color["finance"],   label="Finance"),
        mpatches.Patch(color=cat_color["associes"],  label="Associés"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Spécificités de Lafon par phase — {src} "
                 f"(top {top_n}, seuil |spéc.| ≥ {SEUIL_BANALITE:g})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    path = os.path.join(fig_dir, f"fig_specificites_{src}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ---------------------------------------------------------------------------
# Traitement d'une source
# ---------------------------------------------------------------------------

def traiter_source(csv_path, src, output_dir, fig_dir, top_n, min_count):
    """Pipeline spécificités complet pour une source."""
    print("  Calcul des spécificités hypergéométriques...")
    df_spec = calculer_specificites(csv_path, min_count=min_count)
    if df_spec.empty:
        print("    [SKIP] aucun lemme retenu")
        return

    n_phases = df_spec["phase"].nunique()
    n_vocab = df_spec["lemma"].nunique()
    print(f"    {n_phases} phases × {n_vocab} formes (≥ {min_count} occ.)")

    csv_out = os.path.join(output_dir, f"specificites_{src}.csv")
    df_spec.sort_values(["phase", "specificite", "f"],
                        ascending=[True, False, False]) \
           .to_csv(csv_out, index=False, encoding="utf-8")
    print(f"    -> {csv_out} ({len(df_spec)} lignes)")

    print("  Figure...")
    tracer_specificites(df_spec, src, fig_dir, top_n=top_n)

    # Aperçu console : top 8 sur-emplois par phase
    for phase in sorted(df_spec["phase"].unique()):
        top = (df_spec[(df_spec["phase"] == phase)
                       & (df_spec["specificite"] >= SEUIL_BANALITE)]
               .sort_values(["specificite", "f"],
                            ascending=[False, False]).head(8))
        if top.empty:
            continue
        mots = ", ".join(f"{r.lemma} ({r.specificite:+.1f})"
                         for r in top.itertuples())
        print(f"    {PHASE_SHORT.get(phase, phase)} : {mots}")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spécificités de Lafon (modèle hypergéométrique) par phase")
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--sources", nargs="*", default=["caption", "dialogue"],
                        help="Sources à traiter (défaut : caption dialogue)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top N spécificités positives affichées par phase (défaut : 20)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Fréquence corpus min d'une forme reportée (défaut : 5)")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml")
    args = parser.parse_args()

    load_config(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Spécificités de Lafon — modèle hypergéométrique")
    print(f"  top_n={args.top_n}, min_count={args.min_count}\n")

    for src in args.sources:
        csv_path = os.path.join(args.output_dir, f"lemmes_{src}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {src} : {csv_path} introuvable")
            continue

        print(f"── {src} ──")
        traiter_source(csv_path, src, args.output_dir, fig_dir,
                       args.top_n, args.min_count)
        print()

    print("Terminé.")


if __name__ == "__main__":
    main()
