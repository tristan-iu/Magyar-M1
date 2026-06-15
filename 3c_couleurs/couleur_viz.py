#!/usr/bin/env python3
"""
Visualisations couleur — scatter PCA, scree plot, distributions HSV par phase.

Lit les CSV produits par couleur_batch.py et couleur_espace.py, produit
des figures matplotlib exploitables directement (pas de détour R).

Figures produites :
  1. couleur_pca_scatter_phase.png    — scatter PC1×PC2 coloré par phase
  2. couleur_pca_scatter_cluster.png  — scatter PC1×PC2 coloré par cluster K-means
  3. couleur_pca_scree.png            — variance expliquée par composante
  4. couleur_distributions_phase.png  — boxplots HSV (entropie, S, V, s_inter) par phase

Palette phases depuis config.yaml (sequential blues — reflète la progression).

Usage :
    python couleur_viz.py
    python couleur_viz.py --dpi 200
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COULEUR_DIR = SCRIPT_DIR.parent / "4_data_et_viz" / "couleurs"
RESULTS_DIR = COULEUR_DIR

# Ordre canonique des phases pour toutes les figures
PHASES_ORDER = ["P1", "P2", "P3"]


# ── Chargement CSV ──────────────────────────────────────────────────────────

def charger_csv(path: Path) -> list[dict]:
    """Lit un CSV en liste de dicts."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Figure 1 & 2 : scatter PCA ──────────────────────────────────────────────

def scatter_pca_phase(rows: list[dict], phase_colors: dict, out: Path, dpi: int):
    """Scatter PC1×PC2 coloré par phase."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # On trace P1 en premier (fond), P3 en dernier (dessus) — met l'accent
    # sur la phase qui nous intéresse pour la thèse
    for phase in PHASES_ORDER:
        pts = [(float(r["pc1"]), float(r["pc2"]))
               for r in rows if r["phase"] == phase]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys,
                   c=phase_colors[phase],
                   s=18, alpha=0.7,
                   edgecolors="black", linewidths=0.3,
                   label=f"{phase} (n={len(pts)})")

    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Projection PCA des histogrammes HSV — coloré par phase", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()


def scatter_pca_cluster(rows: list[dict], out: Path, dpi: int):
    """Scatter PC1×PC2 coloré par cluster K-means + marqueur par phase."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Palette clusters (qualitative, Set2)
    cluster_colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
    markers = {"P1": "o", "P2": "s", "P3": "^"}

    clusters = sorted({int(r["cluster"]) for r in rows})

    for cl in clusters:
        for phase in PHASES_ORDER:
            pts = [(float(r["pc1"]), float(r["pc2"]))
                   for r in rows
                   if int(r["cluster"]) == cl and r["phase"] == phase]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(
                xs, ys,
                c=cluster_colors[cl % len(cluster_colors)],
                marker=markers.get(phase, "x"),
                s=22, alpha=0.7,
                edgecolors="black", linewidths=0.3,
                label=f"C{cl} × {phase} (n={len(pts)})",
            )

    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA HSV — cluster K-means (couleur) × phase (marqueur)", fontsize=11)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()


# ── Figure 3 : scree plot ───────────────────────────────────────────────────

def scree_plot(var_rows: list[dict], out: Path, dpi: int):
    """Bar chart variance expliquée + courbe cumulée."""
    fig, ax1 = plt.subplots(figsize=(7, 5))

    labels = [r["composante"] for r in var_rows]
    var_ind = [float(r["variance_expliquee"]) * 100 for r in var_rows]
    var_cum = [float(r["cumul"]) * 100 for r in var_rows]

    ax1.bar(labels, var_ind, color="#2b8cbe", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Variance expliquée (%)", color="#2b8cbe")
    ax1.tick_params(axis="y", labelcolor="#2b8cbe")
    ax1.set_ylim(0, max(var_ind) * 1.15)

    for i, v in enumerate(var_ind):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(labels, var_cum, color="#e34a33", marker="o", linewidth=1.8)
    ax2.set_ylabel("Cumul (%)", color="#e34a33")
    ax2.tick_params(axis="y", labelcolor="#e34a33")
    ax2.set_ylim(0, 105)

    ax1.set_title("PCA — variance expliquée par composante", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()


# ── Figure 4 : distributions HSV par phase ──────────────────────────────────

def distributions_phase(stats_rows: list[dict], pca_rows: list[dict],
                        phase_colors: dict, out: Path, dpi: int):
    """Boxplots de 4 métriques HSV par phase."""
    # On joint les phases depuis pca_rows (qui contient la phase) vers stats_rows
    phase_by_mid = {int(r["message_id"]): r["phase"] for r in pca_rows}

    metrics = [
        ("hsv_entropy",  "Entropie HSV (bits)",       "Diversité chromatique"),
        ("hsv_s_mean",   "Saturation moyenne (0–255)", "Richesse couleur"),
        ("hsv_v_mean",   "Luminosité moyenne (0–255)", "Clarté"),
        ("hsv_s_inter",  "Écart-type S inter-frames", "Cohérence éditoriale"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for ax, (field, ylab, title) in zip(axes, metrics):
        data_by_phase = {p: [] for p in PHASES_ORDER}
        for r in stats_rows:
            mid = int(r["message_id"])
            phase = phase_by_mid.get(mid)
            if phase in data_by_phase:
                try:
                    data_by_phase[phase].append(float(r[field]))
                except (ValueError, KeyError):
                    continue

        data = [data_by_phase[p] for p in PHASES_ORDER]
        bp = ax.boxplot(
            data,
            tick_labels=PHASES_ORDER,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, phase in zip(bp["boxes"], PHASES_ORDER):
            patch.set_facecolor(phase_colors[phase])
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylab, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    fig.suptitle("Distributions HSV par phase", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Visualisations PCA + distributions HSV par phase",
        has_input=False, has_output=False,
    )
    parser.add_argument("--dpi", type=int, default=150,
                        help="Résolution PNG (défaut: 150)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("couleurs_viz", cfg=cfg)

    # Palette depuis config.yaml (sequential blues)
    phase_colors = {p: cfg["phases"][p]["color"] for p in PHASES_ORDER}

    # Vérifications entrées
    pca_csv      = RESULTS_DIR / "couleur_pca.csv"
    clusters_csv = RESULTS_DIR / "couleur_clusters.csv"
    var_csv      = RESULTS_DIR / "couleur_pca_variance.csv"
    stats_csv    = RESULTS_DIR / "couleur_stats.csv"

    for p in [pca_csv, clusters_csv, var_csv, stats_csv]:
        if not p.is_file():
            log.error(f"CSV introuvable : {p}. Lancer couleur_batch + couleur_espace d'abord.")
            sys.exit(1)

    pca_rows      = charger_csv(pca_csv)
    clusters_rows = charger_csv(clusters_csv)
    var_rows      = charger_csv(var_csv)
    stats_rows    = charger_csv(stats_csv)

    # ── Figures ────────────────────────────────────────────────────────────
    out_phase   = RESULTS_DIR / "couleur_pca_scatter_phase.png"
    out_cluster = RESULTS_DIR / "couleur_pca_scatter_cluster.png"
    out_scree   = RESULTS_DIR / "couleur_pca_scree.png"
    out_dist    = RESULTS_DIR / "couleur_distributions_phase.png"

    log.info("Scatter PCA × phase...")
    scatter_pca_phase(pca_rows, phase_colors, out_phase, args.dpi)
    log.info(f"  → {out_phase}")

    log.info("Scatter PCA × cluster...")
    scatter_pca_cluster(clusters_rows, out_cluster, args.dpi)
    log.info(f"  → {out_cluster}")

    log.info("Scree plot...")
    scree_plot(var_rows, out_scree, args.dpi)
    log.info(f"  → {out_scree}")

    log.info("Distributions HSV par phase...")
    distributions_phase(stats_rows, pca_rows, phase_colors, out_dist, args.dpi)
    log.info(f"  → {out_dist}")


if __name__ == "__main__":
    main()
