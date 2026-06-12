#!/usr/bin/env python3
"""
plot_lexico.py — Figures lexicométriques du corpus Magyar.

Ce que ce script lit : les CSVs produits par lexicometrie.py dans --output-dir :
    lemmes_{caption,dialogue,ocr,combined}.csv
    tfidf_{caption,dialogue,combined}.csv
    stats_phases.csv
    temporal_stats_{...}.csv
    volcano_{...}.csv

Ce qu'il produit (PNG dans 4_data_et_viz/, soit le parent de --output-dir) :
    fig_tfidf_{src}.png          barplot TF-IDF horizontal par phase
    fig_heatmap_{src}.png        heatmap lemmes × périodes
    fig_timeseries_{src}.png     courbes de fréquence de termes dans le temps
    fig_ttr_{src}.png            TTR + volume de tokens par période
    fig_volcano_{src}.png        volcano plot (log2FC vs -log10 p_adj BH)
    fig_stats.png                barplot stats descriptives (tokens, types, TTR, hapax)

Usage :
    python plot_lexico.py --output-dir 4_data_et_viz/lexico
    python plot_lexico.py --output-dir ... --track-terms дрон бригада збір

Options CLI notables : --track-terms, --fdr-alpha, --fc-threshold, --top-heatmap, --top-n

Dépendances : matplotlib, numpy, pandas
"""

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config  # noqa: E402

# CSV lus dans 4_data_et_viz/lexico/ ; figures PNG écrites un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config — couleurs et style depuis config.yaml
# ---------------------------------------------------------------------------

_cfg_cache = None


def _get_cfg(config_path=None):
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config(config_path)
    return _cfg_cache


def _get_phase_colors(cfg):
    """Retourne une liste de couleurs indexée par ordre de phase."""
    colors   = [cfg["phases"][pid].get("color", "#999999")
                for pid in sorted(cfg.get("phases", {}).keys())]
    fallback = ["#9B59B6", "#E74C3C", "#1ABC9C"]
    return colors + fallback


def _get_plt_style(cfg):
    """Retourne les rcParams matplotlib depuis config.yaml."""
    return cfg.get("viz", {}).get("plt_style", {
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })


def _construire_phases_depuis_cfg(cfg):
    """
    Reconstruit la liste (start_dt, end_dt, label) depuis config.yaml
    pour superposer les zones de phases sur les graphiques temporels.
    """
    phases = []
    for pid in sorted(cfg.get("phases", {}).keys()):
        pdata = cfg["phases"][pid]
        start = datetime.strptime(pdata["start"], "%Y-%m-%d")
        end   = datetime.strptime(pdata["end"],   "%Y-%m-%d")
        label = f"{pid}_{pdata.get('label', '')}"
        color = pdata.get("color", "#BBBBBB")
        phases.append((start, end, label, color))
    return phases


# ---------------------------------------------------------------------------
# Chargement des données depuis les CSVs
# ---------------------------------------------------------------------------

def charger_compteurs_par_periode(lemmes_csv):
    """
    Reconstruit les compteurs par période à partir de lemmes_{src}.csv.

    Retourne :
      period_counters : {period: Counter({lemma: count})}
      period_totals   : {period: int}
      phase_of_period : {period: phase_label} (premier trouvé, pour les zones)
    """
    df = pd.read_csv(lemmes_csv, dtype=str)
    period_counters = defaultdict(Counter)
    period_totals   = defaultdict(int)
    phase_of_period = {}

    for _, row in df.iterrows():
        period = row.get("period", "")
        lemma  = row.get("lemma",  "")
        phase  = row.get("phase",  None)
        if period and lemma:
            period_counters[period][lemma] += 1
            period_totals[period]          += 1
            if period not in phase_of_period and phase and phase != "nan":
                phase_of_period[period] = phase

    return period_counters, period_totals, phase_of_period


def _verifier_csv(path, label):
    """Retourne True si le CSV existe et n'est pas vide, en affichant un avertissement sinon."""
    if not os.path.exists(path):
        print(f"  [SKIP] {label} : fichier introuvable ({path})")
        return False
    if os.path.getsize(path) == 0:
        print(f"  [SKIP] {label} : fichier vide ({path})")
        return False
    return True


# ---------------------------------------------------------------------------
# Figure 1 — TF-IDF barplot horizontal par phase
# ---------------------------------------------------------------------------

def tracer_tfidf(output_dir, fig_dir, src, top_n=15):
    """Barplot horizontal des top N lemmes TF-IDF par phase."""
    csv_path = os.path.join(output_dir, f"tfidf_{src}.csv")
    if not _verifier_csv(csv_path, f"TF-IDF {src}"):
        return

    df  = pd.read_csv(csv_path)
    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))
    COLORS = _get_phase_colors(cfg)

    phases   = df["phase"].unique()
    n_phases = len(phases)
    fig, axes = plt.subplots(
        1, n_phases,
        figsize=(6 * n_phases, max(6, top_n * 0.4)),
        squeeze=False,
    )

    for idx, phase in enumerate(phases):
        ax  = axes[0][idx]
        sub = df[df["phase"] == phase].head(top_n).sort_values("tfidf")
        ax.barh(sub["lemma"], sub["tfidf"], color=COLORS[idx % len(COLORS)])
        ax.set_title(phase, fontsize=12, fontweight="bold")
        ax.set_xlabel("TF-IDF")
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(f"Top {top_n} TF-IDF — {src}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(fig_dir, f"fig_tfidf_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Heatmap temporelle : top N lemmes × périodes
# ---------------------------------------------------------------------------

def tracer_heatmap_temporelle(output_dir, fig_dir, src, top_n=30, min_count=3):
    """
    Heatmap : lignes = top N lemmes globaux, colonnes = périodes mensuelles.
    Valeur = fréquence relative pour 1 000 tokens — visualisation centrale
    de l'évolution du vocabulaire dans le temps.
    """
    csv_path = os.path.join(output_dir, f"lemmes_{src}.csv")
    if not _verifier_csv(csv_path, f"Heatmap {src}"):
        return

    period_counters, period_totals, _ = charger_compteurs_par_periode(csv_path)
    periods = sorted(period_counters.keys())

    # On sélectionne les top N lemmes par fréquence globale (filtrés min_count)
    global_counter = Counter()
    for c in period_counters.values():
        global_counter.update(c)
    top_lemmes = [w for w, cnt in global_counter.most_common() if cnt >= min_count][:top_n]

    if not top_lemmes or not periods:
        print(f"  [SKIP] Heatmap {src} : vocabulaire insuffisant.")
        return

    # Matrice fréquence relative pour 1 000 tokens
    matrix = np.zeros((len(top_lemmes), len(periods)))
    for j, period in enumerate(periods):
        total = period_totals[period] or 1
        for i, lemma in enumerate(top_lemmes):
            matrix[i, j] = period_counters[period].get(lemma, 0) / total * 1000

    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))
    fig_h = max(8, len(top_lemmes) * 0.38)
    fig_w = max(12, len(periods)   * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Fréq. / 1 000 tokens")

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_lemmes)))
    ax.set_yticklabels(top_lemmes, fontsize=8)
    ax.set_title(f"Heatmap temporelle — {src}  (top {top_n} lemmes)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Période", fontsize=10)

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_heatmap_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Time series : évolution de termes-clés dans le temps
# ---------------------------------------------------------------------------

def tracer_serie_termes(output_dir, fig_dir, src, track_terms=None, min_count=2):
    """
    Courbes de fréquence relative (/ 1 000 tokens) pour chaque terme suivi.
    Si track_terms est None ou vide, auto-sélection des 8 lemmes les plus fréquents.
    """
    csv_path = os.path.join(output_dir, f"lemmes_{src}.csv")
    if not _verifier_csv(csv_path, f"Time series {src}"):
        return

    period_counters, period_totals, _ = charger_compteurs_par_periode(csv_path)
    periods = sorted(period_counters.keys())

    if not track_terms:
        global_counter = Counter()
        for c in period_counters.values():
            global_counter.update(c)
        track_terms = [w for w, cnt in global_counter.most_common(8) if cnt >= min_count]

    # On exclut les termes avec trop peu d'occurrences au global
    valid_terms = [t for t in track_terms
                   if sum(period_counters[p].get(t, 0) for p in periods) >= min_count]
    if not valid_terms:
        print(f"  [SKIP] Time series {src} : aucun terme valide.")
        return

    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))
    fig, ax = plt.subplots(figsize=(max(14, len(periods) * 0.6), 6))
    cmap    = matplotlib.colormaps["tab10"].resampled(len(valid_terms))
    x_vals  = range(len(periods))

    for k, term in enumerate(valid_terms):
        freqs = [
            period_counters[p].get(term, 0) / (period_totals[p] or 1) * 1000
            for p in periods
        ]
        ax.plot(x_vals, freqs, marker="o", markersize=4,
                label=term, color=cmap(k), linewidth=1.8)

    ax.set_xticks(list(x_vals))
    ax.set_xticklabels(periods, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Fréquence / 1 000 tokens", fontsize=10)
    ax.set_title(f"Évolution temporelle des termes — {src}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_timeseries_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 4 — TTR et volume de tokens dans le temps
# ---------------------------------------------------------------------------

def tracer_ttr_temporel(output_dir, fig_dir, src):
    """
    TTR (types / tokens) et volume de tokens par période.
    Les zones de phases sont superposées pour contextualiser les ruptures.
    """
    csv_path = os.path.join(output_dir, f"temporal_stats_{src}.csv")
    if not _verifier_csv(csv_path, f"TTR {src}"):
        return

    df      = pd.read_csv(csv_path).sort_values("period")
    periods = df["period"].tolist()
    ttr_vals = df["ttr"].tolist()
    tok_vals = df["tokens"].tolist()

    if not periods:
        return

    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                    figsize=(max(14, len(periods) * 0.6), 8),
                                    sharex=True)
    x = range(len(periods))

    ax1.plot(x, ttr_vals, color="#4A90D9", linewidth=2, marker="o", markersize=3)
    ax1.fill_between(x, ttr_vals, alpha=0.15, color="#4A90D9")
    ax1.set_ylabel("TTR (types / tokens)", fontsize=10)
    ax1.set_title(f"Richesse lexicale (TTR) et volume — {src}",
                  fontsize=13, fontweight="bold")

    ax2.bar(x, tok_vals, color="#E07B39", alpha=0.7)
    ax2.set_ylabel("Nb tokens", fontsize=10)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(periods, rotation=60, ha="right", fontsize=8)

    # On superpose les zones de phases comme repères visuels
    phases_cfg = _construire_phases_depuis_cfg(cfg)
    for start_dt, end_dt, label, color in phases_cfg:
        s_label = start_dt.strftime("%Y-%m")
        e_label = end_dt.strftime("%Y-%m")
        if s_label in periods and e_label in periods:
            xi = periods.index(s_label)
            xf = periods.index(e_label)
            for axi in (ax1, ax2):
                axi.axvspan(xi - 0.5, xf + 0.5, alpha=0.08, color=color, label=label)
            ax1.text((xi + xf) / 2, max(ttr_vals) * 0.95, label,
                     ha="center", fontsize=7.5, color=color, style="italic")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_ttr_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 5 — Volcano plot
# ---------------------------------------------------------------------------

def tracer_volcano(output_dir, fig_dir, src, fdr_alpha=0.05, fc_threshold=1.0):
    """
    Volcano plot : log2(fold-change) vs -log10(p_adj Benjamini-Hochberg).
    Points rouges = enrichis dans la phase B, bleus = appauvris.
    Lignes pointillées = seuils FDR et |log2FC|.
    """
    csv_path = os.path.join(output_dir, f"volcano_{src}.csv")
    if not _verifier_csv(csv_path, f"Volcano {src}"):
        return

    df  = pd.read_csv(csv_path)
    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))

    neg_log10_threshold = -math.log10(fdr_alpha) if fdr_alpha > 0 else 1.3
    comparisons = df["comparison"].unique()
    n_comp      = len(comparisons)
    fig, axes   = plt.subplots(1, n_comp, figsize=(7 * n_comp, 7), squeeze=False)

    for idx, comp in enumerate(comparisons):
        ax  = axes[0][idx]
        sub = df[df["comparison"] == comp].copy()

        sig  = (sub["neg_log10_padj"] >= neg_log10_threshold) & (sub["log2fc"].abs() >= fc_threshold)
        up   = sig & (sub["log2fc"] > 0)
        down = sig & (sub["log2fc"] < 0)
        ns   = ~sig

        ax.scatter(sub.loc[ns,   "log2fc"], sub.loc[ns,   "neg_log10_padj"],
                   c="#CCCCCC", alpha=0.3, s=12, label="non sig.", zorder=1)
        ax.scatter(sub.loc[up,   "log2fc"], sub.loc[up,   "neg_log10_padj"],
                   c="#E74C3C", alpha=0.8, s=30, label=f"↑ enrichi (FC>{fc_threshold})", zorder=2)
        ax.scatter(sub.loc[down, "log2fc"], sub.loc[down, "neg_log10_padj"],
                   c="#4A90D9", alpha=0.8, s=30, label=f"↓ appauvri (FC<-{fc_threshold})", zorder=2)

        ax.axhline(neg_log10_threshold, color="grey", linestyle="--", linewidth=0.8)
        ax.axvline( fc_threshold,       color="grey", linestyle="--", linewidth=0.8)
        ax.axvline(-fc_threshold,       color="grey", linestyle="--", linewidth=0.8)

        # On annote les 6 points les plus significatifs dans chaque direction
        to_label = pd.concat([
            sub[up  ].nlargest(6, "neg_log10_padj"),
            sub[down].nlargest(6, "neg_log10_padj"),
        ])
        for _, row in to_label.iterrows():
            ax.annotate(row["lemma"], (row["log2fc"], row["neg_log10_padj"]),
                        fontsize=7.5, alpha=0.9,
                        textcoords="offset points", xytext=(4, 4))

        sig_up   = int(up.sum())
        sig_down = int(down.sum())
        ax.set_xlabel("log₂(fold change)", fontsize=11)
        ax.set_ylabel("-log₁₀(p adj. BH)", fontsize=11)
        ax.set_title(f"{comp}\n↑{sig_up} ↓{sig_down} sig. (FDR<{fdr_alpha})",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Volcano — {src}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = os.path.join(fig_dir, f"fig_volcano_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 6 — Stats descriptives globales
# ---------------------------------------------------------------------------

def tracer_stats(output_dir, fig_dir):
    """Barplot 2×2 des stats lexicométriques (tokens, types, TTR, hapax) par phase."""
    csv_path = os.path.join(output_dir, "stats_phases.csv")
    if not _verifier_csv(csv_path, "Stats phases"):
        return

    df  = pd.read_csv(csv_path)
    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))

    metrics = ["tokens", "types", "ttr", "hapax"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flatten(), metrics):
        pivot = df.pivot(index="phase", columns="source", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=30)
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    fig.suptitle("Stats lexicométriques par phase et source",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig_stats.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Figure 7 — Courbe de Pareto (distribution rang-fréquence / Zipf)
# ---------------------------------------------------------------------------

def tracer_pareto(output_dir, fig_dir, src, min_count=1):
    """
    Courbe rang-fréquence en log-log (loi de Zipf).

    Trois zones annotées :
      - Tête : mots ultra-fréquents (déjà filtrés par POS/stopwords)
      - Corps : vocabulaire thématique — zone de travail
      - Queue : hapax et quasi-hapax

    Justifie visuellement le seuil --min-count de lexicometrie.py
    (les hapax dominent en queue de distribution).
    """
    csv_path = os.path.join(output_dir, f"lemmes_{src}.csv")
    if not _verifier_csv(csv_path, f"Pareto {src}"):
        return

    df = pd.read_csv(csv_path, usecols=["lemma"])
    freq = df["lemma"].value_counts().sort_values(ascending=False)

    ranks = np.arange(1, len(freq) + 1)
    freqs = freq.values

    # Zones : tête = top 20, queue = hapax (freq == 1)
    n_hapax = int((freqs == 1).sum())
    hapax_start = len(freqs) - n_hapax

    cfg = _get_cfg()
    matplotlib.rcParams.update(_get_plt_style(cfg))
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.loglog(ranks, freqs, color="#2C3E50", linewidth=1.5, alpha=0.9)

    # Zone tête
    ax.axvspan(1, 20, alpha=0.10, color="#E74C3C")
    ax.text(5, freqs[0] * 0.6, "Tête\n(mots outils)",
            fontsize=9, color="#E74C3C", ha="center", style="italic")

    # Zone queue (hapax)
    if hapax_start > 0:
        ax.axvspan(hapax_start, len(freqs), alpha=0.10, color="#3498DB")
        ax.text(hapax_start + n_hapax * 0.3, 1.3,
                f"Queue — hapax\n({n_hapax} types, {n_hapax/len(freqs)*100:.0f}%)",
                fontsize=9, color="#3498DB", ha="center", style="italic")

    # Zone corps
    mid_rank = int((20 + hapax_start) / 2) if hapax_start > 20 else 100
    mid_freq = freqs[min(mid_rank, len(freqs) - 1)]
    ax.annotate("Corps\n(vocabulaire thématique)",
                xy=(mid_rank, mid_freq),
                fontsize=9, color="#27AE60", ha="center", style="italic",
                xytext=(mid_rank, mid_freq * 5),
                arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))

    # Annotation des 10 premiers lemmes
    for i in range(min(10, len(freq))):
        ax.annotate(freq.index[i], (ranks[i], freqs[i]),
                    fontsize=7, alpha=0.8, rotation=30,
                    textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("Rang (log)", fontsize=11)
    ax.set_ylabel("Fréquence (log)", fontsize=11)
    ax.set_title(f"Distribution rang-fréquence (Zipf) — {src}\n"
                 f"{len(freq)} types, {int(freqs.sum())} tokens, "
                 f"{n_hapax} hapax ({n_hapax/len(freq)*100:.0f}%)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_pareto_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Figures lexicométriques — corpus Magyar")
    p.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                   help="Dossier des CSVs lexicometrie.py (défaut : 4_data_et_viz/lexico)")
    p.add_argument("--fig-dir", default=None,
                   help="Dossier des figures (défaut : parent de --output-dir)")
    p.add_argument("--config", default=None,
                   help="Chemin vers config.yaml (défaut : 0_config/config.yaml)")
    p.add_argument("--sources", nargs="*",
                   default=["caption", "dialogue", "combined"],
                   help="Sources à traiter (défaut : caption dialogue combined)")
    p.add_argument("--top-n", type=int, default=15,
                   help="Top N lemmes pour les barplots TF-IDF (défaut : 15)")
    p.add_argument("--top-heatmap", type=int, default=30,
                   help="Top N lemmes pour la heatmap (défaut : 30)")
    p.add_argument("--track-terms", nargs="*", default=None,
                   help="Termes à suivre dans le time-series (ex : дрон бригада)")
    p.add_argument("--fdr-alpha", type=float, default=0.05,
                   help="Seuil FDR pour le volcano (défaut : 0.05)")
    p.add_argument("--fc-threshold", type=float, default=1.0,
                   help="Seuil |log2FC| pour le volcano (défaut : 1.0)")
    return p.parse_args()


def main():
    args = parse_args()

    # On charge le config une fois (renseigne couleurs et phases)
    _get_cfg(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Génération des figures depuis : {args.output_dir}")
    print(f"Figures écrites dans : {fig_dir}")
    print(f"Sources : {args.sources}\n")

    for src in args.sources:
        print(f"── {src} ──")
        tracer_tfidf(args.output_dir, fig_dir, src, top_n=args.top_n)
        tracer_heatmap_temporelle(args.output_dir, fig_dir, src, top_n=args.top_heatmap)
        tracer_serie_termes(args.output_dir, fig_dir, src, track_terms=args.track_terms)
        tracer_ttr_temporel(args.output_dir, fig_dir, src)
        tracer_volcano(args.output_dir, fig_dir, src,
                     fdr_alpha=args.fdr_alpha, fc_threshold=args.fc_threshold)
        tracer_pareto(args.output_dir, fig_dir, src)

    tracer_stats(args.output_dir, fig_dir)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
