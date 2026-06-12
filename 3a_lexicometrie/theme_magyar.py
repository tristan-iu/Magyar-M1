"""
Thème sémiologique Magyar — matplotlib
Mémoire M1 · Paris 1 Panthéon-Sorbonne

Même logique et mêmes palettes que le bloc R (r_source.R).
Fondé sur Bertin (1967), Tufte (1983), Cleveland & McGill,
Lambert & Zanin (2016).

Usage :
    from theme_magyar import *
    apply_theme()

    fig, ax = plt.subplots()
    ax.plot(dates, vals, color=PAL_PHASE["1_Artisanal"])
    add_phase_lines(ax)
    format_date_axis(ax)
    add_source(fig, "Source : messages_ffrprobe.jsonl")
    fig.savefig("figure.svg")  # SVG recommandé pour le rendu vectoriel
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
import warnings

# ═══════════════════════════════════════════════════════════
# 1. PALETTES (identiques au R)
# ═══════════════════════════════════════════════════════════
#
# Convention de clé : "1_Artisanal" (sans préfixe "P"). C'est le format
# obtenu APRÈS `categories_semantiques.phase_sans_prefixe()`, qui retire le
# "P" des valeurs brutes des CSV ("P1_Artisanal" → "1_Artisanal").
# À ne pas confondre avec `categories_semantiques.PHASE_SHORT`, qui mappe
# l'autre branche : valeur brute "P1_Artisanal" → étiquette courte "P1".
# Les deux ne sont donc pas redondants — ce sont deux étapes distinctes du
# pipeline de normalisation des phases.

PAL_PHASE = {
    "1_Artisanal":      "#2166AC",
    "2_Semi-pro":       "#D6804B",
    "3_Institutionnel": "#4D4D4D",
}

LBL_PHASE = {
    "1_Artisanal":      "Artisanal (sept. 2022 \u2013 d\u00e9c. 2023)",
    "2_Semi-pro":       "Semi-pro (janv. \u2013 sept. 2024)",
    "3_Institutionnel": "Institutionnel (oct. 2024 \u2013 juin 2025)",
}

LBL_PHASE_SHORT = {
    "1_Artisanal":      "Artisanal",
    "2_Semi-pro":       "Semi-pro",
    "3_Institutionnel": "Institutionnel",
}

# Cat\u00e9gorielle (6 max \u2014 seuil pr\u00e9-attentif Bertin)
PAL_CAT = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#525252"]

# S\u00e9quentielle (heatmaps)
PAL_SEQ_LOW  = "#F7FBFF"
PAL_SEQ_HIGH = "#08519C"
SEQ_CMAP     = "Blues"

# Dates fronti\u00e8res de phase
PHASE_DATES = [datetime(2024, 1, 1), datetime(2024, 10, 1)]
PHASE_LABELS_AT_BREAK = ["Semi-pro", "Instit."]


# ═══════════════════════════════════════════════════════════
# 2. TH\u00c8ME (rcParams)
# ═══════════════════════════════════════════════════════════

def apply_theme(base_size=11):
    """
    Configure matplotlib rcParams.
    - Fond blanc, grille Y l\u00e9g\u00e8re, pas de grille X (Tufte)
    - Sans-serif simple (Calibri)
    - 300 dpi savefig
    """
    mpl.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "savefig.facecolor":    "white",

        "axes.grid":            True,
        "axes.grid.axis":       "y",
        "grid.color":           "#D9D9D9",
        "grid.linewidth":       0.3,
        "axes.axisbelow":       True,

        "axes.linewidth":       0.4,
        "axes.edgecolor":       "#333333",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.color":          "#1A1A1A",
        "ytick.color":          "#1A1A1A",
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.size":     3,
        "ytick.major.size":     3,
        "xtick.minor.size":     0,
        "ytick.minor.size":     0,

        "font.family":          "sans-serif",
        "font.size":            base_size,
        "axes.titlesize":       base_size + 3,
        "axes.titleweight":     "bold",
        "axes.labelsize":       base_size,
        "axes.labelcolor":      "#1A1A1A",
        "xtick.labelsize":      base_size - 1,
        "ytick.labelsize":      base_size - 1,

        "legend.frameon":       False,
        "legend.fontsize":      base_size - 1,

        "lines.linewidth":      1.5,
        "lines.markersize":     4,

        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        "figure.figsize":       (10, 5.5),
    })


# ═══════════════════════════════════════════════════════════
# 3. HELPERS
# ═══════════════════════════════════════════════════════════

def add_phase_lines(ax, dates=None, labels=None,
                    colour="#888888", linestyle="--", linewidth=0.5,
                    label_size=8):
    """Fronti\u00e8res de phase : vlines + labels. Appeler APR\u00c8S le trac\u00e9."""
    if dates is None:
        dates = PHASE_DATES
    if labels is None:
        labels = PHASE_LABELS_AT_BREAK

    ymin, ymax = ax.get_ylim()
    for d, lbl in zip(dates, labels):
        ax.axvline(d, ls=linestyle, color=colour, lw=linewidth, zorder=1)
        ax.text(d, ymax * 0.97, f"  \u2190 {lbl}",
                fontsize=label_size, color=colour, fontstyle="italic",
                ha="left", va="top")


def add_phase_bands(ax, alpha=0.06):
    """Bandes color\u00e9es de fond par phase."""
    spans = [
        (datetime(2022, 9, 2),  datetime(2023, 12, 31), PAL_PHASE["1_Artisanal"]),
        (datetime(2024, 1, 1),  datetime(2024, 9, 30),  PAL_PHASE["2_Semi-pro"]),
        (datetime(2024, 10, 1), datetime(2025, 6, 30),  PAL_PHASE["3_Institutionnel"]),
    ]
    for d0, d1, c in spans:
        ax.axvspan(d0, d1, color=c, alpha=alpha, zorder=0)


def format_date_axis(ax, interval_months=3, fmt="%b\\n%Y", rotation=0):
    """Axe X dates horizontal, espacement r\u00e9gulier (pas temporel constant)."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval_months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="center")


def add_source(fig, text, fontsize=7, color="#999999"):
    """Source en bas \u00e0 gauche (toujours citer les donn\u00e9es)."""
    fig.text(0.01, 0.01, text, fontsize=fontsize, color=color, ha="left", va="bottom")


def phase_legend_handles(short=True):
    """Handles de l\u00e9gende pour les 3 phases."""
    lbls = LBL_PHASE_SHORT if short else LBL_PHASE
    return [
        Line2D([0], [0], color=PAL_PHASE[k], lw=2.5, label=lbls[k])
        for k in PAL_PHASE
    ]


def check_n_categories(values, max_n=8, var_name="variable"):
    """Alerte si > seuil pr\u00e9-attentif (Bertin : 8-12 max)."""
    n = len(set(v for v in values if v is not None))
    if n > max_n:
        warnings.warn(
            f"[Bertin] {var_name} a {n} cat\u00e9gories (seuil = {max_n}). "
            f"Regrouper en 'Autres'.",
            stacklevel=2
        )
    return n
