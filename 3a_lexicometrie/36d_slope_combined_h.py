"""
Slopechart TF-IDF P1 → P2 — captions (haut) + dialogue (bas), version paysage.

Variante combinée du slopechart (cf. `36c_slope_tfidf_p1p2.py`) en layout
horizontal : rang 1 à droite, rang 15 à gauche, P1 en haut et P2 en bas.
Les deux panneaux (captions + dialogue) sont empilés sur une même figure.

Source : 4_data_et_viz/lexico/tfidf_{caption,dialogue}.csv
Sortie : 4_data_et_viz/slope_tfidf_p1p2_combined.png

Lancement direct ou via `tableaux_tfidf.py --variant slope_combined`.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories_semantiques import (  # noqa: E402
    categorize, TRADUCTIONS as _TRAD_BASE, COLORS, phase_sans_prefixe,
    STOPWORDS_DIALOGUE,
)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "4_data_et_viz")
os.makedirs(OUT, exist_ok=True)

C_MIL = COLORS["militaire"]["bar"]
C_FIN = COLORS["finance"]["bar"]
C_ASS = COLORS["associes"]["bar"]
C_OUT = "#C8CDD2"


def couleur_lemma(lemma):
    """Retourne la couleur barre selon la catégorie sémantique."""
    return COLORS[categorize(lemma)]["bar"]

# Traductions : base partagée + overrides locaux
TRADUCTIONS = {
    **_TRAD_BASE,
    "жало": "dard/FPV",
    "підрозділ": "unité mil.",
    "хробак": "ver/FPV",
    "реквізит": "coord. banc.",
    "морський": "Marines",
    "рер": "REB",
    "пм": "PM (unité)",
    "засіб": "moyen/engin",
    "кількість": "quantité",
}

TOP_N  = 20   # union des top 20 pour capter les mouvements
SHOW_N = 15   # on n'affiche que les 15 premiers


# ---------------------------------------------------------------------------
# Préparer les données
# ---------------------------------------------------------------------------
def preparer_donnees_slope(csv_path, stopwords=None, top_n=20):
    df = pd.read_csv(csv_path)
    # Normalise "P1_Artisanal" → "1_Artisanal" pour matcher les libellés ci-dessous
    df["phase"] = phase_sans_prefixe(df["phase"])
    df = df[~df["lemma"].str.match(r'^\d+$')]
    if stopwords:
        df = df[~df["lemma"].isin(stopwords)]

    p1 = (df[df["phase"] == "1_Artisanal"]
          .sort_values("tfidf", ascending=False).reset_index(drop=True))
    p1["rank"] = range(1, len(p1) + 1)

    p2 = (df[df["phase"] == "2_Semi-pro"]
          .sort_values("tfidf", ascending=False).reset_index(drop=True))
    p2["rank"] = range(1, len(p2) + 1)

    top_p1 = set(p1.head(top_n)["lemma"])
    top_p2 = set(p2.head(top_n)["lemma"])

    rank_p1 = dict(zip(p1["lemma"], p1["rank"]))
    rank_p2 = dict(zip(p2["lemma"], p2["rank"]))

    rows = []
    for lemma in top_p1 | top_p2:
        rows.append({
            "lemma": lemma,
            "rank_p1": rank_p1.get(lemma),
            "rank_p2": rank_p2.get(lemma),
            "in_p1": lemma in top_p1,
            "in_p2": lemma in top_p2,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dessiner un slope horizontal dans un ax
#   X = rang (inversé : 1 à droite, SHOW_N à gauche)
#   Y = P1 en haut (y=1), P2 en bas (y=0)
#   Labels au-dessus (P1) et en dessous (P2), en rotation
# ---------------------------------------------------------------------------
def tracer_slope_horizontal(ax, sdf, show_n, panel_title):
    # Quels mots sont dans le top visible
    vis_p1 = set(sdf.loc[sdf["in_p1"] & (sdf["rank_p1"] <= show_n), "lemma"])
    vis_p2 = set(sdf.loc[sdf["in_p2"] & (sdf["rank_p2"] <= show_n), "lemma"])
    visible = vis_p1 | vis_p2
    sdf_vis = sdf[sdf["lemma"].isin(visible)].copy()

    # X : rang inversé → x = show_n + 1 - rank (rang 1 à droite)
    def rank_to_x(rank):
        return show_n + 1 - rank

    # « hors top » : position X à gauche du rang show_n
    exit_x = -0.5

    y_top = 1.0   # P1
    y_bot = 0.0   # P2

    ax.set_xlim(exit_x - 1.0, show_n + 1.5)
    ax.set_ylim(-1.1, 2.1)
    ax.axis("off")

    # Lignes horizontales P1 / P2
    ax.axhline(y_top, color="#CCCCCC", linewidth=0.8,
               xmin=0.02, xmax=0.98)
    ax.axhline(y_bot, color="#CCCCCC", linewidth=0.8,
               xmin=0.02, xmax=0.98)

    # Labels P1 / P2 à droite
    ax.text(show_n + 1.3, y_top, "P1", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#555555")
    ax.text(show_n + 1.3, y_bot, "P2", ha="left", va="center",
            fontsize=9, fontweight="bold", color="#555555")

    # Titre du panneau
    ax.set_title(panel_title, fontsize=12, fontweight="bold",
                 color="#2B2B2B", pad=8)

    # Séparateur « hors top »
    sep_x = exit_x + 0.5
    ax.axvline(sep_x, color="#DDDDDD", linewidth=0.6, linestyle="--",
               ymin=0.15, ymax=0.85)
    ax.text(exit_x - 0.3, 0.5, f"hors\ntop {show_n}",
            ha="center", va="center", fontsize=6, color="#AAAAAA",
            style="italic")

    # Numéros de rang le long des axes
    for r in range(1, show_n + 1):
        x = rank_to_x(r)
        ax.text(x, y_top + 0.04, str(r), ha="center", va="bottom",
                fontsize=5.5, color="#CCCCCC")
        ax.text(x, y_bot - 0.04, str(r), ha="center", va="top",
                fontsize=5.5, color="#CCCCCC")

    # Collecter les positions occupées pour alterner les décalages
    # Trier par rang pour un placement prévisible
    rows_p1 = sorted(
        [(row["rank_p1"], row) for _, row in sdf_vis.iterrows() if row["lemma"] in vis_p1],
        key=lambda t: t[0])
    rows_p2 = sorted(
        [(row["rank_p2"], row) for _, row in sdf_vis.iterrows() if row["lemma"] in vis_p2],
        key=lambda t: t[0])

    # Compteurs pour alterner le décalage à chaque rang
    p1_idx = {}
    for i, (rk, _) in enumerate(rows_p1):
        p1_idx[rk] = i
    p2_idx = {}
    for i, (rk, _) in enumerate(rows_p2):
        p2_idx[rk] = i

    # Pentes
    for _, row in sdf_vis.iterrows():
        lemma = row["lemma"]
        col = couleur_lemma(lemma)
        trad = TRADUCTIONS.get(lemma, "")
        label = f"{lemma} ({trad})" if trad else lemma

        in_vis_p1 = lemma in vis_p1
        in_vis_p2 = lemma in vis_p2

        x1 = rank_to_x(row["rank_p1"]) if in_vis_p1 else exit_x
        x2 = rank_to_x(row["rank_p2"]) if in_vis_p2 else exit_x

        is_exit = (not in_vis_p1) or (not in_vis_p2)
        lw    = 1.6 if not is_exit else 0.8
        alpha = 0.85 if not is_exit else 0.35
        lc    = col if not is_exit else C_OUT

        ax.plot([x1, x2], [y_top, y_bot],
                color=lc, linewidth=lw, alpha=alpha, zorder=2,
                solid_capstyle="round")

        dot_s  = 28 if not is_exit else 12
        fs     = 6.0
        fc     = col
        fw     = "bold"
        rot    = 40

        if in_vis_p1:
            ax.scatter(x1, y_top, color=col, s=dot_s, zorder=3,
                       edgecolors="white", linewidths=0.3)
            ax.text(x1, y_top + 0.18, label,
                    ha="right", va="bottom", fontsize=fs, color=fc,
                    fontweight=fw, rotation=rot, rotation_mode="anchor")

        if in_vis_p2:
            ax.scatter(x2, y_bot, color=col, s=dot_s, zorder=3,
                       edgecolors="white", linewidths=0.3)
            ax.text(x2, y_bot - 0.18, label,
                    ha="left", va="top", fontsize=fs, color=fc,
                    fontweight=fw, rotation=rot, rotation_mode="anchor")


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
csv_cap = os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_caption.csv")
csv_dia = os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_dialogue.csv")

sdf_cap = preparer_donnees_slope(csv_cap, stopwords=STOPWORDS_DIALOGUE, top_n=TOP_N)
sdf_dia = preparer_donnees_slope(csv_dia, stopwords=STOPWORDS_DIALOGUE, top_n=TOP_N)

# ---------------------------------------------------------------------------
# Figure : 2 panneaux empilés, format paysage
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 16))
fig.patch.set_facecolor("white")

tracer_slope_horizontal(ax1, sdf_cap, SHOW_N, "Captions (textes publiés)")
tracer_slope_horizontal(ax2, sdf_dia, SHOW_N, "Dialogue (parole transcrite)")

# Titre global
fig.suptitle("Évolution lexicale P1 → P2 — Slopechart TF-IDF, top 15",
             fontsize=16, fontweight="bold", y=0.99, color="#2B2B2B")
fig.text(0.5, 0.965,
         "P1 Artisanal (sept. 2022 – déc. 2023)  →  P2 Semi-pro (janv. – sept. 2024)  ·  mots outils et nombres exclus",
         ha="center", fontsize=10, color="#666666")

# Légende
legend_handles = [
    Line2D([0], [0], color=C_MIL, linewidth=2.5, label="Militaire"),
    Line2D([0], [0], color=C_FIN, linewidth=2.5, label="Finance"),
    Line2D([0], [0], color=C_ASS, linewidth=2.5, label="Associés"),
    Line2D([0], [0], color=C_OUT, linewidth=1.5, alpha=0.5, label="Sorti du top 15"),
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.5, 0.003), ncol=4,
           frameon=False, fontsize=10)

fig.text(0.01, 0.003,
         "Source : TF-IDF par phase · lemmatisation spaCy uk_core_news_trf · traductions DeepL",
         fontsize=7, color="#999999", ha="left", va="bottom")

fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.12, left=0.03, right=0.97)

# Ligne séparatrice entre les deux panneaux (milieu exact entre les deux axes)
mid_y = (ax1.get_position().y0 + ax2.get_position().y1) / 2
fig.add_artist(plt.Line2D(
    [0.03, 0.97], [mid_y, mid_y],
    transform=fig.transFigure, color="#AAAAAA", linewidth=0.8))

out_path = os.path.join(OUT, "slope_tfidf_p1p2_combined.png")
fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Sauvegardé : {out_path}")
