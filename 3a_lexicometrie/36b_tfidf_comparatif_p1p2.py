"""
Tableau comparatif TF-IDF captions P1 vs P2 — côte à côte.

Affiche les 15 lemmes les plus distinctifs en P2 avec, pour chacun :
le rang TF-IDF en P2, la flèche de mouvement vis-à-vis du rang P1
(▲ monte, ▼ descend, NEW = absent du top P1), le lemme original, sa
traduction fr, le score TF-IDF (barre proportionnelle) et le rang P1.

Source : 4_data_et_viz/lexico/tfidf_caption.csv
Sortie : 4_data_et_viz/tab_tfidf_p1p2_caption.png

Lancement direct ou via `tableaux_tfidf.py --variant comparatif_p1p2`.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories_semantiques import (  # noqa: E402
    categorize, TRADUCTIONS as _TRAD_BASE,
    COLORS, HEADER_BG, HEADER_FG, phase_sans_prefixe,
)

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_caption.csv")
OUT  = os.path.join(BASE, "4_data_et_viz", "tab_tfidf_p1p2_caption.png")

TOP_N = 15

# Overrides locaux (annotations plus bavardes que la base partagée).
TRADUCTIONS = {
    **_TRAD_BASE,
    "мадяр":      "Magyar (le commandant)",
    "грн":        "hryvnia (monnaie)",
    "жало":       "dard / drone FPV",
    "збір":       "collecte de fonds",
    "підрозділ":  "unité militaire",
    "зсу":        "Forces armées (ZSU)",
    "хробак":     "ver / drone FPV",
    "хробачий":   "du ver (adj.)",
    "роберт":     "Robert (prénom)",
    "бровді":     "Brovdi (nom)",
    "реквізит":   "coordonnées bancaires",
    "обубас":     "UBAS (système d'arme)",
    "кринки":     "Krynky (lieu de combat)",
    "морський":   "maritime / Marines",
    "рер":        "guerre électronique (REB)",
    "вйо":        "allez ! (interjection)",
    "пм":         "Ptakhyky Madyara (unité)",
    "виявити":    "détecter / repérer",
}

# On ne déplie que les couleurs réutilisées dans la légende (patches).
BAR_FIN = COLORS["finance"]["bar"]
BAR_MIL = COLORS["militaire"]["bar"]
BAR_ASS = COLORS["associes"]["bar"]

# Couleurs de la colonne mouvement (variation de rang P1→P2).
C_UP   = "#1A6B2E"   # vert — monte
C_DOWN = "#C0392B"   # rouge — descend
C_NEW  = "#2166AC"   # bleu — nouveau dans le top
C_SAME = "#888888"   # gris — stable


# ---------------------------------------------------------------------------
# Charger et préparer les données
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV)
# Normalise "P1_Artisanal" → "1_Artisanal" pour matcher les libellés ci-dessous
df["phase"] = phase_sans_prefixe(df["phase"])
# Exclure les nombres purs
df = df[~df["lemma"].str.match(r'^\d+$')]

p1 = (df[df["phase"] == "1_Artisanal"]
      .sort_values("tfidf", ascending=False)
      .reset_index(drop=True))
p1["rank_p1"] = range(1, len(p1) + 1)

p2 = (df[df["phase"] == "2_Semi-pro"]
      .sort_values("tfidf", ascending=False)
      .reset_index(drop=True))
p2["rank_p2"] = range(1, len(p2) + 1)

# Top N de P2 (le tableau montre les mots les plus distinctifs en P2)
top_p2 = p2.head(TOP_N).copy()

# Joindre le rang P1
rank_p1_map = dict(zip(p1["lemma"], p1["rank_p1"]))
top_p2["rank_p1"] = top_p2["lemma"].map(rank_p1_map)
top_p2["tfidf_p1"] = top_p2["lemma"].map(dict(zip(p1["lemma"], p1["tfidf"])))

# Étiquette de mouvement P1 → P2 (texte + couleur).
def etiquette_mouvement(row):
    if pd.isna(row["rank_p1"]):
        return "NEW", C_NEW
    diff = row["rank_p1"] - row["rank_p2"]
    if diff > 0:
        return f"▲{diff}", C_UP
    elif diff < 0:
        return f"▼{abs(diff)}", C_DOWN
    else:
        return "=", C_SAME

top_p2["mvt_label"], top_p2["mvt_color"] = zip(*top_p2.apply(etiquette_mouvement, axis=1))


# ---------------------------------------------------------------------------
# Dessin du tableau comparatif
# ---------------------------------------------------------------------------
def tracer_tableau_comparatif(ax, data, max_tfidf):
    n_rows   = len(data)
    row_h    = 1.0
    title_h  = 1.1
    subhdr_h = 0.9
    total_h  = title_h + subhdr_h + n_rows * row_h

    # Colonnes : Rang P2 | Mvt | Lemme | Traduction | Score P2 (barre) | Rang P1
    # positions X des séparateurs verticaux
    SEP_RANG  = 0.85
    SEP_MVT   = 1.65
    SEP_LEMME = 3.85
    SEP_TRAD  = 6.65
    SEP_SCORE = 9.2  # après les barres, avant rang P1

    col_x      = [0.15, SEP_RANG + 0.12, SEP_MVT + 0.12, SEP_LEMME + 0.12, SEP_TRAD + 0.12, SEP_SCORE + 0.08]
    col_labels = ["Rg P2", "Mvt", "Lemme (uk)", "Traduction (fr)", "Score TF-IDF", "Rg P1"]
    separators = [SEP_RANG, SEP_MVT, SEP_LEMME, SEP_TRAD, SEP_SCORE]

    total_w = 10.2

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ---- Bandeau titre ----
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, total_h - title_h), total_w, title_h,
        boxstyle="square,pad=0", fc=HEADER_BG, ec="none", zorder=1))
    ax.text(0.3, total_h - title_h / 2,
            "Tableau — Les 15 mots les plus distinctifs en P2 (janv. – sept. 2024) — captions",
            va="center", ha="left", fontsize=8.5, fontweight="bold",
            color=HEADER_FG, zorder=2)

    # ---- Sous-en-têtes ----
    subhdr_top = total_h - title_h
    subhdr_bot = subhdr_top - subhdr_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, subhdr_bot), total_w, subhdr_h,
        boxstyle="square,pad=0", fc="#F7F7F7", ec="none", zorder=1))
    ax.axhline(subhdr_top, xmin=0, xmax=1, color="#AAAAAA", linewidth=0.6, zorder=2)
    ax.axhline(subhdr_bot, xmin=0, xmax=1, color="#CCCCCC", linewidth=0.8, zorder=2)

    for x, lbl in zip(col_x, col_labels):
        ax.text(x, subhdr_bot + subhdr_h / 2, lbl,
                va="center", ha="left", fontsize=7.5,
                fontweight="bold", color="#333333", zorder=3)

    for sep in separators:
        ax.plot([sep, sep], [subhdr_bot, subhdr_top],
                color="#CCCCCC", linewidth=0.8, zorder=2)

    # ---- Lignes de données ----
    data_top  = subhdr_bot
    bar_x     = col_x[4]
    bar_w_max = SEP_SCORE - bar_x - 0.15

    for i, (_, row) in enumerate(data.iterrows()):
        lemma    = row["lemma"]
        tfidf    = row["tfidf"]
        trad     = TRADUCTIONS.get(lemma, "—")
        cat      = categorize(lemma)
        bg       = COLORS[cat]["bg"]
        bar_c    = COLORS[cat]["bar"]
        rank_p2  = int(row["rank_p2"])
        rank_p1  = row["rank_p1"]
        mvt_lbl  = row["mvt_label"]
        mvt_col  = row["mvt_color"]

        y_bot = data_top - (i + 1) * row_h
        y_mid = y_bot + row_h * 0.5

        # Fond
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y_bot), total_w, row_h,
            boxstyle="square,pad=0", fc=bg, ec="none", zorder=1))
        ax.plot([0, total_w], [y_bot + row_h, y_bot + row_h],
                color="#E0E0E0", linewidth=0.4, zorder=2)

        for sep in separators:
            ax.plot([sep, sep], [y_bot, y_bot + row_h],
                    color="#E0E0E0", linewidth=0.5, zorder=2)

        # Rang P2
        ax.text(col_x[0] + 0.2, y_mid, str(rank_p2),
                va="center", ha="center", fontsize=9,
                fontweight="bold", color="#1A1A1A", zorder=3)

        # Mouvement
        ax.text(col_x[1] + 0.25, y_mid, mvt_lbl,
                va="center", ha="center", fontsize=8,
                fontweight="bold", color=mvt_col, zorder=3)

        # Lemme
        ax.text(col_x[2], y_mid, lemma,
                va="center", ha="left", fontsize=8.5,
                fontweight="bold", style="italic",
                color="#1A1A1A", zorder=3)

        # Traduction
        ax.text(col_x[3], y_mid, trad,
                va="center", ha="left", fontsize=8,
                color="#333333", zorder=3)

        # Barre TF-IDF
        bar_w = (tfidf / max_tfidf) * bar_w_max
        bar_h = row_h * 0.38
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, y_mid - bar_h / 2), bar_w, bar_h,
            boxstyle="square,pad=0", fc=bar_c, ec="none",
            alpha=0.85, zorder=3))
        ax.text(bar_x + bar_w + 0.1, y_mid,
                f"{tfidf:.4f}",
                va="center", ha="left", fontsize=7.5,
                color="#555555", zorder=3)

        # Rang P1
        rank_p1_str = str(int(rank_p1)) if not pd.isna(rank_p1) else "—"
        ax.text(col_x[5] + 0.3, y_mid, rank_p1_str,
                va="center", ha="center", fontsize=9,
                fontweight="bold", color="#555555", zorder=3)

    # Bordure
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), total_w, total_h,
        boxstyle="square,pad=0", fc="none",
        ec="#AAAAAA", linewidth=0.8, zorder=4))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
max_tfidf = top_p2["tfidf"].max()

fig, ax = plt.subplots(1, 1, figsize=(11, 6.2))
fig.patch.set_facecolor("white")
fig.subplots_adjust(top=0.97, bottom=0.12)

tracer_tableau_comparatif(ax, top_p2, max_tfidf)

legend_handles = [
    mpatches.Patch(color=BAR_MIL, label="Militaire"),
    mpatches.Patch(color=BAR_FIN, label="Finance"),
    mpatches.Patch(color=BAR_ASS, label="Associés"),
    mpatches.Patch(color=C_UP,   label="▲ Monte"),
    mpatches.Patch(color=C_DOWN, label="▼ Descend"),
    mpatches.Patch(color=C_NEW,  label="Nouveau"),
]

ax.legend(
    handles=legend_handles,
    loc="lower right",
    bbox_to_anchor=(1.0, -0.13),
    bbox_transform=ax.transAxes,
    ncol=6,
    frameon=False,
    fontsize=8,
)

ax.text(0.0, -0.06,
        "Source : captions (textes publiés) · TF-IDF par phase · nombres exclus\n"
        "Lemmatisation : spaCy uk_core_news_trf · Traductions : DeepL\n"
        "Rg P1 = rang du même lemme en Phase 1 (sept. 2022 – déc. 2023) · Mvt = variation de rang P1→P2",
        transform=ax.transAxes,
        fontsize=6.5, color="#777777", va="top", ha="left")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Sauvegardé : {OUT}")
