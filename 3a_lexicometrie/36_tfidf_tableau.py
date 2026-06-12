"""
Tableau TF-IDF P1 — 15 mots les plus distinctifs (caption + dialogue).

Produit deux tableaux annotés (un par source) avec barres TF-IDF
proportionnelles, lemme original + traduction fr, et code couleur par
catégorie sémantique (Finance / Militaire / Associés).

Source : 4_data_et_viz/lexico/tfidf_{caption,dialogue}.csv
Sortie : 4_data_et_viz/tab_tfidf_p1_{caption,dialogue}.png

Lancement direct (`python 36_tfidf_tableau.py`) ou via le dispatcher
`tableaux_tfidf.py --variant tableau_3phases`.
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
    COLORS, HEADER_BG, HEADER_FG, STOPWORDS_DIALOGUE,
    phase_sans_prefixe,
)

# On déplie quelques couleurs en variables nommées pour les patches de légende.
BAR_FIN = COLORS["finance"]["bar"]
BAR_MIL = COLORS["militaire"]["bar"]
BAR_ASS = COLORS["associes"]["bar"]

# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

# Ce tableau-ci utilise des annotations plus bavardes que la base
# (ex. "Magyar (le commandant)" vs "Magyar"). On merge les overrides locaux
# par-dessus la base partagée.
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
    "рер":        "guerre électronique (REB)",
    "вйо":        "allez ! (interjection)",
    "йобликів":   "argot militaire (FPV)",
    "доба":       "24 h / journée",
    "сбс":        "sigle / argot",
    "протягом":   "pendant / au cours de",
    "пм":         "Ptakhyky Madyara (unité)",
    "котрий":     "lequel / qui (rel.)",
    "свій":       "son propre (possessif)",
    "от":         "eh bien / donc (particule)",
    "якийсь":     "un certain / quelconque",
    "тисяча":     "mille / millier",
    "кількість":  "quantité / nombre",
    "великий":    "grand / important",
    "бахмут":     "Bakhmut (lieu de combat)",
    "людина":     "personne / homme",
}

TOP_N = 15

# ---------------------------------------------------------------------------
# Dessin d'un tableau pour une phase
# ---------------------------------------------------------------------------
def tracer_tableau(ax, sub, max_tfidf_global, subtitle=""):
    n_rows    = len(sub)
    row_h     = 1.0    # hauteur d'une ligne de données
    title_h   = 1.1    # hauteur du bandeau titre
    subhdr_h  = 0.9    # hauteur de la ligne sous-en-têtes colonnes
    total_h   = title_h + subhdr_h + n_rows * row_h

    # positions des séparateurs verticaux
    SEP1 = 3.05   # entre Lemme et Traduction
    SEP2 = 6.15   # entre Traduction et Score TF-IDF

    col_x      = [0.3, SEP1 + 0.15, SEP2 + 0.15]
    col_labels = ["Lemme (uk)", "Traduction (fr)", "Score TF-IDF"]

    ax.set_xlim(0, 10)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ---- Bandeau titre (une seule ligne) ----
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, total_h - title_h), 10, title_h,
        boxstyle="square,pad=0", fc=HEADER_BG, ec="none", zorder=1))
    title = "Tableau — Les 15 mots les plus distinctifs sur la période (septembre 2022 – décembre 2023)"
    if subtitle:
        title += f" — {subtitle}"
    ax.text(0.3, total_h - title_h / 2,
            title,
            va="center", ha="left", fontsize=8.5, fontweight="bold",
            color=HEADER_FG, zorder=2)

    # ---- Bandeau sous-en-têtes colonnes ----
    subhdr_top = total_h - title_h
    subhdr_bot = subhdr_top - subhdr_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, subhdr_bot), 10, subhdr_h,
        boxstyle="square,pad=0", fc="#F7F7F7", ec="none", zorder=1))
    ax.axhline(subhdr_top, color="#AAAAAA", linewidth=0.6, zorder=2)
    ax.axhline(subhdr_bot, color="#CCCCCC", linewidth=0.8, zorder=2)

    for x, lbl in zip(col_x, col_labels):
        ax.text(x, subhdr_bot + subhdr_h / 2, lbl,
                va="center", ha="left", fontsize=8,
                fontweight="bold", color="#333333", zorder=3)

    # Séparateurs verticaux dans l'en-tête
    for sep in (SEP1, SEP2):
        ax.plot([sep, sep], [subhdr_bot, subhdr_top],
                color="#CCCCCC", linewidth=0.8, zorder=2)

    # ---- Lignes de données (collées sous les en-têtes) ----
    data_top = subhdr_bot
    bar_x     = SEP2 + 0.15
    bar_w_max = 10 - bar_x - 0.6

    for i, (_, row) in enumerate(sub.iterrows()):
        lemma  = row["lemma"]
        tfidf  = row["tfidf"]
        trad   = TRADUCTIONS.get(lemma, "—")
        cat    = categorize(lemma)
        bg     = COLORS[cat]["bg"]
        bar_c  = COLORS[cat]["bar"]

        y_bot = data_top - (i + 1) * row_h
        y_mid = y_bot + row_h * 0.5

        # Fond de ligne
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y_bot), 10, row_h,
            boxstyle="square,pad=0", fc=bg, ec="none", zorder=1))
        # Séparateur horizontal fin
        ax.plot([0, 10], [y_bot + row_h, y_bot + row_h],
                color="#E0E0E0", linewidth=0.4, zorder=2)

        # Séparateurs verticaux
        for sep in (SEP1, SEP2):
            ax.plot([sep, sep], [y_bot, y_bot + row_h],
                    color="#E0E0E0", linewidth=0.5, zorder=2)

        # Lemme (bold italic)
        ax.text(col_x[0], y_mid, lemma,
                va="center", ha="left", fontsize=8.5,
                fontweight="bold", style="italic",
                color="#1A1A1A", zorder=3)

        # Traduction
        ax.text(col_x[1], y_mid, trad,
                va="center", ha="left", fontsize=8,
                color="#333333", zorder=3)

        # Barre proportionnelle
        bar_w = (tfidf / max_tfidf_global) * bar_w_max
        bar_h = row_h * 0.38
        ax.add_patch(mpatches.FancyBboxPatch(
            (bar_x, y_mid - bar_h / 2), bar_w, bar_h,
            boxstyle="square,pad=0", fc=bar_c, ec="none",
            alpha=0.85, zorder=3))

        # Valeur numérique
        ax.text(bar_x + bar_w + 0.12, y_mid,
                f"{tfidf:.4f}",
                va="center", ha="left", fontsize=7.5,
                color="#555555", zorder=3)

    # Bordure extérieure
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 10, total_h,
        boxstyle="square,pad=0", fc="none",
        ec="#AAAAAA", linewidth=0.8, zorder=4))


# ---------------------------------------------------------------------------
# Génération — P1, caption et dialogue
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "csv":       os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_caption.csv"),
        "out":       os.path.join(BASE, "4_data_et_viz", "tab_tfidf_p1_caption.png"),
        "subtitle":  "légendes",
        "stopwords": set(),
        "source":    "Source : captions (textes publiés) · TF-IDF par phase · mots outils exclus",
    },
    {
        "csv":       os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_dialogue.csv"),
        "out":       os.path.join(BASE, "4_data_et_viz", "tab_tfidf_p1_dialogue.png"),
        "subtitle":  "parlé",
        "stopwords": STOPWORDS_DIALOGUE,
        "source":    "Source : dialogues (parole transcrite Whisper) · TF-IDF par phase · mots outils exclus",
    },
]

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

legend_handles = [
    mpatches.Patch(color=BAR_MIL, label="Militaire"),
    mpatches.Patch(color=BAR_FIN, label="Finance"),
    mpatches.Patch(color=BAR_ASS, label="Associés"),
]

for cfg in CONFIGS:
    df = pd.read_csv(cfg["csv"])
    # Normalise "P1_Artisanal" → "1_Artisanal" pour matcher les libellés ci-dessous
    df["phase"] = phase_sans_prefixe(df["phase"])
    sub = (df[df["phase"] == "1_Artisanal"]
           .sort_values("rank")
           .loc[lambda d: ~d["lemma"].str.match(r'^\d+$')]
           .loc[lambda d: ~d["lemma"].isin(cfg["stopwords"])]
           .head(TOP_N)
           .reset_index(drop=True))
    max_tfidf = sub["tfidf"].max()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.8))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.97, bottom=0.12)

    tracer_tableau(ax, sub, max_tfidf, subtitle=cfg["subtitle"])

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.13),
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    ax.text(0.0, -0.06,
            cfg["source"] + "\nLemmatisation : spaCy uk_core_news_trf · Traductions : DeepL",
            transform=ax.transAxes,
            fontsize=6.5, color="#777777", va="top", ha="left")

    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    fig.savefig(cfg["out"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {cfg['out']}")
