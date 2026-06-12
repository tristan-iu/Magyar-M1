"""
TF-IDF P1 — graphique en colonnes (version alternative au tableau).

Top 15 lemmes les plus distinctifs de P1 (caption + dialogue), barres
verticales colorées par catégorie sémantique. Variante visuelle du
tableau `36_tfidf_tableau.py`.

Entrée  : 4_data_et_viz/lexico/tfidf_{caption,dialogue}.csv
Sortie  : 4_data_et_viz/barchart_tfidf_p1_{caption,dialogue}.png

Lancement direct ou via `tableaux_tfidf.py --variant barchart_p1`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categories_semantiques import (  # noqa: E402
    categorize, TRADUCTIONS, COLORS, STOPWORDS_DIALOGUE,
    phase_sans_prefixe,
)

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ  = os.path.join(BASE, "4_data_et_viz")

# On force la couleur "associes" sur la teinte texte (gris fond) plutôt que
# "bar" (gris clair) pour donner davantage de contraste sur les barres.
C_MIL = COLORS["militaire"]["bar"]
C_FIN = COLORS["finance"]["bar"]
C_ASS = COLORS["associes"]["text"]

_CAT_COLOR = {"militaire": C_MIL, "finance": C_FIN, "associes": C_ASS}


def couleur_lemma(lemma):
    """Retourne la couleur de barre selon la catégorie sémantique."""
    return _CAT_COLOR[categorize(lemma)]

TOP_N = 15

# ---------------------------------------------------------------------------
# Configs caption / dialogue
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "csv":       os.path.join(VIZ, "lexico", "tfidf_caption.csv"),
        "out":       os.path.join(VIZ, "barchart_tfidf_p1_caption.png"),
        "subtitle":  "légendes publiées",
        "stopwords": set(),
        "source":    (
            "Source : captions (textes publiés) · TF-IDF par phase · P1 : sept. 2022 – déc. 2023\n"
            "Lemmatisation : spaCy uk_core_news_trf · Traductions : DeepL"
        ),
    },
    {
        "csv":       os.path.join(VIZ, "lexico", "tfidf_dialogue.csv"),
        "out":       os.path.join(VIZ, "barchart_tfidf_p1_dialogue.png"),
        "subtitle":  "parole transcrite",
        "stopwords": STOPWORDS_DIALOGUE,
        "source":    (
            "Source : dialogues (transcription Whisper) · TF-IDF par phase · P1 : sept. 2022 – déc. 2023\n"
            "Lemmatisation : spaCy uk_core_news_trf · Traductions : DeepL"
        ),
    },
]

# ---------------------------------------------------------------------------
# Génération
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(color=C_MIL, label="Militaire"),
    mpatches.Patch(color=C_FIN, label="Finance"),
    mpatches.Patch(color=C_ASS, label="Associés"),
]

for cfg in CONFIGS:
    df  = pd.read_csv(cfg["csv"])
    # Normalise "P1_Artisanal" → "1_Artisanal" pour matcher les libellés ci-dessous
    df["phase"] = phase_sans_prefixe(df["phase"])
    sub = (
        df[df["phase"] == "1_Artisanal"]
        .sort_values("rank")
        .loc[lambda d: ~d["lemma"].str.match(r'^\d+$')]
        .loc[lambda d: ~d["lemma"].isin(cfg["stopwords"])]
        .head(TOP_N)
        .reset_index(drop=True)
    )

    lemmas  = sub["lemma"].tolist()
    scores  = sub["tfidf"].tolist()
    colors  = [couleur_lemma(l) for l in lemmas]
    # Étiquette X : traduction en français, lemme ukrainien en dessous
    xlabels = [
        f"{TRADUCTIONS.get(l, l)}\n{l}"
        for l in lemmas
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(lemmas))
    bars = ax.bar(x, scores, color=colors, width=0.6, zorder=3, alpha=0.9)

    # Valeur au-dessus de chaque barre
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(scores) * 0.012,
            f"{score:.4f}",
            ha="center", va="bottom", fontsize=6.5, color="#555555"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.5, ha="center", va="top")
    ax.tick_params(axis="x", length=0, pad=6)

    ax.set_ylabel("Score TF-IDF", fontsize=9, labelpad=8)
    ax.set_ylim(0, max(scores) * 1.18)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(axis="y", labelsize=8, color="#CCCCCC")

    ax.set_title(
        f"Top {TOP_N} lemmes les plus distinctifs de P1 — {cfg['subtitle']}",
        fontsize=11, fontweight="bold", pad=14, loc="left"
    )
    ax.text(
        1.0, 1.02,
        "P1 : sept. 2022 – déc. 2023",
        transform=ax.transAxes,
        fontsize=8, color="#666666", ha="right", va="bottom"
    )

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=8.5,
    )

    fig.text(
        0.0, -0.04,
        cfg["source"],
        fontsize=6.5, color="#888888", va="top", ha="left",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
    fig.savefig(cfg["out"], dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {cfg['out']}")
