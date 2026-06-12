"""
Slopechart TF-IDF P1 → P2 — captions et dialogue.

Pour chaque source (captions + dialogue), affiche l'évolution du rang
TF-IDF de chaque lemme entre P1 et P2. L'union des top-15 de chaque
phase est tracée ; les lemmes sortis du top sont relégués dans une
zone « Hors top 15 » avec un trait gris pâle.

Source : 4_data_et_viz/lexico/tfidf_{caption,dialogue}.csv
Sorties :
  - 4_data_et_viz/slope_tfidf_p1p2_caption.png
  - 4_data_et_viz/slope_tfidf_p1p2_dialogue.png

Lancement direct ou via `tableaux_tfidf.py --variant slope`.
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
    categorize, TRADUCTIONS as _TRAD_BASE, COLORS, STOPWORDS_DIALOGUE,
    phase_sans_prefixe,
)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "4_data_et_viz")
os.makedirs(OUT, exist_ok=True)

# Couleurs catégories (depuis la palette partagée)
C_MIL = COLORS["militaire"]["bar"]
C_FIN = COLORS["finance"]["bar"]
C_ASS = COLORS["associes"]["bar"]
C_OUT = "#C8CDD2"   # sorti du classement


def couleur_lemma(lemma):
    """Retourne la couleur barre selon la catégorie sémantique."""
    return COLORS[categorize(lemma)]["bar"]

# Traductions : base partagée + overrides spécifiques à ce graphique
TRADUCTIONS = {
    **_TRAD_BASE,
    "жало": "dard/FPV",
    "підрозділ": "unité mil.",
    "хробак": "ver/FPV",
    "реквізит": "coordonnées banc.",
    "морський": "Marines",
    "рер": "REB (guerre élec.)",
    "пм": "PM (unité)",
    "кількість": "quantité",
    "засіб": "moyen/engin",
}


# ---------------------------------------------------------------------------
# Slopechart builder
# ---------------------------------------------------------------------------
def construire_slope(csv_path, out_path, title, subtitle, source_label,
                     top_n=15, stopwords=None):
    df = pd.read_csv(csv_path)
    # Normalise "P1_Artisanal" → "1_Artisanal" pour matcher les libellés ci-dessous
    df["phase"] = phase_sans_prefixe(df["phase"])
    # Exclure nombres purs
    df = df[~df["lemma"].str.match(r'^\d+$')]
    if stopwords:
        df = df[~df["lemma"].isin(stopwords)]

    p1_all = (df[df["phase"] == "1_Artisanal"]
              .sort_values("tfidf", ascending=False)
              .reset_index(drop=True))
    p1_all["rank"] = range(1, len(p1_all) + 1)

    p2_all = (df[df["phase"] == "2_Semi-pro"]
              .sort_values("tfidf", ascending=False)
              .reset_index(drop=True))
    p2_all["rank"] = range(1, len(p2_all) + 1)

    # Union des top_n P1 et top_n P2
    top_p1 = set(p1_all.head(top_n)["lemma"])
    top_p2 = set(p2_all.head(top_n)["lemma"])
    all_lemmas = top_p1 | top_p2

    rank_p1 = dict(zip(p1_all["lemma"], p1_all["rank"]))
    rank_p2 = dict(zip(p2_all["lemma"], p2_all["rank"]))

    # Construire les données du slope
    rows = []
    for lemma in all_lemmas:
        r1 = rank_p1.get(lemma, None)
        r2 = rank_p2.get(lemma, None)
        rows.append({
            "lemma": lemma,
            "rank_p1": r1,
            "rank_p2": r2,
            "in_p1": lemma in top_p1,
            "in_p2": lemma in top_p2,
        })
    sdf = pd.DataFrame(rows)

    # Nombre de positions sur l'axe Y = max des rangs affichés + espace pour "sortis"
    max_rank = top_n

    # Compter combien de mots sortent de chaque côté pour espacer
    exited_left  = [r["lemma"] for _, r in sdf.iterrows() if not r["in_p1"]]
    exited_right = [r["lemma"] for _, r in sdf.iterrows() if not r["in_p2"]]
    n_exited = max(len(exited_left), len(exited_right), 1)

    exit_start = max_rank + 2.5  # début de la zone "sortis"
    exit_spacing = 1.0
    total_h = exit_start + n_exited * exit_spacing + 1.0

    # Attribuer un slot Y unique à chaque mot sorti (chaque côté indépendam.)
    exit_y_left  = {lem: exit_start + i * exit_spacing for i, lem in enumerate(sorted(exited_left))}
    exit_y_right = {lem: exit_start + i * exit_spacing for i, lem in enumerate(sorted(exited_right))}

    # ---------------------------------------------------------------------------
    # Dessin
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, max(11, 7 + total_h * 0.35)))
    fig.patch.set_facecolor("white")

    x_left  = 0.0
    x_right = 1.0
    margin_label = 0.15  # espace pour les labels

    ax.set_xlim(-margin_label - 0.05, x_right + margin_label + 0.05)
    ax.set_ylim(total_h, 0)  # inversé : rang 1 en haut
    ax.axis("off")

    # Colonnes verticales
    ax.axvline(x_left,  color="#CCCCCC", linewidth=0.8, ymin=0.02, ymax=0.96)
    ax.axvline(x_right, color="#CCCCCC", linewidth=0.8, ymin=0.02, ymax=0.96)

    # En-têtes colonnes
    ax.text(x_left,  0.3, "P1\nsept. 2022 – déc. 2023",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")
    ax.text(x_right, 0.3, "P2\njanv. – sept. 2024",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")

    # Ligne séparatrice "sortis"
    sep_y = max_rank + 1.5
    ax.axhline(sep_y, color="#DDDDDD", linewidth=0.6, linestyle="--",
               xmin=0.1, xmax=0.9)
    ax.text(0.5, sep_y + 0.35, "Hors top 15", ha="center", va="top",
            fontsize=7.5, color="#999999", style="italic")

    # Dessiner les pentes
    for _, row in sdf.iterrows():
        lemma = row["lemma"]
        col   = couleur_lemma(lemma)
        trad  = TRADUCTIONS.get(lemma, "")
        label_l = f"{lemma}  ({trad})" if trad else lemma
        label_r = f"({trad})  {lemma}" if trad else lemma

        y1 = row["rank_p1"] if row["in_p1"] else exit_y_left.get(lemma, exit_start)
        y2 = row["rank_p2"] if row["in_p2"] else exit_y_right.get(lemma, exit_start)

        # Ligne de pente
        is_exit = (not row["in_p1"]) or (not row["in_p2"])
        lw  = 2.2 if not is_exit else 1.2
        alpha = 1.0 if not is_exit else 0.5
        line_col = col if not is_exit else C_OUT

        ax.plot([x_left, x_right], [y1, y2],
                color=line_col, linewidth=lw, alpha=alpha, zorder=2,
                solid_capstyle="round")

        # Points et labels
        dot_size = 50 if not is_exit else 25
        if row["in_p1"]:
            ax.scatter(x_left, y1, color=col, s=dot_size, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(x_left - 0.02, y1, label_l,
                    ha="right", va="center", fontsize=7.5, color=col, fontweight="bold")
        else:
            ax.scatter(x_left, y1, color=C_OUT, s=20, zorder=3, marker="x")
            ax.text(x_left - 0.02, y1, label_l,
                    ha="right", va="center", fontsize=7, color="#AAAAAA", style="italic")

        if row["in_p2"]:
            ax.scatter(x_right, y2, color=col, s=dot_size, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(x_right + 0.02, y2, label_r,
                    ha="left", va="center", fontsize=7.5, color=col, fontweight="bold")
        else:
            ax.scatter(x_right, y2, color=C_OUT, s=20, zorder=3, marker="x")
            ax.text(x_right + 0.02, y2, label_r,
                    ha="left", va="center", fontsize=7, color="#AAAAAA", style="italic")

    # Numéros de rang à gauche/droite
    for r in range(1, max_rank + 1):
        ax.text(x_left - margin_label - 0.01, r, str(r),
                ha="right", va="center", fontsize=8, color="#AAAAAA")
        ax.text(x_right + margin_label + 0.01, r, str(r),
                ha="left", va="center", fontsize=8, color="#AAAAAA")

    # Titre
    ax.set_title(title, fontsize=14, fontweight="bold", pad=45)
    ax.text(0.5, 1.03, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#555555")

    # Légende
    legend_handles = [
        Line2D([0], [0], color=C_MIL, linewidth=2.5, label="Militaire"),
        Line2D([0], [0], color=C_FIN, linewidth=2.5, label="Finance"),
        Line2D([0], [0], color=C_ASS, linewidth=2.5, label="Associés"),
        Line2D([0], [0], color=C_OUT, linewidth=1.5, linestyle="-",
               alpha=0.5, label="Sorti du classement"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=4,
              frameon=False, fontsize=8.5)

    # Source
    fig.text(0.08, 0.01, source_label, fontsize=6.5, color="#777777",
             ha="left", va="bottom")

    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.96])
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Sauvegardé : {out_path}")


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------
construire_slope(
    csv_path = os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_caption.csv"),
    out_path = os.path.join(OUT, "slope_tfidf_p1p2_caption.png"),
    title    = "Évolution du lexique des captions — P1 → P2",
    subtitle = "Slopechart TF-IDF, top 15 par phase (nombres exclus)",
    source_label = ("Source : captions · TF-IDF par phase · lemmatisation spaCy uk_core_news_trf\n"
                    "Traductions : DeepL · Catégorisation : Finance / Militaire / Associés"),
    top_n = 15,
)

# ---------------------------------------------------------------------------
# Dialogue (parlé)
# ---------------------------------------------------------------------------
construire_slope(
    csv_path = os.path.join(BASE, "4_data_et_viz", "lexico", "tfidf_dialogue.csv"),
    out_path = os.path.join(OUT, "slope_tfidf_p1p2_dialogue.png"),
    title    = "Évolution du lexique parlé — P1 → P2",
    subtitle = "Slopechart TF-IDF, top 15 par phase (mots outils exclus)",
    source_label = ("Source : transcriptions Whisper · TF-IDF par phase · lemmatisation spaCy uk_core_news_trf\n"
                    "Traductions : DeepL · Mots outils (pronoms, particules) exclus"),
    top_n = 15,
    stopwords = STOPWORDS_DIALOGUE,
)
