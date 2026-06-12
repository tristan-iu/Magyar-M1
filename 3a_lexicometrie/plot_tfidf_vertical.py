"""
Tableau TF-IDF — version verticale empilée sur 3 phases (caption + dialogue
combinés), barres verticales colorées par catégorie sémantique.

Variante visuelle ancienne, conservée pour comparaison avec les tableaux
36*.py (qui partagent les dictionnaires `categories_semantiques`). Ce
script utilise un dictionnaire local légèrement étendu (мilitaire ajoute
`атака`, `операція`, `удар`… qui ne sont pas dans le dictionnaire partagé).

Source : 4_data_et_viz/lexico/tfidf_combined.csv
Sortie : 4_data_et_viz/fig_tfidf_vertical.png
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ---------------------------------------------------------------------------
# Catégorisation sémantique (locale, étendue par rapport à
# `categories_semantiques.MILITAIRE`)
# ---------------------------------------------------------------------------
MILITAIRE = {
    "дрон", "хробак", "підрозділ", "пілот", "птах", "засіб",
    "фпв", "йобликів", "жало", "ціль", "бригада", "нічний",
    "ворог", "бахмут", "окремий", "хробачий", "жах", "бойовий",
    "атака", "операція", "удар", "розвідка", "позиція", "техніка",
}

FINANCE = {
    "збір", "тисяча", "дякувати", "млн", "грн", "реквізити",
    "задонатити", "донат", "переказ", "банк", "гривня", "рахунок",
}

# Tout le reste = associés (gris) — y compris мадяр.

# ---------------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------------
C_MILITAIRE = "#2C5F8A"   # bleu acier
C_FINANCE   = "#C07A20"   # ambre/or
C_ASSOCIES  = "#B8B8B8"   # gris neutre

TOP_N = 15
# CSV lu dans 4_data_et_viz/lexico/ ; figure PNG écrite un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(_REPO, "4_data_et_viz", "fig_tfidf_vertical.png")

# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
csv_path = os.path.join(_REPO, "4_data_et_viz", "lexico", "tfidf_combined.csv")
df = pd.read_csv(csv_path)
# Normalise "P1_Artisanal" → "1_Artisanal" (schéma canonique post-migration)
df["phase"] = df["phase"].str.replace(r'^P(\d)', r'\1', regex=True)

# Noms affichés des phases
PHASE_LABELS = {
    "1_Artisanal":      "P1 — Artisanal (2022–2023)",
    "2_Semi-pro":       "P2 — Semi-pro (2024)",
    "3_Institutionnel": "P3 — Institutionnel (2024–2025)",
}

phases = ["1_Artisanal", "2_Semi-pro", "3_Institutionnel"]

# ---------------------------------------------------------------------------
# Figure : 3 panneaux empilés verticalement, barres verticales
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

fig, axes = plt.subplots(3, 1, figsize=(12, 13))
fig.subplots_adjust(hspace=0.55)

for ax, phase in zip(axes, phases):
    sub = (df[df["phase"] == phase]
           .sort_values("rank")
           .head(TOP_N)
           .reset_index(drop=True))

    # Couleur par catégorie
    colors = []
    for lemma in sub["lemma"]:
        if lemma in MILITAIRE:
            colors.append(C_MILITAIRE)
        elif lemma in FINANCE:
            colors.append(C_FINANCE)
        else:
            colors.append(C_ASSOCIES)

    bars = ax.bar(sub["lemma"], sub["tfidf"], color=colors, width=0.7, edgecolor="none")

    ax.set_title(PHASE_LABELS[phase], fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.set_ylabel("TF-IDF", fontsize=9)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, sub["tfidf"].max() * 1.18)

    # Valeur au dessus de chaque barre
    for bar, val in zip(bars, sub["tfidf"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7, color="#555555",
        )

# Légende (finance + militaire seulement)
legend_handles = [
    mpatches.Patch(color=C_MILITAIRE, label="Militaire"),
    mpatches.Patch(color=C_FINANCE,   label="Finance"),
]
fig.legend(
    handles=legend_handles,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.99),
    frameon=False,
    fontsize=10,
)

fig.suptitle(
    "Top 15 lemmes TF-IDF distinctifs par phase",
    fontsize=13, fontweight="bold", y=1.01,
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Sauvegardé : {OUTPUT}")
