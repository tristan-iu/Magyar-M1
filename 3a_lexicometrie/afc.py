#!/usr/bin/env python3
"""
afc.py — Analyse Factorielle des Correspondances.

Pipeline :
  1. Table de contingence : phases (P1/P2/P3) × lemmes (top N)
  2. AFC via prince (SVD sur les profils chi2)
  3. Biplot Dim1×Dim2 (lignes = phases, colonnes = lemmes)
  4. Dépouillement : contributions + cos2 de chaque modalité par facteur
  5. Alerte hapax : lemmes à forte contribution n'apparaissant que dans 2-3 messages

Lecture : le biplot seul ne suffit pas — toujours croiser avec contributions,
cos2 et retour au contexte (KWIC) avant d'interpréter une opposition factorielle.

Usage :
    python afc.py --output-dir 4_data_et_viz/lexico
    python afc.py --output-dir ... --top-n 100 --sources caption dialogue
"""

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config  # noqa: E402

# CSV de lemmes lus dans 4_data_et_viz/lexico/ ; figures écrites un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import prince


# ---------------------------------------------------------------------------
# Construction de la table de contingence
# ---------------------------------------------------------------------------

def construire_contingence(csv_path, top_n=150, min_count=5, min_messages=3):
    """
    Construit la table de contingence phases × lemmes.

    Retourne :
      ct         : DataFrame (index=phases, columns=lemmes, values=count)
      msg_counts : dict {lemma: nb_messages_distincts} pour détecter les hapax
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
    df = df[df["phase"].notna() & (df["phase"] != "")]

    # Fréquence globale
    freq = df["lemma"].value_counts()
    # Nombre de messages distincts par lemme
    msg_counts = df.groupby("lemma")["message_id"].nunique().to_dict()

    # Filtrer : fréquence >= min_count ET présent dans >= min_messages messages
    valid = [l for l, c in freq.items()
             if c >= min_count and msg_counts.get(l, 0) >= min_messages]
    top_lemmes = valid[:top_n]

    df_filtered = df[df["lemma"].isin(top_lemmes)]

    # Table de contingence
    ct = pd.crosstab(df_filtered["phase"], df_filtered["lemma"])

    # Trier les colonnes par fréquence totale décroissante
    col_order = ct.sum().sort_values(ascending=False).index
    ct = ct[col_order]

    return ct, msg_counts


# ---------------------------------------------------------------------------
# AFC
# ---------------------------------------------------------------------------

def executer_afc(ct):
    """
    Exécute l'AFC via prince.CA.

    Retourne :
      ca            : objet prince.CA fitté
      row_coords    : DataFrame coordonnées des lignes (phases)
      col_coords    : DataFrame coordonnées des colonnes (lemmes)
      eigenvalues   : array valeurs propres
      explained     : array % d'inertie expliquée par facteur
    """
    ca = prince.CA(n_components=min(len(ct) - 1, len(ct.columns) - 1, 5))
    ca = ca.fit(ct)

    row_coords = ca.row_coordinates(ct)
    col_coords = ca.column_coordinates(ct)
    eigenvalues = ca.eigenvalues_
    total_inertia = ca.total_inertia_
    explained = (eigenvalues / total_inertia * 100) if total_inertia > 0 else eigenvalues * 0

    return ca, row_coords, col_coords, eigenvalues, explained


# ---------------------------------------------------------------------------
# Contributions et cos2
# ---------------------------------------------------------------------------

def calculer_contributions(ca, ct):
    """
    Calcule les contributions et cos2 des lignes et colonnes.

    Contribution = part de l'inertie d'un axe expliquée par une modalité.
    Cos2 = qualité de représentation d'une modalité sur un axe.
    """
    row_contrib = ca.row_contributions_
    col_contrib = ca.column_contributions_

    row_cos = ca.row_cosine_similarities(ct)
    col_cos = ca.column_cosine_similarities(ct)

    return row_contrib, col_contrib, row_cos, col_cos


# ---------------------------------------------------------------------------
# Biplot
# ---------------------------------------------------------------------------

def tracer_biplot(row_coords, col_coords, explained, src, fig_dir,
                top_contrib_n=25, col_contrib=None):
    """
    Biplot Dim1 × Dim2 : phases (triangles) + top lemmes (points).
    Seuls les top_contrib_n lemmes avec la plus forte contribution
    sur Dim1 ou Dim2 sont annotés.
    """
    cfg = load_config()
    phase_colors = {
        f"{pid}_{cfg['phases'][pid].get('label', '')}": cfg['phases'][pid].get('color', '#999')
        for pid in sorted(cfg.get('phases', {}).keys())
    }

    fig, ax = plt.subplots(figsize=(14, 10))

    # Sélectionner les lemmes à annoter (top contributions Dim1 + Dim2)
    if col_contrib is not None and len(col_contrib.columns) >= 2:
        max_contrib = col_contrib.iloc[:, :2].max(axis=1)
        top_labels = set(max_contrib.nlargest(top_contrib_n).index)
    else:
        top_labels = set(col_coords.index[:top_contrib_n])

    # Colonnes (lemmes) — points gris, annotés si top contribution
    ax.scatter(col_coords.iloc[:, 0], col_coords.iloc[:, 1],
               c="#CCCCCC", s=15, alpha=0.4, zorder=1)

    for idx, row in col_coords.iterrows():
        if idx in top_labels:
            ax.annotate(idx, (row.iloc[0], row.iloc[1]),
                        fontsize=8, alpha=0.85, color="#2C3E50",
                        textcoords="offset points", xytext=(4, 4))
            ax.scatter(row.iloc[0], row.iloc[1],
                       c="#E74C3C", s=40, alpha=0.8, zorder=3)

    # Lignes (phases) — triangles colorés
    for idx, row in row_coords.iterrows():
        color = phase_colors.get(idx, "#333333")
        ax.scatter(row.iloc[0], row.iloc[1],
                   c=color, s=250, marker="^", zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(idx, (row.iloc[0], row.iloc[1]),
                    fontsize=11, fontweight="bold", color=color,
                    textcoords="offset points", xytext=(10, 10),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Axes
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"Dim 1 ({explained[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"Dim 2 ({explained[1]:.1f}%)", fontsize=12)
    ax.set_title(f"AFC — {src}\n"
                 f"Dim1 ({explained[0]:.1f}%) + Dim2 ({explained[1]:.1f}%) "
                 f"= {explained[0]+explained[1]:.1f}% d'inertie",
                 fontsize=14, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_afc_biplot_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def exporter_contributions(row_contrib, col_contrib, row_cos, col_cos,
                         src, output_dir):
    """Exporte les contributions et cos2 en CSV."""
    # Contributions colonnes (lemmes)
    col_df = col_contrib.copy()
    col_df.columns = [f"contrib_dim{i}" for i in range(len(col_df.columns))]
    for i in range(min(2, len(col_cos.columns))):
        col_df[f"cos2_dim{i}"] = col_cos.iloc[:, i]
    col_df = col_df.sort_values("contrib_dim0", ascending=False)

    path = os.path.join(output_dir, f"afc_contributions_{src}.csv")
    col_df.to_csv(path)
    print(f"    -> {path}")

    # Contributions lignes (phases)
    row_df = row_contrib.copy()
    row_df.columns = [f"contrib_dim{i}" for i in range(len(row_df.columns))]
    for i in range(min(2, len(row_cos.columns))):
        row_df[f"cos2_dim{i}"] = row_cos.iloc[:, i]

    path = os.path.join(output_dir, f"afc_phases_{src}.csv")
    row_df.to_csv(path)
    print(f"    -> {path}")


def verifier_artefacts_hapax(col_contrib, msg_counts, threshold_contrib=5.0,
                          threshold_msgs=3):
    """
    Alerte B4.5 : lemmes à forte contribution qui n'apparaissent
    que dans très peu de messages.
    """
    alerts = []
    for lemma in col_contrib.index:
        max_contrib = col_contrib.loc[lemma].max() * 100
        n_msgs = msg_counts.get(lemma, 0)
        if max_contrib >= threshold_contrib and n_msgs <= threshold_msgs:
            alerts.append((lemma, max_contrib, n_msgs))

    return sorted(alerts, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Traitement d'une source
# ---------------------------------------------------------------------------

def traiter_source(csv_path, src, output_dir, fig_dir, top_n, min_count):
    """Pipeline AFC complet pour une source."""
    # 1. Table de contingence
    print("  Table de contingence...")
    ct, msg_counts = construire_contingence(csv_path, top_n=top_n, min_count=min_count)
    print(f"    {ct.shape[0]} phases × {ct.shape[1]} lemmes, "
          f"{ct.values.sum()} occurrences")

    if ct.shape[0] < 2 or ct.shape[1] < 3:
        print(f"    [SKIP] table trop petite")
        return

    # Export table de contingence
    ct_path = os.path.join(output_dir, f"afc_contingence_{src}.csv")
    ct.to_csv(ct_path)
    print(f"    -> {ct_path}")

    # 2. AFC
    print("  AFC...")
    ca, row_coords, col_coords, eigenvalues, explained = executer_afc(ct)

    print(f"    Valeurs propres : {', '.join(f'{e:.4f}' for e in eigenvalues)}")
    print(f"    Inertie expliquée : {', '.join(f'{e:.1f}%' for e in explained)}")

    # 3. Contributions
    row_contrib, col_contrib, row_cos, col_cos = calculer_contributions(ca, ct)

    # 4. Biplot
    print("  Biplot...")
    tracer_biplot(row_coords, col_coords, explained, src, fig_dir,
                col_contrib=col_contrib)

    # 5. Exports
    print("  Exports...")
    exporter_contributions(row_contrib, col_contrib, row_cos, col_cos,
                         src, output_dir)

    # Coordonnées
    coords_path = os.path.join(output_dir, f"afc_coords_{src}.csv")
    all_coords = pd.concat([
        row_coords.assign(type="phase"),
        col_coords.assign(type="lemma"),
    ])
    all_coords.to_csv(coords_path)
    print(f"    -> {coords_path}")

    # 6. Alerte hapax
    alerts = verifier_artefacts_hapax(col_contrib, msg_counts)
    if alerts:
        print(f"\n    ALERTE B4.5 — lemmes à forte contribution mais peu de messages :")
        for lemma, contrib, n_msgs in alerts[:10]:
            print(f"      {lemma}: contrib max = {contrib:.1f}%, "
                  f"dans seulement {n_msgs} messages")
    else:
        print(f"    Pas d'artefact hapax détecté.")

    # Top contributions Dim1 et Dim2
    print(f"\n    Top 10 contributions Dim1 :")
    top1 = col_contrib.iloc[:, 0].nlargest(10)
    for lemma, val in top1.items():
        print(f"      {lemma}: {val*100:.1f}%")

    if len(col_contrib.columns) >= 2:
        print(f"    Top 10 contributions Dim2 :")
        top2 = col_contrib.iloc[:, 1].nlargest(10)
        for lemma, val in top2.items():
            print(f"      {lemma}: {val*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AFC — Analyse Factorielle des Correspondances (LEXICO §6)")
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--sources", nargs="*", default=["caption", "dialogue"],
                        help="Sources à traiter (défaut : caption dialogue)")
    parser.add_argument("--top-n", type=int, default=150,
                        help="Top N lemmes pour la table de contingence (défaut : 150)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Fréquence min d'un lemme (défaut : 5)")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml")
    args = parser.parse_args()

    load_config(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"AFC — Analyse Factorielle des Correspondances")
    print(f"  Top N : {args.top_n}, min_count : {args.min_count}\n")

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
