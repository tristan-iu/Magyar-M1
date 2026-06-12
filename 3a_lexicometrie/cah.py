#!/usr/bin/env python3
"""
cah.py — Classification Ascendante Hiérarchique Ward.

Pipeline :
  1. DTM message × lemmes, pondéré TF-IDF (L2)
  2. CAH méthode Ward sur distance euclidienne (chaque message = 1 classe
     initiale, fusion qui minimise l'accroissement d'inertie intra à chaque
     étape)
  3. Dendrogramme — couper à k=3
  4. Test chi2 : les 3 clusters sont-ils indépendants des 3 phases ?
  5. Messages migrants : classés dans un cluster ≠ de leur phase

Choix méthodologique assumé : on classifie directement la DTM TF-IDF
(distance euclidienne), approche IR/scikit-learn — et NON les coordonnées
factorielles d'une AFC préalable (distance du chi2), qui serait la voie
Benzécri-Reinert canonique en ADT française. Conséquence : la CAH ici n'est
pas la « CDH de Reinert » (Iramuteq/Rainette) — à formuler comme tel dans
le mémoire.

Usage :
    python cah.py --output-dir 4_data_et_viz/lexico
    python cah.py --output-dir ... --k 3 --top-n 150 --sources caption dialogue
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
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfTransformer


# ---------------------------------------------------------------------------
# Construction de la DTM
# ---------------------------------------------------------------------------

def construire_dtm(csv_path, top_n=150, min_count=5, min_tokens_per_doc=5):
    """
    Construit la DTM (Document-Term Matrix) message × lemmes, pondérée TF-IDF.

    Retourne :
      tfidf_matrix : np.array (n_docs × n_terms)
      terms        : list[str] noms des colonnes
      doc_meta     : DataFrame (message_id, date, phase)
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
    df = df[df["phase"].notna() & (df["phase"] != "")]

    # Top N lemmes par fréquence globale, filtrés min_count
    freq = df["lemma"].value_counts()
    top_lemmes = [l for l, c in freq.items() if c >= min_count][:top_n]
    top_set = set(top_lemmes)

    # DTM brute (count)
    grouped = df[df["lemma"].isin(top_set)].groupby("message_id", sort=False)
    rows = []
    meta_rows = []

    for mid, group in grouped:
        counts = group["lemma"].value_counts()
        if counts.sum() < min_tokens_per_doc:
            continue
        rows.append(counts)
        meta_rows.append({
            "message_id": mid,
            "date": group["date"].iloc[0],
            "phase": group["phase"].iloc[0],
        })

    if not rows:
        return None, None, None

    dtm = pd.DataFrame(rows, columns=top_lemmes).fillna(0).astype(int)
    doc_meta = pd.DataFrame(meta_rows).reset_index(drop=True)

    # TF-IDF
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(dtm.values).toarray()

    return tfidf_matrix, list(dtm.columns), doc_meta


# ---------------------------------------------------------------------------
# CAH Ward
# ---------------------------------------------------------------------------

def executer_cah(tfidf_matrix, k=3):
    """
    CAH méthode Ward sur la matrice TF-IDF.

    Retourne :
      Z       : linkage matrix
      labels  : array cluster assignments (0-indexed)
    """
    Z = linkage(tfidf_matrix, method="ward", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return Z, labels


# ---------------------------------------------------------------------------
# Test chi2
# ---------------------------------------------------------------------------

def tester_chi2(doc_meta, labels):
    """
    Test chi2 d'indépendance : clusters × phases.

    Retourne :
      ct     : table de contingence
      chi2   : statistique chi2
      p      : p-value
      dof    : degrés de liberté
      cramersv : V de Cramér (taille d'effet)
    """
    doc_meta = doc_meta.copy()
    doc_meta["cluster"] = labels

    ct = pd.crosstab(doc_meta["phase"], doc_meta["cluster"])
    chi2, p, dof, expected = chi2_contingency(ct)

    # V de Cramér
    n = ct.values.sum()
    min_dim = min(ct.shape[0], ct.shape[1]) - 1
    cramersv = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return ct, chi2, p, dof, cramersv


# ---------------------------------------------------------------------------
# Messages migrants
# ---------------------------------------------------------------------------

def trouver_migrants(doc_meta, labels, phase_cluster_map):
    """
    Identifie les messages dont le cluster ≠ du cluster dominant de leur phase.

    phase_cluster_map : dict {phase: cluster_dominant}
    """
    doc_meta = doc_meta.copy()
    doc_meta["cluster"] = labels
    doc_meta["expected_cluster"] = doc_meta["phase"].map(phase_cluster_map)
    migrants = doc_meta[doc_meta["cluster"] != doc_meta["expected_cluster"]]
    return migrants


def calculer_mapping_phase_cluster(ct):
    """Pour chaque phase, trouve le cluster dominant (celui avec le plus de messages)."""
    return {phase: ct.loc[phase].idxmax() for phase in ct.index}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def tracer_dendrogramme(Z, doc_meta, src, fig_dir, k=3):
    """Dendrogramme avec ligne de coupe à k clusters."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Couleur de coupe
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=50,
               leaf_rotation=90, leaf_font_size=8,
               color_threshold=Z[-(k-1), 2])

    # Ligne de coupe
    cut_height = Z[-(k-1), 2]
    ax.axhline(cut_height, color="red", linestyle="--", linewidth=1.5,
               label=f"Coupe k={k} (h={cut_height:.1f})")
    ax.legend(fontsize=10)

    ax.set_title(f"Dendrogramme CAH Ward — {src}\n"
                 f"{len(doc_meta)} documents, coupe à k={k}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster (tronqué)", fontsize=11)
    ax.set_ylabel("Distance Ward", fontsize=11)

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_cah_dendro_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def tracer_heatmap_contingence(ct, chi2_val, p_val, cramersv, src, fig_dir):
    """Heatmap de la table de contingence clusters × phases."""
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(ct.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Nb messages")

    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels([f"Cluster {c}" for c in ct.columns], fontsize=10)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index, fontsize=10)

    # Valeurs dans les cellules
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = ct.values[i, j]
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val > ct.values.max() * 0.6 else "black")

    sig = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
    ax.set_title(f"CAH Ward × Phases — {src}\n"
                 f"chi2 = {chi2_val:.1f}, {sig}, V de Cramér = {cramersv:.3f}",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(fig_dir, f"fig_cah_chi2_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ---------------------------------------------------------------------------
# Traitement d'une source
# ---------------------------------------------------------------------------

def traiter_source(csv_path, src, output_dir, fig_dir, k, top_n, min_count):
    """Pipeline CAH complet pour une source."""
    # 1. DTM TF-IDF
    print("  Construction DTM TF-IDF...")
    tfidf_matrix, terms, doc_meta = construire_dtm(
        csv_path, top_n=top_n, min_count=min_count)

    if tfidf_matrix is None:
        print("    [SKIP] pas assez de documents")
        return

    print(f"    {tfidf_matrix.shape[0]} documents × {tfidf_matrix.shape[1]} termes")

    # 2. CAH Ward
    print(f"  CAH Ward (k={k})...")
    Z, labels = executer_cah(tfidf_matrix, k=k)

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"    Cluster {c}: {n} messages ({n/len(labels)*100:.1f}%)")

    # 3. Dendrogramme
    print("  Dendrogramme...")
    tracer_dendrogramme(Z, doc_meta, src, fig_dir, k=k)

    # 4. Test chi2
    print("  Test chi2 (clusters × phases)...")
    ct, chi2_val, p_val, dof, cramersv = tester_chi2(doc_meta, labels)

    print(f"    Table de contingence :")
    print(f"    {ct.to_string()}")
    print(f"\n    chi2 = {chi2_val:.2f}, dof = {dof}, "
          f"p = {p_val:.2e}, V de Cramér = {cramersv:.3f}")

    if p_val < 0.05:
        print(f"    -> SIGNIFICATIF : les clusters ne sont PAS indépendants des phases")
    else:
        print(f"    -> NON significatif : clusters indépendants des phases")

    tracer_heatmap_contingence(ct, chi2_val, p_val, cramersv, src, fig_dir)

    # 5. Messages migrants
    phase_cluster_map = calculer_mapping_phase_cluster(ct)
    print(f"\n    Mapping phase → cluster dominant : {phase_cluster_map}")

    migrants = trouver_migrants(doc_meta, labels, phase_cluster_map)
    n_migrants = len(migrants)
    n_total = len(doc_meta)
    print(f"    Messages migrants : {n_migrants}/{n_total} "
          f"({n_migrants/n_total*100:.1f}%)")

    # Export migrants
    migrants_path = os.path.join(output_dir, f"cah_migrants_{src}.csv")
    migrants_out = migrants[["message_id", "date", "phase", "cluster",
                             "expected_cluster"]].copy()
    migrants_out.to_csv(migrants_path, index=False)
    print(f"    -> {migrants_path}")

    # Export assignments complet
    assign_path = os.path.join(output_dir, f"cah_assignments_{src}.csv")
    assign_df = doc_meta.copy()
    assign_df["cluster"] = labels
    assign_df.to_csv(assign_path, index=False)
    print(f"    -> {assign_path}")

    # Migrants par phase
    if n_migrants > 0:
        print(f"\n    Migrants par phase :")
        for phase in sorted(migrants["phase"].unique()):
            sub = migrants[migrants["phase"] == phase]
            print(f"      {phase}: {len(sub)} migrants")
            # Vers quels clusters migrent-ils ?
            for cl in sorted(sub["cluster"].unique()):
                n = len(sub[sub["cluster"] == cl])
                print(f"        → cluster {cl}: {n}")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CAH Ward + chi2 — corpus Magyar (LEXICO §7)")
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--sources", nargs="*", default=["caption", "dialogue"],
                        help="Sources à traiter (défaut : caption dialogue)")
    parser.add_argument("--k", type=int, default=3,
                        help="Nombre de clusters (défaut : 3)")
    parser.add_argument("--top-n", type=int, default=150,
                        help="Top N lemmes pour la DTM (défaut : 150)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Fréquence min d'un lemme (défaut : 5)")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml")
    args = parser.parse_args()

    load_config(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"CAH Ward — Classification Ascendante Hiérarchique")
    print(f"  k={args.k}, top_n={args.top_n}, min_count={args.min_count}\n")

    for src in args.sources:
        csv_path = os.path.join(args.output_dir, f"lemmes_{src}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {src} : {csv_path} introuvable")
            continue

        print(f"── {src} ──")
        traiter_source(csv_path, src, args.output_dir, fig_dir,
                       args.k, args.top_n, args.min_count)
        print()

    print("Terminé.")


if __name__ == "__main__":
    main()
