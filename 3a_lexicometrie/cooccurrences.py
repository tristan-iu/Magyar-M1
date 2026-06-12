#!/usr/bin/env python3
"""
cooccurrences.py — Co-occurrence PMI + réseau lexical.

Produit pour chaque source (caption, dialogue) un graphe global :
  - edges CSV : Source,Target,Weight(PMI),RawCount  (Gephi-ready)
  - nodes CSV : Id,Label,Freq,Degree,WeightedDegree,Betweenness,Eigenvector,Closeness

Mode --per-phase : produit en plus un jeu de fichiers par phase (P1, P2, P3)
et un tableau synthèse `cooc_centralite_evolution_{src}.csv` croisant
lemme × phase × indicateur (betweenness, eigenvector, degree). C'est ce
tableau qui documente l'évolution de la centralité lexicale (lemmes
pivots du mémoire — мадяр, рахунок, зведений).

Pondération par PMI (Pointwise Mutual Information) :
  PMI(a,b) = log2( P(a,b) / (P(a) * P(b)) )
  Filtre les co-occurrences mécaniques entre mots fréquents.
  Seules les arêtes avec PMI > seuil ET count >= min-count sont gardées.

Fenêtre glissante : 5 tokens (caption), 10 tokens (dialogue).
Top 200 lemmes par fréquence (par défaut) — recalculé indépendamment par
phase en mode --per-phase pour capter le vocabulaire spécifique à chacune.

Indicateurs réseau : betweenness identifie les lemmes qui interagissent avec
une plus grande diversité de voisins (rôle structurant) ; eigenvector pondère
par la centralité des voisins ; densité cible ~0.001 pour un graphe lisible.

Usage :
    python cooccurrences.py --output-dir 4_data_et_viz/lexico
    python cooccurrences.py --output-dir ... --top-n 150 --pmi-threshold 3.0
    python cooccurrences.py --output-dir ... --per-phase
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config  # noqa: E402

# CSV (matrices, arêtes, nœuds, centralités) écrits dans 4_data_et_viz/lexico/.
# Ce script ne produit pas de figure (export Gephi/réseau en aval).
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW_SIZES = {
    "caption":  5,
    "dialogue": 10,
}
DEFAULT_WINDOW = 5

# Seuils PMI spécifiques par source.
# Caption = texte court, vocabulaire ciblé, le bruit statistique est déjà
# filtré par top-N + min-count → seuil PMI élevé (2.0) pour isoler les
# vraies associations thématiques ("414e", "камікадзе", "moнo").
# Dialogue = transcription Whisper, corpus 5× plus volumineux, lexique
# étendu (langage parlé, répétitions, hésitations lemmatisées) → l'entropie
# des distributions marginales est plus haute, ce qui tire mécaniquement
# les PMI vers le bas. Seuil 2.0 ne laisse que 2 nœuds dans le graphe global.
# Test systématique (PMI ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}) : 1.0 donne
# 30 nœuds global / 28-53 par phase, densité comparable aux graphes caption.
PMI_THRESHOLDS = {
    "caption":  2.0,
    "dialogue": 1.0,
}
DEFAULT_PMI_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Co-occurrence + PMI
# ---------------------------------------------------------------------------

def construire_cooccurrence(lemmes_by_message, top_lemmes_set, window):
    """
    Fenêtre glissante sur les lemmes de chaque message.
    Co-occurrence binaire par message (1 par paire par message max).

    Retourne :
      cooc       : Counter {(a, b): count}  avec a < b
      doc_freq   : Counter {lemma: nb_messages_où_il_apparaît}
      n_messages : int
    """
    cooc = Counter()
    doc_freq = Counter()
    n_messages = len(lemmes_by_message)

    for lemmes in lemmes_by_message:
        # Lemmes présents dans ce message (pour doc_freq)
        present = set()
        # Paires vues dans ce message (binaire)
        seen_pairs = set()

        for i, lemma in enumerate(lemmes):
            if lemma not in top_lemmes_set:
                continue
            present.add(lemma)
            for j in range(i + 1, min(i + 1 + window, len(lemmes))):
                other = lemmes[j]
                if other not in top_lemmes_set or other == lemma:
                    continue
                pair = tuple(sorted((lemma, other)))
                seen_pairs.add(pair)

        for pair in seen_pairs:
            cooc[pair] += 1
        for lemma in present:
            doc_freq[lemma] += 1

    return cooc, doc_freq, n_messages


def calculer_pmi(cooc, doc_freq, n_messages, min_count=3):
    """
    Calcule le PMI pour chaque paire de co-occurrence.

    PMI(a,b) = log2( P(a,b) / (P(a) * P(b)) )
    où P(x) = doc_freq(x) / n_messages
    et P(a,b) = cooc(a,b) / n_messages

    Retourne : dict {(a,b): {"pmi": float, "count": int}}
    """
    results = {}
    for (a, b), count in cooc.items():
        if count < min_count:
            continue
        p_ab = count / n_messages
        p_a = doc_freq[a] / n_messages
        p_b = doc_freq[b] / n_messages
        if p_a == 0 or p_b == 0:
            continue
        pmi = math.log2(p_ab / (p_a * p_b))
        results[(a, b)] = {"pmi": round(pmi, 4), "count": count}
    return results


def regrouper_lemmes_par_message(df):
    """Groupe les lemmes par message_id, en préservant l'ordre."""
    result = []
    for _, group in df.groupby("message_id", sort=False):
        result.append(group["lemma"].tolist())
    return result


# ---------------------------------------------------------------------------
# Graphe networkx
# ---------------------------------------------------------------------------

def construire_graphe(pmi_edges, freq_counter, pmi_threshold):
    """
    Construit un graphe pondéré par PMI.
    Ne garde que les arêtes avec PMI >= pmi_threshold.
    """
    G = nx.Graph()

    for (a, b), data in pmi_edges.items():
        if data["pmi"] >= pmi_threshold:
            G.add_edge(a, b, weight=data["pmi"], raw_count=data["count"])

    # Fréquence sur les nœuds
    for node in G.nodes():
        G.nodes[node]["freq"] = freq_counter.get(node, 0)

    return G


def calculer_indicateurs(G):
    """Calcule degré, betweenness, eigenvector, closeness."""
    if len(G) == 0:
        return {}

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    return {
        node: {
            "degree": degree[node],
            "weighted_degree": round(weighted_degree[node], 3),
            "betweenness": round(betweenness[node], 6),
            "eigenvector": round(eigenvector.get(node, 0.0), 6),
            "closeness": round(closeness.get(node, 0.0), 6),
        }
        for node in G.nodes()
    }


# ---------------------------------------------------------------------------
# Export CSV Gephi
# ---------------------------------------------------------------------------

def exporter_edges(G, path):
    """Export edges CSV : Source,Target,Weight,RawCount."""
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "Source": u,
            "Target": v,
            "Weight": data["weight"],
            "RawCount": data.get("raw_count", 0),
        })
    df = pd.DataFrame(rows).sort_values("Weight", ascending=False)
    df.to_csv(path, index=False)
    return len(df)


def exporter_nodes(G, indicators, path):
    """Export nodes CSV Gephi-ready."""
    rows = []
    for node in G.nodes():
        ind = indicators.get(node, {})
        rows.append({
            "Id": node,
            "Label": node,
            "Freq": G.nodes[node].get("freq", 0),
            "Degree": ind.get("degree", 0),
            "WeightedDegree": ind.get("weighted_degree", 0),
            "Betweenness": ind.get("betweenness", 0.0),
            "Eigenvector": ind.get("eigenvector", 0.0),
            "Closeness": ind.get("closeness", 0.0),
        })
    df = pd.DataFrame(rows).sort_values("Betweenness", ascending=False)
    df.to_csv(path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# Traitement d'une source
# ---------------------------------------------------------------------------

def traiter_source(df, src, output_dir, top_n, window, min_count, pmi_threshold,
                   phase_suffix=None):
    """
    Traite une source : co-occurrence, PMI, graphe, indicateurs, export.

    phase_suffix : None pour le corpus entier (fichiers `cooc_*_{src}.csv`),
    sinon un label court (ex. "P1") → fichiers `cooc_*_{src}_{suffix}.csv`.

    Retourne (indicators, freq_dict) ou (None, None) si le traitement est sauté.
    """
    # Fréquences locales (global ou phase-restreintes selon le df reçu)
    freq = df["lemma"].value_counts()
    top_lemmes = set(freq.head(top_n).index)

    if len(top_lemmes) < 10:
        print(f"    [SKIP] < 10 lemmes après filtrage")
        return None, None

    # Co-occurrence
    lemmes_by_msg = regrouper_lemmes_par_message(df)
    cooc, doc_freq, n_messages = construire_cooccurrence(lemmes_by_msg, top_lemmes, window)

    print(f"    {n_messages} messages, {len(cooc)} paires brutes")

    # PMI
    pmi_edges = calculer_pmi(cooc, doc_freq, n_messages, min_count=min_count)
    print(f"    {len(pmi_edges)} paires après min_count={min_count}")

    # Graphe
    G = construire_graphe(pmi_edges, freq.to_dict(), pmi_threshold)

    if len(G) == 0:
        print(f"    [SKIP] graphe vide (pmi_threshold={pmi_threshold})")
        return None, None

    # Indicateurs
    indicators = calculer_indicateurs(G)

    # Construction du suffixe de fichier (phase ou vide)
    tag = f"_{phase_suffix}" if phase_suffix else ""

    # Export matrice brute (carrée, lemme × lemme, valeurs = count)
    matrix_path = os.path.join(output_dir, f"cooc_matrix_{src}{tag}.csv")
    all_lemmes = sorted(top_lemmes)
    lemma_idx = {l: i for i, l in enumerate(all_lemmes)}
    mat = np.zeros((len(all_lemmes), len(all_lemmes)), dtype=int)
    for (a, b), count in cooc.items():
        if a in lemma_idx and b in lemma_idx:
            i, j = lemma_idx[a], lemma_idx[b]
            mat[i, j] = count
            mat[j, i] = count
    df_matrix = pd.DataFrame(mat, index=all_lemmes, columns=all_lemmes)
    df_matrix.to_csv(matrix_path)
    print(f"    Matrice {len(all_lemmes)}×{len(all_lemmes)} -> {matrix_path}")

    # Export
    edges_path = os.path.join(output_dir, f"cooc_edges_{src}{tag}.csv")
    nodes_path = os.path.join(output_dir, f"cooc_nodes_{src}{tag}.csv")

    n_edges = exporter_edges(G, edges_path)
    n_nodes = exporter_nodes(G, indicators, nodes_path)

    # Stats réseau
    density = nx.density(G)
    components = nx.number_connected_components(G)
    top_between = sorted(indicators.items(),
                         key=lambda x: x[1]["betweenness"], reverse=True)[:5]
    top_eigen = sorted(indicators.items(),
                       key=lambda x: x[1]["eigenvector"], reverse=True)[:5]

    print(f"    {n_nodes} nœuds, {n_edges} arêtes, "
          f"densité={density:.4f}, composantes={components}")
    print(f"    Top betweenness : "
          f"{', '.join(f'{k}({v['betweenness']:.3f})' for k, v in top_between)}")
    print(f"    Top eigenvector : "
          f"{', '.join(f'{k}({v['eigenvector']:.3f})' for k, v in top_eigen)}")
    print(f"    -> {edges_path}")
    print(f"    -> {nodes_path}")

    return indicators, freq.to_dict()


# ---------------------------------------------------------------------------
# Évolution de centralité par phase
# ---------------------------------------------------------------------------

def construire_evolution_centralite(indicateurs_par_phase, freq_par_phase, output_path):
    """
    Construit un tableau lemme × phase × indicateur et le sauve en CSV.

    indicateurs_par_phase : dict {phase_short: {lemma: {degree, betweenness, ...}}}
    freq_par_phase       : dict {phase_short: {lemma: freq}}

    Une ligne par lemme ayant apparu dans le graphe d'au moins une phase.
    Colonnes : lemma, {phase}_freq, {phase}_degree, {phase}_betweenness,
               {phase}_eigenvector, {phase}_closeness.
    Les phases absentes sont mises à 0 (lemme hors top-N dans cette phase).
    """
    phases = sorted(indicateurs_par_phase.keys())
    all_lemmes = set()
    for ind in indicateurs_par_phase.values():
        all_lemmes.update(ind.keys())

    rows = []
    for lemma in sorted(all_lemmes):
        row = {"lemma": lemma}
        for phase in phases:
            ind = indicateurs_par_phase.get(phase, {}).get(lemma, {})
            row[f"{phase}_freq"] = freq_par_phase.get(phase, {}).get(lemma, 0)
            row[f"{phase}_degree"] = ind.get("degree", 0)
            row[f"{phase}_betweenness"] = ind.get("betweenness", 0.0)
            row[f"{phase}_eigenvector"] = ind.get("eigenvector", 0.0)
            row[f"{phase}_closeness"] = ind.get("closeness", 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Tri par max betweenness sur l'ensemble des phases — met en tête les
    # lemmes au plus grand rôle structurant toutes phases confondues
    bet_cols = [c for c in df.columns if c.endswith("_betweenness")]
    df["_max_bet"] = df[bet_cols].max(axis=1)
    df = df.sort_values("_max_bet", ascending=False).drop(columns=["_max_bet"])
    df.to_csv(output_path, index=False)
    print(f"    Évolution de centralité ({len(df)} lemmes, "
          f"{len(phases)} phases) -> {output_path}")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def etiquette_phase_courte(phase_value):
    """
    Normalise une valeur de colonne 'phase' (ex. 'P1_Artisanal') en label court
    (ex. 'P1'). Retourne la valeur d'origine si le format n'est pas reconnu.
    """
    if not isinstance(phase_value, str) or not phase_value:
        return None
    # Format attendu dans lemmes_*.csv : 'P1_Artisanal', 'P2_Semi-pro', ...
    token = phase_value.split("_", 1)[0]
    return token if token.startswith("P") and token[1:].isdigit() else phase_value


def main():
    parser = argparse.ArgumentParser(
        description="Co-occurrences PMI + réseau — corpus Magyar (LEXICO §4)")
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--sources", nargs="*", default=["caption", "dialogue"],
                        help="Sources à traiter (défaut : caption dialogue)")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Top N lemmes (défaut : 200)")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Nb min de co-occurrences pour calculer le PMI (défaut : 3)")
    parser.add_argument("--pmi-threshold", type=float, default=None,
                        help="Seuil PMI min pour garder une arête. "
                             "Par défaut : 2.0 caption / 1.0 dialogue "
                             "(le dialogue a plus d'entropie lexicale, donc "
                             "un seuil identique filtrerait trop)")
    parser.add_argument("--per-phase", action="store_true",
                        help="Produit aussi un graphe par phase + tableau "
                             "cooc_centralite_evolution_{src}.csv")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml")
    args = parser.parse_args()

    load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Co-occurrences lexicales (PMI)")
    print(f"  Top N : {args.top_n}, min_count : {args.min_count}, "
          f"PMI threshold : {args.pmi_threshold}, "
          f"per_phase : {args.per_phase}\n")

    for src in args.sources:
        csv_path = os.path.join(args.output_dir, f"lemmes_{src}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {src} : {csv_path} introuvable")
            continue

        window = WINDOW_SIZES.get(src, DEFAULT_WINDOW)
        pmi_threshold = (args.pmi_threshold if args.pmi_threshold is not None
                         else PMI_THRESHOLDS.get(src, DEFAULT_PMI_THRESHOLD))
        print(f"── {src} (fenêtre={window}, PMI≥{pmi_threshold}) ──")

        df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
        df = df[df["phase"].notna() & (df["phase"] != "")]

        # Graphe global (toutes phases confondues)
        print("  [global]")
        traiter_source(df, src, args.output_dir,
                       args.top_n, window, args.min_count, pmi_threshold)
        print()

        if not args.per_phase:
            continue

        # Graphes par phase + collecte des indicateurs pour le tableau
        # d'évolution de centralité.
        indicateurs_par_phase = {}
        freq_par_phase = {}

        for phase_raw in sorted(df["phase"].unique()):
            phase_short = etiquette_phase_courte(phase_raw)
            if phase_short is None:
                continue
            df_phase = df[df["phase"] == phase_raw]
            print(f"  [{phase_short} = {phase_raw}]")
            indicators, freq = traiter_source(
                df_phase, src, args.output_dir,
                args.top_n, window, args.min_count, pmi_threshold,
                phase_suffix=phase_short,
            )
            if indicators is not None:
                indicateurs_par_phase[phase_short] = indicators
                freq_par_phase[phase_short] = freq
            print()

        if indicateurs_par_phase:
            evolution_path = os.path.join(
                args.output_dir, f"cooc_centralite_evolution_{src}.csv"
            )
            construire_evolution_centralite(
                indicateurs_par_phase, freq_par_phase, evolution_path
            )
            print()

    print("Terminé.")


if __name__ == "__main__":
    main()
