#!/usr/bin/env python3
"""
evaluate_lemmas.py — Ground truth pour la lemmatisation spaCy (AUDIT B3).

Deux modes d'utilisation :

  1. Générer un échantillon à annoter manuellement :
     python evaluate_lemmas.py sample \
         --input 4_data_et_viz/lexico/lemmes_combined.csv \
         --output 3a_lexicometrie/evaluation_sample.csv

  2. Calculer les métriques depuis l'échantillon annoté :
     python evaluate_lemmas.py evaluate \
         --input 3a_lexicometrie/evaluation_sample.csv

Protocole :
  On compare un échantillon validé manuellement (vérité terrain) à la
  sortie de l'algorithme. C'est ce qui mesure la fiabilité — ne pas prendre
  les sorties spaCy pour argent comptant sans vérification.

Échantillonnage :
  - 100 tokens par phase (300 total par défaut)
  - Stratifié par POS (proportionnel à la distribution réelle)
  - Contexte : fenêtre de ±5 tokens du même message

Annotation :
  Remplir les colonnes `is_correct` (1 = bon lemme, 0 = erreur) et
  `lemma_corrected` (le bon lemme si erreur). Laisser `lemma_corrected`
  vide si le lemme spaCy est correct.

Dépendances : pandas
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Échantillonnage
# ---------------------------------------------------------------------------

def construire_index_contexte(df):
    """
    Reconstruit l'ordre des tokens par message pour produire un contexte KWIC.

    On groupe par message_id et on conserve l'ordre d'apparition (= l'ordre
    des lignes dans le CSV, qui reflète l'ordre de tokenisation spaCy).

    Entrée : df — DataFrame complet des lemmes
    Sortie : dict {message_id: [(index_df, token), ...]}
    """
    context = {}
    for mid, group in df.groupby("message_id", sort=False):
        context[mid] = list(zip(group.index, group["token"].values))
    return context


def extraire_kwic(context_index, message_id, row_index, window=5):
    """
    Retourne une concordance (KWIC) autour du token cible.

    Format : « ...avant... [TOKEN] ...après... »
    La fenêtre est de ±window tokens.
    """
    tokens = context_index.get(message_id, [])
    # On cherche la position du token dans la séquence du message
    pos = None
    for i, (idx, _tok) in enumerate(tokens):
        if idx == row_index:
            pos = i
            break
    if pos is None:
        return ""

    before = [t for _, t in tokens[max(0, pos - window):pos]]
    target = tokens[pos][1]
    after  = [t for _, t in tokens[pos + 1:pos + 1 + window]]

    return " ".join(before) + " [" + target + "] " + " ".join(after)


def echantillonner_tokens(df, n_per_phase=100, seed=42):
    """
    Échantillonne n tokens par phase, stratifié par POS.

    La stratification est proportionnelle : si NOUN = 60% des tokens d'une
    phase, ~60 des 100 tokens seront des NOUN. Cela reflète la distribution
    réelle du corpus et permet de mesurer le taux d'erreur là où il compte
    le plus (les catégories les plus fréquentes).
    """
    samples = []

    for phase, phase_df in df.groupby("phase"):
        if phase is None or str(phase) == "None" or str(phase) == "nan":
            continue

        total_phase = len(phase_df)
        if total_phase == 0:
            continue

        # On calcule la proportion de chaque POS dans la phase
        pos_counts = phase_df["pos"].value_counts()

        phase_samples = []
        remaining = n_per_phase

        for pos_tag, count in pos_counts.items():
            # Nombre proportionnel, au moins 1 si la catégorie existe
            n_for_pos = max(1, round(n_per_phase * count / total_phase))
            n_for_pos = min(n_for_pos, remaining, count)

            if n_for_pos <= 0:
                continue

            pos_subset = phase_df[phase_df["pos"] == pos_tag]
            sampled = pos_subset.sample(n=min(n_for_pos, len(pos_subset)),
                                        random_state=seed)
            phase_samples.append(sampled)
            remaining -= len(sampled)

            if remaining <= 0:
                break

        if phase_samples:
            samples.append(pd.concat(phase_samples))

    if not samples:
        print("ERREUR : aucun token échantillonné — vérifier les phases.", file=sys.stderr)
        sys.exit(1)

    return pd.concat(samples).sort_values(["phase", "pos", "message_id"])


def generer_echantillon(args):
    """Mode 'sample' : lit le CSV de lemmes et exporte un fichier d'annotation."""
    print(f"Chargement de {args.input}...")
    df = pd.read_csv(args.input, dtype={"message_id": str, "phase": str})
    print(f"  {len(df)} tokens, phases : {sorted(df['phase'].dropna().unique())}")

    # On échantillonne
    print(f"\nÉchantillonnage : {args.n_per_phase} tokens par phase...")
    sampled = echantillonner_tokens(df, n_per_phase=args.n_per_phase, seed=args.seed)
    print(f"  {len(sampled)} tokens sélectionnés")

    # On construit le contexte KWIC pour chaque token
    print("Construction des contextes KWIC...")
    context_index = construire_index_contexte(df)

    kwic_col = []
    for idx, row in sampled.iterrows():
        kwic = extraire_kwic(context_index, row["message_id"], idx, window=args.window)
        kwic_col.append(kwic)

    # On construit le DataFrame de sortie
    output_df = pd.DataFrame({
        "message_id":     sampled["message_id"].values,
        "date":           sampled["date"].values,
        "phase":          sampled["phase"].values,
        "pos":            sampled["pos"].values,
        "token":          sampled["token"].values,
        "lemma_spacy":    sampled["lemma"].values,
        "context":        kwic_col,
        "is_correct":     "",    # à remplir : 1 (correct) ou 0 (erreur)
        "lemma_corrected": "",   # à remplir si is_correct = 0
        "notes":          "",    # commentaire libre (ex : « argot transcarpatique »)
    })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\n-> {args.output} ({len(output_df)} lignes)")
    print(f"\nProchaine étape : ouvrir {args.output} dans un tableur,")
    print("remplir is_correct (1/0) et lemma_corrected (si erreur),")
    print(f"puis lancer : python evaluate_lemmas.py evaluate --input {args.output}")


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluer_echantillon(args):
    """Mode 'evaluate' : lit le fichier annoté et calcule les métriques."""
    print(f"Chargement de {args.input}...")
    df = pd.read_csv(args.input, dtype=str)

    # On vérifie que l'annotation est complète
    annotated = df[df["is_correct"].isin(["0", "1"])].copy()
    annotated["is_correct"] = annotated["is_correct"].astype(int)
    total = len(annotated)
    missing = len(df) - total

    if total == 0:
        print("ERREUR : aucun token annoté (colonne is_correct vide).", file=sys.stderr)
        sys.exit(1)
    if missing > 0:
        print(f"  ATTENTION : {missing} tokens non annotés (ignorés).")

    correct = annotated["is_correct"].sum()
    errors  = total - correct
    rate    = errors / total * 100

    # ── Résultats globaux ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RÉSULTATS GROUND TRUTH — LEMMATISATION spaCy")
    print(f"{'='*60}")
    print(f"  Tokens annotés : {total}")
    print(f"  Corrects       : {correct} ({correct/total*100:.1f}%)")
    print(f"  Erreurs        : {errors} ({rate:.1f}%)")
    print(f"{'='*60}")

    if rate > 10:
        print(f"\n  ⚠  Taux d'erreur > 10% — caveat requis dans la méthodologie du mémoire")
    else:
        print(f"\n  ✓  Taux d'erreur acceptable (< 10%)")

    # ── Par phase ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PAR PHASE")
    print(f"{'─'*60}")
    phase_stats = annotated.groupby("phase").agg(
        total=("is_correct", "count"),
        correct=("is_correct", "sum"),
    )
    phase_stats["errors"]     = phase_stats["total"] - phase_stats["correct"]
    phase_stats["error_rate"] = (phase_stats["errors"] / phase_stats["total"] * 100).round(1)
    print(phase_stats.to_string())

    # ── Par POS ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PAR CATÉGORIE POS")
    print(f"{'─'*60}")
    pos_stats = annotated.groupby("pos").agg(
        total=("is_correct", "count"),
        correct=("is_correct", "sum"),
    )
    pos_stats["errors"]     = pos_stats["total"] - pos_stats["correct"]
    pos_stats["error_rate"] = (pos_stats["errors"] / pos_stats["total"] * 100).round(1)
    pos_stats = pos_stats.sort_values("error_rate", ascending=False)
    print(pos_stats.to_string())

    # ── Erreurs les plus fréquentes ───────────────────────────────────────
    err_df = annotated[annotated["is_correct"] == 0]
    if not err_df.empty:
        print(f"\n{'─'*60}")
        print("  ERREURS FRÉQUENTES (top 20)")
        print(f"{'─'*60}")
        for _, row in err_df.head(20).iterrows():
            correction = row.get("lemma_corrected", "?")
            note       = row.get("notes", "")
            print(f"  {row['token']:20s}  →  {row['lemma_spacy']:20s}  "
                  f"(correct: {correction})  [{row['pos']}] {note}")

    # ── Export résumé ─────────────────────────────────────────────────────
    summary_path = args.input.replace(".csv", "_results.csv")
    summary = pd.concat([
        phase_stats.reset_index().assign(group_by="phase").rename(columns={"phase": "group"}),
        pos_stats.reset_index().assign(group_by="pos").rename(columns={"pos": "group"}),
    ])
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n-> Résumé exporté : {summary_path}")

    return rate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ground truth lemmatisation spaCy — corpus Magyar (AUDIT B3)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sous-commande sample
    sp_sample = subparsers.add_parser("sample",
        help="Générer un échantillon à annoter manuellement")
    sp_sample.add_argument("--input", required=True,
        help="CSV de lemmes (ex : lemmes_combined.csv)")
    sp_sample.add_argument("--output", default="evaluation_sample.csv",
        help="Fichier de sortie pour annotation (défaut : evaluation_sample.csv)")
    sp_sample.add_argument("--n-per-phase", type=int, default=100,
        help="Nombre de tokens par phase (défaut : 100)")
    sp_sample.add_argument("--window", type=int, default=5,
        help="Fenêtre KWIC en tokens de chaque côté (défaut : 5)")
    sp_sample.add_argument("--seed", type=int, default=42,
        help="Seed pour la reproductibilité de l'échantillonnage (défaut : 42)")

    # Sous-commande evaluate
    sp_eval = subparsers.add_parser("evaluate",
        help="Calculer les métriques depuis l'échantillon annoté")
    sp_eval.add_argument("--input", required=True,
        help="CSV annoté (le fichier produit par sample, rempli à la main)")

    args = parser.parse_args()

    if args.command == "sample":
        generer_echantillon(args)
    elif args.command == "evaluate":
        evaluer_echantillon(args)


if __name__ == "__main__":
    main()
