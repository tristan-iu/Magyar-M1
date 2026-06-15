#!/usr/bin/env python3
"""
kwic.py — Keywords In Context.

Outil de close reading : cherche un ou plusieurs lemmes dans les CSV
de lemmes et affiche le contexte linéaire (fenêtre configurable).

Mode --summary : au lieu de dumper les concordances ligne par ligne,
construit un tableau synthétique des voisins gauche/droite les plus
fréquents par phase, scorés par PMI local (sur les fréquences d'unigrammes
de la phase). Sert à alterner lecture délinéarisée (TF-IDF, cooccurrences)
et lecture linéaire (concordances) sans repasser à la lecture intégrale.

Usage :
    python kwic.py --lemma збір --source caption
    python kwic.py --lemma збір сбс --source dialogue --phase P3
    python kwic.py --lemma збір --source caption --export kwic_zbir.csv
    python kwic.py --lemma збір --window 8
    python kwic.py --lemma мадяр --summary
    python kwic.py --lemma сбс --summary --source caption --top 15
"""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import PHASE_SHORT  # noqa: E402

# Les CSV de lemmes sont produits par lexicometrie.py dans 4_data_et_viz/lexico/.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "4_data_et_viz" / "lexico"


def chercher_kwic(csv_path, lemmas, window=5, phase_filter=None):
    """
    Cherche les lemmes dans le CSV et retourne les concordances KWIC.

    Retourne une liste de dicts :
      message_id, date, phase, lemma, pos, context
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})

    if phase_filter:
        df = df[df["phase"].str.contains(phase_filter, case=False, na=False)]

    lemmas_set = set(lemmas)
    results = []

    for mid, group in df.groupby("message_id", sort=False):
        tokens = group["token"].tolist()
        lemma_col = group["lemma"].tolist()
        pos_col = group["pos"].tolist()
        date = group["date"].iloc[0]
        phase = group["phase"].iloc[0] if pd.notna(group["phase"].iloc[0]) else ""

        for i, lemma in enumerate(lemma_col):
            if lemma not in lemmas_set:
                continue

            before = tokens[max(0, i - window):i]
            target = tokens[i]
            after = tokens[i + 1:i + 1 + window]

            context = " ".join(before) + " [" + target + "] " + " ".join(after)

            results.append({
                "message_id": mid,
                "date": date,
                "phase": phase,
                "lemma": lemma,
                "token": target,
                "pos": pos_col[i],
                "context": context,
            })

    return results


def resumer_voisins(csv_path, target_lemma, window=5, top=10):
    """
    Pour un lemme cible, construit un résumé des voisins gauche/droite
    les plus fréquents par phase, scorés par PMI local.

    PMI(neighbor, target | phase)
      = log2( count(neighbor dans fenêtre cible) / expected )
    où expected = count(target) * freq(neighbor) / N_phase
    (N_phase = nb total de tokens de la phase).

    Retourne un DataFrame (phase, position, neighbor, count, pmi).
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
    df = df[df["phase"].notna() & (df["phase"] != "")]

    rows = []
    for phase_raw in sorted(df["phase"].unique()):
        phase_short = PHASE_SHORT.get(phase_raw, phase_raw[:5])
        sub = df[df["phase"] == phase_raw]
        lemma_freq = Counter(sub["lemma"])
        n_phase = lemma_freq.total() if hasattr(lemma_freq, "total") \
            else sum(lemma_freq.values())
        n_target = lemma_freq.get(target_lemma, 0)
        if n_target == 0:
            continue

        left = Counter()
        right = Counter()
        for _, group in sub.groupby("message_id", sort=False):
            lems = group["lemma"].tolist()
            for i, lem in enumerate(lems):
                if lem != target_lemma:
                    continue
                for l in lems[max(0, i - window):i]:
                    if l != target_lemma:
                        left[l] += 1
                for r in lems[i + 1:i + 1 + window]:
                    if r != target_lemma:
                        right[r] += 1

        def _rank(counter, pos):
            for neighbor, c in counter.most_common(top):
                f = lemma_freq.get(neighbor, 0)
                if f == 0 or n_target == 0 or n_phase == 0:
                    continue
                # PMI local : log2( obs / expected ).
                # obs = c / (n_target * window)  — fraction de slots voisins
                # expected = f / n_phase         — fréquence marginale
                p_ctx = c / (n_target * window)
                p_marg = f / n_phase
                if p_marg == 0 or p_ctx == 0:
                    continue
                pmi = math.log2(p_ctx / p_marg)
                rows.append({
                    "phase": phase_short,
                    "position": pos,
                    "neighbor": neighbor,
                    "count": c,
                    "pmi_local": round(pmi, 3),
                    "n_target": n_target,
                })

        _rank(left, "left")
        _rank(right, "right")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="KWIC — Keywords In Context (corpus Magyar)")
    parser.add_argument("--lemma", nargs="+", required=True,
                        help="Lemme(s) à chercher")
    parser.add_argument("--source", default="caption",
                        choices=["caption", "dialogue", "ocr", "combined"],
                        help="Source (défaut : caption)")
    parser.add_argument("--phase", default=None,
                        help="Filtrer par phase (ex : P1, P2, P3)")
    parser.add_argument("--window", type=int, default=5,
                        help="Fenêtre de contexte en tokens (défaut : 5)")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--export", default=None,
                        help="Exporter en CSV (chemin du fichier)")
    parser.add_argument("--summary", action="store_true",
                        help="Mode résumé : voisins gauche/droite top-N par "
                             "phase avec PMI local (au lieu du dump KWIC)")
    parser.add_argument("--top", type=int, default=10,
                        help="En mode --summary : nb de voisins par côté (défaut : 10)")
    args = parser.parse_args()

    out_dir = args.output_dir or str(OUTPUT_DIR)
    csv_path = os.path.join(out_dir, f"lemmes_{args.source}.csv")

    if not os.path.exists(csv_path):
        print(f"Fichier introuvable : {csv_path}", file=sys.stderr)
        sys.exit(1)

    if args.summary:
        if len(args.lemma) != 1:
            print("--summary attend un seul lemme cible", file=sys.stderr)
            sys.exit(2)
        target = args.lemma[0]
        df_sum = resumer_voisins(csv_path, target, args.window, args.top)
        if df_sum.empty:
            print(f"Aucune occurrence de '{target}' dans {args.source}")
            return
        if args.export:
            df_sum.to_csv(args.export, index=False)
            print(f"{len(df_sum)} voisins -> {args.export}")
        else:
            print(f"Voisins de '{target}' dans {args.source} "
                  f"(top {args.top}, fenêtre ±{args.window}) :\n")
            for phase in sorted(df_sum["phase"].unique()):
                sub = df_sum[df_sum["phase"] == phase]
                n_target = sub["n_target"].iloc[0]
                print(f"── {phase} (n={n_target}) ──")
                for pos in ("left", "right"):
                    side = sub[sub["position"] == pos].sort_values(
                        "pmi_local", ascending=False)
                    if side.empty:
                        continue
                    label = "gauche" if pos == "left" else "droite "
                    for _, r in side.iterrows():
                        print(f"  {label} | {r['neighbor']:<20s} "
                              f"count={r['count']:>4}  PMI={r['pmi_local']:>6.2f}")
                print()
        return

    results = chercher_kwic(csv_path, args.lemma, args.window, args.phase)

    if not results:
        print(f"Aucun résultat pour {args.lemma} dans {args.source}"
              + (f" (phase={args.phase})" if args.phase else ""))
        return

    # Tri par date
    results.sort(key=lambda r: r["date"])

    if args.export:
        df = pd.DataFrame(results)
        df.to_csv(args.export, index=False)
        print(f"{len(results)} concordances -> {args.export}")
    else:
        print(f"{len(results)} concordances pour {args.lemma} dans {args.source}"
              + (f" (phase={args.phase})" if args.phase else "") + "\n")
        for r in results:
            phase = r["phase"][:5] if r["phase"] else "    "
            print(f"  {phase}  {r['date'][:10]}  msg_{r['message_id']:>5s}  {r['context']}")


if __name__ == "__main__":
    main()
