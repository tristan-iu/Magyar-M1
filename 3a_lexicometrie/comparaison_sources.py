#!/usr/bin/env python3
"""
comparaison_sources.py — Divergence lexicale caption × dialogue par phase.

Question méthodologique : le canal textuel (caption) et le canal oral
(dialogue) ne se professionnalisent pas au même rythme. On le
mesure lexicalement via :

  1. Jaccard(top-N_caption, top-N_dialogue) par phase — plus c'est bas,
     plus les deux canaux ont des vocabulaires distincts.
  2. Chi² d'indépendance sur la matrice 2×N des fréquences top-N partagées
     — teste si la distribution d'usage diverge entre canaux.
  3. Heatmap 3 phases × {Jaccard, Dice, overlap-coefficient}.

Sorties :
  comparaison_sources_jaccard.csv      — phase × top-N × jaccard/dice/overlap
  comparaison_sources_divergents.csv   — lemmes dans top-N d'une source mais
                                          pas de l'autre (par phase)
  fig_comparaison_sources.png          — heatmap Jaccard + bar chi²

Usage :
    python comparaison_sources.py            # défaut : 4_data_et_viz/lexico
    python comparaison_sources.py --output-dir ... --top-n 100
"""

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config, PHASE_SHORT  # noqa: E402

# CSV lus dans 4_data_et_viz/lexico/ ; figure écrite un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


TOP_N_LIST = [20, 50, 100, 200]


# ---------------------------------------------------------------------------
# Mesures de similarité
# ---------------------------------------------------------------------------

def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def dice(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def overlap(a, b):
    """Coefficient de chevauchement (Szymkiewicz–Simpson)."""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def top_lemmes_par_freq(df, n):
    counts = df["lemma"].value_counts()
    return counts.head(n).index.tolist(), counts


def calculer_divergents(top_cap, top_dia):
    """Lemmes spécifiques à une source (dans top_N d'une mais pas de l'autre)."""
    set_cap, set_dia = set(top_cap), set(top_dia)
    only_cap = [l for l in top_cap if l not in set_dia]
    only_dia = [l for l in top_dia if l not in set_cap]
    return only_cap, only_dia


def chi2_sur_partages(counts_cap, counts_dia, top_n):
    """
    Chi² d'indépendance sur les lemmes partagés dans top-N.
    Table 2×k : [freq_caption, freq_dialogue] × k lemmes partagés.
    """
    set_cap = set(counts_cap.head(top_n).index)
    set_dia = set(counts_dia.head(top_n).index)
    shared = sorted(set_cap & set_dia)
    if len(shared) < 2:
        return None
    obs = np.array([
        [counts_cap.get(l, 0) for l in shared],
        [counts_dia.get(l, 0) for l in shared],
    ], dtype=float)
    # Si une ligne est nulle, chi² n'est pas défini
    if obs.sum(axis=1).min() == 0:
        return None
    chi2, p, dof, _ = chi2_contingency(obs)
    return {"chi2": round(chi2, 2), "p": p, "dof": dof, "n_shared": len(shared)}


def traiter_phase(df_cap, df_dia, phase_label, top_n_list):
    rows_sim = []
    rows_div = []
    chi2_results = {}

    for n in top_n_list:
        top_cap, counts_cap = top_lemmes_par_freq(df_cap, n)
        top_dia, counts_dia = top_lemmes_par_freq(df_dia, n)

        rows_sim.append({
            "phase": phase_label,
            "top_n": n,
            "jaccard": round(jaccard(top_cap, top_dia), 4),
            "dice": round(dice(top_cap, top_dia), 4),
            "overlap": round(overlap(top_cap, top_dia), 4),
            "n_shared": len(set(top_cap) & set(top_dia)),
        })

        if n in (50, 100):
            # On ne sort les listes de divergents qu'à top-50 et top-100
            # pour limiter le bruit
            only_cap, only_dia = calculer_divergents(top_cap, top_dia)
            for rank, lem in enumerate(only_cap, 1):
                rows_div.append({"phase": phase_label, "top_n": n,
                                 "source": "caption_only", "rank": rank, "lemma": lem})
            for rank, lem in enumerate(only_dia, 1):
                rows_div.append({"phase": phase_label, "top_n": n,
                                 "source": "dialogue_only", "rank": rank, "lemma": lem})

        chi2 = chi2_sur_partages(
            df_cap["lemma"].value_counts(),
            df_dia["lemma"].value_counts(),
            n,
        )
        if chi2 is not None:
            chi2_results[(phase_label, n)] = chi2

    return rows_sim, rows_div, chi2_results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def tracer_comparaison(df_sim, chi2_results, fig_dir):
    phases = ["P1", "P2", "P3"]
    top_ns = sorted(df_sim["top_n"].unique())
    # Matrice jaccard phase × top_n
    mat = np.zeros((len(phases), len(top_ns)))
    for i, phase in enumerate(phases):
        for j, n in enumerate(top_ns):
            row = df_sim[(df_sim["phase"] == phase) & (df_sim["top_n"] == n)]
            if not row.empty:
                mat[i, j] = row["jaccard"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    # 1. Heatmap Jaccard
    im = ax1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=0.5, aspect="auto")
    ax1.set_xticks(range(len(top_ns)))
    ax1.set_xticklabels([f"top-{n}" for n in top_ns])
    ax1.set_yticks(range(len(phases)))
    ax1.set_yticklabels(phases)
    ax1.set_title("Jaccard(caption, dialogue) par phase × top-N")
    for i in range(len(phases)):
        for j in range(len(top_ns)):
            ax1.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     color="black", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax1, label="Jaccard")

    # 2. Barres chi² à top-100
    n_ref = 100 if any(k[1] == 100 for k in chi2_results) else top_ns[-1]
    chi2_vals = []
    labels = []
    for phase in phases:
        key = (phase, n_ref)
        if key in chi2_results:
            chi2_vals.append(chi2_results[key]["chi2"])
            labels.append(f"{phase}\nn_shared={chi2_results[key]['n_shared']}")
        else:
            chi2_vals.append(0)
            labels.append(phase)
    ax2.bar(labels, chi2_vals, color=["#E67E22", "#2980B9", "#27AE60"])
    ax2.set_ylabel(f"χ² (top-{n_ref} partagés)")
    ax2.set_title(f"Divergence d'usage entre canaux (χ², top-{n_ref})")
    for i, v in enumerate(chi2_vals):
        ax2.text(i, v + max(chi2_vals) * 0.02, f"{v:.0f}",
                 ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Comparaison lexicale caption × dialogue",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(fig_dir, "fig_comparaison_sources.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--top-n", type=int, nargs="*", default=TOP_N_LIST,
                        help=f"Top-N à évaluer (défaut : {TOP_N_LIST})")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    load_config(args.config)

    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    cap_path = os.path.join(args.output_dir, "lemmes_caption.csv")
    dia_path = os.path.join(args.output_dir, "lemmes_dialogue.csv")
    for p in (cap_path, dia_path):
        if not os.path.exists(p):
            sys.exit(f"[ERREUR] {p} introuvable — lancez lexicometrie.py d'abord")

    df_cap = pd.read_csv(cap_path, dtype={"message_id": str, "phase": str})
    df_dia = pd.read_csv(dia_path, dtype={"message_id": str, "phase": str})
    df_cap = df_cap[df_cap["phase"].notna() & (df_cap["phase"] != "")]
    df_dia = df_dia[df_dia["phase"].notna() & (df_dia["phase"] != "")]

    all_sim = []
    all_div = []
    all_chi2 = {}

    print("Comparaison caption × dialogue")
    for phase_raw, phase_short in PHASE_SHORT.items():
        sub_cap = df_cap[df_cap["phase"] == phase_raw]
        sub_dia = df_dia[df_dia["phase"] == phase_raw]
        if sub_cap.empty or sub_dia.empty:
            print(f"  [SKIP] {phase_short} : corpus vide")
            continue
        print(f"── {phase_short} ── caption={len(sub_cap)} tokens, "
              f"dialogue={len(sub_dia)} tokens")
        rows_sim, rows_div, chi2_results = traiter_phase(
            sub_cap, sub_dia, phase_short, args.top_n)
        all_sim.extend(rows_sim)
        all_div.extend(rows_div)
        all_chi2.update(chi2_results)

        for r in rows_sim:
            print(f"    top-{r['top_n']:<3} | jaccard={r['jaccard']:.3f} "
                  f"dice={r['dice']:.3f} overlap={r['overlap']:.3f} "
                  f"partagés={r['n_shared']}")

    df_sim = pd.DataFrame(all_sim)
    df_div = pd.DataFrame(all_div)

    sim_csv = os.path.join(args.output_dir, "comparaison_sources_jaccard.csv")
    df_sim.to_csv(sim_csv, index=False)
    print(f"\n  -> {sim_csv}")

    div_csv = os.path.join(args.output_dir, "comparaison_sources_divergents.csv")
    df_div.to_csv(div_csv, index=False)
    print(f"  -> {div_csv}")

    # Chi² résumé
    chi2_rows = [{"phase": k[0], "top_n": k[1], **v} for k, v in all_chi2.items()]
    if chi2_rows:
        chi2_csv = os.path.join(args.output_dir, "comparaison_sources_chi2.csv")
        pd.DataFrame(chi2_rows).to_csv(chi2_csv, index=False)
        print(f"  -> {chi2_csv}")

    tracer_comparaison(df_sim, all_chi2, fig_dir)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
