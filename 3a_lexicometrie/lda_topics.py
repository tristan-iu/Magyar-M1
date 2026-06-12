#!/usr/bin/env python3
"""
lda_topics.py — Topic modeling LDA gensim.

Pipeline :
  1. Charge les CSV de lemmes (caption, dialogue)
  2. Construit un DTM (Document-Term Matrix) par message
  3. Boucle k = 3..10 : cohérence c_v + perplexité → courbe
  4. Entraîne le modèle final au K optimal (alpha/beta = 'auto')
  5. Exporte : LDAvis HTML, distribution gamma par message, courbe cohérence

Le prétraitement (lemmatisation, filtrage POS, min_count) conditionne fortement
les résultats — toujours croiser avec l'inspection LDAvis avant d'interpréter.

Usage :
    python lda_topics.py --output-dir 4_data_et_viz/lexico
    python lda_topics.py --output-dir ... --k-min 3 --k-max 10 --min-count 3
    python lda_topics.py --output-dir ... --per-phase --k-min 3 --k-max 6

Mode --per-phase : LDA indépendant par phase (P1/P2/P3). Corpus plus petit
mais plus homogène → la cohérence c_v tend à monter par rapport au LDA global
(le global dialogue plafonne vers 0.45 car corpus mono-thématique). Sorties
indexées par `{src}_{phase}` (lda_topics_caption_P1.csv, lda_vis_dialogue_P2.html…).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_UTILS_DIR = _REPO / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config  # noqa: E402
from categories_semantiques import PHASE_SHORT  # noqa: E402

# CSV/HTML écrits dans 4_data_et_viz/lexico/ ; figures PNG un cran au-dessus
# (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(_REPO / "4_data_et_viz" / "lexico")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Préparation du corpus
# ---------------------------------------------------------------------------


def construire_corpus_depuis_csv(csv_path, min_count=3, min_tokens_per_doc=5,
                          phase_filter=None):
    """
    Lit un CSV de lemmes et construit le corpus gensim.

    phase_filter : None = toutes phases ; sinon valeur exacte de `phase`
    (ex. "P1_Artisanal") pour LDA restreint à une phase.

    Retourne :
      docs       : list[list[str]] — lemmes par message
      dictionary : gensim Dictionary
      bow_corpus : list[list[(id, count)]]
      doc_meta   : DataFrame (message_id, date, phase) pour chaque doc
    """
    df = pd.read_csv(csv_path, dtype={"message_id": str, "phase": str})
    df = df[df["phase"].notna() & (df["phase"] != "")]
    if phase_filter is not None:
        df = df[df["phase"] == phase_filter]

    # Grouper les lemmes par message
    grouped = df.groupby("message_id", sort=False)
    docs = []
    meta_rows = []

    for mid, group in grouped:
        lemmes = group["lemma"].tolist()
        if len(lemmes) < min_tokens_per_doc:
            continue
        docs.append(lemmes)
        meta_rows.append({
            "message_id": mid,
            "date": group["date"].iloc[0],
            "phase": group["phase"].iloc[0],
        })

    doc_meta = pd.DataFrame(meta_rows)

    # Dictionary + filtrage
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=min_count, no_above=0.7)

    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    return docs, dictionary, bow_corpus, doc_meta


# ---------------------------------------------------------------------------
# Boucle cohérence / perplexité
# ---------------------------------------------------------------------------

def chercher_k_optimal(docs, dictionary, bow_corpus, k_min, k_max,
                   passes=15, iterations=200, seed=42):
    """
    Entraîne un LDA pour chaque k et calcule cohérence c_v + perplexité.

    Retourne :
      results : list[dict] avec k, coherence, perplexity
      best_k  : k avec la meilleure cohérence c_v
    """
    results = []

    for k in range(k_min, k_max + 1):
        print(f"    k={k}...", end=" ", flush=True)

        model = models.LdaModel(
            corpus=bow_corpus,
            id2word=dictionary,
            num_topics=k,
            alpha="auto",
            eta="auto",
            passes=passes,
            iterations=iterations,
            random_state=seed,
        )

        # Cohérence c_v
        cm = CoherenceModel(
            model=model,
            texts=docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence = cm.get_coherence()

        # Perplexité
        perplexity = model.log_perplexity(bow_corpus)

        results.append({
            "k": k,
            "coherence": round(coherence, 4),
            "perplexity": round(perplexity, 4),
        })
        print(f"c_v={coherence:.4f}, perplexité={perplexity:.4f}")

    best_k = max(results, key=lambda x: x["coherence"])["k"]
    return results, best_k


# ---------------------------------------------------------------------------
# Modèle final + exports
# ---------------------------------------------------------------------------

def entrainer_modele_final(docs, dictionary, bow_corpus, k,
                      passes=30, iterations=400, seed=42):
    """Entraîne le modèle LDA final avec plus de passes."""
    model = models.LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=k,
        alpha="auto",
        eta="auto",
        passes=passes,
        iterations=iterations,
        random_state=seed,
    )
    return model


def exporter_courbe_coherence(results, src, output_dir, fig_dir):
    """Courbe cohérence c_v + perplexité en fonction de k."""
    ks = [r["k"] for r in results]
    coherences = [r["coherence"] for r in results]
    perplexities = [r["perplexity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(ks, coherences, "o-", color="#E74C3C", linewidth=2, label="Cohérence c_v")
    ax1.set_xlabel("Nombre de topics (k)", fontsize=12)
    ax1.set_ylabel("Cohérence c_v", fontsize=12, color="#E74C3C")
    ax1.tick_params(axis="y", labelcolor="#E74C3C")
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, perplexities, "s--", color="#4A90D9", linewidth=2, label="Log-perplexité")
    ax2.set_ylabel("Log-perplexité", fontsize=12, color="#4A90D9")
    ax2.tick_params(axis="y", labelcolor="#4A90D9")

    best_k = max(results, key=lambda x: x["coherence"])["k"]
    best_c = max(results, key=lambda x: x["coherence"])["coherence"]
    ax1.axvline(best_k, color="grey", linestyle=":", alpha=0.7)
    ax1.annotate(f"k={best_k}\nc_v={best_c:.3f}",
                 xy=(best_k, best_c), fontsize=10,
                 textcoords="offset points", xytext=(15, -15),
                 arrowprops=dict(arrowstyle="->", color="grey"))

    fig.suptitle(f"Choix du nombre de topics — {src}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(fig_dir, f"fig_lda_coherence_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")

    # CSV
    csv_path = os.path.join(output_dir, f"lda_coherence_{src}.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"    -> {csv_path}")


def exporter_mots_topics(model, k, src, output_dir, top_n=15):
    """Exporte les top N mots par topic en CSV."""
    rows = []
    for topic_id in range(k):
        words = model.show_topic(topic_id, topn=top_n)
        for rank, (word, weight) in enumerate(words, 1):
            rows.append({
                "topic": topic_id,
                "rank": rank,
                "word": word,
                "weight": round(weight, 4),
            })

    path = os.path.join(output_dir, f"lda_topics_{src}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    -> {path}")


def exporter_gamma(model, bow_corpus, doc_meta, src, output_dir):
    """
    Exporte la distribution gamma (topic dominant par message).
    Colorable par phase dans un graphique ultérieur.
    """
    rows = []
    for i, bow in enumerate(bow_corpus):
        topic_dist = model.get_document_topics(bow, minimum_probability=0.0)
        topic_probs = {t: p for t, p in topic_dist}
        dominant = max(topic_probs, key=topic_probs.get)

        row = {
            "message_id": doc_meta.iloc[i]["message_id"],
            "date": doc_meta.iloc[i]["date"],
            "phase": doc_meta.iloc[i]["phase"],
            "dominant_topic": dominant,
            "dominant_prob": round(topic_probs[dominant], 4),
        }
        # Toutes les probabilités par topic
        for t, p in sorted(topic_probs.items()):
            row[f"topic_{t}"] = round(p, 4)
        rows.append(row)

    path = os.path.join(output_dir, f"lda_gamma_{src}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"    -> {path}")


def exporter_ldavis(model, bow_corpus, dictionary, src, output_dir):
    """Exporte la visualisation LDAvis en HTML interactif."""
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models

        vis_data = pyLDAvis.gensim_models.prepare(model, bow_corpus, dictionary)
        path = os.path.join(output_dir, f"lda_vis_{src}.html")
        pyLDAvis.save_html(vis_data, path)
        print(f"    -> {path}")
    except Exception as e:
        print(f"    [WARN] LDAvis échoué : {e}")


def tracer_gamma_par_phase(gamma_path, src, fig_dir):
    """Figure : distribution des topics dominants par phase."""
    df = pd.read_csv(gamma_path, dtype={"message_id": str})

    phases = sorted(df["phase"].unique())
    topics = sorted(df["dominant_topic"].unique())
    n_phases = len(phases)

    fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 5), squeeze=False)
    cmap = plt.colormaps.get_cmap("tab10")

    for idx, phase in enumerate(phases):
        ax = axes[0][idx]
        sub = df[df["phase"] == phase]
        counts = sub["dominant_topic"].value_counts().sort_index()
        colors = [cmap(t % 10) for t in counts.index]
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_xlabel("Topic dominant")
        ax.set_ylabel("Nb messages")
        ax.set_title(phase, fontsize=11, fontweight="bold")
        ax.set_xticks(topics)

    fig.suptitle(f"Distribution des topics dominants par phase — {src}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(fig_dir, f"fig_lda_gamma_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# ---------------------------------------------------------------------------
# CLI et main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LDA topic modeling gensim — corpus Magyar (LEXICO §5)")
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de lemmes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--sources", nargs="*", default=["dialogue", "caption"],
                        help="Sources à traiter (défaut : dialogue caption)")
    parser.add_argument("--k-min", type=int, default=3,
                        help="K minimum (défaut : 3)")
    parser.add_argument("--k-max", type=int, default=10,
                        help="K maximum (défaut : 10)")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Fréquence min d'un lemme dans le corpus (défaut : 3)")
    parser.add_argument("--min-tokens", type=int, default=5,
                        help="Nb min de tokens par message pour l'inclure (défaut : 5)")
    parser.add_argument("--per-phase", action="store_true",
                        help="Entraîne un LDA indépendant par phase (P1/P2/P3) "
                             "en plus du LDA global")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml")
    args = parser.parse_args()

    load_config(args.config)

    # On résout fig_dir une fois et on le stocke dans args : le pipeline interne
    # (_executer_pipeline_lda) le relit via args.fig_dir.
    args.fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    print(f"LDA topic modeling (gensim)")
    print(f"  k : {args.k_min}..{args.k_max}, min_count : {args.min_count}, "
          f"min_tokens/doc : {args.min_tokens}, per_phase : {args.per_phase}\n")

    for src in args.sources:
        csv_path = os.path.join(args.output_dir, f"lemmes_{src}.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] {src} : {csv_path} introuvable")
            continue

        print(f"── {src} ──")

        # Global
        print("  [global]")
        _executer_pipeline_lda(csv_path, src, args, phase_filter=None)

        if not args.per_phase:
            continue

        # Par phase : lit les phases depuis le CSV
        df_phases = pd.read_csv(csv_path, dtype={"phase": str},
                                usecols=["phase"])
        phases = sorted(p for p in df_phases["phase"].dropna().unique()
                        if p and p in PHASE_SHORT)
        for phase_raw in phases:
            phase_short = PHASE_SHORT[phase_raw]
            label = f"{src}_{phase_short}"
            print(f"  [{phase_short} = {phase_raw}]")
            _executer_pipeline_lda(csv_path, label, args, phase_filter=phase_raw)

    print("Terminé.")


def _executer_pipeline_lda(csv_path, label, args, phase_filter=None):
    """
    Entraîne un LDA complet (corpus → K optimal → modèle final → exports)
    et préfixe toutes les sorties avec `label` (ex. "caption" ou "caption_P1").
    """
    print("    Construction du corpus...")
    docs, dictionary, bow_corpus, doc_meta = construire_corpus_depuis_csv(
        csv_path, min_count=args.min_count,
        min_tokens_per_doc=args.min_tokens,
        phase_filter=phase_filter,
    )
    print(f"      {len(docs)} documents, {len(dictionary)} termes")

    if len(docs) < 20:
        print(f"      [SKIP] trop peu de documents ({len(docs)})")
        return

    print("    Recherche du K optimal...")
    results, best_k = chercher_k_optimal(
        docs, dictionary, bow_corpus, args.k_min, args.k_max)
    print(f"    -> K optimal : {best_k}")

    exporter_courbe_coherence(results, label, args.output_dir, args.fig_dir)

    print(f"    Entraînement du modèle final (k={best_k})...")
    model = entrainer_modele_final(docs, dictionary, bow_corpus, best_k)

    print("    Exports...")
    exporter_mots_topics(model, best_k, label, args.output_dir)
    exporter_gamma(model, bow_corpus, doc_meta, label, args.output_dir)
    exporter_ldavis(model, bow_corpus, dictionary, label, args.output_dir)

    gamma_path = os.path.join(args.output_dir, f"lda_gamma_{label}.csv")
    # La figure gamma ne vaut que pour le LDA global (besoin des 3 phases)
    if phase_filter is None:
        tracer_gamma_par_phase(gamma_path, label, args.fig_dir)

    print(f"\n    Topics (k={best_k}) :")
    for t in range(best_k):
        words = model.show_topic(t, topn=8)
        w_str = ", ".join(f"{w}({p:.3f})" for w, p in words)
        print(f"      T{t}: {w_str}")
    print()


if __name__ == "__main__":
    main()
