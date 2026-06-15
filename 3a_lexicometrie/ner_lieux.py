#!/usr/bin/env python3
"""
ner_lieux.py — Extraction NER des toponymes (LOC) par phase.

uk_core_news_trf embarque un NER performant sur les toponymes ukrainiens
standards (Бахмут, Кринки, Херсон, Купянськ, Курськ…) sans fine-tuning.
On s'en sert ici en mode exploratoire pour cartographier les théâtres
d'opération évoqués par Magyar à travers les trois phases.

**Scope volontairement restreint à LOC** : PER (commandants, visages) et
ORG (brigades type « 414 ОБр », « СБС ») demanderaient une annotation
manuelle conséquente pour être exploitables au M1 — reporté à un
éventuel M2.

Pipeline :
  1. Lit le JSONL enrichi (caption + dialogue bruts, texte non-lemmatisé)
  2. Exécute spaCy uk_core_news_trf avec NER activé
  3. Filtre entités LOC, normalise (lemma du head token, casse) pour
     fusionner « Бахмут », « Бахмута », « Бахмуті »
  4. Agrège par phase et par source
  5. Exports : toponymes_{caption,dialogue}.csv + figure heatmap top-15

Usage :
    python ner_lieux.py --input <chemin>/messages_clean.jsonl
    python ner_lieux.py --input ... --top-n 20 --min-count 3
    python ner_lieux.py --input ... --start-date 2024-01-01 --limit 50

Validation : à cross-vérifier via kwic.py --summary sur les top 20 pour
estimer la précision. Toute précision < 70 % doit être documentée comme
limite dans le mémoire, sans exploiter les chiffres en valeur.
"""

import os
import re
import sys
import time
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    load_config, read_jsonl, etiquette_phase,
    creer_parser_base, filtrer_eligibles, PHASE_SHORT,
)

# CSV de toponymes écrits dans 4_data_et_viz/lexico/ ; heatmaps PNG un cran
# au-dessus (4_data_et_viz/) pour rejoindre les autres figures du mémoire.
DATA_DIR_DEFAUT = str(Path(__file__).resolve().parents[1] / "4_data_et_viz" / "lexico")

import matplotlib.pyplot as plt
import pandas as pd
import spacy


def normaliser_loc(ent):
    """
    Normalise un span LOC en un toponyme-clef :
      - lemma du head token si disponible (gère les déclinaisons)
      - Title Case (évite la duplication минор/MAJOR)
      - trim de la ponctuation de bord
    """
    head = ent.root
    base = head.lemma_ if head.lemma_ and head.lemma_ != "-PRON-" else head.text
    base = re.sub(r"^[^\w]+|[^\w]+$", "", base)
    return base.strip().lower()


def extraire_locs(nlp, messages, field, min_confidence=0.0):
    """
    Retourne pour chaque message une liste de toponymes (lemmatisés).
    """
    texts = []
    meta = []
    for m in messages:
        if field == "dialogue":
            if (m.get("dialogue_confiance") or 1.0) < min_confidence:
                continue
            text = m.get("dialogue") or ""
        else:
            # "legende" est le nom canonique post-migration ; "caption" reste
            # accepté pour compatibilité CLI mais mappe sur le même champ.
            field_key = "legende" if field in ("caption", "legende") else field
            text = m.get(field_key) or ""
        text = text.strip()
        if not text:
            continue
        date = m.get("date", "")
        phase = etiquette_phase(date) if date else None
        if phase is None:
            continue
        texts.append(text)
        meta.append({"message_id": m.get("message_id"),
                     "date": date, "phase": phase})

    print(f"  {len(texts)} messages à parser ({field})")
    rows = []
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        for ent in doc.ents:
            if ent.label_ != "LOC":
                continue
            lemma = normaliser_loc(ent)
            if len(lemma) < 2:
                continue
            rows.append({
                **meta[i],
                "surface": ent.text,
                "lemma": lemma,
            })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(texts) - i - 1)
            print(f"    {i+1}/{len(texts)}  ({elapsed:.0f}s, eta {eta:.0f}s)")
    return pd.DataFrame(rows)


def agreger(df, min_count):
    """Table lemma × phase × count, avec total."""
    agg = (df.groupby(["lemma", "phase"]).size()
             .unstack(fill_value=0)
             .rename(columns=PHASE_SHORT))
    for col in ("P1", "P2", "P3"):
        if col not in agg.columns:
            agg[col] = 0
    agg = agg[["P1", "P2", "P3"]]
    agg["total"] = agg.sum(axis=1)
    agg = agg[agg["total"] >= min_count]
    agg = agg.sort_values("total", ascending=False)
    return agg


def tracer_heatmap(agg, src, fig_dir, top_n=15):
    if agg.empty:
        return
    top = agg.head(top_n)
    mat = top[["P1", "P2", "P3"]].to_numpy(dtype=float)
    # Normaliser ligne par ligne pour visualiser le pattern temporel
    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    mat_norm = mat / row_max

    fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.35)))
    im = ax.imshow(mat_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["P1", "P2", "P3"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    for i in range(len(top)):
        for j in range(3):
            ax.text(j, i, f"{int(mat[i,j])}",
                    ha="center", va="center", fontsize=9,
                    color="black" if mat_norm[i, j] < 0.6 else "white")
    ax.set_title(f"Toponymes (LOC) top-{top_n} — {src}\n"
                 "(couleur = normalisée par lemme, nombre = count brut)",
                 fontsize=11)
    plt.colorbar(im, ax=ax, label="part de la ligne")
    fig.tight_layout()

    path = os.path.join(fig_dir, f"fig_toponymes_{src}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


def main():
    parser = creer_parser_base(
        "NER lieux — extraction de toponymes par phase (spaCy uk_core_news_trf)",
        has_output=False,
    )
    parser.add_argument("--output-dir", default=DATA_DIR_DEFAUT,
                        help="Dossier des CSVs de toponymes (défaut : 4_data_et_viz/lexico)")
    parser.add_argument("--fig-dir", default=None,
                        help="Dossier des figures (défaut : parent de --output-dir)")
    parser.add_argument("--fields", nargs="*", default=["caption", "dialogue"],
                        help="Champs texte à traiter")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Fréquence min d'un toponyme (défaut : 3)")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Top N toponymes à afficher dans la heatmap (défaut : 15)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Seuil de confiance Whisper pour dialogue (défaut : 0.5)")
    args = parser.parse_args()

    load_config(args.config)

    if not os.path.exists(args.input):
        sys.exit(f"[ERREUR] {args.input} introuvable")
    fig_dir = args.fig_dir or str(Path(args.output_dir).parent)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("Chargement spaCy uk_core_news_trf (NER activé)...")
    t0 = time.time()
    nlp = spacy.load("uk_core_news_trf")
    print(f"  Modèle chargé en {time.time() - t0:.1f}s")

    messages = read_jsonl(args.input)
    print(f"  {len(messages)} messages dans {args.input}")

    # Filtrage optionnel (--ids, --start-date, --end-date, --limit)
    if args.ids or args.start_date or args.end_date or args.limit:
        filtre_ids = set(args.ids) if args.ids else None
        indices = filtrer_eligibles(
            messages, filtre_ids=filtre_ids,
            start_date=args.start_date, end_date=args.end_date,
            limit=args.limit,
        )
        messages = [messages[i] for i in indices]
        print(f"  {len(messages)} messages après filtrage")
    print()

    for field in args.fields:
        print(f"── {field} ──")
        df = extraire_locs(nlp, messages, field, args.min_confidence)
        if df.empty:
            print(f"  [vide] aucun LOC détecté pour {field}")
            continue

        raw_csv = os.path.join(args.output_dir, f"toponymes_{field}_raw.csv")
        df.to_csv(raw_csv, index=False)
        print(f"  -> {raw_csv} ({len(df)} entités)")

        agg = agreger(df, args.min_count)
        agg_csv = os.path.join(args.output_dir, f"toponymes_{field}.csv")
        agg.to_csv(agg_csv)
        print(f"  -> {agg_csv} ({len(agg)} toponymes ≥ {args.min_count})")

        tracer_heatmap(agg, field, fig_dir, top_n=args.top_n)

        print(f"\n  Top {min(10, len(agg))} — {field} :")
        for lemma, row in agg.head(10).iterrows():
            print(f"    {lemma:<25s} "
                  f"P1={int(row['P1']):<4d} "
                  f"P2={int(row['P2']):<4d} "
                  f"P3={int(row['P3']):<4d} "
                  f"tot={int(row['total'])}")
        print()

    print("Terminé.")


if __name__ == "__main__":
    main()
