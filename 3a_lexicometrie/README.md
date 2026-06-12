# 3a_lexicometrie

Lemmatisation ukrainienne (spaCy `uk_core_news_trf`), TF-IDF par phase, et analyses
exploratoires (LDA, AFC, CAH, co-occurrences, KWIC) du corpus Magyar. Traite à la
fois les légendes des publications et le dialogue transcrit par Whisper (étape 2b).

## Dépendances

```bash
pip install -r requirements.txt
python -m spacy download uk_core_news_trf
```

## Scripts

| Script | Rôle |
|--------|------|
| `lexicometrie.py` | Lemmatisation + TF-IDF par phase + volcano + stats lexicométriques |
| `evaluate_lemmas.py` | Échantillonnage stratifié et évaluation manuelle de la qualité spaCy |
| `lda_topics.py` | LDA gensim — topic modeling par phase |
| `afc.py` | Analyse Factorielle des Correspondances (lemmes × phases) |
| `cah.py` | Classification Ascendante Hiérarchique sur les profils lexicaux |
| `specificites.py` | Spécificités de Lafon (modèle hypergéométrique) — formes caractéristiques par phase |
| `cooccurrences.py` | Réseaux de cooccurrences par phase (PMI + centralités) |
| `collocations.py` | Bigrammes significatifs (G² de Dunning + PMI) |
| `kwic.py` | Concordancier (Key Word In Context) |
| `comparaison_sources.py` | Recouvrement lexical caption ↔ dialogue (Jaccard/Dice) |
| `ner_lieux.py` | NER toponymes (LOC) par phase — exploratoire |
| `plot_lexico.py` | Figures principales (volcano, barplots, TTR…) |
| `plot_tfidf_vertical.py` | Tableau TF-IDF avec catégorisation sémantique (militaire/finance/associé) |
| `36_tfidf_tableau.py` | Tableau TF-IDF 3 phases (P1/P2/P3) avec traductions uk→fr |
| `36b_tfidf_comparatif_p1p2.py` | Tableau comparatif TF-IDF P1/P2 par source |
| `36c_slope_tfidf_p1p2.py` | Slope chart TF-IDF P1→P2 |
| `36d_slope_combined_h.py` | Slope chart combiné horizontal |
| `36e_tfidf_barchart_p1.py` | Barplot TF-IDF P1 |

## Utilisation

```bash
# Étape obligatoire — lemmatisation + TF-IDF (produit les CSV dont dépend tout le reste)
python lexicometrie.py --input messages_clean.jsonl
#   --phases "2022-09-01:2023-12-31:Artisanal" …  phases custom "start:end:label" (défaut : config.yaml)
#   --no-phases               corpus traité comme un seul bloc
#   --output-dir DIR          dossier des CSV (défaut : 4_data_et_viz/lexico/, racine repo)
#   --min-confidence 0.5      seuil dialogue_confiance pour inclure un dialogue
#   --stopwords FICHIER       stopwords additionnels (un mot/ligne, ex. stopwords_uk_magyar.txt)
#   --top-n 20                nombre de termes TF-IDF par phase

# Analyses dérivées (lisent les CSV produits par lexicometrie.py)
python lda_topics.py
python afc.py
python cah.py
python specificites.py
python cooccurrences.py
python plot_lexico.py
python plot_tfidf_vertical.py
```

## Phases (config.yaml)

| Phase | Début | Fin | Label |
|---|---|---|---|
| P1 | 2022-09-01 | 2023-12-31 | Artisanal |
| P2 | 2024-01-01 | 2024-09-30 | Semi-pro |
| P3 | 2024-10-01 | 2025-09-30 | Institutionnel |

## Stopwords

`stopwords_uk_magyar.txt` — surcouche aux stopwords spaCy : termes sur-représentés
sans valeur thématique pour ce corpus (jargon Telegram, formules récurrentes,
abréviations militaires hors-vocabulaire spaCy).

## Sorties

Convention de sortie (alignée sur les autres modules, gitignorée — régénérable) :

- **Données brutes (CSV, HTML LDAvis)** → `4_data_et_viz/lexico/`
- **Figures (PNG)** → `4_data_et_viz/` (racine, avec les figures R du mémoire)

Tous les scripts prennent ces chemins par défaut. Surcharge possible :
`--output-dir` (dossier des CSV) et `--fig-dir` (dossier des figures ;
défaut = parent de `--output-dir`).

### CSV principaux (`lexicometrie.py`)
- `lemmes_{caption,dialogue,combined}.csv` — message_id, date, phase, token, lemma, pos
- `tfidf_{caption,dialogue,combined}.csv` — phase, rank, lemma, score TF-IDF
- `stats_phases.csv` — tokens totaux/uniques, TTR, hapax par phase et objet
- `volcano_{caption,dialogue,combined}.csv` — log2fc, p-value, comparaison par paire de phases
- `specificites_{caption,dialogue}.csv` (`specificites.py`) — phase, lemma, f, F, t, T, attendu, specificite (−log₁₀ p signé), signe

### Figures (`plot_lexico.py`, `plot_tfidf_vertical.py`, `specificites.py`) — dans `4_data_et_viz/`
- `fig_tfidf_*.png` — barplot horizontal, top N TF-IDF, un panneau par phase
- `fig_barplot_phases_*.png` — barplot côte-à-côte des top lemmes
- `fig_volcano_*.png` — volcano plot (log2 fold-change vs -log10 p-value chi²)
- `fig_stats.png` — stats lexicométriques (tokens, TTR, hapax) par phase
- `fig_tfidf_vertical.png` — tableau TF-IDF avec catégories militaire/finance/associé
- `fig_specificites_{caption,dialogue}.png` — spécificités de Lafon, top N sur-emplois par phase (3 panneaux)

## Évaluation de la lemmatisation

`evaluate_lemmas.py` produit un échantillon stratifié (100 tokens/phase) à annoter
manuellement, puis calcule l'accuracy spaCy (`is_correct` + `lemma_corrected`).

```bash
python evaluate_lemmas.py sample --input output/lemmes_combined.csv --output eval.csv
# ... annotation manuelle ...
python evaluate_lemmas.py evaluate --input eval_annotated.csv
```

Les échantillons annotés utilisés dans le mémoire (`eval_sample_*.csv`, caption +
dialogue, ground truth manuelle) ne sont **pas publiés** dans le dépôt — ils
contiennent du texte du corpus (gitignorés, disponibles sur demande).

## Champs JSONL lus

- `date` — date du message (ISO 8601)
- `legende` — texte de la légende
- `dialogue` — transcription Whisper
- `dialogue_confiance` — confiance de la transcription (composite QA)
- `message_id` — identifiant du message
