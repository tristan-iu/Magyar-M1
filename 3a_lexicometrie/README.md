# Lexicométrie

Lemmatisation ukrainienne (spaCy `uk_core_news_trf`), TF-IDF par phase et analyses exploratoires (LDA, AFC, CAH, spécificités, cooccurrences, KWIC) du corpus. Les légendes des publications et les dialogues transcrits sont traités comme deux corpus distincts tout au long du module.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download uk_core_news_trf
```

## Utilisation

`lexicometrie.py` est l'étape obligatoire : il lemmatise le corpus et produit les CSV (lemmes, TF-IDF, stats) dont dépendent toutes les analyses dérivées.

```bash
# Étape obligatoire : lemmatisation + TF-IDF
python lexicometrie.py --input messages_clean.jsonl
#   --no-phases               corpus traité comme un seul bloc
#   --output-dir DIR          dossier des CSV (défaut : 4_data_et_viz/lexico/, racine repo)
#   --min-confidence 0.5      seuil dialogue_confiance pour inclure un dialogue
#   --stopwords FICHIER       stopwords additionnels (un mot/ligne, ex. stopwords_uk_magyar.txt)
#   --top-n 20                nombre de termes TF-IDF par phase

# Analyses dérivées (lisent les CSV produits ci-dessus)
python lda_topics.py        # topic modeling LDA (gensim)
python afc.py               # analyse factorielle des correspondances
python cah.py               # classification ascendante hiérarchique
python specificites.py      # spécificités de Lafon (hypergéométrique)
python cooccurrences.py     # réseaux de cooccurrences (PMI), --per-phase pour un export par phase
python collocations.py      # bigrammes significatifs (G² de Dunning)
python kwic.py              # concordancier
python comparaison_sources.py   # recouvrement lexical légendes / dialogues
python ner_lieux.py --input messages_clean.jsonl   # toponymes (exploratoire)

# Figures
python plot_lexico.py
python plot_tfidf_vertical.py
```

Les scripts `36_*.py` produisent les tableaux et slope charts TF-IDF du mémoire à partir des mêmes CSV.

## Output

Tout est écrit dans `4_data_et_viz/` à la racine du dépôt (gitignoré, régénérable) : les données brutes (CSV, HTML LDAvis) dans `4_data_et_viz/lexico/`, les figures PNG à la racine de `4_data_et_viz/` avec les figures R du mémoire. Chaque script accepte `--output-dir` pour surcharger.

`lexicometrie.py` produit `lemmes_{caption,dialogue,combined}.csv` (un token lemmatisé par ligne, avec date, phase et POS), `tfidf_{caption,dialogue,combined}.csv` (termes les mieux classés par phase), `stats_phases.csv` (tokens, types, TTR, hapax par phase et par source) et `volcano_{caption,dialogue,combined}.csv` (log2 fold-change et p-value par paire de phases). Les analyses dérivées écrivent leurs propres CSV et figures préfixés par leur nom.

## Méthodologie

**Deux sources séparées :** les légendes (écrit) et les dialogues (oral transcrit) ne sont jamais fusionnés par défaut. Leurs volumes diffèrent d'un facteur cinq et leurs registres ne sont pas comparables ; chaque analyse sort en version caption, dialogue et, à titre indicatif, combinée.

**Phases :** le découpage temporel vient de `0_config/config.yaml` et n'est jamais codé en dur. L'option `--phases "start:end:label"` permet un découpage custom pour les tests de robustesse.

**Stopwords :** `stopwords_uk_magyar.txt` complète les stopwords spaCy avec les termes sur-représentés sans valeur thématique pour ce corpus (jargon Telegram, formules récurrentes, abréviations militaires hors vocabulaire spaCy).

**Filtre de confiance :** les dialogues dont `dialogue_confiance` est sous 0.5 sont exclus par défaut, pour ne pas faire entrer les transcriptions suspectes (hallucinations Whisper) dans les comptages lexicaux.

**Évaluation de la lemmatisation :** `evaluate_lemmas.py` tire un échantillon stratifié (100 tokens par phase) à annoter manuellement, puis calcule l'accuracy de spaCy sur cette annotation. Les échantillons annotés utilisés dans le mémoire ne sont pas publiés dans le dépôt, ils contiennent du texte du corpus (disponibles sur demande).
