# Analyses statistiques et figures (R)

Scripts d'analyse statistique du corpus enrichi et de production des figures du mémoire. Un script correspond à une analyse et à une ou plusieurs figures PNG. Toutes les figures partagent un thème ggplot commun et des bornes de phases centralisées.

Le module repose sur un socle commun, `r_source.R` (chargement du JSONL, nettoyage, phases, thème, `save_plot()`), chargé par chaque script via un chemin relatif au dépôt : aucun `setwd()` requis. `clean.R` produit un corpus local allégé (`messages_stripped.jsonl`) qui accélère les itérations. Les analyses du mémoire vivent dans `scripts_r/`, numérotées par ordre de création.

## Installation

```r
install.packages(c("jsonlite", "dplyr", "purrr", "lubridate", "tidyr", "ggplot2",
                   "scales", "forcats", "stringr", "stringi", "changepoint", "bcp",
                   "corrplot", "igraph", "visNetwork"))
```

## Utilisation

```bash
# 1. Adapter les chemins en tête de r_source.R (chemin_jsonl) et clean.R (INFILE/OUTFILE)
# 2. Produire le corpus local allégé (optionnel)
Rscript 3b_stats_R/clean.R
# 3. Lancer une analyse
Rscript 3b_stats_R/scripts_r/02_duree_video.R
```

Chaque script porte en tête son nom, la figure produite et sa commande de lancement.

## Output

Le module lit `messages_clean.jsonl` (corpus enrichi canonique, non inclus dans le dépôt, disponible sur demande), et pour certains scripts les CSV produits par `3a_lexicometrie/` ou `2d_vision/`. Il produit les figures dans `4_data_et_viz/`.

## Méthodologie

Les analyses descriptives (médianes, proportions par phase) sont systématiquement doublées de tests non paramétriques (Kruskal-Wallis, chi²) et d'une détection de ruptures indépendante de la périodisation manuelle (PELT via `changepoint`, Bayesian change point via `bcp`) : les bornes de phases sont ainsi testées, pas seulement posées. Les bornes vivent dans `r_source.R` (objet `bornes`), aucune date n'est codée en dur dans les scripts. Les analyses lexicales consomment les sorties de `3a_lexicometrie/` sans recalculer.
