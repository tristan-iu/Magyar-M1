# Configuration centrale et utilitaires partagés

Ce module fournit la configuration commune à tout le pipeline (`config.yaml`) et la bibliothèque partagée `utils.py` qu'importent les scripts d'enrichissement (CLI standard, I/O JSONL, logging, idempotence, phases du corpus). Aucun traitement de données ici : c'est la fondation dont dépendent les modules `1a_scraper/` à `3c_couleurs/`.

## Installation

```bash
cp 0_config/config.example.yaml 0_config/config.yaml
```

Puis adapter les chemins locaux (`paths.*`). `config.yaml` est gitignoré (chemins machine-dépendants), seul le template est versionné ; il documente l'ensemble des sections : chemins, phases du corpus, modèles et seuils.

## Utilisation

Dans un script du pipeline :

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "0_config"))
from utils import (
    creer_parser_base, load_config, init_logger,
    read_jsonl, write_jsonl, filtrer_eligibles,
    etiquette_phase, mettre_a_jour_fiche, SuiviProgression,
)
```

`creer_parser_base()` fournit le parser argparse commun à tous les scripts (`--input`, `--output`, `--limit`, `--overwrite`, `--ids`, `--start-date`, `--end-date`, `--config`), `load_config()` lit `config.yaml`, et `init_logger()` configure la sortie console (INFO) plus un fichier d'erreurs horodaté (WARNING). L'idempotence passe par `est_traite()` et `filtrer_eligibles()`, qui filtrent par ids, dates et type de média et skippent les messages déjà enrichis. `read_jsonl()` et `write_jsonl()` gèrent le corpus, `chemin_fiche()`, `charger_fiche()` et `mettre_a_jour_fiche()` les fiches JSON individuelles (merge incrémental, aucun champ existant n'est réécrasé sans `overwrite`). `SuiviProgression` affiche l'avancement et l'ETA, et `CYRILLIC_RE` est la regex cyrillique partagée entre la transcription et la traduction.

## Méthodologie

Le découpage du corpus en phases est défini une seule fois dans `config.yaml` et appliqué uniformément par `etiquette_phase()` côté Python comme par `r_source.R` côté R : `config.yaml` fait foi pour les bornes, aucune date n'est codée en dur dans les scripts. Les fonctions d'écriture suivent la règle d'idempotence du projet : l'enrichissement ajoute des champs, un champ déjà présent n'est jamais réécrasé sans `--overwrite`.
