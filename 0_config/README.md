# 0_config — Configuration centrale & utilitaires partagés

Ce module fournit la configuration commune à tout le pipeline (`config.yaml`) et la
bibliothèque partagée `utils.py` qu'importent les scripts d'enrichissement (CLI standard,
I/O JSONL, logging, idempotence, phases du corpus). Aucun traitement de données ici :
c'est la fondation dont dépendent les modules `1a_scraper/` → `3a_lexicometrie/`.

## Utilisation

```bash
# 1. Créer sa config locale à partir du template (chemins machine-dépendants)
cp 0_config/config.example.yaml 0_config/config.yaml
# 2. Éditer config.yaml : adapter les chemins paths.* à votre installation
```

`config.yaml` est gitignoré (chemins locaux). Seul `config.example.yaml` est versionné.

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

## Input / Output

- **Input** : `config.yaml` (lu par `load_config()`), et le JSONL/les fiches que manipulent
  les fonctions d'I/O quand un script les appelle.
- **Output** : aucun en propre. `utils.py` est une bibliothèque ; ce sont les scripts
  appelants qui écrivent le JSONL et les fiches via `write_jsonl()` / `mettre_a_jour_fiche()`.

## Contenu

| Fichier | Rôle |
|---|---|
| `config.example.yaml` | Template de configuration (chemins, phases, modèles, seuils). À copier en `config.yaml`. |
| `utils.py` | Bibliothèque partagée — API ci-dessous. |
| `hallucination_patterns.yaml` | Patterns regex d'hallucination Whisper, lus par `2b_transcription/qa_whisper.py`. |

### API `utils.py`

- **CLI** : `creer_parser_base()` (parser argparse avec `--input/--output/--limit/--overwrite/--ids/--start-date/--end-date/--config`), `analyser_date_arg()`
- **Config** : `load_config()`
- **Phases** : `etiquette_phase()` → `"P1"`/`"P2"`/`"P3"` selon la date (bornes dans `config.yaml`)
- **Logging** : `init_logger()` (console INFO + fichier WARNING horodaté)
- **Idempotence** : `est_traite()`, `filtrer_eligibles()` (filtres ids/dates/media_type + skip si déjà enrichi)
- **JSONL** : `read_jsonl()`, `write_jsonl()`
- **Fiches** : `chemin_fiche()`, `charger_fiche()`, `mettre_a_jour_fiche()` (merge incrémental, ne réécrase pas sauf `overwrite=True`)
- **Progression** : `SuiviProgression` (avancement + ETA), `fmt_eta()`
- **Regex** : `CYRILLIC_RE` (partagé Whisper + traduction)

## Méthodologie

Le découpage en phases (P1/P2/P3) est défini dans `config.yaml` et appliqué uniformément
par `etiquette_phase()` côté Python comme par `r_source.R` côté R — `config.yaml` fait foi
pour les bornes. Les fonctions d'enrichissement suivent la règle d'idempotence du projet :
un champ déjà présent n'est jamais réécrasé (sauf `--overwrite`).
