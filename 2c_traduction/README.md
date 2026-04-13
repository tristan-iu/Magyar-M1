# 2c_traduction — Traduction ukrainien → français

Traduction uk→fr des captions, dialogues et segments Whisper du corpus.
**But : consultation uniquement.** Aucune analyse quantitative n'est tirée des traductions — elles servent à naviguer dans le corpus lors de l'interprétation des résultats, pas de variable dépendante.

## Dépendances

```bash
pip install -r requirements.txt
```

## Configuration

Éditer `.env` (présent, non versionné) :

```env
# Moteur 1 — DeepL (optionnel, voir Méthodologie)
DEEPL_AUTH_KEY=votre_clé_deepl

# Moteur 2 — LM Studio (défaut si DeepL absent)
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=                # laisser vide = premier modèle chargé dans LM Studio
```

## Usage

```bash
# Auto-détection moteur (deepl si clé présente, sinon lmstudio)
python translate_srt.py --input /chemin/messages_whisper.jsonl

# Forcer un moteur
python translate_srt.py --input messages_whisper.jsonl --engine lmstudio

# Aperçu sans écrire
python translate_srt.py --input messages_whisper.jsonl --dry-run

# Messages spécifiques
python translate_srt.py --input messages_whisper.jsonl --ids 8 100 200

# Retraduire même si déjà fait
python translate_srt.py --input messages_whisper.jsonl --overwrite

# Fenêtre temporelle
python translate_srt.py --input messages_whisper.jsonl --start-date 2024-01-01
```

### Options CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--input` | requis | JSONL source (avec champs Whisper) |
| `--engine` | auto | `deepl` ou `lmstudio` |
| `--fiches-dir` | config | Dossier des fiches JSON |
| `--dry-run` | false | Afficher sans écrire |
| `--limit` | aucune | N messages max |
| `--ids` | tous | Message IDs spécifiques |
| `--overwrite` | false | Retraduire les messages déjà traduits |
| `--start-date` | aucune | Filtrer à partir de YYYY-MM-DD |
| `--end-date` | aucune | Filtrer jusqu'à YYYY-MM-DD |

## Input / Output

**Entrée :** JSONL avec `caption`, `dialogue` et/ou `whisper_segments` en cyrillique.

**Pas de JSONL de sortie** : les traductions vivent dans les fiches individuelles.

Pour chaque message traité :

| Sortie | Description |
|--------|-------------|
| `caption_fr` dans la fiche JSON | Traduction de la légende Telegram |
| `dialogue_fr` dans la fiche JSON | Traduction du dialogue extrait |
| `robert_magyar_{id}_fr.srt` | Fichier SRT traduit dans `fiches/` |
| `srt_fr_path` dans la fiche JSON | Chemin relatif vers le SRT traduit |

Les messages sans contenu cyrillique sont silencieusement ignorés.

## Méthodologie

**Objectif limité** : la traduction sert uniquement à la consultation du corpus (lecture humaine lors de l'analyse). Aucune variable statistique n'en est dérivée. Une traduction approximative est donc acceptable — le critère est la lisibilité, pas la fidélité terminologique.

**DeepL API vs LM Studio** : DeepL offre la meilleure qualité pour l'ukrainien militaire, mais l'API payante a un quota mensuel limité et un coût non nul sur 1 365 messages. LM Studio (Mistral ou équivalent en local) est le fallback gratuit : qualité légèrement inférieure sur le vocabulaire militaire spécialisé, mais suffisante pour l'usage de consultation. Auto-détection : DeepL si `DEEPL_AUTH_KEY` est présent, LM Studio sinon.

**Traduction par batch (LM Studio)** : les segments Whisper sont envoyés par paquets de 10 lignes numérotées pour limiter les appels LLM. Un chunk dont la réponse est mal parsée (mauvais compte de lignes) repasse en mode individuel segment par segment.

**Idempotence** : `caption_fr` et `dialogue_fr` ne sont pas écrasés sans `--overwrite`. Le SRT est vérifié par existence de fichier.
