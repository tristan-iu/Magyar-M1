# Traduction de l'ukrainien au français

Traduction des légendes, dialogues et sous-titres du corpus, à des fins de consultation uniquement : aucune analyse quantitative n'est tirée des traductions, elles servent à naviguer dans le corpus pendant l'interprétation des résultats. Deux moteurs interchangeables : l'API DeepL et un serveur local compatible OpenAI (LM Studio, Ollama).

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

Compléter le `.env` (non versionné) : une clé `DEEPL_AUTH_KEY` pour le moteur DeepL, ou les variables `LMSTUDIO_*` pour pointer vers un serveur local. Les valeurs par défaut conviennent à LM Studio sur `127.0.0.1:1234`, Ollama s'utilise en changeant `LMSTUDIO_BASE_URL` vers le port 11434.

## Utilisation

```bash
# Moteur auto-détecté : deepl si DEEPL_AUTH_KEY présent, sinon lmstudio
python translate_srt.py --input messages_whisper.jsonl --output messages_whisper.jsonl
#   --engine deepl|lmstudio    force un moteur
#   --dry-run                  aperçu sans écrire
#   --limit N                  N messages max
#   --ids 8 100 200            messages spécifiques
#   --overwrite                retraduit même si déjà fait
#   --start-date / --end-date  fenêtre temporelle (YYYY-MM-DD)
#   --fiches-dir DIR           dossier des fiches JSON (défaut : config.yaml)
```

Les messages sans contenu cyrillique sont ignorés silencieusement. L'idempotence est par champ : `legende_fr` et `dialogue_fr` ne sont pas réécrasés sans `--overwrite`, le SRT traduit est vérifié par existence du fichier.

## Output

Pour chaque message traité, le script écrit `legende_fr` (traduction de la légende Telegram) et `dialogue_fr` (traduction du dialogue transcrit) dans le JSONL et dans la fiche individuelle, et produit un fichier de sous-titres traduit au chemin déductible `fiches/{canal}_{message_id}_fr.srt`. Aucun chemin de SRT n'est persisté dans le JSONL.

## Méthodologie

**Objectif limité :** la traduction sert uniquement à la lecture humaine du corpus. Aucune variable statistique n'en est dérivée, une traduction approximative est donc acceptable : le critère est la lisibilité, pas la fidélité terminologique.

**DeepL ou serveur local :** DeepL offre la meilleure qualité pour l'ukrainien militaire, mais l'API payante a un quota mensuel et un coût non nul sur 1 365 messages. Le moteur local (Mistral ou équivalent via LM Studio) est le fallback gratuit, de qualité légèrement inférieure sur le vocabulaire militaire spécialisé mais suffisante pour la consultation.

**Traduction par lots (moteur local) :** les segments de sous-titres sont envoyés par paquets de 10 lignes numérotées pour limiter les appels LLM. Un paquet dont la réponse est mal parsée (mauvais compte de lignes) repasse automatiquement en mode individuel, segment par segment.

**Limite du moteur local sur les dialogues longs :** le champ `dialogue` complet est traduit en un seul appel, plafonné à 4 096 tokens de sortie. Les dialogues les plus longs du corpus (environ 55 000 caractères) sont donc tronqués silencieusement par le moteur local. DeepL n'a pas cette limite (découpage interne) : pour traduire l'intégralité du corpus, utiliser DeepL, le fallback local convenant à la consultation ponctuelle.
