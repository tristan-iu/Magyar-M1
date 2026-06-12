# 2c_traduction — Traduction ukrainien → français

Traduction uk→fr des captions, dialogues et segments Whisper du corpus.
**But : consultation uniquement.** Aucune analyse quantitative n'est tirée des traductions — elles servent à naviguer dans le corpus lors de l'interprétation des résultats, pas de variable dépendante.

## Dépendances

```bash
pip install -r requirements.txt
```

## Configuration

Copier `.env.example` → `.env` (non versionné) et le compléter :

```env
# Moteur 1 — DeepL (optionnel, voir Méthodologie)
DEEPL_AUTH_KEY=votre_clé_deepl

# Moteur 2 — serveur local OpenAI-compatible (défaut si DeepL absent)
# Fonctionne avec LM Studio (:1234/v1) comme avec Ollama (:11434/v1)
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=                # laisser vide = premier modèle chargé sur le serveur
```

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

## Input / Output

**Entrée :** JSONL avec `legende`, `dialogue` et/ou SRT cyrillique sur disque.

Pour chaque message traité :

| Sortie | Description |
|--------|-------------|
| `legende_fr` dans le JSONL et la fiche | Traduction de la légende Telegram |
| `dialogue_fr` dans le JSONL et la fiche | Traduction du dialogue extrait |
| `robert_magyar_{id}_fr.srt` | Fichier SRT traduit dans `fiches/` (chemin déductible) |

Les messages sans contenu cyrillique sont silencieusement ignorés. Le chemin SRT
français n'est pas persisté en JSONL (déductible : `fiches/{canal}_{message_id}_fr.srt`).

## Méthodologie

**Objectif limité** : la traduction sert uniquement à la consultation du corpus (lecture humaine lors de l'analyse). Aucune variable statistique n'en est dérivée. Une traduction approximative est donc acceptable — le critère est la lisibilité, pas la fidélité terminologique.

**DeepL API vs LM Studio** : DeepL offre la meilleure qualité pour l'ukrainien militaire, mais l'API payante a un quota mensuel limité et un coût non nul sur 1 365 messages. LM Studio (Mistral ou équivalent en local) est le fallback gratuit : qualité légèrement inférieure sur le vocabulaire militaire spécialisé, mais suffisante pour l'usage de consultation. Auto-détection : DeepL si `DEEPL_AUTH_KEY` est présent, LM Studio sinon.

**Traduction par batch (LM Studio)** : les segments Whisper sont envoyés par paquets de 10 lignes numérotées pour limiter les appels LLM. Un chunk dont la réponse est mal parsée (mauvais compte de lignes) repasse en mode individuel segment par segment.

**Limite LM Studio — dialogues longs** : le champ `dialogue` complet est traduit en un seul appel, plafonné à 4096 tokens de sortie. Les dialogues les plus longs du corpus (~55 000 caractères, ex. #889, #1058) sont donc **tronqués silencieusement** par le moteur local. DeepL n'a pas cette limite (découpage interne). Pour traduire l'intégralité du corpus, utiliser DeepL ; le fallback LM Studio convient pour de la consultation ponctuelle.

**Idempotence** : `legende_fr` et `dialogue_fr` ne sont pas écrasés sans `--overwrite`. Le SRT est vérifié par existence de fichier.
