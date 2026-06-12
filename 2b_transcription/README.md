# Transcription Whisper

Transcription audio du corpus via OpenAI Whisper (large-v3), avec pré-filtrage de la parole par Silero-VAD. Produit le texte transcrit (`dialogue`) dans le JSONL, un fichier de sous-titres SRT par vidéo, et des métriques de présence de parole. Le module comprend aussi une segmentation audio parole/musique/silence optionnelle (`audio_segmentation.py`).

## Installation

```bash
pip install -r requirements.txt
```

Requiert ffmpeg installé système et un GPU CUDA. Silero-VAD est chargé via `torch.hub` (clone du dépôt GitHub au premier lancement, mis en cache ensuite), il n'y a pas de paquet pip dédié.

`audio_segmentation.py` demande un venv séparé : inaSpeechSegmenter tourne sous TensorFlow, qui entre en conflit cuDNN avec le venv principal PyTorch. Instructions de création en tête du script.

## Utilisation

```bash
# Corpus complet (input = sortie de l'étape ffprobe).
# Relance après interruption : --input = --output (skip les déjà traités)
python whisper_batch.py --input messages_ffprobe.jsonl --output messages_whisper.jsonl
#   --limit 5      test sur les 5 premiers messages
#   --overwrite    retranscrit même si dialogue existe déjà
#   --model / --language / --vad-threshold   surcharges ponctuelles (défaut : config.yaml)

# Segmentation parole / musique / silence (optionnel, venv séparé)
venv_ina/bin/python audio_segmentation.py --input messages_clean.jsonl \
  --output messages_ina.jsonl --media-dir /chemin/processed
#   --seuil-musique 0.5    seuil du flag alerte_musique_dominante
```

Seuls les messages `media_type="video"` avec `audio_present=true` sont transcrits. Le prompt de conditionnement et les seuils VAD sont lus depuis `0_config/config.yaml`.

## Output

`whisper_batch.py` réécrit le JSONL enrichi des champs ci-dessous et écrit un fichier de sous-titres par vidéo transcrite, au chemin déductible `fiches/{canal}_{message_id}.srt`. Les segments Whisper bruts ne sont pas persistés dans le JSONL (16 MB économisés), ils restent disponibles dans les SRT.

### Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `parole_present` | bool | Présence de parole détectée |
| `parole_duree` | float | Durée de parole en secondes |
| `parole_ratio` | float | Ratio parole / durée totale |
| `dialogue` | string \| null | Texte transcrit. `""` si la transcription est vide ou rejetée par le filtre qualité, `null` si le message n'a jamais été transcrit (pas d'audio) |
| `dialogue_confiance` | float | Score QA composite [0, 1] (1 = très fiable), voir Méthodologie |
| `alerte_no_dialogue` | bool | Pas de dialogue exploitable |
| `alerte_hallucination_phrase` | bool | Phrase d'hallucination connue détectée (outros YouTube...) |
| `alerte_repeated_ngram` | bool | Trigramme dominant > 25 % du texte |
| `alerte_tokens_per_sec_anomaly` | bool | Débit hors `[0.5, 5.0]` tokens/seconde |
| `alerte_compression_high` | bool | `len(tokens)/len(unique) > 2.4` |
| `alerte_low_conf` | bool | Proportion `logprob < -1` trop haute |
| `alerte_high_no_speech` | bool | Proportion `no_speech_prob > 0.6` trop haute |

`whisper_batch.py` produit les quatre premiers champs ; `dialogue_confiance` et les 7 `alerte_*` proviennent d'une passe de QA interne (voir Méthodologie).

### Champs de la segmentation audio

`audio_segmentation.py` écrit dans un JSONL séparé (`messages_ina.jsonl`), hors schéma canonique.

| Champ | Type | Description |
|-------|------|-------------|
| `audio_parole_pure_ratio` | float | Ratio parole seule / durée totale |
| `audio_musique_ratio` | float | Ratio musique / durée totale |
| `audio_parole_sur_musique_ratio` | float | Toujours `0.0` : la segmentation INA est séquentielle (labels exclusifs), le chevauchement n'est pas mesurable. Conservé pour compatibilité |
| `audio_silence_ratio` | float | Ratio silence / durée totale |
| `audio_dominant` | string | `parole`, `musique`, `silence` ou `mixte` |
| `alerte_musique_dominante` | bool | `audio_musique_ratio` au-dessus du seuil (défaut 0.5) |
| `audio_segmentation_modele` | string | Identifiant du modèle |

## Méthodologie

**Deux passes :** la détection d'activité vocale (Silero-VAD) écarte les vidéos sans parole avant de solliciter Whisper, ce qui évite les transcriptions parasites sur des images de drone sans commentaire. La transcription est ensuite forcée en ukrainien, avec un prompt de conditionnement incluant le jargon militaire du corpus (FPV, аеророзвідка, 414 бригада...) pour réduire les erreurs de vocabulaire.

**Filtre qualité :** rejet de la transcription si la confiance globale est inférieure à 0.35 ou si le ratio non cyrillique dépasse 40 %. Ce seuil sacrifie quelques vidéos mixtes (ukrainien et anglais) au profit de la qualité du corpus principal.

**QA des transcriptions :** `dialogue_confiance` et les `alerte_*` ont été produits par une passe de QA interne (script non publié) combinant la médiane des `avg_logprob` Whisper, la proportion de segments basse confiance, la détection de phrases d'hallucination connues (« Дякую за перегляд », outros YouTube...), le `compression_ratio` et le débit tokens/seconde. Ces scores ne sont pas recalculables depuis le seul JSONL : le signal `avg_logprob`/`no_speech_prob` provient des segments Whisper, qui ne sont plus persistés. Les recalculer supposerait de re-transcrire tout le corpus (plusieurs heures GPU).

**Limites :** la qualité de transcription n'a pas été validée contre une vérité-terrain (pas de WER/CER ; le scoring ci-dessus est une QA intrinsèque, pas une mesure d'erreur réelle). Whisper ne distingue pas non plus dialogue parlé et musique chantée : les compilations à forte musique de fond produisent des `dialogue` corrompus ou hallucinatoires, que `alerte_musique_dominante` permet de flagger ; la séparation voix/musique en amont (demucs) n'a pas été mise en œuvre. Le bruit ambiant est en revanche peu problématique sur ce corpus : les drones FPV n'ont pas de micro embarqué, la voix off est enregistrée séparément des images.

**Segmentation INA :** le CNN d'inaSpeechSegmenter est entraîné sur de la radio française (2018), sa généralisation à l'audio militaire ukrainien n'est pas garantie : un sanity-check humain sur quelques vidéos est recommandé avant d'exploiter les résultats. Licence CC-BY-NC (usage non commercial).
