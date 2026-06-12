# 2b_transcription — Transcription Whisper

Transcription audio via OpenAI Whisper (large-v3, GPU CUDA). Produit un texte transcrit (`dialogue`) dans le jsonl, des sous-titres SRT, et des métadonnées de confiance pour chaque message avec audio.

## Limitations méthodologiques

La qualité de transcription n'a pas été validée formellement contre une
vérité-terrain : les résultats sur `dialogue` portent un astérisque
méthodologique. Les limites connues :

- **Pas de ground truth WER/CER.** Une validation rigoureuse demanderait
  30-50 segments corrigés à la main par un·e locuteur·trice natif·ve
  (stratifiés P1/P2/P3 et par niveau de confiance). Le scoring de
  `qa_whisper.py` est une QA *intrinsèque* (confiance estimée sans
  re-transcription), pas une mesure d'erreur réelle.
- **Musique de fond.** Whisper ne distingue pas dialogue parlé et musique
  chantée. Les compilations P3 (forte musique de fond) produisent des
  `dialogue` souvent corrompus ou hallucinatoires. `audio_segmentation.py`
  (voir plus bas) flag ces messages via `alerte_musique_dominante`, mais
  la séparation voix/musique en amont (`demucs`) n'a pas été mise en œuvre.
  Le bruit ambiant est en revanche peu problématique sur ce corpus : les
  drones FPV n'ont pas de micro embarqué, la voix off est enregistrée
  séparément des images.
- **VAD.** La détection d'activité vocale repose sur Silero-VAD ; `pyannote.audio`
  donnerait une borne parole/silence plus précise sur audio bruyant.
- **Pas de cross-model agreement.** Re-transcrire un échantillon stratifié
  avec un second système (ex. `wav2vec2-xls-r-uk`) fournirait une borne
  inférieure de qualité par WER inter-systèmes. Non réalisé.

## Dépendances

```bash
pip install -r requirements.txt
```

Requiert ffmpeg installé système et un GPU CUDA.

## Utilisation

```bash
# Corpus complet (input = sortie de l'étape E1 ffprobe).
# Relance après interruption : --input = --output (skip les déjà traités)
python whisper_batch.py --input messages_ffprobe.jsonl --output messages_whisper.jsonl
#   --limit 5      test sur les 5 premiers messages
#   --overwrite    retranscrit même si dialogue existe déjà
#   --model / --language / --vad-threshold   surcharges ponctuelles (défaut : config.yaml)
```

## Input / Output

**Input** : un JSONL enrichi par E1 (`2a_metadonnees`) — chaque message vidéo
doit porter `media_chemin`, `duree` et `audio_present` (seuls les messages
`media_type="video"` avec `audio_present=true` sont transcrits). Le prompt de
conditionnement et les seuils VAD sont lus depuis `0_config/config.yaml`
(section `models.whisper` / `models.vad`).

**Output** :
- le même JSONL enrichi des champs ci-dessous (écriture idempotente : un
  message déjà pourvu de `dialogue` est skippé sauf `--overwrite`) ;
- un fichier sous-titres `fiches/{canal}_{message_id}.srt` à côté de chaque
  média transcrit (les segments bruts n'y sont conservés que là, pas dans le
  JSONL).

## Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `parole_present` | bool | Présence de parole détectée |
| `parole_duree` | float | Durée de parole en secondes |
| `parole_ratio` | float | Ratio parole / durée totale |
| `dialogue` | string \| null | Texte transcrit. `""` si la transcription est vide ou rejetée par le filtre qualité ; `null` si le message n'a jamais été transcrit (pas d'audio) |
| `dialogue_confiance` | float | Score QA [0, 1] produit par `qa_whisper.py` (1 = très fiable) |
| `alerte_no_dialogue` | bool | Pas de dialogue exploitable |
| `alerte_hallucination_phrase` | bool | Outro YouTube ukr/ru détecté |
| `alerte_repeated_ngram` | bool | Trigramme dominant > 25 % du texte |
| `alerte_tokens_per_sec_anomaly` | bool | Débit hors `[0.5, 5.0]` tok/s |
| `alerte_compression_high` | bool | `len(tokens)/len(unique) > 2.4` |
| `alerte_low_conf` | bool | Proportion `logprob < -1` trop haute |
| `alerte_high_no_speech` | bool | Proportion `no_speech_prob > 0.6` trop haute |

`whisper_batch.py` produit les quatre premiers champs ; `dialogue_confiance`, les
7 `alerte_*` et `qa_patterns_version` (version du YAML de patterns, pour traçabilité)
sont produits par `qa_whisper.py` (section suivante).

Les fichiers SRT sont écrits dans le dossier `fiches/` aux côtés des médias
(chemin déductible : `fiches/{canal}_{message_id}.srt`, pas de champ JSONL).
Les segments Whisper bruts ne sont plus persistés dans le JSONL (16 MB économisés) —
ils sont disponibles dans le SRT si besoin.

## qa_whisper.py — scoring de qualité sans re-transcription

```bash
# Enrichit le JSONL + produit un CSV synthèse pour tri manuel
python qa_whisper.py --input  messages_clean.jsonl \
                     --output messages_clean.jsonl \
                     --csv    output/qa_whisper.csv

# Filtrer les suspects en R / pandas :
#   df_clean = df[df["dialogue_confiance"] >= 0.5]
#   df_suspect = df[df["dialogue_confiance"] < 0.5]
```

Le score combine la médiane des `avg_logprob` Whisper, la proportion de
segments basse confiance, la détection de patterns d'hallucination connus
(« Дякую за перегляд », outros YouTube…), le `compression_ratio` et le
débit tokens/seconde. Seuils documentés en tête du script.

### ⚠️ Les `dialogue_confiance` ne sont pas recalculables tels quels

Le signal `avg_logprob` / `no_speech_prob` provient des **segments Whisper**,
que `whisper_batch.py` ne persiste **plus** dans le JSONL (strippés pour
économiser ~16 MB ; ils restent dans les SRT). Les `dialogue_confiance`
présents dans le corpus ont été calculés lors d'un **run historique** où ces
segments existaient encore : ils sont **valides** mais **non reproductibles**
depuis le seul JSONL canonique.

Pour éviter qu'un rerun n'écrase ces scores valides par du bruit,
`qa_whisper.py` **refuse de tourner** (exit 1, rien écrit) si les messages
éligibles n'ont pas de `whisper_segments` :

```
Aucun message éligible n'a de `whisper_segments` : score fiable impossible (…)
```

Recalculer proprement supposerait de **refaire tourner Whisper** sur tout le
corpus (plusieurs heures GPU). En attendant, on peut forcer un scoring
**dégradé texte-seul** (`logprob` ignoré, score plafonné à 0.3, alertes
`low_conf`/`high_no_speech` désactivées) avec `--allow-degraded` — à n'utiliser
qu'en connaissance de cause, jamais sur le canonique de production.

## audio_segmentation.py — parole / musique / silence (optionnel)

Classifie l'audio de chaque vidéo en segments {parole, musique, silence}
via **inaSpeechSegmenter** (INA, **licence CC-BY-NC** — usage non commercial).
Sert surtout à flagger les compilations P3 à forte musique de fond, où
`dialogue` est peu fiable.

⚠️ **Venv séparé requis.** inaSpeechSegmenter tourne sous TensorFlow 2.16+,
qui entre en conflit cuDNN avec le venv principal (PyTorch). Créer un
`venv_ina/` dédié — instructions en tête du script.

```bash
venv_ina/bin/python audio_segmentation.py \
  --input  messages_clean.jsonl \
  --output messages_ina.jsonl \
  --media-dir /chemin/processed [--seuil-musique 0.5]
```

| Champ | Type | Description |
|-------|------|-------------|
| `audio_parole_pure_ratio` | float | Ratio parole seule / durée totale |
| `audio_musique_ratio` | float | Ratio musique / durée totale |
| `audio_parole_sur_musique_ratio` | float | Toujours `0.0` — la segmentation INA est séquentielle (labels exclusifs), le chevauchement parole/musique n'est pas mesurable. Champ conservé pour compatibilité |
| `audio_silence_ratio` | float | Ratio silence / durée totale |
| `audio_dominant` | string | `parole` \| `musique` \| `silence` \| `mixte` |
| `alerte_musique_dominante` | bool | `audio_musique_ratio > seuil` (défaut 0.5) |
| `audio_segmentation_modele` | string | Identifiant du modèle (`inaSpeechSegmenter-0.7`) |

Le CNN INA est entraîné sur de la radio française (2018) ; sa généralisation
à l'audio drone/militaire ukrainien n'est pas garantie. Faire un sanity-check
humain sur 5-10 vidéos avant d'exploiter les résultats. Ces champs ne font pas
partie du schéma canonique 52 champs et sont produits dans un JSONL séparé
(`messages_ina.jsonl`).



## Méthodologie

La transcription suit un pipeline en deux passes :
1. **Détection d'activité vocale** (Silero-VAD) — écarte les vidéos sans parole 
   avant de solliciter Whisper, évitant les transcriptions parasites sur images 
   de drones sans commentaire.
2. **Transcription** (Whisper large-v3) — forcée en ukrainien (`--language uk`), 
   avec prompt de conditionnement incluant le jargon militaire du corpus 
   (FPV, аеророзвідка, 414 бригада...) pour réduire les erreurs de vocabulaire.

Les segments sont ensuite filtrés : rejet si confidence globale < 0.35 
ou ratio non-cyrillique > 40%. Ce seuil sacrifie quelques vidéos mixtes 
(ukrainien + anglais) au profit de la qualité du corpus principal.
