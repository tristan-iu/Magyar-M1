# 2b_transcription — Transcription Whisper

Transcription audio via OpenAI Whisper (large-v3, GPU CUDA). Produit un texte transcrit (`dialogue`) dans le jsonl, des sous-titres SRT, et des métadonnées de confiance pour chaque message avec audio.

## Dépendances

```bash
pip install -r requirements.txt
```

Requiert ffmpeg installé système et un GPU CUDA.

## Utilisation

```bash
# Lancer sur tout le corpus
python whisper_batch.py --input /chemin/messages_enriched.jsonl --output /chemin/messages_whisper.jsonl

# Tester sur 5 messages
python whisper_batch.py --input messages_enriched.jsonl --output messages_whisper.jsonl --limit 5

# Relancer après interruption (skip les déjà traités)
python whisper_batch.py --input messages_whisper.jsonl --output messages_whisper.jsonl
```

## Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `has_speech` | bool | Présence de parole détectée |
| `dialogue` | string | Texte transcrit |
| `speech_confidence` | float | Confiance moyenne (0–1) |
| `speech_duration` | float | Durée de parole en secondes |
| `speech_ratio` | float | Ratio parole / durée totale |
| `speech_language` | string | Langue détectée (généralement `uk`) |
| `srt_path` | string | Chemin vers le fichier SRT |

Les fichiers SRT sont écrits dans le dossier `fiches/` aux côtés des médias.

# Méthodologie 

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

Reconnait mal avec bruit du fond. Pas un problème pour Magyar (drones n'ont pas de micro intégré, tout son vient de l'exterieur). Ne peut pas reconnaitre entre un dialogue parlé et de la musique, qui constitue la majorité du contenu post oct 2024. 
