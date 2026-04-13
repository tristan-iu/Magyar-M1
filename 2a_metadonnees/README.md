# 2a_metadonnees — Métadonnées ffprobe

Extraction batch des métadonnées techniques (durée, résolution, codec, bitrate, fps, audio)
de chaque fichier média du corpus via `ffprobe`. Première étape du pipeline (E1), prérequis
pour Whisper (durée) et les analyses de forme.

## Dépendances

```bash
sudo apt install ffmpeg
```

Aucune dépendance Python externe (stdlib uniquement).

## Utilisation

```bash
# Test sur 5 messages
python ffprobe_batch.py --input messages.jsonl --output messages_ffprobe.jsonl --limit 5

# Corpus complet
python ffprobe_batch.py --input messages.jsonl --output messages_ffprobe.jsonl

# Relance après interruption (skip les déjà traités)
python ffprobe_batch.py --input messages_ffprobe.jsonl --output messages_ffprobe.jsonl

# Forcer le retraitement
python ffprobe_batch.py --input messages_ffprobe.jsonl --output messages_ffprobe.jsonl --overwrite
```

| Option | Description |
|--------|-------------|
| `--input` | JSONL source |
| `--output` | JSONL destination (peut être identique à `--input`) |
| `--media-dir` | Racine pour résoudre les `media_path` relatifs (défaut : dossier parent de `--input`) |
| `--limit` | Nombre max de messages à traiter |
| `--overwrite` | Retraiter même si les champs existent déjà |
| `--ids` | Liste d'IDs à traiter (debug ciblé) |

## Input / Output

**Lit :** JSONL avec champ `media_path` (chemin relatif vers le fichier)
**Produit :** même JSONL + champs ci-dessous

**Champs ajoutés :**

| Champ | Type | Description |
|-------|------|-------------|
| `duration` | `float` | Durée en secondes |
| `video_codec` | `str` | Codec vidéo (ex: `h264`, `hevc`) |
| `video_bitrate` | `int` | Bitrate vidéo en kbps |
| `fps` | `float` | Images par seconde |
| `has_audio` | `bool` | Présence d'une piste audio |
| `audio_codec` | `str` | Codec audio (ex: `aac`, `opus`) |
| `file_size` | `int` | Taille du fichier en octets |
| `orientation` | `str` | `"vertical"` \| `"horizontal"` \| `"square"` (déduit de `media_dimensions`) |

Photos : `file_size` uniquement (les dimensions sont déjà dans le JSONL via l'API Telegram).
Audio seul : `duration`, `has_audio`, `audio_codec`, `file_size`.

Les erreurs sont loguées dans `logs/ffprobe_errors.log`. Le batch ne s'interrompt jamais :
fichier manquant, timeout (10s) ou stream vide sont skip avec un warning.

## Méthodologie

**ffprobe plutôt que mediainfo ou mutagen :** ffprobe est déjà une dépendance du pipeline
(ffmpeg sert pour Whisper et l'extraction de keyframes). Pas de dépendance Python supplémentaire,
et la sortie JSON est stable entre versions.

**Préférence `format.duration` > `stream.duration` :** pour les vidéos conteneurisées (MP4),
la durée du format est plus fiable que celle du stream vidéo, qui peut être absente ou arrondie
selon l'encodeur. Fallback sur la durée du stream si la durée format est manquante.

**Timeout 10s :** ffprobe lit seulement les métadonnées (pas le contenu), donc 10s est largement
suffisant. Au-delà, c'est un fichier corrompu ou tronqué — on skip et on log.

**Photos ignorées pour la plupart des champs :** ffprobe retourne des streams vidéo pour les
images (codec `mjpeg`, `png`, etc.), mais sans durée ni audio. On détecte ce cas et on ne stocke
que `file_size` pour éviter des champs `duration=null` parasites dans le JSONL.

**`orientation` calculé ici plutôt qu'au scrape :** les dimensions Telegram (API) peuvent
différer de celles du fichier réel après recompression. On calcule l'orientation sur les
dimensions ffprobe, plus fiables.
