# Métadonnées ffprobe

Extraction batch des métadonnées techniques (durée, résolution, bitrate, fps, présence audio) de chaque fichier média du corpus via `ffprobe`. Première étape du pipeline d'enrichissement, prérequis pour la transcription (durée, audio) et pour les analyses de forme.

## Installation

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

Le traitement lui-même est en stdlib pure : la seule dépendance Python (`pyyaml`) vient de la bibliothèque partagée `0_config/utils.py`.

## Utilisation

```bash
# Corpus complet. Relance après interruption : --input = --output (skip les déjà traités)
python ffprobe_batch.py --input messages.jsonl --output messages_ffprobe.jsonl
#   --limit 5                  test sur les 5 premiers messages
#   --overwrite                retraite même si les champs existent déjà
#   --ids 12 345               IDs ciblés (debug)
#   --start-date / --end-date  borne la plage de dates traitée (YYYY-MM-DD)
#   --media-dir DIR            racine des media_chemin relatifs (défaut : dossier parent de --input)
#   --config CHEMIN            config.yaml (défaut : 0_config/config.yaml)
```

## Output

Le script lit un JSONL dont les messages portent un `media_chemin` relatif, et produit le même JSONL enrichi des champs ci-dessous, plus les fiches individuelles mises à jour. Le batch ne s'interrompt jamais : fichier manquant, timeout ou stream vide sont skippés avec un warning, et les erreurs sont loguées dans `logs/`.

### Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `duree` | `float` | Durée en secondes |
| `largeur` | `int` | Largeur en pixels |
| `hauteur` | `int` | Hauteur en pixels |
| `orientation` | `str` | `"vertical"`, `"horizontal"` ou `"square"` (calculé depuis largeur/hauteur) |
| `video_bitrate` | `int` | Bitrate vidéo en kbps |
| `fps` | `float` | Images par seconde |
| `audio_present` | `bool` | Présence d'une piste audio |
| `fichier_taille` | `int` | Taille du fichier en octets |

Les photos ne reçoivent que `largeur`, `hauteur`, `orientation` et `fichier_taille` ; les fichiers audio seuls reçoivent `duree`, `audio_present` et `fichier_taille`.

## Méthodologie

**ffprobe plutôt que mediainfo ou mutagen :** ffprobe est déjà une dépendance du pipeline (ffmpeg sert à la transcription et à l'extraction de keyframes). Pas de dépendance Python supplémentaire, et la sortie JSON est stable entre versions.

**Préférence `format.duration` puis `stream.duration` :** pour les vidéos conteneurisées (MP4), la durée du format est plus fiable que celle du stream vidéo, qui peut être absente ou arrondie selon l'encodeur. Fallback sur la durée du stream si la durée format est manquante.

**Timeout 10 s :** ffprobe lit seulement les métadonnées, pas le contenu, donc 10 secondes suffisent largement. Au-delà, c'est un fichier corrompu ou tronqué : on skip et on log.

**Cas des photos :** ffprobe retourne des streams vidéo pour les images (codec `mjpeg`, `png`, etc.), mais sans durée ni audio. Ce cas est détecté pour ne stocker que les dimensions et la taille, et éviter des champs `duree=null` parasites dans le JSONL.

**`orientation` calculé ici plutôt qu'au scrape :** les dimensions renvoyées par l'API Telegram peuvent différer de celles du fichier réel après recompression. L'orientation est calculée sur les dimensions ffprobe, plus fiables.
