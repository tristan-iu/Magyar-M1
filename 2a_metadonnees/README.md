# 2a_metadonnees — Métadonnées ffprobe

Extraction batch des métadonnées techniques (durée, résolution, bitrate, fps, audio)
de chaque fichier média du corpus via `ffprobe`. Première étape du pipeline (E1), prérequis
pour Whisper (durée) et les analyses de forme.

## Dépendances

```bash
sudo apt install ffmpeg
```

Seule dépendance Python : `pyyaml` (via la bibliothèque partagée `0_config/utils.py`) —
le traitement lui-même est en stdlib pure.

```bash
pip install -r requirements.txt
```

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

## Input / Output

**Lit :** JSONL avec champ `media_chemin` (chemin relatif vers le fichier)
**Produit :** même JSONL + champs ci-dessous

**Champs ajoutés :**

| Champ | Type | Description |
|-------|------|-------------|
| `duree` | `float` | Durée en secondes |
| `largeur` | `int` | Largeur vidéo en pixels |
| `hauteur` | `int` | Hauteur vidéo en pixels |
| `orientation` | `str` | `"vertical"` \| `"horizontal"` \| `"square"` (calculé depuis largeur/hauteur) |
| `video_bitrate` | `int` | Bitrate vidéo en kbps |
| `fps` | `float` | Images par seconde |
| `audio_present` | `bool` | Présence d'une piste audio |
| `fichier_taille` | `int` | Taille du fichier en octets |

Codecs vidéo/audio droppés (zéro variance dans le corpus). Photos : `fichier_taille`,
`largeur`, `hauteur`, `orientation` uniquement. Audio seul : `duree`, `audio_present`, `fichier_taille`.

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
que `fichier_taille` (+ largeur/hauteur/orientation) pour éviter des champs `duree=null` parasites dans le JSONL.

**`orientation` calculé ici plutôt qu'au scrape :** les dimensions Telegram (API) peuvent
différer de celles du fichier réel après recompression. On calcule l'orientation sur les
dimensions ffprobe, plus fiables.
