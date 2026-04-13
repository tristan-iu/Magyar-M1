# Telegram Scraper

Collecte messages et médias depuis un canal Telegram. Produit un fichier JSONL avec métadonnées et hashing pour déduplication.

## Installation

```bash
pip install -r requirements.txt
```

Pour pouvoir utiliser l'API Telegram, créez un fichier `.env` en suivant le format du `.env.example` fourni.

## Utilisation

**Mode scrape** — collecte dans une plage de dates :

```bash
python telegram_scraper.py @chaine 2024-01-01 2024-12-31 ./output
python telegram_scraper.py @chaine 2024-01-01 2024-12-31 ./output --limit 100
python telegram_scraper.py @chaine 2024-01-01 2024-12-31 ./output --no-media
python telegram_scraper.py @chaine 2024-01-01 2024-12-31 ./output --delay 2.0
```

**Mode retry** — retélécharge les médias manquants sans rescraper :

```bash
python telegram_scraper.py @chaine 2024-01-01 2024-12-31 ./output --retry
```

**Mode inject** — intègre de nouveaux messages scrapés dans un JSONL enrichi existant, sans connexion Telegram. Idempotent (seuls les `message_id` absents sont ajoutés) :

```bash
# Chemins depuis config.yaml (paths.raw_path et paths.jsonl_computervision)
python telegram_scraper.py --inject

# Chemins explicites
python telegram_scraper.py --inject --raw ./messages.jsonl --target ./messages_enriched.jsonl

# Prévisualiser sans modifier
python telegram_scraper.py --inject --dry-run
```

## Output 

```
output/
├── messages.jsonl
├── scrape_log.json
└── fiches/
    ├── channel_12345_photo.jpg
    ├── channel_12346_1_photo.jpg
    ├── channel_12346_2_photo.jpg
    └── channel_12347_video.mp4
```

## Champs constituants le jsonl en output

```json
{
  "message_id": 12345,
  "channel": "channelname",
  "date": "2024-01-15T15:30:00",
  "grouped_id": null,
  "is_forwarded": false,
  "media_index": null,
  "caption": "Texte du message",
  "media_type": "video",
  "media_path": "fiches/channel_12345_video.mp4",
  "media_duration": 45,
  "media_dimensions": [1920, 1080],
  "file_hash": "a1b2c3...",
  "perceptual_hash": "d4c3b2...",
  "views": 15000,
  "forwards": 500,
  "reactions": 120,
  "reactions_detail": [{"emoji": "...", "count": 100}]
}
```

## Méthodologie

**Ordre chronologique :** on itère du plus ancien au plus récent (`reverse=True` dans Telethon) pour que le JSONL reflète l'ordre de publication du canal. Essentiel pour les analyses longitudinales, vu que la ligne N précède toujours la ligne N+1 temporellement.

**JSONL plutôt que JSON array :** l'écriture en append (un message à la fois) rend le script crash-safe. Si le scrape s'interrompt à mi-parcours, les données déjà écrites sont intactes. Le streaming ligne par ligne évite de charger l'intégralité du corpus en RAM, important si le corpus est lourd. 

**Double persistance :** chaque message est écrit dans le JSONL maître (pour les analyses batch R/Python) ET dans une fiche JSON individuelle (pour le pipeline d'enrichissement qui travaille message par message : ffprobe, Whisper, OCR, etc.).

**Idempotence :** au démarrage, le script charge les IDs déjà présents dans le JSONL et les skip. On peut relancer après une interruption sans rescraper.

**Hashing :** deux empreintes complémentaires. MD5 (déduplication exacte bit à bit) et pHash perceptuel (déduplication visuelle même après recompression Telegram). Le pHash est calculé sur la frame à t=1s pour les vidéos, pas la première frame (souvent noire).
