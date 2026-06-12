# Telegram Scraper

Collecte messages et médias depuis un canal Telegram. Produit un fichier JSONL
enrichi des métadonnées fournies par la plateforme (vues, réactions, transferts)
et de deux empreintes de hachage (MD5 exact + pHash perceptuel) pour repérer les
doublons. Trois modes : `scrape`, `retry`, `--inject`.

## Installation

```bash
pip install -r requirements.txt
```

Pour pouvoir utiliser l'API de Telegram, créez un fichier `.env` en suivant le format du `.env.example` fourni.

## Utilisation

**Mode scrape** — collecte dans une plage de dates, une seule chaîne Telegram à la fois :

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
# Chemins depuis config.yaml (paths.raw_path et paths.jsonl_clean)
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
└── fiches/
    ├── channel_12345_photo.jpg
    ├── channel_12346_1_photo.jpg
    ├── channel_12346_2_photo.jpg
    └── channel_12347_video.mp4
```

## Champs constituant le jsonl en output

Schéma propre francisé (post-migration avril 2026). Le scraper écrit aussi
`duree`, `largeur`, `hauteur` quand l'API Telegram les fournit (disponibles avant
téléchargement) — l'étape E1 ffprobe les recalcule ensuite sur le fichier réel et
ajoute les autres champs techniques (`audio_present`, `fps`, etc.).

```json
{
  "message_id": 12345,                # ID du message relatif à l'ordre de la chaîne
  "canal": "nom_chaine",              # Nom de la chaîne (URL : t.me/nom_chaine)
  "date": "2024-01-15T15:30:00",      # Publication en UTC (Telegram ne fournit que de l'UTC, stocké sans suffixe timezone)
  "album_id": null,                   # Si présent : grouped_id Telegram (album multi-médias)
  "album_rang": null,                 # Position du média dans l'album (1, 2, 3…)
  "est_transfere": false,             # True si le message est republié depuis une autre chaîne
  "legende": "Texte du message",      # Texte/légende du média
  "liens_externes": [                 # Liens des entités Telegram (inline + URL brutes)
    {"url": "https://x.com/...", "texte": "Twitter"},   # lien inline (ancre affichée)
    {"url": "https://youtu.be/xxxxx", "texte": null}    # URL brute déjà visible dans legende
  ],
  "media_type": "video",              # "photo"|"video"|"audio"|"document"|"other"|null (texte seul) — seuls photo/video/audio sont téléchargés
  "media_chemin": "fiches/canal_12345_video.mp4",  # Chemin relatif du fichier
  "duree": 45,                        # Secondes (métadonnées Telegram — recalculé par ffprobe en E1)
  "largeur": 1920,                    # Pixels (idem)
  "hauteur": 1080,                    # Pixels (idem)
  "fichier_hash": "a1b2c3...",        # MD5 du fichier (déduplication exacte)
  "perceptual_hash": "d4c3b2...",     # pHash visuel (déduplication post-recompression)
  "vues": 15000,                      # Vues totales à l'instant du scraping
  "transferts": 500,                  # Nombre de partages vers d'autres chaînes/utilisateurs
  "reactions": 120,                   # Nombre total de réactions (emojis)
  "reactions_detail": [               # Détail par emoji
    {"emoji": "👍", "count": 100},
    {"emoji": "❤", "count": 20}
  ]
}
```

## Liens externes — entités Telegram

Le champ `liens_externes` capture les liens que le texte brut de la légende ne
contient pas. Telegram stocke les URL « inline » (texte cliquable dont l'URL
n'apparaît pas dans la légende) dans les entités du message
(`MessageEntityTextUrl` / `MessageEntityUrl`), pas dans le texte brut. Le scraper
les extrait à la volée via `extraire_liens_externes()` :

- `MessageEntityTextUrl` → lien inline : `texte` = ancre affichée (ex. « Twitter »)
- `MessageEntityUrl` → URL brute déjà visible dans `legende` : `texte` = `null`
- `[]` = message sans lien

Les offsets d'entités sont décodés en UTF-16LE pour corriger le décalage causé par
les emojis du plan astral (🇺🇦, 🏦…) qui précèdent souvent les URL : Telegram compte
les offsets en code units UTF-16, alors que Python indexe en code points.

## Méthodologie

**Ordre chronologique :** on itère du plus ancien au plus récent (`reverse=True` dans Telethon) pour que le JSONL reflète l'ordre de publication du canal. Essentiel pour les analyses longitudinales, vu que la ligne N précède toujours la ligne N+1 temporellement.

**JSONL plutôt que JSON array :** l'écriture en append (un message à la fois) rend le script crash-safe. Si le scrape s'interrompt à mi-parcours, les données déjà écrites sont intactes. Le streaming ligne par ligne évite de charger l'intégralité du corpus en RAM, important si le corpus est lourd. 

**Double persistance :** chaque message est écrit dans le JSONL maître (pour les analyses batch R/Python) ET dans une fiche JSON individuelle (pour le pipeline d'enrichissement qui travaille message par message : ffprobe, Whisper, OCR, etc.).

**Idempotence :** au démarrage, le script charge les IDs déjà présents dans le JSONL et les skip. On peut relancer après une interruption sans rescraper.

**Hashing :** deux empreintes complémentaires. MD5 (déduplication exacte bit à bit) et pHash perceptuel (déduplication visuelle même après recompression Telegram). Le pHash est calculé sur la frame à t=1s pour les vidéos, pas la première frame (souvent noire).
