#!/usr/bin/env python3
"""
Scrape messages et médias d'un canal Telegram avec l'API Telegram et la librarie Telethon.

Produit un fichier maître JSONL + une fiche JSON par message + les fichiers médias.
Calcule les hashs MD5 et pHash pour la déduplication.

Pipeline :
  1. Connexion Telethon + authentification
  2. Itération chronologique sur les messages (ancien → récent)
  3. Extraction métadonnées + téléchargement média
  4. Écriture JSONL (append, crash-safe) + fiche individuelle

Modes : scrape (défaut), --retry, --inject. Options : --delay, --limit, --no-media, --config

Usage:
    python telegram_scraper.py @channel 2023-01-01 2023-01-15 ./output
    python telegram_scraper.py @channel 2023-01-01 2023-01-15 ./output --limit 50 --no-media
    python telegram_scraper.py @channel 2023-01-01 2023-01-15 ./output --retry  # Retélécharge médias manquants
    python telegram_scraper.py --inject [--raw ...] [--target ...] [--dry-run]  # Injection JSONL, sans connexion

Structure output:
    ./output/
    ├── messages.jsonl                          # Fichier maître
    └── fiches/
        ├── channel_3857_fiche.json             # Métadonnées
        ├── channel_3857_photo.jpg              # Média (si un seul)
        ├── channel_3856_fiche.json
        ├── channel_3856_1_photo.jpg            # Média 1 (si album)
        ├── channel_3856_2_photo.jpg            # Média 2 (si album)
        └── channel_3855_video.mp4

Structure JSON:
    {
      "message_id": 12345,
      "canal": "channelname",
      "date": "2024-01-15T15:30:00",
      "album_id": null,
      "est_transfere": false,
      "album_rang": null,
      "legende": "Contenu du message...",
      "media_type": "video",
      "media_chemin": "fiches/channel_12345_video.mp4",
      "liens_externes": [{"url": "https://x.com/...", "texte": "Twitter"}],
      "duree": 45,
      "largeur": 1920,
      "hauteur": 1080,
      "fichier_hash": "a1b2c3d4e5f6...",
      "perceptual_hash": "d4c3b2a1e5f6...",
      "vues": 15000,
      "transferts": 500,
      "reactions": 120,
      "reactions_detail": [{"emoji": "...", "count": 100}]
    }

Notes:
    - Toutes les dates sont en UTC (sans timezone explicite)
    - album_id identifie les messages appartenant au même album
    - largeur, hauteur en pixels (Telegram, peut être écrasé par ffprobe)
    - fichier_hash: MD5 du fichier, perceptual_hash: pHash pour déduplication visuelle
    - Support proxy optionnel via .env (PROXY_TYPE, PROXY_HOST, PROXY_PORT, etc.)

Dépendances hashing (optionnelles):
    pip install opencv-python pillow imagehash
"""

import os
import sys
import json
import asyncio
import argparse
import random
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Import utils pour logging cohérent et nommage fichiers
# on remonte d'un niveau pour aller chercher 0_config/ — convention projet
_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config, init_logger  # noqa: E402

# Logger module : configuré par init_logger() dans le point d'entrée.
# Console INFO + fichier WARNING horodaté (logs/telegram_scraper_errors.log).
log = logging.getLogger("telegram_scraper")

# Dépendances pour le hashing. Optionnelles parce que sur certaines machines
# (serveur, CI) opencv est pénible à installer. Le scraper fonctionne sans,
# on perd juste le hashing et la possibilité de repérer les duplicats. 
try:
    import cv2
    from PIL import Image
    import imagehash
    HASHING_AVAILABLE = True
except ImportError:
    HASHING_AVAILABLE = False
from telethon.sync import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto,
    MessageMediaDocument,
    ReactionEmoji,
    ReactionCustomEmoji,
    DocumentAttributeVideo,
    PhotoSize,
    PhotoSizeProgressive,
    MessageEntityTextUrl,
    MessageEntityUrl,
)


# ── Configuration & parsing ──────────────────────────────────────────────────
# On gère ici tout ce qui est entrée utilisateur : arguments CLI, credentials,
# proxy. Séparé de la logique de scrape pour pouvoir tester indépendamment.

def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Scrape messages et médias d'une chaîne Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python telegram_scraper.py @channel 2023-01-03 2023-01-16 ./output
  python telegram_scraper.py @channel 2023-01-03 2023-01-16 ./output --no-media
  python telegram_scraper.py @channel 2023-01-01 2023-01-05 ./output --limit 20
  python telegram_scraper.py @channel 2023-01-01 2023-01-05 ./output --retry
  python telegram_scraper.py --inject [--raw raw.jsonl] [--target enriched.jsonl] [--dry-run]
        """
    )
    parser.add_argument("channel", nargs="?", default=None,
                        help="Username (@name) ou URL (https://t.me/name) — requis hors mode --inject")
    parser.add_argument("start_date", nargs="?", default=None,
                        help="Date de début inclusive (YYYY-MM-DD) — requis hors mode --inject")
    parser.add_argument("end_date", nargs="?", default=None,
                        help="Date de fin inclusive (YYYY-MM-DD) — requis hors mode --inject")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Dossier de sortie — requis hors mode --inject")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Délai de base entre requêtes en secondes (défaut: 1.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Nombre maximum de messages à scraper (défaut: illimité)")
    parser.add_argument("--no-media", action="store_true",
                        help="Ne pas télécharger les médias, métadonnées uniquement")
    parser.add_argument("--retry", action="store_true",
                        help="Mode retry: retélécharge uniquement les médias manquants")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml (pour defaults channel/output)")
    parser.add_argument("--inject", action="store_true",
                        help="Mode injection : intègre les nouveaux messages raw dans le JSONL enrichi (pas de connexion Telegram)")
    parser.add_argument("--raw", default=None,
                        help="[--inject] JSONL raw source (défaut : paths.raw_path dans config.yaml)")
    parser.add_argument("--target", default=None,
                        help="[--inject] JSONL enrichi destination (défaut : paths.jsonl_clean depuis config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="[--inject] Affiche les messages à injecter sans modifier le fichier")

    return parser.parse_args()


def charger_identifiants():
    """Charge les credentials Telegram depuis le fichier .env."""
    # on charge deux fois : d'abord le .env global (racine projet), puis celui
    # du dossier scraper. Le second écrase le premier si doublon.
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")

    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    phone = os.getenv("TELEGRAM_PHONE_NUMBER")

    if not all([api_id, api_hash, phone]):
        log.error("Credentials manquants dans .env")
        log.error("Créez un fichier .env avec :")
        log.error("  TELEGRAM_API_ID=your_api_id")
        log.error("  TELEGRAM_API_HASH=your_api_hash")
        log.error("  TELEGRAM_PHONE_NUMBER=+33612345678")
        log.error("Obtenez vos credentials sur : https://my.telegram.org/apps")
        sys.exit(1)

    return int(api_id), api_hash, phone


def charger_config_proxy() -> dict | None:
    """Charge la configuration proxy depuis .env si présente.

    Utile quand le scrape se fait depuis un réseau universitaire qui bloque
    Telegram, ou pour anonymiser les requêtes.

    Entrée : (variables d'environnement PROXY_TYPE, PROXY_HOST, PROXY_PORT)
    Sortie : dict config proxy pour Telethon, ou None si pas de proxy configuré
    """
    proxy_type = os.getenv("PROXY_TYPE")
    proxy_host = os.getenv("PROXY_HOST")
    proxy_port = os.getenv("PROXY_PORT")

    if not all([proxy_type, proxy_host, proxy_port]):
        return None

    # Telethon utilise python-socks, format dict
    proxy_config = {
        'proxy_type': proxy_type.lower(),  # 'socks5', 'socks4', 'http'
        'addr': proxy_host,
        'port': int(proxy_port),
        'rdns': True,  # Remote DNS resolution — IMPORTANT sinon le DNS leak trahit l'IP
    }

    # Credentials proxy optionnels
    proxy_user = os.getenv("PROXY_USER")
    proxy_pass = os.getenv("PROXY_PASS")
    if proxy_user and proxy_pass:
        proxy_config['username'] = proxy_user
        proxy_config['password'] = proxy_pass

    return proxy_config


# ── Parsing & normalisation des données Telegram ─────────────────────────────
# Telegram renvoie des objets Telethon complexes (MessageMediaPhoto, etc.).
# On les normalise ici en types Python simples (str, int, dict) pour le JSONL.
# C'est ce qui garantit un schéma stable en sortie, indépendant de l'API
# Telegram qui peut changer entre versions de Telethon.

def analyser_nom_canal(channel_input: str) -> str:
    """Extrait le username depuis l'input utilisateur.

    Accepte plusieurs formats parce qu'on copie-colle souvent depuis le navigateur
    ou depuis l'app : @robert_magyar, https://t.me/robert_magyar, t.me/robert_magyar

    Entrée : channel_input — str (URL, @mention ou nom brut)
    Sortie : username sans @ ni domaine (ex: "robert_magyar")
    """
    channel = channel_input.strip()
    # ici on gère les URL Telegram (t.me/channel?query=...) et les @mentions
    if "t.me/" in channel:
        channel = channel.split("t.me/")[-1].split("/")[0].split("?")[0]
    if channel.startswith("@"):
        channel = channel[1:]
    return channel


def analyser_date(date_str: str) -> datetime:
    """Parse une date string en datetime UTC."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        log.error("Format de date invalide '%s'. Utilisez YYYY-MM-DD", date_str)
        sys.exit(1)


def format_datetime(dt: datetime | None) -> str | None:
    """Formate un datetime en string UTC sans timezone.

    On stocke SANS timezone dans le JSONL (format ISO 8601 tronqué).
    Choix délibéré : toutes les dates Telegram sont en UTC, on le sait,
    pas besoin de +00:00 partout — ça simplifie le parsing en R/Python après.

    Entrée : dt — datetime (avec ou sans tzinfo), ou None
    Sortie : str "YYYY-MM-DDTHH:MM:SS", ou None si dt est None
    """
    if dt is None:
        return None
    # Convertir en UTC si nécessaire, puis formater sans timezone
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# Types de média qu'on télécharge réellement comme fichier. Les "document"
# (stickers/GIF non analysés) et "other" (sondages, géoloc, webpage) ne sont
# pas des fichiers exploitables par le pipeline → on les laisse sans média.
# Un message texte seul a media_type=None et n'a rien à télécharger.
TYPES_MEDIA_TELECHARGEABLES = ("photo", "video", "audio")


def obtenir_type_media(message) -> str | None:
    """Détermine le type de média du message.

    Telegram distingue Photo (compressée par le serveur) et Document (fichier brut).
    Un Document peut être une vidéo, un audio, une image non compressée, etc.
    On simplifie en 5 catégories : photo, video, audio, document, other.
    Pour le corpus Magyar, on a surtout des vidéos (~921) et des photos (~246).

    Entrée : message — objet Message Telethon
    Sortie : "photo" | "video" | "audio" | "document" | "other" | None (pas de média)
    """
    if not message.media:
        return None
    if isinstance(message.media, MessageMediaPhoto):
        return "photo"
    if isinstance(message.media, MessageMediaDocument):
        doc = message.media.document
        if doc:
            # on se base sur le MIME type plutôt que l'extension — plus fiable
            mime = getattr(doc, 'mime_type', '') or ''
            if mime.startswith('video'):
                return "video"
            if mime.startswith('audio'):
                return "audio"
            if mime.startswith('image'):
                return "photo"
            return "document"
    return "other"


def obtenir_extension_media(media_type: str) -> str:
    """Retourne l'extension de fichier pour un type de média.

    On force des extensions standardisées plutôt que de prendre celles de Telegram
    — ça évite les .webm, .ogg, .webp qui compliquent ffprobe/Whisper après.
    IMPORTANT : l'extension doit être une fonction PURE de media_type (pas du
    mime réel), pour que le mode --retry puisse reconstruire le nom de fichier
    attendu à partir du seul JSONL (cf. nom_fichier_media). Une note vocale opus
    est donc sauvée en .mp3 ; ffprobe/Whisper la lisent au contenu, l'extension
    n'est qu'une étiquette.

    Entrée : media_type — str ("photo", "video", "audio", "document", "other")
    Sortie : extension de fichier sans point (ex: "mp4", "jpg")
    """
    extensions = {
        "photo": "jpg",
        "video": "mp4",
        "audio": "mp3",
        "document": "bin",
        "other": "bin"
    }
    return extensions.get(media_type, "bin")


def nom_fichier_media(
    canal: str, message_id: int, media_type: str, album_rang: int | None = None
) -> str:
    """Construit le nom de fichier d'un média — source unique de vérité.

    Convention : canal_msgid[_rang]_type.ext. Déterministe (dépend uniquement
    de valeurs présentes dans le JSONL) → le mode --retry reconstruit ce nom
    sans avoir besoin de l'objet Message Telethon.

    Entrée : canal — str, message_id — int, media_type — str,
             album_rang — int si album, None si solo
    Sortie : nom de fichier (ex: "robert_magyar_8_video.mp4")
    """
    ext = obtenir_extension_media(media_type)
    if album_rang is not None:
        return f"{canal}_{message_id}_{album_rang}_{media_type}.{ext}"
    return f"{canal}_{message_id}_{media_type}.{ext}"


def extraire_meta_media(message) -> dict:
    """Extrait les métadonnées du média : dimensions [w, h] et durée.

    ATTENTION : ces dimensions viennent de l'API Telegram, pas du fichier lui-même.
    Elles peuvent différer de ce que ffprobe retourne (Telegram compresse/redimensionne).
    D'où le calcul via ffprobe plus tard dans le pipeline (E1).
    Ici on prend quand même les valeurs Telegram parce qu'elles sont disponibles
    AVANT le téléchargement du média, utile pour filtrer sans tout télécharger.

    Entrée : message — objet Message Telethon
    Sortie : dict avec "largeur", "hauteur" et/ou "duree" (float, secondes)
    """
    if not message.media:
        return {}

    metadata = {}
    width, height = None, None

    # Photo : Telegram stocke plusieurs tailles (thumbnail, medium, full).
    # On prend la plus grande — c'est celle qu'on télécharge.
    if isinstance(message.media, MessageMediaPhoto):
        photo = message.media.photo
        if photo and hasattr(photo, 'sizes') and photo.sizes:
            largest = None
            largest_area = 0
            for size in photo.sizes:
                if isinstance(size, (PhotoSize, PhotoSizeProgressive)):
                    w = getattr(size, 'w', 0)
                    h = getattr(size, 'h', 0)
                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest = size

            if largest:
                width = getattr(largest, 'w', None)
                height = getattr(largest, 'h', None)

    # Vidéo/Document : la durée et les dimensions sont dans DocumentAttributeVideo
    elif isinstance(message.media, MessageMediaDocument):
        doc = message.media.document
        if doc and hasattr(doc, 'attributes'):
            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeVideo):
                    metadata["duree"] = attr.duration
                    width = attr.w
                    height = attr.h
                    break

    if width is not None and height is not None:
        metadata["largeur"] = width
        metadata["hauteur"] = height

    return metadata


def est_transfere_message(message) -> bool:
    """Retourne True si le message est un forward, False sinon.

    Important pour le corpus : les forwards ne sont PAS du contenu original de Magyar.
    On les garde dans le JSONL (c'est un choix éditorial — le forward fait partie de
    l'activité du canal) mais on peut les filtrer en analyse.

    Entrée : message — objet Message Telethon
    Sortie : bool
    """
    return message.forward is not None


def extraire_reactions(message) -> dict:
    """Extrait les réactions : total + détail par emoji.

    On dénormalise ici (total + liste de {emoji, count}) pour avoir les deux
    niveaux d'analyse : le total pour les corrélations rapides, le détail
    pour l'analyse des types de réactions (cf. df_reactions en R).
    Les custom emojis (payants, Telegram Premium) sont ignorés — pas d'emoji
    visible côté API, donc inutilisables en analyse.

    Entrée : message — objet Message Telethon
    Sortie : dict {"reactions": int, "reactions_detail": [{"emoji": str, "count": int}]}
    """
    if not message.reactions or not message.reactions.results:
        return {"reactions": 0, "reactions_detail": []}

    reactions_detail = []
    total = 0

    for reaction_count in message.reactions.results:
        count = reaction_count.count
        total += count

        reaction = reaction_count.reaction
        if isinstance(reaction, ReactionEmoji):
            reactions_detail.append({
                "emoji": reaction.emoticon,
                "count": count
            })
        # custom emojis ignorés — pas de représentation textuelle exploitable

    return {
        "reactions": total,
        "reactions_detail": reactions_detail
    }


# ── Liens externes (entités Telegram) ────────────────────────────────────────
# Telegram distingue deux types d'entités URL :
#   - MessageEntityUrl      : URL brute écrite en clair dans le texte
#                             (ex: "https://youtu.be/xxxxx" dans la légende)
#                             → texte = null (déjà visible dans `legende`)
#   - MessageEntityTextUrl  : texte hyperlié — l'URL n'est PAS dans le texte
#                             (ex: le mot "Twitter" qui cache https://x.com/...)
#                             → texte = l'ancre affichée (ex: "Twitter")
#
# IMPORTANT — encodage UTF-16 :
# Telegram stocke les offsets d'entités en UTF-16 code units. Python traite
# les strings en code points Unicode. Les emojis du plan astral (🏦, 🇺🇦, etc.)
# valent 2 code units UTF-16 mais 1 char Python → chaque emoji avant une URL
# décale l'extraction de 1 char. Résultat : "https://youtube.com" devient
# "tps://youtube.com" selon le nombre d'emojis précédents.
# Correction : encoder en UTF-16LE, appliquer l'offset en bytes, décoder.

def _segment_utf16(texte: str, offset: int, length: int) -> str:
    """Extrait un segment en respectant les offsets UTF-16 de Telegram.

    Entrée : texte — str Python, offset/length — en UTF-16 code units (Telegram)
    Sortie : sous-chaîne correcte même si des emojis précèdent l'URL
    """
    encoded = texte.encode("utf-16-le")
    segment = encoded[offset * 2 : (offset + length) * 2]
    return segment.decode("utf-16-le")


def extraire_liens_externes(message) -> list[dict]:
    """Extrait tous les liens URL des entités d'un message Telegram.

    Le texte brut de la légende ne contient pas les liens « inline » (texte
    cliquable dont l'URL est cachée). On les récupère depuis les entités.

    Entrée : message — objet Message Telethon (avec .entities et .message)
    Sortie : liste de {"url": str, "texte": str|None}, vide si aucun lien
    """
    liens = []

    if not message.entities:
        return liens

    texte = message.message or ""

    for ent in message.entities:
        if isinstance(ent, MessageEntityTextUrl):
            # Lien inline : l'URL est dans ent.url, l'ancre dans le texte
            ancre = _segment_utf16(texte, ent.offset, ent.length).strip()
            liens.append({"url": ent.url, "texte": ancre or None})

        elif isinstance(ent, MessageEntityUrl):
            # URL brute : l'URL est dans le texte lui-même (offset + length)
            # On passe par l'extraction UTF-16 pour corriger le décalage emoji
            url = _segment_utf16(texte, ent.offset, ent.length).strip()
            if url:
                liens.append({"url": url, "texte": None})

    return liens


# ── Hashing ──────────────────────────────────────────────────────────────────
# Deux types de hash, deux usages différents :
# - MD5 = empreinte binaire STRICTE. Deux fichiers identiques bit à bit = même MD5.
#   Sert à détecter les doublons exacts (ex: même vidéo postée deux fois).
# - pHash = empreinte PERCEPTUELLE. Deux images visuellement similaires = pHash proches.
#   Sert à détecter les reprises/re-uploads avec recompression.
#   Pour les vidéos, on extrait une frame à t=1s et on hash cette frame.
#
# Pourquoi les deux : MD5 seul rate les re-uploads (Telegram recompresse),
# pHash seul donne des faux positifs sur des images génériques (drapeaux, logos).

def calculer_md5(file_path: str, chunk_size: int = 8192) -> str | None:
    """Calcule le hash MD5 (empreinte binaire stricte).

    Lecture par chunks pour ne pas charger des vidéos de 500Mo en RAM.

    Entrée : file_path — chemin vers le fichier, chunk_size — taille de lecture en octets
    Sortie : str hexdigest MD5, ou None si erreur lecture
    """
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        log.warning("Erreur MD5 sur %s : %s", file_path, e)
        return None


def obtenir_image_pour_phash(media_path: str, media_type: str) -> "Image.Image | None":
    """Récupère une image PIL pour le calcul du pHash.

    Pour les vidéos : on extrait la frame à t=1s plutôt que la première frame,
    parce que beaucoup de vidéos Telegram commencent par un écran noir ou un
    fondu. La frame à 1s est plus représentative du contenu visuel.
    Si la vidéo fait moins de 2s, on prend la première frame quand même.

    Entrée : media_path — chemin vers le fichier, media_type — "photo" ou "video"
    Sortie : objet Image PIL (RGB), ou None si extraction impossible
    """
    if not HASHING_AVAILABLE:
        return None

    try:
        if media_type == 'photo':
            with open(media_path, 'rb') as f:
                return Image.open(f).convert('RGB')

        elif media_type == 'video':
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            # frame à 1s si possible, sinon première frame
            target_frame = int(fps) if duration > 2 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = cap.read()
            cap.release()

            if ret:
                # opencv est en BGR, PIL attend du RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame_rgb)
            return None

    except Exception as e:
        log.warning("Erreur extraction image pour pHash (%s) : %s", media_path, e)
        return None

    return None


def calculer_phash(media_path: str, media_type: str) -> str | None:
    """Calcule le Perceptual Hash (empreinte visuelle).

    hash_size=8 → hash de 64 bits. C'est le défaut d'imagehash, bon compromis
    entre précision et taille. La distance de Hamming entre deux pHash donne
    une mesure de similarité visuelle (0 = identique, >10 = très différent).

    Entrée : media_path — chemin vers le fichier, media_type — "photo" ou "video"
    Sortie : str pHash hexadécimal (ex: "f0e8c4a2b6d1e3f5"), ou None si erreur
    """
    if not HASHING_AVAILABLE:
        return None

    img = obtenir_image_pour_phash(media_path, media_type)
    if img:
        try:
            return str(imagehash.phash(img, hash_size=8))
        except Exception as e:
            log.warning("Erreur calcul pHash : %s", e)
    return None


def hacher_media(file_path: str, media_type: str) -> tuple[str | None, str | None]:
    """Calcule les deux hashs (MD5 + pHash) pour un fichier média."""
    file_hash = calculer_md5(file_path)
    perceptual_hash = calculer_phash(file_path, media_type)
    return file_hash, perceptual_hash


# ── Sérialisation — JSONL + fiches ───────────────────────────────────────────
# Ici on transforme un message Telegram en deux sorties :
# 1) une ligne dans le JSONL maître (pour les analyses batch)
# 2) une fiche JSON individuelle (pour le pipeline d'enrichissement)
#
# Le dict produit par message_vers_dict() est le SCHÉMA DE BASE du corpus.
# Tous les enrichissements ultérieurs (ffprobe, Whisper, OCR) AJOUTENT des
# champs à ce schéma, mais ne modifient jamais les champs existants.

def message_vers_dict(
    message,
    channel_username: str,
    media_path: str | None = None,
    media_index: int | None = None,
    file_hash: str | None = None,
    perceptual_hash: str | None = None
) -> dict:
    """Convertit un message Telegram en dictionnaire (format simplifié).

    C'est LE point de vérité pour le schéma JSONL. Si on veut ajouter un champ
    au corpus brut, c'est ici qu'il faut le faire.

    Entrée : message — objet Message Telethon, channel_username — str,
             media_path — chemin relatif du fichier téléchargé (ou None),
             media_index — numéro dans l'album (ou None), file_hash — MD5 (ou None),
             perceptual_hash — pHash (ou None)
    Sortie : dict conforme au schéma JSONL du corpus
    """

    is_forwarded = est_transfere_message(message)
    reactions_info = extraire_reactions(message)
    media_metadata = extraire_meta_media(message)

    return {
        # Identification — le couple (canal, message_id) est la clé primaire
        "message_id": message.id,
        "canal": channel_username,
        "date": format_datetime(message.date),
        "album_id": message.grouped_id,  # non-null si le message fait partie d'un album

        # Forward — permet de filtrer le contenu non-original
        "est_transfere": is_forwarded,

        # Album — index du média dans l'album (1, 2, 3...), null si message solo
        "album_rang": media_index,

        # Contenu textuel — légende Telegram, peut être null (vidéo sans texte)
        # on peut aussi avoir du texte SANS média (message texte pur)
        "legende": message.text or None,

        # Liens externes — URL des entités Telegram (inline + brutes).
        # [] si aucun lien. Le texte brut de `legende` ne contient pas les
        # liens « inline » (ancre cliquable masquant l'URL) → on les extrait ici.
        "liens_externes": extraire_liens_externes(message),

        # Media — type + chemin relatif vers le fichier téléchargé
        "media_type": obtenir_type_media(message),
        "media_chemin": media_path,
        **media_metadata,  # duree, largeur, hauteur (si disponibles)

        # Hashing — pour déduplication (cf. section hashing plus haut)
        "fichier_hash": file_hash,
        "perceptual_hash": perceptual_hash,

        # Engagement — métriques d'audience du message
        # ATTENTION : ces valeurs sont un snapshot au moment du scrape.
        # Elles évoluent dans le temps (un vieux post continue à accumuler des vues).
        "vues": message.views,
        "transferts": message.forwards,
        **reactions_info,  # reactions (total), reactions_detail (par emoji)
    }


async def telecharger_media(
    client,
    message,
    fiches_dir: Path,
    channel_username: str,
    media_index: int | None = None
) -> tuple[str | None, str | None, str | None]:
    """Télécharge le média et calcule les hashs.

    On télécharge photos, vidéos et audios (cf. TYPES_MEDIA_TELECHARGEABLES).
    Les "documents" (souvent stickers/GIF) et "other" (sondages, géoloc…) ne
    sont pas des fichiers exploitables par le pipeline → ignorés.

    Convention de nommage : channel_msgid[_index]_type.ext
    Le chemin stocké dans le JSONL est RELATIF à output_dir (ex: "fiches/robert_magyar_8_video.mp4").
    Ça permet de déplacer le corpus sans casser les chemins.

    Entrée : client — TelegramClient connecté, message — objet Message Telethon,
             fiches_dir — Path dossier de destination, channel_username — str,
             media_index — int si album, None si solo
    Sortie : tuple (media_path relatif, md5, pHash) ou (None, None, None) si échec
    """
    if not message.media:
        return None, None, None

    media_type = obtenir_type_media(message)
    if media_type not in TYPES_MEDIA_TELECHARGEABLES:
        return None, None, None

    # Nom de fichier déterministe (cf. nom_fichier_media) — réutilisé par --retry
    filename = nom_fichier_media(channel_username, message.id, media_type, media_index)

    fiches_dir.mkdir(parents=True, exist_ok=True)
    filepath = fiches_dir / filename

    try:
        await client.download_media(message, file=str(filepath))
        relative_path = f"fiches/{filename}"

        # hashs calculés APRÈS téléchargement — on hash le fichier local
        file_hash, perceptual_hash = hacher_media(str(filepath), media_type)

        return relative_path, file_hash, perceptual_hash
    except Exception as e:
        # on ne crashe PAS le pipeline — on log et on continue (règle du projet :
        # une erreur sur un message ne doit jamais interrompre la collecte)
        log.warning("Échec téléchargement média msg %s : %s", message.id, e)
        return None, None, None


def enregistrer_fiche(fiches_dir: Path, channel_username: str, message_id: int, msg_dict: dict):
    """Sauvegarde la fiche JSON individuelle.

    Une fiche = un fichier JSON par message. C'est le point d'entrée pour le
    pipeline d'enrichissement : ffprobe/Whisper/OCR ouvrent la fiche, ajoutent
    leurs champs, et la resauvegardent (merge incrémental).

    Entrée : fiches_dir — Path dossier de destination, channel_username — str,
             message_id — int, msg_dict — dict à sérialiser
    Sortie : None (écrit channel_username_messageid_fiche.json sur disque)
    """
    fiches_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{channel_username}_{message_id}_fiche.json"
    fiche_path = fiches_dir / filename

    with open(fiche_path, "w", encoding="utf-8") as f:
        json.dump(msg_dict, f, ensure_ascii=False, indent=2)


def ajouter_message_jsonl(filepath: Path, msg_dict: dict):
    """Ajoute un message au fichier JSONL (crash-safe).

    Mode append ("a") : si le script crashe à mi-parcours, on ne perd pas
    les messages déjà écrits. C'est pour ça qu'on a aussi le mécanisme de
    reprise (ids_existants) dans scraper_canal().

    Entrée : filepath — Path vers le fichier .jsonl, msg_dict — dict à sérialiser
    Sortie : None (ajoute une ligne JSON au fichier)
    """
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(msg_dict, ensure_ascii=False) + "\n")


# ── Rate limiting ────────────────────────────────────────────────────────────
# Telegram impose des limites de requêtes. Si on va trop vite : FloodWaitError
# (ban temporaire de 30s à 24h). Le délai randomisé ±30% évite les patterns
# réguliers que Telegram détecte comme du bot.

async def delai_aleatoire(base_delay: float):
    delay = base_delay * random.uniform(0.7, 1.3)
    await asyncio.sleep(delay)


# ── Mode retry — récupération des médias manquants ───────────────────────────
# Le téléchargement peut échouer (timeout, rate limit, fichier corrompu).
# Plutôt que de tout rescraper, on reparse le JSONL existant et on ne
# retélécharge QUE les médias dont le fichier est absent sur disque.
# Idempotent : on peut relancer autant de fois qu'on veut.

async def retenter_medias_manquants(
    client: TelegramClient,
    channel_name: str,
    output_dir: Path,
    delay: float
):
    """Retélécharge les médias manquants listés dans le JSONL.

    Couvre DEUX cas de média absent sur disque :
      (a) `media_chemin` renseigné mais fichier supprimé/déplacé ;
      (b) `media_chemin` null parce que le téléchargement a échoué PENDANT le
          scrape (timeout, FloodWait, coupure) — cas le plus courant.
    Dans les deux cas, le nom de fichier est déterministe (cf. nom_fichier_media)
    donc on reconstruit le chemin attendu à partir du seul JSONL. Après
    récupération, on réécrit `media_chemin` + les hashs dans le JSONL et la fiche.
    """

    log.info("Mode RETRY : vérification des médias manquants")

    jsonl_path = output_dir / "messages.jsonl"
    if not jsonl_path.exists():
        log.error("Fichier %s introuvable", jsonl_path)
        return

    # On parcourt le JSONL pour trouver les messages à média dont le fichier est
    # absent. On garde (message, chemin_relatif_attendu) — le chemin vient du
    # JSONL s'il est présent, sinon on le reconstruit.
    a_retenter: list[tuple[dict, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("media_type") not in TYPES_MEDIA_TELECHARGEABLES:
                continue
            chemin = msg.get("media_chemin") or "fiches/" + nom_fichier_media(
                msg["canal"], msg["message_id"], msg["media_type"], msg.get("album_rang")
            )
            if not (output_dir / chemin).exists():
                a_retenter.append((msg, chemin))

    if not a_retenter:
        log.info("Tous les médias sont présents !")
        return

    log.info("%d médias manquants à retélécharger", len(a_retenter))

    # on doit se reconnecter à la chaîne pour récupérer les messages par ID
    try:
        entity = await client.get_entity(channel_name)
    except Exception as e:
        log.error("Impossible d'accéder à '%s' : %s", channel_name, e)
        return

    n_succes = 0
    n_echecs = 0
    fiches_dir = output_dir / "fiches"

    # On collecte les champs à répercuter (chemin + hashs) pour réécrire le JSONL
    # et les fiches APRÈS la boucle (réécriture unique, atomique). Sans ça, un
    # média récupéré resterait à `media_chemin`/hash null dans le corpus.
    maj_medias: dict[int, dict] = {}

    for msg, chemin in a_retenter:
        msg_id = msg["message_id"]
        media_type = msg["media_type"]

        try:
            # on récupère le message FRAIS depuis Telegram (pas depuis le JSONL)
            # parce que Telethon a besoin de l'objet Message pour télécharger le média
            message = await client.get_messages(entity, ids=msg_id)

            if message is None:
                # le message a été supprimé entre-temps — ça arrive sur les canaux actifs
                log.warning("Retry msg %s : message supprimé", msg_id)
                n_echecs += 1
                continue

            if not message.media:
                log.warning("Retry msg %s : pas de média", msg_id)
                n_echecs += 1
                continue

            # Télécharger au chemin déterministe (reconstruit ou repris du JSONL)
            filepath = output_dir / chemin
            filepath.parent.mkdir(parents=True, exist_ok=True)

            await client.download_media(message, file=str(filepath))

            # Recalcul des hashs sur le fichier fraîchement récupéré
            file_hash, perceptual_hash = hacher_media(str(filepath), media_type)
            maj_medias[msg_id] = {
                "media_chemin": chemin,
                "fichier_hash": file_hash,
                "perceptual_hash": perceptual_hash,
            }

            log.info("Retry msg %s : ✓", msg_id)
            n_succes += 1

        except Exception as e:
            log.warning("Retry msg %s : ✗ %s", msg_id, e)
            n_echecs += 1

        await delai_aleatoire(delay)

    # Répercussion dans le corpus (JSONL maître + fiches)
    if maj_medias:
        _appliquer_maj_retry(jsonl_path, fiches_dir, maj_medias)

    log.info("Retry terminé — réussis : %d, échecs : %d", n_succes, n_echecs)


def _appliquer_maj_retry(
    jsonl_path: Path,
    fiches_dir: Path,
    maj_medias: dict[int, dict],
) -> None:
    """Répercute media_chemin + hashs récupérés en retry dans le JSONL et les fiches.

    Réécriture atomique du JSONL (tmp + os.replace) pour ne pas corrompre le
    fichier si le script est interrompu. Les fiches sont mises à jour en place
    en ne touchant QUE les champs concernés — les enrichissements éventuels
    (ffprobe, Whisper…) déjà présents sont préservés (règle d'idempotence).

    Entrée : jsonl_path — JSONL maître, fiches_dir — dossier des fiches,
             maj_medias — {message_id: {media_chemin, fichier_hash, perceptual_hash}}
    Sortie : None (réécrit le JSONL et les fiches concernées sur disque)
    """
    # JSONL : on relit tout, on patche les lignes concernées, on réécrit en tmp.
    # On préserve les lignes mal formées telles quelles plutôt que de les perdre.
    lignes: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                lignes.append(line.rstrip("\n"))
                continue
            maj = maj_medias.get(msg.get("message_id"))
            if maj is not None:
                msg.update(maj)
            lignes.append(json.dumps(msg, ensure_ascii=False))

    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for ligne in lignes:
            f.write(ligne + "\n")
    os.replace(tmp_path, jsonl_path)

    # Fiches : merge ciblé des champs récupérés, on préserve le reste
    for msg_id, maj in maj_medias.items():
        # On a besoin du canal pour le nom de fiche — on le relit depuis le JSONL
        # via une recherche légère (les fiches sont nommées canal_msgid_fiche.json).
        # Toutes les fiches partagent le même canal, donc on le déduit du glob.
        fiches = list(fiches_dir.glob(f"*_{msg_id}_fiche.json"))
        if not fiches:
            continue
        fiche_path = fiches[0]
        try:
            with open(fiche_path, "r", encoding="utf-8") as f:
                fiche = json.load(f)
            fiche.update(maj)
            with open(fiche_path, "w", encoding="utf-8") as f:
                json.dump(fiche, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Fiche %s non mise à jour : %s", fiche_path.name, e)

    log.info("Corpus mis à jour pour %d média(s) (JSONL + fiches)", len(maj_medias))


# ── Boucle principale de scrape ──────────────────────────────────────────────
# C'est ici que tout se passe : on itère sur les messages de la chaîne dans
# l'ordre chronologique, on extrait les métadonnées, on télécharge les médias,
# et on écrit dans le JSONL + les fiches.
#
# Propriétés importantes :
# - IDEMPOTENT : si le JSONL existe déjà, on skip les messages déjà scrapés
# - CRASH-SAFE : écriture en append, un message à la fois
# - CHRONOLOGIQUE : reverse=True dans iter_messages (ancien → récent)

async def scraper_canal(
    client: TelegramClient,
    channel_name: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    delay: float,
    limit: int | None,
    telecharger_medias: bool
):
    """Scrape les messages de la chaîne dans la plage de dates."""

    log.info("Connexion à la chaîne : %s", channel_name)

    try:
        entity = await client.get_entity(channel_name)
    except Exception as e:
        log.error("Impossible d'accéder à '%s' : %s", channel_name, e)
        return

    channel_title = getattr(entity, 'title', channel_name)
    channel_id = getattr(entity, 'id', None)

    log.info("Chaîne trouvée : %s (ID %s)", channel_title, channel_id)
    log.info("  Période : %s → %s", start_date.date(), end_date.date())
    log.info("  Limite : %s", limit if limit else "aucune")
    log.info("  Médias : %s", "non" if not telecharger_medias else "oui")

    # Setup output — on crée l'arborescence si elle n'existe pas
    output_dir.mkdir(parents=True, exist_ok=True)
    fiches_dir = output_dir / "fiches"
    fiches_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "messages.jsonl"

    # Reprise après interruption : on charge les IDs déjà scrapés pour les skip.
    # Ça rend le script IDEMPOTENT — on peut le relancer sans rescraper.
    ids_existants = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line)
                    ids_existants.add(msg["message_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if ids_existants:
            log.info("%d messages déjà scrapés, reprise...", len(ids_existants))

    n_messages = 0
    n_nouveaux = 0
    n_medias = 0

    # Gestion des albums Telegram : quand Magyar poste un album (plusieurs photos/vidéos
    # dans un même message), Telegram les envoie comme des messages séparés avec le même
    # grouped_id. On numérote les médias (1, 2, 3...) pour les distinguer dans le nommage.
    compteurs_album = {}  # grouped_id -> compteur

    # iter_messages avec reverse=True : on parcourt du plus ancien au plus récent.
    # offset_date = start_date : on commence à partir de cette date.
    # Telethon interprète offset_date comme "messages AVANT cette date" quand reverse=False, mais comme "messages APRÈS cette date" quand reverse=True.
    async for message in client.iter_messages(
        entity,
        offset_date=start_date,
        reverse=True  # chronologique : ancien → récent
    ):
        if message.date is None:
            continue

        msg_date = message.date.replace(tzinfo=timezone.utc)

        # fenêtre temporelle : on s'arrête dès qu'on dépasse end_date
        if msg_date > end_date:
            break
        if msg_date < start_date:
            continue

        n_messages += 1

        # Vérifier limite (utile pour les tests : --limit 10)
        if limit and n_nouveaux >= limit:
            log.info("Limite atteinte (%d messages)", limit)
            break

        # Skip si déjà scrapé — c'est ici que l'idempotence opère
        if message.id in ids_existants:
            continue

        n_nouveaux += 1

        # Numérotation des médias d'album
        media_index = None
        if message.grouped_id:
            if message.grouped_id not in compteurs_album:
                compteurs_album[message.grouped_id] = 0
            compteurs_album[message.grouped_id] += 1
            media_index = compteurs_album[message.grouped_id]

        # Téléchargement média (si demandé et si le message en contient un)
        media_path = None
        file_hash = None
        perceptual_hash = None

        if telecharger_medias and message.media and obtenir_type_media(message) in TYPES_MEDIA_TELECHARGEABLES:
            log.info("Média msg %s%s", message.id, f" (album #{media_index})" if media_index else "")
            media_path, file_hash, perceptual_hash = await telecharger_media(
                client, message, fiches_dir,
                channel_name, media_index
            )
            if media_path:
                n_medias += 1
            await delai_aleatoire(delay)

        # Construction du dict normalisé
        msg_dict = message_vers_dict(
            message, channel_name, media_path, media_index,
            file_hash, perceptual_hash
        )

        # Double écriture : JSONL maître (analyses batch) + fiche individuelle (pipeline)
        ajouter_message_jsonl(jsonl_path, msg_dict)
        enregistrer_fiche(fiches_dir, channel_name, message.id, msg_dict)

        # Progression visible (n/total dans les logs)
        if n_nouveaux % 25 == 0:
            log.info("... %d nouveaux messages", n_nouveaux)

        # délai réduit entre messages (pas de téléchargement média = plus rapide)
        await delai_aleatoire(delay / 2)

    log.info("Messages parcourus : %d | nouveaux : %d | médias téléchargés : %d",
             n_messages, n_nouveaux, n_medias)

    log.info("Fichier maître : %s", jsonl_path)
    log.info("Fiches : %s", fiches_dir)


# ── Injection JSONL ──────────────────────────────────────────────────────────
# Mode --inject : intègre les nouveaux messages scrapés (raw) dans le JSONL
# enrichi. Idempotent — seuls les message_id absents sont ajoutés. Ne nécessite
# pas de connexion Telegram.

def executer_injection(args) -> None:
    """Intègre les nouveaux messages raw dans le JSONL enrichi."""
    cfg = load_config(args.config)
    corpus_base = Path(cfg["paths"]["corpus_base"])

    cfg_raw = cfg["paths"].get("raw_path", "")
    if args.raw:
        raw_path = Path(args.raw)
    elif cfg_raw:
        raw_path = Path(cfg_raw)
    else:
        log.error("--raw requis ou paths.raw_path absent de config.yaml")
        sys.exit(1)

    target_path = (
        Path(args.target) if args.target
        else corpus_base / cfg["paths"]["jsonl_clean"]
    )

    if not raw_path.is_file():
        log.error("raw introuvable : %s", raw_path)
        sys.exit(1)
    if not target_path.is_file():
        log.error("target introuvable : %s", target_path)
        sys.exit(1)

    log.info("Lecture de %s...", target_path.name)
    with open(target_path, encoding="utf-8") as f:
        ids_existants = {json.loads(line)["message_id"] for line in f if line.strip()}
    log.info("  %d messages existants", len(ids_existants))

    nouvelles_lignes = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m["message_id"] not in ids_existants:
                nouvelles_lignes.append((m["message_id"], m.get("date", "?"), m.get("media_type"), line))

    nouvelles_lignes.sort(key=lambda x: x[0])

    if not nouvelles_lignes:
        log.info("Rien à injecter — le JSONL enrichi est déjà à jour.")
        return

    log.info("%d nouveaux messages à injecter :", len(nouvelles_lignes))
    types: dict[str, int] = {}
    for mid, date_str, mtype, _ in nouvelles_lignes:
        t = mtype or "text"
        types[t] = types.get(t, 0) + 1
        log.info("  msg %s | %s | %s", mid, date_str, mtype or "texte seul")
    log.info("Résumé types : %s", types)

    if args.dry_run:
        log.info("[dry-run] Aucune modification effectuée.")
        return

    with open(target_path, "a", encoding="utf-8") as f:
        for _, _, _, line in nouvelles_lignes:
            f.write(line + "\n")

    total = len(ids_existants) + len(nouvelles_lignes)
    log.info("%d messages injectés dans %s", len(nouvelles_lignes), target_path.name)
    log.info("Total : %d messages", total)

    with open(target_path, encoding="utf-8") as f:
        actual = sum(1 for line in f if line.strip())
    if actual == total:
        log.info("Vérification OK : %d lignes dans le fichier.", actual)
    else:
        log.warning("attendu %d lignes, trouvé %d.", total, actual)


# ── Point d'entrée ───────────────────────────────────────────────────────────
# Trois modes d'exécution :
# 1) Mode normal (défaut) : scrape la chaîne dans la plage de dates
# 2) Mode --retry : retélécharge les médias manquants sans rescraper
# 3) Mode --inject : intègre les nouveaux messages raw dans le JSONL enrichi
#
# La session Telethon est persistée dans telegram_session.session — pas besoin
# de se ré-authentifier à chaque lancement.

async def main(args=None):
    """Point d'entrée principal (modes scrape et retry)."""
    if args is None:
        args = parse_args()

    api_id, api_hash, phone = charger_identifiants()
    proxy_config = charger_config_proxy()

    channel = analyser_nom_canal(args.channel)
    output_dir = Path(args.output_dir).resolve()

    log.info("=== Telegram scraper ===")

    # session persistée dans le dossier du script (pas dans output_dir)
    session_path = Path(__file__).parent / "telegram_session"

    # Créer client avec ou sans proxy
    if proxy_config:
        log.info("Proxy configuré : %s %s:%s",
                 proxy_config['proxy_type'].upper(), proxy_config['addr'], proxy_config['port'])
        client = TelegramClient(str(session_path), api_id, api_hash, proxy=proxy_config)
    else:
        client = TelegramClient(str(session_path), api_id, api_hash)

    try:
        log.info("Connexion à Telegram...")
        await client.start(phone=phone)
        log.info("Connecté !")

        if args.retry:
            await retenter_medias_manquants(
                client=client,
                channel_name=channel,
                output_dir=output_dir,
                delay=args.delay
            )
        else:
            start_date = analyser_date(args.start_date)
            # end_date à 23:59:59 pour inclure toute la journée
            end_date = analyser_date(args.end_date).replace(hour=23, minute=59, second=59)

            await scraper_canal(
                client=client,
                channel_name=channel,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                delay=args.delay,
                limit=args.limit,
                telecharger_medias=not args.no_media
            )

    except KeyboardInterrupt:
        # Ctrl+C propre — les données déjà écrites sont safe (append mode)
        log.warning("Interrompu. Données sauvegardées.")
    finally:
        await client.disconnect()
        log.info("Déconnecté.")


if __name__ == "__main__":
    # On configure le logging partagé (console INFO + fichier WARNING horodaté)
    init_logger("telegram_scraper")

    _args = parse_args()
    if _args.inject:
        executer_injection(_args)
    else:
        if not all([_args.channel, _args.start_date, _args.end_date, _args.output_dir]):
            log.error("channel, start_date, end_date, output_dir sont requis en mode scrape.")
            log.error("  Usage : python telegram_scraper.py @channel YYYY-MM-DD YYYY-MM-DD ./output")
            log.error("  Injection : python telegram_scraper.py --inject [--raw ...] [--target ...]")
            sys.exit(1)
        asyncio.run(main(_args))
