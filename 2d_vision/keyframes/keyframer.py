#!/usr/bin/env python3
"""
Enrichissement vision du corpus — keyframes (ffmpeg), OCR (EasyOCR), détection de scènes.
Enrichit le JSONL avec : ocr_texte, ocr_confiance, ocr_filigrane_texte,
ocr_filigrane_present, scene_coupes, scene_coupes_par_min.

Les keyframes PNG sont écrites sur disque (chemin déductible :
{media_dir}/fiches/keyframes/{canal}_{message_id}_kf_*.png) — pas de champ
JSONL pour les compter, l'idempotence se fait via glob sur le dossier.

Pipeline par message vidéo :
  1. Keyframes — extraction via ffmpeg (1 frame / 10s)
  2. OCR — EasyOCR (ru/uk/en) sur keyframes avec prétraitement Pillow + OpenCV
  3. SceneDetect — PySceneDetect ContentDetector pour les coupes

Pour les photos : OCR seul (pas de keyframes ni scenedetect).

Options CLI : --skip-keyframes, --skip-ocr, --skip-scenedetect,
              --ocr-confidence-threshold, --scene-threshold

Dépendances : ffmpeg, easyocr, scenedetect, opencv-python, Pillow
"""

import glob as globmod
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))

from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    write_jsonl,
    mettre_a_jour_fiche,
    filtrer_eligibles,
    SuiviProgression,
)

WATERMARK_RE = re.compile(
    r"@\w+|t\.me/|magyar|мадяр|414|бригад|dji|gopro|©|telegram",
    re.IGNORECASE,
)


def lire_image_safe(path: str):
    """Lit une image de manière sécurisée (chemins avec accents/unicode).

    Entrée : path — chemin vers le fichier image
    Sortie : ndarray BGR (OpenCV), ou None si lecture impossible
    """
    import numpy as np
    import cv2

    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ── EasyOCR ──────────────────────────────────────────────────────────────────

_easyocr_reader = None


def charger_easyocr(log: logging.Logger, languages=None, gpu=True):
    """Charge EasyOCR une seule fois (GPU avec fallback CPU)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
        except ImportError:
            log.error("EasyOCR introuvable. pip install easyocr")
            sys.exit(1)

        langs = languages or ["ru", "uk", "en"]
        log.info(f"  Chargement EasyOCR ({', '.join(langs)})...")
        try:
            _easyocr_reader = easyocr.Reader(langs, gpu=gpu)
        except Exception as e:
            log.warning(f"  GPU indisponible pour EasyOCR, fallback CPU : {e}")
            _easyocr_reader = easyocr.Reader(langs, gpu=False)
        log.info("  EasyOCR chargé")
    return _easyocr_reader


# ── Prétraitement image ─────────────────────────────────────────────────────
# Pipeline : Pillow (débruitage, netteté, contraste) → OpenCV (CLAHE, upscale).
# Améliore significativement la qualité OCR sur les keyframes compressées.

def pretraiter_image_pour_ocr(image_path: str):
    """Prétraite une image pour améliorer la qualité OCR.

    Entrée : image_path — chemin vers l'image (keyframe ou photo)
    Sortie : ndarray grayscale prétraité (OpenCV), ou None si échec lecture
    """
    import cv2
    import numpy as np

    img = lire_image_safe(image_path)
    if img is None:
        return None

    # Étape 1 : Pillow
    try:
        from PIL import Image, ImageFilter, ImageEnhance, ImageOps

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        # Débruitage doux préservant les arêtes du texte
        pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
        # Renforce les arêtes du texte cyrillique avant l'OCR
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        # Étire l'histogramme en ignorant 1 % des extrêmes (robuste aux outliers de pixels)
        pil_img = ImageOps.autocontrast(pil_img, cutoff=1)
        # Gains empiriques : sharpness 1.3 et contrast 1.2 — calés sur les keyframes Magyar
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except ImportError:
        pass
    except Exception:
        pass

    # Étape 2 : OpenCV — CLAHE + gaussian blur + morpho close + upscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contraste adaptatif local sur tuiles 8×8 (robuste à un éclairage non-uniforme)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # On upscale les petites images : EasyOCR perd les caractères fins sous 800px de large
    height, width = enhanced.shape
    if width < 800:
        scale = 2 if width < 400 else 1.5
        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return enhanced


# ── OCR sur une frame ────────────────────────────────────────────────────────

def est_region_filigrane(bbox, img_height: int, margin: float = 0.15) -> bool:
    """Vérifie si le centroid Y du bbox est dans le top/bottom margin de l'image.

    Entrée : bbox — liste de 4 points [[x,y], ...], img_height — hauteur en pixels,
             margin — fraction de l'image considérée comme zone watermark (0.15 = 15%)
    Sortie : bool
    """
    try:
        ys = [pt[1] for pt in bbox]
        centroid_y = sum(ys) / len(ys)
        return centroid_y < img_height * margin or centroid_y > img_height * (1 - margin)
    except (TypeError, IndexError, ZeroDivisionError):
        return False


def est_filigrane_texte(text: str) -> bool:
    """Vérifie si le texte matche un pattern de watermark."""
    return bool(WATERMARK_RE.search(text))


def ocr_une_frame(
    image_path: str,
    reader,
    confidence_threshold: float,
    log: logging.Logger,
    frame_index: int = 0,
) -> list:
    """OCR sur une image prétraitée.

    Entrée : image_path — chemin vers l'image, reader — instance EasyOCR,
             confidence_threshold — seuil de confiance minimum,
             log — logger, frame_index — numéro de la frame dans la vidéo
    Sortie : liste de dicts {bbox, text, confidence, frame_index, is_watermark}
    """
    pren_traites = pretraiter_image_pour_ocr(image_path)
    if pren_traites is None:
        return []

    try:
        detections = reader.readtext(pren_traites)
    except (RuntimeError, MemoryError) as e:
        # Fatal : OOM GPU ou modèle effondré — log.error pour ne pas confondre avec "pas de texte détecté".
        log.error(f"EasyOCR FATAL sur {os.path.basename(image_path)} : {e}")
        return []
    except Exception as e:
        log.warning(f"EasyOCR erreur récupérable sur {os.path.basename(image_path)} : {e}")
        return []

    if not detections:
        return []

    img_height = int(pren_traites.shape[0])
    results = []

    for bbox, text, conf in detections:
        conf = float(conf)
        if conf < confidence_threshold:
            continue

        is_wm = bool(
            est_region_filigrane(bbox, img_height) or est_filigrane_texte(text)
        )

        results.append(
            {
                "bbox": bbox,
                "text": str(text).strip(),
                "confidence": float(round(conf, 4)),
                "frame_index": int(frame_index),
                "is_watermark": is_wm,
            }
        )

    return results



# ── Fusion OCR multi-frames ─────────────────────────────────────────────────


def fusionner_ocr_textes(all_boxes: list) -> dict:
    """Fusionne et déduplique les textes OCR de plusieurs frames.

    Entrée : all_boxes — liste de dicts produits par ocr_une_frame() (toutes frames)
    Sortie : dict {ocr_texte, ocr_confiance, ocr_filigrane_texte, ocr_filigrane_present}
    """
    content_texts = []
    watermark_texts = []
    confidences = []

    seen_content = set()
    seen_watermark = set()

    for box in all_boxes:
        text = box["text"].strip()
        if not text:
            continue
        norm = text.lower()
        if box["is_watermark"]:
            if norm not in seen_watermark:
                seen_watermark.add(norm)
                watermark_texts.append(text)
        else:
            if norm not in seen_content:
                seen_content.add(norm)
                content_texts.append(text)
                confidences.append(box["confidence"])

    ocr_texte = " ".join(content_texts)
    ocr_filigrane_texte = " ".join(watermark_texts)
    ocr_confiance = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "ocr_texte": ocr_texte,
        "ocr_confiance": ocr_confiance,
        "ocr_filigrane_texte": ocr_filigrane_texte,
        "ocr_filigrane_present": len(watermark_texts) > 0,
    }


# ── Extraction keyframes (ffmpeg) ───────────────────────────────────────────


def extraire_keyframes(video_path: str, channel: str, mid: int,
                      keyframes_dir: str, log: logging.Logger,
                      fps: str = "1/10", compress_level: int = 3,
                      timeout: int = 300) -> int:
    """Extrait des keyframes PNG via ffmpeg à intervalle fixe.

    Entrée : video_path — chemin vidéo, channel — str, mid — id message,
             keyframes_dir — dossier de sortie, log — logger,
             fps — filtre ffmpeg (défaut "1/10" = 1 frame toutes les 10s),
             compress_level — compression PNG (0-9), timeout — secondes max
    Sortie : nombre de keyframes créées (int)
    """
    os.makedirs(keyframes_dir, exist_ok=True)
    pattern = os.path.join(keyframes_dir, f"{channel}_{mid}_kf_%03d.png")

    # On purge les keyframes existantes de ce message avant ré-extraction.
    # ffmpeg -y n'écrase que les fichiers de même nom : sans ce nettoyage, des
    # keyframes orphelines d'un ancien run (autre format %04d, ou vidéo alors
    # plus longue) survivraient et seraient recomptées par l'OCR et le pHash.
    for ext in ("png", "jpg", "jpeg"):
        for ancien in globmod.glob(os.path.join(keyframes_dir, f"{channel}_{mid}_kf_*.{ext}")):
            try:
                os.remove(ancien)
            except OSError as e:
                log.warning(f"  msg {mid} : purge keyframe {ancien} échec : {e}")

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-compression_level", str(compress_level),
        "-vsync", "vfr",
        pattern,
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.warning(f"  msg {mid} : ffmpeg keyframes timeout ({timeout}s)")
        return 0
    except OSError as e:
        log.warning(f"  msg {mid} : ffmpeg keyframes erreur : {e}")
        return 0

    created = globmod.glob(os.path.join(keyframes_dir, f"{channel}_{mid}_kf_*.png"))
    return len(created)


# ── SceneDetect ──────────────────────────────────────────────────────────────


def detecter_scenes(video_path: str, threshold: float,
                  log: logging.Logger, mid: int) -> tuple[int, float]:
    """Détecte les changements de scène via PySceneDetect ContentDetector.

    Entrée : video_path — chemin vidéo, threshold — seuil ContentDetector,
             log — logger, mid — id message (pour les warnings)
    Sortie : tuple (n_cuts int, cuts_per_min float)
    """
    try:
        from scenedetect import detect, ContentDetector
    except ImportError:
        log.warning("  PySceneDetect introuvable, skip scenedetect")
        return 0, 0.0

    try:
        scenes = detect(video_path, ContentDetector(threshold=threshold))
        cuts = max(len(scenes) - 1, 0)

        if scenes:
            last_end = scenes[-1][1].get_seconds()
            cuts_per_min = round(cuts / (last_end / 60), 2) if last_end > 0 else 0.0
        else:
            cuts_per_min = 0.0

        return cuts, cuts_per_min

    except Exception as e:
        log.warning(f"  msg {mid} : scenedetect erreur : {e}")
        return 0, 0.0


# ── Traitement vidéo ────────────────────────────────────────────────────────


def traiter_video(msg: dict, media_file, media_dir,
                  log: logging.Logger, reader,
                  overwrite: bool = False,
                  skip_keyframes: bool = False,
                  skip_ocr: bool = False,
                  skip_scenedetect: bool = False,
                  ocr_confidence_threshold: float = 0.15,
                  scene_threshold: float = 27.0,
                  keyframe_fps: str = "1/10",
                  keyframe_compress_level: int = 3) -> tuple[dict, dict]:
    """Traite une vidéo : keyframes + OCR + scenedetect.

    Entrée : msg — dict message, media_file — Path fichier, media_dir — Path racine corpus,
             log — logger, reader — instance EasyOCR (ou None si skip_ocr),
             overwrite/skip_* — flags de contrôle du pipeline
    Sortie : tuple (nouveaux_champs, champs_fiche) — dicts à merger dans le JSONL et la fiche
    """
    mid = msg.get("message_id")
    channel = msg.get("canal", "robert_magyar")
    nouveaux_champs = {}
    champs_fiche = {}
    keyframes_dir = str(media_dir / "fiches" / "keyframes")
    kf_pattern_png = os.path.join(keyframes_dir, f"{channel}_{mid}_kf_*.png")
    kf_pattern_jpg = os.path.join(keyframes_dir, f"{channel}_{mid}_kf_*.jpg")

    # ── 1. KEYFRAMES ──
    # Idempotence : on regarde si des keyframes existent déjà sur disque.
    # Pas de champ JSONL — le dossier de sortie sert de sentinelle.
    kf_count = 0
    if not skip_keyframes:
        kf_existantes = globmod.glob(kf_pattern_png) + globmod.glob(kf_pattern_jpg)
        if overwrite or not kf_existantes:
            try:
                kf_count = extraire_keyframes(
                    str(media_file), channel, mid, keyframes_dir, log,
                    fps=keyframe_fps, compress_level=keyframe_compress_level,
                )
            except Exception as e:
                log.warning(f"  msg {mid} : keyframes exception : {e}")
                kf_count = 0
        else:
            kf_count = len(kf_existantes)

    # ── 2. OCR ──
    if not skip_ocr and (overwrite or "ocr_texte" not in msg):
        try:
            kf_files = sorted(globmod.glob(kf_pattern_jpg) + globmod.glob(kf_pattern_png))

            all_boxes = []
            for idx, kf_path in enumerate(kf_files):
                boxes = ocr_une_frame(
                    kf_path, reader, ocr_confidence_threshold, log, frame_index=idx
                )
                all_boxes.extend(boxes)

            merged = fusionner_ocr_textes(all_boxes)
            nouveaux_champs.update(merged)
            champs_fiche.update(merged)

            serializable_boxes = []
            for box in all_boxes:
                sb = dict(box)
                sb["bbox"] = [[float(c) for c in pt] for pt in box["bbox"]]
                serializable_boxes.append(sb)
            champs_fiche["ocr_boxes"] = serializable_boxes

        except Exception as e:
            log.warning(f"  msg {mid} : OCR exception : {e}")
            nouveaux_champs["ocr_texte"] = ""
            nouveaux_champs["ocr_confiance"] = 0.0
            nouveaux_champs["ocr_filigrane_texte"] = ""
            nouveaux_champs["ocr_filigrane_present"] = False
            champs_fiche.update(nouveaux_champs)
            champs_fiche["ocr_boxes"] = []

    # ── 3. SCENEDETECT ──
    if not skip_scenedetect and (overwrite or "scene_coupes" not in msg):
        try:
            cuts, cuts_per_min = detecter_scenes(
                str(media_file), scene_threshold, log, mid
            )
            nouveaux_champs["scene_coupes"] = cuts
            nouveaux_champs["scene_coupes_par_min"] = cuts_per_min
            champs_fiche["scene_coupes"] = cuts
            champs_fiche["scene_coupes_par_min"] = cuts_per_min
        except Exception as e:
            log.warning(f"  msg {mid} : scenedetect exception : {e}")
            nouveaux_champs["scene_coupes"] = 0
            nouveaux_champs["scene_coupes_par_min"] = 0.0
            champs_fiche["scene_coupes"] = 0
            champs_fiche["scene_coupes_par_min"] = 0.0

    # On expose le compte de keyframes via la clé privée _kf_count uniquement
    # pour l'affichage de progression — retiré avant l'écriture JSONL côté main.
    nouveaux_champs["_kf_count"] = kf_count

    return nouveaux_champs, champs_fiche


# ── Traitement photo ────────────────────────────────────────────────────────


def traiter_photo(msg: dict, media_file,
                  log: logging.Logger, reader,
                  overwrite: bool = False,
                  skip_ocr: bool = False,
                  ocr_confidence_threshold: float = 0.15) -> tuple[dict, dict]:
    """OCR sur une photo.

    Entrée : msg — dict message, media_file — Path fichier, log — logger,
             reader — instance EasyOCR, overwrite/skip_ocr — flags de contrôle
    Sortie : tuple (nouveaux_champs, champs_fiche) — dicts à merger dans le JSONL et la fiche
    """
    mid = msg.get("message_id")
    nouveaux_champs = {}
    champs_fiche = {}

    if skip_ocr or (not overwrite and "ocr_texte" in msg):
        return nouveaux_champs, champs_fiche

    try:
        boxes = ocr_une_frame(
            str(media_file), reader, ocr_confidence_threshold, log, frame_index=0
        )
        merged = fusionner_ocr_textes(boxes)
        nouveaux_champs.update(merged)
        champs_fiche.update(merged)

        serializable_boxes = []
        for box in boxes:
            sb = dict(box)
            sb["bbox"] = [[float(c) for c in pt] for pt in box["bbox"]]
            serializable_boxes.append(sb)
        champs_fiche["ocr_boxes"] = serializable_boxes

    except Exception as e:
        log.warning(f"  msg {mid} : OCR photo exception : {e}")
        nouveaux_champs["ocr_texte"] = ""
        nouveaux_champs["ocr_confiance"] = 0.0
        nouveaux_champs["ocr_filigrane_texte"] = ""
        nouveaux_champs["ocr_filigrane_present"] = False
        champs_fiche.update(nouveaux_champs)
        champs_fiche["ocr_boxes"] = []

    return nouveaux_champs, champs_fiche


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = creer_parser_base(
        "Enrichissement vision : keyframes (ffmpeg), OCR (EasyOCR), scenedetect",
        has_media_dir=True,
    )
    parser.add_argument("--skip-keyframes", action="store_true", help="Skip extraction keyframes")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR")
    parser.add_argument("--skip-scenedetect", action="store_true", help="Skip PySceneDetect")
    parser.add_argument(
        "--ocr-confidence-threshold", type=float, default=0.15,
        help="Seuil de confiance OCR minimum (défaut: 0.15)",
    )
    parser.add_argument(
        "--scene-threshold", type=float, default=27.0,
        help="Seuil ContentDetector pour scenedetect (défaut: 27.0)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger("vision", cfg=cfg)
    save_every = 50

    ocr_conf_threshold = args.ocr_confidence_threshold
    scene_threshold = args.scene_threshold
    keyframe_fps = "1/10"          # 1 frame toutes les 10 secondes
    keyframe_compress_level = 3    # PNG level 0-9 (équilibre taille/vitesse)
    easyocr_cfg = cfg.get("models", {}).get("easyocr", {})
    ocr_languages = easyocr_cfg.get("languages", ["ru", "uk", "en"])
    ocr_gpu = easyocr_cfg.get("gpu", True)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else input_path.parent

    # ── Vérification ffmpeg ──
    if not args.skip_keyframes:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except FileNotFoundError:
            log.error("ffmpeg introuvable. Installez ffmpeg.")
            sys.exit(1)

    # ── Lecture JSONL ──
    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    messages = read_jsonl(input_path)
    total = len(messages)
    log.info(f"Corpus : {total} messages")

    # ── Filtrer les messages éligibles ──
    filtre_ids = set(args.ids) if args.ids else None
    eligibles = []
    all_indices = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video", "photo"],
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    for i in all_indices:
        msg = messages[i]
        media_type = msg.get("media_type")

        if media_type == "video":
            # Pour les keyframes : pas de champ JSONL — on vérifie via glob plus tard.
            # On considère qu'il faut traiter si OCR ou scenedetect manquent,
            # ou si on demande explicitement les keyframes (le glob filtrera).
            besoin_kf = not args.skip_keyframes
            besoin_ocr = not args.skip_ocr and (args.overwrite or "ocr_texte" not in msg)
            besoin_sd = not args.skip_scenedetect and (args.overwrite or "scene_coupes" not in msg)
            if not (besoin_kf or besoin_ocr or besoin_sd):
                continue
        elif media_type == "photo":
            if args.skip_ocr:
                continue
            if not args.overwrite and "ocr_texte" in msg:
                continue

        eligibles.append(i)

    if args.limit:
        eligibles = eligibles[:args.limit]

    n_eligibles = len(eligibles)
    log.info(f"Messages éligibles (vidéo + photo) : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # ── Chargement EasyOCR (si nécessaire) ──
    reader = None
    if not args.skip_ocr:
        reader = charger_easyocr(log, languages=ocr_languages, gpu=ocr_gpu)

    tracker = SuiviProgression(n_eligibles, label="vision")
    n_traites = 0
    n_erreurs = 0

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            media_type = msg.get("media_type")
            media_chemin_rel = msg.get("media_chemin")

            if not media_chemin_rel:
                continue

            media_file = media_dir / media_chemin_rel

            if not media_file.is_file():
                log.warning(f"msg {mid}\t{media_chemin_rel}\tfichier manquant")
                n_erreurs += 1
                tracker.avancer(rank, mid, "fichier manquant")
                continue

            # On initialise avant le try : si traiter_video/photo lève une
            # exception, ces variables restent définies pour la progression.
            nouveaux_champs, champs_fiche = {}, {}

            try:
                if media_type == "video":
                    nouveaux_champs, champs_fiche = traiter_video(
                        msg, media_file, media_dir, log, reader,
                        overwrite=args.overwrite,
                        skip_keyframes=args.skip_keyframes,
                        skip_ocr=args.skip_ocr,
                        skip_scenedetect=args.skip_scenedetect,
                        ocr_confidence_threshold=ocr_conf_threshold,
                        scene_threshold=scene_threshold,
                        keyframe_fps=keyframe_fps,
                        keyframe_compress_level=keyframe_compress_level,
                    )
                else:  # photo
                    nouveaux_champs, champs_fiche = traiter_photo(
                        msg, media_file, log, reader,
                        overwrite=args.overwrite,
                        skip_ocr=args.skip_ocr,
                        ocr_confidence_threshold=ocr_conf_threshold,
                    )

                # On retire la clé privée _kf_count (purement pour l'affichage)
                # avant d'écrire dans le JSONL et la fiche.
                kf_count_progress = nouveaux_champs.pop("_kf_count", None)

                if nouveaux_champs:
                    msg.update(nouveaux_champs)
                    n_traites += 1
                    mettre_a_jour_fiche(msg, champs_fiche, media_dir / "fiches", overwrite=args.overwrite)

            except Exception as e:
                log.warning(f"msg {mid}\t{media_chemin_rel}\texception : {e}")
                n_erreurs += 1
                kf_count_progress = None

            # Progression
            parts = []
            if kf_count_progress is not None:
                parts.append(f"kf={kf_count_progress}")
            if "ocr_texte" in nouveaux_champs:
                ocr_len = len(nouveaux_champs["ocr_texte"])
                parts.append(f"ocr={ocr_len}c")
            if "scene_coupes" in nouveaux_champs:
                parts.append(f"cuts={nouveaux_champs['scene_coupes']}")
            if nouveaux_champs.get("ocr_filigrane_present"):
                parts.append("wm")
            desc = ", ".join(parts) if parts else "ok"
            tracker.avancer(rank, mid, desc)

            # Sauvegarde intermédiaire
            if n_traites % save_every == 0 and n_traites > 0:
                write_jsonl(messages, output_path)
                log.info(f"  Sauvegarde intermédiaire ({n_traites} traités)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.resumer(errors=n_erreurs, skipped=n_eligibles - n_traites - n_erreurs)


if __name__ == "__main__":
    main()
