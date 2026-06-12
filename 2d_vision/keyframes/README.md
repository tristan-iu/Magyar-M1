# Keyframes, OCR et détection de scènes

Extraction d'images fixes (ffmpeg), OCR cyrillique (EasyOCR) et détection de changements de plan (PySceneDetect) sur les vidéos et photos du corpus. Trois scripts : `keyframer.py` est le pipeline principal (keyframes, OCR, coupes), `scene_detect.py` une détection de scènes autonome avec fallback anti-surdécoupage, et `phash_keyframes.py` calcule un hash perceptuel par keyframe.

## Installation

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
```

EasyOCR télécharge ses modèles (ru, uk, en) au premier lancement. Un GPU CUDA est recommandé pour l'OCR, avec fallback CPU automatique.

## Utilisation

```bash
# 1. Pipeline principal. Relance après interruption : --input = --output
python keyframer.py --input messages_whisper.jsonl --output messages_computervision.jsonl
#   --limit 5      test sur les 5 premiers messages
#   --overwrite    force le retraitement
#   --skip-keyframes / --skip-ocr / --skip-scenedetect   désactive une étape

# 2. Détection de scènes autonome (CSV + graphique mensuel)
python scene_detect.py
#   --ids 42 138 256        messages spécifiques
#   --threshold 27.0        seuil ContentDetector (défaut : config.yaml)
#   --min-scene-len 15      longueur min de scène (frames)
#   --aggregate-only        graphique seul depuis le CSV existant

# 3. pHash par keyframe (après extraction des keyframes)
python phash_keyframes.py
#   --ids 8 1080            messages spécifiques (test)
#   --csv /tmp/test.csv     CSV de sortie alternatif
#   --overwrite             recalcul complet
```

L'idempotence des keyframes se fait par glob sur le dossier de sortie (aucun champ JSONL ne les compte), celle de l'OCR et des scènes par présence des champs dans le JSONL, celle du pHash par les couples (message, keyframe) déjà présents dans le CSV. Relancer sans `--overwrite` ne traite que le nouveau, avec sauvegarde intermédiaire tous les 50 messages.

## Output

`keyframer.py` écrit les keyframes PNG dans `fiches/keyframes/` au format `{canal}_{message_id}_kf_NNN.png` (intervalle fixe, 1 frame / 10 s) et enrichit le JSONL des champs OCR et scènes. `scene_detect.py` produit en plus `results/scene_detection.csv` (une ligne par vidéo : durée, nombre de scènes, coupes par minute, durée moyenne de scène) et `results/scene_monthly.png` (courbe mensuelle des coupes par minute), et ajoute `scene_duree_moyenne` au JSONL. `phash_keyframes.py` écrit `4_data_et_viz/phash_keyframes.csv` (`message_id, media_type, keyframe_index, keyframe_path, phash`), sans toucher au JSONL.

### Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `ocr_texte` | string | Texte détecté par OCR, hors filigrane |
| `ocr_confiance` | float | Confiance OCR moyenne |
| `ocr_filigrane_texte` | string | Texte de filigrane (watermark) détecté |
| `ocr_filigrane_present` | bool | Présence d'un filigrane |
| `scene_coupes` | int | Nombre de changements de plan |
| `scene_coupes_par_min` | float | Rythme de coupe |
| `scene_duree_moyenne` | float | Durée moyenne d'une scène en secondes (`scene_detect.py`) |

## Méthodologie

**Échantillonnage fixe :** une keyframe toutes les 10 secondes plutôt qu'une détection de frames-clés par contenu. Le pas fixe rend les comparaisons entre vidéos homogènes (le nombre de keyframes est proportionnel à la durée) et le nommage déterministe permet l'idempotence par glob.

**Prétraitement OCR :** les keyframes compressées passent par un pipeline Pillow puis OpenCV avant EasyOCR : débruitage médian, renforcement des arêtes du texte (UnsharpMask), étirement d'histogramme, contraste adaptatif local (CLAHE), et upscale des images de moins de 800 px de large, en dessous desquelles EasyOCR perd les caractères fins.

**Filigranes :** un texte OCR est classé filigrane s'il matche un motif connu (handles, t.me, marques de drone...) ou si son centroïde se situe dans les 15 % supérieurs ou inférieurs de l'image, zones habituelles des watermarks. Les filigranes sont stockés à part pour ne pas polluer le texte de contenu.

**Fallback HistogramDetector :** sur les clips courts à montage très rapide, ContentDetector surdécoupe, les changements de luminosité brefs passant pour des coupes. Si plus de 50 coupes sont détectées sur une vidéo de moins de 30 secondes, `scene_detect.py` rejoue la détection avec HistogramDetector, plus robuste à la luminosité mais moins sensible aux coupes franches. Les seuils vivent dans `config.yaml`.

**pHash par keyframe :** le champ `perceptual_hash` du JSONL, calculé par le scraper, ne couvre qu'une seule frame par message (la vignette). Le hash par keyframe donne la granularité nécessaire aux analyses de recyclage visuel : retrouver un même plan réutilisé dans plusieurs compilations.
