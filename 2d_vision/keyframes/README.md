# 2d_vision/keyframes — Keyframes, OCR, détection de scènes

Extraction de keyframes (ffmpeg), OCR cyrillique (EasyOCR) et détection de changements de plan (PySceneDetect) sur les vidéos et photos du corpus.

## Dépendances

```bash
pip install -r requirements.txt
```

Requiert ffmpeg installé système. EasyOCR télécharge les modèles (ru, uk, en) au premier lancement.

## Utilisation

```bash
# Corpus complet. Relance après interruption : --input = --output (skip les déjà traités)
python keyframer.py --input messages_whisper.jsonl --output messages_computervision.jsonl
#   --limit 5      test sur les 5 premiers messages
#   --overwrite    force le retraitement
```

## Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `scene_coupes` | int | Nombre de changements de plan |
| `scene_coupes_par_min` | float | Rythme de coupe (cuts/min) |
| `ocr_texte` | string | Texte détecté par OCR (sans filigrane) |
| `ocr_confiance` | float | Confiance OCR moyenne |
| `ocr_filigrane_texte` | string | Texte de filigrane (watermark) détecté |
| `ocr_filigrane_present` | bool | Présence d'un filigrane |

Les keyframes sont extraites dans `fiches/keyframes/` à intervalle fixe (1 frame / 10 s, constante `keyframe_fps` en tête de `main()`) par `keyframer.py`. Aucun champ JSONL ne les compte : l'idempotence se fait par glob sur le dossier (`{canal}_{message_id}_kf_*.png`).

---

## scene_detect.py — Detection de scenes standalone

Script autonome de detection de changements de plan, avec fallback `HistogramDetector` pour les videos courtes sur-decoupees par `ContentDetector`.

Remplace la fonction `detect_scenes()` de `keyframer.py` tout en la laissant intacte.

### Utilisation

```bash
# Corpus complet (défauts : --input depuis config.yaml, --output messages_scenedetect.jsonl)
python scene_detect.py
#   --limit 10              test sur les 10 premieres videos
#   --ids 42 138 256        messages specifiques
#   --overwrite             retraite les videos deja faites
#   --aggregate-only        graphique seul depuis le CSV existant
#   --csv CHEMIN            CSV de sortie (defaut : results/scene_detection.csv)
#   --threshold 27.0        seuil ContentDetector
#   --min-scene-len 15      longueur min de scene (frames)
```

### Fallback HistogramDetector

Si `ContentDetector` detecte > 50 coupes sur une video < 30 s (typique des compilations FPV rapides), le script bascule automatiquement sur `HistogramDetector(threshold=0.05)` pour eviter le sur-decoupage.

### Sorties

- `results/scene_detection.csv` — une ligne par video (`message_id, date, phase, duration_sec, n_scenes, cuts_per_minute, avg_scene_duration`) — colonnes CSV conservees telles quelles
- `results/scene_monthly.png` — courbe mensuelle des coupes/min avec bandes de phases colorees
- JSONL enrichi : `scene_coupes`, `scene_coupes_par_min`, `scene_duree_moyenne`
- Fiches individuelles mises a jour

### Idempotence

- Charge le CSV existant au demarrage, skip les `message_id` deja traites
- Verifie aussi les champs JSONL (`scene_coupes`, `scene_coupes_par_min`)
- Sauvegarde intermediaire tous les 50 messages

---

## phash_keyframes.py — pHash par keyframe

Calcule un pHash 64-bit (perceptual hash, `imagehash.phash`) pour **chaque**
keyframe d'une vidéo et pour chaque photo. À distinguer du champ JSONL
`perceptual_hash`, calculé par le scraper sur une **seule** frame par message
(la vignette) : ici la granularité est par keyframe, ce qui est la base des
analyses de recyclage visuel (`3b_stats_R/.../55_phash_recyclage.R`).

Script indépendant, destiné à être intégré à `keyframer.py` une fois validé.

```bash
# Corpus complet (input = config.yaml paths.jsonl_clean)
python phash_keyframes.py
#   --ids 8 1080            messages spécifiques (test)
#   --csv /tmp/test.csv     CSV de sortie alternatif
#   --overwrite             recalcul complet
```

**Sortie** : CSV `4_data_et_viz/phash_keyframes.csv` (5 colonnes :
`message_id, media_type, keyframe_index, keyframe_path, phash`). Idempotent :
les couples `(message_id, keyframe_path)` déjà présents sont skippés
(`--overwrite` pour recalcul complet). N'écrit **pas** dans le JSONL.
