# 2d_vision — Keyframes, OCR, détection de scènes

Extraction de keyframes (ffmpeg), OCR cyrillique (EasyOCR) et détection de changements de plan (PySceneDetect) sur les vidéos et photos du corpus.

## Dépendances

```bash
pip install -r requirements.txt
```

Requiert ffmpeg installé système. EasyOCR télécharge les modèles (ru, uk, en) au premier lancement.

## Usage

```bash
# Lancer sur tout le corpus
python keyframer.py --input /chemin/messages_whisper.jsonl --output /chemin/messages_computervision.jsonl

# Tester sur 5 messages
python keyframer.py --input messages_whisper.jsonl --output messages_computervision.jsonl --limit 5

# Relancer après interruption
python keyframer.py --input messages_computervision.jsonl --output messages_computervision.jsonl

# Forcer le retraitement
python keyframer.py --input messages_computervision.jsonl --output messages_computervision.jsonl --overwrite
```

## Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `keyframes_dir` | string | Chemin du dossier keyframes |
| `keyframes_count` | int | Nombre de keyframes extraites |
| `scene_cuts` | int | Nombre de changements de plan |
| `scene_cuts_per_min` | float | Rythme de coupe (cuts/min) |
| `ocr_text` | string | Texte détecté par OCR |
| `ocr_confidence` | float | Confiance OCR moyenne |
| `ocr_watermark_text` | string | Texte de watermark détecté |
| `ocr_has_watermark` | bool | Présence d'un watermark |

Les keyframes sont extraites dans `fiches/keyframes/` à intervalle fixe (`1/10 s`, config `keyframe_fps`) par `keyframer.py`.

---

## scene_detect.py — Detection de scenes standalone

Script autonome de detection de changements de plan, avec fallback `HistogramDetector` pour les videos courtes sur-decoupees par `ContentDetector`.

Remplace la fonction `detect_scenes()` de `keyframer.py` tout en la laissant intacte.

### Usage

```bash
# Tester sur 10 videos
python scene_detect.py --limit 10

# Corpus complet
python scene_detect.py

# Messages specifiques
python scene_detect.py --ids 42 138 256

# Retraiter tout
python scene_detect.py --overwrite

# Graphique seul depuis CSV existant
python scene_detect.py --aggregate-only
```

### Options CLI

| Argument | Defaut | Description |
|----------|--------|-------------|
| `--input` | config JSONL | JSONL source |
| `--output` | `messages_scenedetect.jsonl` | JSONL enrichi |
| `--csv` | `results/scene_detection.csv` | CSV de sortie |
| `--threshold` | 27.0 | Seuil ContentDetector |
| `--min-scene-len` | 15 | Longueur min scene (frames) |
| `--limit` | aucune | N videos max |
| `--ids` | tous | Message IDs specifiques |
| `--overwrite` | false | Retraiter les videos deja faites |
| `--aggregate-only` | false | Ne produire que le graphique |

### Fallback HistogramDetector

Si `ContentDetector` detecte > 50 coupes sur une video < 30 s (typique des compilations FPV rapides), le script bascule automatiquement sur `HistogramDetector(threshold=0.05)` pour eviter le sur-decoupage.

### Sorties

- `results/scene_detection.csv` — une ligne par video (`message_id, date, phase, duration_sec, n_scenes, cuts_per_minute, avg_scene_duration`)
- `results/scene_monthly.png` — courbe mensuelle des coupes/min avec bandes de phases colorees
- JSONL enrichi : `scene_cuts`, `scene_cuts_per_min`, `scene_avg_duration`
- Fiches individuelles mises a jour

### Idempotence

- Charge le CSV existant au demarrage, skip les `message_id` deja traites
- Verifie aussi les champs JSONL (`scene_cuts`, `scene_cuts_per_min`)
- Sauvegarde intermediaire tous les 50 messages

---

## Sous-modules

| Dossier | Description |
|---------|-------------|
| `clip/` | Classification zero-shot CLIP (keyframes) — voir `clip/README.md` |
| `faces/` | Detection de Magyar (InsightFace) — voir `faces/README.md` |
