# 2d_vision — Enrichissement visuel (E4)

Étape E4 du pipeline : analyse de l'image des vidéos et photos du corpus.
Chaque sous-module est **autonome** (son propre `README.md`, `requirements.txt`
et venv possible) et écrit ses champs dans le JSONL et/ou un CSV dans son
`results/`. Tous dépendent de E1 (`2a_metadonnees` — `media_chemin`, `duree`)
et sont parallélisables avec E2/E3.

## Sous-modules

| Dossier | Rôle | Champs JSONL / sortie | README |
|---------|------|-----------------------|--------|
| `keyframes/` | Extraction keyframes (ffmpeg) + OCR cyrillique (EasyOCR) + détection de plans (PySceneDetect) ; pHash par keyframe | `ocr_texte`, `ocr_confiance`, `ocr_filigrane_*`, `scene_coupes*`, `scene_duree_moyenne` ; `phash_keyframes.csv` | [keyframes/README.md](keyframes/README.md) |
| `visages/` | Détection de Magyar (InsightFace + DBSCAN) | `visages_magyar_*`, `visages_unique`, `visages_densite` | [visages/README.md](visages/README.md) |
| `blasons/` | Détection des logos de brigade (SIFT + RANSAC) | `blason_present`, `blason_detecte`, `blason_zone` | [blasons/README.md](blasons/README.md) |
| `clip/` | Classification zero-shot CLIP — ⚠️ limite méthodologique, n'écrit plus dans le JSONL (CSV seulement) | `clip/results/*.csv` | [clip/README.md](clip/README.md) |

## Ordre d'exécution

```
E4   keyframes/keyframer.py      keyframes PNG + OCR + SceneDetect
E4b  keyframes/scene_detect.py   scene_duree_moyenne (standalone, fallback histogramme)
E4c  visages/detect_magyar.py    InsightFace + DBSCAN
E4d  blasons/detect_blasons.py   SIFT + RANSAC
E4e  clip/clip_classify.py       CLIP zero-shot (CSV uniquement — non retenu)
```

`keyframes/phash_keyframes.py` (pHash par keyframe) est indépendant et se lance
après que les keyframes ont été extraites.

## Données de référence (non versionnées)

Les images de référence et embeddings sont **gitignorés** (vie privée + poids) :
`visages/references/magyar/`, `visages/references/*.npy`, `blasons/references/`,
tous les `*/results/` et `*/results_birds/`. Pour rejouer la détection, fournir
ses propres références — `visages/build_reference.py` reconstruit l'embedding
facial à partir d'un dossier d'images.

## Note — HUD_FPV/

`HUD_FPV/` contient des notes préliminaires (`NOTES.md`) pour une détection
d'overlays Betaflight non démarrée. Dossier gitignoré, hors périmètre publié.
