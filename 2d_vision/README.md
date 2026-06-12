# Enrichissement visuel

Analyse de l'image des vidéos et photos du corpus. Chaque sous-module est autonome, avec son propre README et son propre `requirements.txt`, et écrit ses champs dans le JSONL et/ou un CSV dans son dossier `results/`. Tous dépendent des métadonnées ffprobe (`2a_metadonnees`) et sont parallélisables avec la transcription et la traduction.

`keyframes/` extrait les images fixes des vidéos (ffmpeg), y applique l'OCR cyrillique (EasyOCR) et la détection de changements de plan (PySceneDetect), et calcule un pHash par keyframe. `visages/` détecte la présence de Magyar et compte les individus distincts (InsightFace et DBSCAN). `blasons/` détecte les logos de brigade en watermark (SIFT et RANSAC). `clip/` est une classification zero-shot conservée pour traçabilité : ses résultats, peu discriminants sur l'imagerie de guerre, ne sont pas retenus et n'écrivent rien dans le JSONL.

L'ordre d'exécution suit les dépendances : `keyframes/keyframer.py` d'abord, puisque les autres détections consomment ses keyframes, puis `scene_detect.py`, `visages/detect_magyar.py` et `blasons/detect_blasons.py` dans n'importe quel ordre. `keyframes/phash_keyframes.py` se lance après l'extraction des keyframes.

Les images de référence et les embeddings sont gitignorés (`visages/references/`, `blasons/references/`, tous les `results/`). Pour rejouer les détections, fournir ses propres références : `visages/build_reference.py` reconstruit l'embedding facial à partir d'un dossier d'images, et `blasons/README.md` décrit les crops attendus.
