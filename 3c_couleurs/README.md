# Analyse colorimétrique HSV

Statistiques de couleur (espace HSV) sur les keyframes de chaque vidéo et sur les photos du corpus. Trois niveaux d'analyse : agrégats scalaires par message (CSV tidy pour R), histogrammes 3D complets (NPZ pour PCA et similarité), et matrice de similarité temporelle. Se place après l'extraction des keyframes, en parallèle des autres détections visuelles ; le JSONL n'est pas touché.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Extraction stats + histogrammes (idempotent)
python couleur_batch.py
#   --limit 50              test rapide
#   --ids 42 138            messages spécifiques
#   --overwrite             recalcule tout

# 2. Projection PCA + clustering K-means sur les histogrammes
python couleur_espace.py
#   --n-components 5 --n-clusters 4

# 3. Matrice de similarité temporelle + heatmap
python couleur_similarite.py
#   --sample 300            sous-échantillonne pour une heatmap lisible

# Figures diagnostiques (scatter PCA, scree plot, distributions par phase)
python couleur_viz.py

# Image moyenne pixel à pixel par phase
python couleur_images_moyennes.py
```

L'idempotence passe par `couleur_stats.csv` : un message déjà présent est skippé tant que `--overwrite` n'est pas passé.

## Output

Les données et figures diagnostiques sont écrites dans `4_data_et_viz/couleurs/` (gitignoré, régénérable) ; les images moyennes du mémoire (`couleur_images_moyennes.py`) vont dans `4_data_et_viz/`. `couleur_batch.py` produit `couleur_stats.csv` (une ligne par message, champs ci-dessous) et `couleur_histogrammes.npz` (un histogramme HSV 3D aplati de 4 096 cellules par message). `couleur_espace.py` en dérive `couleur_pca.csv` et `couleur_clusters.csv`, `couleur_similarite.py` une matrice cosinus ordonnée par date (`couleur_similarite.npz` et sa heatmap PNG), `couleur_viz.py` les figures diagnostiques, et `couleur_images_moyennes.py` une image moyenne par phase, déclinée par ratio et par source. Les CSV sont relus côté R par `3b_stats_R/scripts_r/50_couleur_hsv.R`.

### Champs de `couleur_stats.csv`

| Champ | Type | Description |
|-------|------|-------------|
| `message_id` | int | Clé de jointure |
| `hsv_h_mean` | float | Teinte moyenne (OpenCV, 0 à 179) |
| `hsv_s_mean` | float | Saturation moyenne (0 à 255) |
| `hsv_v_mean` | float | Luminosité moyenne (0 à 255) |
| `hsv_s_inter` | float | Écart-type de saturation entre frames |
| `hsv_v_inter` | float | Écart-type de luminosité entre frames |
| `hsv_entropy` | float | Entropie de Shannon de l'histogramme HSV 3D (bits) |
| `hsv_n_frames` | int | Nombre de frames analysées (1 pour les photos) |

## Méthodologie

**Espace HSV :** conversion OpenCV BGR vers HSV, trois axes indépendants et interprétables (teinte, saturation, luminosité). L'analyse est pilotée par la saturation et l'entropie : on cherche la cohérence chromatique de la production, pas la teinte dominante, trop dépendante du sujet filmé.

**Histogramme 3D :** par frame, `cv2.calcHist` sur les trois canaux en 16 par 16 par 16 bins, soit 4 096 cellules normalisées en probabilité. Ce pas est un compromis : assez fin pour distinguer des palettes, assez grossier pour rester robuste au bruit de compression vidéo. Les stats HSV étant invariantes à la résolution spatiale, les frames sont réduites à 256 px de grand côté, ce qui accélère d'un facteur dix sans perte de signal.

**Entropie de Shannon :** calculée sur l'histogramme 3D agrégé du message, maximum théorique de 12 bits. Une frame monochrome tend vers 0, une scène multicolore vers le maximum. Une entropie qui décroît au fil du corpus signale un spectre de couleurs qui se resserre, c'est-à-dire une palette de production contrôlée.

**Métriques inter-frames :** les champs `*_inter` mesurent l'écart-type des moyennes frame à frame au sein d'une même vidéo. Une valeur faible signale une palette homogène (color grading, production contrôlée), une valeur élevée un chaos chromatique (smartphone, contextes variés). Pour les photos, ces champs valent 0 par convention.

**Cas limites :** les frames de moins de 100 pixels non noirs sont skippées (transitions ffmpeg, frames corrompues), et les messages sans keyframes ni photo (texte seul) sont exclus.
