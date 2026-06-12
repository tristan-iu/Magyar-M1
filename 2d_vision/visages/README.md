# Détection de visages (InsightFace)

Détecte le visage de Magyar dans les keyframes du corpus, pour quantifier son effacement physique au fil des trois années. Trois scripts : `build_reference.py` construit un embedding de référence à partir de photos, `detect_magyar.py` compare chaque visage détecté à cette référence et compte les individus distincts, `aggregate_magyar.py` agrège les résultats par message, phase et mois.

## Installation

```bash
pip install -r requirements.txt
```

Un GPU CUDA est recommandé (la détection a tourné sur ~32 000 keyframes). Pour un usage CPU seul, remplacer `onnxruntime-gpu` par `onnxruntime`.

Prérequis : placer 5 à 10 photos nettes de Magyar dans `references/magyar/` (jpg, jpeg, png, webp, bmp), en variant les angles, l'éclairage et l'équipement (avec et sans casque). Ce dossier n'est pas versionné.

## Utilisation

```bash
# 1. Embedding de référence (moyenne L2-normalisée des photos)
python build_reference.py
#   --references-dir references/magyar/    dossier source
#   --output references/magyar_embedding.npy

# 2. Détection sur les keyframes et photos du corpus
python detect_magyar.py
#   --limit 10          test sur 10 messages
#   --ids 42 138 256    messages spécifiques
#   --threshold 0.35    seuil cosine (défaut 0.4, voir Méthodologie)
#   --overwrite         retraite tout

# 3. Agrégation par message, phase et mois + graphique
python aggregate_magyar.py
```

Relancer `detect_magyar.py` sans `--overwrite` ne retraite que les nouveaux messages : le CSV existant est chargé au démarrage et les messages déjà présents sont skippés. Le JSONL est sauvegardé tous les 50 messages.

## Output

`detect_magyar.py` écrit `results/magyar_detection.csv` (une ligne par visage détecté, avec sa position temporelle dans la vidéo) et enrichit le JSONL. `aggregate_magyar.py` en dérive `results/magyar_aggregated.csv` (une ligne par message), `results/magyar_by_phase.csv` (statistiques par phase) et `results/magyar_monthly.png` (courbe mensuelle colorée par phase).

### Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `visages_magyar_ratio` | float | Proportion de frames où Magyar est détecté |
| `visages_magyar_detections` | int | Nombre de détections dans le(s) cluster(s) Magyar |
| `visages_magyar_similarite_max` | float | Similarité cosine maximale avec la référence |
| `visages_unique` | int | Individus distincts dans le message (DBSCAN) |
| `visages_densite` | float | Nombre moyen de visages par frame |

La présence de Magyar se dérive de `visages_magyar_detections > 0`, il n'y a pas de champ booléen dédié.

## Méthodologie

**Pourquoi InsightFace :** le modèle `buffalo_l` (backbone ArcFace) produit des embeddings 512d entraînés spécifiquement pour la reconnaissance faciale. Il surpasse DeepFace et FaceNet sur les benchmarks LFW/IJB en conditions difficiles (occultation partielle, angle, basse résolution), conditions représentatives d'un corpus de vidéos de terrain militaire.

**Embedding de référence :** plutôt qu'une photo unique, trop sensible aux variations de pose et de lumière, on moyenne les embeddings de plusieurs photos variées puis on L2-normalise. La moyenne dans l'espace ArcFace converge vers une représentation stable de l'identité.

**Seuil cosine (défaut 0.4) :** valeur calibrée empiriquement sur le corpus. En dessous de 0.35, les faux positifs explosent (visages similaires en tenue militaire) ; au-dessus de 0.5, les faux négatifs deviennent trop nombreux (casque, lunettes FPV). La valeur de `config.yaml` (`insightface.identity_threshold: 0.6`) est volontairement conservative et sert de garde-fou pour d'autres usages.

**Comptage d'individus (DBSCAN) :** tous les embeddings d'un message, toutes frames confondues, sont clusterisés sur distance cosine avec eps 0.55 et min_samples 1. Le choix de min_samples 1 garantit qu'aucun visage n'est classé comme bruit ; eps 0.55 correspond à une similarité minimale de 0.45 pour fusionner deux visages en un même individu, seuil volontairement bas pour absorber les occultations partielles.

**Limites :** casques, lunettes FPV et cagoules réduisent le taux de détection, tout comme le flou, la basse lumière et la compression des keyframes. Ce taux de non-détection est traité comme une donnée du corpus et non comme un défaut à corriger : la raréfaction des détections fait précisément partie de l'analyse.
