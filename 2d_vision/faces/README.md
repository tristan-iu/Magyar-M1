# Detection de presence Magyar (InsightFace)

Detection du visage de Magyar (Robert Brovdi) dans les keyframes du corpus Telegram, pour quantifier son effacement physique entre P1 (artisanal) et P3 (institutionnel).

## Pipeline

```
1. build_reference.py   Calcul embedding moyen a partir de photos de reference
2. detect_magyar.py     Detection sur ~32K keyframes, CSV par frame + enrichissement JSONL
3. aggregate_magyar.py  Agregation par message/phase, stats, graphique mensuel
```

## Installation

```bash
pip install -r requirements.txt
# ou avec GPU :
pip install insightface onnxruntime-gpu numpy opencv-python matplotlib tqdm
```

## Usage

### 1. Preparer les photos de reference

Placer 5-10 photos nettes de Magyar dans `references/magyar/`. Formats acceptes : jpg, jpeg, png, webp, bmp. Privilegier des photos variees (angles, eclairage, avec/sans casque si possible).

### 2. Calculer l'embedding de reference

```bash
python build_reference.py
# Options :
python build_reference.py --references-dir references/magyar/ --output references/magyar_embedding.npy
```

### 3. Detecter Magyar dans les keyframes

```bash
python detect_magyar.py
# Tester sur 10 messages :
python detect_magyar.py --limit 10
# Messages specifiques :
python detect_magyar.py --ids 42 138 256
# Ajuster le seuil :
python detect_magyar.py --threshold 0.35
# Retraiter tout :
python detect_magyar.py --overwrite
```

### 4. Agreger et visualiser

```bash
python aggregate_magyar.py
```

Sorties dans `results/` :
- `magyar_detection.csv` — une ligne par frame
- `magyar_aggregated.csv` — une ligne par message
- `magyar_by_phase.csv` — stats par phase
- `magyar_monthly.png` — courbe mensuelle coloree par phase

## Champs enrichis (JSONL)

| Champ | Type | Description |
|-------|------|-------------|
| `faces_magyar_present` | bool | Magyar detecte dans au moins une frame |
| `faces_magyar_ratio` | float | Proportion de frames avec Magyar |
| `faces_avg_count` | float | Nombre moyen de visages par frame |
| `faces_max_similarity` | float | Similarite cosine maximale avec la reference |

## Seuil de detection

Le seuil par defaut est **0.4** (cosine similarity). Pour l'ajuster :
- **Baisser** (ex: 0.3) : plus de detections, plus de faux positifs
- **Monter** (ex: 0.5) : moins de faux positifs, plus de faux negatifs

Le seuil dans `config.yaml` (`insightface.identity_threshold: 0.6`) est une valeur conservative. Le defaut CLI a 0.4 est plus adapte aux conditions reelles du corpus.

## Limites connues

- **Casques et equipement** : le casque militaire, les lunettes FPV et les cagoules reduisent le taux de detection
- **Qualite des keyframes** : les frames degradees, floues ou en basse lumiere compliquent la detection
- **Occlusions partielles** : visages de profil, partiellement caches
- **Faux negatifs en conditions de combat** : poussieres, fumee, mouvements rapides
- **Taux de non-detection** : c'est une donnee a documenter, pas un defaut a corriger. Le ratio de faux negatifs fait partie de l'analyse

## Idempotence

- `detect_magyar.py` charge le CSV existant au demarrage et skip les frames deja traitees
- Relancer sans `--overwrite` ne retraite que les nouveaux messages
- Le JSONL est sauvegarde tous les 50 messages (configurable via `batch.save_every`)

## Méthodologie

**Pourquoi InsightFace ?** InsightFace (modèle `buffalo_l`, ArcFace backbone) produit des embeddings 512d discriminatifs entraînés spécifiquement sur la reconnaissance faciale. Il surpasse DeepFace et FaceNet sur les benchmarks LFW/IJB en conditions difficiles (occultation partielle, angle, basse résolution) — conditions représentatives d'un corpus de vidéos de terrain militaire.

**Embedding de référence** : plutôt qu'une photo unique (trop sensible aux variations de pose/lumière), on calcule la moyenne L2-normalisée de N embeddings extraits de photos de référence variées. La moyenne dans l'espace des embeddings ArcFace converge vers une représentation stable de l'identité.

**Seuil cosine (défaut 0.4)** : valeur empirique calibrée sur le corpus. En dessous de 0.35 les faux positifs explosent (visages similaires en tenue militaire) ; au-dessus de 0.5 les faux négatifs deviennent trop nombreux (casque, lunettes FPV). La valeur dans `config.yaml` (`insightface.identity_threshold: 0.6`) est conservative et sert de garde-fou pour d'autres usages.

**Comptage d'individus distincts (DBSCAN)** : tous les embeddings d'un message (toutes frames confondues) sont clusterisés via DBSCAN avec distance cosine (eps=0.55, min_samples=1). Le choix de min_samples=1 évite les outliers (-1) et garantit que chaque visage est attribué à un individu. eps=0.55 correspond à une similarité minimale de 0.45 pour que deux visages soient considérés comme le même individu — seuil volontairement bas pour gérer les occultations partielles.
