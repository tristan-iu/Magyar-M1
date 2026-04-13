# CLIP Zero-Shot Classification

Classification zero-shot des ~32K keyframes du corpus Magyar via CLIP (openai/clip-vit-large-patch14).

## Labels

**Set 1 — Contexte visuel (11 labels)** : outdoor battlefield, trench or fortification, military vehicle, drone first-person view, drone aerial surveillance view, indoor command post, person talking to camera, FPV impact explosion, thermal or night vision, text overlay or title card, ruins or destroyed building.

**Set 2 — Presence humaine (5 labels)** : no people visible, single person visible, group of soldiers, dead or wounded body on ground, person running or crawling.

## Usage

```bash
pip install -r requirements.txt

# Classification (GPU recommande)
python clip_classify.py --limit 5        # test rapide
python clip_classify.py                  # corpus complet (~30 min GPU)
python clip_classify.py --batch-size 64  # si VRAM suffisante

# Agregation + graphiques
python aggregate_clip.py
```

## Input / Output

**Entrée :** JSONL enrichi (messages avec `keyframes_count > 0`), keyframes PNG dans `fiches/keyframes/`.

**CSV produit** (`results/clip_classification.csv`) : 1 ligne par keyframe.

| Colonne | Type | Description |
|---------|------|-------------|
| `message_id` | int | ID du message source |
| `frame_filename` | str | Nom du fichier keyframe |
| `date` | str | Date du message |
| `phase` | str | Phase du corpus (P1/P2/P3) |
| `scene_<label>` | float | Probabilité softmax groupe SCENE (9 labels, somme = 1) |
| `content_<label>` | float | Probabilité softmax groupe CONTENT (6 labels, somme = 1) |

**Sorties agrégées** (`aggregate_clip.py`) :

| Fichier | Description |
|---------|-------------|
| `clip_by_message.csv` | Label dominant + probas moyennes par message |
| `clip_by_phase.csv` | Distribution des labels dominants par phase (% messages) |
| `clip_monthly.csv` | Probabilités moyennes mensuelles par label |
| `clip_stacked_bar_phase.png` | Stacked bar chart des labels de scène par phase |
| `clip_monthly_lines.png` | Courbes mensuelles pour 3 catégories clés |
| `clip_heatmap.png` | Heatmap mois × labels de scène |

**Champs JSONL enrichis** (par `aggregate_clip.py --no-enrich-jsonl` pour désactiver) :

| Champ | Type | Description |
|-------|------|-------------|
| `clip_scene_dominant` | str | Label de scène dominant (argmax) |
| `clip_human_dominant` | str | Label de contenu humain dominant |
| `clip_scene_<label>` | float | Score moyen 0–1 pour chaque label scène |
| `clip_human_<label>` | float | Score moyen 0–1 pour chaque label contenu |

## Méthodologie

**Pourquoi CLIP zero-shot ?** CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) permet de classer des images sans données d'entraînement annotées propres au corpus. Pour un corpus de ~32K keyframes militaires ukrainiennes, annoter manuellement un jeu d'entraînement représentatif serait prohibitif. CLIP fournit une classification exploitable directement.

**Modèle `openai/clip-vit-large-patch14`** : le plus grand modèle CLIP publiquement disponible, avec de meilleures performances zero-shot que les variantes ViT-B/32 ou ViT-B/16. Cohérent avec la taille du corpus et la disponibilité GPU.

**Labels en anglais** : CLIP est entraîné majoritairement sur du texte anglais. Les performances zero-shot sur des labels en français ou ukrainien sont notablement inférieures. Les labels ont été rédigés pour être discriminatifs et non ambigus dans le contexte militaire (ex. "first-person view from a flying drone" plutôt que "drone video").

**Deux groupes softmax indépendants** : SCENE (type de plan / source de l'image) et CONTENT (ce qui est visible). Un seul softmax sur tous les labels produirait une compétition artificielle entre catégories conceptuellement orthogonales. Deux passes séparées permettent d'obtenir une probabilité normalisée au sein de chaque groupe sémantique.
