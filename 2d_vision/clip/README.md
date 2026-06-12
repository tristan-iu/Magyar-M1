# Classification zero-shot CLIP (limite méthodologique)

Classification zero-shot des keyframes du corpus via CLIP (`openai/clip-vit-large-patch14`), sous forme de six classifieurs binaires indépendants. Limite documentée : le zero-shot CLIP discrimine mal l'imagerie de guerre (scènes proches, flou, compression, thermique). Les résultats ne sont pas retenus dans le mémoire et ce module n'écrit aucun champ JSONL : seuls des CSV sont produits, conservés pour traçabilité.

## Installation

```bash
pip install -r requirements.txt
```

Un GPU CUDA est recommandé (environ 30 minutes pour le corpus complet).

## Utilisation

```bash
# Classification par keyframe
python clip_classify.py
#   --limit 5         test rapide
#   --batch-size 64   si la VRAM le permet

# Agrégation par message
python aggregate_clip.py
#   --threshold 0.60  seuil des booléens {concept}_flag
```

L'entrée est le JSONL canonique (défaut depuis `config.yaml`) et les keyframes PNG déjà extraites dans `fiches/keyframes/`, retrouvées par glob.

## Output

`clip_classify.py` écrit `results/clip_classification.csv` (une ligne par keyframe) et `aggregate_clip.py` en dérive `results/clip_by_message.csv` (une ligne par message), où chaque concept est agrégé selon sa nature (`mean` pour une présence continue comme le vlog, `max` pour un événement ponctuel comme la carte de statistiques) et doublé d'un booléen `{concept}_flag` au-dessus du seuil.

### Colonnes du CSV par keyframe

| Colonne | Type | Description |
|---------|------|-------------|
| `message_id` | int | ID du message source |
| `frame_filename` | str | Nom du fichier keyframe |
| `date` | str | Date du message |
| `phase` | str | Phase du corpus |
| `clip_vlog` | float | Personne parlant face caméra, style selfie |
| `clip_aerial` | float | Vue aérienne verticale du terrain |
| `clip_fpv` | float | Vidéo de drone FPV (protections d'hélices visibles) |
| `clip_stats` | float | Carte de statistiques (fond noir, chiffres) |
| `clip_screen` | float | Écran ou contrôleur de drone filmé |
| `clip_strike` | float | Impact, explosion ou destruction vue du ciel |

Chaque score est une probabilité entre 0 et 1 issue d'un softmax entre un prompt positif et un prompt négatif propres au concept. Les scores sont indépendants : une frame peut scorer haut sur plusieurs concepts à la fois.

## Méthodologie

**Binaire plutôt que softmax multi-classe :** une softmax sur une dizaine de labels dilue les probabilités entre classes visuellement proches (« drone surveillance » et « FPV » se cannibalisent). Chaque concept reçoit donc sa propre comparaison positif contre négatif, le prompt négatif étant choisi pour être visuellement opposé et non un simple « not X ».

**Labels en anglais :** CLIP est entraîné majoritairement sur des légendes anglaises, ses performances zero-shot sur des labels français ou ukrainiens sont nettement inférieures. Les prompts sont rédigés pour être discriminants dans le contexte militaire (« FPV drone video where the curved propeller guard ring is visible » plutôt que « drone video »).

**Modèle :** `openai/clip-vit-large-patch14`, le plus grand CLIP publiquement disponible, aux meilleures performances zero-shot que les variantes ViT-B.

**Limite retenue (formulation du mémoire) :** « La classification zero-shot CLIP (ViT-L/14) a été testée sur les keyframes du corpus. Les résultats n'ont pas été retenus en raison d'une discrimination insuffisante entre concepts dans le contexte de l'imagerie de guerre. » Aucune relance ni écriture JSONL prévue.
