# 2d_vision/clip — Classification zero-shot CLIP (limite méthodologique)

Classification zero-shot des keyframes du corpus Magyar via CLIP
(`openai/clip-vit-large-patch14`), sous forme de **6 classifieurs binaires
indépendants**. **Limite documentée :** le zero-shot CLIP discrimine mal
l'imagerie de guerre (scènes proches, flou, compression, thermique) ; les
résultats ne sont **pas retenus** dans le mémoire et ce module n'écrit
**aucun champ JSONL** — seuls des CSV sont produits, conservés pour traçabilité.

## Utilisation

```bash
pip install -r requirements.txt

# Classification (GPU recommandé, ~30 min corpus complet)
python clip_classify.py
#   --limit 5         test rapide
#   --batch-size 64   si VRAM suffisante

# Agrégation par message
python aggregate_clip.py
```

## Input / Output

**Entrée :** JSONL canonique (`messages_clean.jsonl`, défaut depuis
`config.yaml`) + keyframes PNG déjà extraites dans `fiches/keyframes/`
(filtrage par glob `{canal}_{message_id}_kf_*` — pas de champ JSONL pour les
keyframes).

**`clip_classify.py` → `results/clip_classification.csv`** (1 ligne par keyframe) :

| Colonne | Type | Description |
|---------|------|-------------|
| `message_id` | int | ID du message source |
| `frame_filename` | str | Nom du fichier keyframe |
| `date` | str | Date du message |
| `phase` | str | Phase du corpus (P1/P2/P3) |
| `clip_vlog` | float | Personne face caméra, style selfie vlog |
| `clip_aerial` | float | Vue aérienne top-down terrain (ISR) |
| `clip_fpv` | float | Drone FPV (prop guards circulaires visibles) |
| `clip_stats` | float | Carte stats institutionnelle (fond noir + chiffres) |
| `clip_screen` | float | Filmer un écran / contrôleur DJI |
| `clip_strike` | float | Impact / explosion / destruction vue aérienne |

Chaque score est une probabilité 0→1 issue d'un `softmax([prompt_positif,
prompt_négatif])` propre au concept. Les scores sont **indépendants** (pas de
compétition softmax entre concepts) : une frame peut scorer haut sur plusieurs.

**`aggregate_clip.py` → `results/clip_by_message.csv`** (1 ligne par message) :
agrège les frames d'un même message selon `CLIP_AGG` (`mean` pour la présence
continue : vlog/aerial/screen/strike ; `max` pour l'événement ponctuel : stats ;
`p75` pour fpv), plus un booléen `{concept}_flag` (score > `--threshold`, défaut 0.60).

## Méthodologie

**Pourquoi binaire plutôt que softmax multi-classe ?** Une softmax sur 9–12
labels dilue les probabilités entre classes visuellement proches (« drone
surveillance » vs « FPV » se cannibalisent). Chaque concept reçoit donc sa
propre comparaison positif/négatif calibrée, le prompt négatif étant choisi
pour être visuellement opposé (pas un simple « not X »).

**Labels en anglais :** CLIP est entraîné majoritairement sur des captions
anglaises ; les performances zero-shot sur des labels FR/UK sont nettement
inférieures. Les prompts sont rédigés pour être discriminants dans le contexte
militaire (ex. « FPV drone video where the curved propeller guard ring is
visible » plutôt que « drone video »).

**Modèle `openai/clip-vit-large-patch14` :** le plus grand CLIP publiquement
disponible, meilleures performances zero-shot que ViT-B/32 ou ViT-B/16.

**Limite retenue (formulation mémoire) :** *« La classification zero-shot CLIP
(ViT-L/14) a été testée sur les keyframes du corpus. Les résultats n'ont pas
été retenus en raison d'une discrimination insuffisante entre concepts dans le
contexte de l'imagerie de guerre. »* Aucune relance ni écriture JSONL prévue.
