# Détection des blasons de brigade (SIFT et RANSAC)

Détecte la présence des blasons de la brigade 414 OBr (« Птахи Мадяра ») en watermark dans les keyframes du corpus. Signal complémentaire à l'OCR du texte « 414 ОБр » : le logo précède ou accompagne le texte, et résiste mieux à la compression vidéo.

## Installation

```bash
pip install -r requirements.txt
```

SIFT est inclus dans `opencv-python` depuis la version 4.4.0 (expiration du brevet Lowe en mars 2020).

Prérequis : placer des crops PNG ou JPG du blason seul (bien cadré, fond minimal) dans des sous-dossiers de `references/`, un sous-dossier par catégorie de logo (par exemple `414_obr/` pour le blason couleur, `414_mono/` pour la variante monochrome). Compter 5 à 10 crops par catégorie, issus de vraies frames vidéo plutôt que de renders officiels, en couvrant les variantes de taille (480p à 1080p) et de contraste du corpus. Ce dossier n'est pas versionné.

## Utilisation

```bash
# Corpus complet (défauts : --input jsonl_clean de config.yaml, --output messages_blasons.jsonl)
python detect_blasons.py
#   --ids 908 1080 1325     messages spécifiques (test)
#   --limit N               N messages max
#   --overwrite             retraite les messages déjà faits
#   --match-threshold 15    seuil inliers RANSAC (voir Méthodologie)
#   --ratio 0.75            ratio test de Lowe
#   --roi-pct 0.30          taille des ROI (fraction de l'image)
#   --rois haut_droite bas_droite   coins à scanner
#   --refs-dir references/  dossier des crops de référence
#   --csv results/blason_detection.csv   CSV per-frame
```

L'idempotence est double : les `message_id` déjà présents dans le CSV sont skippés et le champ `blason_present` du JSONL est vérifié. Sauvegarde intermédiaire tous les 50 messages.

## Output

Le script écrit `results/blason_detection.csv`, une ligne par keyframe analysée (`message_id, keyframe, frame_position, blason_present, n_inliers, blason_detecte, blason_zone`), et enrichit le JSONL et les fiches individuelles des trois champs ci-dessous.

### Champs ajoutés au JSONL

| Champ | Type | Description |
|-------|------|-------------|
| `blason_present` | bool | Blason détecté dans au moins une keyframe |
| `blason_detecte` | str | Catégorie dominante (nom du sous-dossier de référence), `null` sinon |
| `blason_zone` | str | Coin où le blason est le plus souvent détecté (`haut_droite`, `bas_droite`...), `null` sinon |

## Méthodologie

**Régions d'intérêt :** pour chaque keyframe, la recherche se limite aux coins de l'image (30 % de la largeur et de la hauteur), où les logos de brigade apparaissent systématiquement en watermark. Pour les photos, l'image entière est testée en plus des coins, le blason pouvant s'y trouver n'importe où.

**SIFT :** SIFT (*Scale-Invariant Feature Transform*, Lowe 2004) détecte des points d'intérêt locaux dans chaque ROI et les décrit par des vecteurs flottants de 128 dimensions, plus discriminants sur les détails fins (trident, texte cyrillique) que des descripteurs binaires type ORB. Son invariance d'échelle est essentielle ici : le même blason occupe environ 80 px sur une vidéo 480p et 220 px sur une 1080p.

**Appariement et ratio test de Lowe :** les descripteurs de la ROI sont appariés à ceux de chaque référence par force brute (distance euclidienne L2). Le ratio test (seuil 0.75) écarte les appariements ambigus : un match n'est retenu que si le meilleur voisin est nettement meilleur que le second.

**Vérification géométrique RANSAC :** l'étape clé contre les faux positifs. RANSAC estime une homographie entre les points appariés et compte les inliers, les points cohérents avec une transformation géométrique globale. Les coïncidences texturales (végétation, sol, fumée) produisent des appariements dispersés qui ne forment aucune homographie stable et sont éliminées ici.

**Seuil de détection (15 inliers) :** calibré en simulant plusieurs seuils sur l'ensemble des 1 365 messages. En dessous de 12, les faux positifs explosent sur les périodes du corpus sans branding (textures riches en coins : végétation filmée au drone, tissu, équipement), ce qui trahit une détection de texture et non de logo. Au-dessus de 20, les détections légitimes chutent sur les compilations récentes fortement compressées (petits logos flous). À 15, la détection reste rare au début du corpus, où le branding visuel n'existe pas encore, et culmine sur la période institutionnelle : la trajectoire mesurée correspond à la chronologie réelle du branding de la brigade (création officielle du 414e bataillon en janvier 2024).
