# blasons/ — Détection des logos de brigade

Détecte la présence des blasons/logos de la brigade 414 OBr ("Птахи Мадяра")
en watermark dans les keyframes du corpus. Signal complémentaire à l'OCR du
texte "414 ОБр" — le logo précède ou accompagne le texte, et résiste mieux
à la compression vidéo.

## Dépendances

```bash
pip install -r requirements.txt
```

SIFT est disponible dans `opencv-python` depuis la version 4.4.0 (expiration
du brevet Lowe en mars 2020).

## Usage

```bash
# Corpus complet (défaut)
python detect_blasons.py \
  --input  /chemin/messages_faces.jsonl \
  --output /chemin/messages_blasons.jsonl

# Test sur quelques messages connus
python detect_blasons.py --ids 908 1080 1325 --overwrite

# Ajuster le seuil (voir section Méthodologie)
python detect_blasons.py --match-threshold 20 --overwrite
```

## Options CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--input` | config `jsonl_faces` | JSONL source |
| `--output` | `messages_blasons.jsonl` | JSONL enrichi |
| `--refs-dir` | `references/` | Dossier des crops de référence |
| `--match-threshold` | **15** | Seuil inliers RANSAC (voir Méthodologie) |
| `--ratio` | 0.75 | Ratio test de Lowe |
| `--roi-pct` | 0.30 | Taille des ROI en fraction de l'image |
| `--rois` | `haut_droite bas_droite` | Coins à scanner |
| `--csv` | `results/blason_detection.csv` | CSV per-frame |
| `--limit` | aucune | N messages max |
| `--ids` | tous | Message IDs spécifiques |
| `--overwrite` | false | Retraiter les messages déjà faits |

## Champs JSONL produits

| Champ | Type | Description |
|-------|------|-------------|
| `blason_present` | bool | Blason détecté dans au moins une keyframe |
| `blason_detecte` | str | Catégorie dominante (`414_obr` / `414_mono` / `pm_SARG`) |
| `blason_roi` | str | Coin où le blason est le plus souvent détecté |

## Références

Placer des crops PNG/JPG du blason **seul** (bien cadré, fond minimal)
dans des sous-dossiers de `references/` — un sous-dossier par catégorie :

```
references/
├── 414_obr/     # blason couleur teal, "414 ОБр"
├── 414_mono/    # variante monochrome
└── pm_SARG/     # variante "САРГ" (PM/reconn.)
```

Règles pratiques : 5–10 crops par catégorie, issus de vraies frames vidéo
(pas de renders officiels), couvrant les variantes de taille (480p–1080p)
et de contraste.

---

## Méthodologie — pipeline de détection

### 1. Extraction des ROI

Pour chaque keyframe, on extrait les régions d'intérêt dans les coins
(30 % de la largeur et de la hauteur). Les logos de brigade apparaissent
systématiquement en watermark de coin dans les vidéos P2–P3. Pour les
photos, l'image entière est également testée (pas de position fixe).

### 2. SIFT — extraction de points caractéristiques

SIFT (*Scale-Invariant Feature Transform*, Lowe 2004) détecte des points
d'intérêt locaux dans chaque ROI et les décrit par un vecteur de 128
dimensions. L'invariance d'échelle est essentielle ici : le même blason
occupe ~80 px sur une vidéo 480p et ~220 px sur une 1080p (facteur ×2,5).

### 3. Appariement + ratio test de Lowe

Les descripteurs de la ROI sont appariés à ceux de chaque référence via
BFMatcher (distance euclidienne L2). Le ratio test de Lowe (seuil 0,75)
écarte les appariements ambigus : un match n'est retenu que si le
meilleur voisin est significativement meilleur que le second.

### 4. Vérification géométrique RANSAC

L'étape clé contre les faux positifs. RANSAC (*Random Sample Consensus*)
estime une homographie entre les points appariés et compte les **inliers** :
points dont la transformation est géométriquement cohérente avec la
transformation globale. Les coïncidences texturales (végétation, sol,
fumée) produisent des appariements dispersés qui ne forment aucune
homographie stable — elles sont éliminées ici.

### 5. Seuil de détection : 15 inliers

Le seuil par défaut de **15 inliers RANSAC** a été calibré sur le corpus
réel. La simulation sur l'ensemble des 1 365 messages montre :

| Seuil | P1 (artisanal) | P2 (semi-pro) | P3 (institutionnel) |
|-------|---------------|---------------|---------------------|
| 8 | 36 % | 43 % | 75 % |
| 12 | 11 % | 24 % | 67 % |
| **15** | **5 %** | **16 %** | **65 %** |
| 20 | 1 % | 13 % | 54 % |

À seuil 15, le taux P1 (5 %) est cohérent avec la réalité historique :
le branding visuel n'existe pas en P1 — les rares détections correspondent
à des vidéos où Magyar porte un écusson sur l'uniforme en gros plan,
ou à du bruit résiduel. La progression P1→P3 (5 %→65 %) reflète
la montée en puissance du branding institutionnel à partir de P2
(janvier 2024, création officielle du 414e bataillon).

En dessous de 12, les faux positifs P1 explosent (textures riches en
coins : végétation filmée en drone, tissu, équipement), trahissant une
détection de texture et non de logo. Au-dessus de 20, P3 commence à
perdre des détections légitimes (petits logos flous dans les
compilations courtes fortement compressées).

## Sorties

- `results/blason_detection.csv` — une ligne par keyframe analysée
  (`message_id, keyframe, frame_position, blason_present, n_inliers,
  blason_detecte, blason_roi`)
- JSONL enrichi — 3 champs par message
- Fiches individuelles mises à jour (`fiches/robert_magyar_{id}_fiche.json`)
