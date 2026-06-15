#!/usr/bin/env python3
"""
Images moyennes par phase — agrégation pixel-wise sur canvas carré + masque alpha.

Pour chaque phase (P1/P2/P3), on échantillonne 1 image représentative par message
(la keyframe du milieu temporel pour les vidéos, l'image directe pour les photos),
on la place dans un canvas carré 1080×1080 en préservant son ratio natif, et on
moyenne pixel-par-pixel uniquement sur les zones effectivement couvertes (masque
alpha). Aucune distorsion verticale/horizontale ; les pixels jamais couverts
restent noirs.

Sept variantes sont produites en parallèle :
  - global     : toutes les sources (photo + vidéo) confondues
  - photo      : photos uniquement
  - video      : keyframes vidéo uniquement
  - vertical   : ratio w/h ∈ [0.5, 0.7]   (9:16 = 0.56)
  - carre      : ratio w/h ∈ [0.85, 1.18] (1:1 = 1.0)
  - horizontal : ratio w/h ∈ [1.5, 1.9]   (16:9 = 1.78)
  - autre      : ratios hors zones canoniques (4:3 strict, ultrawide, etc.)

Pondération uniforme : chaque message contribue pour 1, indépendamment de sa
résolution native. La résolution n'est utilisée qu'à travers le ratio (préservé
par le placement non-distordu).

Échantillonnage 1 image / message :
  1. Équité statistique — une vidéo longue ne pèse pas plus qu'une courte
  2. Rapidité — 1246 images vs 99 778 keyframes

Figures produites (dans 4_data_et_viz/ par défaut) :
  - 54a_couleur_moyenne_global_{P1,P2,P3}.png     + composite
  - 54b_couleur_moyenne_photo_{P1,P2,P3}.png      + composite
  - 54c_couleur_moyenne_video_{P1,P2,P3}.png      + composite
  - 54d_couleur_moyenne_vertical_{P1,P2,P3}.png   + composite
  - 54e_couleur_moyenne_carre_{P1,P2,P3}.png      + composite
  - 54f_couleur_moyenne_horizontal_{P1,P2,P3}.png + composite
  - 54g_couleur_moyenne_autre_{P1,P2,P3}.png      + composite

Usage :
    python couleur_images_moyennes.py
    python couleur_images_moyennes.py --limit 30
    python couleur_images_moyennes.py --variantes photo video
    python couleur_images_moyennes.py --canvas-size 720 --output-dir /tmp/test
"""

import datetime
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    etiquette_phase,
)

SCRIPT_DIR = Path(__file__).resolve().parent

_MOIS_FR = {
    1: "janv.", 2: "fevr.", 3: "mars", 4: "avr.", 5: "mai", 6: "juin",
    7: "juil.", 8: "aout", 9: "sept.", 10: "oct.", 11: "nov.", 12: "dec."
}

def date_abrev(d: str) -> str:
    """'2022-09-01' → 'sept. 2022'"""
    dt = datetime.date.fromisoformat(d)
    return f"{_MOIS_FR[dt.month]} {dt.year}"

# ── Constantes ──────────────────────────────────────────────────────────────

# Canvas carré pour absorber tous les formats (vertical 9:16, carré, 16:9)
# sans distorsion. 1080 = grand côté typique d'une vidéo Telegram HD.
CANVAS_SIZE = 1080

# Sortie centralisée comme toutes les autres figures du mémoire (01_…53_…).
OUTPUT_DIR_DEFAUT = Path(__file__).resolve().parents[1] / "4_data_et_viz"

# Préfixe + suffixes a..g — ordre stable pour le tri alphabétique des PNG.
# a-c : découpe par source (global/photo/video).
# d-f : découpe par ratio canonique (vertical 9:16 / carré 1:1 / horizontal 16:9).
# g   : bucket résiduel pour les ratios hors intervalles canoniques.
PREFIXE = "54"
VARIANTES = ("global", "photo", "video", "vertical", "carre", "horizontal", "autre")
SUFFIXES = {
    "global": "a", "photo": "b", "video": "c",
    "vertical": "d", "carre": "e", "horizontal": "f", "autre": "g",
}

# Intervalles ratio w/h pour la classification format (intervalles fermés).
# 9:16 = 0.5625, 1:1 = 1.0, 16:9 = 1.778. Les intervalles sont resserrés autour
# des formats canoniques Telegram pour éviter qu'un 4:3 (1.33) "bleed" dans
# la zone centrale du composite horizontal. Tout ce qui tombe hors des trois
# intervalles est routé vers "autre" (4:3, ultrawide, ratios atypiques).
RATIO_VERTICAL   = (0.50, 0.70)
RATIO_CARRE      = (0.85, 1.18)
RATIO_HORIZONTAL = (1.50, 1.90)


# ── Helpers ─────────────────────────────────────────────────────────────────

def lire_image(path: Path) -> np.ndarray | None:
    """Lecture robuste (chemins unicode via buffer binaire)."""
    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def source_representative(msg: dict, keyframes_dir: Path, corpus_base: Path) -> Path | None:
    """Retourne 1 image représentative par message (keyframe du milieu pour
    les vidéos, photo directe sinon), ou None.

    Détection des keyframes par glob — pas de champ JSONL pour les compter."""
    channel = msg.get("canal", "robert_magyar")
    mid = msg["message_id"]

    if msg.get("media_type") == "video":
        pattern = f"{channel}_{mid}_kf_"
        kfs = sorted(
            p for p in keyframes_dir.iterdir()
            if p.name.startswith(pattern) and p.suffix.lower() in (".jpg", ".png")
        )
        if kfs:
            return kfs[len(kfs) // 2]

    if msg.get("media_type") == "photo" and msg.get("media_chemin"):
        photo = corpus_base / msg["media_chemin"]
        if photo.is_file():
            return photo

    return None


def placer_dans_canvas(img: np.ndarray, taille: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize en préservant le ratio (grand côté = taille), placement centré
    dans un canvas carré taille×taille.

    Entrée : img BGR (cv2), taille — côté du canvas carré
    Sortie : (canvas_img float64 H×W×3, canvas_mask float64 H×W)
             mask = 1.0 où l'image est posée, 0.0 ailleurs
    """
    h, w = img.shape[:2]
    scale = taille / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # cv2.resize attend (width, height)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (taille - new_h) // 2
    pad_left = (taille - new_w) // 2

    canvas_img = np.zeros((taille, taille, 3), dtype=np.float64)
    canvas_mask = np.zeros((taille, taille), dtype=np.float64)
    canvas_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_resized
    canvas_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1.0

    return canvas_img, canvas_mask


def etiqueter_image(img: np.ndarray, date_range: str, variante: str, n: int, phase_name: str) -> np.ndarray:
    """Ajoute un bandeau texte en bas (plage de dates, nom, variante, n messages)."""
    h, w = img.shape[:2]
    bandeau_h = max(40, h // 20)
    out = np.zeros((h + bandeau_h, w, 3), dtype=np.uint8)
    out[:h] = img
    out[h:] = (30, 30, 30)

    txt = f"{date_range}  |  {phase_name}  |  {variante}  |  n={n}"
    cv2.putText(out, txt, (12, h + bandeau_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def variantes_pour_message(msg: dict, variantes_actives: tuple[str, ...]) -> list[str]:
    """Retourne les variantes liées à la source pour un message.
    - global : toujours (si activée)
    - photo  : ssi media_type == "photo"
    - video  : ssi media_type == "video"
    """
    sortie = []
    mtype = msg.get("media_type")
    if "global" in variantes_actives:
        sortie.append("global")
    if "photo" in variantes_actives and mtype == "photo":
        sortie.append("photo")
    if "video" in variantes_actives and mtype == "video":
        sortie.append("video")
    return sortie


def buckets_format_pour_image(img: np.ndarray, variantes_actives: tuple[str, ...]) -> list[str]:
    """Retourne le bucket format (vertical/carre/horizontal/autre) pour une image
    selon son ratio w/h natif (avant resize). Une image n'est jamais dans deux
    buckets format à la fois — les trois intervalles canoniques sont disjoints,
    et "autre" est le résidu.

    Lecture des dimensions sur l'image plutôt que sur le JSONL : robuste aux
    keyframes dont la résolution peut différer de celle de la vidéo source.
    """
    h, w = img.shape[:2]
    if h <= 0:
        return []
    ratio = w / h
    if "vertical" in variantes_actives and RATIO_VERTICAL[0] <= ratio <= RATIO_VERTICAL[1]:
        return ["vertical"]
    if "carre" in variantes_actives and RATIO_CARRE[0] <= ratio <= RATIO_CARRE[1]:
        return ["carre"]
    if "horizontal" in variantes_actives and RATIO_HORIZONTAL[0] <= ratio <= RATIO_HORIZONTAL[1]:
        return ["horizontal"]
    if "autre" in variantes_actives:
        return ["autre"]
    return []


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Images moyennes par phase (canvas carré + masque alpha, 3 variantes photo/video/global)",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source (défaut: messages_clean.jsonl)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAUT),
                        help=f"Répertoire de sortie (défaut: {OUTPUT_DIR_DEFAUT})")
    parser.add_argument("--canvas-size", type=int, default=CANVAS_SIZE,
                        help=f"Côté du canvas carré (défaut: {CANVAS_SIZE})")
    parser.add_argument("--variantes", nargs="+", default=list(VARIANTES),
                        choices=list(VARIANTES),
                        help=f"Sous-ensemble parmi {VARIANTES} (défaut: les 3)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("couleurs_moyennes", cfg=cfg)

    corpus_base   = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])
    input_path = (Path(args.input) if args.input
                  else corpus_base / "messages_clean.jsonl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taille = args.canvas_size
    variantes_actives = tuple(args.variantes)

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages — canvas {taille}×{taille}")
    log.info(f"Variantes : {variantes_actives}")
    log.info(f"Sortie : {output_dir}")

    # ── Regroupement par phase ─────────────────────────────────────────────
    par_phase = defaultdict(list)
    for m in messages:
        if not m.get("date"):
            continue
        p = etiquette_phase(m["date"], cfg=cfg)
        if p:
            par_phase[p].append(m)
    for p in sorted(par_phase):
        log.info(f"  {p} : {len(par_phase[p])} messages")

    # Filtre --limit / --ids appliqué globalement (pas par phase)
    filtre_ids = set(args.ids) if args.ids else None
    if filtre_ids:
        par_phase = {p: [m for m in msgs if m["message_id"] in filtre_ids]
                     for p, msgs in par_phase.items()}
    if args.limit:
        # On garde proportionnellement les premiers messages — debug seulement
        total = sum(len(v) for v in par_phase.values())
        if total > args.limit:
            ratio = args.limit / total
            par_phase = {p: msgs[:max(1, int(len(msgs) * ratio))]
                         for p, msgs in par_phase.items()}

    phase_names = {p: cfg["phases"][p]["label"] for p in par_phase}
    phase_dates = {
        p: f"{date_abrev(cfg['phases'][p]['start'])} - {date_abrev(cfg['phases'][p]['end'])}"
        for p in par_phase
    }

    # ── Accumulateurs ──────────────────────────────────────────────────────
    # Clé : (phase, variante). Valeur : (sum_img H×W×3 float64, sum_mask H×W float64)
    sum_img = {}
    sum_mask = {}
    counts = defaultdict(int)
    skipped = defaultdict(int)
    for p in par_phase:
        for v in variantes_actives:
            sum_img[(p, v)] = np.zeros((taille, taille, 3), dtype=np.float64)
            sum_mask[(p, v)] = np.zeros((taille, taille), dtype=np.float64)

    # ── Accumulation pixel-wise ────────────────────────────────────────────
    for phase in sorted(par_phase):
        log.info(f"\nPhase {phase} ({phase_names[phase]}) — accumulation...")
        for msg in tqdm(par_phase[phase], desc=phase, unit="msg"):
            src = source_representative(msg, keyframes_dir, corpus_base)
            if src is None:
                skipped[(phase, "no_source")] += 1
                continue
            img = lire_image(src)
            if img is None:
                skipped[(phase, "read_error")] += 1
                continue

            canvas_img, canvas_mask = placer_dans_canvas(img, taille)
            # Broadcast du masque sur les 3 canaux pour l'accumulation pondérée
            canvas_img_masked = canvas_img * canvas_mask[..., None]

            # Variantes par source (global/photo/video) + par format (vert/carre/horiz)
            variantes_cibles = (
                variantes_pour_message(msg, variantes_actives)
                + buckets_format_pour_image(img, variantes_actives)
            )
            for v in variantes_cibles:
                sum_img[(phase, v)] += canvas_img_masked
                sum_mask[(phase, v)] += canvas_mask
                counts[(phase, v)] += 1

    # ── Calcul moyennes par pixel ──────────────────────────────────────────
    log.info("\nCalcul des moyennes pixel-par-pixel...")
    moyennes = {}
    for (phase, v), s_img in sum_img.items():
        s_mask = sum_mask[(phase, v)]
        n = counts[(phase, v)]
        if n == 0:
            log.warning(f"  {phase}/{v} : aucune image utilisable")
            continue
        # Division pixel-par-pixel : pixels non couverts → moyenne = 0 (fond noir)
        # On évite division par zéro en clippant le denominateur à 1.
        mean_img = s_img / np.maximum(s_mask[..., None], 1.0)
        mean_img = mean_img.clip(0, 255).astype(np.uint8)
        # Pixels jamais couverts : forcer noir explicitement (sécurité)
        mean_img[s_mask == 0] = 0
        moyennes[(phase, v)] = mean_img
        couverture = float((s_mask > 0).mean()) * 100
        log.info(f"  {phase}/{v:6s} : n={n}  couverture={couverture:.1f}%  "
                 f"skipped(no_src={skipped[(phase, 'no_source')]}, "
                 f"read_err={skipped[(phase, 'read_error')]})")

    # ── Sauvegarde individuelle + étiquetage ───────────────────────────────
    etiquetees = defaultdict(dict)  # variante → {phase → image étiquetée}
    for (phase, v), img in moyennes.items():
        n = counts[(phase, v)]
        labeled = etiqueter_image(img, phase_dates[phase], v, n, phase)
        suffixe = SUFFIXES[v]
        out = output_dir / f"{PREFIXE}{suffixe}_couleur_moyenne_{v}_{phase}.png"
        cv2.imwrite(str(out), labeled)
        etiquetees[v][phase] = labeled
        log.info(f"  → {out.name}")

    # ── Composites horizontaux (P1 | P2 | P3) par variante ─────────────────
    log.info("")
    for v in variantes_actives:
        ordre = [p for p in ["P1", "P2", "P3"] if p in etiquetees.get(v, {})]
        if len(ordre) >= 2:
            composite = np.hstack([etiquetees[v][p] for p in ordre])
            suffixe = SUFFIXES[v]
            out = output_dir / f"{PREFIXE}{suffixe}_couleur_moyenne_{v}_composite.png"
            cv2.imwrite(str(out), composite)
            log.info(f"  → {out.name}")


if __name__ == "__main__":
    main()
