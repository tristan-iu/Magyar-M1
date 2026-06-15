#!/usr/bin/env python3
"""
Espace couleur — PCA + K-means sur les histogrammes HSV.

Lit les histogrammes 3D produits par couleur_batch.py et les projette
dans un espace à faible dimension (PCA) + cluster non-supervisé (K-means).

Le but est de tester si les phases du corpus (P1/P2/P3) sont visuellement
séparables à partir de la seule distribution couleur des pixels. Si oui,
c'est un signal fort que l'homogénéisation de palette reflète bien la
professionnalisation (et pas juste l'évolution du contenu).

Pipeline :
  1. Charge le NPZ (N × 4096) + JSONL (pour phase & date)
  2. PCA sur les histogrammes — 3 composantes (couvre typiquement 40-60% variance)
  3. K-means k=3 sur les composantes PCA (pas sur les histogrammes bruts —
     moins bruité, dimensions comparables)
  4. Sorties CSV : coordonnées PCA + cluster assigné par message_id

Sorties :
  - 4_data_et_viz/couleurs/couleur_pca.csv       — message_id, phase, date, pc1, pc2, pc3
  - 4_data_et_viz/couleurs/couleur_clusters.csv  — message_id, phase, cluster, pc1, pc2
  - 4_data_et_viz/couleurs/couleur_pca_variance.csv — composante, variance expliquée, cumul

Usage :
    python couleur_espace.py
    python couleur_espace.py --n-components 5 --n-clusters 4
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
COULEUR_DIR = SCRIPT_DIR.parent / "4_data_et_viz" / "couleurs"


def main():
    parser = creer_parser_base(
        "PCA + K-means sur histogrammes HSV des messages",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source pour récupérer phase & date (défaut: messages_blasons.jsonl)")
    parser.add_argument(
        "--npz",
        default=str(COULEUR_DIR / "couleur_histogrammes.npz"),
        help="NPZ histogrammes (produit par couleur_batch.py)",
    )
    parser.add_argument("--n-components", type=int, default=3,
                        help="Nombre de composantes PCA à conserver (défaut: 3)")
    parser.add_argument("--n-clusters", type=int, default=3,
                        help="Nombre de clusters K-means (défaut: 3, comme les phases)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Graine K-means pour reproductibilité")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("couleurs_espace", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    input_path = (Path(args.input) if args.input
                  else corpus_base / "messages_blasons.jsonl")
    npz_path = Path(args.npz)

    if not npz_path.is_file():
        log.error(f"NPZ introuvable : {npz_path}. Lancer couleur_batch.py d'abord.")
        sys.exit(1)
    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)

    # ── Chargement ──────────────────────────────────────────────────────────
    log.info(f"Chargement NPZ : {npz_path}")
    with np.load(npz_path) as data:
        mids = data["message_ids"]
        hists = data["histograms"]
    log.info(f"Matrice : {hists.shape}")

    # Index phase & date par message_id depuis le JSONL
    log.info(f"Lecture JSONL : {input_path}")
    messages = read_jsonl(input_path)
    meta = {
        m["message_id"]: (etiquette_phase(m["date"], cfg=cfg), m["date"][:10])
        for m in messages if m.get("date")
    }

    # ── PCA ─────────────────────────────────────────────────────────────────
    log.info(f"PCA — {args.n_components} composantes")
    pca = PCA(n_components=args.n_components, random_state=args.random_state)
    coords = pca.fit_transform(hists)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    for i, (v, c) in enumerate(zip(var_ratio, cum_var), start=1):
        log.info(f"  PC{i} : {v*100:5.2f}%  (cumul {c*100:5.2f}%)")

    # ── K-means ─────────────────────────────────────────────────────────────
    log.info(f"K-means — k={args.n_clusters}")
    km = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init=10,
    )
    clusters = km.fit_predict(coords)

    # ── Export CSV ──────────────────────────────────────────────────────────
    pca_path      = COULEUR_DIR / "couleur_pca.csv"
    clusters_path = COULEUR_DIR / "couleur_clusters.csv"
    var_path      = COULEUR_DIR / "couleur_pca_variance.csv"
    pca_path.parent.mkdir(parents=True, exist_ok=True)

    pc_cols = [f"pc{i+1}" for i in range(args.n_components)]

    with open(pca_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message_id", "phase", "date"] + pc_cols)
        for mid, coord in zip(mids, coords):
            phase, date = meta.get(int(mid), (None, None))
            writer.writerow([int(mid), phase, date] + [round(float(x), 4) for x in coord])

    with open(clusters_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message_id", "phase", "date", "cluster", "pc1", "pc2"])
        for mid, coord, cl in zip(mids, coords, clusters):
            phase, date = meta.get(int(mid), (None, None))
            writer.writerow([
                int(mid), phase, date, int(cl),
                round(float(coord[0]), 4),
                round(float(coord[1]), 4),
            ])

    with open(var_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["composante", "variance_expliquee", "cumul"])
        for i, (v, c) in enumerate(zip(var_ratio, cum_var), start=1):
            writer.writerow([f"PC{i}", round(float(v), 5), round(float(c), 5)])

    log.info(f"Écrit :\n  {pca_path}\n  {clusters_path}\n  {var_path}")

    # ── Diagnostic : table de contingence phase × cluster ───────────────────
    # Utile pour lire immédiatement si K-means retrouve les phases
    phases_order = ["P1", "P2", "P3"]
    crosstab = {p: [0] * args.n_clusters for p in phases_order}
    missing = 0
    for mid, cl in zip(mids, clusters):
        phase, _ = meta.get(int(mid), (None, None))
        if phase in crosstab:
            crosstab[phase][int(cl)] += 1
        else:
            missing += 1

    log.info("\nTable de contingence phase × cluster :")
    header = "       " + "  ".join(f"C{i:<2}" for i in range(args.n_clusters))
    log.info(header)
    for p in phases_order:
        row = "  ".join(f"{n:>3}" for n in crosstab[p])
        log.info(f"  {p} : {row}")
    if missing:
        log.info(f"  (hors phases : {missing})")


if __name__ == "__main__":
    main()
