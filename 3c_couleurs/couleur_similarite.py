#!/usr/bin/env python3
"""
Matrice de similarité visuelle entre messages (distance cosinus HSV).

Lit les histogrammes 3D produits par couleur_batch.py, calcule la distance
cosinus entre tous les couples de messages, ordonne les messages par date.
La heatmap résultante montre directement l'homogénéisation : si les vidéos
P3 se ressemblent visuellement, le coin bas-droit sera uniformément
"similaire" (sombre en distance, clair en similarité).

Pipeline :
  1. Charge NPZ + JSONL (pour dates)
  2. Ordonne les messages par date
  3. Similarité cosinus = 1 − distance cosinus sur histogrammes normalisés
  4. Sauvegarde matrice NPZ + heatmap PNG avec frontières de phases

Sorties :
  - 4_data_et_viz/couleurs/couleur_similarite.npz — matrice (N, N) + message_ids ordonnés
  - 4_data_et_viz/couleurs/couleur_similarite.png — heatmap temporelle

Usage :
    python couleur_similarite.py
    python couleur_similarite.py --sample 300   # sous-échantillonne pour lisibilité
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        "Matrice de similarité cosinus HSV ordonnée par date",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source pour les dates (défaut: messages_blasons.jsonl)")
    parser.add_argument(
        "--npz",
        default=str(COULEUR_DIR / "couleur_histogrammes.npz"),
        help="NPZ histogrammes (produit par couleur_batch.py)",
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Sous-échantillonne à N messages pour la heatmap (défaut: tous)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Ne pas générer la heatmap PNG (matrice NPZ seulement)")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("couleurs_similarite", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    input_path = (Path(args.input) if args.input
                  else corpus_base / "messages_blasons.jsonl")
    npz_path = Path(args.npz)

    if not npz_path.is_file():
        log.error(f"NPZ introuvable : {npz_path}. Lancer couleur_batch.py d'abord.")
        sys.exit(1)

    # ── Chargement ──────────────────────────────────────────────────────────
    with np.load(npz_path) as data:
        mids = data["message_ids"]
        hists = data["histograms"]
    log.info(f"Matrice : {hists.shape}")

    messages = read_jsonl(input_path)
    dates = {m["message_id"]: m["date"] for m in messages if m.get("date")}

    # ── Tri par date ────────────────────────────────────────────────────────
    # On n'aligne que les messages dont on connaît la date
    pairs = [
        (int(mid), dates.get(int(mid)), i)
        for i, mid in enumerate(mids)
        if dates.get(int(mid))
    ]
    pairs.sort(key=lambda p: p[1])

    order_idx = np.array([p[2] for p in pairs])
    ordered_ids = np.array([p[0] for p in pairs], dtype=np.int64)
    ordered_dates = [p[1] for p in pairs]
    ordered_hists = hists[order_idx]

    # ── Sous-échantillonnage éventuel ───────────────────────────────────────
    if args.sample and args.sample < len(ordered_ids):
        step = len(ordered_ids) / args.sample
        take = np.round(np.arange(args.sample) * step).astype(int)
        ordered_ids = ordered_ids[take]
        ordered_dates = [ordered_dates[i] for i in take]
        ordered_hists = ordered_hists[take]
        log.info(f"Sous-échantillonnage : {len(ordered_ids)} messages")

    # ── Matrice de similarité cosinus ───────────────────────────────────────
    log.info("Calcul cosine_similarity...")
    sim = cosine_similarity(ordered_hists).astype(np.float32)
    log.info(f"Similarité : moyenne {sim.mean():.3f}, "
             f"médiane {np.median(sim):.3f}, min {sim.min():.3f}")

    # ── Sauvegarde NPZ ──────────────────────────────────────────────────────
    out_npz = COULEUR_DIR / "couleur_similarite.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        similarity=sim,
        message_ids=ordered_ids,
        dates=np.array(ordered_dates),
    )
    log.info(f"NPZ : {out_npz}")

    if args.no_plot:
        return

    # ── Heatmap PNG ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Frontières de phases : index où la phase change
    phases_seq = [etiquette_phase(d, cfg=cfg) for d in ordered_dates]
    boundaries = [
        i for i in range(1, len(phases_seq))
        if phases_seq[i] != phases_seq[i - 1]
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="magma", aspect="equal", vmin=0.0, vmax=1.0)

    # Lignes aux frontières de phases
    for b in boundaries:
        ax.axvline(b - 0.5, color="#00ff88", linewidth=0.8, alpha=0.8)
        ax.axhline(b - 0.5, color="#00ff88", linewidth=0.8, alpha=0.8)

    # Ticks temporels : ~8 labels datés
    n = len(ordered_dates)
    tick_positions = np.linspace(0, n - 1, 8).astype(int)
    tick_labels = [ordered_dates[i][:7] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_title("Similarité cosinus HSV (ordonnée par date)", fontsize=11)
    plt.colorbar(im, ax=ax, label="cos(θ)")
    plt.tight_layout()

    out_png = COULEUR_DIR / "couleur_similarite.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info(f"PNG : {out_png}")


if __name__ == "__main__":
    main()
