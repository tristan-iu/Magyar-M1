#!/usr/bin/env python3
"""
Agregation des resultats de detection Magyar (CSV per-face) par message,
par phase et par mois. Produit des CSVs agreges et un graphique mensuel.

Pipeline :
  1. Lecture du CSV per-face produit par detect_magyar.py
  2. Agregation par message (% frames Magyar, nb individus, position temporelle)
  3. Agregation par phase
  4. Graphique mensuel (% presence Magyar + individus non-Magyar)

Options CLI : --csv, --output-dir, --config
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config, setup_logging  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def read_perface_csv(csv_path: Path) -> list[dict]:
    """Lit le CSV per-face et convertit les types numériques.

    Entrée : csv_path — Path vers le CSV produit par detect_magyar.py
    Sortie : liste de dicts avec types convertis (int, float, bool)
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["message_id"] = int(row["message_id"])
            row["frame_index"] = int(row["frame_index"])
            row["frame_position"] = float(row["frame_position"])
            row["face_index"] = int(row["face_index"])
            row["cluster_id"] = int(row["cluster_id"])
            row["is_magyar"] = row["is_magyar"].strip().lower() == "true"
            row["similarity"] = float(row["similarity"])
            row["det_score"] = float(row["det_score"])
            row["n_faces_in_frame"] = int(row["n_faces_in_frame"])
            rows.append(row)
    return rows


def aggregate_by_message(rows: list[dict]) -> list[dict]:
    """Agrège les lignes per-face en une ligne par message.

    Entrée : rows — liste de dicts (sortie de read_perface_csv)
    Sortie : liste de dicts agrégés (1 ligne par message_id)
    """
    msg_data = defaultdict(lambda: {
        "date": "", "phase": "",
        "frames": set(),
        "frames_with_magyar": set(),
        "faces": [],             # toutes les detections (face_index >= 0)
        "clusters": set(),
        "magyar_clusters": set(),
        "magyar_positions": [],  # frame_position des frames avec Magyar
        "all_positions": [],     # frame_position de toutes les frames
    })

    for row in rows:
        mid = row["message_id"]
        d = msg_data[mid]
        d["date"] = row["date"]
        d["phase"] = row["phase"]

        fname = row["frame_filename"]
        d["frames"].add(fname)
        d["all_positions"].append(row["frame_position"])

        if row["face_index"] >= 0:
            d["faces"].append(row)
            d["clusters"].add(row["cluster_id"])
            if row["is_magyar"]:
                d["frames_with_magyar"].add(fname)
                d["magyar_clusters"].add(row["cluster_id"])
                d["magyar_positions"].append(row["frame_position"])

    msg_rows = []
    for mid in sorted(msg_data.keys()):
        d = msg_data[mid]
        total_frames = len(d["frames"])
        n_faces = len(d["faces"])
        frames_w_magyar = len(d["frames_with_magyar"])
        n_unique = len(d["clusters"])
        n_non_magyar = n_unique - len(d["magyar_clusters"])

        pct_magyar = (round(frames_w_magyar / total_frames * 100, 2)
                      if total_frames > 0 else 0.0)
        avg_faces = (round(n_faces / total_frames, 2)
                     if total_frames > 0 else 0.0)
        max_sim = max((f["similarity"] for f in d["faces"]), default=0.0)

        # Position temporelle moyenne de Magyar dans la video
        magyar_avg_pos = (round(np.mean(d["magyar_positions"]), 4)
                          if d["magyar_positions"] else None)
        magyar_first_pos = (round(min(d["magyar_positions"]), 4)
                            if d["magyar_positions"] else None)
        magyar_last_pos = (round(max(d["magyar_positions"]), 4)
                           if d["magyar_positions"] else None)

        msg_rows.append({
            "message_id": mid,
            "date": d["date"],
            "phase": d["phase"],
            "total_frames": total_frames,
            "total_faces": n_faces,
            "frames_with_magyar": frames_w_magyar,
            "pct_magyar": pct_magyar,
            "avg_faces": avg_faces,
            "max_similarity": round(max_sim, 4),
            "n_unique_individuals": n_unique,
            "n_non_magyar": n_non_magyar,
            "magyar_avg_position": magyar_avg_pos,
            "magyar_first_position": magyar_first_pos,
            "magyar_last_position": magyar_last_pos,
        })

    return msg_rows


def aggregate_by_phase(msg_rows: list[dict]) -> list[dict]:
    """Agrège les lignes per-message en une ligne par phase.

    Entrée : msg_rows — sortie de aggregate_by_message
    Sortie : liste de dicts (1 ligne par phase, triée par nom de phase)
    """
    phase_data = defaultdict(lambda: {
        "n_messages": 0,
        "n_messages_with_magyar": 0,
        "pct_frames_magyar": [],
        "avg_faces_list": [],
        "unique_individuals": [],
        "non_magyar_list": [],
        "magyar_positions": [],
    })

    for row in msg_rows:
        phase = row["phase"]
        if not phase:
            continue
        pd = phase_data[phase]
        pd["n_messages"] += 1
        if row["frames_with_magyar"] > 0:
            pd["n_messages_with_magyar"] += 1
        pd["pct_frames_magyar"].append(row["pct_magyar"])
        pd["avg_faces_list"].append(row["avg_faces"])
        pd["unique_individuals"].append(row["n_unique_individuals"])
        pd["non_magyar_list"].append(row["n_non_magyar"])
        if row["magyar_avg_position"] is not None:
            pd["magyar_positions"].append(row["magyar_avg_position"])

    phase_rows = []
    for phase in sorted(phase_data.keys()):
        pd = phase_data[phase]
        n = pd["n_messages"]
        phase_rows.append({
            "phase": phase,
            "n_messages": n,
            "n_messages_with_magyar": pd["n_messages_with_magyar"],
            "pct_messages_with_magyar": (
                round(pd["n_messages_with_magyar"] / n * 100, 2)
                if n > 0 else 0.0),
            "avg_pct_frames_magyar": (
                round(np.mean(pd["pct_frames_magyar"]), 2)
                if pd["pct_frames_magyar"] else 0.0),
            "avg_faces_per_frame": (
                round(np.mean(pd["avg_faces_list"]), 2)
                if pd["avg_faces_list"] else 0.0),
            "avg_unique_individuals": (
                round(np.mean(pd["unique_individuals"]), 2)
                if pd["unique_individuals"] else 0.0),
            "avg_non_magyar_individuals": (
                round(np.mean(pd["non_magyar_list"]), 2)
                if pd["non_magyar_list"] else 0.0),
            "magyar_avg_position": (
                round(np.mean(pd["magyar_positions"]), 4)
                if pd["magyar_positions"] else None),
        })

    return phase_rows


def plot_monthly(msg_rows: list[dict], cfg: dict, output_dir: Path, log):
    """Trace deux courbes mensuelles : % frames Magyar + nb individus non-Magyar.

    Entrée : msg_rows — sortie de aggregate_by_message, cfg — dict config,
             output_dir — Path de sortie, log — logger
    Sortie : None (écrit magyar_monthly.png dans output_dir)
    """
    plt_style = cfg.get("viz", {}).get("plt_style", {})
    for k, v in plt_style.items():
        try:
            plt.rcParams[k] = v
        except (KeyError, ValueError):
            pass

    phase_colors = {}
    for pid, pdata in cfg.get("phases", {}).items():
        phase_colors[pid] = pdata.get("color", "#999999")

    # Agreger par mois
    monthly = defaultdict(lambda: {
        "pct_list": [], "non_magyar_list": [], "phase": "",
    })
    for row in msg_rows:
        if not row["date"]:
            continue
        month = row["date"][:7]
        monthly[month]["pct_list"].append(row["pct_magyar"])
        monthly[month]["non_magyar_list"].append(row["n_non_magyar"])
        monthly[month]["phase"] = row["phase"]

    months = sorted(monthly.keys())
    if not months:
        log.warning("Pas de donnees mensuelles pour le graphique.")
        return

    avg_pcts = [np.mean(monthly[m]["pct_list"]) for m in months]
    avg_non_magyar = [np.mean(monthly[m]["non_magyar_list"]) for m in months]
    phases = [monthly[m]["phase"] for m in months]
    colors = [phase_colors.get(p, "#999999") for p in phases]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Bandes de couleur par phase (sur les deux axes)
    phase_spans = cfg.get("phases", {})
    for ax in (ax1, ax2):
        for pid, pdata in phase_spans.items():
            start = pdata["start"][:7]
            end = pdata["end"][:7]
            idx_start = None
            idx_end = None
            for i, m in enumerate(months):
                if m >= start and idx_start is None:
                    idx_start = i
                if m <= end:
                    idx_end = i
            if idx_start is not None and idx_end is not None:
                ax.axvspan(
                    idx_start - 0.5, idx_end + 0.5,
                    alpha=0.1, color=pdata.get("color", "#999999"),
                    label=f"{pid} — {pdata.get('label', '')}",
                )

    # Courbe 1 : % frames avec Magyar
    ax1.plot(range(len(months)), avg_pcts, color="#333333",
             linewidth=1.5, zorder=3)
    ax1.scatter(range(len(months)), avg_pcts, c=colors, s=40,
                zorder=4, edgecolors="white", linewidths=0.5)
    ax1.set_ylabel("% frames avec Magyar")
    ax1.set_title("Presence de Magyar et individus non-Magyar par mois")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_ylim(bottom=0)

    # Courbe 2 : nb moyen d'individus non-Magyar
    ax2.plot(range(len(months)), avg_non_magyar, color="#555555",
             linewidth=1.5, zorder=3)
    ax2.scatter(range(len(months)), avg_non_magyar, c=colors, s=40,
                zorder=4, edgecolors="white", linewidths=0.5)
    ax2.set_ylabel("Individus non-Magyar (moy.)")
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    png_path = output_dir / "magyar_monthly.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Graphique : {png_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Agregation des resultats de detection Magyar (CSV per-face)"
    )
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR / "results" / "magyar_detection.csv"),
        help="CSV per-face produit par detect_magyar.py",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results"),
        help="Dossier de sortie pour CSVs agreges et graphique",
    )
    parser.add_argument("--config", default=None, help="Chemin config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = setup_logging("insightface", cfg=cfg)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        log.error(f"CSV introuvable : {csv_path}. "
                  "Lancez detect_magyar.py d'abord.")
        sys.exit(1)

    # Lire le CSV per-face
    rows = read_perface_csv(csv_path)
    log.info(f"{len(rows)} lignes lues depuis {csv_path}")

    if not rows:
        log.error("CSV vide.")
        sys.exit(1)

    # ── Agregation par message ──
    msg_rows = aggregate_by_message(rows)

    agg_msg_path = output_dir / "magyar_aggregated.csv"
    agg_fields = [
        "message_id", "date", "phase", "total_frames", "total_faces",
        "frames_with_magyar", "pct_magyar", "avg_faces", "max_similarity",
        "n_unique_individuals", "n_non_magyar",
        "magyar_avg_position", "magyar_first_position", "magyar_last_position",
    ]
    with open(agg_msg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(msg_rows)
    log.info(f"CSV agrege par message : {agg_msg_path} ({len(msg_rows)} msgs)")

    # ── Agregation par phase ──
    phase_rows = aggregate_by_phase(msg_rows)

    phase_path = output_dir / "magyar_by_phase.csv"
    phase_fields = [
        "phase", "n_messages", "n_messages_with_magyar",
        "pct_messages_with_magyar", "avg_pct_frames_magyar",
        "avg_faces_per_frame", "avg_unique_individuals",
        "avg_non_magyar_individuals", "magyar_avg_position",
    ]
    with open(phase_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=phase_fields)
        writer.writeheader()
        writer.writerows(phase_rows)

    log.info(f"CSV agrege par phase : {phase_path}")
    for pr in phase_rows:
        pos = pr["magyar_avg_position"]
        pos_str = f", pos moy={pos}" if pos is not None else ""
        log.info(
            f"  {pr['phase']} : "
            f"{pr['pct_messages_with_magyar']}% msgs avec Magyar, "
            f"{pr['avg_unique_individuals']} individus moy, "
            f"{pr['avg_non_magyar_individuals']} non-Magyar moy"
            f"{pos_str}"
        )

    # ── Graphique mensuel ──
    plot_monthly(msg_rows, cfg, output_dir, log)


if __name__ == "__main__":
    main()
