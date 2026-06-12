#!/usr/bin/env python3
"""
Détection de scènes standalone via PySceneDetect.
Enrichit le JSONL avec scene_coupes, scene_coupes_par_min, scene_duree_moyenne.
Produit un CSV par vidéo et un graphique mensuel des coupes/min.

Pipeline :
  1. Chargement JSONL + CSV existant (idempotence)
  2. ContentDetector sur chaque vidéo (fallback HistogramDetector si sur-découpage)
  3. Écriture CSV per-vidéo + enrichissement JSONL + fiche individuelle
  4. Graphique mensuel des coupes/min par phase

Options CLI : --input, --output, --csv, --threshold, --min-scene-len,
              --limit, --ids, --overwrite, --aggregate-only
"""

import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    write_jsonl,
    mettre_a_jour_fiche,
    etiquette_phase,
    filtrer_eligibles,
    fmt_eta,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FIELDNAMES = [
    "message_id", "date", "phase", "duration_sec",
    "n_scenes", "cuts_per_minute", "avg_scene_duration",
]


def charger_csv_termine(csv_path: Path) -> dict[int, dict]:
    """Charge le CSV existant pour permettre la reprise sans retraitement.

    Entrée : csv_path — Path vers le fichier CSV
    Sortie : dict {message_id (int): row dict} — vide si le fichier n'existe pas
    """
    done = {}
    if not csv_path.is_file():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done[int(row["message_id"])] = row
    return done


def detecter_scenes_video(video_path: str, threshold: float,
                        min_scene_len: int, log, mid: int,
                        fallback_max_cuts: int,
                        fallback_max_duration_s: float,
                        histogram_threshold: float) -> dict:
    """Détecte les scènes avec ContentDetector, fallback HistogramDetector.

    Si ContentDetector produit un nombre anormalement élevé de coupes sur une
    vidéo courte (> fallback_max_cuts coupes pour < fallback_max_duration_s
    secondes), on re-détecte avec HistogramDetector — plus robuste aux
    changements de luminosité rapides (cf. config.yaml §models.scenedetect).

    Entrée : video_path — chemin vidéo, threshold — seuil ContentDetector,
             min_scene_len — longueur min de scène en frames, log — logger,
             mid — id message (pour les warnings),
             fallback_max_cuts / fallback_max_duration_s — déclenchent le
             fallback HistogramDetector si dépassés,
             histogram_threshold — seuil HistogramDetector (delta YUV).
    Sortie : dict {n_scenes, cuts_per_minute, avg_scene_duration, duration_sec}
    """
    from scenedetect import detect, ContentDetector

    try:
        scenes = detect(video_path, ContentDetector(
            threshold=threshold, min_scene_len=min_scene_len
        ))
    except (RuntimeError, OSError, ValueError) as e:
        # On remonte en erreur : sinon "0 coupes" est un faux signal indistinguable d'un plan-séquence.
        log.error(f"msg {mid} : scenedetect erreur (vidéo cassée ou décodage HS) : {e}")
        return {"n_scenes": 0, "cuts_per_minute": 0.0, "avg_scene_duration": 0.0, "duration_sec": 0.0}

    n_scenes = len(scenes)
    n_cuts = max(n_scenes - 1, 0)

    # Durée totale de la vidéo
    if scenes:
        duration = scenes[-1][1].get_seconds()
    else:
        # Durée directement depuis la vidéo si aucune scène détectée
        try:
            from scenedetect import open_video
            video = open_video(video_path)
            duration = video.duration.get_seconds()
        except Exception:
            duration = 0.0

    # Fallback : trop de coupes sur vidéo courte → HistogramDetector plus robuste
    if n_cuts > fallback_max_cuts and duration < fallback_max_duration_s:
        try:
            from scenedetect import HistogramDetector
            scenes = detect(video_path, HistogramDetector(
                threshold=histogram_threshold, min_scene_len=min_scene_len
            ))
            n_scenes = len(scenes)
            n_cuts = max(n_scenes - 1, 0)
            if scenes:
                duration = scenes[-1][1].get_seconds()
        except Exception as e:
            log.warning(f"msg {mid} : HistogramDetector fallback erreur : {e}")

    cuts_per_min = round(n_cuts / (duration / 60), 2) if duration > 0 else 0.0
    avg_scene_dur = round(duration / n_scenes, 2) if n_scenes > 0 else duration

    return {
        "n_scenes": n_scenes,
        "cuts_per_minute": cuts_per_min,
        "avg_scene_duration": avg_scene_dur,
        "duration_sec": round(duration, 2),
    }


def tracer_coupes_mensuelles(csv_path: Path, out_path: Path, cfg: dict):
    """Trace la courbe mensuelle des coupes/min avec bandes de phases.

    Entrée : csv_path — Path vers le CSV scene_detection, out_path — Path PNG de sortie,
             cfg — dict config (phases, couleurs, style matplotlib)
    Sortie : None (écrit le PNG sur disque)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("month")["cuts_per_minute"].mean().reset_index()
    monthly.columns = ["month", "avg_cuts_per_min"]

    # On applique le style matplotlib depuis config
    style = cfg.get("viz", {}).get("plt_style", {})
    for k, v in style.items():
        try:
            plt.rcParams[k] = v
        except KeyError:
            pass

    phases_cfg = cfg.get("phases", {})
    fig, ax = plt.subplots(figsize=(14, 6))

    # On trace les bandes colorées par phase
    for pid, pdata in phases_cfg.items():
        ax.axvspan(
            pd.Timestamp(pdata["start"]), pd.Timestamp(pdata["end"]),
            alpha=0.1, color=pdata["color"], label=f"{pid} ({pdata['label']})"
        )

    ax.plot(monthly["month"], monthly["avg_cuts_per_min"],
            marker="o", markersize=5, linewidth=1.5, color="0.3")

    ax.set_title("Coupes par minute — moyenne mensuelle (PySceneDetect)")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Coupes / minute")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = creer_parser_base(
        "PySceneDetect standalone — detection de scenes",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None, help="JSONL source (defaut depuis config)")
    parser.add_argument("--output", default=None, help="JSONL enrichi (defaut: messages_scenedetect.jsonl)")
    parser.add_argument("--csv", default=str(SCRIPT_DIR / "results" / "scene_detection.csv"))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Seuil ContentDetector (defaut: config.yaml models.scenedetect.content_threshold)")
    parser.add_argument("--min-scene-len", type=int, default=None,
                        help="Longueur min de scene en frames (defaut: config.yaml)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Ne produire que le graphique depuis le CSV existant")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("scene_detect", cfg=cfg)

    # On résout les seuils : CLI prioritaire, sinon config.yaml, sinon dur (sécurité).
    sd_cfg = cfg.get("models", {}).get("scenedetect", {})
    threshold = args.threshold if args.threshold is not None else sd_cfg.get("content_threshold", 27.0)
    min_scene_len = args.min_scene_len if args.min_scene_len is not None else sd_cfg.get("min_scene_len", 15)
    fallback_max_cuts = sd_cfg.get("fallback_max_cuts", 50)
    fallback_max_duration_s = sd_cfg.get("fallback_max_duration_s", 30)
    histogram_threshold = sd_cfg.get("histogram_threshold", 0.05)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    fiches_dir = Path(cfg["paths"]["fiches_dir"])
    save_every = 50

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = csv_path.parent

    # Aggregate-only mode
    if args.aggregate_only:
        if not csv_path.is_file():
            log.error(f"CSV introuvable : {csv_path}")
            sys.exit(1)
        tracer_coupes_mensuelles(csv_path, results_dir / "scene_monthly.png", cfg)
        return

    input_path = Path(args.input) if args.input else corpus_base / cfg["paths"]["jsonl_clean"]
    output_path = Path(args.output) if args.output else corpus_base / "messages_scenedetect.jsonl"

    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    # On filtre les vidéos éligibles
    filtre_ids = set(args.ids) if args.ids else None
    indices_eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video"],
        champs_a_verifier=["scene_coupes", "scene_coupes_par_min"] if not args.overwrite else None,
        overwrite=args.overwrite,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Also check CSV for idempotence
    csv_termine = charger_csv_termine(csv_path) if not args.overwrite else {}
    if not args.overwrite:
        indices_eligibles = [i for i in indices_eligibles
                           if messages[i]["message_id"] not in csv_termine]
    if args.limit:
        indices_eligibles = indices_eligibles[:args.limit]

    n_eligibles = len(indices_eligibles)
    log.info(f"Videos eligibles : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien a faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        # Still generate plot if CSV exists
        if csv_path.is_file():
            tracer_coupes_mensuelles(csv_path, results_dir / "scene_monthly.png", cfg)
        return

    # Open CSV
    csv_mode = "w" if args.overwrite else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if args.overwrite or not csv_termine:
        csv_writer.writeheader()

    n_traites = 0
    n_erreurs = 0
    t0 = time.time()

    try:
        for rank, idx in enumerate(tqdm(indices_eligibles, desc="SceneDetect", unit="vid")):
            msg = messages[idx]
            mid = msg["message_id"]
            msg_date = msg.get("date", "")
            phase = etiquette_phase(msg_date, cfg) if msg_date else ""

            # Build video path
            media_chemin_rel = msg.get("media_chemin", "")
            if not media_chemin_rel:
                n_erreurs += 1
                continue
            fichier_video = corpus_base / media_chemin_rel
            if not fichier_video.is_file():
                log.warning(f"msg {mid} : video introuvable {fichier_video}")
                n_erreurs += 1
                continue

            result = detecter_scenes_video(
                str(fichier_video), threshold, min_scene_len, log, mid,
                fallback_max_cuts=fallback_max_cuts,
                fallback_max_duration_s=fallback_max_duration_s,
                histogram_threshold=histogram_threshold,
            )

            # Write CSV row
            csv_writer.writerow({
                "message_id": mid,
                "date": msg_date,
                "phase": phase or "",
                "duration_sec": result["duration_sec"],
                "n_scenes": result["n_scenes"],
                "cuts_per_minute": result["cuts_per_minute"],
                "avg_scene_duration": result["avg_scene_duration"],
            })

            # Enrich JSONL
            msg["scene_coupes"] = max(result["n_scenes"] - 1, 0)
            msg["scene_coupes_par_min"] = result["cuts_per_minute"]
            msg["scene_duree_moyenne"] = result["avg_scene_duration"]

            # Enrich fiche
            mettre_a_jour_fiche(msg, {
                "scene_coupes": msg["scene_coupes"],
                "scene_coupes_par_min": result["cuts_per_minute"],
                "scene_duree_moyenne": result["avg_scene_duration"],
                "scene_n_scenes": result["n_scenes"],
            }, fiches_dir, overwrite=True)

            n_traites += 1

            if n_traites % save_every == 0:
                write_jsonl(messages, output_path)
                csv_file.flush()
                log.info(f"  Sauvegarde intermediaire ({n_traites} traites)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        csv_file.close()
        write_jsonl(messages, output_path)

    elapsed = time.time() - t0
    skipped = n_eligibles - n_traites - n_erreurs
    log.info(
        f"\nTermine en {fmt_eta(elapsed)} — "
        f"{n_traites} traites, {skipped} skippes, {n_erreurs} erreurs."
    )
    log.info(f"CSV : {csv_path}")
    log.info(f"JSONL : {output_path}")

    # Generate plot
    if csv_path.is_file():
        tracer_coupes_mensuelles(csv_path, results_dir / "scene_monthly.png", cfg)


if __name__ == "__main__":
    main()
