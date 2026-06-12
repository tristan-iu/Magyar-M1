#!/usr/bin/env python3
"""
Detection de Magyar (Robert Brovdi) et comptage d'individus distincts
dans les keyframes du corpus.

Pipeline par message :
  1. InsightFace detecte tous les visages dans chaque keyframe
  2. Chaque embedding est compare a la reference Magyar (cosine similarity)
  3. Tous les embeddings du message sont clusterises via DBSCAN (distance cosine)
     → nombre d'individus distincts
  4. CSV per-face avec position temporelle pour analyse intra-video

Champs JSONL produits :
  - visages_magyar_ratio          (float) — proportion de frames avec Magyar
  - visages_densite               (float) — nb moyen de visages par frame
  - visages_magyar_similarite_max (float) — score cosine max vs reference
  - visages_unique                (int)   — nb d'individus distincts (DBSCAN)
  - visages_magyar_detections     (int)   — nb de detections dans le cluster Magyar
                                            (presence : visages_magyar_detections > 0)

CSV per-face (1 ligne = 1 visage detecte) :
  message_id, frame_filename, frame_index, frame_position,
  face_index, cluster_id, is_magyar, similarity, det_score,
  n_faces_in_frame, date, phase

Options CLI : --threshold, --cluster-eps, --limit, --ids, --overwrite

Usage :
    python detect_magyar.py
    python detect_magyar.py --limit 10
    python detect_magyar.py --ids 42 138 --overwrite
    python detect_magyar.py --threshold 0.35 --cluster-eps 0.55
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
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

# CSV per-face : 1 ligne = 1 visage detecte dans 1 keyframe
CSV_FIELDNAMES = [
    "message_id", "frame_filename", "frame_index", "frame_position",
    "face_index", "cluster_id", "is_magyar", "similarity", "det_score",
    "n_faces_in_frame", "date", "phase",
]


def lire_image_safe(path: str):
    """Lit une image de manière sécurisée (chemins avec accents/unicode).

    Entrée : path — chemin vers le fichier image
    Sortie : ndarray BGR (OpenCV), ou None si lecture impossible
    """
    import cv2
    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcule la similarité cosine entre deux vecteurs d'embeddings.

    Entrée : a, b — vecteurs numpy 1D (embeddings InsightFace 512d)
    Sortie : float entre 0 et 1 (0 si l'un des vecteurs est nul)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def analyser_index_frame(filename: str) -> int:
    """Extrait l'index numérique d'un nom de keyframe.

    Exemple : 'robert_magyar_123_kf_0005.png' → 5
    Pour les photos directes (pas de _kf_), retourne 1.

    Entrée : filename — nom de fichier (str)
    Sortie : index int (1-based)
    """
    m = re.search(r"_kf_(\d+)", filename)
    return int(m.group(1)) if m else 1


def clusteriser_visages(embeddings: list[np.ndarray], eps: float = 0.55) -> np.ndarray:
    """Clusterise les embeddings faciaux via DBSCAN sur distance cosine.

    eps = seuil de distance cosine (1 − similarité) : eps=0.55 → similarité
    minimale 0.45 pour que deux visages soient le même individu.
    min_samples=1 garantit qu'aucun visage n'est classé bruit (pas de label -1).

    Entrée : embeddings — liste de vecteurs numpy 512d, eps — seuil distance cosine
    Sortie : array numpy de labels de cluster (int, 0-based)
    """
    n = len(embeddings)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])

    # Matrice (N, 512)
    emb_matrix = np.stack(embeddings)

    # L2-normaliser (InsightFace le fait deja, mais par securite)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    emb_norm = emb_matrix / norms

    # Matrice de distance cosine : dist = 1 - sim
    sim_matrix = emb_norm @ emb_norm.T
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

    # DBSCAN avec metric precomputed
    clustering = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(dist_matrix)

    return labels


def charger_msgs_termines(csv_path: Path) -> set[int]:
    """Charge les message_id déjà présents dans le CSV (idempotence).

    Entrée : csv_path — Path vers le CSV per-face
    Sortie : set d'int (message_id) — vide si le fichier n'existe pas
    """
    done = set()
    if not csv_path.is_file():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(int(row["message_id"]))
    return done


def main():
    parser = creer_parser_base(
        "Detection de Magyar + comptage individus distincts (InsightFace + DBSCAN)",
        has_input=False, has_output=False,
    )
    parser.add_argument("--input", default=None,
                        help="JSONL source (defaut depuis config)")
    parser.add_argument("--output", default=None,
                        help="JSONL enrichi (defaut: messages_faces.jsonl)")
    parser.add_argument(
        "--embedding",
        default=str(SCRIPT_DIR / "references" / "magyar_embedding.npy"),
        help="Chemin .npy embedding de reference",
    )
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Seuil cosine similarity pour Magyar (defaut: 0.4)")
    parser.add_argument("--cluster-eps", type=float, default=0.55,
                        help="DBSCAN eps en distance cosine (defaut: 0.55, "
                             "i.e. similarite >= 0.45 pour meme individu)")
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR / "results" / "magyar_detection.csv"),
        help="CSV de sortie per-face",
    )
    args = parser.parse_args()

    # Config
    cfg = load_config(args.config) if args.config else load_config()
    log = init_logger("insightface", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])
    fiches_dir = Path(cfg["paths"]["fiches_dir"])
    save_every = 50

    # Chemins I/O
    input_path = (Path(args.input) if args.input
                  else corpus_base / cfg["paths"]["jsonl_clean"])
    output_path = (Path(args.output) if args.output
                   else corpus_base / "messages_faces.jsonl")
    embedding_path = Path(args.embedding)
    csv_path = Path(args.csv)

    # Verifications
    if not input_path.is_file():
        log.error(f"JSONL introuvable : {input_path}")
        sys.exit(1)
    if not embedding_path.is_file():
        log.error(f"Embedding introuvable : {embedding_path}. "
                  "Lancez build_reference.py d'abord.")
        sys.exit(1)

    # Charger embedding de reference
    ref_embedding = np.load(embedding_path)
    log.info(f"Embedding de reference charge ({ref_embedding.shape[0]}d)")

    # Charger InsightFace
    from insightface.app import FaceAnalysis

    model_pack = cfg.get("models", {}).get("insightface", {}).get(
        "model_pack", "buffalo_l")
    app = FaceAnalysis(
        name=model_pack,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    log.info(f"InsightFace charge (modele: {model_pack})")

    # Charger JSONL
    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    # Filtrer messages avec keyframes ou photos
    filtre_ids = set(args.ids) if args.ids else None
    indices_eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video", "photo"],
        champs_a_verifier=["visages_magyar_detections", "visages_unique"],
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    # Pour les photos, exiger un media_chemin ; pour les videos, on glob les
    # keyframes dans la boucle (kf_files vide → skip avec warning)
    indices_eligibles = [
        i for i in indices_eligibles
        if messages[i].get("media_type") == "video"
        or messages[i].get("media_chemin")
    ]

    if args.limit:
        indices_eligibles = indices_eligibles[:args.limit]

    n_eligibles = len(indices_eligibles)
    log.info(f"Messages eligibles (keyframes + photos) : {n_eligibles}")
    log.info(f"Seuil Magyar : {args.threshold} | "
             f"DBSCAN eps : {args.cluster_eps}")

    if n_eligibles == 0:
        log.info("Rien a faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # CSV : idempotence au niveau message
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    msgs_termines = charger_msgs_termines(csv_path) if not args.overwrite else set()
    log.info(f"Messages deja dans le CSV : {len(msgs_termines)}")

    # Ouvrir CSV en append (ou ecriture si overwrite)
    csv_mode = "w" if args.overwrite else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if args.overwrite or not msgs_termines:
        csv_writer.writeheader()

    n_traites = 0
    n_erreurs = 0
    t0 = time.time()

    try:
        for rank, idx in enumerate(tqdm(indices_eligibles,
                                        desc="InsightFace", unit="msg")):
            msg = messages[idx]
            mid = msg["message_id"]
            channel = msg.get("canal", "robert_magyar")
            msg_date = msg.get("date", "")
            phase = etiquette_phase(msg_date, cfg) if msg_date else None

            # ── Lister les keyframes (ou photo directe) ──
            kf_pattern = f"{channel}_{mid}_kf_"
            kf_files = sorted(
                p for p in keyframes_dir.iterdir()
                if p.name.startswith(kf_pattern)
                and p.suffix in (".jpg", ".png")
            )

            est_photo = False
            if not kf_files:
                media_chemin_rel = msg.get("media_chemin", "")
                fichier_photo = corpus_base / media_chemin_rel if media_chemin_rel else None
                if fichier_photo and fichier_photo.is_file():
                    kf_files = [fichier_photo]
                    est_photo = True
                else:
                    log.warning(f"msg {mid} : aucune keyframe ni photo")
                    n_erreurs += 1
                    continue

            total_kf = len(kf_files)

            # ── Traiter chaque keyframe — collecter TOUS les embeddings ──
            # donnees_visages : toutes les detections du message, indexees par frame
            donnees_visages = []
            resumes_frames = []

            for kf_idx, kf_path in enumerate(kf_files):
                kf_name = kf_path.name
                frame_index = 1 if est_photo else analyser_index_frame(kf_name)

                img = lire_image_safe(str(kf_path))
                if img is None:
                    log.warning(f"msg {mid} : impossible de lire {kf_name}")
                    continue

                try:
                    faces = app.get(img)
                except Exception as e:
                    log.warning(
                        f"msg {mid} : InsightFace erreur sur {kf_name} : {e}")
                    continue

                n_faces = len(faces)
                magyar_dans_frame = False
                frame_sim_max = 0.0

                for fi, face in enumerate(faces):
                    sim = cosine_similarity(face.embedding, ref_embedding)
                    is_magyar = sim >= args.threshold
                    if is_magyar:
                        magyar_dans_frame = True
                    if sim > frame_sim_max:
                        frame_sim_max = sim

                    donnees_visages.append({
                        "embedding": face.embedding,
                        "frame_name": kf_name,
                        "frame_index": frame_index,
                        "kf_idx": kf_idx,
                        "face_index": fi,
                        "n_faces_in_frame": n_faces,
                        "similarity": sim,
                        "is_magyar": is_magyar,
                        "det_score": float(face.det_score),
                    })

                resumes_frames.append({
                    "frame_name": kf_name,
                    "n_faces": n_faces,
                    "magyar_detected": magyar_dans_frame,
                    "max_similarity": frame_sim_max,
                })

            # ── Skip si aucune frame traitee ──
            if not resumes_frames:
                continue

            # ── Clustering DBSCAN ──
            n_uniques = 0
            n_detections_magyar = 0
            cluster_labels = np.array([], dtype=int)

            if donnees_visages:
                all_embeddings = [fd["embedding"] for fd in donnees_visages]
                cluster_labels = clusteriser_visages(
                    all_embeddings, eps=args.cluster_eps)
                n_uniques = int(len(set(cluster_labels)))

                # Identifier le(s) cluster(s) Magyar
                clusters_magyar = set()
                for i, fd in enumerate(donnees_visages):
                    if fd["is_magyar"]:
                        clusters_magyar.add(int(cluster_labels[i]))

                # Compter les detections dans les clusters Magyar
                for i, fd in enumerate(donnees_visages):
                    if int(cluster_labels[i]) in clusters_magyar:
                        n_detections_magyar += 1

            # ── Ecrire CSV per-face (apres clustering) ──
            if mid not in msgs_termines:
                for i, fd in enumerate(donnees_visages):
                    # Position dans la video : 0.0 (debut) → 1.0 (fin)
                    if total_kf > 1:
                        frame_pos = round(fd["kf_idx"] / (total_kf - 1), 4)
                    else:
                        frame_pos = 0.0

                    cid = int(cluster_labels[i]) if len(cluster_labels) > i else -1

                    csv_writer.writerow({
                        "message_id": mid,
                        "frame_filename": fd["frame_name"],
                        "frame_index": fd["frame_index"],
                        "frame_position": frame_pos,
                        "face_index": fd["face_index"],
                        "cluster_id": cid,
                        "is_magyar": fd["is_magyar"],
                        "similarity": round(fd["similarity"], 4),
                        "det_score": round(fd["det_score"], 4),
                        "n_faces_in_frame": fd["n_faces_in_frame"],
                        "date": msg_date,
                        "phase": phase or "",
                    })

                # Ecrire aussi les frames sans visages (pour completude)
                frames_avec_visages = {fd["frame_name"] for fd in donnees_visages}
                for kf_idx, fs in enumerate(resumes_frames):
                    if fs["frame_name"] not in frames_avec_visages:
                        if total_kf > 1:
                            frame_pos = round(kf_idx / (total_kf - 1), 4)
                        else:
                            frame_pos = 0.0
                        csv_writer.writerow({
                            "message_id": mid,
                            "frame_filename": fs["frame_name"],
                            "frame_index": analyser_index_frame(fs["frame_name"]),
                            "frame_position": frame_pos,
                            "face_index": -1,
                            "cluster_id": -1,
                            "is_magyar": False,
                            "similarity": 0.0,
                            "det_score": 0.0,
                            "n_faces_in_frame": 0,
                            "date": msg_date,
                            "phase": phase or "",
                        })

            # ── Agregation niveau message ──
            total_frames = len(resumes_frames)
            frames_avec_magyar = sum(
                1 for fs in resumes_frames if fs["magyar_detected"])
            ratio_magyar = (round(frames_avec_magyar / total_frames, 4)
                           if total_frames > 0 else 0.0)
            moy_visages = (
                round(sum(fs["n_faces"] for fs in resumes_frames)
                      / total_frames, 2)
                if total_frames > 0 else 0.0)
            max_sim = max(
                (fs["max_similarity"] for fs in resumes_frames), default=0.0)

            # ── Enrichir JSONL ──
            msg["visages_magyar_ratio"] = ratio_magyar
            msg["visages_densite"] = moy_visages
            msg["visages_magyar_similarite_max"] = round(max_sim, 4)
            msg["visages_unique"] = n_uniques
            msg["visages_magyar_detections"] = n_detections_magyar

            # ── Enrichir fiche ──
            resume_clusters = []
            if donnees_visages and len(cluster_labels) > 0:
                for cid in sorted(set(cluster_labels)):
                    mask = cluster_labels == cid
                    c_sims = [donnees_visages[i]["similarity"]
                              for i in range(len(donnees_visages)) if mask[i]]
                    c_magyar = [donnees_visages[i]["is_magyar"]
                                for i in range(len(donnees_visages)) if mask[i]]
                    c_frames = set(
                        donnees_visages[i]["frame_name"]
                        for i in range(len(donnees_visages)) if mask[i])
                    # Position temporelle du cluster : premiere et derniere frame
                    c_positions = [
                        donnees_visages[i]["kf_idx"] / max(total_kf - 1, 1)
                        for i in range(len(donnees_visages)) if mask[i]
                    ]
                    resume_clusters.append({
                        "cluster_id": int(cid),
                        "n_detections": int(mask.sum()),
                        "n_frames": len(c_frames),
                        "is_magyar": any(c_magyar),
                        "avg_similarity_to_ref": round(
                            float(np.mean(c_sims)), 4),
                        "first_position": round(min(c_positions), 4),
                        "last_position": round(max(c_positions), 4),
                    })

            champs_fiche = {
                "visages_magyar_ratio": ratio_magyar,
                "visages_densite": moy_visages,
                "visages_magyar_similarite_max": round(max_sim, 4),
                "visages_unique": n_uniques,
                "visages_magyar_detections": n_detections_magyar,
                "visages_total_frames": total_frames,
                "visages_frames_avec_magyar": frames_avec_magyar,
                "visages_seuil": args.threshold,
                "visages_cluster_eps": args.cluster_eps,
                "visages_clusters": resume_clusters,
                "visages_details": [
                    {
                        "frame": fs["frame_name"],
                        "n_faces": fs["n_faces"],
                        "magyar_detected": fs["magyar_detected"],
                        "max_similarity": round(fs["max_similarity"], 4),
                    }
                    for fs in resumes_frames
                ],
            }
            mettre_a_jour_fiche(msg, champs_fiche, fiches_dir,
                                overwrite=args.overwrite)

            n_traites += 1

            tqdm.write(
                f"  msg {mid}: {len(donnees_visages)} visages, "
                f"{n_uniques} individus, "
                f"magyar={frames_avec_magyar}/{total_frames} frames"
            )

            # Sauvegarde intermediaire
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
        f"{n_traites} enrichis, {skipped} skippes, {n_erreurs} erreurs."
    )
    log.info(f"CSV : {csv_path}")
    log.info(f"JSONL : {output_path}")


if __name__ == "__main__":
    main()
