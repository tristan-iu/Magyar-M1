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
  - faces_magyar_present   (bool)  — Magyar detecte dans au moins 1 frame
  - faces_magyar_ratio     (float) — proportion de frames avec Magyar
  - faces_avg_count        (float) — nb moyen de visages par frame
  - faces_max_similarity   (float) — score cosine max vs reference
  - faces_unique_count     (int)   — nb d'individus distincts (DBSCAN)
  - faces_magyar_detections (int)  — nb de detections dans le cluster Magyar

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
    build_base_parser,
    load_config,
    setup_logging,
    read_jsonl,
    write_jsonl,
    update_fiche,
    phase_label,
    filter_eligible,
    fmt_eta,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# CSV per-face : 1 ligne = 1 visage detecte dans 1 keyframe
CSV_FIELDNAMES = [
    "message_id", "frame_filename", "frame_index", "frame_position",
    "face_index", "cluster_id", "is_magyar", "similarity", "det_score",
    "n_faces_in_frame", "date", "phase",
]


def imread_safe(path: str):
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


def parse_frame_index(filename: str) -> int:
    """Extrait l'index numérique d'un nom de keyframe.

    Exemple : 'robert_magyar_123_kf_0005.png' → 5
    Pour les photos directes (pas de _kf_), retourne 1.

    Entrée : filename — nom de fichier (str)
    Sortie : index int (1-based)
    """
    m = re.search(r"_kf_(\d+)", filename)
    return int(m.group(1)) if m else 1


def cluster_faces(embeddings: list[np.ndarray], eps: float = 0.55) -> np.ndarray:
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


def load_existing_csv_messages(csv_path: Path) -> set[int]:
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
    parser = build_base_parser(
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
    log = setup_logging("insightface", cfg=cfg)

    corpus_base = Path(cfg["paths"]["corpus_base"])
    keyframes_dir = Path(cfg["paths"]["keyframes_dir"])
    fiches_dir = Path(cfg["paths"]["fiches_dir"])
    save_every = 50

    # Chemins I/O
    input_path = (Path(args.input) if args.input
                  else corpus_base / cfg["paths"]["jsonl_faces"])
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
    ids_filter = set(args.ids) if args.ids else None
    eligible_indices = filter_eligible(
        messages,
        ids_filter=ids_filter,
        check_fields=["faces_magyar_present", "faces_unique_count"],
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    eligible_indices = [
        i for i in eligible_indices
        if (messages[i].get("keyframes_count") or 0) > 0
        or (messages[i].get("media_type") == "photo"
            and messages[i].get("media_path"))
    ]

    if args.limit:
        eligible_indices = eligible_indices[:args.limit]

    n_eligible = len(eligible_indices)
    log.info(f"Messages eligibles (keyframes + photos) : {n_eligible}")
    log.info(f"Seuil Magyar : {args.threshold} | "
             f"DBSCAN eps : {args.cluster_eps}")

    if n_eligible == 0:
        log.info("Rien a faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # CSV : idempotence au niveau message
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done_msgs = load_existing_csv_messages(csv_path) if not args.overwrite else set()
    log.info(f"Messages deja dans le CSV : {len(done_msgs)}")

    # Ouvrir CSV en append (ou ecriture si overwrite)
    csv_mode = "w" if args.overwrite else "a"
    csv_file = open(csv_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if args.overwrite or not done_msgs:
        csv_writer.writeheader()

    processed = 0
    errors = 0
    t0 = time.time()

    try:
        for rank, idx in enumerate(tqdm(eligible_indices,
                                        desc="InsightFace", unit="msg")):
            msg = messages[idx]
            mid = msg["message_id"]
            channel = msg.get("channel", "robert_magyar")
            msg_date = msg.get("date", "")
            phase = phase_label(msg_date, cfg) if msg_date else None

            # ── Lister les keyframes (ou photo directe) ──
            kf_pattern = f"{channel}_{mid}_kf_"
            kf_files = sorted(
                p for p in keyframes_dir.iterdir()
                if p.name.startswith(kf_pattern)
                and p.suffix in (".jpg", ".png")
            )

            is_photo = False
            if not kf_files:
                media_path = msg.get("media_path", "")
                photo_file = corpus_base / media_path if media_path else None
                if photo_file and photo_file.is_file():
                    kf_files = [photo_file]
                    is_photo = True
                else:
                    log.warning(f"msg {mid} : aucune keyframe ni photo")
                    errors += 1
                    continue

            total_kf = len(kf_files)

            # ── Traiter chaque keyframe — collecter TOUS les embeddings ──
            # face_data : toutes les detections du message, indexees par frame
            face_data = []
            frame_summaries = []

            for kf_idx, kf_path in enumerate(kf_files):
                kf_name = kf_path.name
                frame_index = 1 if is_photo else parse_frame_index(kf_name)

                img = imread_safe(str(kf_path))
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
                frame_magyar = False
                frame_max_sim = 0.0

                for fi, face in enumerate(faces):
                    sim = cosine_similarity(face.embedding, ref_embedding)
                    is_magyar = sim >= args.threshold
                    if is_magyar:
                        frame_magyar = True
                    if sim > frame_max_sim:
                        frame_max_sim = sim

                    face_data.append({
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

                frame_summaries.append({
                    "frame_name": kf_name,
                    "n_faces": n_faces,
                    "magyar_detected": frame_magyar,
                    "max_similarity": frame_max_sim,
                })

            # ── Skip si aucune frame traitee ──
            if not frame_summaries:
                continue

            # ── Clustering DBSCAN ──
            n_unique = 0
            magyar_detections = 0
            cluster_labels = np.array([], dtype=int)

            if face_data:
                all_embeddings = [fd["embedding"] for fd in face_data]
                cluster_labels = cluster_faces(
                    all_embeddings, eps=args.cluster_eps)
                n_unique = int(len(set(cluster_labels)))

                # Identifier le(s) cluster(s) Magyar
                magyar_clusters = set()
                for i, fd in enumerate(face_data):
                    if fd["is_magyar"]:
                        magyar_clusters.add(int(cluster_labels[i]))

                # Compter les detections dans les clusters Magyar
                for i, fd in enumerate(face_data):
                    if int(cluster_labels[i]) in magyar_clusters:
                        magyar_detections += 1

            # ── Ecrire CSV per-face (apres clustering) ──
            if mid not in done_msgs:
                for i, fd in enumerate(face_data):
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
                frames_with_faces = {fd["frame_name"] for fd in face_data}
                for kf_idx, fs in enumerate(frame_summaries):
                    if fs["frame_name"] not in frames_with_faces:
                        if total_kf > 1:
                            frame_pos = round(kf_idx / (total_kf - 1), 4)
                        else:
                            frame_pos = 0.0
                        csv_writer.writerow({
                            "message_id": mid,
                            "frame_filename": fs["frame_name"],
                            "frame_index": parse_frame_index(fs["frame_name"]),
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
            total_frames = len(frame_summaries)
            frames_with_magyar = sum(
                1 for fs in frame_summaries if fs["magyar_detected"])
            magyar_ratio = (round(frames_with_magyar / total_frames, 4)
                           if total_frames > 0 else 0.0)
            avg_faces = (
                round(sum(fs["n_faces"] for fs in frame_summaries)
                      / total_frames, 2)
                if total_frames > 0 else 0.0)
            max_sim = max(
                (fs["max_similarity"] for fs in frame_summaries), default=0.0)

            # ── Enrichir JSONL ──
            msg["faces_magyar_present"] = frames_with_magyar > 0
            msg["faces_magyar_ratio"] = magyar_ratio
            msg["faces_avg_count"] = avg_faces
            msg["faces_max_similarity"] = round(max_sim, 4)
            msg["faces_unique_count"] = n_unique
            msg["faces_magyar_detections"] = magyar_detections

            # ── Enrichir fiche ──
            clusters_summary = []
            if face_data and len(cluster_labels) > 0:
                for cid in sorted(set(cluster_labels)):
                    mask = cluster_labels == cid
                    c_sims = [face_data[i]["similarity"]
                              for i in range(len(face_data)) if mask[i]]
                    c_magyar = [face_data[i]["is_magyar"]
                                for i in range(len(face_data)) if mask[i]]
                    c_frames = set(
                        face_data[i]["frame_name"]
                        for i in range(len(face_data)) if mask[i])
                    # Position temporelle du cluster : premiere et derniere frame
                    c_positions = [
                        face_data[i]["kf_idx"] / max(total_kf - 1, 1)
                        for i in range(len(face_data)) if mask[i]
                    ]
                    clusters_summary.append({
                        "cluster_id": int(cid),
                        "n_detections": int(mask.sum()),
                        "n_frames": len(c_frames),
                        "is_magyar": any(c_magyar),
                        "avg_similarity_to_ref": round(
                            float(np.mean(c_sims)), 4),
                        "first_position": round(min(c_positions), 4),
                        "last_position": round(max(c_positions), 4),
                    })

            fiche_fields = {
                "faces_magyar_present": frames_with_magyar > 0,
                "faces_magyar_ratio": magyar_ratio,
                "faces_avg_count": avg_faces,
                "faces_max_similarity": round(max_sim, 4),
                "faces_unique_count": n_unique,
                "faces_magyar_detections": magyar_detections,
                "faces_total_frames": total_frames,
                "faces_frames_with_magyar": frames_with_magyar,
                "faces_threshold": args.threshold,
                "faces_cluster_eps": args.cluster_eps,
                "faces_clusters": clusters_summary,
                "faces_details": [
                    {
                        "frame": fs["frame_name"],
                        "n_faces": fs["n_faces"],
                        "magyar_detected": fs["magyar_detected"],
                        "max_similarity": round(fs["max_similarity"], 4),
                    }
                    for fs in frame_summaries
                ],
            }
            update_fiche(msg, fiche_fields, fiches_dir,
                         overwrite=args.overwrite)

            processed += 1

            tqdm.write(
                f"  msg {mid}: {len(face_data)} visages, "
                f"{n_unique} individus, "
                f"magyar={frames_with_magyar}/{total_frames} frames"
            )

            # Sauvegarde intermediaire
            if processed % save_every == 0:
                write_jsonl(messages, output_path)
                csv_file.flush()
                log.info(f"  Sauvegarde intermediaire ({processed} traites)")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        csv_file.close()
        write_jsonl(messages, output_path)

    elapsed = time.time() - t0
    skipped = n_eligible - processed - errors
    log.info(
        f"\nTermine en {fmt_eta(elapsed)} — "
        f"{processed} enrichis, {skipped} skippes, {errors} erreurs."
    )
    log.info(f"CSV : {csv_path}")
    log.info(f"JSONL : {output_path}")


if __name__ == "__main__":
    main()
