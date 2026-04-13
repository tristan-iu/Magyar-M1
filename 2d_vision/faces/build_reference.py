#!/usr/bin/env python3
"""
Calcul de l'embedding de reference de Magyar a partir de photos.
Detecte le visage principal dans chaque photo, calcule la moyenne L2-normalisee.

Pipeline :
  1. Lister les images dans references/magyar/
  2. InsightFace detecte le visage principal dans chaque photo
  3. Moyenne L2-normalisee des embeddings → references/magyar_embedding.npy

Options CLI : --references-dir, --output, --config
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path(__file__).resolve().parents[2] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import load_config, setup_logging  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser(
        description="Calcul embedding de reference Magyar (InsightFace)"
    )
    parser.add_argument(
        "--references-dir",
        default=str(Path(__file__).parent / "references" / "magyar"),
        help="Dossier contenant les photos de reference (defaut: references/magyar/)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "references" / "magyar_embedding.npy"),
        help="Chemin de sortie .npy (defaut: references/magyar_embedding.npy)",
    )
    parser.add_argument("--config", default=None, help="Chemin config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    log = setup_logging("insightface", cfg=cfg)

    refs_dir = Path(args.references_dir)
    if not refs_dir.is_dir():
        log.error(f"Dossier de references introuvable : {refs_dir}")
        sys.exit(1)

    # Lister les images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(
        p for p in refs_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        log.error(f"Aucune image trouvee dans {refs_dir}")
        sys.exit(1)

    log.info(f"{len(images)} photos de reference trouvees dans {refs_dir}")

    # Charger InsightFace
    from insightface.app import FaceAnalysis

    model_pack = cfg.get("models", {}).get("insightface", {}).get("model_pack", "buffalo_l")
    app = FaceAnalysis(name=model_pack, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    log.info(f"InsightFace charge (modele: {model_pack})")

    # Extraire les embeddings
    embeddings = []
    for img_path in images:
        img = imread_safe(str(img_path))
        if img is None:
            log.warning(f"  Impossible de lire : {img_path.name}")
            continue

        faces = app.get(img)
        if not faces:
            log.warning(f"  Aucun visage detecte : {img_path.name}")
            continue

        # Prendre le visage le plus grand (aire du bbox)
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(best.embedding)
        log.info(f"  {img_path.name} : visage detecte (score {best.det_score:.3f})")

    if not embeddings:
        log.error("Aucun embedding extrait. Verifiez les photos de reference.")
        sys.exit(1)

    # Moyenne L2-normalisee
    mean_emb = np.mean(embeddings, axis=0)
    mean_emb = mean_emb / np.linalg.norm(mean_emb)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, mean_emb)

    log.info(f"\nEmbedding sauvegarde : {output_path}")
    log.info(f"  {len(embeddings)}/{len(images)} photos utilisees")
    log.info(f"  Dimension : {mean_emb.shape[0]}")


if __name__ == "__main__":
    main()
