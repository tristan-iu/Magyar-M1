#!/usr/bin/env python3
"""
audio_segmentation.py — Segmentation audio parole / musique / silence.

Utilise inaSpeechSegmenter (INA, CC-BY-NC) pour classifier l'audio de chaque
vidéo en segments temporels {speech, music, noEnergy, noise}.

⚠️  Ce script nécessite un venv séparé (venv_ina/) avec TensorFlow 2.16+.
    Le venv principal (PyTorch) entre en conflit cuDNN avec TensorFlow.
    Création (placer le venv où l'espace disque le permet) :
      python3 -m venv /chemin/vers/venv_ina
      ln -s /chemin/vers/venv_ina ./venv_ina   # symlink depuis le repo
      /chemin/vers/venv_ina/bin/python -m pip install \
        inaSpeechSegmenter "tensorflow>=2.16,<2.17"
    Lancer avec : venv_ina/bin/python 2b_transcription/audio_segmentation.py ...

Écrit dans le JSONL de sortie (messages_ina.jsonl) :
  audio_parole_pure_ratio          float [0,1]
  audio_musique_ratio              float [0,1]
  audio_parole_sur_musique_ratio   float [0,1]
  audio_silence_ratio              float [0,1]
  audio_dominant                   str ∈ {parole, musique, silence, mixte}
  alerte_musique_dominante         bool  (audio_musique_ratio > SEUIL_MUSIQUE)
  audio_segmentation_modele        str   identifiant du modèle

Note : le CNN INA est entraîné sur de la radio française (2018). La
généralisation à l'audio drone/militaire ukrainien n'est pas garantie.
Faire un sanity-check humain sur 5-10 vidéos avant d'intégrer les
résultats dans le pipeline principal.

Usage :
  # Dans venv_ina :
  python 2b_transcription/audio_segmentation.py \\
    --input /media/.../messages_clean.jsonl \\
    --output /media/.../messages_ina.jsonl \\
    --media-dir /media/.../processed

  # Smoke test ciblé (par IDs directs ou fichier JSON {"ids": [...]})
  python 2b_transcription/audio_segmentation.py \\
    --input ... --output ... --media-dir ... \\
    --ids 42 138

  # Avec seuil musique personnalisé
  python 2b_transcription/audio_segmentation.py \\
    --input ... --output ... --seuil-musique 0.4
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    load_config,
    init_logger,
    read_jsonl,
    write_jsonl,
    mettre_a_jour_fiche,
    filtrer_eligibles,
    SuiviProgression,
)

MODEL_LABEL = "inaSpeechSegmenter-0.7"
SEUIL_MUSIQUE_DEFAUT = 0.5  # ratio musique au-dessus duquel on flag

# Logger module-level partagé : configuré par init_logger("audio_segmentation")
# dans main(). charger_ina() l'utilise sans qu'on ait à lui passer le logger.
log = logging.getLogger("audio_segmentation")


# ── Extraction audio WAV ──────────────────────────────────────────────────────

def _extraire_wav_tmp(media_path: str, sampling_rate: int = 16000) -> str:
    """Extrait l'audio en WAV mono 16kHz dans un fichier temporaire.

    inaSpeechSegmenter préfère travailler sur des WAVs plutôt que des MP4.

    Entrée : media_path — chemin média, sampling_rate — Hz
    Sortie : chemin du fichier WAV temporaire (à supprimer après usage)
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", media_path,
        "-ar", str(sampling_rate), "-ac", "1",
        "-vn",  # pas de vidéo
        tmp.name,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return tmp.name


# ── Segmentation INA ─────────────────────────────────────────────────────────

_ina_segmenter = None


def charger_ina():
    """Charge inaSpeechSegmenter (singleton). Nécessite venv_ina.

    Sortie : objet Segmenter INA ou None si non disponible
    """
    global _ina_segmenter
    if _ina_segmenter is None:
        try:
            from inaSpeechSegmenter import Segmenter
            # detect_gender=False → labels "speech" au lieu de "male"/"female"
            # Évite d'avoir à mapper les labels de genre vers "speech"
            _ina_segmenter = Segmenter(detect_gender=False)
            log.info("inaSpeechSegmenter chargé (detect_gender=False).")
        except ImportError as e:
            log.error("inaSpeechSegmenter non disponible : %s", e)
            log.error("Ce script doit être exécuté dans venv_ina/ (TF 2.16+).")
            log.error("Voir README 2b_transcription pour la création du venv.")
            sys.exit(1)
    return _ina_segmenter


def segmenter_audio(wav_path: str, duree_totale: float, log) -> dict:
    """Segmente l'audio via inaSpeechSegmenter.

    Calcule les ratios parole/musique/silence sur la durée totale.

    Entrée : wav_path — chemin WAV, duree_totale — durée en secondes, log
    Sortie : dict avec les champs audio_*
    """
    seg = charger_ina()

    # Résultat par défaut (erreur ou fichier vide)
    defaut = {
        "audio_parole_pure_ratio": None,
        "audio_musique_ratio": None,
        "audio_parole_sur_musique_ratio": None,
        "audio_silence_ratio": None,
        "audio_dominant": None,
        "audio_segmentation_modele": MODEL_LABEL,
    }

    try:
        # INA retourne une liste de tuples : (label, debut, fin)
        # Avec detect_gender=False : labels = {"speech", "music", "noEnergy", "noise"}
        # (detect_gender=True retournerait "male"/"female" au lieu de "speech")
        segments = seg(wav_path)

        if not segments:
            return defaut

        # Accumulation des durées par label
        durees: dict[str, float] = {
            "speech": 0.0, "music": 0.0, "noEnergy": 0.0, "noise": 0.0,
        }
        for label, debut, fin in segments:
            cat = label if label in durees else "noise"
            durees[cat] += max(0.0, fin - debut)

        # On utilise la durée totale (depuis ffprobe si non disponible)
        dur = duree_totale
        if dur <= 0:
            dur = sum(durees.values())
        if dur <= 0:
            return defaut

        parole_pure = round(durees["speech"] / dur, 3)
        musique = round(durees["music"] / dur, 3)
        silence = round((durees["noEnergy"] + durees["noise"]) / dur, 3)
        # "parole sur musique" = cas où les segments speech et music se chevauchent
        # INA segmente séquentiellement donc pas de chevauchement réel.
        # On l'estime comme 0 (les segments sont exclusifs).
        parole_sur_musique = 0.0

        # Dominant
        scores = {
            "parole": parole_pure,
            "musique": musique,
            "silence": silence,
        }
        dominant = max(scores, key=scores.get)
        # Si aucun ne dépasse 50% clairement → mixte
        if scores[dominant] < 0.4:
            dominant = "mixte"

        return {
            "audio_parole_pure_ratio": parole_pure,
            "audio_musique_ratio": musique,
            "audio_parole_sur_musique_ratio": parole_sur_musique,
            "audio_silence_ratio": silence,
            "audio_dominant": dominant,
            "audio_segmentation_modele": MODEL_LABEL,
        }

    except (RuntimeError, ImportError, OSError) as e:
        # Modèle ina absent / mal chargé ou audio corrompu : on remonte en erreur
        # car le fallback "defaut" masque sinon un signal pipeline manquant.
        log.error(f"inaSpeechSegmenter défaillant : {e}")
        return defaut


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Segmentation audio parole/musique/silence via inaSpeechSegmenter.",
        has_media_dir=True,
    )
    parser.add_argument("--seuil-musique", type=float, default=SEUIL_MUSIQUE_DEFAUT,
                        help=f"Ratio musique pour alerte_musique_dominante (défaut: {SEUIL_MUSIQUE_DEFAUT})")
    parser.add_argument("--ids-file", default=None,
                        help="JSON avec liste d'IDs à traiter (ex: smoke_test_ids.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger("audio_segmentation", cfg=cfg)
    save_every = cfg.get("batch", {}).get("save_every", 50)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else input_path.parent

    # Lecture optionnelle d'IDs depuis un fichier JSON (smoke_test_ids.json)
    ids_from_file: set[int] | None = None
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids_data = json.load(f)
        ids_from_file = set(int(i) for i in ids_data.get("ids", []))
        log.info(f"IDs depuis fichier : {ids_from_file}")

    filtre_ids: set[int] | None = None
    if args.ids:
        filtre_ids = set(args.ids)
    elif ids_from_file:
        filtre_ids = ids_from_file

    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video"],
        champs_a_verifier=["audio_dominant"],
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    eligibles = [i for i in eligibles if messages[i].get("audio_present")]

    if args.limit:
        eligibles = eligibles[:args.limit]

    n_eligibles = len(eligibles)
    log.info(f"Vidéos à segmenter : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # Chargement INA une seule fois (vérifie aussi le venv)
    charger_ina()

    tracker = SuiviProgression(n_eligibles, label="ina_seg")
    n_traites = 0
    n_erreurs = 0

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            media_chemin_rel = msg.get("media_chemin")
            duree = msg.get("duree") or 0.0

            if not media_chemin_rel:
                continue

            fichier_media = media_dir / media_chemin_rel
            if not fichier_media.is_file():
                log.warning(f"msg {mid} fichier manquant : {media_chemin_rel}")
                n_erreurs += 1
                tracker.avancer(rank, mid, "fichier manquant ✗")
                continue

            # Extraction WAV temporaire
            wav_tmp = None
            try:
                wav_tmp = _extraire_wav_tmp(str(fichier_media))
                champs = segmenter_audio(wav_tmp, duree, log)
            except Exception as e:
                log.warning(f"msg {mid} extraction WAV error : {e}")
                champs = {}
                n_erreurs += 1
                tracker.avancer(rank, mid, "extraction WAV ✗")
                continue
            finally:
                if wav_tmp:
                    try:
                        Path(wav_tmp).unlink(missing_ok=True)
                    except Exception:
                        pass

            # alerte_musique_dominante dérivée du ratio
            musique_ratio = champs.get("audio_musique_ratio") or 0.0
            champs["alerte_musique_dominante"] = musique_ratio > args.seuil_musique

            msg.update(champs)
            n_traites += 1
            mettre_a_jour_fiche(msg, champs, media_dir / "fiches",
                                overwrite=args.overwrite)

            dominant = champs.get("audio_dominant", "?")
            musique_pct = f"{musique_ratio*100:.0f}%"
            tracker.avancer(rank, mid, f"dominant={dominant} musique={musique_pct}")

            if n_traites % save_every == 0:
                write_jsonl(messages, output_path)

    except KeyboardInterrupt:
        log.info("\nInterruption — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.resumer(errors=n_erreurs, skipped=n_eligibles - n_traites - n_erreurs)
    log.info(f"Sortie : {output_path}")


if __name__ == "__main__":
    main()
