#!/usr/bin/env python3
"""
Transcription audio batch via Whisper + Silero-VAD.
Enrichit le JSONL avec : dialogue, parole_present, parole_duree, parole_ratio.
Le SRT est écrit à côté du média (chemin déductible : channel_mid.srt).
Les segments bruts ne sont plus persistés (ils gonflaient le JSONL de ~16 MB) :
conséquence — les `avg_logprob`/`no_speech_prob` calculés ici ne sont disponibles
qu'au moment du run, et `dialogue_confiance` (score QA composite dérivé de ces
signaux) n'est pas recalculable a posteriori sans re-transcrire.

Pipeline par message :
  1. VAD (Silero) — détecte la présence/durée de parole
  2. Whisper — transcription (si parole détectée ou --vad-gate désactivé)
  3. Filtrage qualité — rejette les hallucinations (confidence, ratio cyrillique)
  4. SRT — génère un fichier sous-titres à côté du média

Options CLI : --model, --language, --vad-threshold, --min-speech-ms, --min-silence-ms,
              --vad-gate, + options standard (--input, --output, --media-dir, --limit,
              --overwrite, --ids, --start-date, --end-date, --config)

Dépendances : openai-whisper, torch, torchaudio
"""
import logging
import math
import os
import re
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
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
    CYRILLIC_RE,
)

# Logger module-level partagé : configuré par init_logger("whisper") dans main().
# Les singletons de chargement (charger_whisper/charger_vad) l'utilisent sans
# qu'on ait à leur passer le logger en argument.
log = logging.getLogger("whisper")

LATIN_ARTIFACT_RE = re.compile(r"[a-zA-Z]+")


def fmt_srt_time(seconds: float) -> str:
    """Convertit des secondes en timestamp SRT HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


# ── Chargement modèles (une seule fois) ─────────────────────────────────────

_whisper_model = None
_vad_model = None
_vad_utils = None


def charger_whisper(model_name: str = "large-v3"):
    """Charge le modèle Whisper sur GPU si disponible, sinon CPU.

    Entrée : model_name — str identifiant du modèle (ex: "large-v3", "medium")
    Sortie : modèle Whisper chargé (singleton — chargé une seule fois par session)
    """
    global _whisper_model

    if _whisper_model is None:
        import whisper

        log.info("Chargement Whisper '%s' sur CPU...", model_name)
        _whisper_model = whisper.load_model(model_name, device="cpu")

        if torch.cuda.is_available():
            _whisper_model = _whisper_model.to(device="cuda", dtype=torch.float16)

            # Fix dtype: LayerNorm en fp32, reste en fp16
            for m in _whisper_model.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.float()

        p = next(_whisper_model.parameters())
        log.info("Whisper chargé (%s, %s)", p.device, p.dtype)

    return _whisper_model


def _lire_audio_ffmpeg(audio_path: str, sampling_rate: int = 16000) -> "torch.Tensor":
    """Lecture audio via ffmpeg subprocess — contourne torchaudio/torchcodec.

    Entrée : audio_path — chemin vers le fichier audio/vidéo, sampling_rate — int (Hz)
    Sortie : Tensor float32 mono (samples,)
    """
    import subprocess
    import numpy as np
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", str(sampling_rate), "-ac", "1", "-f", "f32le", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return torch.from_numpy(audio)


def charger_vad():
    """Charge Silero-VAD une seule fois."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        try:
            log.info("Chargement Silero-VAD...")
            _vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            _vad_utils = {
                "get_speech_timestamps": utils[0],
                "read_audio": _lire_audio_ffmpeg,  # bypass torchaudio/torchcodec
            }
            log.info("Silero-VAD chargé")
        except Exception as e:
            log.warning("Erreur chargement VAD : %s", e)
            return None, None
    return _vad_model, _vad_utils


# ── VAD ──────────────────────────────────────────────────────────────────────

def detecter_parole(audio_path: str, log,
                          threshold: float = 0.35,
                          min_speech_ms: int = 200,
                          min_silence_ms: int = 200) -> dict:
    """Détecte la parole via Silero-VAD.

    Entrée : audio_path — chemin vers le fichier, log — logger,
             threshold — seuil de détection VAD (0–1),
             min_speech_ms / min_silence_ms — durées minimales en ms
    Sortie : dict {"parole_presente": bool, "parole_duree": float (secondes)}
    """
    result = {"parole_presente": False, "parole_duree": 0.0}

    model, utils = charger_vad()
    if model is None or utils is None:
        # VAD indisponible → on suppose parole présente pour ne pas bloquer
        result["parole_presente"] = True
        return result

    try:
        SAMPLING_RATE = 16000
        wav = utils["read_audio"](audio_path, sampling_rate=SAMPLING_RATE)

        speech_timestamps = utils["get_speech_timestamps"](
            wav,
            model,
            sampling_rate=SAMPLING_RATE,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=True,
        )

        if speech_timestamps:
            total_speech = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
            result["parole_duree"] = round(total_speech, 2)
            result["parole_presente"] = True

    except Exception as e:
        log.warning(f"Erreur VAD : {e}")
        # Fallback optimiste
        result["parole_presente"] = True

    return result


# ── Transcription Whisper ────────────────────────────────────────────────────

def transcrire(audio_path: str, model_name: str, log,
               language: str = None, prompt: str = "") -> dict:
    """Transcrit l'audio via Whisper.

    Entrée : audio_path — chemin vers le fichier, model_name — str,
             log — logger, language — code langue forcé (ex: "uk") ou None,
             prompt — texte de conditionnement initial
    Sortie : dict {"text": str, "language": str, "segments": list, "confidence": float|None}
    """
    model = charger_whisper(model_name)
    result = {"text": "", "language": None, "segments": [], "confidence": None}

    try:
        transcription = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False,
            initial_prompt=prompt,
            # Sous 0.2 de no_speech_prob on considère qu'il y a de la parole : seuil
            # bas (défaut Whisper 0.6) car le pré-VAD Silero a déjà écarté le silence.
            no_speech_threshold=0.2,
            # Seuil log-probabilité min : rejette les segments où Whisper a peu de confiance
            logprob_threshold=-0.5,
            # Au-dessus de 2.0 = ratio de compression suspect = signal d'hallucination (boucles, répétitions)
            compression_ratio_threshold=2.0,
        )
        result["text"] = transcription.get("text", "").strip()
        result["language"] = transcription.get("language")
        result["segments"] = transcription.get("segments", [])

        # Confidence = exp(moyenne des avg_logprob)
        logprobs = [s["avg_logprob"] for s in result["segments"] if "avg_logprob" in s]
        if logprobs:
            result["confidence"] = round(math.exp(sum(logprobs) / len(logprobs)), 4)

    except (RuntimeError, OSError) as e:
        # OOM GPU = fatal pour ce message, on log en erreur pour ne pas masquer le trou dans le corpus.
        if "out of memory" in str(e).lower():
            log.error(f"GPU OOM Whisper sur {audio_path} : {e}")
        else:
            log.warning(f"Erreur Whisper (runtime/IO) sur {audio_path} : {e}")

    return result


# ── Nettoyage segments ───────────────────────────────────────────────────────

def nettoyer_segments(segments: list) -> list:
    """Normalise les segments Whisper (champs utiles uniquement).

    Sert à générer le SRT et à construire la chaîne `dialogue`. Les segments
    ne sont PAS persistés dans le JSONL (cf. docstring module) ; `avg_logprob`
    et `no_speech_prob` sont conservés ici seulement pour le run courant.
    """
    clean = []
    for s in segments:
        clean.append({
            "id": s.get("id"),
            "start": s.get("start"),
            "end": s.get("end"),
            "text": s.get("text", "").strip(),
            "avg_logprob": s.get("avg_logprob"),
            "no_speech_prob": s.get("no_speech_prob"),
        })
    return clean


# ── SRT ──────────────────────────────────────────────────────────────────────

def generer_srt(segments: list, output_path: str) -> bool:
    """Génère un fichier SRT à partir des segments nettoyés."""
    if not segments:
        return False
    try:
        idx = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for seg in segments:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                idx += 1
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                f.write(f"{idx}\n")
                f.write(f"{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n")
                f.write(f"{text}\n\n")
        return True
    except Exception:
        return False


# ── Filtre post-transcription ────────────────────────────────────────────────

def filtrer_segments(segments: list,
                    max_non_cyr_ratio: float = 0.4,
                    max_consecutive_latin: int = 4) -> list:
    """Filtre les segments individuels contenant des artefacts.

    Deux critères d'exclusion :
    - ratio non-cyrillique > seuil (hallucinations en alphabet latin)
    - bloc de lettres latines consécutives >= max_consecutive_latin

    Entrée : segments — liste de dicts Whisper nettoyés,
             max_non_cyr_ratio — seuil ratio non-cyrillique (0–1),
             max_consecutive_latin — longueur max d'un bloc latin autorisé
    Sortie : liste de segments filtrés (sous-ensemble de l'entrée)
    """
    filtered = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # Blocs de lettres latines consécutives
        if any(len(b) >= max_consecutive_latin for b in LATIN_ARTIFACT_RE.findall(text)):
            continue
        # Ratio non-cyrillique parmi les lettres
        letters = [c for c in text if c.isalpha()]
        if letters:
            cyrillic_count = sum(1 for c in letters if CYRILLIC_RE.match(c))
            if (1.0 - cyrillic_count / len(letters)) > max_non_cyr_ratio:
                continue
        filtered.append(seg)
    return filtered


def est_transcription_valide(confidence: float | None,
                           min_confidence: float = 0.35) -> bool:
    """Rejette si la confidence globale est trop basse."""
    if confidence is not None and confidence < min_confidence:
        return False
    return True



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = creer_parser_base(
        "Transcription audio batch (Whisper + Silero-VAD)",
        has_media_dir=True,
    )
    parser.add_argument("--model", default=None, help="Modèle Whisper (défaut: depuis config)")
    parser.add_argument("--language", default=None, help="Langue forcée (ex: uk, ru, en). Défaut: depuis config")
    parser.add_argument("--vad-threshold", type=float, default=None, help="Seuil VAD (défaut: depuis config)")
    parser.add_argument("--min-speech-ms", type=int, default=None, help="Durée min parole en ms")
    parser.add_argument("--min-silence-ms", type=int, default=None, help="Durée min silence en ms")
    parser.add_argument("--vad-gate", action="store_true",
                        help="Skip la transcription si le VAD ne détecte pas de parole")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = init_logger("whisper", cfg=cfg)
    save_every = 50

    # Defaults depuis config, CLI override
    whisper_cfg = cfg.get("models", {}).get("whisper", {})
    vad_cfg = cfg.get("models", {}).get("vad", {})
    model_name = args.model or whisper_cfg.get("model_name", "large-v3")
    language = args.language or whisper_cfg.get("language", "uk")
    whisper_prompt = whisper_cfg.get("prompt", "")
    vad_threshold = args.vad_threshold or vad_cfg.get("threshold", 0.35)
    min_speech_ms = args.min_speech_ms or vad_cfg.get("min_speech_ms", 200)
    min_silence_ms = args.min_silence_ms or vad_cfg.get("min_silence_ms", 200)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    media_dir = Path(args.media_dir).resolve() if args.media_dir else input_path.parent

    # ── Lecture JSONL ──
    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    messages = read_jsonl(input_path)
    total = len(messages)
    log.info(f"Corpus : {total} messages")

    # ── Filtrer les messages éligibles ──
    # Seuls media_type="video" + has_audio=true, et pas déjà traités
    filtre_ids = set(args.ids) if args.ids else None
    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        media_types=["video"],
        champs_a_verifier=["dialogue"],
        overwrite=args.overwrite,
        limit=None,  # appliqué après le filtre has_audio
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Filtre supplémentaire : audio_present
    eligibles = [i for i in eligibles if messages[i].get("audio_present")]

    if args.limit:
        eligibles = eligibles[:args.limit]

    n_eligibles = len(eligibles)
    log.info(f"Vidéos avec audio à transcrire : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # ── Chargement modèles (une seule fois) ──
    charger_vad()
    charger_whisper(model_name)

    tracker = SuiviProgression(n_eligibles, label="whisper")
    n_traites = 0
    n_erreurs = 0

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            channel = msg.get("canal", "robert_magyar")
            media_chemin_rel = msg.get("media_chemin")
            duration = msg.get("duree")  # ffprobe

            if not media_chemin_rel:
                continue

            fichier_media = media_dir / media_chemin_rel

            # ── Fichier manquant ──
            if not fichier_media.is_file():
                log.warning(f"msg {mid}\t{media_chemin_rel}\tfichier manquant")
                n_erreurs += 1
                tracker.avancer(rank, mid, "fichier manquant ✗")
                continue

            # ── VAD ──
            vad = detecter_parole(str(fichier_media), log,
                                        threshold=vad_threshold,
                                        min_speech_ms=min_speech_ms,
                                        min_silence_ms=min_silence_ms)

            if not vad["parole_presente"] and args.vad_gate:
                nouveaux_champs = {
                    "dialogue": "",
                    "parole_present": False,
                    "parole_duree": 0.0,
                    "parole_ratio": 0.0,
                }
                msg.update(nouveaux_champs)
                n_traites += 1
                mettre_a_jour_fiche(msg, nouveaux_champs, media_dir / "fiches", overwrite=args.overwrite)
                tracker.avancer(rank, mid, "pas de parole")
                if n_traites % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Whisper ──
            whisper_result = transcrire(str(fichier_media), model_name, log,
                                        language=language, prompt=whisper_prompt)

            parole_dur = vad["parole_duree"]
            parole_ratio = round(parole_dur / duration, 3) if duration and duration > 0 else 0.0

            if not whisper_result["text"]:
                nouveaux_champs = {
                    "dialogue": "",
                    "parole_present": False,
                    "parole_duree": parole_dur,
                    "parole_ratio": parole_ratio,
                }
                msg.update(nouveaux_champs)
                n_traites += 1
                mettre_a_jour_fiche(msg, nouveaux_champs, media_dir / "fiches", overwrite=args.overwrite)
                tracker.avancer(rank, mid, "transcription vide")
                if n_traites % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Filtre qualité post-transcription ──
            cleaned = nettoyer_segments(whisper_result["segments"])

            # 1. Rejet global si confidence trop basse
            if not est_transcription_valide(whisper_result["confidence"]):
                cleaned = []
            else:
                # 2. Filtrage par segment (artefacts latins, ratio non-cyrillique)
                cleaned = filtrer_segments(cleaned)

            if not cleaned:
                nouveaux_champs = {
                    "dialogue": "",
                    "parole_present": False,
                    "parole_duree": parole_dur,
                    "parole_ratio": parole_ratio,
                }
                msg.update(nouveaux_champs)
                n_traites += 1
                mettre_a_jour_fiche(msg, nouveaux_champs, media_dir / "fiches", overwrite=args.overwrite)
                conf = f"{whisper_result['confidence']:.2f}" if whisper_result["confidence"] else "?"
                tracker.avancer(rank, mid, f"rejeté (conf={conf})")
                if n_traites % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Résultats ──
            dialogue = " ".join(s["text"] for s in cleaned if s.get("text"))

            # SRT (dans le même dossier que le média) — chemin déductible :
            # {fiches_dir}/{canal}_{message_id}.srt
            srt_filename = f"{channel}_{mid}.srt"
            media_folder = fichier_media.parent
            generer_srt(cleaned, str(media_folder / srt_filename))

            nouveaux_champs = {
                "dialogue": dialogue,
                "parole_present": True,
                "parole_duree": parole_dur,
                "parole_ratio": parole_ratio,
            }
            msg.update(nouveaux_champs)
            n_traites += 1
            mettre_a_jour_fiche(msg, nouveaux_champs, media_dir / "fiches", overwrite=args.overwrite)

            lang = whisper_result["language"] or "?"
            conf = f"{whisper_result['confidence']:.2f}" if whisper_result["confidence"] else "?"
            n_seg = len(cleaned)
            n_raw = len(whisper_result["segments"])
            dropped = f", -{n_raw - n_seg} filtrés" if n_seg < n_raw else ""
            tracker.avancer(rank, mid, f"{lang}, {n_seg} seg{dropped}, conf={conf}")

            if n_traites % save_every == 0:
                write_jsonl(messages, output_path)

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.resumer(errors=n_erreurs, skipped=n_eligibles - n_traites - n_erreurs)


if __name__ == "__main__":
    main()
