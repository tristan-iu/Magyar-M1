#!/usr/bin/env python3
"""
Transcription audio batch via Whisper + Silero-VAD.
Enrichit le JSONL avec : dialogue, has_speech, speech_language,
speech_confidence, speech_duration, speech_ratio, whisper_segments, srt_path.

Pipeline par message :
  1. VAD (Silero) — détecte la présence/durée de parole
  2. Whisper — transcription (si parole détectée ou --vad-gate désactivé)
  3. Filtrage qualité — rejette les hallucinations (confidence, ratio cyrillique)
  4. SRT — génère un fichier sous-titres à côté du média

Options CLI : --model, --language, --vad-threshold, --vad-gate, --limit, --overwrite

Dépendances : openai-whisper, torch, torchaudio
"""
import json
import math
import os
import re
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    build_base_parser,
    load_config,
    setup_logging,
    read_jsonl,
    write_jsonl,
    update_fiche,
    filter_eligible,
    ProgressTracker,
    CYRILLIC_RE,
)

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


def load_whisper(model_name: str = "large-v3"):
    """Charge le modèle Whisper sur GPU si disponible, sinon CPU.

    Entrée : model_name — str identifiant du modèle (ex: "large-v3", "medium")
    Sortie : modèle Whisper chargé (singleton — chargé une seule fois par session)
    """
    global _whisper_model

    if _whisper_model is None:
        import whisper

        print(f"  Chargement Whisper '{model_name}' sur CPU...")
        _whisper_model = whisper.load_model(model_name, device="cpu")

        if torch.cuda.is_available():
            _whisper_model = _whisper_model.to(device="cuda", dtype=torch.float16)

            # Fix dtype: LayerNorm en fp32, reste en fp16
            for m in _whisper_model.modules():
                if isinstance(m, torch.nn.LayerNorm):
                    m.float()

        p = next(_whisper_model.parameters())
        print("WHISPER", p.device, p.dtype)

    return _whisper_model


def _read_audio_ffmpeg(audio_path: str, sampling_rate: int = 16000) -> "torch.Tensor":
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


def load_vad():
    """Charge Silero-VAD une seule fois."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        try:
            print("  Chargement Silero-VAD...")
            _vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            _vad_utils = {
                "get_speech_timestamps": utils[0],
                "read_audio": _read_audio_ffmpeg,  # bypass torchaudio/torchcodec
            }
            print("  Silero-VAD chargé")
        except Exception as e:
            print(f"ATTENTION : Erreur chargement VAD : {e}")
            return None, None
    return _vad_model, _vad_utils


# ── VAD ──────────────────────────────────────────────────────────────────────

def detect_voice_activity(audio_path: str, log,
                          threshold: float = 0.35,
                          min_speech_ms: int = 200,
                          min_silence_ms: int = 200) -> dict:
    """Détecte la parole via Silero-VAD.

    Entrée : audio_path — chemin vers le fichier, log — logger,
             threshold — seuil de détection VAD (0–1),
             min_speech_ms / min_silence_ms — durées minimales en ms
    Sortie : dict {"has_speech": bool, "speech_duration": float (secondes)}
    """
    result = {"has_speech": False, "speech_duration": 0.0}

    model, utils = load_vad()
    if model is None or utils is None:
        # VAD indisponible → on suppose parole présente pour ne pas bloquer
        result["has_speech"] = True
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
            result["speech_duration"] = round(total_speech, 2)
            result["has_speech"] = True

    except Exception as e:
        log.warning(f"Erreur VAD : {e}")
        # Fallback optimiste
        result["has_speech"] = True

    return result


# ── Transcription Whisper ────────────────────────────────────────────────────

def transcribe(audio_path: str, model_name: str, log,
               language: str = None, prompt: str = "") -> dict:
    """Transcrit l'audio via Whisper.

    Entrée : audio_path — chemin vers le fichier, model_name — str,
             log — logger, language — code langue forcé (ex: "uk") ou None,
             prompt — texte de conditionnement initial
    Sortie : dict {"text": str, "language": str, "segments": list, "confidence": float|None}
    """
    model = load_whisper(model_name)
    result = {"text": "", "language": None, "segments": [], "confidence": None}

    try:
        transcription = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False,
            initial_prompt=prompt,
            no_speech_threshold=0.2,
            logprob_threshold=-0.5,
            compression_ratio_threshold=2.0,
        )
        result["text"] = transcription.get("text", "").strip()
        result["language"] = transcription.get("language")
        result["segments"] = transcription.get("segments", [])

        # Confidence = exp(moyenne des avg_logprob)
        logprobs = [s["avg_logprob"] for s in result["segments"] if "avg_logprob" in s]
        if logprobs:
            result["confidence"] = round(math.exp(sum(logprobs) / len(logprobs)), 4)

    except Exception as e:
        log.warning(f"Erreur Whisper sur {audio_path} : {e}")

    return result


# ── Nettoyage segments ───────────────────────────────────────────────────────

def clean_segments(segments: list) -> list:
    """Nettoie les segments Whisper pour inclusion directe dans le JSONL."""
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

def generate_srt(segments: list, output_path: str) -> bool:
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

def filter_segments(segments: list,
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


def is_valid_transcription(confidence: float | None,
                           min_confidence: float = 0.35) -> bool:
    """Rejette si la confidence globale est trop basse."""
    if confidence is not None and confidence < min_confidence:
        return False
    return True



# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = build_base_parser(
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
    log = setup_logging("whisper", cfg=cfg)
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
    ids_filter = set(args.ids) if args.ids else None
    eligible = filter_eligible(
        messages,
        ids_filter=ids_filter,
        media_types=["video"],
        check_fields=["dialogue"],
        overwrite=args.overwrite,
        limit=None,  # appliqué après le filtre has_audio
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Filtre supplémentaire : has_audio
    eligible = [i for i in eligible if messages[i].get("has_audio")]

    if args.limit:
        eligible = eligible[:args.limit]

    n_eligible = len(eligible)
    log.info(f"Vidéos avec audio à transcrire : {n_eligible}")

    if n_eligible == 0:
        log.info("Rien à faire.")
        if input_path != output_path:
            write_jsonl(messages, output_path)
        return

    # ── Chargement modèles (une seule fois) ──
    load_vad()
    load_whisper(model_name)

    tracker = ProgressTracker(n_eligible, label="whisper")
    processed = 0
    errors = 0

    try:
        for rank, idx in enumerate(eligible):
            msg = messages[idx]
            mid = msg.get("message_id", "?")
            channel = msg.get("channel", "robert_magyar")
            media_path_rel = msg.get("media_path")
            duration = msg.get("duration")  # ffprobe

            if not media_path_rel:
                continue

            media_file = media_dir / media_path_rel
            basename = os.path.basename(media_path_rel)

            # ── Fichier manquant ──
            if not media_file.is_file():
                log.warning(f"msg {mid}\t{media_path_rel}\tfichier manquant")
                errors += 1
                tracker.tick(rank, mid, "fichier manquant ✗")
                continue

            # ── VAD ──
            print(f"[{rank+1}/{n_eligible}] msg {mid} {basename} — VAD...", end="", flush=True)
            vad = detect_voice_activity(str(media_file), log,
                                        threshold=vad_threshold,
                                        min_speech_ms=min_speech_ms,
                                        min_silence_ms=min_silence_ms)

            if not vad["has_speech"] and args.vad_gate:
                new_fields = {
                    "dialogue": "", "has_speech": False, "speech_language": None,
                    "speech_confidence": None, "speech_duration": 0.0,
                    "speech_ratio": 0.0, "whisper_segments": [],
                    "srt_path": None,
                }
                msg.update(new_fields)
                processed += 1
                update_fiche(msg, new_fields, media_dir / "fiches", overwrite=args.overwrite)
                tracker.tick(rank, mid, "pas de parole")
                if processed % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Whisper ──
            print(f"\r[{rank+1}/{n_eligible}] msg {mid} {basename} — Whisper...", end="", flush=True)
            whisper_result = transcribe(str(media_file), model_name, log,
                                        language=language, prompt=whisper_prompt)

            speech_dur = vad["speech_duration"]
            speech_ratio = round(speech_dur / duration, 3) if duration and duration > 0 else 0.0

            if not whisper_result["text"]:
                new_fields = {
                    "dialogue": "", "has_speech": False,
                    "speech_language": whisper_result["language"],
                    "speech_confidence": None, "speech_duration": speech_dur,
                    "speech_ratio": speech_ratio, "whisper_segments": [],
                    "srt_path": None,
                }
                msg.update(new_fields)
                processed += 1
                update_fiche(msg, new_fields, media_dir / "fiches", overwrite=args.overwrite)
                tracker.tick(rank, mid, "transcription vide")
                if processed % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Filtre qualité post-transcription ──
            cleaned = clean_segments(whisper_result["segments"])

            # 1. Rejet global si confidence trop basse
            if not is_valid_transcription(whisper_result["confidence"]):
                cleaned = []
            else:
                # 2. Filtrage par segment (artefacts latins, ratio non-cyrillique)
                cleaned = filter_segments(cleaned)

            if not cleaned:
                new_fields = {
                    "dialogue": "", "has_speech": False,
                    "speech_language": whisper_result["language"],
                    "speech_confidence": whisper_result["confidence"],
                    "speech_duration": speech_dur,
                    "speech_ratio": speech_ratio, "whisper_segments": [],
                    "srt_path": None,
                }
                msg.update(new_fields)
                processed += 1
                update_fiche(msg, new_fields, media_dir / "fiches", overwrite=args.overwrite)
                conf = f"{whisper_result['confidence']:.2f}" if whisper_result["confidence"] else "?"
                tracker.tick(rank, mid, f"rejeté (conf={conf})")
                if processed % save_every == 0:
                    write_jsonl(messages, output_path)
                continue

            # ── Résultats ──
            dialogue = " ".join(s["text"] for s in cleaned if s.get("text"))

            # SRT (dans le même dossier que le média)
            srt_filename = f"{channel}_{mid}.srt"
            media_folder = media_file.parent
            generate_srt(cleaned, str(media_folder / srt_filename))
            media_rel_dir = str(Path(media_path_rel).parent)

            new_fields = {
                "dialogue": dialogue, "has_speech": True,
                "speech_language": whisper_result["language"],
                "speech_confidence": whisper_result["confidence"],
                "speech_duration": speech_dur, "speech_ratio": speech_ratio,
                "whisper_segments": cleaned,
                "srt_path": f"{media_rel_dir}/{srt_filename}",
            }
            msg.update(new_fields)
            processed += 1
            update_fiche(msg, new_fields, media_dir / "fiches", overwrite=args.overwrite)

            lang = whisper_result["language"] or "?"
            conf = f"{whisper_result['confidence']:.2f}" if whisper_result["confidence"] else "?"
            n_seg = len(cleaned)
            n_raw = len(whisper_result["segments"])
            dropped = f", -{n_raw - n_seg} filtrés" if n_seg < n_raw else ""
            tracker.tick(rank, mid, f"{lang}, {n_seg} seg{dropped}, conf={conf}")

            if processed % save_every == 0:
                write_jsonl(messages, output_path)

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        write_jsonl(messages, output_path)

    tracker.summary(errors=errors, skipped=n_eligible - processed - errors)


if __name__ == "__main__":
    main()
