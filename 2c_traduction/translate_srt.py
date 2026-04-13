#!/usr/bin/env python3
"""
Traduction uk→fr du corpus Magyar (caption, dialogue, whisper_segments).

Pas de JSONL de sortie : les traductions vivent dans les fiches individuelles
(caption_fr, dialogue_fr) et dans les fichiers robert_magyar_{id}_fr.srt.

Pipeline :
  1. Sélectionner les messages avec du contenu cyrillique à traduire
  2. Détecter le moteur disponible (deepl → lmstudio)
  3. Traduire caption / dialogue / segments Whisper
  4. Écrire dans la fiche JSON + fichier SRT

Options CLI : --input, --engine, --fiches-dir, --dry-run, --limit, --ids,
              --overwrite, --start-date, --end-date, --config
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Import utils / config ────────────────────────────────────────────────────

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    CYRILLIC_RE,
    ProgressTracker,
    build_base_parser,
    filter_eligible,
    load_config,
    load_fiche,
    read_jsonl,
    setup_logging,
    update_fiche,
)


def _fmt_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def format_srt(segments: list[dict]) -> str:
    """Reconstruit un fichier SRT à partir de segments {start, end, text}."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(blocks) + "\n"


# ── Détection langue ──────────────────────────────────────────────────────────

def _needs_translation(text: str) -> bool:
    """True si le texte contient au moins 2 caractères cyrilliques.

    Entrée : text — str quelconque
    Sortie : bool
    """
    if not text or not text.strip():
        return False
    return len(CYRILLIC_RE.findall(text)) >= 2


# ── Translateurs ──────────────────────────────────────────────────────────────

class DeepLTranslator:
    name = "deepl"

    def __init__(self):
        import deepl
        auth_key = os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
        if not auth_key:
            raise EnvironmentError("DEEPL_AUTH_KEY absent du .env")
        self.client = deepl.Translator(auth_key)

    def translate(self, text: str) -> str:
        if not _needs_translation(text):
            return text
        result = self.client.translate_text(text, source_lang="UK", target_lang="FR")
        return result.text

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Traduit une liste de textes en un seul appel API DeepL (préserve l'ordre).

        Entrée : texts — liste de str (certains peuvent ne pas nécessiter de traduction)
        Sortie : liste de str traduits (même ordre, même longueur)
        """
        if not texts:
            return []
        indices = [i for i, t in enumerate(texts) if _needs_translation(t)]
        if not indices:
            return texts[:]
        results = self.client.translate_text(
            [texts[i] for i in indices], source_lang="UK", target_lang="FR"
        )
        out = texts[:]
        for pos, result in zip(indices, results):
            out[pos] = result.text
        return out


class LMStudioTranslator:
    name = "lmstudio"

    _SYSTEM_PROMPT = (
        "Tu es un traducteur expert en ukrainien militaire (conflit russo-ukrainien 2022-2025). "
        "Traduis le texte ukrainien en français. "
        "Retourne uniquement la traduction, sans explication ni commentaire."
    )

    def __init__(self):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai requis pour LM Studio")

        base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        api_key  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
        self._system_prompt = self._SYSTEM_PROMPT
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

        model_override = os.environ.get("LMSTUDIO_MODEL")
        if model_override:
            self._model = model_override
        else:
            models = self._client.models.list()
            if not models.data:
                raise ConnectionError("Aucun modèle chargé dans LM Studio")
            self._model = models.data[0].id
        print(f"  LM Studio — modèle : {self._model}  url : {base_url}")

    def _call(self, user_text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            max_tokens=4096,  # assez large pour les dialogues longs
        )
        return resp.choices[0].message.content.strip()

    def translate(self, text: str) -> str:
        if not _needs_translation(text):
            return text
        return self._call(text)

    def translate_batch(self, texts: list[str], chunk_size: int = 10) -> list[str]:
        """Traduit par chunks numérotés ; fallback segment par segment si la réponse est malformée.

        On envoie les segments par paquets de chunk_size pour limiter les appels LLM,
        mais un chunk mal parsé (mauvais compte de lignes) repasse en individuel.

        Entrée : texts — liste de str, chunk_size — taille des paquets (défaut 10)
        Sortie : liste de str traduits (même ordre, même longueur)
        """
        if not texts:
            return []
        indices = [i for i, t in enumerate(texts) if _needs_translation(t)]
        if not indices:
            return texts[:]

        out = texts[:]
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_idx = indices[chunk_start:chunk_start + chunk_size]
            numbered = "\n".join(f"{j + 1}. {texts[i]}" for j, i in enumerate(chunk_idx))
            prompt = (
                "Traduis chaque ligne numérotée de l'ukrainien vers le français. "
                "Conserve exactement les numéros. Retourne uniquement les lignes traduites.\n\n"
                + numbered
            )
            translated_chunk = None
            try:
                raw = self._call(prompt)
                translated_chunk = _parse_numbered(raw, expected_count=len(chunk_idx))
            except Exception:
                pass

            if translated_chunk is not None:
                for j, i in enumerate(chunk_idx):
                    out[i] = translated_chunk[j]
            else:
                # Fallback individuel pour ce chunk
                for i in chunk_idx:
                    try:
                        out[i] = self._call(texts[i])
                    except Exception:
                        pass  # conserve l'original
        return out


def _parse_numbered(raw: str, expected_count: int) -> list[str] | None:
    """Parse une réponse LLM numérotée ("1. texte\n2. texte\n...").

    Entrée : raw — réponse brute du LLM, expected_count — nb de lignes attendues
    Sortie : liste de str ou None si le compte ne correspond pas
    """
    lines = []
    for line in raw.strip().splitlines():
        m = re.match(r"^\d+\.\s*(.*)", line.strip())
        if m:
            lines.append(m.group(1).strip())
    return lines if len(lines) == expected_count else None


# ── Sélection du moteur ───────────────────────────────────────────────────────

def detect_engine(force: str | None):
    """Retourne un translateur disponible, ou quitte avec une erreur.

    Auto-détection : essaie deepl d'abord, puis lmstudio. Si --engine est fourni,
    tente uniquement ce moteur et échoue explicitement s'il est indisponible.

    Entrée : force — nom du moteur imposé ou None
    Sortie : instance DeepLTranslator ou LMStudioTranslator
    """
    candidates = [
        ("deepl",    lambda: DeepLTranslator()),
        ("lmstudio", lambda: LMStudioTranslator()),
    ]
    if force:
        for name, factory in candidates:
            if name == force:
                try:
                    t = factory()
                    print(f"Moteur : {name}")
                    return t
                except Exception as e:
                    print(f"ERREUR : --engine {force} demandé mais indisponible : {e}")
                    sys.exit(1)
        print(f"ERREUR : moteur inconnu '{force}'")
        sys.exit(1)

    for name, factory in candidates:
        try:
            t = factory()
            print(f"Moteur auto-détecté : {name}")
            return t
        except Exception:
            continue

    print(
        "ERREUR : aucun moteur disponible.\n"
        "  → Ajouter DEEPL_AUTH_KEY dans 2d_traduction/.env\n"
        "  → Ou démarrer LM Studio (http://127.0.0.1:1234) et installer : pip install openai"
    )
    sys.exit(1)


# ── Traitement d'un message ───────────────────────────────────────────────────

def process_message(
    msg: dict,
    translator,
    fiches_dir: Path,
    overwrite: bool,
    dry_run: bool,
    log,
) -> tuple[bool, bool, bool]:
    """Traduit caption / dialogue / whisper_segments d'un message.

    Chaque opération est indépendante : un échec SRT ne bloque pas caption/dialogue.
    Écrit directement dans la fiche individuelle + fichier SRT.

    Entrée : msg — dict message JSONL, translator — instance DeepL ou LMStudio,
             fiches_dir — Path, overwrite — bool, dry_run — bool, log — logger
    Sortie : tuple (did_caption bool, did_dialogue bool, did_srt bool)
    """
    mid = msg["message_id"]
    fiche = load_fiche(msg, fiches_dir)

    caption      = fiche.get("caption")      or msg.get("caption")
    dialogue     = fiche.get("dialogue")     or msg.get("dialogue")
    whisper_segs = fiche.get("whisper_segments") or msg.get("whisper_segments") or []

    if not caption and not dialogue and not whisper_segs:
        return False, False, False

    srt_fr_name = f"robert_magyar_{mid}_fr.srt"
    srt_fr_path = fiches_dir / srt_fr_name

    do_caption  = bool(caption)      and (overwrite or "caption_fr"  not in fiche)
    do_dialogue = bool(dialogue)     and (overwrite or "dialogue_fr" not in fiche)
    do_srt      = bool(whisper_segs) and (overwrite or not srt_fr_path.exists())

    if not do_caption and not do_dialogue and not do_srt:
        return False, False, False

    # Chaque opération est indépendante : un échec SRT ne détruit pas caption/dialogue.

    did_caption = did_dialogue = did_srt = False

    # ── Caption ────────────────────────────────────────────────────────────
    if do_caption:
        if dry_run:
            print(f"  [dry-run] caption  msg {mid}: {caption[:70]!r}")
            did_caption = True
        else:
            try:
                caption_fr = translator.translate(caption)
                update_fiche(msg, {"caption_fr": caption_fr}, fiches_dir, overwrite=overwrite)
                did_caption = True
                log.info(f"msg {mid} | caption | OK")
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    raise
                log.warning(f"msg {mid} | caption | ERROR:{e}")

    # ── Dialogue ───────────────────────────────────────────────────────────
    if do_dialogue:
        if dry_run:
            print(f"  [dry-run] dialogue msg {mid}: {dialogue[:70]!r}")
            did_dialogue = True
        else:
            try:
                dialogue_fr = translator.translate(dialogue)
                update_fiche(msg, {"dialogue_fr": dialogue_fr}, fiches_dir, overwrite=overwrite)
                did_dialogue = True
                log.info(f"msg {mid} | dialogue | OK")
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    raise
                log.warning(f"msg {mid} | dialogue | ERROR:{e}")

    # ── Whisper segments → SRT FR ──────────────────────────────────────────
    if do_srt:
        texts = [seg.get("text", "") for seg in whisper_segs]
        if dry_run:
            print(f"  [dry-run] {len(texts)} segments → {srt_fr_name}  msg {mid}")
            did_srt = True
        else:
            try:
                translated_texts = translator.translate_batch(texts)
                translated_segs = [
                    {**seg, "text": t}
                    for seg, t in zip(whisper_segs, translated_texts)
                ]
                srt_fr_path.write_text(format_srt(translated_segs), encoding="utf-8")
                update_fiche(msg, {"srt_fr_path": f"fiches/{srt_fr_name}"}, fiches_dir, overwrite=overwrite)
                did_srt = True
                log.info(f"msg {mid} | srt | OK:{len(translated_segs)} segments")
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    raise
                log.warning(f"msg {mid} | srt | ERROR:{e}")

    return did_caption, did_dialogue, did_srt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(Path(__file__).parent / ".env")

    parser = build_base_parser(
        "Traduction uk→fr : caption / dialogue / whisper_segments → fiche + SRT",
        has_input=True,
        has_output=False,
    )
    parser.add_argument(
        "--engine",
        choices=["deepl", "lmstudio"],
        default=None,
        help="Forcer un moteur (défaut : auto-détection deepl → lmstudio)",
    )
    parser.add_argument(
        "--fiches-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dossier des fiches JSON (défaut : config.yaml paths.fiches_dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher ce qui serait traduit sans rien écrire",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("translation", cfg=cfg)

    input_path = Path(args.input).resolve()
    fiches_dir = (args.fiches_dir or Path(cfg["paths"]["fiches_dir"])).resolve()

    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)
    if not fiches_dir.is_dir():
        log.error(f"Dossier fiches introuvable : {fiches_dir}")
        sys.exit(1)

    translator = detect_engine(args.engine)

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    ids_filter = set(args.ids) if args.ids else None
    eligible = filter_eligible(
        messages,
        ids_filter=ids_filter,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    n_eligible = len(eligible)
    log.info(f"Messages à traiter : {n_eligible}")

    if n_eligible == 0:
        log.info("Rien à faire.")
        return

    tracker = ProgressTracker(n_eligible, label="traduction")
    processed = 0
    errors = 0

    try:
        for rank, idx in enumerate(eligible):
            msg = messages[idx]
            mid = msg.get("message_id", "?")

            try:
                did_caption, did_dialogue, did_srt = process_message(
                    msg=msg,
                    translator=translator,
                    fiches_dir=fiches_dir,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    log=log,
                )
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    log.error(f"Quota DeepL dépassé après {rank} messages.")
                    break
                log.warning(f"msg {mid} | FATAL:{e}")
                errors += 1
                tracker.tick(rank, mid, "erreur ✗")
                continue

            parts = (
                (["caption"] if did_caption else [])
                + (["dialogue"] if did_dialogue else [])
                + (["srt"] if did_srt else [])
            )
            if parts:
                processed += 1
                tracker.tick(rank, mid, ", ".join(parts) + " ✓")
            else:
                tracker.tick(rank, mid, "skip")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier.")

    tracker.summary(errors=errors, skipped=n_eligible - processed - errors)


if __name__ == "__main__":
    main()
