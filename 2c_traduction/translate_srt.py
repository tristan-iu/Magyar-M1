#!/usr/bin/env python3
"""
Traduction uk→fr du corpus Magyar (legende, dialogue, whisper_segments).

Écrit `legende_fr` et `dialogue_fr` dans le JSONL ET dans la fiche.
Le SRT français vit dans les fichiers robert_magyar_{id}_fr.srt à côté du média.

Pipeline :
  1. Sélectionner les messages avec du contenu cyrillique à traduire
  2. Détecter le moteur disponible (deepl → lmstudio)
  3. Traduire legende / dialogue / segments Whisper
  4. Écrire dans le JSONL + fiche JSON + fichier SRT

Options CLI : --input, --output, --engine, --fiches-dir, --dry-run, --limit,
              --ids, --overwrite, --start-date, --end-date, --config
"""
import logging
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
    SuiviProgression,
    creer_parser_base,
    filtrer_eligibles,
    load_config,
    charger_fiche,
    read_jsonl,
    init_logger,
    mettre_a_jour_fiche,
    write_jsonl,
)

# Logger module : configuré par init_logger("translation") dans main().
# Les classes/fonctions ci-dessous l'utilisent via cette variable globale.
log = logging.getLogger("translation")


def _fmt_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def formater_srt(segments: list[dict]) -> str:
    """Reconstruit un fichier SRT à partir de segments {start, end, text}."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(
            f"{i}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n{seg['text']}\n"
        )
    return "\n".join(blocks) + "\n"


def _analyser_srt_uk(srt_path: Path) -> list[dict]:
    """Parse un fichier SRT ukrainien en liste de segments {start, end, text}.

    Remplace la lecture de `whisper_segments` depuis la fiche (champ supprimé
    du schéma propre post-migration). Lit directement le fichier .srt sur disque.

    Convention fichier : {canal}_{message_id}.srt (dans le même dossier que le média)

    Entrée : srt_path — chemin vers le fichier SRT ukrainien
    Sortie : liste de dicts {start: float, end: float, text: str}
    """
    if not srt_path.is_file():
        return []

    segments: list[dict] = []
    try:
        contenu = srt_path.read_text(encoding="utf-8")
        # Format SRT : blocs séparés par une ligne vide
        # Chaque bloc : numéro \n timestamp --> timestamp \n texte
        blocs = [b.strip() for b in contenu.split("\n\n") if b.strip()]
        for bloc in blocs:
            lignes = [l.strip() for l in bloc.splitlines() if l.strip()]
            if len(lignes) < 3:
                continue  # bloc incomplet
            # Ligne 1 : numéro (on l'ignore)
            # Ligne 2 : "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            # Ligne 3+ : texte (peut s'étaler sur plusieurs lignes)
            timestamp_ligne = lignes[1]
            if " --> " not in timestamp_ligne:
                continue
            debut_str, fin_str = timestamp_ligne.split(" --> ", 1)
            texte = " ".join(lignes[2:])

            def _ts_to_sec(ts: str) -> float:
                ts = ts.replace(",", ".")
                parties = ts.strip().split(":")
                if len(parties) == 3:
                    h, m, s = parties
                    return int(h) * 3600 + int(m) * 60 + float(s)
                return 0.0

            segments.append({
                "start": _ts_to_sec(debut_str),
                "end": _ts_to_sec(fin_str),
                "text": texte,
            })
    except Exception:
        pass  # SRT corrompu : on retourne liste vide, le caller gère

    return segments


# ── Détection langue ──────────────────────────────────────────────────────────

def _a_besoin_traduction(text: str) -> bool:
    """True si le texte contient au moins 2 caractères cyrilliques.

    Entrée : text — str quelconque
    Sortie : bool
    """
    if not text or not text.strip():
        return False
    return len(CYRILLIC_RE.findall(text)) >= 2


# ── Translateurs ──────────────────────────────────────────────────────────────

class TraducteurDeepL:
    name = "deepl"

    def __init__(self):
        import deepl
        auth_key = os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
        if not auth_key:
            raise EnvironmentError("DEEPL_AUTH_KEY absent du .env")
        self.client = deepl.Translator(auth_key)

    def traduire(self, text: str) -> str:
        if not _a_besoin_traduction(text):
            return text
        result = self.client.translate_text(text, source_lang="UK", target_lang="FR")
        return result.text

    def traduire_lot(self, texts: list[str]) -> list[str]:
        """Traduit une liste de textes en un seul appel API DeepL (préserve l'ordre).

        Entrée : texts — liste de str (certains peuvent ne pas nécessiter de traduction)
        Sortie : liste de str traduits (même ordre, même longueur)
        """
        if not texts:
            return []
        indices = [i for i, t in enumerate(texts) if _a_besoin_traduction(t)]
        if not indices:
            return texts[:]
        results = self.client.translate_text(
            [texts[i] for i in indices], source_lang="UK", target_lang="FR"
        )
        out = texts[:]
        for pos, result in zip(indices, results):
            out[pos] = result.text
        return out


class TraducteurLMStudio:
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
        log.info("  LM Studio — modèle : %s  url : %s", self._model, base_url)

    def _call(self, user_text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.1,
            # ⚠️ Tronque silencieusement les dialogues très longs : les plus
            # gros du corpus font ~55k caractères (#889, #1058) → bien au-delà
            # de 4096 tokens de sortie. DeepL n'a pas cette limite (découpe en
            # interne). Le fallback LMStudio reste destiné à la consultation ;
            # pour traduire le corpus complet, préférer DeepL.
            max_tokens=4096,
        )
        return resp.choices[0].message.content.strip()

    def traduire(self, text: str) -> str:
        if not _a_besoin_traduction(text):
            return text
        return self._call(text)

    def traduire_lot(self, texts: list[str], chunk_size: int = 10) -> list[str]:
        """Traduit par chunks numérotés ; fallback segment par segment si la réponse est malformée.

        On envoie les segments par paquets de chunk_size pour limiter les appels LLM,
        mais un chunk mal parsé (mauvais compte de lignes) repasse en individuel.

        Entrée : texts — liste de str, chunk_size — taille des paquets (défaut 10)
        Sortie : liste de str traduits (même ordre, même longueur)
        """
        if not texts:
            return []
        indices = [i for i, t in enumerate(texts) if _a_besoin_traduction(t)]
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
            lot_traduit = None
            try:
                raw = self._call(prompt)
                lot_traduit = _analyser_numerote(raw, expected_count=len(chunk_idx))
            except (ValueError, KeyError, AttributeError, IndexError) as e:
                # On signale le passage en mode individuel (parsing ou API LMStudio défaillant)
                # pour ne pas masquer un bug silencieux ; le fallback ci-dessous prend le relais.
                log.warning("[traduire_lot] chunk parse failed, fallback individual: %s", e)

            if lot_traduit is not None:
                for j, i in enumerate(chunk_idx):
                    out[i] = lot_traduit[j]
            else:
                # Fallback individuel pour ce chunk
                for i in chunk_idx:
                    try:
                        out[i] = self._call(texts[i])
                    except Exception:
                        pass  # conserve l'original
        return out


def _analyser_numerote(raw: str, expected_count: int) -> list[str] | None:
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

def selectionner_moteur(force: str | None):
    """Retourne un translateur disponible, ou quitte avec une erreur.

    Auto-détection : essaie deepl d'abord, puis lmstudio. Si --engine est fourni,
    tente uniquement ce moteur et échoue explicitement s'il est indisponible.

    Entrée : force — nom du moteur imposé ou None
    Sortie : instance TraducteurDeepL ou TraducteurLMStudio
    """
    candidates = [
        ("deepl",    lambda: TraducteurDeepL()),
        ("lmstudio", lambda: TraducteurLMStudio()),
    ]
    if force:
        for name, factory in candidates:
            if name == force:
                try:
                    t = factory()
                    log.info("Moteur : %s", name)
                    return t
                except Exception as e:
                    log.error("ERREUR : --engine %s demandé mais indisponible : %s", force, e)
                    sys.exit(1)
        log.error("ERREUR : moteur inconnu '%s'", force)
        sys.exit(1)

    for name, factory in candidates:
        try:
            t = factory()
            log.info("Moteur auto-détecté : %s", name)
            return t
        except Exception:
            continue

    log.error(
        "ERREUR : aucun moteur disponible.\n"
        "  → Ajouter DEEPL_AUTH_KEY dans 2c_traduction/.env\n"
        "  → Ou démarrer un serveur local OpenAI-compatible (LM Studio :1234 ou "
        "Ollama :11434) ; définir LMSTUDIO_BASE_URL et installer : pip install openai"
    )
    sys.exit(1)


# ── Traitement d'un message ───────────────────────────────────────────────────

def traiter_message(
    msg: dict,
    traducteur,
    fiches_dir: Path,
    overwrite: bool,
    dry_run: bool,
    log,
) -> tuple[bool, bool, bool]:
    """Traduit legende / dialogue / whisper_segments d'un message.

    Chaque opération est indépendante : un échec SRT ne bloque pas legende/dialogue.
    Écrit dans le JSONL (msg) + fiche individuelle + fichier SRT.

    Entrée : msg — dict message JSONL, traducteur — instance DeepL ou LMStudio,
             fiches_dir — Path, overwrite — bool, dry_run — bool, log — logger
    Sortie : tuple (did_legende bool, did_dialogue bool, did_srt bool)
    """
    mid = msg["message_id"]
    channel = msg.get("canal", "robert_magyar")
    fiche = charger_fiche(msg, fiches_dir)

    legende      = msg.get("legende") or fiche.get("legende")
    dialogue     = msg.get("dialogue") or fiche.get("dialogue")

    # On lit le SRT UK directement sur disque (whisper_segments absent du schéma propre).
    # Convention : {canal}_{message_id}.srt dans le même dossier que fiches_dir.
    srt_uk_path = fiches_dir / f"{channel}_{mid}.srt"
    whisper_segs = _analyser_srt_uk(srt_uk_path)

    if not legende and not dialogue and not whisper_segs:
        return False, False, False

    srt_fr_name = f"{channel}_{mid}_fr.srt"
    srt_fr_path = fiches_dir / srt_fr_name

    do_legende  = bool(legende)      and (overwrite or "legende_fr"  not in msg)
    do_dialogue = bool(dialogue)     and (overwrite or "dialogue_fr" not in msg)
    do_srt      = bool(whisper_segs) and (overwrite or not srt_fr_path.exists())

    if not do_legende and not do_dialogue and not do_srt:
        return False, False, False

    did_legende = did_dialogue = did_srt = False

    # ── Légende ────────────────────────────────────────────────────────────
    if do_legende:
        if dry_run:
            print(f"  [dry-run] legende  msg {mid}: {legende[:70]!r}")
            did_legende = True
        else:
            try:
                legende_fr = traducteur.traduire(legende)
                msg["legende_fr"] = legende_fr
                mettre_a_jour_fiche(msg, {"legende_fr": legende_fr}, fiches_dir, overwrite=overwrite)
                did_legende = True
                log.info(f"msg {mid} | legende | OK")
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    raise
                log.warning(f"msg {mid} | legende | ERROR:{e}")

    # ── Dialogue ───────────────────────────────────────────────────────────
    if do_dialogue:
        if dry_run:
            print(f"  [dry-run] dialogue msg {mid}: {dialogue[:70]!r}")
            did_dialogue = True
        else:
            try:
                dialogue_fr = traducteur.traduire(dialogue)
                msg["dialogue_fr"] = dialogue_fr
                mettre_a_jour_fiche(msg, {"dialogue_fr": dialogue_fr}, fiches_dir, overwrite=overwrite)
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
                textes_traduits = traducteur.traduire_lot(texts)
                segments_traduits = [
                    {**seg, "text": t}
                    for seg, t in zip(whisper_segs, textes_traduits)
                ]
                srt_fr_path.write_text(formater_srt(segments_traduits), encoding="utf-8")
                did_srt = True
                log.info(f"msg {mid} | srt | OK:{len(segments_traduits)} segments")
            except Exception as e:
                if "QuotaExceeded" in type(e).__name__:
                    raise
                log.warning(f"msg {mid} | srt | ERROR:{e}")

    return did_legende, did_dialogue, did_srt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(Path(__file__).parent / ".env")

    parser = creer_parser_base(
        "Traduction uk→fr : legende / dialogue / whisper_segments → JSONL + fiche + SRT",
        has_input=True,
        has_output=True,
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
    log = init_logger("translation", cfg=cfg)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    fiches_dir = (args.fiches_dir or Path(cfg["paths"]["fiches_dir"])).resolve()

    if not input_path.is_file():
        log.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)
    if not fiches_dir.is_dir():
        log.error(f"Dossier fiches introuvable : {fiches_dir}")
        sys.exit(1)

    traducteur = selectionner_moteur(args.engine)

    messages = read_jsonl(input_path)
    log.info(f"Corpus : {len(messages)} messages")

    filtre_ids = set(args.ids) if args.ids else None
    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    n_eligibles = len(eligibles)
    log.info(f"Messages à traiter : {n_eligibles}")

    if n_eligibles == 0:
        log.info("Rien à faire.")
        return

    tracker = SuiviProgression(n_eligibles, label="traduction")
    n_traites = 0
    n_erreurs = 0
    save_every = 50

    try:
        for rank, idx in enumerate(eligibles):
            msg = messages[idx]
            mid = msg.get("message_id", "?")

            try:
                did_legende, did_dialogue, did_srt = traiter_message(
                    msg=msg,
                    traducteur=traducteur,
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
                n_erreurs += 1
                tracker.avancer(rank, mid, "erreur ✗")
                continue

            parts = (
                (["legende"] if did_legende else [])
                + (["dialogue"] if did_dialogue else [])
                + (["srt"] if did_srt else [])
            )
            if parts:
                n_traites += 1
                tracker.avancer(rank, mid, ", ".join(parts) + " ✓")
                if not args.dry_run and n_traites % save_every == 0:
                    write_jsonl(messages, output_path)
            else:
                tracker.avancer(rank, mid, "skip")

    except KeyboardInterrupt:
        log.info("\nInterruption clavier — sauvegarde en cours...")
    finally:
        if not args.dry_run:
            write_jsonl(messages, output_path)

    tracker.resumer(errors=n_erreurs, skipped=n_eligibles - n_traites - n_erreurs)


if __name__ == "__main__":
    main()
