"""
utils.py — Utilitaires partagés du pipeline Magyar.

Sections :
  - CLI         : build_base_parser(), parse_date_arg()
  - PHASES      : phase_label()
  - LOGGING     : setup_logging()
  - IDEMPOTENCE : is_processed()
  - JSONL I/O   : read_jsonl(), write_jsonl()
  - FICHES      : fiche_path(), load_fiche(), update_fiche()
  - PROGRESSION : ProgressTracker, fmt_eta()
  - FILTRAGE    : filter_eligible()
  - REGEX       : CYRILLIC_RE (partagé whisper + traduction)

Import :
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[N] / "0_config"))
    from utils import setup_logging, write_jsonl, update_fiche, ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
import yaml

# ---------------------------------------------------------------------------
# CHARGEMENT CONFIG
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path: Path | None = None) -> dict:
    """Charge config.yaml. Retourne un dict YAML."""
    p = path or _CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI — ARGUMENT PARSER FACTORY
# ---------------------------------------------------------------------------
# Fournit un argparse.ArgumentParser pré-rempli avec les arguments communs
# à tous les scripts du pipeline (input, output, limit, overwrite, ids,
# start-date, end-date, config, media-dir). Chaque script peut ajouter
# ses propres arguments spécifiques par-dessus.
# ---------------------------------------------------------------------------

def build_base_parser(
    description: str,
    *,
    has_input: bool = True,
    has_output: bool = True,
    has_media_dir: bool = False,
) -> argparse.ArgumentParser:
    """
    Factory qui retourne un ArgumentParser avec les arguments CLI standards.

    Paramètres de contrôle :
      - has_input     : ajoute --input  (JSONL source)
      - has_output    : ajoute --output (JSONL destination)
      - has_media_dir : ajoute --media-dir (racine médias)
    """
    parser = argparse.ArgumentParser(description=description)

    if has_input:
        parser.add_argument("--input", required=True, help="JSONL source")
    if has_output:
        parser.add_argument("--output", required=True,
                            help="JSONL destination (peut être le même fichier)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Traiter au maximum N messages")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retraiter même si déjà fait (désactive l'idempotence)")
    parser.add_argument("--ids", type=int, nargs="+", metavar="ID",
                        help="Traiter uniquement ces message_id (ex: --ids 42 138)")
    parser.add_argument("--start-date", type=parse_date_arg, default=None,
                        metavar="YYYY-MM-DD",
                        help="Ne traiter que les messages >= cette date")
    parser.add_argument("--end-date", type=parse_date_arg, default=None,
                        metavar="YYYY-MM-DD",
                        help="Ne traiter que les messages <= cette date")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml (défaut : 0_config/config.yaml)")
    if has_media_dir:
        parser.add_argument(
            "--media-dir",
            help="Racine pour résoudre les media_path relatifs "
                 "(défaut : dossier parent de --input)",
        )

    return parser


def parse_date_arg(s: str) -> date:
    """Parse une chaîne YYYY-MM-DD en datetime.date. Utilisé comme type= dans argparse."""
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Format de date invalide : {s!r}. Attendu : YYYY-MM-DD"
        )


# ---------------------------------------------------------------------------
# PHASES DU CORPUS 
# ---------------------------------------------------------------------------
# Ces fonctions ont été crées après avoir parcouru l'ensemble du corpus 
# d'avoir établi le plan du mémoire, dans le but de faciliter le traitement
# des données. Aucune des analyses n'a été faite en présupposant que ce 
# découpage est objectif.
# ---------------------------------------------------------------------------

_PHASE_DATES: list[tuple[str, date, date]] | None = None

def _init_phases(cfg: dict | None = None) -> list[tuple[str, date, date]]:
    """Initialise la liste de phases depuis config.yaml."""
    global _PHASE_DATES
    if _PHASE_DATES is not None:
        return _PHASE_DATES
    c = cfg or load_config()
    phases = []
    for pid, pdata in c["phases"].items():
        start = date.fromisoformat(pdata["start"])
        end   = date.fromisoformat(pdata["end"])
        phases.append((pid, start, end))
    _PHASE_DATES = sorted(phases, key=lambda x: x[1])
    return _PHASE_DATES


def phase_label(dt: str | datetime | date, cfg: dict | None = None) -> str | None:
    """
    Retourne l'identifiant de phase (P1/P2/P3) pour une date donnée.
    Accepte : chaîne ISO, datetime, date.
    Retourne None si hors de toutes les phases définies.

    >>> phase_label("2023-06-15")
    'P1'
    >>> phase_label("2024-05-01")
    'P2'
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt[:19])
    if isinstance(dt, datetime):
        dt = dt.date()
    for pid, start, end in _init_phases(cfg):
        if start <= dt <= end:
            return pid
    return None


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def setup_logging(
    script_name: str,
    log_file: str | Path | None = None,
    cfg: dict | None = None,
) -> logging.Logger:
    """
    Configure et retourne un logger avec :
    - handler console (INFO) — messages courts
    - handler fichier  (WARNING) — erreurs horodatées dans logs/

    Si log_file est None, construit le chemin depuis config.yaml (paths.logs_dir).
    """
    c = cfg or load_config()

    if log_file is None:
        logs_dir = Path(c["paths"]["logs_dir"])
        log_file = logs_dir / f"{script_name}_errors.log"

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # Fichier
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.WARNING)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s\t%(levelname)s\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# IDEMPOTENCE
# ---------------------------------------------------------------------------

def is_processed(msg: dict, fields: list[str] | str) -> bool:
    """
    Retourne True si tous les champs listés sont déjà présents dans msg
    (et non-None). Utilisé pour skiper les messages déjà traités.

    >>> is_processed({"dialogue": "..."}, "dialogue")
    True
    >>> is_processed({"a": 1}, ["a", "b"])
    False
    """
    if isinstance(fields, str):
        fields = [fields]
    return all(msg.get(f) is not None for f in fields)


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Lit un fichier JSONL, retourne une liste de dicts."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(messages: list[dict], output_path: str | Path) -> None:
    """
    Réécrit le JSONL entier (sauvegarde incrémentale).
    Crée les dossiers parents si nécessaire.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# FICHES INDIVIDUELLES
# ---------------------------------------------------------------------------

def fiche_path(msg: dict, fiches_dir: str | Path) -> Path:
    """
    Retourne le chemin de la fiche JSON individuelle pour un message.
    Convention : {fiches_dir}/{channel}_{id}_fiche.json
    """
    channel = msg.get("channel", "robert_magyar")
    mid = msg["message_id"]
    return Path(fiches_dir) / f"{channel}_{mid}_fiche.json"


def load_fiche(msg: dict, fiches_dir: str | Path) -> dict:
    """Charge la fiche individuelle. Retourne {} si elle n'existe pas."""
    p = fiche_path(msg, fiches_dir)
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def update_fiche(
    msg: dict,
    new_fields: dict,
    fiches_dir: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Merge incrémental de new_fields dans la fiche individuelle.
    Si overwrite=False (défaut), les champs existants ne sont pas écrasés.
    """
    p = fiche_path(msg, fiches_dir)
    p.parent.mkdir(parents=True, exist_ok=True)

    existing = load_fiche(msg, fiches_dir)
    if overwrite:
        existing.update(new_fields)
    else:
        for k, v in new_fields.items():
            if k not in existing or existing[k] is None:
                existing[k] = v

    with open(p, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# PROGRESSION & ETA
# ---------------------------------------------------------------------------

def fmt_eta(seconds: float) -> str:
    """
    Formate un nombre de secondes en chaîne lisible.
    >>> fmt_eta(45)   → '45s'
    >>> fmt_eta(125)  → '2m05s'
    >>> fmt_eta(3700) → '1h01m'
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m:02d}m"


class ProgressTracker:
    """
    Tracker de progression pour les batchs.

    Usage :
        tracker = ProgressTracker(total=len(eligible), label="whisper")
        for i, msg in enumerate(eligible):
            tracker.tick(i, msg["message_id"])
            # ... traitement ...
        tracker.summary(errors, skipped)
    """

    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.t0 = time.time()

    def tick(self, rank: int, msg_id: int | str = "", extra: str = "") -> None:
        elapsed = time.time() - self.t0
        if rank > 0:
            eta = fmt_eta(elapsed / rank * (self.total - rank))
        else:
            eta = "~"
        prefix = f"[{self.label}] " if self.label else ""
        print(
            f"{prefix}[{rank+1}/{self.total}] msg {msg_id} "
            f"{extra} (ETA {eta})",
            flush=True,
        )

    def summary(self, errors: int = 0, skipped: int = 0) -> None:
        elapsed = time.time() - self.t0
        processed = self.total - skipped - errors
        print(
            f"\nTerminé en {fmt_eta(elapsed)} — "
            f"{processed} succès, {skipped} skippés, {errors} erreurs."
        )


# ---------------------------------------------------------------------------
# FILTRAGE BATCH (CLI helpers)
# ---------------------------------------------------------------------------
# Utiles pour le déboggage et l'itération sur les scripts, et dans leur 
# réutilisation sur d'autres corpus.
# ---------------------------------------------------------------------------

def filter_eligible(
    messages: list[dict],
    *,
    ids_filter: set[int] | None = None,
    media_types: list[str] | None = None,
    check_fields: list[str] | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[int]:
    """
    Retourne les indices des messages à traiter selon les filtres.

    - ids_filter   : ne traiter que ces message_id
    - media_types  : ne traiter que ces types (ex: ["video", "audio"])
    - check_fields : skipper si tous ces champs sont déjà présents (idempotence)
    - overwrite    : si True, désactive le check d'idempotence
    - limit        : max N messages
    - start_date   : ne garder que les messages dont le champ "date" >= start_date
    - end_date     : ne garder que les messages dont le champ "date" <= end_date
    """
    eligible = []
    for i, msg in enumerate(messages):
        # Filtre par ID
        if ids_filter and msg.get("message_id") not in ids_filter:
            continue
        # Filtre par type de média
        if media_types and msg.get("media_type") not in media_types:
            continue
        # Filtre par dates
        if start_date or end_date:
            raw = msg.get("date")
            if raw:
                try:
                    msg_date = datetime.fromisoformat(raw[:19]).date()
                except (ValueError, TypeError):
                    msg_date = None
            else:
                msg_date = None
            if msg_date is None:
                continue
            if start_date and msg_date < start_date:
                continue
            if end_date and msg_date > end_date:
                continue
        # Idempotence
        if not overwrite and check_fields and is_processed(msg, check_fields):
            continue
        eligible.append(i)

    if limit:
        eligible = eligible[:limit]
    return eligible


# ---------------------------------------------------------------------------
# REGEX PATTERNS
# ---------------------------------------------------------------------------

# Caractères cyrilliques — partagé par whisper_batch + translate_srt
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


