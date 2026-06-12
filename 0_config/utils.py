"""
utils.py — Utilitaires partagés du pipeline Magyar.

Sections :
  - CLI         : creer_parser_base(), analyser_date_arg()
  - PHASES      : etiquette_phase()
  - LOGGING     : init_logger()
  - IDEMPOTENCE : est_traite()
  - JSONL I/O   : read_jsonl(), write_jsonl()
  - FICHES      : chemin_fiche(), charger_fiche(), mettre_a_jour_fiche()
  - PROGRESSION : SuiviProgression, fmt_eta()
  - FILTRAGE    : filtrer_eligibles()
  - REGEX       : CYRILLIC_RE (partagé whisper + traduction)

Import :
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[N] / "0_config"))
    from utils import init_logger, write_jsonl, mettre_a_jour_fiche, ...
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
# CLI — FABRIQUE D'ARGUMENT PARSER
# ---------------------------------------------------------------------------
# Fournit un argparse.ArgumentParser pré-rempli avec les arguments communs
# à tous les scripts du pipeline (input, output, limit, overwrite, ids,
# start-date, end-date, config, media-dir). Chaque script peut ajouter
# ses propres arguments spécifiques par-dessus.
# ---------------------------------------------------------------------------

def creer_parser_base(
    description: str,
    *,
    has_input: bool = True,
    has_output: bool = True,
    has_media_dir: bool = False,
) -> argparse.ArgumentParser:
    """
    Fabrique qui retourne un ArgumentParser avec les arguments CLI standards.

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
    parser.add_argument("--start-date", type=analyser_date_arg, default=None,
                        metavar="YYYY-MM-DD",
                        help="Ne traiter que les messages >= cette date")
    parser.add_argument("--end-date", type=analyser_date_arg, default=None,
                        metavar="YYYY-MM-DD",
                        help="Ne traiter que les messages <= cette date")
    parser.add_argument("--config", default=None,
                        help="Chemin vers config.yaml (défaut : 0_config/config.yaml)")
    if has_media_dir:
        parser.add_argument(
            "--media-dir",
            help="Racine pour résoudre les media_chemin relatifs "
                 "(défaut : dossier parent de --input)",
        )

    return parser


def analyser_date_arg(s: str) -> date:
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
# Ces fonctions ont été créées après avoir parcouru l'ensemble du corpus
# et établi le plan du mémoire, dans le but de faciliter le traitement
# des données. Aucune des analyses n'a été faite en présupposant que ce
# découpage est objectif.
# ---------------------------------------------------------------------------

_PHASE_DATES: list[tuple[str, date, date]] | None = None

def _initialiser_phases(cfg: dict | None = None) -> list[tuple[str, date, date]]:
    """Initialise la liste de phases depuis config.yaml."""
    global _PHASE_DATES
    if _PHASE_DATES is not None:
        return _PHASE_DATES
    config = cfg or load_config()
    phases = []
    for phase_id, phase_data in config["phases"].items():
        debut = date.fromisoformat(phase_data["start"])
        fin = date.fromisoformat(phase_data["end"])
        phases.append((phase_id, debut, fin))
    _PHASE_DATES = sorted(phases, key=lambda x: x[1])
    return _PHASE_DATES


def etiquette_phase(dt: str | datetime | date, cfg: dict | None = None) -> str | None:
    """
    Retourne l'identifiant de phase (P1/P2/P3) pour une date donnée.
    Accepte : chaîne ISO, datetime, date.
    Retourne None si hors de toutes les phases définies.

    >>> etiquette_phase("2023-06-15")
    'P1'
    >>> etiquette_phase("2024-05-01")
    'P2'
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt[:19])
    if isinstance(dt, datetime):
        dt = dt.date()
    for phase_id, debut, fin in _initialiser_phases(cfg):
        if debut <= dt <= fin:
            return phase_id
    return None


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def init_logger(
    nom_script: str,
    fichier_log: str | Path | None = None,
    cfg: dict | None = None,
) -> logging.Logger:
    """
    Configure et retourne un logger avec :
    - handler console (INFO) — messages courts
    - handler fichier  (WARNING) — erreurs horodatées dans logs/

    Si fichier_log est None, construit le chemin depuis config.yaml (paths.logs_dir).
    """
    config = cfg or load_config()

    if fichier_log is None:
        dossier_logs = Path(config["paths"]["logs_dir"])
        fichier_log = dossier_logs / f"{nom_script}_errors.log"

    fichier_log = Path(fichier_log)
    fichier_log.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(nom_script)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # Fichier
    fh = logging.FileHandler(fichier_log, encoding="utf-8")
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

def est_traite(msg: dict, fields: list[str] | str) -> bool:
    """
    Retourne True si tous les champs listés sont déjà présents dans msg
    (et non-None). Utilisé pour skiper les messages déjà traités.

    >>> est_traite({"dialogue": "..."}, "dialogue")
    True
    >>> est_traite({"a": 1}, ["a", "b"])
    False
    """
    if isinstance(fields, str):
        fields = [fields]
    return all(msg.get(f) is not None for f in fields)


# ---------------------------------------------------------------------------
# LECTURE / ÉCRITURE JSONL
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    """Lit un fichier JSONL, retourne une liste de dicts."""
    chemin = Path(path)
    with open(chemin, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(messages: list[dict], output_path: str | Path) -> None:
    """
    Réécrit le JSONL entier (sauvegarde incrémentale).
    Crée les dossiers parents si nécessaire.
    """
    chemin_sortie = Path(output_path)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# FICHES INDIVIDUELLES
# ---------------------------------------------------------------------------

def chemin_fiche(msg: dict, dossier_fiches: str | Path) -> Path:
    """
    Retourne le chemin de la fiche JSON individuelle pour un message.
    Convention : {dossier_fiches}/{canal}_{id}_fiche.json
    Tolère l'ancien `channel` pour les fiches archives pré-migration.
    """
    canal = msg.get("canal") or msg.get("channel", "robert_magyar")
    mid = msg["message_id"]
    return Path(dossier_fiches) / f"{canal}_{mid}_fiche.json"


def charger_fiche(msg: dict, dossier_fiches: str | Path) -> dict:
    """Charge la fiche individuelle. Retourne {} si elle n'existe pas."""
    p = chemin_fiche(msg, dossier_fiches)
    if p.is_file():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def mettre_a_jour_fiche(
    msg: dict,
    nouveaux_champs: dict,
    dossier_fiches: str | Path,
    overwrite: bool = False,
) -> None:
    """
    Merge incrémental de nouveaux_champs dans la fiche individuelle.
    Si overwrite=False (défaut), les champs existants ne sont pas écrasés.
    """
    p = chemin_fiche(msg, dossier_fiches)
    p.parent.mkdir(parents=True, exist_ok=True)

    existant = charger_fiche(msg, dossier_fiches)
    if overwrite:
        existant.update(nouveaux_champs)
    else:
        for k, v in nouveaux_champs.items():
            if k not in existant or existant[k] is None:
                existant[k] = v

    with open(p, "w", encoding="utf-8") as f:
        json.dump(existant, f, ensure_ascii=False, indent=2)


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


class SuiviProgression:
    """
    Tracker de progression pour les batchs.

    Usage :
        suivi = SuiviProgression(total=len(eligibles), label="whisper")
        for i, msg in enumerate(eligibles):
            suivi.avancer(i, msg["message_id"])
            # ... traitement ...
        suivi.resumer(errors, skipped)
    """

    def __init__(self, total: int, label: str = ""):
        self.total = total
        self.label = label
        self.t0 = time.time()

    def avancer(self, rank: int, msg_id: int | str = "", extra: str = "") -> None:
        ecoule = time.time() - self.t0
        if rank > 0:
            eta = fmt_eta(ecoule / rank * (self.total - rank))
        else:
            eta = "~"
        prefixe = f"[{self.label}] " if self.label else ""
        print(
            f"{prefixe}[{rank+1}/{self.total}] msg {msg_id} "
            f"{extra} (ETA {eta})",
            flush=True,
        )

    def resumer(self, errors: int = 0, skipped: int = 0) -> None:
        ecoule = time.time() - self.t0
        traites = self.total - skipped - errors
        print(
            f"\nTerminé en {fmt_eta(ecoule)} — "
            f"{traites} succès, {skipped} skippés, {errors} erreurs."
        )


# ---------------------------------------------------------------------------
# FILTRAGE DES MESSAGES
# ---------------------------------------------------------------------------
# Utiles pour le déboggage et l'itération sur les scripts, et dans leur
# réutilisation sur d'autres corpus.
# ---------------------------------------------------------------------------

def filtrer_eligibles(
    messages: list[dict],
    *,
    filtre_ids: set[int] | None = None,
    media_types: list[str] | None = None,
    champs_a_verifier: list[str] | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[int]:
    """
    Retourne les indices des messages à traiter selon les filtres.

    - filtre_ids        : ne traiter que ces message_id
    - media_types       : ne traiter que ces types (ex: ["video", "audio"])
    - champs_a_verifier : skipper si tous ces champs sont déjà présents (idempotence)
    - overwrite         : si True, désactive le check d'idempotence
    - limit             : max N messages
    - start_date        : ne garder que les messages dont le champ "date" >= start_date
    - end_date          : ne garder que les messages dont le champ "date" <= end_date
    """
    eligibles = []
    for i, msg in enumerate(messages):
        # Filtre par ID
        if filtre_ids and msg.get("message_id") not in filtre_ids:
            continue
        # Filtre par type de média
        if media_types and msg.get("media_type") not in media_types:
            continue
        # Filtre par dates
        if start_date or end_date:
            brut = msg.get("date")
            if brut:
                try:
                    date_msg = datetime.fromisoformat(brut[:19]).date()
                except (ValueError, TypeError):
                    date_msg = None
            else:
                date_msg = None
            if date_msg is None:
                continue
            if start_date and date_msg < start_date:
                continue
            if end_date and date_msg > end_date:
                continue
        # Idempotence
        if not overwrite and champs_a_verifier and est_traite(msg, champs_a_verifier):
            continue
        eligibles.append(i)

    if limit:
        eligibles = eligibles[:limit]
    return eligibles


# ---------------------------------------------------------------------------
# EXPRESSIONS RÉGULIÈRES
# ---------------------------------------------------------------------------

# Caractères cyrilliques — partagé par whisper_batch + translate_srt
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
