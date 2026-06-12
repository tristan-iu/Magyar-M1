#!/usr/bin/env python3
"""
qa_whisper.py — Score de confiance des transcriptions Whisper.

Évalue `dialogue` depuis `whisper_segments` (openai-whisper — logprob +
no_speech_prob). ⚠️ whisper_batch ne persiste plus ces segments dans le JSONL
(strippés, −16 MB) : les `dialogue_confiance` du corpus viennent d'un run
historique, valides mais non reproductibles depuis le seul JSONL. Pour ne pas
les écraser, ce script REFUSE de tourner (exit 1, rien écrit) si les messages
éligibles n'ont pas de segments — sauf `--allow-degraded` (mode texte-seul :
logprob ignoré, score plafonné à 0.3). Voir README.

Ajoute au JSONL :
  - dialogue_confiance              : float [0, 1]  — score composite
  - alerte_low_conf                 : bool — proportion logprob < -1 trop haute
  - alerte_high_no_speech           : bool — proportion no_speech_prob > 0.6 trop haute
  - alerte_hallucination_phrase     : bool — pattern YAML détecté
  - alerte_compression_high         : bool — len(tokens)/len(unique) > 2.4
  - alerte_repeated_ngram           : bool — un trigramme occupe > 25% du texte
  - alerte_tokens_per_sec_anomaly   : bool — débit hors [0.5, 5.0] tok/s
  - alerte_no_dialogue              : bool — pas de dialogue exploitable
  - qa_patterns_version             : str  — version du YAML de patterns

Patterns d'hallucination : externalisés dans 0_config/hallucination_patterns.yaml.
Version des patterns injectée dans le champ `qa_patterns_version`.

Usage :
    python qa_whisper.py --input  messages_clean.jsonl \\
                         --output messages_clean.jsonl

    # Avec fichier patterns personnalisé
    python qa_whisper.py --input ... --output ... \\
                         --patterns-file 0_config/hallucination_patterns.yaml

    # Exporter CSV synthèse
    python qa_whisper.py --input ... --output ... --csv output/qa_whisper.csv

    # Forcer un scoring dégradé (sans segments) — à éviter sur le canonique
    python qa_whisper.py --input ... --output ... --allow-degraded
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    creer_parser_base,
    filtrer_eligibles,
    read_jsonl,
    init_logger,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Chargement des patterns depuis hallucination_patterns.yaml (externalisé)
# ---------------------------------------------------------------------------

_PATTERNS_CACHE: dict[str, tuple["re.Pattern", str]] = {}  # path → (pattern, version)

# Patterns de secours si le YAML est introuvable (rétro-compat v1)
# Note : "дякую\s+за\s+перегляд" retiré — faux positif sur le signe de Magyar
_FALLBACK_PATTERNS = [
    r"дякую\s+за\s+увагу",
    r"підпишіться\s+на\s+канал",
    r"не\s+забудьте\s+підписатися",
    r"субтитри\s+(виконано|створено|зроблено)",
    r"переклад\s+з\s+",
    r"спасибо\s+за\s+просмотр",
    r"подписывайтесь\s+на\s+канал",
    r"ставьте\s+лайк",
    r"^\s*\.{3,}\s*$",
    r"^\s*-\s*$",
]

_DEFAULT_YAML = Path(_UTILS_DIR) / "hallucination_patterns.yaml"


def charger_patterns(yaml_path: Path | None = None) -> tuple["re.Pattern", str]:
    """Charge et compile les patterns depuis le YAML. Cache par chemin.

    Entrée : yaml_path — chemin YAML (défaut : 0_config/hallucination_patterns.yaml)
    Sortie : (pattern compilé, version str)
    """
    chemin = str(yaml_path or _DEFAULT_YAML)
    if chemin in _PATTERNS_CACHE:
        return _PATTERNS_CACHE[chemin]

    patterns: list[str] = []
    version = "fallback"
    try:
        import yaml
        with open(chemin, encoding="utf-8") as f:
            hal = yaml.safe_load(f)
        version = str(hal.get("version", "?"))
        for categorie, liste in hal.get("patterns", {}).items():
            if isinstance(liste, list):
                patterns.extend(str(p) for p in liste)
    except Exception:
        patterns = _FALLBACK_PATTERNS

    compiled = re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)
    _PATTERNS_CACHE[chemin] = (compiled, version)
    return compiled, version


# ---------------------------------------------------------------------------
# Seuils
# ---------------------------------------------------------------------------
LOGPROB_BAD          = -1.0   # avg_logprob < -1.0 = segment peu fiable (OpenAI)
LOGPROB_GREAT        = -0.3   # au-dessus : transcription propre
NO_SPEECH_BAD        = 0.6    # no_speech_prob > 0.6 = probable silence transcrit
TOKENS_PER_SEC_MIN   = 0.5    # < 0.5 = trop peu de texte pour la durée
TOKENS_PER_SEC_MAX   = 5.0    # > 5.0 = débit inhumain, probable hallucination
COMPRESSION_RATIO_MAX = 2.4   # > 2.4 = répétition pathologique (seuil Whisper)
REPEATED_NGRAM_MAX   = 0.25   # >25% du texte = même trigramme → hallucination


# ---------------------------------------------------------------------------
# Calcul des signaux bruts
# ---------------------------------------------------------------------------
def _jetons(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _trigrammes(tokens: list[str]) -> list[tuple[str, str, str]]:
    return [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]


def calculer_qa(msg: dict, halluc_pattern: "re.Pattern | None" = None) -> dict:
    """Retourne un dict de signaux QA pour un message (ou {} si pas de dialogue).

    Lit whisper_segments pour logprob/no_speech_prob.
    Si whisper_segments absent, les métriques logprob sont None (mode dégradé).

    Entrée : msg — dict message, halluc_pattern — pattern compilé (auto-chargé si None)
    Sortie : dict de signaux QA ou {} si pas de dialogue
    """
    segments = msg.get("whisper_segments") or []
    dialogue = (msg.get("dialogue") or "").strip()
    duration = msg.get("parole_duree") or msg.get("duree") or 0.0

    if not dialogue:
        return {}

    # On accepte sans segments (mode dégradé — métriques logprob seront None)
    logprobs = [s.get("avg_logprob") for s in segments if s.get("avg_logprob") is not None]
    nsprobs  = [s.get("no_speech_prob") for s in segments if s.get("no_speech_prob") is not None]

    tokens = _jetons(dialogue)
    n_tok  = len(tokens)
    trigs  = _trigrammes(tokens)

    # Répétition : ratio du trigramme le plus fréquent sur total
    if trigs:
        top_trig_count = Counter(trigs).most_common(1)[0][1]
        repeated_ratio = top_trig_count / len(trigs)
    else:
        repeated_ratio = 0.0

    # compression_ratio : approximation len(tokens) / len(tokens uniques)
    compression = n_tok / max(1, len(set(tokens))) if tokens else 0.0

    tps = n_tok / duration if duration > 0 else 0.0

    # Pattern hallucination (charge depuis YAML si non fourni)
    if halluc_pattern is None:
        halluc_pattern, _ = charger_patterns()

    return {
        "n_segments":       len(segments),
        "n_tokens":         n_tok,
        "duration_s":       round(duration, 1),
        "tokens_per_sec":   round(tps, 2),
        "logprob_median":   round(statistics.median(logprobs), 3) if logprobs else None,
        "logprob_p10":      round(statistics.quantiles(logprobs, n=10)[0], 3)
                            if len(logprobs) >= 10 else
                            (round(min(logprobs), 3) if logprobs else None),
        "prop_low_conf":    round(sum(1 for x in logprobs if x < LOGPROB_BAD) / len(logprobs), 3)
                            if logprobs else 0.0,
        "prop_high_nsp":    round(sum(1 for x in nsprobs if x > NO_SPEECH_BAD) / len(nsprobs), 3)
                            if nsprobs else 0.0,
        "compression":      round(compression, 2),
        "repeated_ratio":   round(repeated_ratio, 3),
        "hallucination_hit": bool(halluc_pattern.search(dialogue)),
    }


# ---------------------------------------------------------------------------
# Score composite + flags
# ---------------------------------------------------------------------------
ALERTE_FIELDS = (
    "alerte_low_conf",
    "alerte_high_no_speech",
    "alerte_hallucination_phrase",
    "alerte_compression_high",
    "alerte_repeated_ngram",
    "alerte_tokens_per_sec_anomaly",
    "alerte_no_dialogue",
)


def calculer_score(qa: dict) -> tuple[float, dict[str, bool]]:
    """
    Combine les signaux en un score [0, 1] et 7 alertes booléennes.

    Stratégie : on part d'un score basé sur la médiane de logprob
    (remappée de [-2, 0] vers [0, 1]) et on applique des pénalités
    additives pour chaque signal suspect.
    """
    alertes = {field: False for field in ALERTE_FIELDS}

    if not qa:
        alertes["alerte_no_dialogue"] = True
        return 0.0, alertes

    # Score de base depuis la médiane de logprob
    lp = qa.get("logprob_median")
    if lp is None:
        base = 0.3
    else:
        # -2.0 → 0,  -1.0 → 0.5,  0.0 → 1
        base = max(0.0, min(1.0, (lp + 2.0) / 2.0))

    # Pénalités additives (pondérées)
    penalty = 0.0

    if qa["prop_low_conf"] > 0.3:
        penalty += 0.20 * qa["prop_low_conf"]
        alertes["alerte_low_conf"] = True

    if qa["prop_high_nsp"] > 0.3:
        penalty += 0.15 * qa["prop_high_nsp"]
        alertes["alerte_high_no_speech"] = True

    if qa["hallucination_hit"]:
        penalty += 0.30
        alertes["alerte_hallucination_phrase"] = True

    if qa["compression"] > COMPRESSION_RATIO_MAX:
        penalty += 0.25
        alertes["alerte_compression_high"] = True

    if qa["repeated_ratio"] > REPEATED_NGRAM_MAX:
        penalty += 0.30
        alertes["alerte_repeated_ngram"] = True

    tps = qa["tokens_per_sec"]
    if tps and (tps < TOKENS_PER_SEC_MIN or tps > TOKENS_PER_SEC_MAX):
        penalty += 0.15
        alertes["alerte_tokens_per_sec_anomaly"] = True

    score = max(0.0, min(1.0, base - penalty))
    return round(score, 3), alertes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def construire_parser() -> argparse.ArgumentParser:
    p = creer_parser_base(
        "Score de confiance des transcriptions Whisper — enrichit le JSONL.",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Si fourni, écrit aussi un CSV synthèse (une ligne par message).",
    )
    p.add_argument(
        "--suspect-threshold",
        type=float,
        default=0.5,
        help="Seuil de confiance en dessous duquel un message est compté "
             "comme suspect dans le rapport (défaut : 0.5).",
    )
    p.add_argument(
        "--patterns-file",
        default=None,
        help="Chemin vers le YAML de patterns d'hallucination "
             "(défaut : 0_config/hallucination_patterns.yaml).",
    )
    p.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Forcer le scoring même sans whisper_segments (mode texte-seul : "
             "le signal logprob est ignoré, score plafonné à 0.3). Par défaut "
             "le script refuse pour ne pas écraser les dialogue_confiance "
             "historiques valides. Voir README.",
    )
    return p


def main() -> int:
    args = construire_parser().parse_args()
    log  = init_logger("qa_whisper")

    # Chargement patterns depuis YAML (avec version pour traçabilité)
    yaml_path = Path(args.patterns_file) if args.patterns_file else None
    halluc_pattern, patterns_version = charger_patterns(yaml_path)
    log.info(f"Patterns hallucination version={patterns_version} chargés.")

    messages = read_jsonl(args.input)
    log.info(f"Chargé {len(messages)} messages depuis {args.input}")

    champ_confiance = "dialogue_confiance"
    champs_a_traiter = [champ_confiance, *ALERTE_FIELDS]

    # ── Filtrage des messages à scorer ──
    # On route par filtrer_eligibles pour honorer --ids / --start-date /
    # --end-date (ajoutés par creer_parser_base) ET l'idempotence. Sans ça, la
    # boucle scorait TOUT le corpus même avec --ids ciblé : un --overwrite
    # destiné à un seul message clobberait tous les dialogue_confiance.
    filtre_ids = set(args.ids) if args.ids else None
    eligibles = filtrer_eligibles(
        messages,
        filtre_ids=filtre_ids,
        champs_a_verifier=champs_a_traiter,
        overwrite=args.overwrite,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
    )

    # ── Garde-fou : segments Whisper absents ──
    # Le score s'appuie sur whisper_segments (avg_logprob, no_speech_prob), que
    # whisper_batch ne persiste plus dans le JSONL (strippés, −16 MB). Sans eux,
    # calculer_score plafonne à 0.3 (base quand logprob_median is None) → tout
    # devient « suspect » et les dialogue_confiance historiques valides seraient
    # écrasés par du bruit. On refuse de tourner, sauf --allow-degraded explicite.
    avec_dialogue = [i for i in eligibles if (messages[i].get("dialogue") or "").strip()]
    sans_segments = [i for i in avec_dialogue if not messages[i].get("whisper_segments")]
    if avec_dialogue and len(sans_segments) == len(avec_dialogue) and not args.allow_degraded:
        log.error(
            "Aucun message éligible n'a de `whisper_segments` : score fiable "
            "impossible (le signal avg_logprob/no_speech_prob vient de Whisper "
            "et n'est plus persisté dans le JSONL)."
        )
        log.error(
            "Les `dialogue_confiance` actuels proviennent d'un run historique et "
            "restent VALIDES. Les recalculer maintenant les écraserait par un "
            "plancher 0.3 (mode dégradé)."
        )
        log.error(
            "Pour scorer quand même en mode texte-seul, relancer avec "
            "--allow-degraded. Rien n'a été écrit."
        )
        return 1

    if sans_segments and args.allow_degraded:
        log.warning(
            "%d/%d messages scorés sans whisper_segments : mode dégradé "
            "(logprob ignoré, score plafonné à 0.3, alertes low_conf/"
            "high_no_speech jamais déclenchées).",
            len(sans_segments), len(avec_dialogue),
        )

    n_scored = 0
    n_suspect = 0
    n_no_dialogue = 0
    n_skipped = len(messages) - len(eligibles)

    rows_csv = []

    for idx in eligibles:
        msg = messages[idx]

        # On passe le pattern pour éviter de le recharger à chaque appel
        qa = calculer_qa(msg, halluc_pattern)

        if not qa:
            msg[champ_confiance] = None
            for field in ALERTE_FIELDS:
                msg[field] = False
            msg["alerte_no_dialogue"] = True
            msg["qa_patterns_version"] = patterns_version
            n_no_dialogue += 1
            continue

        score, alertes = calculer_score(qa)
        msg[champ_confiance] = score
        msg.update(alertes)
        msg["qa_patterns_version"] = patterns_version
        n_scored += 1
        if score < args.suspect_threshold:
            n_suspect += 1

        if args.csv:
            triggered = [f for f, v in alertes.items() if v]
            rows_csv.append({
                "message_id":  msg.get("message_id"),
                "date":        msg.get("date"),
                "confiance":   score,
                "alertes":     "|".join(triggered),
                **{k: v for k, v in qa.items() if k != "hallucination_hit"},
            })

    write_jsonl(messages, args.output)
    log.info(
        f"Écrit {args.output} — scorés={n_scored}, "
        f"skippés idempotence={n_skipped}, sans dialogue={n_no_dialogue}, "
        f"suspects (<{args.suspect_threshold})={n_suspect}"
    )

    if args.csv and rows_csv:
        import csv
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        log.info(f"CSV synthèse écrit : {out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
