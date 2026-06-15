#!/usr/bin/env python3
"""
lexicometrie.py — Pipeline de lemmatisation et d'analyse lexicale du corpus Magyar.

Ce que ce script lit : un JSONL enrichi (champs legende, dialogue, ocr_texte, date,
    message_id, dialogue_confiance).
Ce qu'il produit : des CSV dans --output-dir —
    lemmes_{caption,dialogue,ocr,combined}.csv  : tokens lemmatisés par message
    tfidf_{caption,dialogue,combined}.csv        : scores TF-IDF par phase
    stats_phases.csv                             : tokens, types, TTR, hapax par phase
    temporal_stats_{...}.csv                     : mêmes stats par période temporelle
    volcano_{...}.csv                            : log2FC + p-valeur ajustée (BH) par paire

Ce script ne produit AUCUNE figure — elles viennent des analyses dérivées
(lda_topics, afc, cah…) et des scripts R (3b_stats_R).

Pipeline interne :
  1. Chargement du JSONL et des phases depuis config.yaml
  2. Chargement du modèle spaCy uk_core_news_trf
  3. Lemmatisation (filtres POS + stopwords) de caption, dialogue, OCR
  4. Export CSV lemmes (base de toutes les analyses aval)
  5. Calcul TF-IDF par phase + export CSV
  6. Stats descriptives (TTR, hapax) par phase et par période
  7. Calcul volcano (chi2 + correction FDR Benjamini-Hochberg) + export CSV

Options CLI notables : --stopwords, --top-n, --min-confidence, --temporal-bin

Dépendances : spacy uk_core_news_trf, scikit-learn, scipy, statsmodels, pandas
"""

import math
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

_UTILS_DIR = Path(__file__).resolve().parents[1] / "0_config"
sys.path.insert(0, str(_UTILS_DIR))
from utils import (  # noqa: E402
    load_config,
    read_jsonl,
    creer_parser_base,
)

import pandas as pd
import spacy
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_cache = None


def _get_cfg(config_path=None):
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config(config_path)
    return _cfg_cache


def _construire_phases_par_defaut(cfg):
    """Construit la liste de phases (start, end, label) depuis config.yaml."""
    phases = []
    for pid in sorted(cfg.get("phases", {}).keys()):
        pdata = cfg["phases"][pid]
        start = datetime.strptime(pdata["start"], "%Y-%m-%d")
        end   = datetime.strptime(pdata["end"],   "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )
        label = f"{pid}_{pdata.get('label', '')}"
        phases.append((start, end, label))
    return phases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = creer_parser_base("Lexicométrie — corpus Magyar", has_output=False)
    p.add_argument(
        "--phases", nargs="*", default=None,
        help='Phases nommées : "start:end:label". Si omis, phases depuis config.yaml.',
    )
    p.add_argument("--no-phases", action="store_true",
                   help="Désactiver les phases nommées (analyse temporelle seule)")
    p.add_argument("--temporal-bin", choices=["month", "quarter", "year"],
                   default="month",
                   help="Granularité temporelle pour les stats par période (défaut : month)")
    p.add_argument("--output-dir",
                   default=os.path.join(
                       os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "4_data_et_viz", "lexico"
                   ))
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Seuil dialogue_confiance pour inclure un dialogue (défaut : 0.5)")
    p.add_argument("--min-count", type=int, default=5,
                   help="Fréquence minimale d'un lemme pour le volcano (défaut : 5). "
                        "N.B. le TF-IDF par phase n'applique PAS ce filtre (min_df=1).")
    p.add_argument("--stopwords", default=None,
                   help="Fichier texte de stopwords additionnels, un mot par ligne")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top N lemmes dans le CSV TF-IDF par phase (défaut : 20)")
    p.add_argument("--ocr-field", default="ocr_texte",
                   help="Champ JSON pour le texte OCR (défaut : ocr_texte)")
    return p.parse_args()


def analyser_phases(phase_strings):
    """Parse les phases CLI au format 'start:end:label'."""
    phases = []
    for s in phase_strings:
        parts = s.split(":")
        if len(parts) != 3:
            print(f"ERREUR format phase : {s!r}  (attendu start:end:label)", file=sys.stderr)
            sys.exit(1)
        start = datetime.strptime(parts[0], "%Y-%m-%d")
        end   = datetime.strptime(parts[1], "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        phases.append((start, end, parts[2]))
    return phases


def assigner_phase(dt, phases):
    """Retourne le label de phase correspondant à dt, ou None si hors bornes."""
    for start, end, label in phases:
        if start <= dt <= end:
            return label
    return None


def etiquette_periode(dt, bin_type):
    """Retourne un label de période : '2022-09', '2022-Q3' ou '2022'."""
    if bin_type == "month":
        return dt.strftime("%Y-%m")
    elif bin_type == "quarter":
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"
    else:
        return str(dt.year)


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

def charger_stopwords(path, nlp):
    """
    Construit l'ensemble des stopwords à partir des défauts spaCy
    et d'un fichier optionnel (un mot par ligne).

    Entrée : path — chemin vers le fichier texte (ou None), nlp — modèle spaCy
    Sortie : set de chaînes de caractères en minuscules
    """
    stops = set(nlp.Defaults.stop_words)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    stops.add(w)
    return stops


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------

def lemmatiser_texte(text, nlp, stopwords, exclude_pos):
    """
    Lemmatise un texte ukrainien et retourne les tokens filtrés.

    Filtres appliqués dans l'ordre :
      1. POS dans exclude_pos — exclut catégories grammaticales (lues depuis config)
      2. Longueur lemme < 2 — élimine les tokens d'un seul caractère
      3. Lemme dans stopwords — liste spaCy + fichier custom

    Entrée : text — str, nlp — modèle spaCy, stopwords — set,
             exclude_pos — set de tags POS universels à exclure
    Sortie : liste de dicts {token, lemma, pos}
    """
    doc     = nlp(text)
    results = []
    for tok in doc:
        # On exclut les catégories POS sans charge sémantique thématique
        if tok.pos_ in exclude_pos:
            continue
        lemma = tok.lemma_.lower()
        # On exclut les lemmes trop courts (chiffres isolés, résidus de tokenisation)
        if len(lemma) < 2:
            continue
        if lemma in stopwords:
            continue
        results.append({"token": tok.text, "lemma": lemma, "pos": tok.pos_})
    return results


def traiter_messages(messages, phases, nlp, stopwords, exclude_pos,
                     min_confidence, temporal_bin, ocr_field):
    """
    Lemmatise chaque message pour les quatre sources textuelles.

    Retourne un dict {source: [(phase, period, msg_id, date_str, dt, lemmes), ...]}
    Sources : caption, dialogue, ocr, combined

    Le champ dialogue est exclu si dialogue_confiance < min_confidence —
    évite d'injecter des transcriptions peu fiables dans le corpus lexical.
    """
    sources = ("caption", "dialogue", "ocr", "combined")
    results = {s: [] for s in sources}
    total = len(messages)
    t0    = time.time()

    for i, msg in enumerate(messages):
        mid      = msg.get("message_id", i)
        date_str = msg.get("date", "")
        try:
            dt = datetime.fromisoformat(date_str)
        except (ValueError, TypeError):
            continue

        # Affectation de phase nommée — None si le message est hors bornes
        if phases is not None:
            phase = assigner_phase(dt, phases)
        else:
            phase = "corpus"

        period = etiquette_periode(dt, temporal_bin)

        # On exclut le dialogue si la confiance composite QA est insuffisante.
        # `dialogue_confiance` (score QA composite) remplace l'ancien `speech_confidence`
        # — score 0-1 mais avec pénalités de qualité (hallucination, répétition, logprob).
        caption_text  = (msg.get("legende")    or "").strip()
        conf          = msg.get("dialogue_confiance", 0) or 0
        dialogue_text = (msg.get("dialogue")   or "").strip() if conf >= min_confidence else ""
        ocr_text_raw  = (msg.get(ocr_field)    or "").strip()

        cap_l = lemmatiser_texte(caption_text,  nlp, stopwords, exclude_pos) if caption_text  else []
        dia_l = lemmatiser_texte(dialogue_text, nlp, stopwords, exclude_pos) if dialogue_text else []
        ocr_l = lemmatiser_texte(ocr_text_raw,  nlp, stopwords, exclude_pos) if ocr_text_raw  else []

        combined_text = " ".join(filter(None, [caption_text, dialogue_text, ocr_text_raw]))
        com_l = lemmatiser_texte(combined_text, nlp, stopwords, exclude_pos) if combined_text else []

        record_base = (phase, period, mid, date_str, dt)
        for src, lemmes in [("caption", cap_l), ("dialogue", dia_l),
                             ("ocr", ocr_l), ("combined", com_l)]:
            if lemmes:
                results[src].append((*record_base, lemmes))

        # Progression en console
        elapsed   = time.time() - t0
        speed     = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (total - i - 1) / speed if speed > 0 else 0
        print(f"\r  {i+1}/{total} ({100*(i+1)/total:.1f}%) — {remaining:.0f}s restant",
              end="", flush=True)

    print()
    return results


# ---------------------------------------------------------------------------
# Export CSV — lemmes
# ---------------------------------------------------------------------------

def exporter_lemmes(results, src, output_dir):
    """Exporte les lemmes d'une source en CSV plat (un token par ligne)."""
    rows = []
    for phase, period, mid, date_str, dt, lemmes in results[src]:
        for lem in lemmes:
            rows.append({
                "message_id": mid,
                "date":       date_str,
                "period":     period,
                "phase":      phase,
                "token":      lem["token"],
                "lemma":      lem["lemma"],
                "pos":        lem["pos"],
            })
    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"lemmes_{src}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {path} ({len(df)} lignes)")
    return df


# ---------------------------------------------------------------------------
# TF-IDF par phase nommée
# ---------------------------------------------------------------------------

def calculer_tfidf(results, src, top_n, output_dir):
    """
    Calcule le TF-IDF par phase nommée et exporte le CSV.

    Chaque phase est traitée comme un seul document agrégé.
    use_idf=False quand il n'y a qu'une phase (évite division par zéro).
    Les messages hors phase (phase=None) sont ignorés.
    """
    # On agrège les lemmes par phase — chaque phase = un pseudo-document
    docs_par_phase = defaultdict(list)
    for phase, period, _mid, _date, _dt, lemmes in results[src]:
        if phase is None:
            continue
        docs_par_phase[phase].extend([lem["lemma"] for lem in lemmes])

    if not docs_par_phase:
        print(f"  TF-IDF {src} : aucune phase nommée disponible.")
        return pd.DataFrame()

    phase_labels = sorted(docs_par_phase.keys())
    documents    = [" ".join(docs_par_phase[p]) for p in phase_labels]
    use_idf      = len(documents) > 1

    # token_pattern exige un premier caractère alphabétique — évite que
    # CountVectorizer extraie des chiffres isolés depuis des lemmes comme "23-го"
    # (spaCy conserve le tiret dans le lemme ; le split au tiret produisait "23")
    vectorizer   = CountVectorizer(token_pattern=r"(?u)\b[^\W\d_]\w*\b")
    dtm          = vectorizer.fit_transform(documents)
    transformer  = TfidfTransformer(use_idf=use_idf)
    tfidf_matrix = transformer.fit_transform(dtm)
    feat_names   = vectorizer.get_feature_names_out()

    rows = []
    for idx, phase in enumerate(phase_labels):
        scores  = tfidf_matrix[idx].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        for rank, j in enumerate(top_idx, 1):
            if scores[j] > 0:
                rows.append({
                    "phase": phase, "rank": rank,
                    "lemma": feat_names[j], "tfidf": round(float(scores[j]), 6),
                })

    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"tfidf_{src}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {path} ({len(df)} lignes)")
    return df


# ---------------------------------------------------------------------------
# Stats descriptives par phase
# ---------------------------------------------------------------------------

def calculer_stats(results, output_dir):
    """Calcule tokens, types, TTR et hapax par phase et par source textuelle."""
    rows = []
    for src in ("caption", "dialogue", "ocr", "combined"):
        lemmes_par_phase  = defaultdict(list)
        n_msgs_par_phase = defaultdict(int)
        for phase, period, _mid, _date, _dt, lemmes in results[src]:
            # On skip les messages hors phase (cohérent avec calculer_tfidf l. 310).
            if phase is None:
                continue
            lemmes_par_phase[phase].extend([lem["lemma"] for lem in lemmes])
            n_msgs_par_phase[phase] += 1

        for phase in sorted(lemmes_par_phase.keys()):
            lemmes = lemmes_par_phase[phase]
            counts = Counter(lemmes)
            total  = len(lemmes)
            unique = len(counts)
            ttr    = unique / total if total > 0 else 0
            hapax  = sum(1 for c in counts.values() if c == 1)
            rows.append({
                "source":   src, "phase": phase,
                "messages": n_msgs_par_phase[phase],
                "tokens":   total, "types": unique,
                "ttr":      round(ttr, 4), "hapax": hapax,
            })

    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, "stats_phases.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {path}")
    return df


# ---------------------------------------------------------------------------
# Stats temporelles par période
# ---------------------------------------------------------------------------

def calculer_stats_temporelles(results, src, output_dir):
    """Calcule TTR, tokens et types par période temporelle (mois / trimestre / an)."""
    lemmes_par_periode  = defaultdict(list)
    n_msgs_par_periode = defaultdict(int)
    for phase, period, _mid, _date, dt, lemmes in results[src]:
        lemmes_par_periode[period].extend([lem["lemma"] for lem in lemmes])
        n_msgs_par_periode[period] += 1

    rows = []
    for period in sorted(lemmes_par_periode.keys()):
        lemmes = lemmes_par_periode[period]
        counts = Counter(lemmes)
        total  = len(lemmes)
        unique = len(counts)
        rows.append({
            "period":   period,
            "messages": n_msgs_par_periode[period],
            "tokens":   total,
            "types":    unique,
            "ttr":      round(unique / total, 4) if total > 0 else 0,
            "hapax":    sum(1 for c in counts.values() if c == 1),
        })

    df   = pd.DataFrame(rows)
    path = os.path.join(output_dir, f"temporal_stats_{src}.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  -> {path} ({len(df)} lignes)")
    return df


# ---------------------------------------------------------------------------
# Volcano — chi2 + correction FDR Benjamini-Hochberg
# ---------------------------------------------------------------------------

def calculer_donnees_volcano(results, src, min_count):
    """
    Calcule les fold-changes et p-valeurs ajustées entre chaque paire de phases.

    Pour chaque lemme présent dans au moins une phase avec count >= min_count :
      - log2(fold-change) entre fréquences relatives (avec pseudo-count 0.5)
      - chi2 de contingence (correction de Yates)
      - correction FDR Benjamini-Hochberg par paire de phases

    La pseudo-fréquence (+ 0.5) évite les log2(0) et stabilise les ratios
    sur les lemmes rares — convention standard en transcriptomique, adaptée ici.

    Entrée : results dict, src — source textuelle, min_count — fréquence minimale
    Sortie : DataFrame avec colonnes comparison, lemma, log2fc, pvalue_raw, pvalue_adj, ...
    """
    compteurs_par_phase = defaultdict(Counter)
    totaux_par_phase   = defaultdict(int)
    for phase, period, _mid, _date, _dt, lemmes in results[src]:
        if phase is None:
            continue
        for lem in lemmes:
            compteurs_par_phase[phase][lem["lemma"]] += 1
            totaux_par_phase[phase] += 1

    phases = sorted(compteurs_par_phase.keys())
    if len(phases) < 2:
        return pd.DataFrame()

    # Vocabulaire filtré : au moins min_count occurrences dans AU MOINS une phase
    all_lemmes = {
        lemma
        for c in compteurs_par_phase.values()
        for lemma, cnt in c.items()
        if cnt >= min_count
    }

    rows = []
    for pa, pb in combinations(phases, 2):
        total_a = totaux_par_phase[pa]
        total_b = totaux_par_phase[pb]
        if total_a == 0 or total_b == 0:
            continue

        for lemma in all_lemmes:
            count_a = compteurs_par_phase[pa].get(lemma, 0)
            count_b = compteurs_par_phase[pb].get(lemma, 0)
            if count_a == 0 and count_b == 0:
                continue

            # Pseudo-fréquences pour éviter log2(0)
            freq_a = (count_a + 0.5) / total_a
            freq_b = (count_b + 0.5) / total_b
            log2fc = math.log2(freq_b / freq_a)

            rest_a = total_a - count_a
            rest_b = total_b - count_b
            table  = [[count_a, count_b], [rest_a, rest_b]]
            try:
                _, pval, _, _ = chi2_contingency(table, correction=True)
            except ValueError:
                pval = 1.0

            rows.append({
                "comparison": f"{pa} → {pb}",
                "lemma":      lemma,
                "log2fc":     round(log2fc, 4),
                "pvalue_raw": pval,
                "count_a":    count_a,
                "count_b":    count_b,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Correction FDR par paire de phases (Benjamini-Hochberg)
    adj_list = []
    for comp, sub in df.groupby("comparison"):
        _, pvals_adj, _, _ = multipletests(sub["pvalue_raw"].values, method="fdr_bh")
        adj_list.append(pd.Series(pvals_adj, index=sub.index))
    df["pvalue_adj"] = pd.concat(adj_list)

    # Forme canonique pour le volcano plot : -log10(p_adj), plafonné à 50
    df["neg_log10_padj"] = df["pvalue_adj"].apply(
        lambda p: min(-math.log10(p) if p > 0 else 50, 50)
    )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg = _get_cfg(args.config)

    # On lit les seuils depuis config, surchargeable par CLI
    min_count = args.min_count

    # On lit les POS à exclure depuis config — évite de hardcoder dans le script
    spacy_cfg   = cfg.get("models", {}).get("spacy", {})
    exclude_pos = set(spacy_cfg.get(
        "excluded_pos",
        ["PUNCT", "SPACE", "SYM", "NUM", "X",
         "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ"]
    ))

    # Phases
    if args.no_phases:
        print("Mode sans phases nommées — analyse temporelle seule.")
        phases = None
    elif args.phases:
        phases = analyser_phases(args.phases)
        print(f"Phases CLI : {[p[2] for p in phases]}")
    else:
        phases = _construire_phases_par_defaut(cfg)
        print(f"Phases depuis config : {[p[2] for p in phases]}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Chargement de {args.input}...")
    messages = read_jsonl(args.input)
    print(f"  {len(messages)} messages chargés.")

    # Filtrage optionnel par date / ids / limit
    if args.ids or args.start_date or args.end_date or args.limit:
        from utils import filtrer_eligibles
        filtre_ids = set(args.ids) if args.ids else None
        indices_eligibles = filtrer_eligibles(
            messages, filtre_ids=filtre_ids,
            start_date=args.start_date, end_date=args.end_date,
            limit=args.limit,
        )
        messages = [messages[i] for i in indices_eligibles]
        print(f"  {len(messages)} messages après filtrage.")

    print("Chargement du modèle spaCy uk_core_news_trf...")
    t0  = time.time()
    nlp = spacy.load("uk_core_news_trf")
    print(f"  Modèle chargé en {time.time() - t0:.1f}s.")

    stopwords = charger_stopwords(args.stopwords, nlp)
    print(f"  {len(stopwords)} stopwords  |  {len(exclude_pos)} POS exclus : {sorted(exclude_pos)}")

    # ── Lemmatisation ─────────────────────────────────────────────────────────
    print("\nLemmatisation en cours...")
    results = traiter_messages(
        messages, phases, nlp, stopwords, exclude_pos,
        args.min_confidence, args.temporal_bin, args.ocr_field,
    )
    for src in ("caption", "dialogue", "ocr", "combined"):
        print(f"  {src} : {len(results[src])} messages avec texte")

    # ── Export CSV lemmes ──────────────────────────────────────────────────────
    print("\nExport lemmes...")
    for src in ("caption", "dialogue", "ocr", "combined"):
        if results[src]:
            exporter_lemmes(results, src, args.output_dir)

    # ── TF-IDF par phase ──────────────────────────────────────────────────────
    if phases:
        print("\nTF-IDF par phase...")
        for src in ("caption", "dialogue", "combined"):
            calculer_tfidf(results, src, args.top_n, args.output_dir)

    # ── Stats descriptives ────────────────────────────────────────────────────
    print("\nStats descriptives...")
    calculer_stats(results, args.output_dir)

    print("\nStats temporelles...")
    for src in ("caption", "dialogue", "combined"):
        calculer_stats_temporelles(results, src, args.output_dir)

    # ── Volcano (chi2 + FDR) ──────────────────────────────────────────────────
    if phases:
        print("\nVolcano (chi2 + Benjamini-Hochberg)...")
        for src in ("caption", "dialogue", "combined"):
            df_vol = calculer_donnees_volcano(results, src, min_count)
            if not df_vol.empty:
                path = os.path.join(args.output_dir, f"volcano_{src}.csv")
                df_vol.to_csv(path, index=False, encoding="utf-8")
                print(f"  -> {path} ({len(df_vol)} lignes)")
            else:
                print(f"  Volcano {src} : pas assez de données.")

    print(f"\nTerminé. CSVs dans : {args.output_dir}")


if __name__ == "__main__":
    main()
