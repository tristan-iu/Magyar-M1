"""
categories_semantiques.py — Constantes partagées des scripts lexicométriques.

Centralise :
  - les catégories sémantiques FINANCE / MILITAIRE / ASSOCIÉS (catégorie par
    défaut pour tout lemme non listé)
  - la table de traductions uk→fr utilisée par les tableaux TF-IDF
  - la palette `COLORS` partagée par les figures de `tableaux_tfidf.py`
  - la table `PHASE_SHORT` qui mappe les valeurs brutes de la colonne `phase`
    des CSV `lemmes_*.csv` ('P1_Artisanal' …) vers leur étiquette courte
    ('P1' …), utilisée par tous les scripts qui agrègent par phase.

Union extraite des 5 anciens scripts `36*.py` avant refacto d'avril 2026.
"""

# ---------------------------------------------------------------------------
# Catégories sémantiques
# ---------------------------------------------------------------------------

FINANCE = {
    "банк", "гривня", "грн", "донат", "дякувати", "задонатити", "збір",
    "млн", "моно", "моноbank", "отримувач", "переказ", "приватбанк",
    "рахунок", "реквізит", "реквізити", "тисяча",
}

MILITAIRE = {
    "бахмут", "бригада", "вйо", "ворог", "доба", "дрон", "жало", "жах",
    "засіб", "знищити", "зсу", "йобликів", "кожний", "кринки", "макітра",
    "морський", "наступний", "нічний", "обубас", "окремий", "пм", "птах",
    "підрозділ", "пілот", "рер", "сбс", "спокійний", "тихий", "фпв",
    "хробак", "хробачий", "хробачу", "ціль",
}

# Tout le reste = ASSOCIÉS (catégorie par défaut). Inclut мадяр, відео,
# качати, тіктоці, roberto, bovdi, etc.


def categorize(lemma: str) -> str:
    """Retourne 'finance', 'militaire' ou 'associes'."""
    if lemma in FINANCE:
        return "finance"
    if lemma in MILITAIRE:
        return "militaire"
    return "associes"


# Mots outils à exclure du parlé (pronoms, particules, verbes génériques).
# Partagé par les scripts 36* qui dressent les tableaux/barres TF-IDF du
# dialogue : sans ce filtre, le haut de classement est saturé de mots vides.
STOPWORDS_DIALOGUE = {
    "котрий", "свій", "от", "бачити", "працювати", "якийсь",
    "дивитися", "робити", "знати", "казати", "великий",
    "кожний", "наступний", "якось", "щось", "той", "цей",
    "такий", "який", "там", "тут", "так", "але", "вже",
    "ще", "навіть", "просто", "тому", "тоді", "коли",
    "потім", "дуже", "зараз", "один", "два", "три",
}


# ---------------------------------------------------------------------------
# Palette partagée (cohérente entre tous les tableaux)
# ---------------------------------------------------------------------------

COLORS = {
    "finance":   {"text": "#1A6B2E", "bg": "#EAF5EC", "bar": "#2E9E4F"},
    "militaire": {"text": "#8B1A1A", "bg": "#FAE8E8", "bar": "#C0392B"},
    "associes":  {"text": "#888888", "bg": "#F2F2F2", "bar": "#B0B0B0"},
}
HEADER_BG = "#2B2B2B"
HEADER_FG = "#FFFFFF"


# ---------------------------------------------------------------------------
# Étiquettes courtes des phases
# ---------------------------------------------------------------------------
# Format des CSV lemmes_*.csv : 'P1_Artisanal', 'P2_Semi-pro', ...
# Format raccourci utilisé dans les colonnes/légendes des sorties : 'P1', 'P2', 'P3'.

PHASE_SHORT = {
    "P1_Artisanal":     "P1",
    "P2_Semi-pro":      "P2",
    "P3_Institutionnel": "P3",
}


def phase_sans_prefixe(series):
    """
    Retire le 'P' initial des étiquettes de phase ('P1_Artisanal' → '1_Artisanal').

    Entrée : Series pandas dont les valeurs commencent par 'P' + chiffre
             ('P1_Artisanal', 'P2_Semi-pro', 'P3_Institutionnel'…).
    Sortie : Series avec le 'P' initial retiré ('1_Artisanal'…).

    Centralise le `str.replace(r'^P(\\d)', r'\\1')` répété dans les scripts 36*.
    """
    return series.str.replace(r'^P(\d)', r'\1', regex=True)


# ---------------------------------------------------------------------------
# Traductions uk → fr
# ---------------------------------------------------------------------------

TRADUCTIONS = {
    "бахмут":    "Bakhmout",
    "бачити":    "voir",
    "бригада":   "brigade",
    "бровді":    "Brovdi",
    "великий":   "grand",
    "виявити":   "détecter",
    "вйо":       "allez !",
    "ворог":     "ennemi",
    "відео":     "vidéo",
    "військовий": "militaire (adj.)",
    "грн":       "hryvnia",
    "дивитися":  "regarder",
    "доба":      "24 h",
    "дрон":      "drone",
    "дякувати":  "remercier",
    "жало":      "dard / FPV",
    "жах":       "horreur",
    "засіб":     "moyen / engin",
    "збір":      "collecte",
    "знати":     "savoir",
    "знищити":   "détruire",
    "зсу":       "ZSU",
    "йобликів":  "argot FPV",
    "казати":    "dire",
    "качати":    "télécharger",
    "кожний":    "chaque",
    "котрий":    "lequel/qui",
    "кринки":    "Krynky",
    "кількість": "quantité",
    "людина":    "personne",
    "мадяр":     "Magyar",
    "макітра":   "tête (argot)",
    "млн":       "million",
    "моно":      "Monobank",
    "морський":  "maritime",
    "місяць":    "mois",
    "наступний": "suivant",
    "нічний":    "nocturne",
    "обубас":    "UBAS",
    "око":       "œil",
    "окремий":   "séparé / distinct",
    "от":        "donc (part.)",
    "отримувач": "destinataire",
    "пм":        "Ptakhyky Madyara",
    "працювати": "travailler",
    "приватбанк": "PrivatBank",
    "протягом":  "pendant",
    "птах":      "oiseau / drone",
    "підрозділ": "unité",
    "пілот":     "pilote",
    "реквізит":  "coord. bancaires",
    "рер":       "guerre élec.",
    "роберт":    "Robert",
    "робити":    "faire",
    "рік":       "année",
    "сбс":       "sigle",
    "свій":      "son propre",
    "спокійний": "calme",
    "тисяча":    "mille",
    "тихий":     "silencieux",
    "тож":       "donc/alors",
    "тіктоці":   "TikTok",
    "україна":   "Ukraine",
    "фпв":       "FPV",
    "хробак":    "ver / FPV",
    "хробачий":  "ver (adj.)",
    "хробачу":   "ver (acc.)",
    "ціль":      "cible",
    "якийсь":    "un certain",
}


def translate(lemma: str) -> str:
    """Retourne la traduction fr ou le lemme uk si absent du dictionnaire."""
    return TRADUCTIONS.get(lemma, lemma)
