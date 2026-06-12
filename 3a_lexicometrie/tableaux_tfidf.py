"""
Dispatcher des variantes de tableaux/graphiques TF-IDF.

Les cinq scripts 36*.py partagent maintenant le module `categories_semantiques`
(dictionnaires FINANCE / MILITAIRE / TRADUCTIONS + palette COLORS). Ce fichier
regroupe leurs CLI en un seul point d'entrée.

Usage :
    python tableaux_tfidf.py --variant tableau_3phases
    python tableaux_tfidf.py --variant comparatif_p1p2
    python tableaux_tfidf.py --variant slope
    python tableaux_tfidf.py --variant slope_combined
    python tableaux_tfidf.py --variant barchart_p1
    python tableaux_tfidf.py --variant all

Chaque variante lit les CSV TF-IDF produits par `lexicometrie.py` dans
4_data_et_viz/lexico/ et écrit ses figures dans 4_data_et_viz/.
"""

import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = {
    "tableau_3phases":   HERE / "36_tfidf_tableau.py",
    "comparatif_p1p2":   HERE / "36b_tfidf_comparatif_p1p2.py",
    "slope":             HERE / "36c_slope_tfidf_p1p2.py",
    "slope_combined":    HERE / "36d_slope_combined_h.py",
    "barchart_p1":       HERE / "36e_tfidf_barchart_p1.py",
}


def executer_variante(name: str) -> None:
    script = VARIANTS[name]
    if not script.exists():
        raise FileNotFoundError(f"Script introuvable : {script}")
    print(f"\n=== Variant : {name} ({script.name}) ===")
    runpy.run_path(str(script), run_name="__main__")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Génère une ou toutes les variantes de tableaux/graphiques TF-IDF."
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
        help="Variante à générer (défaut : all).",
    )
    args = parser.parse_args()

    targets = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    for name in targets:
        executer_variante(name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
