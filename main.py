from src.data_loader import charger_donnees
from src.data_cleaning import nettoyer_donnees
from src.hypothesis_tests import (
    test_vegetarian_calories,
    test_cuisine_calories,
    test_correlation_temps_calories
)
from src.recommender import (
    creer_recommandations_similarite,
    recommander_recettes,
    recommander_par_criteres
)
from src.statistics import statistiques_generales

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    print("PROJET: Analyse de Recettes du Monde Entier")

    df = charger_donnees("Receipes.csv")
    df_clean = nettoyer_donnees(df)

    df_clean.to_csv("recipes_cleaned.csv", index=False)
    print("\n Données nettoyées sauvegardées: recipes_cleaned.csv")

    statistiques_generales(df_clean)

    print("ÉTAPE 3: ANALYSES STATISTIQUES")
    r1 = test_vegetarian_calories(df_clean)
    r2 = test_cuisine_calories(df_clean)
    r3 = test_correlation_temps_calories(df_clean)

    similarity_matrix = creer_recommandations_similarite(df_clean)

    recommander_recettes(df_clean, similarity_matrix, recipe_idx=0, n=5)

    recommander_par_criteres(
        df_clean,
        cuisine="Italian",
        max_calories=400,
        max_time=60,
        n=5
    )

    recommander_par_criteres(
        df_clean,
        vegetarian=True,
        max_calories=500,
        max_time=45,
        n=5
    )

    print("\nRésumé des tests statistiques:")
    print(f" Test 1 - p-value = {r1['p_value']:.4f}")
    if r2:
        print(f" Test 2 - p-value = {r2['p_value']:.4f}")
    print(f" Test 3 - r = {r3['r']:.4f}, p-value = {r3['p_value']:.4f}")


if __name__ == "__main__":
    main()
