import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ============================================================================
# ÉTAPE 4 : SYSTÈME DE RECOMMANDATION
# ============================================================================

def creer_recommandations_similarite(df):
    """
    Création d'un système de recommandation basé sur la similarité cosinus
    """

    print("\nÉTAPE 4: SYSTÈME DE RECOMMANDATION PAR SIMILARITÉ")
    print("Création de la matrice de similarité...")

    features = [
        'calories_per_serving',
        'total_time_minutes',
        'servings',
        'is_vegetarian',
        'is_vegan',
        'is_gluten_free',
        'is_dairy_free'
    ]

    data = df[features].copy()
    data = data.fillna(0)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    similarity_matrix = cosine_similarity(data_scaled)

    print("Matrice de similarité créée")
    print(f"Dimensions: {similarity_matrix.shape}")

    return similarity_matrix


def recommander_recettes(df, similarity_matrix, recipe_idx, n=5):
    """
    Recommande n recettes similaires à une recette donnée
    """

    print(f"\nRECOMMANDATIONS POUR LA RECETTE {recipe_idx}")
    print("=" * 50)

    recipe = df.iloc[recipe_idx]

    print(f"Recette de référence:")
    print(f"  Nom: {recipe['recipe_name']}")
    print(f"  Cuisine: {recipe['cuisine']}")
    print(f"  Calories: {recipe['calories_per_serving']:.0f}")
    print(f"  Temps total: {recipe['total_time_minutes']:.0f} minutes")
    print(f"  Végétarien: {'Oui' if recipe['is_vegetarian'] else 'Non'}")

    sim_scores = list(enumerate(similarity_matrix[recipe_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:n+1]

    print(f"\nTop {n} recettes similaires:")

    for i, (idx, score) in enumerate(sim_scores, 1):
        r = df.iloc[idx]
        print(
            f"{i}. {r['recipe_name']} "
            f"(Cuisine: {r['cuisine']}, "
            f"Calories: {r['calories_per_serving']:.0f}, "
            f"Temps: {r['total_time_minutes']:.0f} min, "
            f"Similarité: {score:.3f})"
        )


def recommander_par_criteres(
    df,
    cuisine=None,
    vegetarian=None,
    vegan=None,
    gluten_free=None,
    max_calories=None,
    max_time=None,
    n=5
):
    """
    Recommande des recettes selon des critères spécifiques
    """

    print("\nRECOMMANDATIONS PAR CRITÈRES")
    print("=" * 50)

    filtered_df = df.copy()

    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'] == cuisine]
        print(f"Filtre cuisine: {cuisine}")

    if vegetarian is not None:
        filtered_df = filtered_df[filtered_df['is_vegetarian'] == vegetarian]
        print(f"Filtre végétarien: {'Oui' if vegetarian else 'Non'}")

    if vegan is not None:
        filtered_df = filtered_df[filtered_df['is_vegan'] == vegan]
        print(f"Filtre vegan: {'Oui' if vegan else 'Non'}")

    if gluten_free is not None:
        filtered_df = filtered_df[filtered_df['is_gluten_free'] == gluten_free]
        print(f"Filtre sans gluten: {'Oui' if gluten_free else 'Non'}")

    if max_calories:
        filtered_df = filtered_df[
            filtered_df['calories_per_serving'] <= max_calories
        ]
        print(f"Calories max: {max_calories}")

    if max_time:
        filtered_df = filtered_df[
            filtered_df['total_time_minutes'] <= max_time
        ]
        print(f"Temps max: {max_time} minutes")

    print(f"\nNombre de recettes correspondantes: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("Aucune recette ne correspond aux critères.")
        return

    recommandations = (
        filtered_df
        .sort_values(by='calories_per_serving')
        .head(n)
    )

    print(f"\nTop {len(recommandations)} recommandations:")

    for i, (_, r) in enumerate(recommandations.iterrows(), 1):
        print(
            f"{i}. {r['recipe_name']} "
            f"(Cuisine: {r['cuisine']}, "
            f"Calories: {r['calories_per_serving']:.0f}, "
            f"Temps: {r['total_time_minutes']:.0f} min)"
        )
