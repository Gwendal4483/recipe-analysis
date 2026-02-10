import pandas as pd
import numpy as np

# ============================================================================
# ÉTAPE 2 : NETTOYAGE DES DONNÉES
# ============================================================================

def nettoyer_donnees(df):
    """
    Nettoie et prépare les données
    """

    df_clean = df.copy()

    print("\n1. Valeurs manquantes:")
    print(df_clean.isnull().sum())

    df_clean['dietary_restrictions'] = df_clean['dietary_restrictions'].replace(
        "['nan']", np.nan
    )

    print("\n valeur aberrantes")
    #on a des recettes avec des temps de préparation ou de cuisson très élevés, on va les filtrer 
    df_clean = df_clean[
        (df_clean['cooking_time_minutes'] <= 300) &
        (df_clean['prep_time_minutes'] <= 120)
    ]
    

    print("\n2. Création des catégories alimentaires...")

    df_clean['is_vegetarian'] = df_clean['dietary_restrictions'].apply(
        lambda x: 'vegetarian' in str(x).lower() if pd.notna(x) else False
    )

    df_clean['is_vegan'] = df_clean['dietary_restrictions'].apply(
        lambda x: 'vegan' in str(x).lower() if pd.notna(x) else False
    )

    df_clean['is_gluten_free'] = df_clean['dietary_restrictions'].apply(
        lambda x: 'gluten-free' in str(x).lower() if pd.notna(x) else False
    )

    df_clean['is_dairy_free'] = df_clean['dietary_restrictions'].apply(
        lambda x: 'dairy-free' in str(x).lower() if pd.notna(x) else False
    )

    df_clean['total_time_minutes'] = (
        df_clean['cooking_time_minutes'] + df_clean['prep_time_minutes']
    )

    df_clean['total_calories'] = (
        df_clean['calories_per_serving'] * df_clean['servings']
    )

    print(f"\n Données nettoyées: {len(df_clean)} recettes")
    print(f" Recettes végétariennes: {df_clean['is_vegetarian'].sum()}")
    print(f" Recettes vegan: {df_clean['is_vegan'].sum()}")
    print(f" Recettes sans gluten: {df_clean['is_gluten_free'].sum()}")
    print(f" Nouvelles colonnes créées: total_time_minutes, total_calories")

    return df_clean
