import pandas as pd

# ============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ============================================================================

def charger_donnees(filepath):
    """
    Charge le dataset de recettes depuis Kaggle
    Dataset: Recipes from Around the World
    """
    print("Chargement des données...")
    df = pd.read_csv(filepath, encoding="latin-1")
    print(f"{len(df)} recettes chargées")

    print("Aperçu des données:")
    print(df.head(3))

    return df
