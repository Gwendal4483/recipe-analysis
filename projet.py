"""
Projet d'Analyse de Recettes et Recommandation Nutritionnelle
Dataset: Recipes from Around the World (Kaggle)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

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
    
    # Afficher un aperçu
    print("Aperçu des données:")
    print(df.head(3))
    
    return df


# ============================================================================
# ÉTAPE 2 : NETTOYAGE DES DONNÉES
# ============================================================================

def nettoyer_donnees(df):
    """
    Nettoie et prépare les données
    """
    
    # Copie pour ne pas modifier l'original
    df_clean = df.copy()
    
    # 1. Gérer les valeurs manquantes
    print("\n1. Valeurs manquantes:")
    print(df_clean.isnull().sum())
    
    # Remplacer 'nan' en string par NaN
    df_clean['dietary_restrictions'] = df_clean['dietary_restrictions'].replace("['nan']", np.nan)
    
    # 2. Créer des catégories binaires pour les restrictions alimentaires
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
    
    # 3. Calculer le temps total
    df_clean['total_time_minutes'] = df_clean['cooking_time_minutes'] + df_clean['prep_time_minutes']
    
    # 4. Calculer les calories totales
    df_clean['total_calories'] = df_clean['calories_per_serving'] * df_clean['servings']
    
    # 5. Statistiques de nettoyage
    print(f"\n Données nettoyées: {len(df_clean)} recettes")
    print(f" Recettes végétariennes: {df_clean['is_vegetarian'].sum()}")
    print(f" Recettes vegan: {df_clean['is_vegan'].sum()}")
    print(f" Recettes sans gluten: {df_clean['is_gluten_free'].sum()}")
    print(f" Nouvelles colonnes créées: total_time_minutes, total_calories")
    
    return df_clean

# ============================================================================
# ÉTAPE 3 : TESTS D'HYPOTHÈSES STATISTIQUES
# ============================================================================

def test_vegetarian_calories(df):
    """
    TEST 1: Les plats végétariens ont-ils moins de calories ?
    H0: calories_vegetarian >= calories_non_vegetarian
    H1: calories_vegetarian < calories_non_vegetarian
    """

    print(" TEST 1: Les plats végétariens ont-ils moins de calories ?")

    
    # Séparer les données
    veg = df[df['is_vegetarian'] == True]['calories_per_serving'].dropna()
    non_veg = df[df['is_vegetarian'] == False]['calories_per_serving'].dropna()
    
    # Statistiques descriptives
    print(f"\nVégétarien:")
    print(f"  n = {len(veg)}")
    print(f"  Moyenne = {veg.mean():.2f} calories/portion")
    print(f"  Écart-type = {veg.std():.2f}")
    print(f"  Min = {veg.min():.0f}, Max = {veg.max():.0f}")
    
    print(f"\nNon-végétarien:")
    print(f"  n = {len(non_veg)}")
    print(f"  Moyenne = {non_veg.mean():.2f} calories/portion")
    print(f"  Écart-type = {non_veg.std():.2f}")
    print(f"  Min = {non_veg.min():.0f}, Max = {non_veg.max():.0f}")
    
    # Test t de Student (unilatéral)
    t_stat, p_value = ttest_ind(veg, non_veg, alternative='less')
    
    print(f"\nTest t de Student (unilatéral):")
    print(f"  Statistique t = {t_stat:.4f}")
    print(f"  P-value = {p_value:.4f}")
    print(f"  Seuil α = 0.05")
    
    # Conclusion
    print(f"\n Conclusion:")
    if p_value < 0.05:
        print(f"   REJET de H0 (p={p_value:.4f} < 0.05)")
        print(f"   Les plats végétariens ont SIGNIFICATIVEMENT moins de calories")
        print(f"   Différence moyenne: {non_veg.mean() - veg.mean():.1f} calories/portion")
    else:
        print(f"   NON-REJET de H0 (p={p_value:.4f} >= 0.05)")
        print(f"   Pas de différence significative entre les deux groupes")
    
    # Visualisation
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Boxplot
    plt.subplot(1, 2, 1)
    plt.boxplot([veg, non_veg], labels=['Végétarien', 'Non-végétarien'])
    plt.ylabel('Calories par portion')
    plt.title('Distribution des calories par type')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Violinplot
    plt.subplot(1, 2, 2)
    data_plot = pd.DataFrame({
        'Calories': list(veg) + list(non_veg),
        'Type': ['Végétarien']*len(veg) + ['Non-végétarien']*len(non_veg)
    })
    sns.violinplot(data=data_plot, x='Type', y='Calories')
    plt.title('Distribution détaillée des calories')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test1_vegetarian_calories.png', dpi=300, bbox_inches='tight')
    print("\n Graphique sauvegardé: test1_vegetarian_calories.png")
    
    return {'p_value': p_value, 'veg_mean': veg.mean(), 'non_veg_mean': non_veg.mean()}


def test_cuisine_calories(df):
    """
    TEST 2: Y a-t-il des différences de calories entre les cuisines ?
    H0: Toutes les cuisines ont la même moyenne de calories
    H1: Au moins une cuisine diffère
    """

    print("TEST 2: Différences caloriques entre cuisines (ANOVA)")
    
    # Obtenir les cuisines avec au moins 3 recettes
    cuisine_counts = df['cuisine'].value_counts()
    cuisines_valides = cuisine_counts[cuisine_counts >= 3].index.tolist()
    
    print(f"\nCuisines analysées (≥3 recettes): {len(cuisines_valides)}")
    
    # Créer les groupes
    groupes = []
    noms_cuisines = []
    
    for cuisine in cuisines_valides[:10]:  # Top 10 cuisines
        calories = df[df['cuisine'] == cuisine]['calories_per_serving'].dropna()
        if len(calories) >= 3:
            groupes.append(calories)
            noms_cuisines.append(cuisine)
            print(f"  {cuisine}: n={len(calories)}, moyenne={calories.mean():.1f}")
    
    # Test ANOVA
    if len(groupes) >= 2:
        f_stat, p_value = f_oneway(*groupes)
        
        print(f"\n Test ANOVA:")
        print(f"  F-statistique = {f_stat:.4f}")
        print(f"  P-value = {p_value:.4f}")
        print(f"  Seuil α = 0.05")
        
        print(f"\n Conclusion:")
        if p_value < 0.05:
            print(f"  REJET de H0 (p={p_value:.4f} < 0.05)")
            print(f"  Il existe des différences significatives entre les cuisines")
        else:
            print(f"  NON-REJET de H0 (p={p_value:.4f} >= 0.05)")
            print(f"  Pas de différence significative entre les cuisines")
        
        # Visualisation
        plt.figure(figsize=(14, 6))
        
        # Préparer les données pour le plot
        plot_data = []
        for cuisine in noms_cuisines:
            calories = df[df['cuisine'] == cuisine]['calories_per_serving'].dropna()
            for cal in calories:
                plot_data.append({'Cuisine': cuisine, 'Calories': cal})
        
        plot_df = pd.DataFrame(plot_data)
        
        # Boxplot
        plt.subplot(1, 2, 1)
        sns.boxplot(data=plot_df, x='Cuisine', y='Calories')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution des calories par cuisine')
        plt.ylabel('Calories par portion')
        plt.grid(True, alpha=0.3)
        
        # Barplot des moyennes
        plt.subplot(1, 2, 2)
        moyennes = df.groupby('cuisine')['calories_per_serving'].mean().sort_values(ascending=False).head(10)
        moyennes.plot(kind='barh')
        plt.xlabel('Calories moyennes par portion')
        plt.title('Top 10 cuisines par calories moyennes')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test2_cuisine_calories.png', dpi=300, bbox_inches='tight')
        print("\n Graphique sauvegardé: test2_cuisine_calories.png")
        
        return {'p_value': p_value, 'f_statistic': f_stat}
    else:
        print("\n Pas assez de groupes pour ANOVA")
        return None


def test_correlation_temps_calories(df):
    """
    TEST 3: Corrélation entre temps de préparation et calories
    H0: ρ = 0 (pas de corrélation)
    H1: ρ ≠ 0 (corrélation existe)
    """

    print(" TEST 3: Corrélation temps total vs calories")
    
    # Préparer les données
    data = df[['total_time_minutes', 'calories_per_serving']].dropna()
    
    # Test de corrélation de Pearson
    r, p_value = pearsonr(data['total_time_minutes'], data['calories_per_serving'])
    
    print(f"\nNombre d'observations: {len(data)}")
    print(f"\n Test de corrélation de Pearson:")
    print(f"  Coefficient r = {r:.4f}")
    print(f"  P-value = {p_value:.4f}")
    print(f"  Seuil α = 0.05")
    
    # Interprétation de la force
    abs_r = abs(r)
    if abs_r < 0.3:
        force = "faible"
    elif abs_r < 0.7:
        force = "modérée"
    else:
        force = "forte"
    
    print(f"\n Conclusion:")
    if p_value < 0.05:
        print(f"   REJET de H0 (p={p_value:.4f} < 0.05)")
        print(f"   Corrélation {force} et significative (r={r:.3f})")
        if r > 0:
            print(f"   Plus le temps de préparation augmente, plus les calories augmentent")
        else:
            print(f"   Plus le temps de préparation augmente, moins il y a de calories")
    else:
        print(f"   NON-REJET de H0 (p={p_value:.4f} >= 0.05)")
        print(f"   Pas de corrélation significative")
    
    # Visualisation
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Scatter plot avec tendance
    plt.subplot(1, 2, 1)
    plt.scatter(data['total_time_minutes'], data['calories_per_serving'], alpha=0.6, s=50)
    
    # Ligne de tendance
    z = np.polyfit(data['total_time_minutes'], data['calories_per_serving'], 1)
    p = np.poly1d(z)
    plt.plot(data['total_time_minutes'], p(data['total_time_minutes']), 
             "r--", alpha=0.8, linewidth=2, label=f'Tendance (r={r:.3f})')
    
    plt.xlabel('Temps total (minutes)')
    plt.ylabel('Calories par portion')
    plt.title(f'Corrélation: Temps vs Calories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribution jointe
    plt.subplot(1, 2, 2)
    plt.hexbin(data['total_time_minutes'], data['calories_per_serving'], 
               gridsize=20, cmap='YlOrRd')
    plt.colorbar(label='Nombre de recettes')
    plt.xlabel('Temps total (minutes)')
    plt.ylabel('Calories par portion')
    plt.title('Densité: Temps vs Calories')
    
    plt.tight_layout()
    plt.savefig('test3_correlation_temps_calories.png', dpi=300, bbox_inches='tight')
    print("\n Graphique sauvegardé: test3_correlation_temps_calories.png")
    
    return {'r': r, 'p_value': p_value}

# ============================================================================
# ÉTAPE 4 : SYSTÈME DE RECOMMANDATION
# ============================================================================

def creer_recommandations_similarite(df):
    """
    Système de recommandation basé sur la similarité cosine
    """

    print("ÉTAPE 4: SYSTÈME DE RECOMMANDATION")

    
    print("\n1. Préparation des features pour la similarité...")
    
    # Features numériques
    features_cols = ['calories_per_serving', 'cooking_time_minutes', 
                     'prep_time_minutes', 'servings']
    
    # Normaliser
    scaler = StandardScaler()
    features = scaler.fit_transform(df[features_cols].fillna(df[features_cols].mean()))
    
    # Calculer la similarité
    similarity_matrix = cosine_similarity(features)
    
    print(f"  Matrice de similarité calculée: {similarity_matrix.shape}")
    
    return similarity_matrix


def recommander_recettes(df, similarity_matrix, recipe_idx, n=5):
    """
    Recommande des recettes similaires basées sur la similarité cosine
    """

    print(" RECOMMANDATION PAR SIMILARITÉ")

    
    # Obtenir les similarités
    similarities = similarity_matrix[recipe_idx]
    
    # Top N recettes similaires (exclure la recette elle-même)
    similar_indices = similarities.argsort()[::-1][1:n+1]
    
    # Recette de référence
    ref = df.iloc[recipe_idx]
    print(f"\n Recette de référence:")
    print(f"  Nom: {ref['recipe_name']}")
    print(f"  Cuisine: {ref['cuisine']}")
    print(f"  Calories: {ref['calories_per_serving']:.0f}/portion")
    print(f"  Temps total: {ref['total_time_minutes']:.0f} min")
    print(f"  Portions: {ref['servings']:.0f}")
    
    # Recommandations
    print(f"\n Top {n} recettes similaires:")
    for i, idx in enumerate(similar_indices, 1):
        row = df.iloc[idx]
        print(f"\n{i}. {row['recipe_name']}")
        print(f"   Cuisine: {row['cuisine']}")
        print(f"   Calories: {row['calories_per_serving']:.0f}/portion")
        print(f"   Temps: {row['total_time_minutes']:.0f} min")
        print(f"   Similarité: {similarities[idx]:.3f}")


def recommander_par_criteres(df, cuisine=None, max_calories=None, 
                             max_time=None, vegetarian=None, n=5):
    """
    Recommande des recettes selon des critères spécifiques
    """

    print(" RECOMMANDATION PAR CRITÈRES")
    
    # Commencer avec toutes les recettes
    filtered = df.copy()
    
    # Appliquer les filtres
    print("\nCritères appliqués:")
    
    if cuisine:
        filtered = filtered[filtered['cuisine'] == cuisine]
        print(f"   Cuisine: {cuisine}")
    
    if max_calories:
        filtered = filtered[filtered['calories_per_serving'] <= max_calories]
        print(f"   Calories max: {max_calories}/portion")
    
    if max_time:
        filtered = filtered[filtered['total_time_minutes'] <= max_time]
        print(f"   Temps max: {max_time} minutes")
    
    if vegetarian is not None:
        filtered = filtered[filtered['is_vegetarian'] == vegetarian]
        print(f"   Végétarien: {'Oui' if vegetarian else 'Non'}")
    
    # Trier par calories (du moins au plus calorique)
    filtered = filtered.sort_values('calories_per_serving')
    
    print(f"\n Top {min(len(filtered), n)} recettes trouvées (sur {len(filtered)} correspondantes):")
    
    for i, (idx, row) in enumerate(filtered.head(n).iterrows(), 1):
        print(f"\n{i}. {row['recipe_name']}")
        print(f"   Cuisine: {row['cuisine']}")
        print(f"   Calories: {row['calories_per_serving']:.0f}/portion")
        print(f"   Temps total: {row['total_time_minutes']:.0f} min")
        print(f"   Portions: {row['servings']:.0f}")
        restrictions = row['dietary_restrictions']
        if pd.notna(restrictions) and restrictions != "['nan']":
            print(f"   Restrictions: {restrictions}")


def statistiques_generales(df):
    """
    Affiche des statistiques générales sur le dataset
    """

    print("STATISTIQUES GÉNÉRALES")

    
    print(f"\nNombre total de recettes: {len(df)}")
    print(f"\nNombre de cuisines différentes: {df['cuisine'].nunique()}")
    
    print(f"\nStatistiques caloriques:")
    print(f"  Moyenne: {df['calories_per_serving'].mean():.1f} calories/portion")
    print(f"  Médiane: {df['calories_per_serving'].median():.1f} calories/portion")
    print(f"  Min: {df['calories_per_serving'].min():.0f}")
    print(f"  Max: {df['calories_per_serving'].max():.0f}")
    
    print(f"\n⏱Statistiques de temps:")
    print(f"  Temps moyen total: {df['total_time_minutes'].mean():.1f} minutes")
    print(f"  Temps médian: {df['total_time_minutes'].median():.1f} minutes")
    
    print(f"\nRestrictions alimentaires:")
    print(f"  Végétarien: {df['is_vegetarian'].sum()} recettes ({df['is_vegetarian'].sum()/len(df)*100:.1f}%)")
    print(f"  Vegan: {df['is_vegan'].sum()} recettes ({df['is_vegan'].sum()/len(df)*100:.1f}%)")
    print(f"  Sans gluten: {df['is_gluten_free'].sum()} recettes ({df['is_gluten_free'].sum()/len(df)*100:.1f}%)")
    print(f"  Sans produits laitiers: {df['is_dairy_free'].sum()} recettes ({df['is_dairy_free'].sum()/len(df)*100:.1f}%)")
    
    print(f"\nTop 5 cuisines:")
    for i, (cuisine, count) in enumerate(df['cuisine'].value_counts().head(5).items(), 1):
        print(f"  {i}. {cuisine}: {count} recettes")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Pipeline complet du projet
    """

    print("PROJET: Analyse de Recettes du Monde Entier")

    
    # ÉTAPE 1: Chargement
    df = charger_donnees('Receipes.csv')
    
    # ÉTAPE 2: Nettoyage
    df_clean = nettoyer_donnees(df)
    
    # Sauvegarder les données nettoyées
    df_clean.to_csv('recipes_cleaned.csv', index=False)
    print(f"\n  Données nettoyées sauvegardées: recipes_cleaned.csv")
    
    # Statistiques générales
    statistiques_generales(df_clean)
    
    # ÉTAPE 3: Tests statistiques
    print("ÉTAPE 3: ANALYSES STATISTIQUES")
    
    result1 = test_vegetarian_calories(df_clean)
    result2 = test_cuisine_calories(df_clean)
    result3 = test_correlation_temps_calories(df_clean)
    
    # ÉTAPE 4: Recommandation
    similarity_matrix = creer_recommandations_similarite(df_clean)
    
    # Exemple 1: Recommandation par similarité
    recommander_recettes(df_clean, similarity_matrix, recipe_idx=0, n=5)
    
    # Exemple 2: Recommandation par critères (plats italiens, légers, rapides)
    recommander_par_criteres(
        df_clean, 
        cuisine='Italian',
        max_calories=400,
        max_time=60,
        n=5
    )
    
    # Exemple 3: Recommandation végétarienne
    recommander_par_criteres(
        df_clean,
        vegetarian=True,
        max_calories=500,
        max_time=45,
        n=5
    )
    

    print("\nFichiers générés:")
    print("    recipes_cleaned.csv")
    print("    test1_vegetarian_calories.png")
    print("    test2_cuisine_calories.png")
    print("    test3_correlation_temps_calories.png")
    
    print("\nRésumé des tests statistiques:")
    print(f"  Test 1 - Végétarien vs Calories: p-value = {result1['p_value']:.4f}")
    if result2:
        print(f"Test 2 - ANOVA Cuisines: p-value = {result2['p_value']:.4f}")
    print(f"  Test 3 - Corrélation Temps/Calories: r = {result3['r']:.4f}, p-value = {result3['p_value']:.4f}")


if __name__ == "__main__":
    main()
