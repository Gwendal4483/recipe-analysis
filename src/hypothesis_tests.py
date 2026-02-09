import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, f_oneway, pearsonr

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
        print(
            f"   Différence moyenne: "
            f"{non_veg.mean() - veg.mean():.1f} calories/portion"
        )
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
        'Type': ['Végétarien'] * len(veg) + ['Non-végétarien'] * len(non_veg)
    })
    sns.violinplot(data=data_plot, x='Type', y='Calories')
    plt.title('Distribution détaillée des calories')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        'test1_vegetarian_calories.png',
        dpi=300,
        bbox_inches='tight'
    )
    print("\n Graphique sauvegardé: test1_vegetarian_calories.png")

    return {
        'p_value': p_value,
        'veg_mean': veg.mean(),
        'non_veg_mean': non_veg.mean()
    }


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

    groupes = []
    noms_cuisines = []

    for cuisine in cuisines_valides[:10]:
        calories = df[df['cuisine'] == cuisine]['calories_per_serving'].dropna()
        if len(calories) >= 3:
            groupes.append(calories)
            noms_cuisines.append(cuisine)
            print(
                f"  {cuisine}: n={len(calories)}, "
                f"moyenne={calories.mean():.1f}"
            )

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

        plot_data = []
        for cuisine in noms_cuisines:
            calories = df[df['cuisine'] == cuisine]['calories_per_serving'].dropna()
            for cal in calories:
                plot_data.append({'Cuisine': cuisine, 'Calories': cal})

        plot_df = pd.DataFrame(plot_data)

        plt.subplot(1, 2, 1)
        sns.boxplot(data=plot_df, x='Cuisine', y='Calories')
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution des calories par cuisine')
        plt.ylabel('Calories par portion')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        (
            df.groupby('cuisine')['calories_per_serving']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .plot(kind='barh')
        )
        plt.xlabel('Calories moyennes par portion')
        plt.title('Top 10 cuisines par calories moyennes')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            'test2_cuisine_calories.png',
            dpi=300,
            bbox_inches='tight'
        )
        print("\n Graphique sauvegardé: test2_cuisine_calories.png")

        return {'p_value': p_value, 'f_statistic': f_stat}

    else:
        print("\n Pas assez de groupes pour ANOVA")
        return None


def test_correlation_temps_calories(df):
    """
    TEST 3: Corrélation entre temps de préparation et calories
    H0: ρ = 0
    H1: ρ ≠ 0
    """

    print(" TEST 3: Corrélation temps total vs calories")

    data = df[['total_time_minutes', 'calories_per_serving']].dropna()

    r, p_value = pearsonr(
        data['total_time_minutes'],
        data['calories_per_serving']
    )

    print(f"\nNombre d'observations: {len(data)}")

    print(f"\n Test de corrélation de Pearson:")
    print(f"  Coefficient r = {r:.4f}")
    print(f"  P-value = {p_value:.4f}")
    print(f"  Seuil α = 0.05")

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

    plt.subplot(1, 2, 1)
    plt.scatter(
        data['total_time_minutes'],
        data['calories_per_serving'],
        alpha=0.6,
        s=50
    )

    z = np.polyfit(
        data['total_time_minutes'],
        data['calories_per_serving'],
        1
    )
    p = np.poly1d(z)

    plt.plot(
        data['total_time_minutes'],
        p(data['total_time_minutes']),
        "r--",
        alpha=0.8,
        linewidth=2,
        label=f'Tendance (r={r:.3f})'
    )

    plt.xlabel('Temps total (minutes)')
    plt.ylabel('Calories par portion')
    plt.title('Corrélation: Temps vs Calories')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hexbin(
        data['total_time_minutes'],
        data['calories_per_serving'],
        gridsize=20,
        cmap='YlOrRd'
    )
    plt.colorbar(label='Nombre de recettes')
    plt.xlabel('Temps total (minutes)')
    plt.ylabel('Calories par portion')
    plt.title('Densité: Temps vs Calories')

    plt.tight_layout()
    plt.savefig(
        'test3_correlation_temps_calories.png',
        dpi=300,
        bbox_inches='tight'
    )
    print("\n Graphique sauvegardé: test3_correlation_temps_calories.png")

    return {'r': r, 'p_value': p_value}
