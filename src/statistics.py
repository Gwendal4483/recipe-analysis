def statistiques_generales(df):
    print("STATISTIQUES GÉNÉRALES")

    print(f"\nNombre total de recettes: {len(df)}")
    print(f"\nNombre de cuisines différentes: {df['cuisine'].nunique()}")

    print(f"\nStatistiques caloriques:")
    print(f"  Moyenne: {df['calories_per_serving'].mean():.1f}")
    print(f"  Médiane: {df['calories_per_serving'].median():.1f}")
    print(f"  Min: {df['calories_per_serving'].min():.0f}")
    print(f"  Max: {df['calories_per_serving'].max():.0f}")

    print(f"\n⏱Statistiques de temps:")
    print(f"  Temps moyen total: {df['total_time_minutes'].mean():.1f}")
    print(f"  Temps médian: {df['total_time_minutes'].median():.1f}")

    print(f"\nRestrictions alimentaires:")
    print(f"  Végétarien: {df['is_vegetarian'].sum()}")
    print(f"  Vegan: {df['is_vegan'].sum()}")
    print(f"  Sans gluten: {df['is_gluten_free'].sum()}")
    print(f"  Sans produits laitiers: {df['is_dairy_free'].sum()}")

    print(f"\nTop 5 cuisines:")
    for i, (cuisine, count) in enumerate(
        df['cuisine'].value_counts().head(5).items(), 1
    ):
        print(f"  {i}. {cuisine}: {count} recettes")
