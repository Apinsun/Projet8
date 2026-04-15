from pathlib import Path

import pandas as pd
import numpy as np
import argparse
import os

# Script permettant d'analyser les données brutes et de générer des données synthétiques pour tester l'API et le monitoring de drift.
# avec -desc pour décrire les données et aider à configurer Pydantic
# avec -cd pour créer un dataset synthétique perturbé (Data Drift artificiel)

def action_describe(file_path):
    """
    Analyse les données brutes pour aider à configurer Pydantic.
    Focus sur le Top 10 des variables les plus importantes (SHAP).
    """
    print(f"\n📊 Analyse du fichier : {file_path}")
    df = pd.read_csv(file_path)
    
    # Le Top 10 SHAP (brut, avant Feature Engineering)
    features_cles = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
        'DAYS_BIRTH', 'CODE_GENDER', 'NAME_EDUCATION_TYPE'
    ]
    
    print("\n--- STATISTIQUES POUR PYDANTIC ---")
    for col in features_cles:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            dtype = df[col].dtype
            
            print(f"🔸 {col} ({dtype}):")
            
            # Si c'est un nombre (int, float)
            if np.issubdtype(dtype, np.number):
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"   Min: {min_val:.2f} | Max: {max_val:.2f} | Manquants: {nan_pct:.1f}%")
                
                if min_val >= 0:
                    print(f"   👉 Conseil Pydantic: Field(..., ge=0, description='...')")
                else:
                    print(f"   👉 Conseil Pydantic: Field(..., le=0) (valeurs négatives)")
                    
            # Si c'est du texte (catégories)
            else:
                valeurs_uniques = df[col].dropna().unique()
                print(f"   Valeurs uniques: {valeurs_uniques}")
                print(f"   Manquants: {nan_pct:.1f}%")
                print(f"   👉 Conseil Pydantic: str = Field(..., description=\"Valeurs possibles: {valeurs_uniques[:3]}...\")")
                
    print("----------------------------------\n")

def action_create_data(file_path, output_path, n_samples=1000):
    """
    Génère des données synthétiques par 'Perturbation' pour tester l'API et le Drift.
    """
    print(f"\n🧬 Création de {n_samples} clients synthétiques...")
    df = pd.read_csv(file_path)
    
    # 1. On tire au hasard n_samples vrais clients
    df_synthetic = df.sample(n=n_samples, replace=True).copy()
    
    # 2. On applique des mutations (Data Drift artificiel !)
    # Ex: On simule une crise économique où les revenus baissent de 10% à 40%
    if 'AMT_INCOME_TOTAL' in df_synthetic.columns:
        bruit_revenu = np.random.uniform(0.6, 0.9, size=n_samples)
        df_synthetic['AMT_INCOME_TOTAL'] = df_synthetic['AMT_INCOME_TOTAL'] * bruit_revenu
        
    # Ex: On simule des montants de crédits demandés plus élevés (+0% à +50%)
    if 'AMT_CREDIT' in df_synthetic.columns:
        bruit_credit = np.random.uniform(1.0, 1.5, size=n_samples)
        df_synthetic['AMT_CREDIT'] = df_synthetic['AMT_CREDIT'] * bruit_credit

    # 3. On injecte volontairement des valeurs manquantes (NaN) pour tester la robustesse de l'API
    cols_to_corrupt = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'AMT_ANNUITY']
    for col in cols_to_corrupt:
        if col in df_synthetic.columns:
            # On met 15% des valeurs à NaN
            mask = np.random.rand(n_samples) < 0.15
            df_synthetic.loc[mask, col] = np.nan
            
    # 4. On crée de faux ID clients pour les repérer facilement (ex: commençant par 999)
    df_synthetic['SK_ID_CURR'] = range(9990000, 9990000 + n_samples)
    
    # 5. Sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_synthetic.to_csv(output_path, index=False)
    print(f"✅ Données sauvegardées dans : {output_path}")

if __name__ == "__main__":


    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "dataset_raw.csv"
    print(f"Chargement des données depuis {data_path}...")
    df = pd.read_csv(data_path)


    parser = argparse.ArgumentParser(description="Boîte à outils Data pour l'API Prêt à Dépenser")
    
    # L'argument --file est obligatoire pour savoir sur quoi on travaille
    parser.add_argument("--file", type=str, default=data_path, help="Chemin vers le dataset source")
    
    # Les actions possibles
    parser.add_argument("-desc", "--describe", action="store_true", help="Affiche les stats des features clés pour Pydantic")
    parser.add_argument("-cd", "--create_data", action="store_true", help="Génère un dataset synthétique perturbé")
    
    args = parser.parse_args()
    
    if args.describe:
        action_describe(args.file)
        
    if args.create_data:
        output_file = "data/processed/synthetic_api_test.csv"
        action_create_data(args.file, output_file)
        
    if not args.describe and not args.create_data:
        print("⚠️ Aucune action spécifiée. Utilisez --help pour voir les options.")