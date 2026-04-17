import pandas as pd
import json
import os

def profile_dataset(csv_path="data/dataset_raw.csv", output_file="feature_rules.json"):
    print(f"🔍 Analyse du fichier : {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Fichier introuvable à l'emplacement : {csv_path}")
        return

    # On ignore les colonnes qui ne sont pas des features pour le modèle
    cols_to_ignore = ['SK_ID_CURR', 'TARGET']
    features = [col for col in df.columns if col not in cols_to_ignore]
    
    report = {}
    
    for col in features:
        col_type = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        missing_pct = round((missing_count / len(df)) * 100, 2)
        
        col_info = {
            "type_pandas": col_type,
            "valeurs_manquantes_pct": missing_pct,
            "est_optionnel": missing_pct > 0 # Si > 0, on devra utiliser Optional dans Pydantic
        }
        
        # --- Variables Catégorielles (Texte / Objets) ---
        if col_type == 'object':
            # On récupère les valeurs uniques (en ignorant les NaN)
            unique_vals = df[col].dropna().unique().tolist()
            col_info["type_pydantic"] = "str"
            col_info["valeurs_autorisees"] = unique_vals
            
        # --- Variables Numériques (Int / Float) ---
        else:
            col_info["type_pydantic"] = "int" if "int" in col_type else "float"
            col_info["min"] = float(df[col].min())
            col_info["max"] = float(df[col].max())
            
            # Déduction des règles métier
            if col_info["min"] >= 0:
                col_info["regle_metier"] = "Strictement positif ou nul (ge=0)"
            elif col_info["max"] <= 0:
                col_info["regle_metier"] = "Strictement négatif ou nul (le=0)"
            else:
                col_info["regle_metier"] = "Peut être positif ou négatif"

        report[col] = col_info

    # Sauvegarde du rapport dans un fichier JSON pour une lecture facile
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Analyse terminée ! Rapport sauvegardé dans '{output_file}'")
    print(f"📊 Colonnes analysées : {len(features)}")

if __name__ == "__main__":
    profile_dataset()
