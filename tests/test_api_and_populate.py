import requests
import time
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# L'URL de Hugging Face
API_URL = "https://apinsun-projet8.hf.space/predict"

def send_data(drift_mode=False, n_requests=50):
    print(f"🚀 Mode {'DRIFT ⚠️' if drift_mode else 'NORMAL ✅'} | Envoi de {n_requests} requêtes...")

    try:
        # 1. CHARGEMENT DU DATASET RÉEL
            BASE_DIR = Path(__file__).resolve().parent.parent
            data_path = BASE_DIR / "data" / "dataset_raw.csv"
            print(f"Chargement des données depuis {data_path}...")
            df = pd.read_csv(data_path)
            print(f"📊 Dataset chargé : {len(df)} lignes disponibles.")
            print(f"Colonnes trouvées : {list(df.columns)[:5]}")
    except FileNotFoundError:
        print("❌ Erreur : Fichier 'datatset_raw.csv' introuvable.")
        return

    # 2. SÉLECTION ALÉATOIRE DE N CLIENTS
    # On prend un échantillon au hasard pour simuler de vrais flux
    sample_df = df.sample(n=n_requests, random_state=42) # random_state pour la reproductibilité (optionnel)

    # Nettoyage rapide si besoin (ex: remplacer les NaN par 0 ou les ignorer)
    # Dans la réalité, Pydantic n'aime pas les NaN, on les gère ici
    #sample_df = sample_df.fillna(0) # ou une autre méthode d'imputation selon ton modèle
    sample_df = sample_df.replace({np.nan: None})

    for i, row in enumerate(sample_df.to_dict(orient='records')):
        
        # --- LOGIQUE DE SIMULATION DE DRIFT SUR LES VRAIES DONNÉES ---
        if drift_mode:
            # On vérifie que la valeur n'est pas None avant de faire le calcul
            if row.get("AMT_CREDIT") is not None:
                row["AMT_CREDIT"] *= 2.5
            
            if row.get("EXT_SOURCE_1") is not None:
                row["EXT_SOURCE_1"] *= 0.5
                
            if row.get("EXT_SOURCE_2") is not None:
                row["EXT_SOURCE_2"] *= 0.5
            

        # On ajoute notre fameux flag pour protéger la prod
        row["is_test"] = True

        # 🧹 NETTOYAGE AVANT ENVOI
        # On sauvegarde l'ID dans une variable pour pouvoir l'afficher plus tard
        client_id = row.get('SK_ID_CURR', 'Inconnu')
        row.pop('SK_ID_CURR', None)
        row.pop('TARGET', None)

        try:
            res = requests.post(API_URL, json=row)
            if res.status_code == 200:
                print(f"[{i+1}/{n_requests}] Envoyé | Client ID: {client_id} | Score: {res.json().get('score_defaut')}")
            else:
                 print(f"[{i+1}/{n_requests}] ❌ Erreur {res.status_code} : {res.text}")
        except Exception as e:
            print(f"Erreur de connexion : {e}")
        
        time.sleep(0.01) # Petite pause pour le serveur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peupler la BDD depuis un CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--drift", "-d", action="store_true", help="Induire un drift volontaire sur le CSV")
    group.add_argument("--no-drift", "-nd", action="store_true", help="Données normales tirées du CSV")
    
    # On peut même ajouter un argument pour le nombre de requêtes
    parser.add_argument("--count", "-c", type=int, default=50, help="Nombre de lignes à envoyer")

    args = parser.parse_args()
    send_data(drift_mode=args.drift, n_requests=args.count)
