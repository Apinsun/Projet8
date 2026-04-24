import requests
import time
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# L'URL du nouvel endpoint Batch
API_URL_BATCH = "https://apinsun-projet8.hf.space/predict_batch"
#API_URL_BATCH = "http://0.0.0.0:7860/predict_batch" # Décommenter pour tester en local avec Docker

def send_data(drift_mode=False, n_requests=50):
    print(f"🚀 Mode {'DRIFT ⚠️' if drift_mode else 'NORMAL ✅'} | Préparation d'un batch de {n_requests} clients...")

    try:
        # 1. CHARGEMENT DU DATASET RÉEL
        BASE_DIR = Path(__file__).resolve().parent.parent
        data_path = BASE_DIR / "data" / "dataset_raw.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("❌ Erreur : Fichier 'dataset_raw.csv' introuvable.")
        return

    # 2. SÉLECTION ALÉATOIRE
    sample_df = df.sample(n=n_requests, random_state=42)
    sample_df = sample_df.replace({np.nan: None})

    # 3. PRÉPARATION DU PAYLOAD GÉANT
    payload = []
    client_ids = []

    for row in sample_df.to_dict(orient='records'):
        # --- LOGIQUE DE SIMULATION DE DRIFT ---
        if drift_mode:
            if row.get("AMT_CREDIT") is not None: row["AMT_CREDIT"] *= 2.5
            if row.get("EXT_SOURCE_1") is not None: row["EXT_SOURCE_1"] *= 0.5
            if row.get("EXT_SOURCE_2") is not None: row["EXT_SOURCE_2"] *= 0.5

        row["is_test"] = True
        
        # Sauvegarde de l'ID et nettoyage
        client_ids.append(row.get('SK_ID_CURR', 'Inconnu'))
        row.pop('SK_ID_CURR', None)
        row.pop('TARGET', None)

        # On ajoute le client à notre liste
        payload.append(row)

    # 4. ENVOI UNIQUE
    print("🌐 Envoi du batch à l'API en une seule requête réseau...")
    start_time = time.perf_counter()
    
    try:
        res = requests.post(API_URL_BATCH, json=payload)
        end_time = time.perf_counter()
        
        if res.status_code == 200:
            data = res.json()
            print(f"✅ Succès ! {data['batch_size']} prédictions traitées.")
            print(f"⏱️  Temps d'exécution serveur (API + BDD) : {data['execution_time_total_ms']} ms")
            print(f"⏱️  Temps total perçu par ce script (Réseau compris) : {round(end_time - start_time, 2)} secondes")
        else:
            print(f"❌ Erreur {res.status_code} : {res.text}")
    except Exception as e:
        print(f"Erreur de connexion : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peupler la BDD depuis un CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--drift", "-d", action="store_true", help="Induire un drift volontaire")
    group.add_argument("--no-drift", "-nd", action="store_true", help="Données normales")
    
    parser.add_argument("--count", "-c", type=int, default=50, help="Nombre de lignes à envoyer")

    args = parser.parse_args()
    send_data(drift_mode=args.drift, n_requests=args.count)
