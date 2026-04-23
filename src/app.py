import os
import sys
from typing import Literal, Optional,List
import time
import json
import logging
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib
from supabase import create_client, Client
#permet de bien trouver model_wrapper.py même si on lance l'API depuis un autre dossier
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_wrapper import ModelWrapper
from schema import ClientData

# 1. Initialisation de l'API
app = FastAPI(
    title="API Prêt à Dépenser",
    description="API de scoring crédit utilisant un modèle LightGBM complet",
    version="1.0.0"
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Configuration d'un logger basique
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)

# On ne crée le client que si les clés sont présentes
supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Chargement de notre piepeline (Modèle + Preprocessing)
try:
    pipeline = joblib.load("model/model.pkl")
    print("✅ Modèle chargé avec succès !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    pipeline = None

def save_to_supabase_background(data_to_insert):
    """Fonction qui tournera en arrière-plan pour ne pas bloquer l'API."""
    if supabase_client:
        try:
            # Cette fonction accepte aussi bien un dictionnaire (predict) qu'une liste de dictionnaires (predict_batch)
            supabase_client.table("predictions_logs").insert(data_to_insert).execute()
            print("✅ Logs insérés dans Supabase avec succès (en arrière-plan).")
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde Supabase (background) : {e}")

# 4. Le Endpoint de prédiction pour un client unique
@app.post("/predict")
async def predict_score(client: ClientData, background_tasks: BackgroundTasks):
    # ⏱️ 1. DÉMARRAGE DU CHRONOMÈTRE
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible.")

    # 1. Convertir les données reçues en dictionnaire puis en DataFrame (9 colonnes obligatoires)
    client_dict = client.model_dump()
    # On retire "is_test" du dictionnaire (le modèle ML n'a pas besoin de le voir)
    client_dict.pop("is_test", None)
    client_dict.pop("SK_ID_CURR", None)
    # 🧹 LA TRADUCTION DES VIDES
    # On remplace les None (Python) par des np.nan (Pandas/Maths)
    for key, value in client_dict.items():
        if value is None:
            client_dict[key] = np.nan

    df_client = pd.DataFrame([client_dict])

    # 2. On accède au vrai pipeline Scikit-Learn (qui est caché dans notre ModelWrapper)
    sk_pipe = pipeline.pipeline if hasattr(pipeline, "pipeline") else pipeline
    
    # 3. On récupère la liste exacte des 120+ colonnes attendues par le modèle
    expected_features = sk_pipe.feature_names_in_
    
    # --- Si on nous envoie des colonnes ne faisant référence à rien on renvoie une erreur plutôt que de ne rien dire ---
    # On pourrait juste les enlever et faire tourner le modèle mais je trouve que c'est plus judicieux d'avertir l'utilisateur
    colonnes_envoyees = set(client_dict.keys())
    colonnes_attendues = set(expected_features)
    
    colonnes_inconnues = colonnes_envoyees - colonnes_attendues
    
    if colonnes_inconnues:
        # On arrête tout et on prévient l'utilisateur des champs exacts qui posent problème
        raise HTTPException(
            status_code=400, 
            detail=f"Champs non reconnus par le modèle : {', '.join(colonnes_inconnues)}"
        )
    # ----------------------------------

    # 4. On redimensionne le DataFrame. 
    # Pandas va garder nos colonnes et créer les autres en les remplissant de NaN
    df_client = df_client.reindex(columns=expected_features)
    # ------------------------------------

    try:
            # 1. On récupère la probabilité de défaut (entre 0 et 1) grâce à la méthode predict_proba de notre ModelWrapper
            proba = float(pipeline.predict_proba(df_client)[0])
            
            # 2. On récupère la décision avec notre seuil optimisé (renvoie [1] ou [0])
            prediction = int(pipeline.predict_class(df_client)[0])

            # 3. La logique métier
            decision = "Refusé" if prediction == 1 else "Accordé"

            # ⏱️ 2. ARRÊT DU CHRONOMÈTRE
            end_time = time.perf_counter()
            execution_time_ms = round((end_time - start_time) * 1000, 2) # Conversion en millisecondes

            # 📝 3. LE LOGGING STRUCTURÉ (JSON)
            log_entry = {
                "event": "api_prediction",
                "decision": decision,
                "score": float(proba),
                "execution_time_ms": execution_time_ms,
                "status": "success"
            }
            # On imprime le dictionnaire sous forme de chaîne JSON stricte
            print(json.dumps(log_entry))

            data_to_log = {
                "client_features": client.model_dump(exclude={"is_test"}),
                "score_defaut": float(proba),
                "decision": decision,
                "execution_time_ms": execution_time_ms,
                "is_test": client.is_test
            }
            background_tasks.add_task(save_to_supabase_background, data_to_log)

            

            return {
                "score_defaut": round(proba, 4),
                "decision": decision,
                "seuil_utilise": float(pipeline.threshold) if hasattr(pipeline, "threshold") else 0.5,
                "message": "Le client présente un risque élevé." if decision == "Refusé" else "Dossier solide.",
                "execution_time_ms": execution_time_ms
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")



# Petite route de test pour vérifier que l'API est vivante
@app.get("/")
def read_root():
    return {"status": "API en ligne", "message": "Bienvenue sur l'API Prêt à Dépenser -> testez https://apinsun-projet8.hf.space/docs pour la doc interactive !"}

@app.post("/predict_batch")
async def predict_batch(clients: List[ClientData], background_tasks: BackgroundTasks):
    """
    Endpoint optimisé pour traiter plusieurs clients d'un coup.
    """
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible.")

    # 1. Préparation des données en masse
    client_dicts = []
    is_test_flags = []
    
    for c in clients:
        c_dict = c.model_dump()
        # On sauvegarde le flag is_test de chaque client pour Supabase
        is_test_flags.append(c_dict.pop("is_test", False))
        c_dict.pop("SK_ID_CURR", None)
        
        # 🧹 LA TRADUCTION DES VIDES
        for key, value in c_dict.items():
            if value is None:
                c_dict[key] = np.nan
        client_dicts.append(c_dict)

    # Création du DataFrame d'un seul coup
    df_clients = pd.DataFrame(client_dicts)

    # 2. Vérification des colonnes
    sk_pipe = pipeline.pipeline if hasattr(pipeline, "pipeline") else pipeline
    expected_features = sk_pipe.feature_names_in_
    
    colonnes_envoyees = set(df_clients.columns)
    colonnes_attendues = set(expected_features)
    colonnes_inconnues = colonnes_envoyees - colonnes_attendues
    
    if colonnes_inconnues:
        raise HTTPException(
            status_code=400, 
            detail=f"Champs non reconnus par le modèle : {', '.join(colonnes_inconnues)}"
        )

    # On aligne les colonnes
    df_clients = df_clients.reindex(columns=expected_features)

    try:
        # 3. Prédiction vectorisée (ultra rapide)
        probas = pipeline.predict_proba(df_clients)
        predictions = pipeline.predict_class(df_clients)

        end_time = time.perf_counter()
        execution_time_ms = round((end_time - start_time) * 1000, 2)

        # 4. Préparation des logs et de la réponse
        results = []
        logs_to_insert = []
        
        for i in range(len(clients)):
            proba = float(probas[i])
            pred = int(predictions[i])
            decision = "Refusé" if pred == 1 else "Accordé"
            
            # La réponse pour l'utilisateur
            results.append({
                "score_defaut": round(proba, 4),
                "decision": decision,
                "seuil_utilise": float(pipeline.threshold) if hasattr(pipeline, "threshold") else 0.5,
                "message": "Le client présente un risque élevé." if decision == "Refusé" else "Dossier solide."
            })

            # Le log pour Supabase
            if supabase_client:
                original_dict = clients[i].model_dump(exclude={"is_test"})
                logs_to_insert.append({
                    "client_features": original_dict,
                    "score_defaut": proba,
                    "decision": decision,
                    "execution_time_ms": execution_time_ms / len(clients), # Temps moyen estimé par requête
                    "is_test": is_test_flags[i]
                })

        # 5. Bulk Insert Supabase (1 seul appel réseau vers la BDD !)
            background_tasks.add_task(save_to_supabase_background, logs_to_insert)

        return {
            "execution_time_total_ms": execution_time_ms,
            "batch_size": len(clients),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction batch : {str(e)}")