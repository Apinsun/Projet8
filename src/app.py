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

# 3. Extraction des features attendues par le modèle (pour la validation d'entrée)

if pipeline is not None:
    sk_pipe = pipeline.pipeline if hasattr(pipeline, "pipeline") else pipeline
    # On stocke sous forme de Tuple (très rapide à lire)
    EXPECTED_FEATURES = tuple(sk_pipe.feature_names_in_)
    # On stocke sous forme de Set pour la comparaison mathématique ultra-rapide
    EXPECTED_FEATURES_SET = set(EXPECTED_FEATURES)
    # On met le seuil en cache (remplace 0.5 par ton attribut exact si différent)
    THRESHOLD = float(pipeline.threshold) if hasattr(pipeline, "threshold") else 0.5

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
    # ⏱️ 1. DÉMARRAGE DU CHRONOMÈTRE & VÉRIFICATION DES DONNEES D'ENTRÉE
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible.")

    # Utiliser Pydantic pour exclure directement les champs inutiles (Plus rapide que .pop())
    client_dict = client.model_dump(exclude={"is_test", "SK_ID_CURR"})

    # Vérification des colonnes inconnues avec notre Set en cache
    colonnes_inconnues = set(client_dict.keys()) - EXPECTED_FEATURES_SET
    
    if colonnes_inconnues:
        raise HTTPException(
            status_code=400, 
            detail=f"Champs non reconnus par le modèle : {', '.join(colonnes_inconnues)}"
        )
    
    # 🧹 LA TRADUCTION DES VIDES
    # On remplace les None (Python) par des np.nan (Pandas/Maths)
    client_dict = {k: (np.nan if v is None else v) for k, v in client_dict.items()}

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
            #On récupère à la fois la classe et la proba
            prediction, proba = pipeline.predict_classe_and_proba(df_client)
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
                "seuil_utilise": THRESHOLD,
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
    Endpoint ultra-optimisé pour traiter plusieurs clients d'un coup.
    """
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible.")

    # 0. Vérification rapide des colonnes (basée sur le 1er client, Pydantic garantit l'homogénéité)
    if clients:
        client_dict_test = clients[0].model_dump(exclude={"is_test", "SK_ID_CURR"})
        colonnes_inconnues = set(client_dict_test.keys()) - EXPECTED_FEATURES_SET
        if colonnes_inconnues:
            raise HTTPException(
                status_code=400, 
                detail=f"Champs non reconnus par le modèle : {', '.join(colonnes_inconnues)}"
            )

    # 1. Préparation des données en masse
    # Extraction des flags is_test en une ligne
    is_test_flags = [c.is_test for c in clients]
    
    # Création de la liste de dictionnaires, exclusion des champs inutiles et traitement des None -> NaN en une passe
    client_dicts = [
        {k: (np.nan if v is None else v) for k, v in c.model_dump(exclude={"is_test", "SK_ID_CURR"}).items()}
        for c in clients
    ]

    # Création du DataFrame et Reindex global d'un seul coup 
    df_clients = pd.DataFrame(client_dicts, columns=EXPECTED_FEATURES)

    try:
        # Inférence en batch
        predictions, probas = pipeline.predict_classe_and_proba(df_clients)

        end_time = time.perf_counter()
        execution_time_ms = round((end_time - start_time) * 1000, 2)
        avg_time_per_request = execution_time_ms / len(clients) if clients else 0

        # 3. Préparation des logs et de la réponse
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
                "seuil_utilise": THRESHOLD,
                "message": "Le client présente un risque élevé." if decision == "Refusé" else "Dossier solide."
            })

            # Le log pour Supabase
            if supabase_client:
                original_dict = clients[i].model_dump(exclude={"is_test"})
                logs_to_insert.append({
                    "client_features": original_dict,
                    "score_defaut": proba,
                    "decision": decision,
                    "execution_time_ms": avg_time_per_request,
                    "is_test": is_test_flags[i]
                })

        # 4. On insère dans la BDD également par batch
        if supabase_client and logs_to_insert:
            background_tasks.add_task(save_to_supabase_background, logs_to_insert)

        return {
            "execution_time_total_ms": execution_time_ms,
            "batch_size": len(clients),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction batch : {str(e)}")