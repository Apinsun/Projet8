import os
import sys
from typing import Literal, Optional
import time
import json
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import joblib
from supabase import create_client, Client
#permet de bien trouver model_wrapper.py même si on lance l'API depuis un autre dossier
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_wrapper import ModelWrapper

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

# 3. Le Pydantic Model pour valider les données d'entrée
# Basée sur les features les plus importantes identifiées par SHAP, mais avec une tolérance pour les autres (extra='allow')
class ClientData(BaseModel):
    # --- 1. Les Scores Externes (Généralement entre 0 et 1) ---

    #Score optionnel car il y a beaucoup de valeurs manquantes dans notre bdd de base, surtout pour 1 (56%) et 3 (19%)
    EXT_SOURCE_1: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score externe 1 (Optionnel)")
    EXT_SOURCE_2: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score externe 2 (Optionnel)")
    EXT_SOURCE_3: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score externe 3 (Optionnel)")

# --- 2. Les Montants Financiers (Strictement obligatoires) ---
    # On met ge=0 car on ne peut pas avoir de crédit ou de prix négatif
    AMT_CREDIT: float = Field(..., ge=0, description="Montant du crédit demandé")
    AMT_ANNUITY: float = Field(..., ge=0, description="Montant des annuités")
    AMT_GOODS_PRICE: float = Field(..., ge=0, description="Prix du bien financé")

    # --- 3. Les Informations Démographiques ---
    # ge=-30000 (environ 82 ans) et le=-7000 (environ 19 ans) pour rester cohérent
    DAYS_BIRTH: int = Field(..., ge=-30000, le=-7000, description="Âge en jours (négatif)")
    
    # Utilisation de Literal pour restreindre aux seules valeurs connues par ton modèle
    CODE_GENDER: Literal['M', 'F', 'XNA'] = Field(
        ..., description="Genre du client"
    )
    NAME_EDUCATION_TYPE: Literal[
        'Secondary / secondary special', 
        'Higher education', 
        'Incomplete higher', 
        'Lower secondary', 
        'Academic degree'
    ] = Field(..., description="Niveau d'éducation")

    # flag pour différencier les données de test
    is_test: bool = False

    # --- 4. Tolérance pour les autres variables (pour le Feature Engineering) ---
    model_config = ConfigDict(extra='allow')

# 4. Le Endpoint (La route)
@app.post("/predict")
async def predict_score(client: ClientData):
    # ⏱️ 1. DÉMARRAGE DU CHRONOMÈTRE
    start_time = time.perf_counter()
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Le modèle n'est pas disponible.")

    # 1. Convertir les données reçues en dictionnaire puis en DataFrame (9 colonnes obligatoires)
    client_dict = client.model_dump()
    # On retire "is_test" du dictionnaire (le modèle ML n'a pas besoin de le voir)
    client_dict.pop("is_test", None)

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

            if supabase_client:
                try:
                    # On logge les données en asynchrone (ou dans un bloc try pour ne pas bloquer l'API si la DB plante)
                    data_to_log = {
                        "client_features": client_dict, # Convertit l'objet Pydantic en dictionnaire
                        "score_defaut": float(proba),
                        "decision": decision,
                        "execution_time_ms": execution_time_ms,
                        "is_test": client.is_test
                    }
                    supabase_client.table("predictions_logs").insert(data_to_log).execute()
                except Exception as e:
                    print(f"Erreur lors de la sauvegarde Supabase : {e}")

            

            return {
                "score_defaut": round(proba, 4),
                "decision": decision,
                "seuil_utilise": float(pipeline.threshold) if hasattr(pipeline, "threshold") else 0.5,
                "message": "Le client présente un risque élevé." if decision == "Refusé" else "Dossier solide."
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")


# Petite route de test pour vérifier que l'API est vivante
@app.get("/")
def read_root():
    return {"status": "API en ligne", "message": "Bienvenue sur l'API Prêt à Dépenser -> testez https://apinsun-projet8.hf.space/docs pour la doc interactive !"}