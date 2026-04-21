import sys
import time
import joblib
import pandas as pd
import numpy as np
import onnxruntime as rt
from pathlib import Path

# --- 1. CONFIGURATION DES ALIAS ---
try:
    import src.model_wrapper as model_wrapper
    import src.custom_transformer as custom_transformer
    sys.modules['model_wrapper'] = model_wrapper
    sys.modules['custom_transformer'] = custom_transformer
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data" / "dataset_raw.csv"
model_pkl_path = BASE_DIR / "model" / "model.pkl"
model_onnx_path = BASE_DIR / "model" / "model.onnx"

print("📦 Chargement des données et des modèles...")

# --- 2. CHARGEMENT DES MODÈLES ---
# Modèle Classique
wrapper = joblib.load(model_pkl_path)
classic_model = wrapper

# Modèle ONNX
sess = rt.InferenceSession(str(model_onnx_path))
# On récupère le Feature Engineer Python (la première étape du pipeline classique)
custom_fe_step = wrapper.pipeline.steps[0][1] 

# --- 3. PRÉPARATION DE LA DONNÉE ---
df_test = pd.read_csv(data_path, nrows=1)
columns_to_drop = ['SK_ID_CURR', 'TARGET']
features = df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns])

def predict_onnx(df):
    """Version optimisée (vectorisée) sans boucle for lente"""
    df_fe = custom_fe_step.transform(df)
    
    # Optimisation : On traite toutes les colonnes d'un coup en Pandas
    num_cols = df_fe.select_dtypes(include=np.number).columns
    cat_cols = df_fe.select_dtypes(exclude=np.number).columns
    
    df_fe[num_cols] = df_fe[num_cols].astype(np.float32)
    df_fe[cat_cols] = df_fe[cat_cols].fillna('').astype(str)
    
    # Création ultra-rapide du dictionnaire
    onnx_inputs = {c: df_fe[c].values.reshape(-1, 1) for c in df_fe.columns}
        
    return sess.run(None, onnx_inputs)[1]

# --- LE GRAND MATCH ---
N_RUNS = 1000
print(f"\n🏁 TEST 1 : Une requête à la fois ({N_RUNS} itérations)...")

start_time = time.perf_counter()
for _ in range(N_RUNS):
    classic_model.predict_proba(features)
time_classic_single = time.perf_counter() - start_time
print(f"🐌 Scikit-Learn : {time_classic_single:.4f}s")

start_time = time.perf_counter()
for _ in range(N_RUNS):
    predict_onnx(features)
time_onnx_single = time.perf_counter() - start_time
print(f"🚀 ONNX (Optimisé) : {time_onnx_single:.4f}s")

print(f"\n🏁 TEST 2 : Traitement par lot (Batch de {N_RUNS} clients d'un coup)...")
# On duplique la ligne 1000 fois pour créer un gros dataset
features_batch = pd.concat([features]*N_RUNS, ignore_index=True)

start_time = time.perf_counter()
classic_model.predict_proba(features_batch)
time_classic_batch = time.perf_counter() - start_time
print(f"🐌 Scikit-Learn (Batch) : {time_classic_batch:.4f}s")

start_time = time.perf_counter()
predict_onnx(features_batch)
time_onnx_batch = time.perf_counter() - start_time
print(f"🚀 ONNX (Batch) : {time_onnx_batch:.4f}s")
