import cProfile
import sys
import os
from pathlib import Path
import pstats
import joblib
import pandas as pd
import time

try:
    # On importe depuis le nouveau chemin
    from src.model_wrapper import ModelWrapper
    from src.custom_transformer import FeatureEngineerTransformer
    # On crée les alias pour que Joblib ne soit pas perdu
    import src.model_wrapper as model_wrapper
    import src.custom_transformer as custom_transformer
    
    sys.modules['model_wrapper'] = model_wrapper
    sys.modules['custom_transformer'] = custom_transformer
    
    print("✅ Alias de modules configurés.")
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)
BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "dataset_raw.csv"
model_path = BASE_DIR / "model" / "model.pkl"

print("📦 Chargement du modèle...")

model = joblib.load(model_path)

print("📊 Chargement d'une ligne de test...")
# On charge juste la première ligne de du dataset pour simuler un client
df_test = pd.read_csv(data_path, nrows=1)

# On retire la target et l'ID comme on le fait dans l'API
columns_to_drop = ['SK_ID_CURR', 'TARGET']
features = df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns])

def simulate_production_traffic():
    """Simule 1000 prédictions pour avoir des statistiques mesurables"""
    for _ in range(1000):
        model.predict_proba(features)

print("⏱️ Lancement de cProfile (Simulation de 1000 requêtes)...")

# 1. On démarre le profiler
profiler = cProfile.Profile()
profiler.enable()

# 2. On lance notre fonction
simulate_production_traffic()

# 3. On arrête le profiler
profiler.disable()

# 4. On affiche les résultats formatés
print("\n=== RÉSULTATS DU PROFILING ===")
stats = pstats.Stats(profiler)
# On trie par 'cumtime' (temps cumulé passé dans la fonction)
stats.sort_stats('cumtime')
# On affiche le Top 15 des fonctions les plus lentes
stats.print_stats(15)
