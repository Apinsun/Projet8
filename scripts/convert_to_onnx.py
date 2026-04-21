import joblib
import sys
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
import lightgbm as lgb
from skl2onnx import update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

# 1. Configuration LightGBM pour ONNX
update_registered_converter(
    lgb.LGBMClassifier, 'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes, convert_lightgbm,
    # --- ON CORRIGE LA LIGNE CI-DESSOUS ---
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']} 
)

# 2. Gestion des alias
try:
    import src.model_wrapper as model_wrapper
    import src.custom_transformer as custom_transformer
    sys.modules['model_wrapper'] = model_wrapper
    sys.modules['custom_transformer'] = custom_transformer
except ImportError as e:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "model" / "model.pkl"
data_path = BASE_DIR / "data" / "dataset_raw.csv"

# 3. Chargement
print("📦 Chargement du pipeline complet...")
wrapper = joblib.load(model_path)
full_pipeline = wrapper.pipeline 

# 4. Le Grand Split !
custom_fe_step = full_pipeline.steps[0][1]
from sklearn.pipeline import Pipeline
onnx_pipeline = Pipeline(full_pipeline.steps[1:])

print("✂️ Pipeline séparé avec succès.")

# --- 🩹 LA RUSTINE MAGIQUE POUR ONNX ---
from sklearn.impute import SimpleImputer

def patch_categorical_imputers(estimator):
    """Parcourt le pipeline et modifie les imputers de texte pour ONNX"""
    if hasattr(estimator, 'steps'):
        for _, step in estimator.steps:
            patch_categorical_imputers(step)
    if hasattr(estimator, 'transformers_'):
        for _, step, _ in estimator.transformers_:
            patch_categorical_imputers(step)
    if isinstance(estimator, SimpleImputer):
        if hasattr(estimator, 'statistics_'):
            # Si l'imputer a appris à boucher les trous avec du texte, c'est un imputer catégoriel
            if len(estimator.statistics_) > 0 and isinstance(estimator.statistics_[0], str):
                estimator.missing_values = '' # Remplacement de np.nan par une chaîne vide

patch_categorical_imputers(onnx_pipeline)
print("🩹 Imputers catégoriels patchés pour ONNX.")
# ---------------------------------------

# 5. Simulation de l'entrée pour comprendre la structure des données
df_test = pd.read_csv(data_path, nrows=1)
columns_to_drop = ['SK_ID_CURR', 'TARGET']
features = df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns])

features_transformed = custom_fe_step.transform(features)

# --- 🧹 NETTOYAGE DES DONNÉES ENTRANTES POUR ONNX ---
# ONNX ne supporte pas d'avaler des NaN dans des colonnes de texte
for col in features_transformed.columns:
    if features_transformed[col].dtype == 'object':
        features_transformed[col] = features_transformed[col].fillna('')
# ----------------------------------------------------

# 6. Cartographie exacte des colonnes pour ONNX
print("📊 Analyse des types de colonnes pour ONNX...")
initial_types = []

for col in features_transformed.columns:
    dtype = str(features_transformed[col].dtype)
    
    # 💡 L'ASTUCE ICI : Entiers ET Décimaux deviennent des Floats pour ONNX
    if 'int' in dtype or 'float' in dtype:
        initial_types.append((col, FloatTensorType([None, 1])))
    else:
        initial_types.append((col, StringTensorType([None, 1])))

# 7. Conversion
print("⚡ Conversion vers ONNX en cours (seulement Preprocessing lourd + LightGBM)...")
onx = convert_sklearn(
    onnx_pipeline, 
    initial_types=initial_types, 
    # --- 🎯 LA CORRECTION EST ICI ---
    target_opset={'': 12, 'ai.onnx.ml': 3}, 
    options={lgb.LGBMClassifier: {'zipmap': False}}
)

# 8. Sauvegarde
onnx_path = BASE_DIR / "model" / "model.onnx"
with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())

print(f"🎉 VICTOIRE ABSOLUE ! Modèle converti et sauvegardé dans {onnx_path}")