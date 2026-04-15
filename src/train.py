import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import clone
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
import shap
import joblib

from custom_transformer import FeatureEngineerTransformer
from model_wrapper import ModelWrapper

# Adaptateur spécifique à MLflow
class MLflowAdapter(mlflow.pyfunc.PythonModel):
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def predict(self, context, model_input):
        return self.wrapper.predict_class(model_input)

def calcul_cout_metier(y_true, y_proba, seuil):
    """Calcule le coût financier basé sur un seuil donné."""
    y_pred = (y_proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (10 * fn) + (1 * fp)

def train_and_log_experiment(run_name, pipeline, params, X_train, X_valid, y_train, y_valid):
    """
    Entraîne, optimise le seuil et logge le modèle LightGBM unique.
    """
    with mlflow.start_run(run_name=run_name): 
        
        mlflow.log_params(params)

        # --- CROSS-VALIDATION SUR LE TRAIN SET (OPTIMISÉE) ---
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Évaluation en Cross-Validation (Boucle unique)...")
        
        y_train_proba_oof = np.zeros(len(y_train))
        cv_auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train, y_train)):
            clone_pipe = clone(pipeline)
            X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_fold_val, y_fold_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            clone_pipe.fit(X_fold_train, y_fold_train)
            fold_preds = clone_pipe.predict_proba(X_fold_val)[:, 1]
            y_train_proba_oof[val_idx] = fold_preds
            
            fold_auc = roc_auc_score(y_fold_val, fold_preds)
            cv_auc_scores.append(fold_auc)
            print(f"   - Fold {fold + 1} | AUC : {fold_auc:.4f}")

        cv_auc_scores = np.array(cv_auc_scores)
        mlflow.log_metric("cv_auc_mean", cv_auc_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_auc_scores.std())

        # ---  OPTIMISATION DU SEUIL (SUR LE TRAIN SET NEUTRE) ---
        print("Recherche du seuil optimal métier...")
        seuils = np.linspace(0.1, 0.9, 81)
        couts_train = [calcul_cout_metier(y_train, y_train_proba_oof, s) for s in seuils]
        meilleur_seuil = seuils[np.argmin(couts_train)]

        # --- ENTRAÎNEMENT FINAL ---
        print(f"🚀 Lancement de l'entraînement final sur tout le Train Set...")
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_valid)[:, 1]
        
        cout_valid_reel = calcul_cout_metier(y_valid, y_proba, meilleur_seuil)
        auc_valid_reel = roc_auc_score(y_valid, y_proba)
        
        # Logs des métriques
        mlflow.log_metric("roc_auc", auc_valid_reel)
        mlflow.log_metric("optimal_threshold", meilleur_seuil)
        mlflow.log_metric("min_business_cost", cout_valid_reel)

        # --- GÉNÉRATION DES GRAPHIQUES ---
        fpr, tpr, _ = roc_curve(y_valid, y_proba)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {auc_valid_reel:.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("Courbe ROC")
        ax.legend()
        mlflow.log_figure(fig_roc, "plots/roc_curve.png")
        plt.close(fig_roc)

        couts_valid_pour_graphe = [calcul_cout_metier(y_valid, y_proba, s) for s in seuils]
        fig_cost, ax = plt.subplots(figsize=(10, 6))
        ax.plot(seuils, couts_valid_pour_graphe, color='darkblue', lw=2, label='Coût sur Validation')
        ax.scatter(meilleur_seuil, cout_valid_reel, color='red', s=100, marker='*', label=f'Seuil Appliqué ({meilleur_seuil:.2f})')
        ax.axvline(x=meilleur_seuil, color='red', linestyle='--', alpha=0.5)
        ax.set_title("Optimisation du Coût Métier en fonction du Seuil", fontsize=14)
        ax.set_xlabel("Seuil de décision (Probability Threshold)")
        ax.set_ylabel("Coût total (10*FN + 1*FP)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        mlflow.log_figure(fig_cost, "plots/cost_optimization.png")
        plt.close(fig_cost)

        # --- SAUVEGARDE DU MODÈLE ---
        wrapped_model = ModelWrapper(pipeline=pipeline, threshold=meilleur_seuil)
        # pour l'API
        joblib.dump(wrapped_model, "model/model.pkl")
        # Log dans MLflow (via l'adaptateur)
        mlflow.pyfunc.log_model(
        artifact_path="model", 
        python_model=MLflowAdapter(wrapped_model)
)
        
        print(f"✅ Run terminé : AUC={auc_valid_reel:.4f}, Seuil={meilleur_seuil:.2f}")

        # --- GRAPHIQUES SHAP ---
        try:
            classifier = pipeline.named_steps['classifier']
            
            # ASTUCE : Applique toutes les étapes du pipeline SAUF la dernière (le classifieur)
            # Cela permet d'inclure ton feature engineering ET ton preprocessor
            X_valid_transformed = pipeline[:-1].transform(X_valid)

            explainer = shap.Explainer(classifier)
            X_sample = X_valid_transformed.sample(n=min(1000, len(X_valid_transformed)), random_state=42)
            shap_obj = explainer(X_sample)

            if len(shap_obj.shape) == 3:
                shap_obj = shap_obj[:, :, 1]

            fig_global, ax_global = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_obj, max_display=11, show=False) 
            plt.tight_layout()
            mlflow.log_figure(fig_global, "plots/shap_0_global.png")
            plt.close(fig_global)

            y_proba_sample = pipeline.predict_proba(X_valid.loc[X_sample.index])[:, 1]
            sorted_indices = np.argsort(y_proba_sample)
            lowest_5_idx = sorted_indices[:5]
            highest_5_idx = sorted_indices[-5:]

            for i, idx in enumerate(highest_5_idx):
                fig_local, ax_local = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_obj[idx], max_display=10, show=False)
                plt.tight_layout()
                mlflow.log_figure(fig_local, f"plots/shap_local_highest_risk_{i+1}.png")
                plt.close(fig_local)

            for i, idx in enumerate(lowest_5_idx):
                fig_local, ax_local = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(shap_obj[idx], max_display=10, show=False)
                plt.tight_layout()
                mlflow.log_figure(fig_local, f"plots/shap_local_lowest_risk_{i+1}.png")
                plt.close(fig_local)

            print("✅ Graphiques SHAP loggés avec succès dans MLflow !")
            
        except Exception as e:
            print(f"⚠️ Erreur non-critique lors de la génération SHAP : {e}")

def main():

    BASE_DIR = Path(__file__).resolve().parent.parent
    data_path = BASE_DIR / "data" / "dataset_raw.csv"
    print(f"Chargement des données depuis {data_path}...")
    df = pd.read_csv(data_path)

    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

    print("Séparation en Train / Validation...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y 
    )

    mlflow.set_experiment("Home_Credit_Default_Risk")

    # Paramètres figés issus de notre ancienne optimisation
    lgbm_params = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 34,
        "max_depth": 8,
        "scale_pos_weight": 10, 
        "random_state": 42,
        "verbose": -1
    }

    categorical_features = X_train.select_dtypes(include=['object']).columns

    # 1. Préparation des colonnes dynamiques
    preprocessor = ColumnTransformer(
        transformers=[
            # Pour le texte : on bouche les trous par la valeur la plus fréquente, puis on encode
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ]), make_column_selector(dtype_include=['object', 'category'])),
            
            # Pour les nombres : on bouche les trous par la médiane 
            # Ces deux étapes permettent de gérer les données manquantes et éviter un crash en prod -> penser à définir les features vitales
            ('num', SimpleImputer(strategy='median'), make_column_selector(dtype_exclude=['object', 'category']))
        ],
        remainder='passthrough',
        verbose_feature_names_out=False # Garde les noms de colonnes propres
    ).set_output(transform="pandas")

    # 2. Le Super-Pipeline global
    pipeline_lgbm = Pipeline([
        ('feature_engineering', FeatureEngineerTransformer()), # Étape 1 : Les calculs
        ('preprocessor', preprocessor),                        # Étape 2 : Imputations et Encodage
        ('classifier', LGBMClassifier(**lgbm_params))          # Étape 3 : Le modèle LightGBM
    ])
    
    print("Lancement du modèle LightGBM...")
    train_and_log_experiment(
        run_name="LGBM_Production_Model", 
        pipeline=pipeline_lgbm,
        params={**lgbm_params, "model": "LGBM"},
        X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
    )

if __name__ == "__main__":
    main()
