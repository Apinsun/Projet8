import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
    """
    Applique le nettoyage et crée les ratios métiers à la volée. (basé sur le scipt du projet6 qui préparait les données pour le modèle)
    """
    def fit(self, X, y=None):
        # On sauvegarde les noms des colonnes du dataset d'entraînement
        # crucial pour l'API qui utilise feature_names_in_ pour réindexer les données entrantes par exemple
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.values
        return self # Pas de calcul global nécessaire lors du fit

    def transform(self, X, y=None):
        X_ = X.copy()
        # Dans ce dataset, la valeur 365243 jours (1000 ans) est une erreur courante pour les chômeurs/retraités.
        # 1. Correction de l'anomalie
        if 'DAYS_EMPLOYED' in X_.columns:
            X_['DAYS_EMPLOYED'] = X_['DAYS_EMPLOYED'].replace({365243: np.nan})
            
        # 2. Ratios métiers (avec vérification de la présence des colonnes)
        if 'AMT_CREDIT' in X_.columns and 'AMT_INCOME_TOTAL' in X_.columns:
            X_['CREDIT_INCOME_PERCENT'] = X_['AMT_CREDIT'] / X_['AMT_INCOME_TOTAL']
            
        if 'AMT_ANNUITY' in X_.columns and 'AMT_INCOME_TOTAL' in X_.columns:
            X_['ANNUITY_INCOME_PERCENT'] = X_['AMT_ANNUITY'] / X_['AMT_INCOME_TOTAL']
            
        if 'AMT_ANNUITY' in X_.columns and 'AMT_CREDIT' in X_.columns:
            X_['CREDIT_TERM'] = X_['AMT_ANNUITY'] / X_['AMT_CREDIT']
            
        if 'DAYS_EMPLOYED' in X_.columns and 'DAYS_BIRTH' in X_.columns:
            X_['DAYS_EMPLOYED_PERCENT'] = X_['DAYS_EMPLOYED'] / X_['DAYS_BIRTH']
            
        return X_