import numpy as np

class ModelWrapper:
    """
    Classe simple pour encapsuler le pipeline et le seuil.
    """
    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, X):
        # Renvoie la probabilité brute (utile pour le monitoring/drift)
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_class(self, X):
        # Renvoie 0 ou 1 selon le seuil
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)