# Utilisation d'une version de Python légère (slim)
FROM python:3.12-slim

# Empêcher Python de générer des fichiers .pyc et forcer l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système (Libgomp1 est crucial pour LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation de Poetry
RUN pip install poetry

# Copie uniquement des fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# Installation des dépendances de PROD uniquement (--only main)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

# Copie du code source et du modèle
COPY src/ ./src/
COPY model/ ./model/

# Port par défaut pour Hugging Face
EXPOSE 7860

# Commande de lancement (0.0.0.0 est obligatoire pour être accessible de l'extérieur)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
