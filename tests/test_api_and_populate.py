import requests
import random
import time

# L'URL de ton API locale (ou celle de Hugging Face si tu veux tester la prod)
API_URL = "https://apinsun-projet8.hf.space/predict"

print("🚀 Démarrage de la simulation d'appels API...")

# On va générer 50 requêtes pour peupler la base
for i in range(50):
    # Génération de données aléatoires (mais réalistes)
    payload = {
        "EXT_SOURCE_1": round(random.uniform(0.1, 0.9), 3),
        "EXT_SOURCE_2": round(random.uniform(0.1, 0.9), 3),
        "EXT_SOURCE_3": round(random.uniform(0.1, 0.9), 3),
        "AMT_CREDIT": random.randint(50000, 500000),
        "AMT_ANNUITY": random.randint(5000, 50000),
        "AMT_GOODS_PRICE": random.randint(50000, 500000),
        "DAYS_BIRTH": random.randint(-25000, -8000),
        "CODE_GENDER": random.choice(["M", "F"]),
        "NAME_EDUCATION_TYPE": random.choice(["Higher education", "Secondary / secondary special"]),
        "is_test": True  # <--- LE FAMEUX FLAG POUR PROTÉGER LA PROD
    }

    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"[{i+1}/50] ✅ Succès | Score: {result['score_defaut']} | Temps: {result['execution_time_ms']}ms")
        else:
            print(f"[{i+1}/50] ❌ Erreur {response.status_code}: {response.text}")
            
    except Exception as e:
         print(f"[{i+1}/50] ❌ Erreur de connexion : {e}")

    # Petite pause pour ne pas surcharger le serveur
    time.sleep(0.1)

print("🎉 Simulation terminée !")