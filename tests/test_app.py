from fastapi.testclient import TestClient
from src.app import app  # Ajuste le chemin si besoin selon l'arborescence

client = TestClient(app)

# ==========================================
# LE CLIENT PARFAIT (Pour passer la douane Pydantic)
# ==========================================
VALID_PAYLOAD = {
    # Flag test
    "is_test": True,
    
    # Catégorielles obligatoires
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "F",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "WEEKDAY_APPR_PROCESS_START": "WEDNESDAY",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    
    # Numériques obligatoires
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 150000.0,
    "AMT_CREDIT": 500000.0,
    "AMT_ANNUITY": 25000.0,
    "REGION_POPULATION_RELATIVE": 0.02,
    "DAYS_BIRTH": -15000,
    "DAYS_EMPLOYED": -2000,
    "DAYS_REGISTRATION": -5000.0,
    "DAYS_ID_PUBLISH": -2500,
    "DAYS_LAST_PHONE_CHANGE": -1000.0,
    "FLAG_MOBIL": 1,
    "FLAG_EMP_PHONE": 1,
    "FLAG_WORK_PHONE": 0,
    "FLAG_CONT_MOBILE": 1,
    "FLAG_PHONE": 0,
    "FLAG_EMAIL": 0,
    "CNT_FAM_MEMBERS": 2.0,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "HOUR_APPR_PROCESS_START": 10,
    "REG_REGION_NOT_LIVE_REGION": 0,
    "REG_REGION_NOT_WORK_REGION": 0,
    "LIVE_REGION_NOT_WORK_REGION": 0,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "REG_CITY_NOT_WORK_CITY": 0,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "FLAG_DOCUMENT_2": 0, "FLAG_DOCUMENT_3": 1, "FLAG_DOCUMENT_4": 0,
    "FLAG_DOCUMENT_5": 0, "FLAG_DOCUMENT_6": 0, "FLAG_DOCUMENT_7": 0,
    "FLAG_DOCUMENT_8": 0, "FLAG_DOCUMENT_9": 0, "FLAG_DOCUMENT_10": 0,
    "FLAG_DOCUMENT_11": 0, "FLAG_DOCUMENT_12": 0, "FLAG_DOCUMENT_13": 0,
    "FLAG_DOCUMENT_14": 0, "FLAG_DOCUMENT_15": 0, "FLAG_DOCUMENT_16": 0,
    "FLAG_DOCUMENT_17": 0, "FLAG_DOCUMENT_18": 0, "FLAG_DOCUMENT_19": 0,
    "FLAG_DOCUMENT_20": 0, "FLAG_DOCUMENT_21": 0,
    
    # Optionnelles (mais on les met pour tester un flux classique complet)
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5
}


def test_racine_api_fonctionne():
    """Test vérifiant que l'API est bien en ligne sur la route /"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction_rejetee_si_credit_negatif():
    """Vérifie que Pydantic bloque un crédit négatif (Erreur 422)"""
    client_data = VALID_PAYLOAD.copy()
    client_data["AMT_CREDIT"] = -100 
    
    response = client.post("/predict", json=client_data)
    assert response.status_code == 422
    assert "AMT_CREDIT" in response.text

def test_prediction_rejetee_si_colonne_inconnue():
        """Vérifie que l'API bloque les colonnes fantômes (Erreur 422)"""
        client_data = VALID_PAYLOAD.copy()
        client_data["CHIEN"] = "Médor"
        
        response = client.post("/predict", json=client_data)
        
        assert response.status_code == 422
        # On vérifie le message d'erreur standard de Pydantic
        assert "Extra inputs are not permitted" in response.text 
        # (Optionnel) On peut même vérifier qu'il cite bien la colonne fautive !
        assert "CHIEN" in response.text

def test_prediction_acceptee_payload_complet():
    """Vérifie que l'API accepte les données parfaites"""
    client_data = VALID_PAYLOAD.copy()
    
    response = client.post("/predict", json=client_data)
    assert response.status_code == 200

    data = response.json()
    print(response.json())
    assert isinstance(data["score_defaut"], float)
    assert data["decision"] in ["Accordé", "Refusé"]
    assert 0 <= data["score_defaut"] <= 1

def test_prediction_acceptee_si_ext_source1_manquant():
    """Vérifie que l'API accepte les données même s'il manque une variable Optionnelle"""
    client_data = VALID_PAYLOAD.copy()
    
    # On retire EXT_SOURCE_1 (Pydantic doit l'accepter car on l'a mis en Optional dans schemas.py)
    if "EXT_SOURCE_1" in client_data:
        del client_data["EXT_SOURCE_1"]
        
    response = client.post("/predict", json=client_data)
    print(response.json())
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["score_defaut"], float)

def test_prediction_batch_acceptee():
    """Vérifie que la route /predict_batch accepte bien une liste de clients et renvoie le bon format"""
    # 1. On crée deux clients à partir de notre modèle parfait
    client_1 = VALID_PAYLOAD.copy()
    
    client_2 = VALID_PAYLOAD.copy()
    # On change juste une valeur pour faire genre c'est un client différent
    client_2["AMT_CREDIT"] = 100000.0 

    # 2. On prépare la liste (le format attendu par la route batch)
    payload_batch = [client_1, client_2]

    # 3. On envoie à l'API
    response = client.post("/predict_batch", json=payload_batch)
    
    # 4. Assertions : Vérification du statut HTTP
    assert response.status_code == 200, f"Erreur de l'API: {response.text}"

    # 5. Assertions : Vérification du contenu de la réponse
    data = response.json()
    assert "batch_size" in data
    assert data["batch_size"] == 2
    
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    # Vérification de la structure de la première prédiction renvoyée
    premiere_pred = data["predictions"][0]
    assert "score_defaut" in premiere_pred
    assert "decision" in premiere_pred
    assert premiere_pred["decision"] in ["Accordé", "Refusé"]
