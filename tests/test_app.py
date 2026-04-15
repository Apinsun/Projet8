from fastapi.testclient import TestClient
from src.app import app  # On importe notre API FastAPI

# On instancie le faux navigateur/client
client = TestClient(app)

def test_racine_api_fonctionne():
    """Test vérifiant que l'API est bien en ligne sur la route /"""
    
    # 1. Arrange : L'URL que l'on veut tester
    url = "/"

    # 2. Act : On simule une requête GET
    response = client.get(url)

    # 3. Assert : On vérifie que le code HTTP est 200 (Succès)
    assert response.status_code == 200
    
    # 3b. Assert : On vérifie que le message de bienvenue est exact
    assert response.json() == {"status": "API en ligne", "message": "Bienvenue sur l'API Prêt à Dépenser -> testez https://apinsun-projet8.hf.space/docs pour la doc interactive !"}

def test_prediction_rejetee_si_credit_negatif():
    """Vérifie que Pydantic bloque un crédit négatif (Erreur 422)"""
    
    # 1. Arrange : On crée un client avec une erreur volontaire (AMT_CREDIT = -100)
    client_data = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": -100,  # <--- L'ERREUR EST ICI
        "AMT_ANNUITY": 5000,
        "AMT_GOODS_PRICE": 50000,
        "DAYS_BIRTH": -15000,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Higher education"
    }

    # 2. Act : On envoie la requête
    response = client.post("/predict", json=client_data)

    # 3. Assert : FastAPI doit renvoyer 422 (Unprocessable Entity) pour erreur de validation
    assert response.status_code == 422
    
    # On vérifie que le message d'erreur mentionne bien le champ en faute
    assert "AMT_CREDIT" in response.text

def test_prediction_rejetee_si_colonne_inconnue():
    """Vérifie que l'API bloque les colonnes fantômes (Erreur 400)"""
    
    # 1. Arrange : On ajoute une colonne "CHIEN" qui n'existe pas dans le modèle
    client_data = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": 100000,
        "AMT_ANNUITY": 5000,
        "AMT_GOODS_PRICE": 50000,
        "DAYS_BIRTH": -15000,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Higher education",
        "CHIEN": "Médor"  # <--- L'INTRUS
    }

    # 2. Act : On envoie la requête
    response = client.post("/predict", json=client_data)

    # 3. Assert : Ici on attend 400 (notre erreur personnalisée), pas 422
    assert response.status_code == 400
    assert "Champs non reconnus" in response.json()["detail"]

def test_prediction_acceptee_si_nombre_colonne_minimum():
    """Vérifie que l'API accepte les données avec le nombre minimum de colonnes"""
    
    # 1. Arrange : On crée un client avec les colonnes minimales requises
    client_data = {
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": 100000,
        "AMT_ANNUITY": 5000,
        "AMT_GOODS_PRICE": 50000,
        "DAYS_BIRTH": -15000,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Higher education"
    }

    # 2. Act : On envoie la requête
    response = client.post("/predict", json=client_data)

    # 3. Assert : Ici on attend 400 (notre erreur personnalisée), pas 422
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["score_defaut"], float)
    assert data["decision"] in ["Accordé", "Refusé"]

    assert  0 <= data["score_defaut"] <=1

def test_prediction_acceptee_si_ext_source1_manquant():
    """Vérifie que l'API accepte les données avec le nombre minimum de colonnes"""
    
    # 1. Arrange : On crée un client avec les colonnes minimales requises
    client_data = {
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "AMT_CREDIT": 100000,
        "AMT_ANNUITY": 5000,
        "AMT_GOODS_PRICE": 50000,
        "DAYS_BIRTH": -15000,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Higher education"
    }

    # 2. Act : On envoie la requête
    response = client.post("/predict", json=client_data)

    # 3. Assert : Ici on attend 400 (notre erreur personnalisée), pas 422
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["score_defaut"], float)
    assert data["decision"] in ["Accordé", "Refusé"]

    assert  0 <= data["score_defaut"] <=1