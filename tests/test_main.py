from fastapi.testclient import TestClient
from app.main import app


#setup del Client e test del raggiungimento della root

client = TestClient(app)
def test_get_root():
    response = client.get("/")

    assert response.status_code == 200, "Errore: la root non ha risposto con 200!"


#test per la post activity
def test_predict_sentiment():


    text = {"text": "I absolutely love this project"}
    response = client.post("/predict", json = text)
    data = response.json()
    print(data)
    assert response.status_code == 200, "Errore: la post activity non ha funzionato!"
    assert "label" in data, "Nella risposta non è stata trovata la label"
    assert "score" in data, "Nella risposta non è stata trovata lo score"


#eseguo un test sull'input errato
def test_predict_invalid_input():
    # Invio un JSON totalmente sbagliato (una chiave inesistente e un numero)
    bad_payload = {"testo_sbagliato": 12345}
    response = client.post("/predict", json=bad_payload)
    
   
    #verificato a priori che il tipo di errore per questo genere di test è 422
    assert response.status_code == 422, "Errore: l'API ha accettato un input non valido!"
