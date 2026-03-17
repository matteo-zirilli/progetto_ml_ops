from fastapi.testclient import TestClient
from app.main import app


#setup del Client e test del raggiungimento della root

client = TestClient(app)
def test_get_root():
    response = client.get("/")

    assert response.status_code == 200, "Errore: la root non ha risposto con 200!"