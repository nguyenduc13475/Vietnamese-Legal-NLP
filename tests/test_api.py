from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_extract_endpoint():
    payload = {"text": "Bên A cho Bên B thuê căn nhà số 123."}
    response = client.post("/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert "intent" in data["results"][0]
    assert "entities" in data["results"][0]