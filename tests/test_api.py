from fastapi.testclient import TestClient

from src.api.app import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    client = TestClient(app)
    payload = {
        "records": [
            {
                "age": 30,
                "job": "admin.",
                "marital": "single",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 100,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": 1.1,
                "cons.price.idx": 93.2,
                "cons.conf.idx": -34.6,
                "euribor3m": 4.9,
                "nr.employed": 5200.0,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 422)
