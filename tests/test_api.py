import pytest
from fastapi.testclient import TestClient

from creditlens import api

APPLICANT = {
    "revolving_utilization": 0.85, "age": 29, "past_due_30_59": 2, "past_due_60_89": 0,
    "past_due_90_plus": 0, "debt_ratio": 0.45, "monthly_income": 4200, "open_credit_lines": 7,
    "real_estate_loans": 0, "dependents": 1,
}


@pytest.fixture()
def client(card):
    # swap in the small scorecard from conftest
    original = api.get_model
    api.get_model = lambda: card
    yield TestClient(api.app)
    api.get_model = original


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score(client):
    r = client.post("/score", json=APPLICANT)
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["probability_of_default"] < 1
    assert body["decision"] in {"approve", "manual_review", "decline"}
    assert body["grade"] in {"A", "B", "C", "D", "E"}
    assert isinstance(body["score"], int)


def test_missing_income(client):
    r = client.post("/score", json={**APPLICANT, "monthly_income": None})
    assert r.status_code == 200


def test_bad_input(client):
    assert client.post("/score", json={**APPLICANT, "age": 12}).status_code == 422
    assert client.post("/score", json={**APPLICANT, "revolving_utilization": -1}).status_code == 422


def test_approve_and_decline(client):
    clean = {**APPLICANT, "revolving_utilization": 0.03, "past_due_30_59": 0, "age": 60}
    risky = {**APPLICANT, "revolving_utilization": 1.0, "past_due_30_59": 3, "past_due_90_plus": 3,
             "past_due_60_89": 2, "age": 23}
    assert client.post("/score", json=clean).json()["decision"] == "approve"
    assert client.post("/score", json=risky).json()["decision"] == "decline"


def test_scorecard_endpoint(client):
    body = client.get("/scorecard").json()
    assert body["pdo"] == 20 and len(body["rows"]) > 0
