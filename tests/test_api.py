import pytest
from fastapi.testclient import TestClient
from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert "version" in data


def test_predict_endpoint_success(client):
    """Verify single patient prediction via POST /predict."""
    payload = {
        "age": 65,
        "sex": "Male",
        "ambulance_arrival": 1,
        "walking_ability": 0,
        "altered_consciousness": 1,
        "chest_pain": 1,
        "difficulty_breathing": 1,
        "abdominal_pain": 0,
        "injury_trauma": 0,
        "bleeding": 0,
        "fever": 0,
        "headache": 0,
        "vomiting": 0,
        "heart_rate": 120.0,
        "systolic_bp": 85.0,
        "diastolic_bp": 50.0,
        "spo2": 86.0,
        "respiratory_rate": 28.0,
        "temperature": 37.0,
        "gcs": 10,
        "pain_severity": 8,
        "oxygen_requirement": 1,
        "known_cardiac_history": 1,
        "known_hypertension": 1,
        "known_diabetes": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "criticality_score" in data
    assert data["criticality_score"] >= 1.0 and data["criticality_score"] <= 10.0
    assert data["urgency_tier"] in ["Low", "Moderate", "Elevated", "High", "Critical"]
    assert "inference_latency_ms" in data
    assert len(data["red_flags"]) > 0  # Should trigger red flags for shock / hypoxemia


def test_predict_endpoint_validation_error(client):
    """Verify 422 unprocessable entity on invalid input schema."""
    invalid_payload = {
        "age": 150,  # Out of range
        "sex": "UnknownSex",
        "systolic_bp": 60.0,
        "diastolic_bp": 90.0  # DBP >= SBP violation
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_batch_predict_endpoint(client):
    """Verify batch prediction via POST /batch-predict."""
    patient1 = {
        "age": 25, "sex": "Female", "ambulance_arrival": 0, "walking_ability": 1,
        "altered_consciousness": 0, "chest_pain": 0, "difficulty_breathing": 0,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 0,
        "headache": 0, "vomiting": 0, "heart_rate": 70.0, "systolic_bp": 115.0,
        "diastolic_bp": 75.0, "spo2": 99.0, "respiratory_rate": 14.0, "temperature": 36.8,
        "gcs": 15, "pain_severity": 1, "oxygen_requirement": 0,
        "known_cardiac_history": 0, "known_hypertension": 0, "known_diabetes": 0
    }
    patient2 = {
        "age": 70, "sex": "Male", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 1, "chest_pain": 1, "difficulty_breathing": 1,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 0,
        "headache": 0, "vomiting": 0, "heart_rate": 130.0, "systolic_bp": 80.0,
        "diastolic_bp": 45.0, "spo2": 82.0, "respiratory_rate": 30.0, "temperature": 36.5,
        "gcs": 9, "pain_severity": 9, "oxygen_requirement": 1,
        "known_cardiac_history": 1, "known_hypertension": 1, "known_diabetes": 1
    }
    batch_payload = {"patients": [patient1, patient2]}
    response = client.post("/batch-predict", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["criticality_score"] < data[1]["criticality_score"]
