"""
Unit Tests for Triage AI Agent Tools
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
from src.agent.tools import TriageTools


@pytest.fixture
def sample_payload():
    return {
        "age": 55, "sex": "Male", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 0, "chest_pain": 1, "difficulty_breathing": 1,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 0,
        "headache": 0, "vomiting": 0, "heart_rate": 105.0, "systolic_bp": 92.0,
        "diastolic_bp": 60.0, "spo2": 89.0, "respiratory_rate": 24.0, "temperature": 37.1,
        "gcs": 14, "pain_severity": 7, "oxygen_requirement": 1,
        "known_cardiac_history": 1, "known_hypertension": 1, "known_diabetes": 0
    }


def test_get_current_patient_context(sample_payload):
    tools = TriageTools(current_payload=sample_payload)
    res = tools.get_current_patient_context()
    assert res["status"] == "success"
    assert "demographics" in res["structured_data"]
    assert res["structured_data"]["vitals"]["spo2_percent"] == 89.0
    assert "Chest Pain" in res["structured_data"]["symptoms"]


def test_get_current_prediction(sample_payload):
    tools = TriageTools(current_payload=sample_payload)
    res = tools.get_current_prediction()
    assert res["status"] == "success"
    assert isinstance(res["criticality_score"], float)
    assert res["criticality_score"] >= 1.0 and res["criticality_score"] <= 10.0
    assert res["urgency_tier"] in ["Low", "Moderate", "Elevated", "High", "Critical"]


def test_get_shap_explanation(sample_payload):
    tools = TriageTools(current_payload=sample_payload)
    res = tools.get_shap_explanation()
    assert res["status"] == "success"
    assert "top_contributing_factors" in res
    assert len(res["top_contributing_factors"]) > 0


def test_run_what_if_prediction_real_ml(sample_payload):
    tools = TriageTools(current_payload=sample_payload)
    
    # Original SpO2 is 89%
    # Test what-if SpO2 improves to 99%
    what_if_res = tools.run_what_if_prediction("spo2", 99.0)
    
    assert what_if_res["status"] == "success"
    assert what_if_res["feature_modified"] == "spo2"
    assert what_if_res["baseline_value"] == 89.0
    assert what_if_res["hypothetical_value"] == 99.0
    # Improving SpO2 from 89% to 99% must lower or equal the criticality score under learned relationships
    assert what_if_res["hypothetical_criticality"] <= what_if_res["baseline_criticality"]
    assert what_if_res["score_delta"] <= 0.0


def test_get_model_information():
    tools = TriageTools()
    res = tools.get_model_information()
    assert res["status"] == "success"
    assert res["r2_score"] >= 0.90
    assert "Tuned XGBoost" in res["model_name"]


def test_get_project_information():
    tools = TriageTools()
    res = tools.get_project_information("decisions")
    assert res["status"] == "success"
    assert "content_excerpt" in res
