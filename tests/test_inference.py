"""
Unit and Integration Tests for Production Inference Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System
"""

import pytest
from pydantic import ValidationError
from src.predict import PatientPayload, ClinicalInferenceService


def test_payload_validation_success():
    """Verify standard payload passes Pydantic validation."""
    valid_data = {
        "age": 52,
        "sex": "Female",
        "ambulance_arrival": 1,
        "walking_ability": 1,
        "altered_consciousness": 0,
        "chest_pain": 0,
        "difficulty_breathing": 0,
        "abdominal_pain": 0,
        "injury_trauma": 0,
        "bleeding": 0,
        "fever": 0,
        "headache": 0,
        "vomiting": 0,
        "heart_rate": 78.0,
        "systolic_bp": 124.0,
        "diastolic_bp": 82.0,
        "spo2": 98.0,
        "respiratory_rate": 16.0,
        "temperature": 37.0,
        "gcs": 15,
        "pain_severity": 1,
        "oxygen_requirement": 0,
        "known_cardiac_history": 0,
        "known_hypertension": 1,
        "known_diabetes": 0
    }
    payload = PatientPayload(**valid_data)
    assert payload.age == 52
    assert payload.sex == "Female"


def test_payload_validation_fails_on_out_of_bounds_and_invariants():
    """Verify validation errors are raised for invalid inputs and BP violations."""
    invalid_data = {
        "age": 120,  # Invalid (>95)
        "sex": "InvalidCategory",
        "systolic_bp": 80.0,
        "diastolic_bp": 90.0  # Invalid (DBP >= SBP)
    }
    with pytest.raises(ValidationError):
        PatientPayload(**invalid_data)


def test_safety_guardrails_trigger_on_critical_vitals():
    """Verify that hard red flags are identified for coma, hypoxemia, or shock."""
    service = ClinicalInferenceService.__new__(ClinicalInferenceService)
    
    # Comatose patient
    flags = service.check_safety_red_flags({"gcs": 6, "spo2": 95.0, "systolic_bp": 110.0})
    assert any("COMA" in f for f in flags)
    
    # Severe Hypoxemia
    flags = service.check_safety_red_flags({"gcs": 15, "spo2": 82.0, "systolic_bp": 110.0})
    assert any("RESPIRATORY FAILURE" in f for f in flags)
    
    # Profound Shock
    flags = service.check_safety_red_flags({"gcs": 15, "spo2": 98.0, "systolic_bp": 72.0, "heart_rate": 120.0})
    assert any("HYPOTENSIVE SHOCK" in f for f in flags)
