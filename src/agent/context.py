"""
Patient Context Management for Triage AI Agent
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Extracts, normalizes, and packages active patient telemetry, vitals,
symptoms, and ML predictions from Streamlit session state or caller payloads.
"""

import os
import sys
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath("."))


def normalize_patient_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes raw patient dictionary into clinical values and human-readable tags.
    """
    age = raw_payload.get("age", 50)
    sex = raw_payload.get("sex", "Unknown")
    hr = raw_payload.get("heart_rate", 80.0)
    sbp = raw_payload.get("systolic_bp", 120.0)
    dbp = raw_payload.get("diastolic_bp", 80.0)
    spo2 = raw_payload.get("spo2", 98.0)
    rr = raw_payload.get("respiratory_rate", 16.0)
    temp = raw_payload.get("temperature", 37.0)
    gcs = raw_payload.get("gcs", 15)
    pain = raw_payload.get("pain_severity", 0)
    
    # Derived parameters
    shock_index = round(hr / max(sbp, 1.0), 2)
    pulse_pressure = round(sbp - dbp, 1)
    
    # Active symptoms list
    active_symptoms = []
    if raw_payload.get("chest_pain"): active_symptoms.append("Chest Pain")
    if raw_payload.get("difficulty_breathing"): active_symptoms.append("Difficulty Breathing / Dyspnea")
    if raw_payload.get("altered_consciousness"): active_symptoms.append("Altered Mental State")
    if raw_payload.get("abdominal_pain"): active_symptoms.append("Abdominal Pain")
    if raw_payload.get("injury_trauma"): active_symptoms.append("Physical Trauma / Injury")
    if raw_payload.get("bleeding"): active_symptoms.append("Active Bleeding")
    if raw_payload.get("fever"): active_symptoms.append("Fever / Chills")
    if raw_payload.get("headache"): active_symptoms.append("Headache")
    if raw_payload.get("vomiting"): active_symptoms.append("Nausea / Vomiting")
    if not active_symptoms:
        active_symptoms.append("None reported")
        
    # Comorbidities
    comorbidities = []
    if raw_payload.get("known_cardiac_history"): comorbidities.append("Known Cardiac Disease")
    if raw_payload.get("known_hypertension"): comorbidities.append("Hypertension")
    if raw_payload.get("known_diabetes"): comorbidities.append("Diabetes")
    if not comorbidities:
        comorbidities.append("None documented")

    return {
        "demographics": {
            "age": age,
            "sex": sex,
            "is_geriatric": bool(age >= 65)
        },
        "vitals": {
            "heart_rate_bpm": hr,
            "blood_pressure_mmhg": f"{sbp:.0f}/{dbp:.0f}",
            "systolic_bp": sbp,
            "diastolic_bp": dbp,
            "spo2_percent": spo2,
            "respiratory_rate_bpm": rr,
            "temperature_celsius": temp,
            "gcs_score": gcs,
            "pain_score": pain,
            "oxygen_requirement": bool(raw_payload.get("oxygen_requirement", 0))
        },
        "derived_indices": {
            "shock_index": shock_index,
            "shock_status": "Severe (>1.1)" if shock_index > 1.1 else ("Elevated (0.9-1.1)" if shock_index >= 0.9 else "Normal (0.5-0.7)"),
            "pulse_pressure": pulse_pressure
        },
        "symptoms": active_symptoms,
        "comorbidities": comorbidities,
        "mode_of_arrival": "Ambulance" if raw_payload.get("ambulance_arrival") else "Walk-in",
        "can_walk": bool(raw_payload.get("walking_ability"))
    }


def format_patient_summary_text(normalized_data: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats normalized patient dictionary into a concise clinical markdown text block for the LLM.
    """
    demo = normalized_data["demographics"]
    vitals = normalized_data["vitals"]
    indices = normalized_data["derived_indices"]
    
    text = (
        f"**Patient**: {demo['age']}-year-old {demo['sex']} (Arrival: {normalized_data['mode_of_arrival']})\n"
        f"**Vitals**: HR {vitals['heart_rate_bpm']:.0f} bpm | BP {vitals['blood_pressure_mmhg']} mmHg | "
        f"SpO2 {vitals['spo2_percent']:.0f}% | RR {vitals['respiratory_rate_bpm']:.0f} bpm | "
        f"Temp {vitals['temperature_celsius']:.1f}°C | GCS {vitals['gcs_score']}/15 | Pain {vitals['pain_score']}/10\n"
        f"**Calculated Indices**: Shock Index: {indices['shock_index']} ({indices['shock_status']}) | "
        f"Pulse Pressure: {indices['pulse_pressure']} mmHg\n"
        f"**Documented Symptoms**: {', '.join(normalized_data['symptoms'])}\n"
        f"**Past Medical History**: {', '.join(normalized_data['comorbidities'])}\n"
    )
    
    if prediction:
        text += (
            f"**Current ML Prediction**: Acuity Score {prediction.get('criticality_score', 'N/A')}/10.0 | "
            f"Urgency Tier: [{prediction.get('urgency_tier', 'N/A').upper()}]\n"
            f"**Routing Recommendation**: {prediction.get('clinical_routing_guidance', 'Standard triage evaluation')}\n"
        )
        if prediction.get("red_flags"):
            text += f"**Active Safety Red Flags**: {'; '.join(prediction['red_flags'])}\n"
            
    return text
