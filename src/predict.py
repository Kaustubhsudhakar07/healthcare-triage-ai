"""
Production Clinical Inference and Decision-Support Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Handles payload schema validation (Pydantic), pipeline transformation,
model execution, hard physiological safety guardrails, and SHAP explanations.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import (
    score_to_urgency_tier,
    clip_criticality_scores,
    TARGET_CONTINUOUS,
    TARGET_CATEGORICAL,
    ID_COLUMN
)
from src.explainability import ClinicalExplainer, clean_feature_name


class PatientPayload(BaseModel):
    """
    Strict Pydantic schema for pre-hospital patient triage inputs.
    """
    age: int = Field(default=45, ge=16, le=95, description="Patient age in years")
    sex: str = Field(default="Male", description="Biological sex ('Male', 'Female', 'Other')")
    ambulance_arrival: int = Field(default=1, ge=0, le=1, description="Transit mode (1: EMS, 0: Walk-in)")
    walking_ability: int = Field(default=1, ge=0, le=1, description="Ambulatory status (1: Yes, 0: No)")
    altered_consciousness: int = Field(default=0, ge=0, le=1, description="Altered mental status (1: Yes, 0: No)")
    chest_pain: int = Field(default=0, ge=0, le=1, description="Chest pain complaint (1: Yes, 0: No)")
    difficulty_breathing: int = Field(default=0, ge=0, le=1, description="Dyspnea / breathing difficulty (1: Yes, 0: No)")
    abdominal_pain: int = Field(default=0, ge=0, le=1, description="Acute abdominal pain (1: Yes, 0: No)")
    injury_trauma: int = Field(default=0, ge=0, le=1, description="Physical trauma / collision / fall (1: Yes, 0: No)")
    bleeding: int = Field(default=0, ge=0, le=1, description="Active external or internal bleeding (1: Yes, 0: No)")
    fever: int = Field(default=0, ge=0, le=1, description="Fever / chills (1: Yes, 0: No)")
    headache: int = Field(default=0, ge=0, le=1, description="Severe headache (1: Yes, 0: No)")
    vomiting: int = Field(default=0, ge=0, le=1, description="Nausea / vomiting (1: Yes, 0: No)")
    heart_rate: float = Field(default=75.0, ge=30.0, le=250.0, description="Heart rate in bpm")
    systolic_bp: float = Field(default=120.0, ge=50.0, le=260.0, description="Systolic blood pressure in mmHg")
    diastolic_bp: float = Field(default=80.0, ge=30.0, le=160.0, description="Diastolic blood pressure in mmHg")
    spo2: float = Field(default=98.0, ge=50.0, le=100.0, description="Blood oxygen saturation %")
    respiratory_rate: float = Field(default=16.0, ge=6.0, le=60.0, description="Respiratory rate in breaths/min")
    temperature: float = Field(default=37.0, ge=33.0, le=43.0, description="Body temperature in °C")
    gcs: int = Field(default=15, ge=3, le=15, description="Glasgow Coma Scale score (3-15)")
    pain_severity: int = Field(default=2, ge=0, le=10, description="Pain rating (0-10 NRS)")
    oxygen_requirement: int = Field(default=0, ge=0, le=1, description="Pre-hospital supplemental O2 administered")
    known_cardiac_history: int = Field(default=0, ge=0, le=1, description="History of MI / CHF / CAD")
    known_hypertension: int = Field(default=0, ge=0, le=1, description="Known diagnosed hypertension")
    known_diabetes: int = Field(default=0, ge=0, le=1, description="Known diagnosed diabetes")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v not in {"Male", "Female", "Other"}:
            raise ValueError(f"Invalid sex '{v}'. Must be 'Male', 'Female', or 'Other'.")
        return v

    @field_validator("diastolic_bp")
    @classmethod
    def validate_bp(cls, v: float, info) -> float:
        # SBP comparison if present
        if "systolic_bp" in info.data and v >= info.data["systolic_bp"]:
            raise ValueError(f"Diastolic BP ({v}) must be strictly less than Systolic BP ({info.data['systolic_bp']}).")
        return v


class ClinicalInferenceService:
    """
    Singleton-style inference service managing pipeline execution,
    hard safety overrides, and SHAP feature attribution.
    """
    def __init__(self, pipeline_path: str = "models/pipeline.joblib", train_path: str = "data/processed/train.csv"):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline model file not found at: {pipeline_path}")
            
        self.pipeline = joblib.load(pipeline_path)
        
        # Initialize Explainer with training baseline sample
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            feature_cols = [c for c in train_df.columns if c not in [ID_COLUMN, TARGET_CONTINUOUS, TARGET_CATEGORICAL]]
            bg_sample = train_df[feature_cols].sample(min(300, len(train_df)), random_state=42)
            self.explainer = ClinicalExplainer(self.pipeline, bg_sample)
        else:
            self.explainer = None

    def check_safety_red_flags(self, payload: Dict[str, Any]) -> List[str]:
        """
        Evaluates hard clinical safety guardrails (immediate resuscitation criteria).
        """
        red_flags = []
        
        # 1. Airway / Coma Emergency
        if payload.get("gcs", 15) <= 8:
            red_flags.append("CRITICAL COMA / AIRWAY RISK: GCS <= 8 (Immediate intubation / airway protection indicated).")
            
        # 2. Severe Hypoxemia
        if payload.get("spo2", 98.0) < 88.0:
            red_flags.append(f"SEVERE RESPIRATORY FAILURE: SpO2 is {payload.get('spo2')}% (< 88%). High-flow O2 / ventilation required.")
            
        # 3. Profound Shock / Hypotension
        sbp = payload.get("systolic_bp", 120.0)
        hr = payload.get("heart_rate", 75.0)
        shock_idx = hr / max(sbp, 1.0)
        if sbp < 85.0:
            red_flags.append(f"PROFOUND HYPOTENSIVE SHOCK: Systolic BP is {sbp} mmHg (< 85 mmHg). Immediate IV access & fluid resuscitation.")
        elif shock_idx >= 1.2:
            red_flags.append(f"HIGH SHOCK INDEX ({shock_idx:.2f}): Severe circulatory compromise.")
            
        # 4. Exsanguinating Hemorrhage
        if payload.get("bleeding", 0) == 1 and payload.get("injury_trauma", 0) == 1 and sbp < 100:
            red_flags.append("HEMORRHAGIC TRAUMA RISK: Active bleeding with decompensating blood pressure.")
            
        # 5. Acute Coronary Syndrome Red Flag
        if payload.get("chest_pain", 0) == 1 and payload.get("known_cardiac_history", 0) == 1 and payload.get("age", 40) >= 50:
            red_flags.append("HIGH-RISK CARDIAC SUSPICION: Chest pain in patient with known cardiac history. Immediate ECG triage bay.")
            
        return red_flags

    def predict(self, payload_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction, safety checks, and local explanation.
        """
        # Validate schema via Pydantic
        validated = PatientPayload(**payload_dict)
        data_df = pd.DataFrame([validated.model_dump()])
        
        # Model Prediction
        raw_pred = self.pipeline.predict(data_df)
        score = float(clip_criticality_scores(raw_pred)[0])
        tier = str(score_to_urgency_tier([score])[0])
        
        # Hard Safety Guardrails
        red_flags = self.check_safety_red_flags(validated.model_dump())
        safety_override = False
        
        # If critical red flags exist but model predicted < 6.5, apply safety floor
        if len(red_flags) > 0 and score < 7.0:
            safety_override = True
            recommended_tier = "High" if score < 6.5 else tier
        else:
            recommended_tier = tier
            
        # Triage Protocol & Routing Guidance
        routing_guidance = {
            "Low": "Queue for routine ED intake and ambulatory waiting area. Standard nursing check within 120 min.",
            "Moderate": "Assign to Acute Care bed. Clinical evaluation and vitals monitoring within 60 min.",
            "Elevated": "Assign to Monitored ED bay. Physician assessment and diagnostic workup within 30 min.",
            "High": "Priority High-Acuity bay. Immediate physician evaluation, continuous ECG/SpO2 telemetry, IV line.",
            "Critical": "IMMEDIATE RESUSCITATION / TRAUMA BAY (Code 1). Multidisciplinary emergency resuscitation team activation."
        }[recommended_tier]
        
        # Compute Local SHAP Explanation
        explanation = None
        if self.explainer is not None:
            try:
                explanation = self.explainer.explain_instance(data_df)
            except Exception as e:
                explanation = {"error": f"SHAP explanation failed: {str(e)}"}
                
        return {
            "criticality_score": score,
            "urgency_tier": recommended_tier,
            "raw_urgency_tier": tier,
            "safety_guardrails_triggered": len(red_flags) > 0,
            "safety_override_applied": safety_override,
            "red_flags": red_flags,
            "clinical_routing_guidance": routing_guidance,
            "explanation": explanation
        }


# Quick test
if __name__ == "__main__":
    if os.path.exists("models/pipeline.joblib"):
        service = ClinicalInferenceService()
        sample_patient = {
            "age": 68,
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
            "heart_rate": 118.0,
            "systolic_bp": 92.0,
            "diastolic_bp": 55.0,
            "spo2": 85.0,
            "respiratory_rate": 28.0,
            "temperature": 36.9,
            "gcs": 11,
            "pain_severity": 8,
            "oxygen_requirement": 1,
            "known_cardiac_history": 1,
            "known_hypertension": 1,
            "known_diabetes": 0
        }
        res = service.predict(sample_patient)
        print("Inference Result:")
        print(f"  Criticality Score: {res['criticality_score']}")
        print(f"  Urgency Tier:      {res['urgency_tier']}")
        print(f"  Red Flags:         {res['red_flags']}")
        if res["explanation"]:
            print(f"  Top Factors:       {res['explanation']['narrative']}")
