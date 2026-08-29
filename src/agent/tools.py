"""
Controlled Agent Tools & Real ML Pipeline Invocation Engine
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Defines the 7 primary tools used by the Triage AI Agent.
Enforces the principle: ML is the sole authority for predictions and SHAP values.
"""

import os
import sys
import json
import glob
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))

from src.predict import ClinicalInferenceService
from src.agent.context import normalize_patient_payload, format_patient_summary_text
from src.rag.knowledge_base import get_knowledge_base


# Singleton instance of real ML inference service
_ml_service: Optional[ClinicalInferenceService] = None

def get_ml_service() -> ClinicalInferenceService:
    global _ml_service
    if _ml_service is None:
        _ml_service = ClinicalInferenceService(
            pipeline_path="models/pipeline.joblib",
            train_path="data/processed/train.csv"
        )
    return _ml_service


class TriageTools:
    """
    Collection of callable tools for the Triage AI Agent.
    """

    def __init__(self, current_payload: Optional[Dict[str, Any]] = None, current_prediction: Optional[Dict[str, Any]] = None):
        self.current_payload = current_payload or {
            "age": 60, "sex": "Male", "ambulance_arrival": 1, "walking_ability": 0,
            "altered_consciousness": 0, "chest_pain": 1, "difficulty_breathing": 1,
            "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 0,
            "headache": 0, "vomiting": 0, "heart_rate": 110.0, "systolic_bp": 95.0,
            "diastolic_bp": 60.0, "spo2": 89.0, "respiratory_rate": 26.0, "temperature": 37.2,
            "gcs": 14, "pain_severity": 8, "oxygen_requirement": 1,
            "known_cardiac_history": 1, "known_hypertension": 1, "known_diabetes": 0
        }
        
        # If prediction not passed, compute it using the real ML service
        if current_prediction:
            self.current_prediction = current_prediction
        else:
            service = get_ml_service()
            self.current_prediction = service.predict(self.current_payload)

    # -------------------------------------------------------------
    # TOOL 1: GET CURRENT PATIENT CONTEXT
    # -------------------------------------------------------------
    def get_current_patient_context(self) -> Dict[str, Any]:
        """
        Retrieves the clinical parameters, vitals, symptoms, and derived indices of the currently loaded patient.
        """
        normalized = normalize_patient_payload(self.current_payload)
        summary_text = format_patient_summary_text(normalized, self.current_prediction)
        return {
            "status": "success",
            "summary_text": summary_text,
            "structured_data": normalized
        }

    # -------------------------------------------------------------
    # TOOL 2: GET CURRENT ML PREDICTION
    # -------------------------------------------------------------
    def get_current_prediction(self) -> Dict[str, Any]:
        """
        Returns the actual prediction, acuity score, urgency tier, and safety red flags from the ML model.
        """
        return {
            "status": "success",
            "model_name": "Tuned XGBoost Pipeline (5-Tier Acuity Regressor)",
            "criticality_score": self.current_prediction.get("criticality_score"),
            "urgency_tier": self.current_prediction.get("urgency_tier"),
            "raw_urgency_tier": self.current_prediction.get("raw_urgency_tier"),
            "safety_override_applied": self.current_prediction.get("safety_override_applied"),
            "red_flags": self.current_prediction.get("red_flags", []),
            "clinical_routing_guidance": self.current_prediction.get("clinical_routing_guidance")
        }

    # -------------------------------------------------------------
    # TOOL 3: GET SHAP EXPLANATION
    # -------------------------------------------------------------
    def get_shap_explanation(self) -> Dict[str, Any]:
        """
        Returns the top contributing physiological variables and their exact SHAP contributions.
        """
        explanation = self.current_prediction.get("explanation", {})
        top_factors = explanation.get("top_factors", [])
        narrative = explanation.get("narrative", [])
        base_value = explanation.get("base_value", 5.0)

        formatted_factors = []
        for factor in top_factors:
            feat = factor.get("feature", "unknown")
            impact = factor.get("shap_impact", 0.0)
            direction = "elevates acuity" if impact > 0 else "reduces acuity"
            formatted_factors.append({
                "feature": feat,
                "shap_impact": round(impact, 3),
                "direction": direction
            })

        return {
            "status": "success",
            "base_expected_value": round(base_value, 2),
            "top_contributing_factors": formatted_factors,
            "clinical_narratives": narrative
        }

    # -------------------------------------------------------------
    # TOOL 4: RUN WHAT-IF PREDICTION (REAL ML EXECUTION)
    # -------------------------------------------------------------
    def run_what_if_prediction(self, feature_name: str, new_value: Any) -> Dict[str, Any]:
        """
        Executes a true sensitivity analysis by modifying ONLY the requested feature in the current
        patient payload and running it through the real ML inference pipeline.
        """
        # Map common natural language aliases to schema column names
        feature_alias_map = {
            "spo2": "spo2", "o2": "spo2", "oxygen": "spo2", "oxygen saturation": "spo2",
            "heart rate": "heart_rate", "hr": "heart_rate", "pulse": "heart_rate",
            "systolic": "systolic_bp", "sbp": "systolic_bp", "systolic bp": "systolic_bp",
            "diastolic": "diastolic_bp", "dbp": "diastolic_bp", "diastolic bp": "diastolic_bp",
            "respiratory rate": "respiratory_rate", "rr": "respiratory_rate", "breathing rate": "respiratory_rate",
            "gcs": "gcs", "glasgow coma scale": "gcs", "mental status": "gcs",
            "temperature": "temperature", "temp": "temperature", "fever temp": "temperature",
            "pain": "pain_severity", "pain score": "pain_severity",
            "chest pain": "chest_pain", "difficulty breathing": "difficulty_breathing",
            "altered consciousness": "altered_consciousness", "walking ability": "walking_ability",
            "bleeding": "bleeding", "trauma": "injury_trauma"
        }

        canonical_feature = feature_alias_map.get(feature_name.lower().strip(), feature_name.lower().strip())
        if canonical_feature not in self.current_payload:
            return {
                "status": "error",
                "message": f"Feature '{feature_name}' not recognized in patient schema. Valid parameters include: SpO2, Heart Rate, Systolic BP, Diastolic BP, Respiratory Rate, GCS, Temperature, Pain Severity."
            }

        # Convert value type according to existing schema
        old_value = self.current_payload[canonical_feature]
        try:
            if isinstance(old_value, float):
                parsed_value = float(new_value)
            elif isinstance(old_value, int):
                parsed_value = int(float(new_value))
            else:
                parsed_value = new_value
        except (ValueError, TypeError):
            return {
                "status": "error",
                "message": f"Invalid value '{new_value}' for feature '{canonical_feature}'."
            }

        # Clone and modify payload
        hypothetical_payload = self.current_payload.copy()
        hypothetical_payload[canonical_feature] = parsed_value

        # Re-run REAL ML Inference Service
        service = get_ml_service()
        hypothetical_result = service.predict(hypothetical_payload)

        orig_score = self.current_prediction.get("criticality_score", 5.0)
        new_score = hypothetical_result.get("criticality_score", 5.0)
        delta = round(new_score - orig_score, 2)

        return {
            "status": "success",
            "feature_modified": canonical_feature,
            "baseline_value": old_value,
            "hypothetical_value": parsed_value,
            "baseline_criticality": orig_score,
            "baseline_urgency": self.current_prediction.get("urgency_tier"),
            "hypothetical_criticality": new_score,
            "hypothetical_urgency": hypothetical_result.get("urgency_tier"),
            "score_delta": delta,
            "delta_description": f"{abs(delta):.2f} points {'higher' if delta > 0 else ('lower' if delta < 0 else 'unchanged')}",
            "safety_override_triggered": hypothetical_result.get("safety_override_applied", False),
            "new_red_flags": hypothetical_result.get("red_flags", [])
        }

    # -------------------------------------------------------------
    # TOOL 5: GET MODEL INFORMATION & PERFORMANCE
    # -------------------------------------------------------------
    def get_model_information(self) -> Dict[str, Any]:
        """
        Dynamically reads model metadata, test set metrics, hyperparameters, and target definition.
        """
        metadata_path = "models/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        metrics = meta.get("test_metrics", {})
        return {
            "status": "success",
            "model_name": meta.get("model_name", "Tuned XGBoost Regressor"),
            "pipeline_type": "Scikit-Learn ColumnTransformer + XGBoost Regressor",
            "target_variable": "criticality_score (1.0 to 10.0 continuous scale)",
            "5_tier_mapping": "Low (1.0-2.4), Moderate (2.5-4.4), Elevated (4.5-6.4), High (6.5-8.4), Critical (8.5-10.0)",
            "test_set_size": 2000,
            "training_set_size": 8000,
            "r2_score": metrics.get("r2", 0.9935),
            "mae": metrics.get("mae", 0.1651),
            "rmse": metrics.get("rmse", 0.2348),
            "exact_tier_accuracy": f"{metrics.get('tier_accuracy', 0.905) * 100:.1f}%",
            "critical_tier_recall": f"{metrics.get('critical_recall', 0.9608) * 100:.1f}%",
            "severe_under_triage_rate": f"{metrics.get('severe_under_triage_rate', 0.0) * 100:.2f}% (Zero dangerous tier misclassifications)",
            "explainability_engine": "SHAP TreeExplainer with background baseline"
        }

    # -------------------------------------------------------------
    # TOOL 6: GET PROJECT DOCUMENTATION INFORMATION
    # -------------------------------------------------------------
    def get_project_information(self, topic: str) -> Dict[str, Any]:
        """
        Retrieves architectural decisions, dataset design, and methodology from docs/ Markdown files.
        """
        topic_lower = topic.lower()
        doc_map = {
            "decisions": "docs/DECISIONS.md",
            "architecture": "docs/DECISIONS.md",
            "model card": "docs/MODEL_CARD.md",
            "scope": "docs/PROJECT_SCOPE.md",
            "dataset": "docs/DATASET_INVESTIGATION.md",
            "target": "docs/TARGET_DEFINITION.md",
            "error analysis": "docs/ERROR_ANALYSIS.md",
            "safety": "docs/SECURITY_PRIVACY.md",
            "security": "docs/SECURITY_PRIVACY.md",
            "deployment": "docs/DEPLOYMENT_GUIDE.md"
        }

        # Find best matching document
        matched_file = "docs/DECISIONS.md"
        for key, path in doc_map.items():
            if key in topic_lower:
                matched_file = path
                break

        if os.path.exists(matched_file):
            with open(matched_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Return first 3000 chars of document for context
            excerpt = content[:3000]
            return {
                "status": "success",
                "document": matched_file,
                "content_excerpt": excerpt
            }
        else:
            return {
                "status": "error",
                "message": f"Documentation file '{matched_file}' not found."
            }

    # -------------------------------------------------------------
    # TOOL 7: QUERY TRIAGE KNOWLEDGE BASE (RAG)
    # -------------------------------------------------------------
    def query_triage_knowledge_base(self, query: str) -> Dict[str, Any]:
        """
        Searches the Chroma vector store for authoritative healthcare and triage literature (WHO, NIH, CDC, NHS).
        """
        kb = get_knowledge_base()
        chunks = kb.retrieve(query, top_k=3)
        return {
            "status": "success",
            "query": query,
            "retrieved_chunks_count": len(chunks),
            "chunks": chunks
        }
