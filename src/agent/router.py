"""
Lightweight Intent & Tool Selection Router
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Analyzes user queries to select appropriate deterministic tool executions
without relying on slow, unconstrained multi-agent loops.
"""

import re
from typing import Dict, Any, List, Tuple


class IntentRouter:
    """
    Deterministic regex and keyword-based intent router for emergency triage inquiries.
    """

    @staticmethod
    def route_query(query: str) -> Dict[str, Any]:
        """
        Classifies user query into an intent and specifies tools to invoke.
        """
        q = query.lower().strip()

        # 1. Check What-If Sensitivity Questions
        what_if_match = re.search(r"\b(what if|suppose|what happens if|hypothetically)\b", q)
        if what_if_match or (q.startswith("if ") and any(v in q for v in ["changes", "becomes", "is", "drops", "rises", "removed"])):
            extracted_feature, extracted_val = IntentRouter._extract_what_if_params(q)
            if extracted_feature and extracted_val is not None:
                return {
                    "intent": "WHAT_IF_ANALYSIS",
                    "tools": ["run_what_if_prediction", "get_current_patient_context"],
                    "params": {
                        "feature": extracted_feature,
                        "value": extracted_val
                    },
                    "description": f"Sensitivity analysis for {extracted_feature} = {extracted_val}"
                }

        # 2. Check SHAP / Feature Attribution Questions
        if any(k in q for k in ["shap", "waterfall", "feature importance", "top factors", "drivers", "contribution", "which vitals increased"]):
            return {
                "intent": "SHAP_EXPLANATION",
                "tools": ["get_shap_explanation", "get_current_prediction"],
                "params": {},
                "description": "SHAP feature impact and directional contribution analysis"
            }

        # 3. Check Current Patient Acuity / Vitals / Assessment Questions
        if any(k in q for k in ["this patient", "for this patient", "why high", "why critical", "why moderate", "why low", "urgency tier", "current criticality", "patient's current", "vital signs for this patient", "current patient"]):
            return {
                "intent": "PATIENT_EXPLANATION",
                "tools": ["get_current_patient_context", "get_current_prediction", "get_shap_explanation"],
                "params": {},
                "description": "Holistic current patient clinical assessment"
            }

        # 4. Check Model Performance & Metrics Questions
        if any(k in q for k in ["accuracy", "r2", "r^2", "mae", "rmse", "metric", "f1", "recall", "precision", "test set", "reliable", "how reliable", "performance", "under-triage"]):
            return {
                "intent": "MODEL_PERFORMANCE",
                "tools": ["get_model_information"],
                "params": {},
                "description": "Model evaluation metrics and benchmark figures"
            }

        # 5. Check Project Architecture & Engineering Questions
        if any(k in q for k in ["why xgboost", "why synthetic", "why streamlit", "why docker", "architecture", "pipeline work", "decisions", "limitations"]):
            return {
                "intent": "PROJECT_QUESTION",
                "tools": ["get_project_information"],
                "params": {"topic": q},
                "description": "Architectural decision records and system design documentation"
            }

        # 6. Check Clinical Concept / Healthcare Definitions (RAG)
        if any(k in q for k in ["what is gcs", "what does gcs mean", "gcs", "spo2", "shock index", "pulse pressure", "news2", "start triage", "hypoxia", "sepsis", "tachycardia", "hypotension", "vital"]):
            return {
                "intent": "GENERAL_HEALTHCARE",
                "tools": ["query_triage_knowledge_base"],
                "params": {"query": query},
                "description": "RAG retrieval from authoritative clinical knowledge base"
            }

        # Default fallback: Treat as patient or general inquiry
        return {
            "intent": "GENERAL_QUERY",
            "tools": ["get_current_patient_context", "query_triage_knowledge_base"],
            "params": {"query": query},
            "description": "Combined patient context and clinical literature search"
        }

    @staticmethod
    def _extract_what_if_params(query: str) -> Tuple[str, Any]:
        """
        Parses requested feature name and numerical/boolean target value from text.
        """
        q = query.lower()
        feature = ""
        value = None

        # Feature mappings
        if "spo2" in q or "oxygen" in q or "o2" in q:
            feature = "spo2"
        elif "heart rate" in q or "hr" in q or "pulse" in q:
            feature = "heart_rate"
        elif "systolic" in q or "sbp" in q:
            feature = "systolic_bp"
        elif "diastolic" in q or "dbp" in q:
            feature = "diastolic_bp"
        elif "respiratory rate" in q or "breathing rate" in q or "rr" in q:
            feature = "respiratory_rate"
        elif "gcs" in q or "coma scale" in q:
            feature = "gcs"
        elif "temp" in q or "temperature" in q:
            feature = "temperature"
        elif "pain" in q:
            feature = "pain_severity"
        elif "difficulty breathing" in q:
            feature = "difficulty_breathing"
            value = 0 if any(neg in q for neg in ["no", "removed", "resolved", "false", "0"]) else 1
            return feature, value
        elif "chest pain" in q:
            feature = "chest_pain"
            value = 0 if any(neg in q for neg in ["no", "removed", "resolved", "false", "0"]) else 1
            return feature, value

        # Number extraction
        num_matches = re.findall(r"[-+]?\d*\.?\d+", q)
        if num_matches:
            val_str = num_matches[-1]
            value = float(val_str) if "." in val_str else int(val_str)

        return feature, value
