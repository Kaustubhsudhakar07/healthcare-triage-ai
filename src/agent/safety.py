"""
Clinical Safety Guardrails & Adversarial Defense Layer
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Enforces clinical boundaries:
- Intercepts requests for medical diagnosis, drug prescriptions, and dosages
- Prohibits false reassurance or advice to bypass emergency care
- Defends against prompt injection and natural-language model state manipulation
- Appends mandatory clinical disclaimers to model sensitivity analyses
"""

import re
from typing import Dict, Any, Tuple, Optional


DISCLAIMER_TEXT = (
    "\n\n*Clinical Disclaimer: This system is an AI-assisted decision-support prototype. "
    "Predictions and what-if sensitivity analyses reflect statistical patterns in synthetic data "
    "and must never replace clinical judgment, protocolized resuscitation, or physical physician assessment.*"
)

# Regex patterns for clinical safety violations
DIAGNOSIS_PATTERNS = [
    r"\b(diagnose|diagnosis|what disease|what condition does|what is wrong with|what is the illness|cure)\b",
    r"\b(do (i|they|he|she) have (a|an|the)?)\b",
    r"\b(stroke|heart attack|myocardial infarction|pulmonary embolism|pneumonia|appendicitis|cancer|covid)\b"
]

PRESCRIPTION_PATTERNS = [
    r"\b(prescribe|prescription|medication|medicine|what drug|dosage|dose|how many mg|administer)\b",
    r"\b(give (him|her|them|me)?.*\b(aspirin|morphine|epinephrine|antibiotic|antibiotics|insulin|pill|pills|drug|drugs))\b",
    r"\b(treat with|what pill|which drug|cure this patient)\b"
]

AVOID_CARE_PATTERNS = [
    r"\b(stay home|avoid.*hospital|skip.*hospital|cancel (the)? ambulance|delay.*hospital|avoid care)\b",
    r"\b(do (i|we) really need (an)? ambulance|is it safe to not go|is it safe for the patient to stay home)\b",
    r"\b(safe to stay home|safe not to go|avoid going to)\b"
]

ADVERSARIAL_INJECTION_PATTERNS = [
    r"\b(ignore (all|previous|your) instructions|system prompt|override (the)? model|change (the)? prediction|force (the)? score)\b",
    r"\b(make (the)? patient (safe|low risk)|jailbreak|dan mode|bypass safety|output confidential)\b",
    r"\b(act as|pretend to be|forget (all)? rules)\b"
]


class ClinicalSafetyGuard:
    """
    Pre- and post-processing safety guardrails for the Triage AI Assistant.
    """

    @staticmethod
    def pre_screen_query(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Screens incoming user queries against clinical boundaries and adversarial injections.
        Returns: (is_safe, refusal_reason, standard_response)
        """
        q_lower = query.lower().strip()

        # 1. Check Adversarial Prompt Injections
        for pattern in ADVERSARIAL_INJECTION_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    False,
                    "ADVERSARIAL_PROMPT_INJECTION",
                    "I cannot alter system instructions, bypass safety rules, or arbitrarily modify the model's prediction score. "
                    "All predictions are generated directly by the underlying machine learning pipeline based on actual patient telemetry."
                )

        # 2. Check Diagnosis Inquiries
        for pattern in DIAGNOSIS_PATTERNS:
            if re.search(pattern, q_lower):
                # Allow general knowledge queries like "What is the diagnosis criteria for sepsis?"
                if not any(k in q_lower for k in ["criteria", "definition", "literature", "guideline", "what is"]):
                    return (
                        False,
                        "MEDICAL_DIAGNOSIS_PROHIBITED",
                        "As a pre-hospital triage decision-support system, I am not authorized or designed to diagnose medical conditions or diseases. "
                        "My role is strictly limited to estimating physiological acuity and routing urgency. "
                        "A definitive medical diagnosis requires clinical examination, diagnostic imaging, and laboratory testing by a licensed physician."
                    )

        # 3. Check Prescription / Medication / Dosage Requests
        for pattern in PRESCRIPTION_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    False,
                    "PRESCRIPTION_OR_DOSAGE_PROHIBITED",
                    "I cannot prescribe medications, suggest pharmaceutical treatments, or calculate drug dosages. "
                    "All pharmacotherapy must be ordered by qualified medical directors or emergency physicians in accordance with local EMS clinical protocols."
                )

        # 4. Check Avoidance of Emergency Hospital Care
        for pattern in AVOID_CARE_PATTERNS:
            if re.search(pattern, q_lower):
                return (
                    False,
                    "AVOID_CARE_PROHIBITED",
                    "I cannot advise anyone to avoid emergency hospital care or cancel an ambulance. "
                    "Pre-hospital triage tools are designed for acute prioritization, not discharge safety. "
                    "Anyone experiencing acute symptoms, abnormal vitals, or distress must be evaluated in person by emergency healthcare professionals."
                )

        return (True, None, None)

    @staticmethod
    def post_process_response(response_text: str, is_what_if: bool = False) -> str:
        """
        Appends mandatory safety reminders and sanitizes output.
        """
        # Strip any simulated internal thinking tokens if present
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        
        if is_what_if:
            if "model sensitivity analysis" not in cleaned.lower():
                cleaned += (
                    "\n\n*Note: This is a model sensitivity analysis examining learned mathematical correlations. "
                    "It is not a clinical treatment recommendation or physiological prognosis.*"
                )

        if "Clinical Disclaimer:" not in cleaned:
            cleaned += DISCLAIMER_TEXT

        return cleaned
