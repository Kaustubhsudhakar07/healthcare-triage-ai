"""
Primary Triage AI Agent Orchestrator
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Implements single-agent architecture with controlled tool calling, Gemini LLM synthesis,
RAG literature grounding, execution tracing, and clinical safety enforcement.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.abspath("."))

from src.agent.safety import ClinicalSafetyGuard
from src.agent.router import IntentRouter
from src.agent.tools import TriageTools
from src.agent.prompts import TRIAGE_AGENT_SYSTEM_PROMPT, build_agent_context_prompt


class TriageAIAgent:
    """
    Primary Triage AI Agent connecting user queries to real ML tools, RAG, and Gemini.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.client = None
        self._init_gemini_client()

    def _init_gemini_client(self):
        """Initializes Google GenAI client if API key is provided."""
        if not self.api_key:
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"Warning: Could not initialize Google GenAI client: {e}")
            self.client = None

    def set_api_key(self, api_key: str):
        """Allows dynamic API key update from UI input."""
        self.api_key = api_key
        self._init_gemini_client()

    def answer_query(
        self,
        query: str,
        current_payload: Optional[Dict[str, Any]] = None,
        current_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end agentic workflow: Pre-Screen -> Route -> Tools -> Gemini -> Post-Process.
        """
        tool_activity_trace = []
        retrieved_sources = []
        t0 = time.perf_counter()

        # Step 1: Safety Pre-Screening
        is_safe, refusal_reason, safety_response = ClinicalSafetyGuard.pre_screen_query(query)
        if not is_safe:
            tool_activity_trace.append(f"Safety Pre-Screen: Triggered [{refusal_reason}]")
            tool_activity_trace.append("Action: Standard Clinical Boundary Refusal Enforced")
            return {
                "answer": safety_response,
                "tool_activity": tool_activity_trace,
                "sources": [],
                "intent": "SAFETY_RESTRICTED",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "is_what_if": False
            }

        tool_activity_trace.append("Safety Pre-Screen: PASSED (Clinical boundaries verified)")

        # Step 2: Intent Routing
        routing_info = IntentRouter.route_query(query)
        intent = routing_info["intent"]
        tools_to_run = routing_info["tools"]
        params = routing_info.get("params", {})
        tool_activity_trace.append(f"Intent Classified: {intent} ({routing_info['description']})")

        # Step 3: Tool Execution
        tools_engine = TriageTools(
            current_payload=current_payload,
            current_prediction=current_prediction
        )
        tool_outputs = {}
        is_what_if = False

        for tool_name in tools_to_run:
            if tool_name == "get_current_patient_context":
                tool_activity_trace.append("Tool Invoked: `get_current_patient_context()` (Read active session vitals)")
                tool_outputs["patient_context"] = tools_engine.get_current_patient_context()

            elif tool_name == "get_current_prediction":
                tool_activity_trace.append("Tool Invoked: `get_current_prediction()` (Read active XGBoost inference)")
                tool_outputs["prediction"] = tools_engine.get_current_prediction()

            elif tool_name == "get_shap_explanation":
                tool_activity_trace.append("Tool Invoked: `get_shap_explanation()` (Retrieved SHAP local attribution factors)")
                tool_outputs["shap_factors"] = tools_engine.get_shap_explanation()

            elif tool_name == "run_what_if_prediction":
                is_what_if = True
                feat = params.get("feature", "spo2")
                val = params.get("value", 95.0)
                tool_activity_trace.append(f"Tool Invoked: `run_what_if_prediction({feat}={val})` [EXECUTING REAL ML PIPELINE]")
                what_if_res = tools_engine.run_what_if_prediction(feat, val)
                tool_outputs["what_if_result"] = what_if_res
                
                if what_if_res.get("status") == "success":
                    delta_txt = what_if_res["delta_description"]
                    tool_activity_trace.append(
                        f"Real ML Output: Base {what_if_res['baseline_criticality']:.1f} -> "
                        f"Hypothetical {what_if_res['hypothetical_criticality']:.1f} ({delta_txt})"
                    )

            elif tool_name == "get_model_information":
                tool_activity_trace.append("Tool Invoked: `get_model_information()` (Read models/model_metadata.json)")
                tool_outputs["model_metadata"] = tools_engine.get_model_information()

            elif tool_name == "get_project_information":
                topic = params.get("topic", "decisions")
                tool_activity_trace.append(f"Tool Invoked: `get_project_information('{topic}')` (Parsed docs/ Markdown)")
                tool_outputs["project_documentation"] = tools_engine.get_project_information(topic)

            elif tool_name == "query_triage_knowledge_base":
                rag_query = params.get("query", query)
                tool_activity_trace.append(f"Tool Invoked: `query_triage_knowledge_base('{rag_query[:40]}...')` [Chroma Vector Search]")
                rag_res = tools_engine.query_triage_knowledge_base(rag_query)
                tool_outputs["rag_literature"] = rag_res

                for c in rag_res.get("chunks", []):
                    retrieved_sources.append({
                        "title": c.get("title", "Clinical Reference"),
                        "source": c.get("source", "Standard Guidelines"),
                        "authority": c.get("authority", "Emergency Protocol")
                    })
                tool_activity_trace.append(f"RAG Retrieved: {len(retrieved_sources)} authoritative literature chunks")

        # Step 4: LLM Generation (Gemini) or Fallback Synthesis
        if self.client:
            try:
                tool_activity_trace.append("LLM Generation: Synthesizing natural language response via Gemini API...")
                context_prompt = build_agent_context_prompt(query, intent, tool_outputs)
                
                # Call Gemini via official modern google.genai SDK
                # Prefer fast, reliable gemini-2.5-flash or gemini-1.5-flash
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        {"role": "user", "parts": [{"text": f"{TRIAGE_AGENT_SYSTEM_PROMPT}\n\n{context_prompt}"}]}
                    ]
                )
                raw_answer = response.text
            except Exception as e:
                tool_activity_trace.append(f"Warning: Gemini API call failed ({e}). Using deterministic fallback synthesis.")
                raw_answer = self._generate_deterministic_fallback(query, intent, tool_outputs, is_what_if)
        else:
            tool_activity_trace.append("LLM Notice: No GEMINI_API_KEY detected. Using grounded deterministic fallback synthesis.")
            raw_answer = self._generate_deterministic_fallback(query, intent, tool_outputs, is_what_if)

        # Step 5: Safety Post-Processing
        final_answer = ClinicalSafetyGuard.post_process_response(raw_answer, is_what_if=is_what_if)
        tool_activity_trace.append("Safety Post-Process: Verified disclaimers & non-diagnostic framing")

        total_latency = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "answer": final_answer,
            "tool_activity": tool_activity_trace,
            "sources": retrieved_sources,
            "intent": intent,
            "latency_ms": total_latency,
            "is_what_if": is_what_if
        }

    def _generate_deterministic_fallback(
        self,
        query: str,
        intent: str,
        tool_outputs: Dict[str, Any],
        is_what_if: bool
    ) -> str:
        """
        Provides accurate, grounded fallback response using exact tool values
        if Gemini API is unavailable or offline.
        """
        if is_what_if and "what_if_result" in tool_outputs:
            res = tool_outputs["what_if_result"]
            if res.get("status") == "success":
                return (
                    f"### What-If Sensitivity Analysis (Real ML Pipeline Execution)\n\n"
                    f"- **Modified Feature**: `{res['feature_modified']}`\n"
                    f"- **Baseline Value**: `{res['baseline_value']}` ➡️ **Hypothetical Value**: `{res['hypothetical_value']}`\n"
                    f"- **Baseline Prediction**: Score **{res['baseline_criticality']:.1f} / 10.0** (`{res['baseline_urgency']}` tier)\n"
                    f"- **Hypothetical Prediction**: Score **{res['hypothetical_criticality']:.1f} / 10.0** (`{res['hypothetical_urgency']}` tier)\n"
                    f"- **Acuity Shift**: **{res['delta_description']}**\n\n"
                    f"Under the model's learned non-linear decision trees, adjusting this vital sign produced a shift of "
                    f"{abs(res['score_delta']):.2f} points on the calibrated 1-10 acuity scale."
                )

        if intent == "SHAP_EXPLANATION" and "shap_factors" in tool_outputs:
            shap = tool_outputs["shap_factors"]
            factors_md = "\n".join([f"- **{f['feature']}**: {f['shap_impact']:+.2f} points ({f['direction']})" for f in shap.get("top_contributing_factors", [])[:5]])
            return (
                f"### SHAP Local Attribution Breakdown\n\n"
                f"The model's baseline expected score across the population is **{shap.get('base_expected_value', 5.0):.1f}**.\n\n"
                f"For this patient, the top physiological drivers shifting the prediction are:\n{factors_md}\n\n"
                f"**Clinical Narrative Summary**:\n" +
                "\n".join([f"- {n}" for n in shap.get("clinical_narratives", [])[:3]])
            )

        if intent == "MODEL_PERFORMANCE" and "model_metadata" in tool_outputs:
            m = tool_outputs["model_metadata"]
            return (
                f"### Model Architecture & Test Evaluation Benchmarks\n\n"
                f"- **Model**: {m.get('model_name')}\n"
                f"- **R² Goodness of Fit**: **{m.get('r2_score')}**\n"
                f"- **Mean Absolute Error (MAE)**: **{m.get('mae')}** points\n"
                f"- **5-Tier Exact Accuracy**: **{m.get('exact_tier_accuracy')}**\n"
                f"- **Critical Tier Recall**: **{m.get('critical_tier_recall')}**\n"
                f"- **Severe Under-Triage Rate**: **{m.get('severe_under_triage_rate')}**\n"
                f"- **Evaluation Cohort**: Stratified held-out test partition ($N=2,000$)"
            )

        if intent == "GENERAL_HEALTHCARE" and "rag_literature" in tool_outputs:
            chunks = tool_outputs["rag_literature"].get("chunks", [])
            if chunks:
                primary = chunks[0]
                return (
                    f"### Clinical Reference: {primary.get('title')}\n\n"
                    f"{primary.get('content')}\n\n"
                    f"*Source: {primary.get('source')} ({primary.get('authority')})*"
                )

        if "patient_context" in tool_outputs:
            ctx = tool_outputs["patient_context"]
            pred = tool_outputs.get("prediction", {})
            return (
                f"### Current Patient Clinical Assessment\n\n"
                f"{ctx.get('summary_text', '')}\n"
                f"The estimated acuity score is **{pred.get('criticality_score', 'N/A')}/10.0** (`{pred.get('urgency_tier', 'N/A')}`). "
                f"Recommended routing: {pred.get('clinical_routing_guidance', 'Standard intake')}."
            )

        return (
            "I have analyzed your inquiry. Please refer to the current patient telemetry, "
            "SHAP factor attribution charts, or model benchmark audits on the dashboard."
        )
