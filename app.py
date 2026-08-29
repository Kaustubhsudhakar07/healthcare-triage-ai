"""
AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
Interactive Streamlit Web Dashboard
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.abspath("."))

from src.predict import ClinicalInferenceService, PatientPayload
from src.preprocessing import score_to_urgency_tier, clip_criticality_scores, URGENCY_ORDER
from src.monitoring import ClinicalDriftMonitor, calculate_psi
from src.agent.agent import TriageAIAgent

# Page configuration
st.set_page_config(
    page_title="Pre-Hospital Criticality & Triage AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Mode / Glassmorphism Medical Dashboard
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }
    
    /* Metric Score Badge */
    .score-badge {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1;
        margin-bottom: 8px;
    }
    
    /* Triage Tier Banners */
    .tier-low {
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid #2ecc71;
        color: #2ecc71;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .tier-moderate {
        background: rgba(241, 196, 15, 0.15);
        border: 1px solid #f1c40f;
        color: #f1c40f;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .tier-elevated {
        background: rgba(230, 126, 34, 0.15);
        border: 1px solid #e67e22;
        color: #e67e22;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .tier-high {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid #e74c3c;
        color: #e74c3c;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .tier-critical {
        background: rgba(155, 89, 182, 0.2);
        border: 2px solid #9b59b6;
        color: #d7bde2;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 0 15px rgba(155, 89, 182, 0.4);
    }
    
    /* Card Container */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(8px);
    }
    
    /* Red Flag Box */
    .red-flag-box {
        background: rgba(220, 38, 38, 0.15);
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        color: #fca5a5;
        font-size: 0.95rem;
    }
    
    /* Footer Disclaimer */
    .disclaimer-text {
        font-size: 0.82rem;
        color: #94a3b8;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 16px;
        margin-top: 32px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inference_service():
    """Cached singleton for inference engine."""
    return ClinicalInferenceService(
        pipeline_path="models/pipeline.joblib",
        train_path="data/processed/train.csv"
    )


# Presets for Rapid Field Simulation
PRESETS = {
    "Custom / Manual Entry": None,
    "Severe Cardiogenic Shock (Code 1)": {
        "age": 72, "sex": "Male", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 1, "chest_pain": 1, "difficulty_breathing": 1,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 0,
        "headache": 0, "vomiting": 0, "heart_rate": 132.0, "systolic_bp": 78.0,
        "diastolic_bp": 45.0, "spo2": 84.0, "respiratory_rate": 32.0, "temperature": 36.4,
        "gcs": 9, "pain_severity": 9, "oxygen_requirement": 1,
        "known_cardiac_history": 1, "known_hypertension": 1, "known_diabetes": 1
    },
    "Severe Hypoxemic Respiratory Failure (COPD/Asthma)": {
        "age": 64, "sex": "Female", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 0, "chest_pain": 0, "difficulty_breathing": 1,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 1,
        "headache": 0, "vomiting": 0, "heart_rate": 115.0, "systolic_bp": 140.0,
        "diastolic_bp": 85.0, "spo2": 76.0, "respiratory_rate": 36.0, "temperature": 38.6,
        "gcs": 14, "pain_severity": 5, "oxygen_requirement": 1,
        "known_cardiac_history": 0, "known_hypertension": 1, "known_diabetes": 0
    },
    "Major Polytrauma & Hemorrhagic Shock": {
        "age": 32, "sex": "Male", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 1, "chest_pain": 0, "difficulty_breathing": 1,
        "abdominal_pain": 1, "injury_trauma": 1, "bleeding": 1, "fever": 0,
        "headache": 1, "vomiting": 0, "heart_rate": 145.0, "systolic_bp": 72.0,
        "diastolic_bp": 40.0, "spo2": 89.0, "respiratory_rate": 28.0, "temperature": 35.2,
        "gcs": 7, "pain_severity": 10, "oxygen_requirement": 1,
        "known_cardiac_history": 0, "known_hypertension": 0, "known_diabetes": 0
    },
    "Geriatric Sepsis with Altered Mental Status": {
        "age": 84, "sex": "Female", "ambulance_arrival": 1, "walking_ability": 0,
        "altered_consciousness": 1, "chest_pain": 0, "difficulty_breathing": 1,
        "abdominal_pain": 0, "injury_trauma": 0, "bleeding": 0, "fever": 1,
        "headache": 0, "vomiting": 1, "heart_rate": 122.0, "systolic_bp": 88.0,
        "diastolic_bp": 50.0, "spo2": 91.0, "respiratory_rate": 26.0, "temperature": 39.4,
        "gcs": 11, "pain_severity": 4, "oxygen_requirement": 1,
        "known_cardiac_history": 1, "known_hypertension": 1, "known_diabetes": 1
    },
    "Stable Ambulatory Walk-in (Low Acuity)": {
        "age": 28, "sex": "Female", "ambulance_arrival": 0, "walking_ability": 1,
        "altered_consciousness": 0, "chest_pain": 0, "difficulty_breathing": 0,
        "abdominal_pain": 1, "injury_trauma": 0, "bleeding": 0, "fever": 0,
        "headache": 0, "vomiting": 0, "heart_rate": 72.0, "systolic_bp": 118.0,
        "diastolic_bp": 76.0, "spo2": 99.0, "respiratory_rate": 14.0, "temperature": 36.8,
        "gcs": 15, "pain_severity": 3, "oxygen_requirement": 0,
        "known_cardiac_history": 0, "known_hypertension": 0, "known_diabetes": 0
    }
}


def render_gauge_chart(score: float, tier: str) -> go.Figure:
    """Renders a modern semicircular gauge for criticality score."""
    colors = {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "Elevated": "#e67e22",
        "High": "#e74c3c",
        "Critical": "#9b59b6"
    }
    tier_color = colors.get(tier, "#3498db")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Acuity Score: {tier.upper()}", 'font': {'size': 20, 'color': tier_color}},
        number={'font': {'size': 44, 'color': "#ffffff"}, 'suffix': " / 10"},
        gauge={
            'axis': {'range': [1.0, 10.0], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': tier_color, 'thickness': 0.3},
            'bgcolor': "rgba(30, 41, 59, 0.6)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [1.0, 2.5], 'color': 'rgba(46, 204, 113, 0.2)'},
                {'range': [2.5, 4.5], 'color': 'rgba(241, 196, 15, 0.2)'},
                {'range': [4.5, 6.5], 'color': 'rgba(230, 126, 34, 0.2)'},
                {'range': [6.5, 8.5], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [8.5, 10.0], 'color': 'rgba(155, 89, 182, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260
    )
    return fig


def render_shap_waterfall_chart(factors: list) -> go.Figure:
    """Renders horizontal bar chart of top SHAP contributors."""
    df_factors = pd.DataFrame(factors[:8])
    df_factors["color"] = df_factors["shap_impact"].apply(lambda x: "#ef4444" if x > 0 else "#10b981")
    df_factors = df_factors.iloc[::-1]  # Invert order for top on top
    
    fig = go.Figure(go.Bar(
        x=df_factors["shap_impact"],
        y=df_factors["feature"],
        orientation='h',
        marker_color=df_factors["color"],
        text=df_factors["shap_impact"].apply(lambda x: f"{'+' if x>0 else ''}{x:.2f}"),
        textposition="auto"
    ))
    
    fig.update_layout(
        title="Top Physiological Drivers (SHAP Local Attribution)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Impact on Criticality Score (Points)",
            zeroline=True,
            zerolinecolor="#64748b",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            tickfont=dict(color="#e2e8f0")
        ),
        margin=dict(l=10, r=20, t=40, b=30),
        height=320
    )
    return fig


def main():
    # Header Banner
    st.markdown("""
    <div class="header-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2.1rem; color: #f8fafc; font-weight: 800;">
                    🚑 Pre-Hospital Criticality & Triage AI
                </h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                    AI-Assisted Emergency Acuity Prediction & Clinical Decision Support System
                </p>
            </div>
            <div style="text-align: right;">
                <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #93c5fd; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                    Model: Tuned XGBoost Pipeline (R²: 0.994)
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize services
    try:
        service = load_inference_service()
    except Exception as e:
        st.error(f"Failed to load clinical inference service: {e}")
        st.stop()

    @st.cache_resource
    def load_triage_agent():
        return TriageAIAgent()

    agent = load_triage_agent()

    # Sidebar: Optional Gemini API Key configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Triage AI Assistant")
    gemini_key_input = st.sidebar.text_input(
        "Google Gemini API Key (Optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Provide your Gemini API key to enable live Generative AI synthesis. If left blank, grounded deterministic fallback synthesis will be used."
    )
    if gemini_key_input:
        os.environ["GEMINI_API_KEY"] = gemini_key_input
        agent.set_api_key(gemini_key_input)

    # Navigation Tabs
    tab_triage, tab_agent, tab_queue, tab_benchmarks, tab_drift, tab_about = st.tabs([
        "🩺 Live Field Patient Triage",
        "🤖 Triage AI Assistant",
        "📋 Dispatch & Ambulance Queue",
        "📊 Model Benchmarks & Safety Audits",
        "📡 Real-Time Drift & Telemetry Monitor",
        "📖 Scope & Clinical Guidance"
    ])
    
    # -------------------------------------------------------------
    # TAB 1: LIVE FIELD TRIAGE
    # -------------------------------------------------------------
    with tab_triage:
        col_input, col_output = st.columns([1.15, 1.0], gap="large")
        
        with col_input:
            st.subheader("Field Patient Observation Intake")
            
            # Quick Presets Selector
            selected_preset_name = st.selectbox(
                "⚡ Quick Clinical Presets (Demo Scenarios):",
                list(PRESETS.keys()),
                index=0
            )
            preset_vals = PRESETS[selected_preset_name] or {}
            
            # Input Form Sections
            with st.expander("👤 1. Demographics & Transport Mode", expanded=True):
                c1, c2, c3 = st.columns(3)
                age = c1.slider("Age (Years)", 16, 95, value=preset_vals.get("age", 58))
                sex = c2.selectbox("Biological Sex", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(preset_vals.get("sex", "Male")))
                ambulance_arrival = c3.selectbox("Arrival Mode", [("EMS / Ambulance", 1), ("Walk-in / Private", 0)], format_func=lambda x: x[0], index=0 if preset_vals.get("ambulance_arrival", 1) == 1 else 1)[1]
                
            with st.expander("🩺 2. Field Vital Signs", expanded=True):
                v1, v2, v3 = st.columns(3)
                spo2 = v1.slider("SpO2 Saturation (%)", 50.0, 100.0, value=float(preset_vals.get("spo2", 96.0)), step=0.5)
                heart_rate = v2.slider("Heart Rate (bpm)", 30.0, 220.0, value=float(preset_vals.get("heart_rate", 82.0)), step=1.0)
                systolic_bp = v3.slider("Systolic BP (mmHg)", 50.0, 240.0, value=float(preset_vals.get("systolic_bp", 124.0)), step=1.0)
                
                v4, v5, v6 = st.columns(3)
                # Enforce DBP < SBP constraint
                max_dbp = min(140.0, systolic_bp - 5.0)
                diastolic_bp = v4.slider("Diastolic BP (mmHg)", 30.0, max_dbp, value=float(min(preset_vals.get("diastolic_bp", 78.0), max_dbp)), step=1.0)
                respiratory_rate = v5.slider("Resp Rate (breaths/min)", 8.0, 50.0, value=float(preset_vals.get("respiratory_rate", 16.0)), step=1.0)
                temperature = v6.slider("Temperature (°C)", 34.0, 42.0, value=float(preset_vals.get("temperature", 37.0)), step=0.1)
                
                v7, v8, v9 = st.columns(3)
                gcs = v7.slider("Glasgow Coma Scale (GCS)", 3, 15, value=int(preset_vals.get("gcs", 15)))
                pain_severity = v8.slider("Pain Score (0–10 NRS)", 0, 10, value=int(preset_vals.get("pain_severity", 2)))
                oxygen_requirement = v9.checkbox("Pre-Hospital O2 Administered", value=bool(preset_vals.get("oxygen_requirement", 0)))
                
            with st.expander("⚠️ 3. Observable Symptoms & Clinical Red Flags", expanded=True):
                s1, s2, s3 = st.columns(3)
                walking_ability = s1.checkbox("Ambulatory (Can Walk)", value=bool(preset_vals.get("walking_ability", 1)))
                altered_consciousness = s2.checkbox("Altered Mental State", value=bool(preset_vals.get("altered_consciousness", 0)))
                difficulty_breathing = s3.checkbox("Difficulty Breathing", value=bool(preset_vals.get("difficulty_breathing", 0)))
                
                s4, s5, s6 = st.columns(3)
                chest_pain = s4.checkbox("Chest Pain", value=bool(preset_vals.get("chest_pain", 0)))
                injury_trauma = s5.checkbox("Physical Trauma", value=bool(preset_vals.get("injury_trauma", 0)))
                bleeding = s6.checkbox("Active Bleeding", value=bool(preset_vals.get("bleeding", 0)))
                
                s7, s8, s9 = st.columns(3)
                abdominal_pain = s7.checkbox("Abdominal Pain", value=bool(preset_vals.get("abdominal_pain", 0)))
                vomiting = s8.checkbox("Vomiting / Nausea", value=bool(preset_vals.get("vomiting", 0)))
                fever = s9.checkbox("Subjective Fever", value=bool(preset_vals.get("fever", 0)))
                
                s10, s11, s12 = st.columns(3)
                headache = s10.checkbox("Severe Headache", value=bool(preset_vals.get("headache", 0)))
                known_cardiac_history = s11.checkbox("Known Cardiac History", value=bool(preset_vals.get("known_cardiac_history", 0)))
                known_hypertension = s12.checkbox("Known Hypertension", value=bool(preset_vals.get("known_hypertension", 0)))
                known_diabetes = st.checkbox("Known Diabetes Mellitus", value=bool(preset_vals.get("known_diabetes", 0)))
                
            # Build payload dict
            payload = {
                "age": age, "sex": sex, "ambulance_arrival": int(ambulance_arrival),
                "walking_ability": int(walking_ability), "altered_consciousness": int(altered_consciousness),
                "chest_pain": int(chest_pain), "difficulty_breathing": int(difficulty_breathing),
                "abdominal_pain": int(abdominal_pain), "injury_trauma": int(injury_trauma),
                "bleeding": int(bleeding), "fever": int(fever), "headache": int(headache),
                "vomiting": int(vomiting), "heart_rate": heart_rate, "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp, "spo2": spo2, "respiratory_rate": respiratory_rate,
                "temperature": temperature, "gcs": gcs, "pain_severity": pain_severity,
                "oxygen_requirement": int(oxygen_requirement), "known_cardiac_history": int(known_cardiac_history),
                "known_hypertension": int(known_hypertension), "known_diabetes": int(known_diabetes)
            }
            
        with col_output:
            st.subheader("AI Decision-Support Output")
            
            # Run prediction
            try:
                result = service.predict(payload)
                st.session_state["current_patient_payload"] = payload
                st.session_state["current_prediction_result"] = result
            except Exception as ex:
                st.error(f"Inference error: {ex}")
                st.stop()
                
            score = result["criticality_score"]
            tier = result["urgency_tier"]
            
            # Gauge & Tier Display
            st.plotly_chart(render_gauge_chart(score, tier), use_container_width=True)
            
            tier_classes = {
                "Low": "tier-low",
                "Moderate": "tier-moderate",
                "Elevated": "tier-elevated",
                "High": "tier-high",
                "Critical": "tier-critical"
            }
            badge_class = tier_classes.get(tier, "tier-moderate")
            
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 16px;">
                <span class="{badge_class}">
                    Operational Urgency: {tier.upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Safety Red Flag Alerts
            if result["red_flags"]:
                st.markdown("#### 🚨 Critical Safety Guardrail Triggers")
                for rf in result["red_flags"]:
                    st.markdown(f'<div class="red-flag-box">⚠️ <b>{rf}</b></div>', unsafe_allow_html=True)
                    
            if result.get("safety_override_applied"):
                st.warning("⚠️ **Safety Override Active:** Ground-truth safety flags triggered an emergency tier elevation.")
                
            # Clinical Routing Recommendation
            st.markdown("#### 🏥 Recommended Emergency Bay Routing")
            st.info(result["clinical_routing_guidance"])
            
            # SHAP Local Explainability
            if result.get("explanation") and "top_factors" in result["explanation"]:
                st.markdown("#### 🔍 Explainable AI (Local Factor Attribution)")
                st.plotly_chart(render_shap_waterfall_chart(result["explanation"]["top_factors"]), use_container_width=True)
                
                with st.expander("📝 View Clinical Narrative Breakdown", expanded=False):
                    for narr in result["explanation"]["narrative"]:
                        st.markdown(f"- {narr}")

    # -------------------------------------------------------------
    # TAB 2: TRIAGE AI ASSISTANT
    # -------------------------------------------------------------
    with tab_agent:
        st.subheader("🤖 Triage AI Decision-Support Assistant")
        st.markdown("Conversational agent powered by **Google Gemini**, **Real ML What-If Sensitivity**, **SHAP attributions**, and **Chroma RAG** literature grounding.")

        active_payload = st.session_state.get("current_patient_payload", PRESETS["Severe Cardiogenic Shock (Code 1)"])
        active_result = st.session_state.get("current_prediction_result")
        if not active_result:
            active_result = service.predict(active_payload)

        score = active_result.get("criticality_score", 5.0)
        tier = active_result.get("urgency_tier", "Moderate")
        tier_color = {"Low": "#2ecc71", "Moderate": "#f1c40f", "Elevated": "#e67e22", "High": "#e74c3c", "Critical": "#9b59b6"}.get(tier, "#3498db")

        # Current Patient Assessment Card
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em;">Active Patient Assessment</span>
                    <h3 style="margin: 4px 0; color: #f8fafc;">{active_payload.get('age', 50)}-year-old {active_payload.get('sex', 'Patient')} ({('Ambulance Arrival' if active_payload.get('ambulance_arrival') else 'Walk-in')})</h3>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">
                        HR: <b>{active_payload.get('heart_rate', 80):.0f}</b> bpm | BP: <b>{active_payload.get('systolic_bp', 120):.0f}/{active_payload.get('diastolic_bp', 80):.0f}</b> | SpO2: <b>{active_payload.get('spo2', 98):.0f}%</b> | GCS: <b>{active_payload.get('gcs', 15)}/15</b> | RR: <b>{active_payload.get('respiratory_rate', 16):.0f}</b>
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: {tier_color};">{score:.1f} <span style="font-size: 1.1rem; color: #94a3b8;">/ 10</span></div>
                    <span style="background: {tier_color}33; color: {tier_color}; border: 1px solid {tier_color}; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 0.8rem;">{tier.upper()}</span>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">Model: Tuned XGBoost Pipeline</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested Questions (Clickable chips)
        st.markdown("##### 💡 Suggested Questions (Click to Send):")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            if st.button("❓ Why is this patient high risk?", key="sq_high", use_container_width=True):
                st.session_state["queued_user_query"] = "Why is this patient high risk?"
        with s_col2:
            if st.button("📊 Explain SHAP results", key="sq_shap", use_container_width=True):
                st.session_state["queued_user_query"] = "Explain the SHAP results"
        with s_col3:
            if st.button("🔄 What if SpO2 = 95%?", key="sq_whatif", use_container_width=True):
                st.session_state["queued_user_query"] = "What if SpO2 changes to 95%?"
        with s_col4:
            if st.button("📖 What does GCS mean?", key="sq_gcs", use_container_width=True):
                st.session_state["queued_user_query"] = "What does GCS mean?"

        s_col5, s_col6, s_col7 = st.columns(3)
        with s_col5:
            if st.button("🎯 How reliable is this prediction?", key="sq_rel", use_container_width=True):
                st.session_state["queued_user_query"] = "How reliable is this prediction?"
        with s_col6:
            if st.button("⚙️ Why was XGBoost chosen?", key="sq_xgb", use_container_width=True):
                st.session_state["queued_user_query"] = "Why did you choose XGBoost for this system?"
        with s_col7:
            if st.button("⚠️ What are model limitations?", key="sq_lim", use_container_width=True):
                st.session_state["queued_user_query"] = "What are the limitations of the model?"

        # Chat history container
        if "triage_chat_messages" not in st.session_state:
            st.session_state["triage_chat_messages"] = [
                {
                    "role": "assistant",
                    "content": f"Hello! I am your **Triage AI Assistant**. I can analyze the active patient's vitals (Criticality: **{score:.1f}/10**, Tier: **{tier}**), explain SHAP factors, run real ML what-if simulations, and query clinical literature. How can I help you?",
                    "tool_activity": ["Session Initialized: Triage AI Agent Ready"],
                    "sources": []
                }
            ]

        st.markdown("---")
        st.markdown("##### 💬 Clinical Triage Dialogue")

        for msg in st.session_state["triage_chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_activity"):
                    with st.expander("🔎 AI Tool Activity", expanded=False):
                        for step in msg["tool_activity"]:
                            st.code(step, language="text")
                if msg.get("sources"):
                    st.markdown("**📚 Retrieved Authoritative Citations:**")
                    for src in msg["sources"]:
                        st.markdown(f"- **{src.get('title')}**: *{src.get('source')}* ({src.get('authority')})")

        # Chat Input
        user_input = st.chat_input("Ask a clinical, what-if, or architectural question...")
        query_to_run = user_input or st.session_state.pop("queued_user_query", None)

        if query_to_run:
            st.session_state["triage_chat_messages"].append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing telemetry, running tools, and synthesizing grounded response..."):
                    if gemini_key_input:
                        agent.set_api_key(gemini_key_input)
                    response = agent.answer_query(
                        query=query_to_run,
                        current_payload=active_payload,
                        current_prediction=active_result
                    )
                    st.markdown(response["answer"])
                    if response.get("tool_activity"):
                        with st.expander("🔎 AI Tool Activity", expanded=True):
                            for step in response["tool_activity"]:
                                st.code(step, language="text")
                    if response.get("sources"):
                        st.markdown("**📚 Retrieved Authoritative Citations:**")
                        for src in response["sources"]:
                            st.markdown(f"- **{src.get('title')}**: *{src.get('source')}* ({src.get('authority')})")

            st.session_state["triage_chat_messages"].append({
                "role": "assistant",
                "content": response["answer"],
                "tool_activity": response.get("tool_activity", []),
                "sources": response.get("sources", [])
            })
            st.rerun()

    # -------------------------------------------------------------
    # TAB 3: DISPATCH & AMBULANCE QUEUE
    # -------------------------------------------------------------
    with tab_queue:
        st.subheader("🚑 Active Inbound Emergency Transit Queue")
        st.markdown("Simulated live feed of inbound emergency transports with automated AI priority ranking.")
        
        # Load sample from test data
        test_df = pd.read_csv("data/processed/test.csv").head(15).copy()
        
        # Batch inference
        batch_scores = []
        batch_tiers = []
        for _, row in test_df.iterrows():
            row_dict = row.to_dict()
            res = service.predict(row_dict)
            batch_scores.append(res["criticality_score"])
            batch_tiers.append(res["urgency_tier"])
            
        test_df["Estimated_Acuity"] = batch_scores
        test_df["Urgency_Tier"] = batch_tiers
        
        # Sort by urgency
        tier_weight = {"Critical": 5, "High": 4, "Elevated": 3, "Moderate": 2, "Low": 1}
        test_df["Priority_Rank"] = test_df["Urgency_Tier"].map(tier_weight)
        queue_df = test_df.sort_values(by=["Priority_Rank", "Estimated_Acuity"], ascending=[False, False]).reset_index(drop=True)
        
        display_cols = ["patient_id", "age", "sex", "heart_rate", "systolic_bp", "spo2", "gcs", "Estimated_Acuity", "Urgency_Tier"]
        
        st.dataframe(
            queue_df[display_cols].style.background_gradient(
                subset=["Estimated_Acuity"], cmap="YlOrRd"
            ),
            use_container_width=True
        )
        
        st.download_button(
            "📥 Export Dispatch Queue to CSV",
            queue_df[display_cols].to_csv(index=False),
            file_name="inbound_triage_queue.csv",
            mime="text/csv"
        )

    # -------------------------------------------------------------
    # TAB 3: MODEL BENCHMARKS & AUDITS
    # -------------------------------------------------------------
    with tab_benchmarks:
        st.subheader("📊 Rigorous Multi-Model Benchmarking & Safety Evaluation")
        
        if os.path.exists("experiments/benchmark_results.csv"):
            bench_df = pd.read_csv("experiments/benchmark_results.csv")
            st.dataframe(bench_df, use_container_width=True)
            
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            if os.path.exists("reports/figures/confusion_matrix.png"):
                st.image("reports/figures/confusion_matrix.png", caption="5-Tier Urgency Confusion Matrix")
            if os.path.exists("reports/figures/predicted_vs_actual.png"):
                st.image("reports/figures/predicted_vs_actual.png", caption="Predicted vs Actual Acuity")
                
        with col_fig2:
            if os.path.exists("reports/figures/under_triage_by_acuity.png"):
                st.image("reports/figures/under_triage_by_acuity.png", caption="Under-Triage & Accuracy by Tier")
            if os.path.exists("reports/figures/shap_summary_beeswarm.png"):
                st.image("reports/figures/shap_summary_beeswarm.png", caption="Global SHAP Beeswarm Feature Impact")

    # -------------------------------------------------------------
    # TAB 4: REAL-TIME DATA DRIFT & TELEMETRY MONITOR
    # -------------------------------------------------------------
    with tab_drift:
        st.subheader("📡 Clinical Data Drift & Population Stability Monitoring")
        st.markdown("Real-time telemetry audit tracking distribution shifts between historical baseline training data and active field patient batches using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests.")
        
        if os.path.exists("data/processed/train.csv") and os.path.exists("data/processed/test.csv"):
            train_baseline = pd.read_csv("data/processed/train.csv")
            test_current = pd.read_csv("data/processed/test.csv")
            
            c_drift1, c_drift2 = st.columns([1, 2], gap="large")
            with c_drift1:
                st.markdown("#### ⚙️ Audit Controls")
                drift_scenario = st.selectbox(
                    "Select Telemetry Stream Scenario:",
                    ["Standard Operational Flow (Clean Test Cohort)", "Simulated Severe Hypoxemic Wave (SpO2 Drop -15%)", "Simulated Trauma Crisis Surge (Heart Rate +40 bpm)"]
                )
                
                # Apply scenario modifications
                eval_df = test_current.copy()
                if "Hypoxemic" in drift_scenario:
                    eval_df["spo2"] = np.clip(eval_df["spo2"] - 15.0, 50.0, 100.0)
                elif "Trauma" in drift_scenario:
                    eval_df["heart_rate"] = np.clip(eval_df["heart_rate"] + 40.0, 30.0, 220.0)
                    
                monitor = ClinicalDriftMonitor(train_baseline)
                drift_results = monitor.audit_feature_drift(eval_df)
                
                status_color = {"STABLE": "#2ecc71", "WARNING": "#f1c40f", "ACTION_REQUIRED": "#ef4444"}.get(drift_results["overall_status"], "#3498db")
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.6); border: 2px solid {status_color}; border-radius: 10px; padding: 16px; margin: 16px 0; text-align: center;">
                    <div style="color: #94a3b8; font-size: 0.9rem;">POPULATION STABILITY STATUS</div>
                    <div style="color: {status_color}; font-size: 1.6rem; font-weight: 800;">{drift_results['overall_status']}</div>
                    <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 4px;">Drifted Features: {drift_results['drifted_feature_count']} / {drift_results['total_features_evaluated']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_drift2:
                st.markdown("#### 📊 Feature-Level PSI Breakdown")
                psi_data = []
                for feat, details in drift_results["details"].items():
                    psi_data.append({
                        "Feature": feat.replace("_", " ").title(),
                        "PSI": details["psi"],
                        "KS p-value": details["p_value"],
                        "Status": details["status"]
                    })
                df_psi = pd.DataFrame(psi_data)
                
                fig_psi = px.bar(
                    df_psi, x="PSI", y="Feature", orientation="h",
                    color="Status",
                    color_discrete_map={"STABLE": "#2ecc71", "WARNING": "#f1c40f", "DRIFT_DETECTED": "#ef4444"},
                    title="Population Stability Index (PSI) per Physiological Vital"
                )
                fig_psi.add_vline(x=0.10, line_dash="dash", line_color="#f1c40f", annotation_text="Warning (0.10)")
                fig_psi.add_vline(x=0.20, line_dash="dash", line_color="#ef4444", annotation_text="Drift (0.20)")
                fig_psi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, font=dict(color="#e2e8f0"))
                st.plotly_chart(fig_psi, use_container_width=True)
                
            # Distribution Comparison Plot
            st.markdown("#### 📈 Distribution Shift Deep-Dive")
            selected_feat = st.selectbox("Select Vital to Compare Distributions:", [f for f in ["spo2", "heart_rate", "systolic_bp", "respiratory_rate", "gcs", "temperature"] if f in eval_df.columns])
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=train_baseline[selected_feat], name="Baseline (Train)", opacity=0.6, marker_color="#3b82f6", nbinsx=30))
            fig_dist.add_trace(go.Histogram(x=eval_df[selected_feat], name="Active Telemetry Stream", opacity=0.6, marker_color="#ef4444", nbinsx=30))
            fig_dist.update_layout(
                barmode="overlay",
                title=f"Distribution Comparison: Baseline vs. Active Telemetry ({selected_feat.upper()})",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: SCOPE & CLINICAL GUIDANCE
    # -------------------------------------------------------------
    with tab_about:
        st.subheader("📖 Project Scope, Boundaries & Clinical Safety Rationale")
        st.markdown("""
        ### Intended Purpose & Operational Role
        This application is an **AI-Assisted Emergency Decision-Support Prototype** designed for pre-hospital triage prioritization.
        
        #### Core Capabilities:
        - **Continuous Acuity Estimation:** Predicts a calibrated $1.0 - 10.0$ Patient Criticality Score from non-invasive field vitals and observable symptoms.
        - **5-Tier Operational Triage:** Maps acuity into standard triage tiers (*Low*, *Moderate*, *Elevated*, *High*, *Critical*).
        - **Local Explainability (XAI):** Uses SHAP values to quantify exactly which physiological variables (e.g. hypoxemia, altered GCS, tachycardia) drove the acuity estimation.
        - **Hard Safety Guardrails:** Implements immediate clinical override checks for profound coma ($GCS \\le 8$), extreme hypoxia ($SpO_2 < 88\\%$), and decompensated shock ($SBP < 85$ mmHg).
        
        ---
        
        ### Hard Operational Boundaries & Exclusions:
        1. **No Medical Diagnosis:** The model does not diagnose diseases (e.g. Myocardial Infarction, Sepsis, Stroke).
        2. **No Treatment / Drug Recommendations:** The system never prescribes medications, dosages, or procedures.
        3. **Human-in-the-Loop:** Licensed paramedics, emergency medical technicians, and triage physicians always maintain clinical authority.
        """)

    # Bottom Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-text">
        <b>MANDATORY RESEARCH & PORTFOLIO DISCLAIMER:</b> This system is an educational machine learning engineering prototype. 
        It is trained on a physiologically modeled synthetic dataset ($N = 10,000$) and has <b>not</b> been approved by regulatory bodies (FDA, CE, MDR) 
        for autonomous medical decision-making. Pre-hospital clinical personnel and triage physicians retain sole responsibility for patient care.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
