"""
Synthetic Patient Criticality Dataset Generator
AI-Assisted Pre-Hospital Patient Criticality Prediction System

Generates N=10,000 realistic pre-hospital patient records with physiological
vitals, signs, symptoms, and mathematically formulated criticality targets.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_patient_data(
    n_samples: int = 10000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a DataFrame of synthetic pre-hospital patient encounters.
    
    Parameters
    ----------
    n_samples : int
        Number of patient records to generate.
    random_seed : int
        Random seed for full reproducibility.
        
    Returns
    -------
    pd.DataFrame
        Complete dataset with features and target variables.
    """
    np.random.seed(random_seed)
    
    # 1. Identifier
    patient_ids = [f"PT-{10001 + i}" for i in range(n_samples)]
    
    # 2. Demographics & Baseline Medical History
    # Age: Bimodal mixture of young/adult and geriatric emergency presentations
    is_elderly = np.random.binomial(1, 0.40, size=n_samples)
    age_young = np.random.normal(38, 14, size=n_samples)
    age_old = np.random.normal(74, 9, size=n_samples)
    age = np.where(is_elderly == 1, age_old, age_young)
    age = np.clip(np.round(age), 16, 95).astype(int)
    
    # Biological Sex
    sex = np.random.choice(["Male", "Female", "Other"], size=n_samples, p=[0.50, 0.48, 0.02])
    
    # Medical History (age-dependent probability)
    prob_htn = 0.15 + 0.45 * (age / 95.0)
    prob_cardiac = 0.05 + 0.35 * (age / 95.0)
    prob_dm = 0.08 + 0.25 * (age / 95.0)
    
    known_hypertension = np.random.binomial(1, prob_htn)
    known_cardiac_history = np.random.binomial(1, prob_cardiac)
    known_diabetes = np.random.binomial(1, prob_dm)
    
    # 3. Latent Clinical Archetypes / Acuity Profiles (Unobserved driving state)
    # 0: Stable / Mild (45%), 1: Respiratory (18%), 2: Hemodynamic / Trauma (15%),
    # 3: Neurological (10%), 4: Systemic / Severe Multi-organ (12%)
    archetype = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.45, 0.18, 0.15, 0.10, 0.12])
    
    # 4. Symptoms & Physical Signs
    # Walking ability (START primary triage marker)
    prob_walking = np.where(
        archetype == 0, 0.92,
        np.where(archetype == 1, 0.60,
        np.where(archetype == 2, 0.25,
        np.where(archetype == 3, 0.10, 0.15)))
    )
    walking_ability = np.random.binomial(1, prob_walking)
    
    # Altered consciousness
    prob_altered_consciousness = np.where(
        archetype == 3, 0.85,
        np.where(archetype == 4, 0.55,
        np.where(archetype == 2, 0.30,
        np.where(archetype == 1, 0.15, 0.03)))
    )
    altered_consciousness = np.random.binomial(1, prob_altered_consciousness)
    
    # Chest pain
    prob_cp = np.where(
        archetype == 4, 0.45,
        np.where(archetype == 0, 0.12, 0.20)
    ) + 0.20 * known_cardiac_history
    prob_cp = np.clip(prob_cp, 0.0, 0.95)
    chest_pain = np.random.binomial(1, prob_cp)
    
    # Difficulty breathing
    prob_dyspnea = np.where(
        archetype == 1, 0.88,
        np.where(archetype == 4, 0.60,
        np.where(archetype == 2, 0.25, 0.08))
    )
    difficulty_breathing = np.random.binomial(1, prob_dyspnea)
    
    # Abdominal pain
    prob_abdpain = np.where(archetype == 0, 0.22, np.where(archetype == 4, 0.35, 0.15))
    abdominal_pain = np.random.binomial(1, prob_abdpain)
    
    # Injury / Trauma
    prob_trauma = np.where(archetype == 2, 0.70, np.where(archetype == 0, 0.08, 0.05))
    injury_trauma = np.random.binomial(1, prob_trauma)
    
    # Active Bleeding
    prob_bleed = np.where(injury_trauma == 1, 0.65, 0.02)
    bleeding = np.random.binomial(1, prob_bleed)
    
    # Fever
    prob_fever = np.where(archetype == 4, 0.60, np.where(archetype == 1, 0.25, 0.10))
    fever = np.random.binomial(1, prob_fever)
    
    # Headache
    prob_headache = np.where(archetype == 3, 0.50, np.where(known_hypertension == 1, 0.35, 0.15))
    headache = np.random.binomial(1, prob_headache)
    
    # Vomiting
    prob_vomit = np.where(abdominal_pain == 1, 0.45, np.where(altered_consciousness == 1, 0.35, 0.10))
    vomiting = np.random.binomial(1, prob_vomit)
    
    # 5. Field Physiological Vital Signs
    # SpO2 (%)
    spo2_base = np.where(
        archetype == 1, np.random.normal(86, 6, size=n_samples),
        np.where(archetype == 4, np.random.normal(89, 5, size=n_samples),
        np.random.normal(97.5, 1.8, size=n_samples))
    )
    spo2_base -= 4.0 * difficulty_breathing
    spo2 = np.clip(np.round(spo2_base, 1), 68.0, 100.0)
    
    # Oxygen requirement
    prob_o2 = np.where(spo2 < 93.0, 0.90, np.where(difficulty_breathing == 1, 0.60, 0.05))
    oxygen_requirement = np.random.binomial(1, np.clip(prob_o2, 0.0, 1.0))
    
    # Respiratory Rate (breaths/min)
    rr_base = 16.0 + 0.6 * np.maximum(0, 95.0 - spo2) + 6.0 * difficulty_breathing + 3.0 * fever
    rr_noise = np.random.normal(0, 2.5, size=n_samples)
    respiratory_rate = np.clip(np.round(rr_base + rr_noise, 1), 8.0, 48.0)
    
    # Systolic Blood Pressure (mmHg)
    sbp_base = 122.0 + 15.0 * known_hypertension - 35.0 * bleeding - 25.0 * (archetype == 2)
    sbp_noise = np.random.normal(0, 12, size=n_samples)
    systolic_bp = np.clip(np.round(sbp_base + sbp_noise, 1), 60.0, 235.0)
    
    # Diastolic Blood Pressure (mmHg)
    pulse_pressure = np.random.normal(42, 6, size=n_samples) + 8.0 * (systolic_bp > 150)
    pulse_pressure = np.clip(pulse_pressure, 20.0, 80.0)
    diastolic_bp = np.clip(np.round(systolic_bp - pulse_pressure, 1), 35.0, 135.0)
    
    # Heart Rate (bpm)
    hr_base = 76.0 + 0.35 * np.maximum(0, 100.0 - systolic_bp) + 0.3 * np.maximum(0, 95.0 - spo2) + 12.0 * fever + 10.0 * chest_pain + 15.0 * bleeding
    hr_noise = np.random.normal(0, 9, size=n_samples)
    heart_rate = np.clip(np.round(hr_base + hr_noise, 1), 38.0, 195.0)
    
    # Body Temperature (°C)
    temp_base = 36.8 + 1.8 * fever - 0.8 * (bleeding == 1) * (systolic_bp < 90)
    temp_noise = np.random.normal(0, 0.4, size=n_samples)
    temperature = np.clip(np.round(temp_base + temp_noise, 1), 34.2, 41.2)
    
    # Glasgow Coma Scale (GCS, 3 - 15)
    gcs_raw = 15 - 6 * altered_consciousness - 4 * (spo2 < 82) - 3 * (systolic_bp < 80)
    gcs_noise = np.random.choice([-1, 0, 1], size=n_samples, p=[0.15, 0.70, 0.15])
    gcs = np.clip(gcs_raw + gcs_noise, 3, 15).astype(int)
    
    # Pain Severity (0 - 10 NRS)
    pain_base = (
        4.0 * injury_trauma +
        3.5 * chest_pain +
        3.0 * abdominal_pain +
        2.0 * headache +
        1.5 * (1 - walking_ability)
    )
    pain_noise = np.random.normal(0, 1.5, size=n_samples)
    pain_severity = np.clip(np.round(pain_base + pain_noise), 0, 10).astype(int)
    
    # Ambulance Arrival (Mode of transit)
    prob_amb = 0.15 + 0.40 * (1 - walking_ability) + 0.25 * (gcs < 13) + 0.15 * injury_trauma
    ambulance_arrival = np.random.binomial(1, np.clip(prob_amb, 0.10, 0.95))
    
    # 6. Target Criticality Derivation (Mathematical Latent Acuity Formulation)
    # A. Respiratory Derangement Component
    s_resp = (
        2.5 * np.maximum(0, (96.0 - spo2) / 10.0) ** 1.3 +
        1.5 * (np.abs(respiratory_rate - 16.0) / 10.0) +
        1.2 * difficulty_breathing +
        1.0 * oxygen_requirement
    )
    
    # B. Hemodynamic Derangement Component
    shock_index = heart_rate / np.maximum(systolic_bp, 40.0)
    s_hemo = (
        3.0 * np.maximum(0, shock_index - 0.70) +
        2.0 * np.maximum(0, (90.0 - systolic_bp) / 15.0) +
        1.8 * bleeding +
        1.0 * injury_trauma
    )
    
    # C. Neurological Derangement Component
    s_neuro = (
        2.2 * ((15.0 - gcs) / 3.0) +
        1.8 * altered_consciousness +
        1.2 * (1.0 - walking_ability)
    )
    
    # D. Systemic Derangement Component
    s_systemic = (
        1.0 * (np.abs(temperature - 37.0) / 1.5) +
        0.15 * pain_severity +
        1.5 * (chest_pain * known_cardiac_history) +
        0.8 * abdominal_pain
    )
    
    # Non-linear Multi-system Synergistic Interactions
    i_resp_hemo = 1.2 * ((s_resp > 2.0) & (s_hemo > 2.0)).astype(float)
    i_neuro_resp = 1.5 * ((gcs <= 8) & (spo2 < 90.0)).astype(float)
    m_age = 1.0 + 0.25 * np.maximum(0, (age - 65.0) / 30.0)
    
    # Raw latent severity score
    psi_raw = (s_resp + s_hemo + s_neuro + s_systemic + i_resp_hemo + i_neuro_resp) * m_age
    
    # Stochastic clinical observation noise
    noise_epsilon = np.random.normal(0, 0.35, size=n_samples)
    psi_noisy = psi_raw + noise_epsilon
    
    # Calibrated Sigmoidal min-max mapping to [1.0, 10.0]
    # Smooth, realistic triage distribution:
    # Low (~28%), Moderate (~26%), Elevated (~22%), High (~14%), Critical (~10%)
    mu_0 = 6.8
    scale_s = 3.2
    raw_score = 1.0 + 9.0 / (1.0 + np.exp(-(psi_noisy - mu_0) / scale_s))
    criticality_score = np.clip(np.round(raw_score, 1), 1.0, 10.0)
    
    # Urgency Category Mapping (5 discrete triage tiers)
    def assign_urgency_tier(score: float) -> str:
        if score < 2.5:
            return "Low"
        elif score < 4.5:
            return "Moderate"
        elif score < 6.5:
            return "Elevated"
        elif score < 8.5:
            return "High"
        else:
            return "Critical"
            
    urgency_category = [assign_urgency_tier(s) for s in criticality_score]
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "sex": sex,
        "ambulance_arrival": ambulance_arrival,
        "walking_ability": walking_ability,
        "altered_consciousness": altered_consciousness,
        "chest_pain": chest_pain,
        "difficulty_breathing": difficulty_breathing,
        "abdominal_pain": abdominal_pain,
        "injury_trauma": injury_trauma,
        "bleeding": bleeding,
        "fever": fever,
        "headache": headache,
        "vomiting": vomiting,
        "heart_rate": heart_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "spo2": spo2,
        "respiratory_rate": respiratory_rate,
        "temperature": temperature,
        "gcs": gcs,
        "pain_severity": pain_severity,
        "oxygen_requirement": oxygen_requirement,
        "known_cardiac_history": known_cardiac_history,
        "known_hypertension": known_hypertension,
        "known_diabetes": known_diabetes,
        "criticality_score": criticality_score,
        "urgency_category": urgency_category
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient criticality dataset.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--output_path", type=str, default="data/raw/patient_criticality_data.csv", help="Target CSV filepath.")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"Generating {args.n_samples} synthetic patient records (seed={args.seed})...")
    df = generate_synthetic_patient_data(n_samples=args.n_samples, random_seed=args.seed)
    
    df.to_csv(args.output_path, index=False)
    print(f"Dataset successfully written to {args.output_path}")
    print(f"Shape: {df.shape}")
    print("\nUrgency Category Distribution:")
    print(df["urgency_category"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))
    print("\nCriticality Score Summary:")
    print(df["criticality_score"].describe())


if __name__ == "__main__":
    main()
