# Project Requirements & Problem Formulation

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Status:** Phase 0 Completed / Baseline Formulation
- **Classification:** Educational & Portfolio Machine Learning Engineering Decision-Support Prototype

---

## 1. Executive Summary & Problem Formulation

### 1.1 What problem are we solving?
In emergency medical response, the time window between field contact (e.g., in an ambulance, remote clinic, or disaster zone) and arrival at the emergency department (ED) is critical. Emergency dispatchers, paramedics, and triage nurses must rapidly assess patient acuity with limited diagnostic tools. Subjective assessments or manual triage protocols under high-stress conditions can lead to:
- **Under-triage:** Misclassifying a critically deteriorating patient as stable, resulting in fatal treatment delays.
- **Over-triage:** Inappropriately routing stable patients to high-resource trauma bays, overwhelming emergency resuscitation teams and causing resource misallocation.

This project develops an **AI-assisted decision-support prototype** that ingests basic field-accessible patient vitals, observable symptoms, and transit metadata to estimate a **standardized Patient Criticality Score (1–10)** and map it to an **Urgency Category (Low, Moderate, Elevated, High, Critical)** with local feature attributions (Explainable AI).

---

## 2. Target Users and Operational Context

### 2.1 Who would use the prototype?
1. **Paramedics / Emergency Medical Technicians (EMTs):** Entering field observations via tablet or mobile interface during pre-hospital transport to alert the receiving hospital's triage desk.
2. **Ambulance Operators & Field Dispatchers:** Logging preliminary patient condition for emergency coordination.
3. **Emergency Department Triage Nurses:** Receiving automated pre-arrival notifications with acuity predictions and contributing factors to prepare resuscitation bays before the ambulance arrives.
4. **Rural / Remote Health Workers:** Providing standardized initial acuity estimation when specialized emergency physicians are unavailable.

### 2.2 Operational Environment
- **Pre-hospital transport:** Ambulances, emergency medical vehicles, primary health centers.
- **Field constraints:** Limited network bandwidth, high stress, noisy measurements, incomplete information (e.g., patient unconscious or unable to provide full history).

---

## 3. Information Availability (Inputs)

### 3.1 What information is available before hospital arrival?
Only non-invasive, immediately observable, and rapidly measurable parameters are available pre-hospital:
- **Demographics:** Age, Biological Sex.
- **Transit Metadata:** Ambulance Arrival Mode (Emergency transport vs. self-presentation).
- **Physical Signs & Observable Symptoms:**
  - Ambulatory/Walking status (can the patient walk?)
  - Altered consciousness / responsiveness
  - Active bleeding
  - Injury / physical trauma
  - Chest pain
  - Difficulty breathing (dyspnea)
  - Abdominal pain
  - Vomiting
  - Fever / chills
  - Severe headache
- **Field Vital Signs:**
  - Heart Rate (bpm)
  - Systolic Blood Pressure (mmHg)
  - Diastolic Blood Pressure (mmHg)
  - Blood Oxygen Saturation ($SpO_2$, %)
  - Respiratory Rate (breaths/min)
  - Body Temperature (°C)
  - Glasgow Coma Scale (GCS, 3–15)
- **High-Yield Justified Clinical Context:**
  - Reported Pain Severity (0–10 Numeric Rating Scale)
  - Oxygen Requirement / Supplemental $O_2$ in transit
  - High-risk medical history flags (Known Cardiac, Hypertension, Diabetes)

### 3.2 What information is explicitly NOT available?
- Laboratory blood tests (troponin, lactate, arterial blood gases, CBC, renal panel).
- Imaging (CT scans, X-rays, MRI, ultrasound).
- Invasive hemodynamic monitoring (central venous pressure, arterial lines).
- Definitive specialist diagnoses.

---

## 4. Expected Outputs & Prediction Task

### 4.1 What the model SHOULD predict:
1. **Criticality Score ($1.0 - 10.0$):** A continuous or fine-grained ordinal index reflecting acute physiological distress and urgent resource need.
2. **Urgency Category:**
   - **Low (1.0 – 2.4):** Non-urgent, stable vitals, ambulatory.
   - **Moderate (2.5 – 4.4):** Mildly abnormal vitals or localized pain without severe distress.
   - **Elevated (4.5 – 6.4):** Moderate physiological derangement or high-risk symptoms requiring timely evaluation.
   - **High (6.5 – 8.4):** Severe derangement in vital signs, potential life-threat, urgent resuscitation needed.
   - **Critical (8.5 – 10.0):** Immediate threat to life/limb, profound hypoxia, shock, coma ($GCS \le 8$), active major trauma/hemorrhage.
3. **Model Confidence / Prediction Intervals:** Quantifying the model's certainty.
4. **Local Feature Explanations (Explainable AI):** Quantifying which specific physiological factors drove the acuity score (e.g., "+2.1 criticality due to $SpO_2 = 82\%$ and altered consciousness").

### 4.2 What the model MUST NOT predict:
1. **Medical Diagnoses:** The model must never output disease labels (e.g., "Acute Myocardial Infarction", "Appendicitis", "Subdural Hematoma").
2. **Medication & Treatment Recommendations:** The model must never advise specific drug dosages or surgical interventions.
3. **Autonomous Admission/Discharge Decisions:** The model must not dictate clinical routing without human sign-off.

---

## 5. Justification for Machine Learning

### Why is ML appropriate for this problem?
1. **Multivariate Non-Linear Interactions:** While simple heuristics (e.g., MEWS or single-parameter thresholds) capture extreme abnormalities, subtle combinations of borderline vitals (e.g., borderline tachycardia + mild tachypnea + diaphoresis) often signify early compensated shock that linear thresholds miss.
2. **Tolerance to Missing or Noisy Inputs:** Pre-hospital data frequently contains missing fields (e.g., temperature unmeasured during a rapid trauma transit). Trained tree-based and regularized models can learn robust predictive patterns despite partial feature availability.
3. **Rapid, Deterministic Inference:** ML inference executes in sub-millisecond timeframes, making it suitable for low-connectivity offline tablet deployments in ambulances.

---

## 6. Known Limitations & Safety Boundaries

1. **Synthetic Data Foundation:** The model is trained on a scientifically structured synthetic dataset representing 10,000 simulated emergency encounters. It has **NOT** been trained on real clinical electronic health records (EHR).
2. **No Clinical Validation:** The estimated score is an engineering index inspired by triage principles (ESI, NEWS2) but has **no regulatory approval** (FDA/CE Mark/MDR) for medical decision-making.
3. **Atypical Presentations:** Synthetic datasets may underrepresent rare clinical syndromes or edge cases (e.g., silent hypoxemia in elderly patients).
4. **Human-in-the-Loop Mandate:** The application is strictly a **decision-support tool**; clinical judgment by paramedics and triage physicians always supersedes model output.

---

## 7. Requirements Before Any Real-World Clinical Use

To transition this architectural prototype into a regulated clinical software as a medical device (SaMD), the following mandatory regulatory and clinical milestones would be required:
1. **Real-World Clinical Data Acquisition:** Retrospective and prospective pre-hospital data collection from accredited EMS agencies under IRB (Institutional Review Board) approval.
2. **Clinical Validation Studies:** Multi-center randomized or observational clinical trials comparing triage accuracy against expert emergency medicine consensus.
3. **Fairness & Bias Audits:** Rigorous subgroup performance auditing across age brackets, sexes, and demographic groups.
4. **Regulatory Approval:** Compliance with FDA 21 CFR Part 820 (Quality System Regulation), FDA SaMD guidance, and EU Medical Device Regulation (EU MDR 2017/745).
5. **Cybersecurity & Privacy Compliance:** HIPAA (Health Insurance Portability and Accountability Act) and GDPR compliance with end-to-end data encryption at rest and in transit.
