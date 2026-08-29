# Target Definition & Synthetic Acuity Formulation

## Document Information
- **Project Title:** AI-Assisted Pre-Hospital Patient Criticality Prediction & Emergency Triage Support System
- **Document Version:** 1.0.0
- **Status:** Phase 1 Completed / Target Formulation Finalized
- **Primary Metric Targets:** `criticality_score` (Continuous $1.0 - 10.0$) and `urgency_category` (5 Levels)

---

## 1. Target Philosophy & Design Objectives

In emergency medicine, patient triage requires assessing immediate life threats, physiological stability, and required level of care. Clinical systems like the **Emergency Severity Index (ESI)** (5 levels), **National Early Warning Score (NEWS2)** (0–20), and **Simple Triage and Rapid Treatment (START)** rely on physiological thresholds and clinical discernment.

To create an effective machine learning prototype, we define:
1. **`criticality_score` (Continuous / Fine-Grained Index, $1.0 - 10.0$):** Reflects the overall burden of acute physiological distress and urgency of emergency intervention.
2. **`urgency_category` (Categorical, 5 Bins):** Discrete operational bins for triage dashboard filtering and emergency bay assignment.

---

## 2. Mathematical Definition of Synthetic Acuity

The target is generated via a **multi-factorial latent physiological derangement model** combined with non-linear interaction terms and stochastic noise. 

> [!IMPORTANT]
> **No Target Leakage:** The internal latent variables (e.g., $S_{\text{resp}}$, $S_{\text{hemo}}$, $S_{\text{neuro}}$) are intermediate simulation calculations and are **never** exported to the dataset. The machine learning model is exposed **only** to the observable patient features ($SpO_2$, HR, BP, symptoms, etc.).

### 2.1 Component Derangement Formulations

The latent physiological acuity $\Lambda$ is composed of four primary physiological domain scores:

#### A. Respiratory Derangement Component ($S_{\text{resp}}$)
Calculates respiratory compromise based on hypoxemia, tachypnea/bradypnea, dyspnea, and oxygen need:
$$S_{\text{resp}} = 2.5 \cdot \max\left(0, \frac{96 - SpO_2}{10}\right)^{1.3} + 1.5 \cdot \frac{|RR - 16|}{10} + 1.2 \cdot \text{difficulty\_breathing} + 1.0 \cdot \text{oxygen\_requirement}$$

#### B. Hemodynamic & Perfusion Component ($S_{\text{hemo}}$)
Calculates shock, hypotension, extreme tachycardia, and blood loss:
$$\text{Shock Index} = \frac{\text{Heart Rate}}{\text{Systolic BP}}$$
$$S_{\text{hemo}} = 3.0 \cdot \max(0, \text{Shock Index} - 0.7) + 2.0 \cdot \max\left(0, \frac{90 - SBP}{15}\right) + 1.8 \cdot \text{bleeding} + 1.0 \cdot \text{injury\_trauma}$$

#### C. Neurological & Functional Component ($S_{\text{neuro}}$)
Calculates cerebral compromise, coma depth, and mobility:
$$S_{\text{neuro}} = 2.2 \cdot \left(\frac{15 - GCS}{3}\right) + 1.8 \cdot \text{altered\_consciousness} + 1.2 \cdot (1 - \text{walking\_ability})$$

#### D. Systemic & Inflammatory Component ($S_{\text{systemic}}$)
Calculates thermal instability, severe pain, and high-risk acute syndromes:
$$S_{\text{systemic}} = 1.0 \cdot \frac{|Temp - 37.0|}{1.5} + 0.15 \cdot \text{pain\_severity} + 1.5 \cdot (\text{chest\_pain} \times \text{known\_cardiac\_history}) + 0.8 \cdot \text{abdominal\_pain}$$

---

### 2.2 Non-Linear Synergistic Interactions

In clinical pathophysiology, multi-system failure is multiplicative rather than purely additive. We incorporate compound risk terms:
- **Hypoxia + Shock Interaction:** $I_{\text{resp-hemo}} = 1.2 \times (S_{\text{resp}} > 2.0) \times (S_{\text{hemo}} > 2.0)$
- **Coma + Airway/Hypoxia Interaction:** $I_{\text{neuro-resp}} = 1.5 \times (GCS \le 8) \times (SpO_2 < 90)$
- **Geriatric Frailty Multiplier:** $M_{\text{age}} = 1.0 + 0.25 \times \max\left(0, \frac{\text{Age} - 65}{30}\right)$

---

### 2.3 Composite Score & Stochastic Noise

The raw latent severity score $\Psi$ is calculated as:
$$\Psi = \left( S_{\text{resp}} + S_{\text{hemo}} + S_{\text{neuro}} + S_{\text{systemic}} + I_{\text{resp-hemo}} + I_{\text{neuro-resp}} \right) \times M_{\text{age}}$$

To reflect real-world clinical ambiguity, measurement variance, and unobserved patient factors, we add zero-mean Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 = 0.35)$:
$$\Psi_{\text{noisy}} = \Psi + \epsilon$$

Finally, the score is mapped onto the fixed interval $[1.0, 10.0]$ via a calibrated sigmoidal min-max scaling function and rounded to 1 decimal place:
$$\text{criticality\_score} = \text{clip}\left( 1.0 + 9.0 \times \frac{1}{1 + \exp\left(-\frac{\Psi_{\text{noisy}} - \mu_0}{s}\right)}, 1.0, 10.0 \right)$$
where $\mu_0$ and $s$ are calibration parameters selected such that stable patients center around $1.5 - 3.0$ and moribund/shock patients reach $8.5 - 10.0$.

---

## 3. Operational Urgency Categories

To support rapid triage dispatch and visual alert color-coding, the continuous `criticality_score` is mapped into 5 categorical bands:

| Urgency Category | Criticality Range | Typical Clinical Presentation (Simulated) | Triage Action / Operational Priority | Color Indicator |
| :--- | :---: | :--- | :--- | :---: |
| **Low** | $1.0 - 2.4$ | Normal vitals, ambulatory, minor isolated pain, no red flags. | Routine ED intake; non-urgent queue. | 🟢 Green |
| **Moderate** | $2.5 - 4.4$ | Mild vital deviations (mild fever, tachycardia), moderate pain, non-urgent symptoms. | Urgent care / standard ED bed within 60 min. | 🟡 Yellow |
| **Elevated** | $4.5 - 6.4$ | Moderate vital abnormalities, severe pain, non-ambulatory, potential infection/cardiac risk. | Direct evaluation within 30 min; monitoring required. | 🟠 Orange |
| **High** | $6.5 - 8.4$ | Marked tachypnea/hypoxia ($SpO_2 < 90\%$), shock, severe trauma, active bleed, chest pain + cardiac history. | Immediate physician evaluation; high resource priority. | 🔴 Crimson |
| **Critical** | $8.5 - 10.0$ | Profound shock ($SBP < 80$, $HR > 140$), $GCS \le 8$ (coma/unresponsive), severe hypoxia ($SpO_2 < 80\%$), life-threatening multi-organ failure. | Immediate resuscitation bay / Trauma Team activation. | 🟣 Purple |

---

## 4. Why This is NOT Clinical Validation

1. **Simulated Heuristic Model:** The formulas above are based on clinical domain logic inspired by NEWS2, ESI, and trauma triage indices, but have **not** been derived from real prospective survival or ICU admission outcomes.
2. **Pedagogical Purpose:** The target provides a challenging, non-linear, multi-modal machine learning benchmark that tests regression and classification algorithms on physiological data.
3. **Explicit Disclaimer:** The 1–10 score must always be presented as an educational engineering prototype index.
