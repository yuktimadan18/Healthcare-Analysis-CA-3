# Milestone 8 Stage 1: Dashboard Planning, Requirements, and Architecture

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Date:** October 21, 2026
**Auditor:** Antigravity AI
**Scope:** Comprehensive design specifications, page architecture, data flows, safety safeguards, and phased implementation roadmap for Milestone 8: Interactive Dashboard Development.

---

## 1. Executive Summary & Dashboard Purpose

Milestone 8 translates the empirical, statistical, modeling, and explainability findings from Milestones 1 through 7 into an interactive research dashboard titled:

> **Population Health Analytics & Explainable ML — CDC BRFSS 2024**

### 1.1 Core Mission
The dashboard serves as an interactive population health surveillance, model evaluation, and algorithmic audit interface for epidemiologists, biostatisticians, and health policy researchers. It allows users to:
1. Explore complex survey distributions across demographic, chronic condition, and behavioral risk factor strata in the CDC BRFSS 2024 cohort ($N = 452,464$).
2. Inspect supervised model discrimination, calibration, and class-imbalance trade-offs across four evaluated algorithms.
3. Understand the epidemiological justification for the locked operational screening threshold ($\tau = 0.080$).
4. Interactively audit global feature rankings, unscaled continuous risk gradients, and individual case explanation archetypes.

### 1.2 Binding Ethical & Surveillance Constraints
In accordance with `PROJECT_RULES.md` and public health guidelines:
- **No Clinical Diagnosis:** The dashboard classifies self-reported history of coronary heart disease or myocardial infarction (`_MICHD`). It **does not** provide clinical diagnostic advice, predict future disease onset, or calculate personal medical risk.
- **No Free-Form "Symptom Checker" Input:** The dashboard will not provide a free-form form where users type in personal health attributes to receive a personal diagnostic prediction.
- **No Raw Data Exposure:** Anonymized survey records and precomputed aggregates are utilized exclusively; raw survey respondent records will not be exposed.
- **Locked Threshold Integrity:** The operational threshold remains strictly fixed at **$0.080$**. Interactive threshold sliders function solely as visual exploration tools for hypothetical sensitivity-specificity trade-offs and cannot modify the validated decision threshold.

---

## 2. Technical Framework & Dependency Architecture

### 2.1 Dependency Inspection Findings
As documented in [`reports/milestone8_stage1_dashboard_data_inventory.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone8_stage1_dashboard_data_inventory.md):
- `plotly` is **installed** (`version 7.1.0`).
- `pandas` (`3.0.6`), `numpy` (`2.5.4`), `scikit-learn` (`1.9.1`), and `joblib` (`1.6.0`) are **installed**.
- Web server frameworks (`streamlit`, `dash`, `gradio`, `flask`, `fastapi`) are **not currently installed** in `.venv`.

### 2.2 Framework Options for User Decision:
To comply with the rule against installing packages without explicit authorization, the architecture provides two modular implementation options:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              DASHBOARD SERVING OPTIONS                                 │
├────────────────────────────────────────────┬───────────────────────────────────────────┤
│ Option A: Zero-Dependency Plotly.js SPA    │ Option B: Idiomatic Streamlit App         │
│ (DEFAULT - Ready Immediately)              │ (Requires Explicit User Approval)         │
├────────────────────────────────────────────┼───────────────────────────────────────────┤
│ • Pure HTML5, Modern CSS, Vanilla JS       │ • Requires pip install streamlit          │
│ • Uses Plotly.js (CDN or local asset)      │ • Multi-page Streamlit application        │
│ • Zero pip packages required               │ • Idiomatic Python syntax                 │
│ • Served via python -m http.server 8501    │ • Served via streamlit run                │
│ • Instant sub-second page loads            │ • Requires continuous Python kernel loop  │
└────────────────────────────────────────────┴───────────────────────────────────────────┘
```

**Recommendation:**
Both options share the exact same underlying modular Python backend, data precomputation pipelines, and chart logic (`dashboard/components/`). The architecture is completely decoupled from the rendering layer.

---

## 3. Detailed Page Architecture & Content Specifications

The dashboard is structured into **six cohesive, modular sections**:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│        DASHBOARD HEADER: Title, CDC BRFSS Badge, Locked Threshold (0.080), Nav       │
├───────────────┬──────────────────────────────────────────────────────────────────────┤
│               │  [Section A] Overview & Project Scope                                │
│               ├──────────────────────────────────────────────────────────────────────┤
│               │  [Section B] Population Health Explorer (Unweighted vs. Weighted)    │
│  NAVIGATION   ├──────────────────────────────────────────────────────────────────────┤
│    SIDEBAR    │  [Section C] Model Performance & Validation vs. Test Audit           │
│               ├──────────────────────────────────────────────────────────────────────┤
│               │  [Section D] Threshold Optimization & Trade-Off Analysis             │
│               ├──────────────────────────────────────────────────────────────────────┤
│               │  [Section E] Explainable AI: Global & Local Archetype Audits         │
│               ├──────────────────────────────────────────────────────────────────────┤
│               │  [Section F] Methodological Disclosures & Responsible Use            │
└───────────────┴──────────────────────────────────────────────────────────────────────┘
```

---

### Section A: Overview & Project Scope
- **Purpose:** Contextualize the surveillance research, data provenance, target definition, and cohort dimensions.
- **Key Display Cards:**
  - *Analytical Cohort:* $452,464$ validated survey records (from $457,670$ raw records; $5,206$ missing omissions).
  - *Target Outcome (`_MICHD`):* $42,338$ reported CAD/MI cases ($9.36\%$ unweighted sample prevalence, $6.80\%$ survey-weighted US civilian population prevalence).
  - *Sampling Architecture:* Complex, multistage probability sampling across 54 US states and territories.
  - *Champion Model:* Unweighted Histogram Gradient Boosting Classifier (GBDT) with frozen preprocessor ($190$ features).
  - *Locked Threshold:* $\tau = 0.080$ (Balanced population screening cutoff, Youden's J criterion).
- **Core Visualizations & Tables:**
  - Summary metrics cards (Cohort Size, Prevalence, Baseline PR-AUC $0.3701$, Locked Threshold $0.080$).
  - Candidate outcome feasibility comparison table (`results/tables/outcome_feasibility_analysis.csv`).
  - Candidate outcomes prevalence bar chart (`results/figures/fig4_candidate_outcomes_prevalence.png` or interactive Plotly equivalent).
- **Prominent Banner:** High-visibility banner stating: *"Surveillance and epidemiological classification model. Not a clinical diagnostic tool or personal health risk calculator."*

---

### Section B: Population Health Explorer
- **Purpose:** Allow interactive exploration of population risk factor distributions, sociodemographic patterns, and multimorbidity clustering.
- **Interactive Controls:**
  - *Domain Selector:* Demographics (Age, Sex, Race, Education, Income), Health Status (General Health, Physical/Mental Health Days), Chronic Comorbidities (Diabetes, Stroke, Kidney Disease, COPD, Arthritis), Behavioral/Access Factors (Smoking, BMI, Routine Checkup, Healthcare Costs).
  - *Weighting Toggle:* Unweighted Respondent Counts ($N$) vs. Survey-Weighted Population Estimates (`_LLCPWT`).
- **Core Visualizations:**
  - Interactive distribution histograms and bar charts for selected variables.
  - Stratified disease prevalence curves (e.g. reported CAD/MI prevalence across age bands and general health ratings).
  - Chronic comorbidity correlation heatmap (`results/figures/fig5_chronic_comorbidity_correlation.png` / Plotly equivalent).
  - Multivariable survey-weighted odds ratios forest plot (`results/tables/multivariable_logistic_regression_or.csv`).
- **Survey Design Disclosure:** Explicit note indicating that survey-weighted metrics represent descriptive population-weighted point estimates; design-based standard errors require complex survey variance estimation (`_PSU`, `_STSTR`).

---

### Section C: Model Performance & Comparative Evaluation
- **Purpose:** Transparently compare all four candidate models and audit validation versus final holdout test results.
- **Interactive Controls:**
  - *Partition View:* Validation Partition ($N = 67,870$) vs. Final Holdout Test Partition ($N = 67,870$).
  - *Metric Focus:* Discrimination (PR-AUC, ROC-AUC) vs. Calibration (Brier Score, Brier Skill Score) vs. Threshold-Dependent (Sensitivity, Specificity, F1).
- **Core Visualizations & Tables:**
  - Model comparison summary table (`results/tables/model_validation_comparison.csv` and `results/tables/final_test_evaluation_summary.csv`).
  - Interactive ROC curves (`results/figures/fig10_roc_curves_validation.png` / Plotly equivalent).
  - Interactive Precision-Recall curves (`results/figures/fig11_pr_curves_validation.png` / Plotly equivalent).
  - Calibration curves across decile bins (`results/figures/fig12_calibration_curves_validation.png`), explaining why the unweighted GBDT was chosen (natural calibration, Brier $0.0700$) over class-weighted models (distorted probabilities, negative Brier skill score).
  - Holdout test confusion matrix display at default ($0.50$) vs. locked ($0.080$) thresholds.

---

### Section D: Threshold Optimization & Trade-Off Analysis
- **Purpose:** Demonstrate the epidemiological rationale for the locked operational threshold ($\tau = 0.080$) and visualize trade-offs between screening sensitivity and false-positive burden.
- **Interactive Controls:**
  - *Threshold Exploration Slider:* Interactive slider $\tau \in [0.02, 0.90]$ dynamically updating confusion matrix counts and rate metrics.
  - *Persistent Lock Indicator:* Non-interactive display badge showing: `Locked Operational Threshold: 0.080 (Youden's J Cutoff)`.
- **Core Visualizations & Tables:**
  - Dual-axis trade-off plot: Sensitivity, Specificity, Precision (PPV), NPV, and Balanced Accuracy across threshold values (`results/tables/threshold_optimization_analysis.csv`).
  - Dynamic 2x2 confusion matrix showing TP, FP, TN, and FN.
  - Trade-off curve (`results/figures/fig13_threshold_tradeoffs.png` / Plotly equivalent).
- **Epidemiological Context:**
  - At default $\tau = 0.50$, the model captures only $10.7\%$ of cases ($89.3\%$ missed).
  - At locked $\tau = 0.080$, sensitivity rises to **$83.3\%$** on validation ($83.9\%$ on test) with an NPV of **$97.6\%$**, optimizing early population screening capture while maintaining a manageable follow-up burden.
- **Safety Safeguard:** Visual slider exploration includes a persistent banner: *"Hypothetical exploration only. The operational decision threshold is locked at 0.080 in accordance with formal project protocol."*

---

### Section E: Explainable AI (XAI)
- **Purpose:** Provide complete transparency into global population feature drivers and individual prediction attributions.
- **Interactive Sub-Sections:**
  1. *Global Grouped Permutation Importance:*
     - Interactive horizontal bar chart of 38 original predictors with error bars (PR-AUC drop).
     - Indicator dilution comparison toggle: Grouped Variable Level vs. 190 Indicator Level.
     - Source: `results/tables/xai_global_variable_importance.csv` and `results/tables/xai_transformed_feature_importance.csv`.
  2. *Unscaled Partial Dependence Profiles:*
     - Interactive multi-panel curves for continuous age (`_AGE80`), physical health limitation days (`PHYSHLTH_DAYS`), and body mass index (`BMI_NUMERIC`).
     - Display of reference lines: Cohort prevalence ($9.36\%$) and operational screening threshold ($0.080$).
     - Highlighting the critical age threshold crossing at **$54$ years of age**.
  3. *Representative Case Archetype Audits:*
     - Detailed inspection of the 4 validated validation archetypes: True Positive (`56065`), False Positive (`279669`), True Negative (`253891`), and False Negative (`44471`).
     - Interactive directional attribution waterfall / bar plots showing the top risk-elevating (crimson) and risk-mitigating (steel blue) factors.
     - Local Ridge linear surrogate weights and local fidelity metric ($R^2 > 0.93$).
     - Sources: `results/explanations/xai_case_archetypes_summary.csv` and `results/explanations/xai_local_case_explanations.csv`.
- **Attribution Methodology Disclosure:** Explicit label: *"Custom grouped marginal perturbation ($\Delta p$) and local Ridge surrogate models. These are model-agnostic custom interpretability methods, NOT SHAP or LIME."*

---

### Section F: Limitations & Responsible Use
- **Purpose:** Dedicated public health and ethical disclosure section consolidating all project boundaries.
- **Key Disclosures:**
  1. *Cross-Sectional Surveillance Boundary:* Classification of self-reported historical physician diagnoses (`_MICHD`), not prospective incidence or biological causation.
  2. *Clinical Diagnostic Validity:* Clinical screening validity and rule-out capability in clinical practice have not been established.
  3. *Survey Design Constraints:* Models were trained on unweighted sample loss; population estimates represent descriptive point estimates without design-based variance linearization.
  4. *Demographic Predictor Considerations:* Biological sex and race adjustments reflect surveillance disparities, access barriers, and social determinants of health; they must never justify clinical denial of care.
  5. *Guidance for Researchers:* Permissible uses (population health surveillance, hypothesis generation) vs. impermissible uses (individual medical diagnosis, insurance underwriting, autonomous clinical triage).

---

## 4. Assessment of Interactive Individual-Record Prediction

### 4.1 Methodological & Ethical Assessment
| Proposed Capability | Methodological Justification | Ethical / Clinical Risk | Recommendation |
|:---|:---|:---|:---:|
| **Free-Form Patient Clinical Risk Calculator** | **UNJUSTIFIED.** Model was trained on cross-sectional self-reported survey answers to classify historical disease history, not to predict personal future disease onset. | **HIGH RISK.** Lay users may mistake model outputs for medical advice or confirmed diagnosis, leading to false reassurance or unnecessary distress. | **REJECTED.** Free-form patient symptom entry will NOT be implemented. |
| **Raw Survey Respondent Browser** | **UNJUSTIFIED.** Browsing arbitrary raw survey rows offers no analytical value over precomputed summaries. | **MODERATE RISK.** Potential risk of survey re-identification or misinterpretation of survey records as clinical patient charts. | **REJECTED.** Raw row browser will NOT be implemented. |
| **Prespecified Archetype What-If Simulator** | **JUSTIFIED.** Auditing the 4 validated, anonymized case archetypes (TP, FP, TN, FN) and allowing controlled toggling of specific risk factors (e.g. changing health rating from "Poor" to "Excellent") illustrates non-linear model sensitivity. | **LOW RISK (With Guardrails).** Anonymized, prespecified, and framed strictly as algorithmic auditing. | **APPROVED.** Included strictly within the case archetype audit view with persistent non-causal disclaimers. |

---

## 5. Performance Strategy & Data Flow Architecture

### 5.1 Precomputation Strategy (Sub-Second UI Loading)
The full raw/metadata dataset is large ($452,464$ rows, $8.3\,\text{MB}$ compressed). Loading and filtering $452\text{k}$ rows on every dashboard interaction would cause severe lag ($5-15$ second delays).

**Solution: The Precomputed Summary Data Layer**
In Stage 2, a lightweight precomputation script (`dashboard/precompute_dashboard_data.py`) will pre-aggregate:
1. `dashboard/data/population_distributions.json`: Precomputed univariate and bivariate histograms, contingency counts, and survey-weighted percentages for all supported variables.
2. `dashboard/data/model_metrics.json`: Consolidated validation and test metrics, ROC/PR coordinate arrays, and calibration deciles.
3. `dashboard/data/threshold_curve.json`: Full threshold grid coordinates ($89$ points).
4. `dashboard/data/xai_summaries.json`: Global importance ranks, unscaled PDP curves, and archetype explanation records.

**Performance Impact:** The entire precomputed dashboard data bundle is $< 1.5\,\text{MB}$, enabling instantaneous, sub-second rendering with zero runtime compute overhead.

---

## 6. Directory Structure & Modular Implementation Sequence

### 6.1 Proposed Dashboard Directory Structure
```text
dashboard/
├── data/                                 # Precomputed lightweight JSON/CSV bundles
│   ├── population_distributions.json
│   ├── model_metrics.json
│   ├── threshold_curve.json
│   └── xai_summaries.json
├── components/                           # Decoupled visualization and data modules
│   ├── __init__.py
│   ├── overview_view.py                  # Section A logic
│   ├── population_view.py                # Section B logic
│   ├── performance_view.py               # Section C logic
│   ├── threshold_view.py                 # Section D logic
│   ├── xai_view.py                       # Section E logic
│   └── limitations_view.py               # Section F logic
├── assets/                               # CSS styles, logos, responsive layout tokens
│   ├── styles.css
│   └── favicon.ico
├── precompute_dashboard_data.py          # Offline summary generation pipeline
├── app.py                                # Main application entry point (Streamlit or HTTP SPA)
└── README.md                             # Dashboard launch instructions & user guide
```

---

## 7. Phased Implementation Roadmap for Subsequent Stages

| Milestone 8 Stage | Scope & Deliverables | Constraints |
|:---:|:---|:---|
| **Stage 1 (Current)** | **Planning, Requirements & Architecture Only.** Complete data inventory and architectural plan. | Zero code changes; no dependencies installed. |
| **Stage 2** | **Data Precomputation & Backend Modules.** Implement `precompute_dashboard_data.py` to extract lightweight JSON summaries from existing CSVs and cohort metadata; implement modular visualization functions in `dashboard/components/`. | Read-only data access; no model retraining. |
| **Stage 3** | **Interactive Dashboard Implementation.** Build the interactive UI across Sections A–F; integrate Plotly interactive charts; incorporate locked threshold safeguards and responsible-use disclosures. | Zero threshold modification; strict non-causal labeling. |
| **Stage 4** | **Verification, Automated Testing & Documentation.** Implement automated tests in `tests/test_dashboard.py` (verifying page rendering, data consistency, threshold lock invariant, and zero test partition leakage); create `reports/milestone8_dashboard_audit.md` and update `README.md`. | Full test suite passing; clean documentation. |

---

## 8. Summary & Next Steps

This plan delivers a modular, lightweight, scientifically grounded, and ethically safeguarded dashboard architecture.

Awaiting your review and explicit approval of [`reports/milestone8_stage1_dashboard_plan.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone8_stage1_dashboard_plan.md) and [`reports/milestone8_stage1_dashboard_data_inventory.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone8_stage1_dashboard_data_inventory.md) before proceeding to **Stage 2: Dashboard Data Preparation and Backend Modules**.
