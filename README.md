# CDC BRFSS 2024 Healthcare Analytics

## Project Overview
This project builds an explainable, reproducible healthcare analytics and predictive modeling framework using the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024** combined landline and cellular survey dataset.

The analytics workflow strictly follows the standardized guidelines established in `PROJECT_RULES.md`, emphasizing complex survey design principles, prevention of data leakage, modular engineering, and clinical/ethical awareness (distinguishing predictive associations from clinical diagnoses).

> **Current Project Status**: Milestones 1 through 7 are complete. Supervised models classifying self-reported cardiovascular disease history (`_MICHD`) have been developed, evaluated, and comprehensively explained using Explainable AI (XAI). Validation-only model comparison selected **HistGradientBoostingClassifier (Unweighted GBDT)** as Champion ($\text{PR-AUC} = 0.3701$, $\text{ROC-AUC} = 0.8449$, $\text{Brier} = 0.0700$), with its operational screening threshold locked at $\tau = 0.080$ (Youden's J cutoff). Final test holdout evaluation confirmed an unweighted Sensitivity of 83.91%, Specificity of 70.45%, NPV of 97.69%, PR-AUC of 0.3808, and ROC-AUC of 0.8492. Milestone 7 implemented global and local XAI: (1) bijective feature mapping from 38 original predictors to 190 transformed features, (2) grouped permutation importance over 10 repeats, (3) unscaled partial dependence profiles for age, physical health limitations, and BMI, (4) transparent local marginal perturbation attributions ($\Delta p$) and local Ridge linear surrogate models ($R^2 > 0.93$) across representative validation quadrants (TP, FP, TN, FN), and (5) the executed analysis notebook `notebooks/05_explainable_ai_analysis.ipynb`. All attributions describe mathematical model behavior in cross-sectional surveillance data and do not establish causal intervention effects or clinical diagnostic validity.

---

## Milestone 7: Explainable Artificial Intelligence (XAI)

Milestone 7 delivers a methodologically rigorous, dependency-safe interpretability framework auditing the frozen champion GBDT model strictly on the **validation partition** ($N = 67,870$), keeping the holdout test partition quarantined.

### 1. Explanation Methodologies
- **Bijective Feature Mapping:** Bridges 38 original survey variables to 190 preprocessed features (`results/tables/xai_feature_mapping_audit.csv`), preventing dummy confusion.
- **Global Grouped Permutation Importance:** Simultaneously permutes all dummy indicators belonging to each parent variable across 10 reproducible repeats (seed 42), eliminating indicator dilution. Evaluated primarily by **PR-AUC decrease** due to class imbalance ($9.36\%$ prevalence).
- **Unscaled Partial Dependence Profiles (PDP):** Uses recursion PDP unscaled via fitted `StandardScaler` statistics to natural survey units (years, days, $\text{kg/m}^2$) and transformed via logistic sigmoid to marginal predicted probabilities.
- **Local Marginal Perturbation Attribution ($\Delta p$):** Transparent, model-agnostic local method evaluating risk shifts $\Delta p_v = f(x) - \mathbb{E}_{z \sim \mathcal{D}_{\text{bg}}}[f(x_{C_v \leftarrow z_{C_v}})]$ relative to an empirical background sample ($M = 200$).
- **Local Sparse Linear Surrogates:** Fits weighted Ridge regressions on $N = 1,000$ neighborhood perturbations weighted by an exponential distance kernel, demonstrating high local fidelity ($R^2 \in [0.931, 0.969]$).

### 2. Key Empirical Findings
- **Top Population Predictors:** `_AGE80` (Rank 1, mean PR-AUC drop $0.0772$), `GENHLTH` (Rank 2, drop $0.0434$), `_SEX` (Rank 3, drop $0.0380$), `CVDSTRK3` (Rank 4, drop $0.0306$), and `PERSDOC3` (Rank 5, drop $0.0127$).
- **Age Risk Trajectory:** Partial dependence reveals an accelerating risk curve that crosses the locked screening threshold ($0.080$) at approximately **$54$ years of age**, rising above $23\%$ by age 80.
- **Representative Case Archetypes:**
  - *True Positive (`VALIDATION_ID_56065`, $\hat{p} = 0.267$):* Driven by age 76, fair health, kidney disease, and diabetes.
  - *False Positive (`VALIDATION_ID_279669`, $\hat{p} = 0.166$):* Screening alert driven by poor health and 30 days of physical illness without reported diagnosis.
  - *True Negative (`VALIDATION_ID_253891`, $\hat{p} = 0.016$):* Protected by young age (41) and complete absence of chronic disease despite Class I obesity.
  - *False Negative (`VALIDATION_ID_44471`, $\hat{p} = 0.044$):* Favorable self-reported vitality ("Very Good" health) and normal BMI mask disease history, suppressing predicted risk below threshold despite age 73.

### 3. How to Run XAI Pipelines and Notebook
```powershell
# 1. Run global importance and partial dependence analysis (Generates Figs 16-18)
.venv\Scripts\python.exe src/xai/global_importance.py

# 2. Run local explanations and archetype analysis (Generates Fig 19)
.venv\Scripts\python.exe src/xai/local_explanations.py

# 3. Run all unit tests (19 passed)
.venv\Scripts\python.exe -m unittest discover tests

# 4. View executed XAI notebook
# notebooks/05_explainable_ai_analysis.ipynb
```

---

## Repository Structure
```text
Project/
├── .venv/                  # Virtual environment (Python 3.13.14)
├── .vscode/                # Workspace editor settings
├── config/                 # Pipeline and model configuration files
├── dashboard/              # Interactive dashboard application (Milestone 8)
├── data/
│   ├── raw/                # Original CDC BRFSS 2024 ZIP archive
│   ├── interim/            # Transformed / partitioned intermediate data
│   └── processed/          # Analysis- and modeling-ready datasets
├── docs/                   # Methodological documentation & project decisions
├── models/                 # Serialized model artifacts, lock summaries, and baselines
├── notebooks/              # Modular Jupyter analysis notebooks
│   ├── 01_data_quality_and_eda.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_feature_engineering_and_splits.ipynb
│   ├── 04_model_development_and_evaluation.ipynb
│   └── 05_explainable_ai_analysis.ipynb
├── reports/                # Generated reports and summaries
├── results/                # Evaluation metrics, charts, and artifacts
│   ├── explanations/       # Local XAI archetype tables and attributions
│   ├── figures/            # Visualizations (Figs 01-19)
│   └── tables/             # Audit and performance tables
├── src/                    # Modular source code package
│   ├── analysis/           # Statistical analysis modules
│   ├── data/               # Ingestion and validation scripts
│   ├── evaluation/         # Metrics and performance evaluation
│   ├── features/           # Feature selection and transformation
│   ├── models/             # Model training pipelines
│   ├── preprocessing/      # Cleaning and missing-value handling
│   └── xai/                # Explainable AI (global & local interpretability)
├── tests/                  # Automated unit test suites
├── .gitignore              # Environment, data, and secret exclusions
├── PROJECT_RULES.md        # Binding project execution constraints & workflow
├── README.md               # Project overview and setup documentation
└── requirements.txt        # Verified core dependencies
```

---

## Environment & Setup Instructions

### 1. Prerequisites & Interpreter
The project uses the pre-configured local virtual environment:
* **Operating System:** Windows (AMD64)
* **Python Executable:** `.venv\Scripts\python.exe` (Python 3.13.14)

### 2. Dependencies
Core packages have been pre-installed in the virtual environment. To inspect or reproduce the environment:
```powershell
.venv\Scripts\python.exe -m pip list
```
The exact installed versions of core packages are pinned in `requirements.txt`.

### 3. Raw Data
The raw CDC BRFSS 2024 survey dataset archive resides at:
```text
data/raw/LLCP2024XPT.zip
```
* **Integrity:** SHA-256 verified and zip CRC checked.
* The original archive must remain preserved in `data/raw/` and is excluded from version control.

---

## Milestone Roadmap
- [x] **Milestone 1:** Repository audit, environment verification, and baseline configuration.
- [x] **Milestone 2:** Official CDC codebook & dataset metadata inspection.
- [x] **Milestone 3:** Data quality assessment & exploratory data analysis (EDA).
- [x] **Milestone 4:** Statistical analysis & survey-weighted knowledge discovery.
- [x] **Milestone 5:** Evidence-based predictive target selection & task formulation.
- [x] **Milestone 6:** Baseline and Improved machine learning model training & evaluation.
- [x] **Milestone 7:** Model evaluation, fairness checks, and explainability (XAI).
- [x] **Milestone 8 (Stages 1–3):** Interactive dashboard architecture, data precomputation, and zero-dependency standalone Plotly.js UI.
- [x] **Milestone 8 (Stage 4):** Dashboard integration audit and final packaging.
- [x] **Milestone 9:** Reproducibility verification, documentation, and final delivery.

---

## Milestone 8: Interactive Healthcare Analytics Dashboard

Milestone 8 delivers an academic single-page application (SPA) titled:
**"Population Health Analytics & Explainable ML — CDC BRFSS 2024"**

Built on the zero-dependency architecture approved in Stage 1, the dashboard consumes lightweight, precomputed JSON bundles (`dashboard/data/`, total 172 KB) and uses HTML5, CSS3, Vanilla JavaScript, and Plotly.js.

### 1. Dashboard Structure & Features
* **Section 1: Overview & Scope:** Cohort dimensions ($N = 452,464$), chronic condition feasibility assessment, champion model definition, methodology flow diagram, and prominent public health disclaimers.
* **Section 2: Population Health Explorer:** Interactive demographic and health construct selector (12 variables) supporting both unweighted sample counts and survey-weighted US population percentages, continuous risk factor summaries, and multivariable odds ratios.
* **Section 3: Model Performance:** Cross-architecture comparison of 4 supervised models, validation ROC, Precision-Recall, and Calibration deciles, plus final holdout test evaluation table ($N = 67,870$).
* **Section 4: Threshold Analysis:** Interactive threshold exploration slider ($\tau \in [0.02, 0.65]$) with real-time 2x2 confusion matrix updates, compared side-by-side with the immutable locked operational cutoff ($\tau = 0.080$).
* **Section 5: Explainable AI (XAI):** Global grouped permutation importance (38 original predictors), transformed indicator ranking, unscaled partial dependence profiles (Age, Physical Health Days, BMI), and 4 representative validation case archetypes (TP, FP, TN, FN) with local directional attributions ($\Delta p$) and Ridge surrogate weights ($R^2 > 0.85$).
* **Section 6: Limitations & Responsible Use:** Comprehensive guardrails detailing cross-sectional surveillance boundaries, lack of prospective validation, and strict prohibition against individual clinical diagnosis or symptom checking.

### 2. How to Launch the Dashboard Locally

Run Python's built-in HTTP server using the project virtual environment:

```powershell
# Recommended: Serve from dashboard directory
.venv\Scripts\python.exe -m http.server 8501 --directory dashboard
# Open in browser: http://localhost:8501/
```

Alternatively, from the repository root:
```powershell
# Alternative: Serve from project root
.venv\Scripts\python.exe -m http.server 8501
# Open in browser: http://localhost:8501/dashboard/
```

> **Network Note:** Plotly.js is loaded via CDN (`https://cdn.plot.ly/plotly-2.35.2.min.js`). An active internet connection is required to fetch the Plotly library for chart rendering. All underlying data bundles and styling are hosted locally with zero external package dependencies.

### 3. How to Run Dashboard Tests
```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```
