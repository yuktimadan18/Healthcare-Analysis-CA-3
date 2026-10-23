# CDC BRFSS 2024 Healthcare Analytics

**An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling**


<p align="center">
  <a href="https://yuktimadan18.github.io/Healthcare-Analytics-CA-3/">
    <strong>🌐 View Live Interactive Dashboard</strong>
  </a>
</p>

<p align="center">
  <a href="https://yuktimadan18.github.io/Healthcare-Analytics-CA-3/">
    <img src="https://img.shields.io/badge/Dashboard-Live%20on%20GitHub%20Pages-2E8B8B?style=for-the-badge" alt="Live Dashboard" />
  </a>
</p>


A reproducible healthcare analytics project using the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 combined landline and cellular survey dataset. The project combines population-health exploration, survey-aware statistical analysis, supervised machine learning, model evaluation, explainable AI (XAI), and an interactive dashboard.

> **Responsible-use statement:** This project studies associations in cross-sectional, self-reported public-health surveillance data. Its predictions are not clinical diagnoses, do not establish causation, and must not be used to make individual medical decisions.

---

## Table of Contents

- [Project at a Glance](#project-at-a-glance)
- [Problem Statement and Objectives](#problem-statement-and-objectives)
- [Dataset and Target Definition](#dataset-and-target-definition)
- [End-to-End Workflow](#end-to-end-workflow)
- [Analysis and Modeling](#analysis-and-modeling)
- [Model Evaluation](#model-evaluation)
- [Explainable AI](#explainable-ai)
- [Interactive Dashboard](#interactive-dashboard)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Environment and Setup](#environment-and-setup)
- [Reproducibility and Testing](#reproducibility-and-testing)
- [Project Roadmap](#project-roadmap)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Data Source](#data-source)

---

## Project at a Glance

| Item | Description |
|---|---|
| Data source | CDC Behavioral Risk Factor Surveillance System (BRFSS), 2024 |
| Data format | Combined landline and cellular survey data |
| Analytical task | Classification of self-reported cardiovascular disease history |
| Target variable | `_MICHD` |
| Eligible cohort | 452,464 respondents |
| Positive target records | 42,338 |
| Negative target records | 410,126 |
| Unweighted positive prevalence | 9.36% |
| Survey-weighted prevalence reported by the project | 6.80% |
| Modeling approach | Comparison of four supervised models |
| Selected model | `HistGradientBoostingClassifier` (unweighted GBDT) |
| Explainability | Grouped permutation importance, partial dependence, local perturbation explanations, and local surrogate models |
| Dashboard | HTML, CSS, JavaScript, precomputed JSON, and Plotly.js |
| Intended use | Educational research and population-health analysis |

## Problem Statement and Objectives

Cardiovascular disease is an important population-health concern, but public-health survey data are complex: observations are self-reported, outcomes may be imbalanced, and the survey design affects how population estimates should be interpreted. A useful analytical workflow must therefore do more than train a classifier. It should document data quality, examine population patterns, prevent data leakage, evaluate model performance beyond accuracy, explain model behavior, and communicate limitations clearly.

This project aims to:

1. Inspect and validate the BRFSS 2024 data and its documentation.
2. Explore data quality, variable distributions, and population-health patterns.
3. Conduct statistical analysis while distinguishing unweighted sample summaries from survey-weighted estimates.
4. Define a defensible predictive target and prevent target leakage.
5. Build reproducible preprocessing and feature-engineering steps.
6. Compare supervised classifiers using validation data and evaluate the selected model on a held-out test partition.
7. Explain global model behavior and selected local predictions.
8. Present the analytical outputs through an interactive dashboard.
9. Document reproducibility, ethical boundaries, and the limits of the evidence.

## Dataset and Target Definition

### Source

The dataset is the CDC BRFSS 2024 combined landline and cellular survey file.

- **Official data and documentation:** https://www.cdc.gov/brfss/annual_data/annual_2024.html
- The original archive is expected locally at `data/raw/LLCP2024XPT.zip`.
- The raw archive is excluded from version control; users should obtain it from the official CDC source.

### Cohort and target

The project reports **452,464 eligible respondents** after applying its target eligibility rules.

The target, `_MICHD`, represents a self-reported history of myocardial infarction or coronary heart disease, constructed from the source survey items `CVDINFR4` and `CVDCRHD4`.

| Target class | Records | Share of eligible cohort |
|---|---:|---:|
| Positive (`_MICHD` = 1) | 42,338 | 9.36% |
| Negative (`_MICHD` = 0) | 410,126 | 90.64% |
| **Total** | **452,464** | **100%** |

The positive class is the minority class, at approximately **9.69 negative records per positive record**. This imbalance is one reason the project reports precision-recall performance and threshold-dependent metrics instead of relying on accuracy alone.

### Survey design considerations

The project keeps survey design fields such as `_LLCPWT`, `_STSTR`, `_PSU`, and `_STATE` separate from the model predictor matrix. The project reports both unweighted sample results and survey-weighted population estimates where implemented. These quantities answer different questions and should not be treated as interchangeable.

The target-source variables used to construct `_MICHD` are excluded from predictors to reduce direct target leakage.

## End-to-End Workflow

The project is organized as a sequence of analytical stages. The numbered notebooks provide the main narrative of the work; supporting scripts, configuration files, results, and tests provide implementation and validation details.

1. **Repository and environment audit** — review the project structure, establish configuration and execution conventions, and verify the Python environment.
2. **Data documentation and ingestion** — inspect the official CDC codebook and metadata, validate the source data, and establish the target definition.
3. **Data quality assessment and EDA** — examine data quality, missingness, distributions, and exploratory population-health patterns.
4. **Statistical analysis** — investigate health-related patterns and calculate survey-weighted summaries where implemented.
5. **Target formulation and feature engineering** — establish the prediction task, exclude target-source variables from predictors, prepare features, and create separated data partitions.
6. **Model development and evaluation** — compare supervised models on validation data, select a champion model, lock the operating threshold, and report held-out test performance.
7. **Explainable AI** — investigate global predictor importance, partial dependence, and representative local predictions on the validation partition.
8. **Interactive dashboard** — package selected precomputed results in an interactive web interface.
9. **Final verification and delivery** — complete the integration audit, verify the final documentation and reproducibility instructions, and record the final delivery status.

### Data split and leakage prevention

The recorded modeling workflow uses a stratified **70% training / 15% validation / 15% test split**, with random seed **42**. Preprocessing is fitted using training data and then applied to the other partitions. Model selection and XAI analysis use the validation partition; the test partition is reserved for final evaluation.

The project reports a transformation from **38 original predictors to 190 transformed features**. See the feature-engineering notebook and the feature-mapping audit table for implementation details.

## Analysis and Modeling

### Exploratory and statistical analysis

The early analysis stages are designed to establish the quality and context of the data before predictive modeling. The project contains dedicated notebooks for data quality and EDA, statistical analysis, and feature engineering/splitting.

The repository includes generated figures and tables under `results/figures/` and `results/tables/`. Refer to those artifacts and the notebooks for the exact variables, statistical tests, estimates, and interpretation produced by each analysis. Do not interpret unweighted sample proportions as population estimates.

### Predictive modeling

The project compares four supervised model architectures. Validation-only model comparison selected `HistGradientBoostingClassifier` (unweighted gradient-boosted decision trees) as the champion, using **PR-AUC** as the primary model-selection metric because the positive class is relatively uncommon.

The operational threshold was selected on validation data using Youden's J criterion and locked at **0.080** before final test evaluation. This is a project-specific screening threshold, not a clinically validated cutoff.

## Model Evaluation

### Champion model results

| Metric | Reported result |
|---|---:|
| Validation PR-AUC | 0.3701 |
| Validation ROC-AUC | 0.8449 |
| Validation Brier score | 0.0700 |
| Locked operating threshold | 0.080 |
| Test PR-AUC | 0.3808 |
| Test ROC-AUC | 0.8492 |
| Test sensitivity | 83.91% |
| Test specificity | 70.45% |
| Test negative predictive value (NPV) | 97.69% |
| Test precision | 22.67% |
| Test F1 score | 0.3569 |
| Test balanced accuracy | 77.18% |

The test confusion matrix at the locked threshold was:

| | Predicted positive | Predicted negative |
|---|---:|---:|
| **Actual positive** | TP = 5,328 | FN = 1,022 |
| **Actual negative** | FP = 18,179 | TN = 43,341 |

These results illustrate the trade-off associated with the chosen threshold: sensitivity is relatively high, while precision is lower. A positive prediction is therefore not evidence that a respondent has a clinically confirmed condition. The high NPV is specific to this dataset, target definition, evaluation split, and threshold; it should not be assumed to generalize to another population.

## Explainable AI

Milestone 7 implemented global and local interpretability analyses for the frozen champion model on the **validation partition** (`N = 67,870`). The holdout test partition was kept separate from the explanation analysis.

### Methods

- **Bijective feature mapping:** links 38 original survey variables to 190 transformed features in `results/tables/xai_feature_mapping_audit.csv`.
- **Grouped permutation importance:** permutes indicators belonging to the same original variable together, using 10 reproducible repeats with seed 42. PR-AUC decrease is emphasized because of class imbalance.
- **Partial dependence profiles:** reports profiles for age, physical health limitation days, and BMI in interpretable units.
- **Local marginal perturbation attribution (`Δp`):** estimates how a model prediction changes when a feature group is perturbed relative to an empirical background sample (`M = 200`).
- **Local sparse linear surrogates:** fits weighted Ridge models to local perturbations to describe behavior near selected validation examples.

### Selected findings

The reported top five population predictors by grouped permutation importance were:

| Rank | Predictor | Mean PR-AUC decrease |
|---:|---|---:|
| 1 | `_AGE80` | 0.0772 |
| 2 | `GENHLTH` | 0.0434 |
| 3 | `_SEX` | 0.0380 |
| 4 | `CVDSTRK3` | 0.0306 |
| 5 | `PERSDOC3` | 0.0127 |

The reported age partial-dependence profile crosses the locked model threshold of 0.080 at approximately age 54 and rises above 23% by age 80. This is a model-derived marginal pattern, **not** a clinical age threshold or causal estimate.

The local analysis includes representative true-positive, false-positive, true-negative, and false-negative validation examples. These are model-behavior examples and should not be interpreted as individual medical assessments.

### Re-run the XAI scripts

From the project root in Windows PowerShell:

```powershell
.venv\Scripts\python.exe src/xai/global_importance.py
.venv\Scripts\python.exe src/xai/local_explanations.py
```

The executed notebook is `notebooks/05_explainable_ai_analysis.ipynb`. Generated figures and tables are saved in the project results directories according to the scripts' configuration.

## Interactive Dashboard

The dashboard is an academic single-page application titled **“Population Health Analytics & Explainable ML — CDC BRFSS 2024.”** It uses HTML, CSS, vanilla JavaScript, precomputed JSON data bundles, and Plotly.js.

### Dashboard sections

1. **Overview and scope** — cohort information, target context, champion model, methodology, and responsible-use notices.
2. **Population Health Explorer** — interactive exploration of 12 demographic and health constructs, including unweighted counts and survey-weighted percentages where available.
3. **Model Performance** — comparison of four supervised models and presentation of evaluation plots and final holdout metrics.
4. **Threshold Analysis** — interactive threshold slider with a dynamically updated confusion matrix, alongside the locked threshold of 0.080.
5. **Explainable AI** — grouped feature importance, transformed-feature ranking, partial dependence profiles, and representative local cases.
6. **Limitations and Responsible Use** — boundaries of cross-sectional surveillance data and warnings against individual clinical diagnosis.

### Run locally

From the repository root, run:

```powershell
.venv\Scripts\python.exe -m http.server 8501 --directory dashboard
```

Then open: http://localhost:8501/

Alternatively, serve from the repository root:

```powershell
.venv\Scripts\python.exe -m http.server 8501
```

Then open: http://localhost:8501/dashboard/

**Network requirement:** the current dashboard loads Plotly.js from `https://cdn.plot.ly/plotly-2.35.2.min.js`. An internet connection is required for the chart library to load. The dashboard data bundles and styling are stored in the repository.

> **Live deployment:** Add the public GitHub Pages URL here after deployment has been configured and the published site has been verified. A GitHub repository link is not itself a live dashboard link.

## Key Findings

- The target has an unweighted prevalence of **9.36%** in the eligible cohort, making class imbalance a relevant modeling consideration.
- The selected model achieved a held-out test PR-AUC of **0.3808** and ROC-AUC of **0.8492**.
- At the locked 0.080 threshold, test sensitivity was **83.91%**, specificity was **70.45%**, and precision was **22.67%**. These metrics should be interpreted together.
- Age, general health, sex, stroke history, and having a personal doctor were the five highest-ranked predictors in the reported grouped permutation analysis.
- The XAI analyses provide descriptions of model behavior, not evidence of causation or clinical validity.
- The dashboard makes selected analysis results explorable without requiring users to rerun the modeling pipeline.

## Repository Structure

```text
Project/
├── config/                 # Pipeline and model configuration
├── dashboard/              # Interactive dashboard and precomputed JSON
│   ├── components/         # Dashboard data, chart, and view components
│   └── data/               # Dashboard-ready result bundles
├── data/
│   ├── raw/                # Original CDC archive (excluded from Git)
│   ├── interim/            # Intermediate data (local)
│   └── processed/          # Processed data (local)
├── docs/                   # Methodological documentation and decisions
├── notebooks/
│   ├── 01_data_quality_and_eda.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_feature_engineering_and_splits.ipynb
│   ├── 04_model_development_and_evaluation.ipynb
│   └── 05_explainable_ai_analysis.ipynb
├── reports/                # Reports and summaries
├── results/
│   ├── explanations/       # Local explanation outputs
│   ├── figures/            # Generated figures
│   └── tables/             # Audit and performance tables
├── src/
│   ├── analysis/           # Statistical analysis
│   ├── data/               # Data ingestion and validation
│   ├── evaluation/         # Evaluation metrics
│   ├── features/           # Feature selection and transformation
│   ├── models/             # Model training pipelines
│   ├── preprocessing/      # Cleaning and missing-data handling
│   └── xai/                # Explainability pipelines
├── tests/                  # Automated tests
├── .gitignore
├── PROJECT_RULES.md        # Project workflow and constraints
├── README.md
└── requirements.txt        # Core dependency pins
```

**Note:** The raw survey archive, local datasets, virtual environment, and serialized model artifacts may be excluded from version control. The repository alone may not be sufficient to reproduce every analysis unless the official data and any required local artifacts are obtained and placed as expected.

## Environment and Setup

The original development environment recorded for this project is Windows (AMD64) with Python 3.13.14 and a project-local virtual environment.

### Dependencies

If the project virtual environment is already available, inspect installed packages with:

```powershell
.venv\Scripts\python.exe -m pip list
```

The project dependency pins are recorded in `requirements.txt`. To create a new environment from the repository root, if the required Python version is installed:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Some analyses may require the raw BRFSS archive and local intermediate or model artifacts that are not committed to Git. Download the official dataset from the CDC link above and follow the project's configuration and documentation before attempting a full pipeline rerun.

## Reproducibility and Testing

The project uses configuration files, modular source code, notebooks, generated result artifacts, and automated tests to support reproducibility.

Run the test suite from the project root:

```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

**Update this section with the latest verified test count** after running the command locally. Avoid hard-coding a count in multiple sections of the README unless it is refreshed whenever tests change.

For reproducibility, preserve the documented split seed, use the same target definition and preprocessing path, and ensure that model selection and explanation development do not use the final holdout test partition.

## Project Roadmap

The existing milestone roadmap is:

- [x] **Milestone 1:** Repository audit, environment verification, and baseline configuration.
- [x] **Milestone 2:** Official CDC codebook and dataset metadata inspection.
- [x] **Milestone 3:** Data quality assessment and exploratory data analysis.
- [x] **Milestone 4:** Statistical analysis and survey-weighted knowledge discovery.
- [x] **Milestone 5:** Evidence-based predictive target selection and task formulation.
- [x] **Milestone 6:** Baseline and improved model training and evaluation.
- [x] **Milestone 7:** Model evaluation, fairness checks, and explainability.
- [x] **Milestone 8 (Stages 1–3):** Dashboard architecture, data precomputation, and interface implementation.
- [x] **Milestone 8 (Stage 4):** Dashboard integration audit and final packaging.
- [x] **Milestone 9:** Reproducibility verification, documentation, and final delivery.

Only mark the remaining milestones complete after their acceptance checks have been run and the results recorded.

## Limitations and Responsible Use

- **Self-reported outcome:** `_MICHD` is constructed from self-reported survey responses and is not a clinically adjudicated diagnosis.
- **Cross-sectional design:** the analysis does not establish temporal ordering or causal relationships.
- **Survey design:** survey weights and design variables matter for population inference. The project reports weighted summaries where implemented, but the reported model metrics are unweighted; full complex-survey variance estimation for all analyses is not claimed here.
- **Class imbalance:** precision, recall, specificity, NPV, PR-AUC, and the threshold must be considered together. The chosen threshold is not a medical decision threshold.
- **Generalizability:** performance on this held-out partition does not establish external, prospective, or clinical validation.
- **Model explanations:** permutation importance, partial dependence, and local surrogate explanations describe aspects of the fitted model. They do not prove causation or that changing a predictor would change disease risk.
- **False positives and false negatives:** both occur at the chosen threshold. The model must not be used as a substitute for professional clinical assessment.
- **Reproducibility dependencies:** raw data and some local artifacts are not tracked in Git and may need to be acquired or generated separately.

## Data Source

Centers for Disease Control and Prevention (CDC). **Behavioral Risk Factor Surveillance System (BRFSS): Annual Survey Data, 2024.**

Official source and documentation: https://www.cdc.gov/brfss/annual_data/annual_2024.html

---

*This repository is an academic healthcare analytics project. It is intended for research and educational exploration, not for diagnosis, treatment, or individual-level clinical decision-making.*
