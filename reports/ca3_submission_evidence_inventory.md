# CA-3 Submission Evidence Inventory & Synthesis Report
## For IEEE Conference Paper (6 Pages) and Oral Presentation (12–15 Slides)

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Date:** October 21, 2026
**Author:** Research Team / Antigravity AI
**Scope:** Stage 1 — Authoritative evidence inventory, quantitative baseline, citation audit, and methodological synthesis.

---

## 1. Project Title & Research Objectives

### 1.1 Formal Title
> **"An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024"**

### 1.2 Primary Research Objective
To develop an end-to-end, reproducible, leakage-safe epidemiological analytics and machine learning framework for classifying self-reported history of coronary heart disease or myocardial infarction (`_MICHD`) in the **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024** cohort ($N = 452,464$). The study bridges the gap between public health survey methodology and modern machine learning by integrating:
1. Complex survey weighting and design awareness;
2. Empirical candidate outcome feasibility auditing;
3. Strict target leakage elimination;
4. Preprocessing with frozen parameter quarantine;
5. Transparent baseline (Logistic Regression) vs. nonlinear gradient boosting (HistGradientBoosting);
6. Natural calibration preservation and Youden's J threshold optimization for imbalanced population screening;
7. Model-agnostic, dependency-safe global and local explainability (grouped permutation, partial dependence, marginal perturbation, local Ridge surrogates);
8. An interactive, zero-dependency academic research surveillance dashboard.

### 1.3 Epidemiological & Regulatory Task Framing
- **Cross-Sectional Surveillance Classification:** The modeling target `_MICHD` records whether a survey respondent has **ever received a physician diagnosis** of myocardial infarction or coronary heart disease. It does **not** represent prospective prediction of future cardiovascular incidents, etiological causality, or real-time clinical diagnosis.
- **Clinical Utility Disclosure:** Although the champion model achieves high screening sensitivity ($80.4\% - 83.9\%$) and high negative predictive value ($97.7\% - 98.4\%$), **clinical diagnostic validity and clinical screening utility have not been established.** The framework is strictly an epidemiological risk stratification and surveillance research platform, not a medical device.

---

## 2. Research Gaps & Distinct Project Contributions

### 2.1 Research Gaps Identified in Literature
Based on the comprehensive 26-page literature review and recent empirical studies (e.g., Clark et al. 2021 [1]; Morgenstern et al. 2020 [27]; Islam et al. 2026 [17]; Lam et al. 2026 [18]):

| Identified Research Gap | Description in Existing Literature | Citation Support |
| :--- | :--- | :--- |
| **1. Predetermined Disease Silos** | Most studies arbitrarily predetermine a single disease (e.g., diabetes, asthma, depression) without empirical comparison of candidate public health outcomes or codebook structure. | [2], [4], [7], [9], [11], [12], [20] |
| **2. Target Leakage & Data Snooping** | Common failure to account for CDC calculated variable logic, inadvertently including constituent questions (e.g., keeping heart attack `CVDINFR4` when predicting `_MICHD`) or fitting preprocessors across full datasets prior to splitting. | [14], [17], [29], [32] |
| **3. Complex Survey Weight Misuse** | Machine learning studies either ignore survey expansion weights (`_LLCPWT`) completely or incorrectly apply raw weights as cell frequencies in loss functions, distorting sample variance by orders of magnitude. | [18], [27], [30] |
| **4. Default Threshold Failure on Skewed Data** | Pervasive reliance on the arbitrary 0.50 classification cutoff in imbalanced cohorts ($\sim 7\%-9\%$ prevalence), which silently misses $85\%-90\%$ of positive cases while reporting deceptive $\sim 91\%$ accuracy. | [3], [4], [9], [21], [24] |
| **5. Calibration Distortion from Resampling** | Heavy use of synthetic oversampling (SMOTE, ADASYN) destroys predicted probability calibration, rendering risk probabilities clinically uninterpretable. | [2], [3], [9], [22], [27] |
| **6. Uncritical XAI Claims (SHAP/LIME Default)** | Treating post-hoc feature importance as causal biological truth, or applying one-hot permutation that dilutes multi-category features and violates indicator constraints. | [8], [10], [16], [34], [35], [38] |

### 2.2 Substantive Contributions of the Implemented Project
1. **Systematic 10-Outcome Feasibility Audit:** Evaluated 10 chronic conditions across 457,670 raw records before selecting `_MICHD` based on public health burden, completeness (98.86%), and non-trivial class balance ($9.36\%$ unweighted, $6.80\%$ weighted; $1:9.69$ ratio).
2. **Strict Leakage-Safe Architecture:** Excluded direct mathematical constituents (`CVDINFR4`, `CVDCRHD4`) and high-missingness routing artifacts (`POORHLTH`, `PREDIAB2`). Preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) was fitted strictly on `X_train` and applied frozen via `transform()` to validation and test partitions.
3. **Dual Metric Evaluation & Weight Normalization:** Preserved raw unweighted loss during gradient estimation to maintain natural calibration, while reporting normalized survey-weighted population estimates on holdout evaluation.
4. **Principled Decision Threshold Optimization:** Identified operational threshold $\tau = 0.080$ via validation Youden's J maximization, elevating sensitivity from $10.9\%$ to $83.9\%$ on holdout test data with $97.7\%$ NPV.
5. **Dependency-Safe, Grouped XAI:** Implemented grouped permutation importance (shuffling all 190 dummy columns across 38 parent variables simultaneously to eliminate categorical dilution), unscaled continuous partial dependence, and local Ridge surrogate models ($R^2 \in [0.931, 0.969]$) without external dependencies.
6. **Zero-Dependency Academic Dashboard:** Implemented a modular Plotly.js SPA visualizing population strata, validation trade-offs, calibration deciles, global/local attributions, and non-causal governance disclosures.

### 2.3 Critical Inconsistency Resolution: Proposed Review vs. Implemented Project
> [!IMPORTANT]
> **Explicit Methodological Correction:** The literature review frequently discusses **SHAP** and **LIME** as generic healthcare XAI tools. However, in the implemented project, neither `shap` nor `lime` was installed or utilized. The actual implemented XAI methods are:
> 1. **Grouped Permutation Feature Importance** (PR-AUC loss metric, 10 repeats);
> 2. **Recursion Partial Dependence Profiles** (dynamically unscaled to epidemiological units);
> 3. **Grouped Marginal Perturbation Attribution ($\Delta p$)** against an empirical background ($M = 200$);
> 4. **Local Ridge Linear Surrogates** with exponential distance weighting ($N = 1,000$ perturbations).
>
> *The IEEE paper and presentation must strictly present these implemented methods and avoid claiming SHAP or LIME was used.*

---

## 3. Dataset Description, Cohort Definition & Class Distribution

### 3.1 Raw Dataset Provenance & Integrity
- **Data Source:** Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS) 2024 Annual Public Release.
- **Raw Archive:** `data/raw/LLCP2024XPT.zip` (83,007,956 bytes, ~79.16 MB).
- **Cryptographic Hash (SHA-256):** `d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4`.
- **Integrity Status:** Verified bit-for-bit identical to live CDC distribution server (`zipfile.testzip()` passed with 0 CRC errors).
- **Survey Size:** Exactly **457,670 respondents** across all 50 states, DC, and US territories; **301 raw variables**.
- *Source Documentation:* [`docs/PROJECT_DECISIONS.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/docs/PROJECT_DECISIONS.md) (Milestones 1 & 2).

### 3.2 Target Definition & Cohort Derivation
- **Target Variable:** `_MICHD` (Calculated variable: Reported history of coronary heart disease or myocardial infarction).
- **Binary Coding:**
  - Positive ($y = 1$): `_MICHD == 1` (Respondent reported physician-diagnosed heart attack or angina/CHD).
  - Negative ($y = 0$): `_MICHD == 2` (Respondent reported no history of heart attack or angina/CHD).
  - Excluded ($y = \text{NaN}$): `_MICHD == BLANK` or missing ($N = 5,206$, $1.14\%$ of raw survey).
- **Final Validated Analytical Cohort:** Exactly **452,464 respondents** (98.86% retention).
- *Source Artifact:* [`results/tables/dataset_partition_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/dataset_partition_summary.csv).

### 3.3 Cohort Class Distribution & Imbalance
*Source:* [`results/tables/dataset_partition_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/dataset_partition_summary.csv)

| Cohort Partition | Total Records | Positive ($y=1$) | Negative ($y=0$) | Unweighted Prevalence | Weighted Prevalence (`_LLCPWT`) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (70.0%)** | 316,724 | 29,637 | 287,087 | **9.36%** | **6.82%** | 1 : 9.69 |
| **Validation (15.0%)** | 67,870 | 6,351 | 61,519 | **9.36%** | **6.77%** | 1 : 9.69 |
| **Holdout Test (15.0%)**| 67,870 | 6,350 | 61,520 | **9.36%** | **6.76%** | 1 : 9.69 |
| **Full Eligible Cohort** | **452,464** | **42,338** | **410,126** | **9.36%** | **6.80%** | **1 : 9.69** |

*Methodological Note:* The divergence between unweighted sample prevalence ($9.36\%$) and survey-weighted population prevalence ($6.80\%$) reflects disproportionate telephone response rates among older individuals, requiring explicit survey weighting during population inference.

---

## 4. Preprocessing, Leakage Prevention & Partitioning Protocol

### 4.1 Strict Target Leakage Elimination
*Source:* [`results/tables/feature_inclusion_exclusion_audit.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/feature_inclusion_exclusion_audit.csv)
- **Constituent Variables Excluded:**
  - `CVDINFR4` (Ever told you had a heart attack / myocardial infarction): Direct mathematical constituent of `_MICHD`. Banned.
  - `CVDCRHD4` (Ever told you had angina or coronary heart disease): Direct mathematical constituent of `_MICHD`. Banned.
- **Survey Design Variables Isolated:**
  - `_LLCPWT` (Final expansion weight), `_STSTR` (Stratum), `_PSU` (Primary sampling unit), and `_STATE` (FIPS code) were isolated in `cohort_metadata.csv.gz` for evaluation and subgroup auditing; excluded from predictive features.
- **Redundant / Structural Skip Exclusions (16 variables):**
  - Excluded variables with high structural missingness (`POORHLTH` [41.5% skip], `PREDIAB2` [65.2% skip]) or coarse binned duplicates (`_AGE65YR`, `_RFHLTH`, `_BMI5CAT`, `_RFSMOK3`, `_TOTINDA`).

### 4.2 Feature Engineering & Matrix Dimensionality
The feature matrix is constructed from **38 source predictors**:
- **Continuous Numeric Features (4 variables):**
  1. `_AGE80`: Imputed reported age in years (18–80, top-coded at 80).
  2. `PHYSHLTH_DAYS`: Days of poor physical health in past 30 days (0–30; code 88 recoded to 0; 77/99 recoded to NaN).
  3. `MENTHLTH_DAYS`: Days of poor mental health in past 30 days (0–30; code 88 recoded to 0; 77/99 recoded to NaN).
  4. `BMI_NUMERIC`: Body mass index calculated as `_BMI5 / 100.0` ($\text{kg/m}^2$).
- **Categorical Features (34 variables):**
  - Encompasses biological sex (`_SEX`), race/ethnicity (`_RACE`), education (`_EDUCAG`), income (`_INCOMG1`), marital status (`MARITAL`), employment (`EMPLOY1`), urbanicity (`_METSTAT`, `_URBSTAT`), general health (`GENHLTH`), healthcare coverage (`_HLTHPL2`, `PERSDOC3`, `MEDCOST1`, `CHECKUP1`), comorbid conditions (`CVDSTRK3`, `DIABETE4`, `ASTHMA3`, `CHCCOPD3`, `ADDEPEV3`, `CHCKDNY2`, `HAVARTH4`, `CHCSCNC1`, `CHCOCNC1`), behavioral risks (`SMOKE100`, `_SMOKER3`, `_CURECI3`, `DRNKANY6`, `_RFDRHV9`), and functional disabilities (`DEAF`, `BLIND`, `DECIDE`, `DIFFWALK`, `DIFFDRES`, `DIFFALON`).
- **Missing Value Handling Strategy:**
  - *Numeric Features:* Median imputation followed by `StandardScaler` standard normal scaling ($\mu = 0, \sigma = 1$).
  - *Categorical Features:* Genuine missingness was imputed with an explicit numeric sentinel (`-1.0`), ensuring no collision with valid BRFSS codes, followed by `OneHotEncoder(handle_unknown='ignore')`. Refusal (`9`/`99`) and don't-know (`7`/`77`) codes were retained as distinct valid behavioral categories.
- **Transformed Feature Dimensionality:** Exactly **190 preprocessed features** (4 continuous + 164 valid one-hot categories + 22 explicit -1.0 missing sentinel indicators).
- *Source Artifact:* [`models/preprocessor.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/preprocessor.joblib).

### 4.3 Data Quarantine & Frozen Pipeline Contract
- **Stratified Disjoint Splitting:** Partitions were created using `train_test_split(test_size=0.30, stratify=y, random_state=42)`, followed by splitting the temporary 30% holdout evenly into 15% validation and 15% test splits.
- **Disjointness Verified:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Frozen Pipeline Contract:** `models/preprocessor.joblib` was fitted **strictly on `X_train`** ($N = 316,724$). Validation and test sets were transformed via `.transform()` with zero parameter refitting, preventing all information leakage across folds.

---

## 5. Candidate Models & Hyperparameter Configurations

Four candidate supervised learning models were developed using native `scikit-learn 1.9.1`, without external boosting libraries, using fixed random seed 42:

*Source:* [`src/models/model_config.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/models/model_config.py) & [`reports/milestone6_model_development_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone6_model_development_audit.md)

| Model Key | Model Description | Algorithm Class | Key Hyperparameters & Weighting | Artifact Location |
| :--- | :--- | :--- | :--- | :--- |
| `logistic_regression_unweighted` | Baseline Logistic Regression (Unweighted) | Linear Model | `C=1.0`, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`, `class_weight=None` | [`models/baseline_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/baseline_logistic_regression.joblib) |
| `logistic_regression_balanced` | Baseline Logistic Regression (Class-Weighted) | Linear Model | `C=1.0`, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`, `class_weight='balanced'` ($w_1 \approx 5.34, w_0 \approx 0.55$) | [`models/weighted_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_logistic_regression.joblib) |
| `hist_gradient_boosting_unweighted` | **HistGradientBoosting (Champion Model)** | GBDT | `max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=1.0`, `random_state=42`, `class_weight=None` | [`models/improved_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib) |
| `hist_gradient_boosting_balanced` | HistGradientBoosting (Class-Weighted) | GBDT | `max_iter=150`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=1.0`, `random_state=42`, `class_weight='balanced'` | [`models/weighted_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_hist_gradient_boosting.joblib) |

---

## 6. Computing Environment & Installed Package Manifest

*Source:* Environment inspection via Python 3.13 runtime on host system and [`requirements.txt`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/requirements.txt):

```text
Platform: Windows 11 (10.0.26300-SP0) AMD64
Python Interpreter: .venv\Scripts\python.exe (Python 3.13.14, 64-bit)
Core Processing:
  - numpy == 2.5.4
  - pandas == 3.0.6
  - scipy == 1.18.1
SAS Ingestion:
  - pyreadstat == 1.3.6
Modeling & Statistics:
  - scikit-learn == 1.9.1
  - statsmodels == 0.15.0
  - joblib == 1.6.0
Visualization & Dashboard:
  - matplotlib == 3.11.2
  - seaborn == 0.13.2
  - plotly == 7.1.0
Interactive Notebooks:
  - ipykernel == 7.4.0
  - jupyterlab == 4.6.4
  - notebook == 7.6.3
  - openpyxl == 3.1.5
```

---

## 7. Model Selection, Calibration Audit & Validation Results

### 7.1 Primary Selection Metric Rationale: PR-AUC
In low-prevalence surveillance settings ($9.36\%$ unweighted prevalence, $\approx 1:10$ ratio), the **Precision-Recall Area Under the Curve (PR-AUC / Average Precision)** was designated as the primary selection criterion. Unlike ROC-AUC—which can appear deceptively elevated ($> 0.84$) due to overwhelming true negative volume—PR-AUC directly reflects positive predictive value across all operational recall thresholds.

### 7.2 Validation Set Comparison Across All 4 Candidates ($N = 67,870$)
*Source:* [`results/tables/model_validation_comparison.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv)

| Model Key & Architecture | Validation PR-AUC | Validation ROC-AUC | Validation Brier Score | Brier Skill Score (BSS) | Mean Predicted Probability | Sensitivity (@ 0.50) | Specificity (@ 0.50) | F1 Score (@ 0.50) | Rank by PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`hist_gradient_boosting_unweighted` (Champion)** | **0.3701** | **0.8449** | **0.0700** | **+0.1746** | **0.0934** | 10.66% | 99.19% | 0.1799 | **1** |
| `hist_gradient_boosting_balanced` | 0.3687 | 0.8450 | 0.1652 | -0.9481 | 0.3468 | 80.35% | 73.47% | 0.3675 | 2 |
| `logistic_regression_unweighted` | 0.3658 | 0.8413 | 0.0705 | +0.1691 | 0.0933 | 12.61% | 98.92% | 0.2049 | 3 |
| `logistic_regression_balanced` | 0.3646 | 0.8413 | 0.1661 | -0.9580 | 0.3547 | 78.40% | 74.85% | 0.3715 | 4 |

### 7.3 Calibration & Probability Integrity Findings
- **Unweighted Models Preserve Natural Calibration:** The unweighted GBDT achieved a Brier score of **$0.0700$** and a positive Brier Skill Score ($+0.1746$), with a mean predicted probability of **$0.0934$**, aligning almost perfectly with the true empirical prevalence ($0.0936$). Decile calibration curves ([`fig12_calibration_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig12_calibration_curves_validation.png)) follow the ideal diagonal.
- **Class-Weighting Distorts Probabilities:** Using `class_weight='balanced'` severely distorted predicted probabilities toward $\sim 0.35$, degrading the Brier score to $0.1652$ and producing severe negative Brier Skill Scores ($-0.9481$). Consequently, unweighted training combined with post-hoc threshold tuning was mathematically confirmed as superior to heuristic class weighting.

### 7.4 Threshold Selection via Youden's J Index
*Source:* [`results/tables/threshold_optimization_analysis.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/threshold_optimization_analysis.csv) & [`models/model_selection_summary.json`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json)
- Evaluated decision thresholds across $\tau \in [0.02, 0.90]$ with step $0.01$.
- At default $\tau = 0.50$, the champion model detects only $10.66\%$ of positive cases (missing $89.34\%$).
- The optimal operational threshold was selected by maximizing **Youden's J Index** ($J = \text{Sensitivity} + \text{Specificity} - 1$) on the validation partition, identifying **$\tau = 0.080$**:
  - Threshold $\tau = 0.080$: Sensitivity = **83.29%**, Specificity = **70.28%**, PPV = **22.44%**, NPV = **97.60%**, F1 Score = **0.3536**, Youden's J = **0.5357**.
- **Formal Lock:** The champion model (`hist_gradient_boosting_unweighted`) and operational threshold ($\tau = 0.080$) were formally locked into `models/model_selection_summary.json` prior to opening the holdout test set.

---

## 8. Final Holdout Test Set Evaluation & Exact Confusion Matrices

Holdout test evaluation was performed in a single forward pass ($N = 67,870$) strictly after model and threshold locking.

### 8.1 Quantitative Test Performance Summary
*Source:* [`results/tables/final_test_evaluation_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/final_test_evaluation_summary.csv)

| Evaluation Mode | Cutoff ($\tau$) | PR-AUC | ROC-AUC | Sensitivity | Specificity | PPV (Precision) | NPV | F1 Score | Balanced Accuracy | Brier Score | Youden's J |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted Sample** | 0.500 | 0.3808 | 0.8492 | 10.91% | 99.26% | 60.26% | 91.52% | 0.1848 | 55.09% | 0.0694 | 0.1017 |
| **Unweighted Sample** | **0.080** | **0.3808** | **0.8492** | **83.91%** | **70.45%** | **22.67%** | **97.70%** | **0.3569** | **77.18%** | **0.0694** | **0.5436** |
| **Survey-Weighted (`_LLCPWT`)** | 0.500 | 0.3594 | 0.8726 | 9.34% | 99.58% | 61.72% | 93.81% | 0.1623 | 54.46% | 0.0514 | 0.0892 |
| **Survey-Weighted (`_LLCPWT`)** | **0.080** | **0.3594** | **0.8726** | **80.44%** | **79.46%** | **22.12%** | **98.42%** | **0.3469** | **79.95%** | **0.0514** | **0.5990** |

### 8.2 Exact Confusion Matrix Counts
*Sources:*
- Holdout Test Counts: Figure 14 ([`results/figures/fig14_champion_test_confusion_matrices.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig14_champion_test_confusion_matrices.png)) & Test Holdout ($N = 67,870$).
- Validation Counts: [`models/model_selection_summary.json`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json) lines 27–30 ($N = 67,870$).

```
====================================================================================================
HOLDOUT TEST CONFUSION MATRIX (N = 67,870) — DEFAULT THRESHOLD (τ = 0.500)
----------------------------------------------------------------------------------------------------
                                Predicted Negative (No CAD)    Predicted Positive (CAD)       Total
Actual Negative (No CAD):       TN = 61,063                    FP = 457                       61,520
Actual Positive (CAD):          FN = 5,657                     TP = 693                        6,350
Total:                          66,720                         1,150                          67,870
Metrics: Sensitivity = 10.91% | Specificity = 99.26% | PPV = 60.26% | NPV = 91.52%
====================================================================================================

====================================================================================================
HOLDOUT TEST CONFUSION MATRIX (N = 67,870) — LOCKED OPERATIONAL THRESHOLD (τ = 0.080)
----------------------------------------------------------------------------------------------------
                                Predicted Negative (No CAD)    Predicted Positive (CAD)       Total
Actual Negative (No CAD):       TN = 43,341                    FP = 18,179                    61,520
Actual Positive (CAD):          FN = 1,022                     TP = 5,328                      6,350
Total:                          44,363                         23,507                         67,870
Metrics: Sensitivity = 83.91% | Specificity = 70.45% | PPV = 22.67% | NPV = 97.70%
====================================================================================================

====================================================================================================
VALIDATION SET CONFUSION MATRIX (N = 67,870) — LOCKED OPERATIONAL THRESHOLD (τ = 0.080)
----------------------------------------------------------------------------------------------------
                                Predicted Negative (No CAD)    Predicted Positive (CAD)       Total
Actual Negative (No CAD):       TN = 43,236                    FP = 18,283                    61,519
Actual Positive (CAD):          FN = 1,061                     TP = 5,290                      6,351
Total:                          44,297                         23,573                         67,870
Metrics: Sensitivity = 83.29% | Specificity = 70.28% | PPV = 22.44% | NPV = 97.60%
====================================================================================================
```

*Key Analytical Outcome:* Adjusting the threshold to $\tau = 0.080$ rescues **$4,635$ additional true positive cases** ($5,328$ vs. $693$) on the holdout test set, increasing disease detection by nearly $8\times$ while maintaining a $97.7\%$ negative predictive value.

---

## 9. Explainable AI (XAI) Methods, Findings & Fidelity Measurements

*Source:* Milestone 7 Audit Reports ([`reports/milestone7_global_importance_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md), [`reports/milestone7_stage3_local_explainability_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_stage3_local_explainability_audit.md)). All XAI evaluations were conducted strictly on the **validation partition** ($N = 67,870$).

### 9.1 Global Grouped Permutation Importance
- **Methodology:** Standard permutation of one-hot encoded columns dilutes feature importance across dummy variables and can generate unrealistic combinations. To resolve this, all dummy columns belonging to a parent survey variable were shuffled simultaneously using a shared permutation index over 10 random shuffles (seed 42).
- **Ranking Metric:** PR-AUC reduction ($\Delta \text{PR-AUC}$) from baseline validation PR-AUC ($0.3701$).
- **Top 10 Global Predictors:**
  *Source:* [`results/tables/xai_global_variable_importance.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv)
  1. `_AGE80` (Continuous Age): $\Delta \text{PR-AUC} = \mathbf{0.0772} \pm 0.0016$
  2. `GENHLTH` (General Health Status): $\Delta \text{PR-AUC} = \mathbf{0.0434} \pm 0.0017$
  3. `_SEX` (Biological Sex): $\Delta \text{PR-AUC} = \mathbf{0.0380} \pm 0.0013$
  4. `CVDSTRK3` (Stroke History): $\Delta \text{PR-AUC} = \mathbf{0.0306} \pm 0.0014$
  5. `PERSDOC3` (Personal Doctor Access): $\Delta \text{PR-AUC} = \mathbf{0.0127} \pm 0.0007$
  6. `CHCKDNY2` (Kidney Disease History): $\Delta \text{PR-AUC} = \mathbf{0.0122} \pm 0.0006$
  7. `_RACE` (Race / Ethnicity): $\Delta \text{PR-AUC} = \mathbf{0.0101} \pm 0.0007$
  8. `DIABETE4` (Diabetes History): $\Delta \text{PR-AUC} = \mathbf{0.0097} \pm 0.0005$
  9. `_SMOKER3` (Smoking Status): $\Delta \text{PR-AUC} = \mathbf{0.0076} \pm 0.0005$
  10. `DIFFWALK` (Serious Difficulty Walking): $\Delta \text{PR-AUC} = \mathbf{0.0075} \pm 0.0006$

### 9.2 Partial Dependence Profiles (PDP)
- **Continuous Unscaling:** Standardized values were unscaled back to natural units using training mean and scale parameters from `models/preprocessor.joblib`.
- **Profiles Observed:**
  - *Age (`_AGE80`):* Non-linear upward risk gradient accelerating markedly after age 55; marginal probability rises from $< 0.04$ at age 30 to $> 0.16$ at age 80.
  - *Physical Health Days (`PHYSHLTH_DAYS`):* Steeper increase in risk between 0 and 10 days of physical limitation, plateauing towards 30 days.
  - *BMI (`BMI_NUMERIC`):* Gentle, steady risk increase across normal, overweight, and class I/II obesity.

### 9.3 Local Explainability & Surrogate Fidelity
- **Methods:** Evaluated 4 validation case archetypes (representing the median predicted probability of each confusion matrix quadrant at $\tau = 0.080$):
  - *Grouped Marginal Perturbation ($\Delta p$):* Measures directional probability change relative to an empirical background ($M = 200$).
  - *Local Ridge Surrogates:* Weighted linear models fitted on $N = 1,000$ local Gaussian/multinomial perturbations weighted by an exponential distance kernel.
- **Local Surrogate Fidelity ($R^2$):**
  *Source:* [`results/explanations/xai_case_archetypes_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/explanations/xai_case_archetypes_summary.csv)
  - True Positive Archetype (`VALIDATION_ID_56065`, $\hat{p} = 0.2673$): Local Surrogate **$R^2 = 0.9587$**
  - False Positive Archetype (`VALIDATION_ID_279669`, $\hat{p} = 0.1664$): Local Surrogate **$R^2 = 0.9691$**
  - True Negative Archetype (`VALIDATION_ID_253891`, $\hat{p} = 0.0163$): Local Surrogate **$R^2 = 0.9312$**
  - False Negative Archetype (`VALIDATION_ID_44471`, $\hat{p} = 0.0436$): Local Surrogate **$R^2 = 0.9458$**
  - *Mean Local Surrogate Fidelity across Archetypes:* **$R^2 = 0.9512$** (high explanatory fidelity).

---

## 10. Statistical Analysis, Survey Weighting & Non-Causal Disclosures

### 10.1 Multivariable Survey-Weighted Logistic Regression
*Source:* [`results/tables/multivariable_logistic_regression_or.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/multivariable_logistic_regression_or.csv) ($N = 385,120$ complete cases across survey-weighted variables; weights normalized to sample size to prevent artificial SE deflation):
- **Age 65+ Years (`age_65plus`):** Adjusted Odds Ratio **$\text{aOR} = 4.57$** (95% CI: $4.45 - 4.70$; Cluster-robust CI: $4.32 - 4.84$, $p < 0.001$).
- **Fair/Poor General Health (`fair_poor_health`):** **$\text{aOR} = 3.10$** (95% CI: $3.01 - 3.19$; Cluster-robust CI: $2.93 - 3.29$, $p < 0.001$).
- **Kidney Disease (`chckdny2`):** **$\text{aOR} = 2.41$** (95% CI: $2.31 - 2.52$, $p < 0.001$).
- **Diabetes (`diabetes`):** **$\text{aOR} = 1.99$** (95% CI: $1.92 - 2.05$, $p < 0.001$).
- **Male Biological Sex (`male_sex`):** **$\text{aOR} = 1.73$** (95% CI: $1.68 - 1.78$, $p < 0.001$).
- **Current Smoker (`current_smoker`):** **$\text{aOR} = 1.58$** (95% CI: $1.52 - 1.64$, $p < 0.001$).
- **Physical Inactivity (`physical_inactivity`):** **$\text{aOR} = 1.21$** (95% CI: $1.17 - 1.24$, $p < 0.001$).
- **Overweight/Obese (`bmi_overweight_obese`):** **$\text{aOR} = 1.20$** (95% CI: $1.17 - 1.24$, $p < 0.001$).

### 10.2 Survey Weighting Disclosures
1. Machine learning models were trained by optimizing unweighted sample loss across individual respondents.
2. Reported survey-weighted test metrics reflect population-weighted point estimates calculated using CDC sample weights (`_LLCPWT`).
3. These estimates do not incorporate complex-survey variance estimation (e.g., Taylor series linearization across primary sampling units `_PSU` and strata `_STSTR`).

---

## 11. Figure & Table Inventory Suitable for Paper & Slides

### 11.1 Recommended Figures for 6-Page IEEE Paper & Presentation
All figures exist in high-resolution 300 DPI PNG format under `results/figures/`:

| Figure ID & Path | Title / Content | Best Fit (Paper vs. Slides) | Core Message |
| :--- | :--- | :---: | :--- |
| [`fig7_odds_ratios_forest_plot.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig7_odds_ratios_forest_plot.png) | Multivariable Logistic Regression Forest Plot | Both (Paper Sec. III, Slide 4) | Displays adjusted odds ratios for clinical & behavioral risk factors. |
| [`fig11_pr_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig11_pr_curves_validation.png) | Precision-Recall Curves (Validation Set) | Both (Paper Sec. IV, Slide 6) | Demonstrates GBDT superiority over Logistic Regression against 9.36% baseline. |
| [`fig12_calibration_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig12_calibration_curves_validation.png) | Decile Reliability Calibration Diagrams | Paper (Sec. IV) | Shows unweighted models follow the diagonal while weighted models distort risk. |
| [`fig13_threshold_tradeoffs.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig13_threshold_tradeoffs.png) | Decision Threshold Metric Trade-off Curves | Both (Paper Sec. IV, Slide 7) | Justifies selection of $\tau = 0.080$ via Youden's J maximization. |
| [`fig14_champion_test_confusion_matrices.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig14_champion_test_confusion_matrices.png) | Test Confusion Matrices (0.50 vs. 0.080) | Both (Paper Sec. V, Slide 8) | Shows 8x surge in detected cases (693 to 5,328) on the holdout test partition. |
| [`fig16_global_variable_importance.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig16_global_variable_importance.png) | Grouped Permutation Variable Importance | Both (Paper Sec. V, Slide 9) | Highlights top predictors: Age, General Health, Sex, Stroke, Healthcare Access. |
| [`fig18_partial_dependence_profiles.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png) | Unscaled Partial Dependence Profiles | Paper (Sec. V) | Illustrates non-linear age threshold effect (>55 years) on predicted probability. |
| [`fig19_local_case_attributions.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig19_local_case_attributions.png) | 4-Quadrant Local Case Archetypes ($\Delta p$) | Both (Paper Sec. V, Slide 10) | Contrasts positive attributions in TP/FP against protective attributions in TN/FN. |

### 11.2 Recommended Tables for 6-Page IEEE Paper
1. **Table I: Cohort Derivation and Partition Class Distribution** (from Section 3.3).
2. **Table II: Multivariable Survey-Weighted Odds Ratios for `_MICHD`** (from Section 10.1).
3. **Table III: Validation Performance Comparison of Candidate Models** (from Section 7.2).
4. **Table IV: Final Holdout Test Evaluation and Confusion Matrix Summary** (from Section 8.1 & 8.2).

---

## 12. GitHub Repository Status & Packaging Tasks

### 12.1 Git Repository Status
- **Current State:** The local project directory is **not yet initialized as a Git repository** (`fatal: not a git repository`).
- **Ignore Rules Verified:** `.gitignore` is properly configured, excluding large datasets (`data/raw/`, `data/interim/`, `data/processed/`), virtual environments (`.venv/`), and compiled caches while retaining source code, tests, documentation, and notebooks.

### 12.2 Remaining Packaging Tasks for Final Submission
1. **Initialize Git Repository:** Run `git init` and configure local commit metadata.
2. **Verify Staging Exclusions:** Confirm that large data archives (`LLCP2024XPT.zip`, `.csv.gz` files) and model binaries are excluded by `.gitignore`.
3. **Stage Core Code and Assets:** Stage `src/`, `tests/`, `dashboard/`, `docs/`, `reports/`, `notebooks/`, `README.md`, `requirements.txt`, and `PROJECT_RULES.md`.
4. **Initial Commit:** Create a structured initial commit (`v1.0.0-ca3-submission`).
5. **Add Remote & Push:** Connect to user's designated GitHub remote repository.

---

## 13. Comprehensive Reference Audit (40 Candidate Papers)

Every reference from the provided literature review was audited against official digital publication databases (Crossref, PubMed, IEEE Xplore, ScienceDirect).

| Ref # | Cited Authors & Title | Journal & Publication Year | Documented DOI | Verified Status & Notes |
| :---: | :--- | :--- | :--- | :--- |
| **[1]** | C. R. Clark et al., "Predicting self-rated health across the life course..." | *J. Gen. Intern. Med.*, 2021 | `10.1007/s11606-020-06438-1` | **VERIFIED.** (Vol. 36, No. 5, pp. 1181–1188). |
| **[2]** | Z. Ullah et al., "Detecting high-risk factors and early diagnosis of diabetes..." | *Comput. Intell. Neurosci.*, 2022 | `10.1155/2022/2557795` | **VERIFIED WITH CORRECTION.** Review cited *J. Healthc. Eng.*; actual journal is *Comput. Intell. Neurosci.* (Hindawi). |
| **[3]** | M. M. Chowdhury et al., "An investigation of machine learning algorithms..." | *Healthc. Anal.*, 2024 | `10.1016/j.health.2023.100297` | **VERIFIED.** (Vol. 5, Art. 100297). |
| **[4]** | N. Budhathoki et al., "Predicting asthma using imbalanced data modeling..." | *PLOS ONE*, 2023 | `10.1371/journal.pone.0295427` | **VERIFIED.** (Vol. 18, No. 12, Art. e0295427). |
| **[5]** | S. Kamal et al., "Using machine learning to predict poor mental health status..." | *Am. J. Geriatr. Psychiatry*, 2023 | `10.1016/j.jagp.2022.12.234` | **VERIFIED.** (Vol. 31, No. 3, Suppl., pp. S71–S72; conference abstract). |
| **[6]** | A. M. Oladayo et al., "Using machine learning algorithms to investigate factors..." | *Spec. Care Dentist.*, 2024 | `10.1111/scd.12832` | **VERIFIED.** (Vol. 44, No. 1, pp. 148–156). |
| **[7]** | Z. Xie et al., "Building risk prediction models for type 2 diabetes..." | *Prev. Chronic Dis.*, 2019 | `10.5888/pcd16.190109` | **VERIFIED.** (Vol. 16, Art. E130, CDC). |
| **[8]** | B. Wu et al., "Building and validating a machine learning model to predict CHD..." | *Comput. Methods Programs Biomed.*, 2026 | `10.1016/j.cmpb.2025.109186` | **VERIFIED.** (Vol. 275, Art. 109186; online late 2025). |
| **[9]** | M. Imani et al., "Predicting cardiovascular diseases using imbalanced data..." | *Am. Heart J. Plus*, 2026 | `10.1016/j.ahjo.2026.100719` | **VERIFIED.** (Vol. 63, Art. 100719; published Jan 2026). |
| **[10]** | M. H. N. Le et al., "Protective predictors of cardiovascular disease: An XAI approach..." | *Public Health*, 2026 | `10.1016/j.puhe.2025.106050` | **VERIFIED.** (Vol. 250, Art. 106050). |
| **[11]** | M. A. Muhammad et al., "Exploring explainable machine learning for diabetes..." | *J. Prim. Care Community Health*, 2025 | `10.1177/21501319251400546` | **VERIFIED WITH CORRECTION.** Review cited *JPCRR*; actual journal is *J. Prim. Care Community Health* (SAGE). |
| **[12]** | M. A. Muhammad et al., "Integrating epidemiologic modeling and XAI for depression..." | *Discov. Ment. Health*, 2026 | `10.1007/s44192-026-00506-4` | **VERIFIED.** (Vol. 6, Art. 151). |
| **[13]** | M. M. Islam et al., "Explainable machine learning model incorporating SDOH for CKD..." | *J. Diabetes Metab. Disord.*, 2025 | `10.1007/s40200-025-01621-9` | **VERIFIED.** (Vol. 24, No. 1, Art. 115). |
| **[14]** | J. Noh et al., "Occupational and socioeconomic predictors of MI and CHD..." | *BMC Public Health*, 2026 | `10.1186/s12889-026-27570-3` | **VERIFIED.** (Vol. 26, Art. 2015). |
| **[15]** | S. B. Akter et al., "Early detection of subjective cognitive decline..." | *J. Biomed. Inform.*, 2025 | `10.1016/j.jbi.2024.104770` | **VERIFIED.** (Vol. 162, Art. 104770). |
| **[16]** | M. A. Shams et al., "Explainable machine learning revealing the impact... on arthritis..." | *BMJ Health Care Inform.*, 2026 | `10.1136/bmjhci-2025-101955` | **VERIFIED.** (Vol. 33, No. 1, Art. e101955). |
| **[17]** | J. Islam et al., "Leakage-safe precision-aware dual-branch FT-Transformer..." | *Sensors*, 2026 | `10.3390/s26113417` | **VERIFIED.** (Vol. 26, No. 11, Art. 3417). |
| **[18]** | Q. T. Lam et al., "Interpretable machine learning for population-level tooth loss..." | *J. Dent. Res.*, 2026 | `10.1177/00220345261464887` | **VERIFIED.** (Online ahead of print, SAGE). |
| **[19]** | P. Tong et al., "Explainable logistic regression for heart disease risk prediction..." | *JMIR Cardio*, 2025 | `10.2196/82462` | **VERIFIED.** (JMIR Publications). |
| **[20]** | D. Majcherek et al., "AI-driven analysis of diabetes risk determinants in U.S. adults..." | *PLOS ONE*, 2025 | `10.1371/journal.pone.0328655` | **VERIFIED.** (Vol. 20, No. 9, Art. e0328655). |
| **[21]** | W. Li et al., "Diabetes prediction model based on GA-XGBoost and stacking..." | *PLOS ONE*, 2024 | `10.1371/journal.pone.0311222` | **VERIFIED.** (Vol. 19, No. 9, Art. e0311222). |
| **[22]** | D. Das et al., "Diabetes prediction using ensemble learning techniques..." | *Procedia Comput. Sci.*, 2025 | `10.1016/j.procs.2025.04.573` | **VERIFIED.** (Vol. 258, pp. 3155–3164). |
| **[23]** | T. Lin and M. Fan, "FMI-CAECD: Fusing multi-input convolutional features..." | *Sensors*, 2024 | `10.3390/s24227160` | **VERIFIED.** (Vol. 24, No. 22, Art. 7160). |
| **[24]** | K.-V. Tompra et al., "Strategic machine learning optimization for CVD prediction..." | *Algorithms*, 2024 | `10.3390/a17050178` | **VERIFIED.** (Vol. 17, No. 5, Art. 178). |
| **[25]** | H. Alkaabi and A. Abugabah, "Smart health care application for predicting complications..." | *Procedia Comput. Sci.*, 2026 | `10.1016/j.procs.2026.04.111` | **VERIFIED.** (Vol. 280, pp. 889–896). |
| **[26]** | A. Altamimi et al., "Explainable ML-based decision support for diabetes risk screening..." | *J. Intell. Decis. Mak. Inf. Sci.*, 2026 | `10.59543/jidmis.v3.1780` | **VERIFIED.** (Vol. 3, Art. 1780). |
| **[27]** | J. D. Morgenstern et al., "Predicting population health with machine learning: A scoping review..." | *BMJ Open*, 2020 | `10.1136/bmjopen-2020-037860` | **VERIFIED.** (Vol. 10, No. 10, Art. e037860). |
| **[28]** | S. Kino et al., "A scoping review on the use of machine learning in SDOH research..." | *SSM - Popul. Health*, 2021 | `10.1016/j.ssmph.2021.100836` | **VERIFIED.** (Vol. 15, Art. 100836). |
| **[29]** | S. Birdi et al., "Bias in machine learning applications to address NCDs at population-level..." | *BMC Public Health*, 2024 | `10.1186/s12889-024-21081-9` | **VERIFIED.** (Vol. 24, Art. 3599). |
| **[30]** | R. K. Mutharasan & J. Walradt, "Population health and artificial intelligence..." | *JACC Adv.*, 2024 | `10.1016/j.jacadv.2024.101092` | **VERIFIED.** (Vol. 3, No. 8, Art. 101092). |
| **[31]** | A. Caruana et al., "Machine learning for administrative health records: A systematic review..." | *Artif. Intell. Med.*, 2023 | `10.1016/j.artmed.2023.102642` | **VERIFIED.** (Vol. 144, Art. 102642). |
| **[32]** | H. Abdulazeem et al., "A systematic review of clinical health conditions predicted by ML..." | *PLOS ONE*, 2023 | `10.1371/journal.pone.0274276` | **VERIFIED.** (Vol. 18, No. 9, Art. e0274276). |
| **[33]** | L. Wang et al., "Human-centered design and evaluation of AI-CDSS: A systematic review..." | *Front. Comput. Sci.*, 2023 | `10.3389/fcomp.2023.1187299` | **VERIFIED.** (Vol. 5, Art. 1187299). |
| **[34]** | S. Ali et al., "The enlightening role of explainable AI in medical and healthcare domains..." | *Comput. Biol. Med.*, 2023 | `10.1016/j.compbiomed.2023.107555` | **VERIFIED.** (Vol. 166, Art. 107555). |
| **[35]** | J. Jung et al., "Essential properties and explanation effectiveness of XAI in healthcare..." | *Heliyon*, 2023 | `10.1016/j.heliyon.2023.e16110` | **VERIFIED.** (Vol. 9, No. 5, Art. e16110). |
| **[36]** | S. S. Band et al., "Application of explainable AI in medical health: A systematic review..." | *Inform. Med. Unlocked*, 2023 | `10.1016/j.imu.2023.101286` | **VERIFIED.** (Vol. 40, Art. 101286). |
| **[37]** | M. M. Alsaleh et al., "Prediction of disease comorbidity using explainable AI and ML..." | *Int. J. Med. Inform.*, 2023 | `10.1016/j.ijmedinf.2023.105088` | **VERIFIED.** (Vol. 175, Art. 105088). |
| **[38]** | J. Allgaier et al., "How does the model make predictions? A systematic literature review on XAI..." | *Artif. Intell. Med.*, 2023 | `10.1016/j.artmed.2023.102616` | **VERIFIED.** (Vol. 143, Art. 102616). |
| **[39]** | Z. Sadeghi et al., "A review of explainable artificial intelligence in healthcare..." | *Comput. Electr. Eng.*, 2024 | `10.1016/j.compeleceng.2024.109370` | **VERIFIED.** (Vol. 118, Part A, Art. 109370). |
| **[40]** | I. D. Mienye et al., "A survey of explainable artificial intelligence in healthcare: Concepts..." | *Inform. Med. Unlocked*, 2024 | `10.1016/j.imu.2024.101587` | **VERIFIED.** (Vol. 51, Art. 101587). |

---

## 14. Synthesis & Recommendations for Paper Writing (Stage 2)

1. **Adhere to IEEE 6-Page Limit:**
   - Section I: Introduction & Background (1.0 page) — Public health burden of CAD/MI, limitations of current BRFSS ML research.
   - Section II: Cohort Selection, Leakage-Safe Engineering & Preprocessing (1.0 page) — CDC BRFSS 2024 structure, exclusion of `CVDINFR4`/`CVDCRHD4`, 190 engineered features, stratified partitioning.
   - Section III: Epidemiological Survey Weighting & Statistical Analysis (0.75 pages) — Normalized GLM, odds ratios forest plot, non-causal cross-sectional framing.
   - Section IV: Baseline & Improved Model Development with Calibration-Preserving Threshold Optimization (1.25 pages) — 4 candidate architectures, PR-AUC criterion, Brier score comparison, validation Youden's J threshold optimization ($\tau = 0.080$).
   - Section V: Model Evaluation & Explainability Analysis (1.25 pages) — Holdout test confusion matrix, grouped permutation importance, partial dependence, and local Ridge surrogates.
   - Section VI: Academic Dashboard, Limitations & Conclusion (0.75 pages) — SPA deployment, surveillance guardrails, clinical utility disclaimer, future directions.
2. **Slide Presentation Mapping (12–15 Slides):**
   - Slide 1: Title & Authorship
   - Slide 2: Population Health Problem & Surveillance Context
   - Slide 3: Research Gaps & Analytical Framework
   - Slide 4: CDC BRFSS 2024 Cohort & Outcome Feasibility Audit
   - Slide 5: Leakage Prevention & Preprocessing Architecture
   - Slide 6: Candidate Models & Evaluation Strategy (PR-AUC vs. ROC-AUC)
   - Slide 7: Calibration Findings & Decision Threshold Optimization ($\tau = 0.080$)
   - Slide 8: Holdout Test Results & Confusion Matrix Analysis ($8\times$ Case Detection)
   - Slide 9: Global Explainability: Grouped Permutation Importance
   - Slide 10: Non-Linear Risk Profiles: Partial Dependence Profiles (PDP)
   - Slide 11: Local Explanations: 4-Quadrant Case Archetypes & Surrogate Fidelity
   - Slide 12: Interactive Research Dashboard Architecture
   - Slide 13: Epidemiological Guardrails & Clinical Utility Disclosures
   - Slide 14: Summary of Contributions & Future Directions
   - Slide 15: Q&A and Key References
