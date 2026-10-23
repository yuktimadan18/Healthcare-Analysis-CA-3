# Milestone 7 Audit Report: Global Explainable AI (XAI)

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Target Variable:** `_MICHD` (Binary classification of reported history of myocardial infarction or coronary heart disease)
**Execution Date:** October 20, 2026
**Analysis Partition:** Validation Partition Strictly ($N = 67,870$, $6,351$ positive cases, prevalence: $9.358\%$)
**Model Under Audit:** `HistGradientBoostingClassifier` (Unweighted GBDT Champion Model, frozen from Milestone 6)
**Artifacts Frozen:** `models/improved_hist_gradient_boosting.joblib`, `models/preprocessor.joblib`, `models/model_selection_summary.json`

---

## 1. Executive Summary

Milestone 7, Stage 2 delivers a methodologically rigorous global explainability analysis for the frozen `HistGradientBoostingClassifier` champion model. All global importance evaluations and partial dependence profiles were conducted strictly on the **validation partition** ($N = 67,870$). The holdout test partition remained completely untouched during feature importance estimation, preserving its status as an unbiased holdout dataset.

### Core Methodological Safeguards Enforced:
1. **Grouped Permutation Integrity:** For each of the 38 original survey predictors, all one-hot encoded dummy columns belonging to that predictor were permuted simultaneously across rows using a shared permutation index (`perm_idx`). Individual dummy indicators within a multi-category variable were never permuted independently during grouped analysis, preserving the structural relationship across levels for each permuted respondent while breaking the association with the target and remaining features.
2. **Imbalance-Aware Primary Metric (PR-AUC):** Because reported CAD/MI history is imbalanced ($9.358\%$ validation prevalence), **PR-AUC (Average Precision) decrease** served as the primary importance metric. ROC-AUC decreases are reported in parallel. Empirical small negative importance drops are preserved without truncation, reflecting standard empirical noise on non-informative features.
3. **Dual-Level Feature Resolution:**
   - **Grouped Variable Level (38 original predictors):** Captures the net predictive contribution of the parent survey construct, eliminating the artificial importance dilution caused by one-hot encoding.
   - **Indicator Feature Level (190 transformed features):** Dissects individual dummy categories to identify specific high-risk contrasts (e.g., Male vs. Female, Poor vs. Excellent health).
4. **Reliably Unscaled Partial Dependence:** Partial dependence profiles were computed for the three key continuous numeric predictors (`_AGE80`, `PHYSHLTH_DAYS`, and `BMI_NUMERIC`). Standardized values from the transformed feature space were mapped back to their natural epidemiological units using the frozen `StandardScaler` mean and scale parameters, and log-odds decision function outputs were transformed into marginal predicted probabilities via the logistic sigmoid function.
5. **Strict Non-Causal & Non-Clinical Framing:** All feature importances and partial dependence profiles describe associative statistical response patterns of the machine learning model within cross-sectional surveillance data; they do not represent causal intervention levers, etiological mechanisms, or clinical diagnostic validity.

---

## 2. Validation Baseline Performance

Before any column permutation, the frozen champion model was evaluated on the unpermuted validation partition ($N = 67,870$, 190 transformed features):

- **Validation Baseline PR-AUC:** **$0.370088$**
- **Validation Baseline ROC-AUC:** **$0.844895$**
- **Prevalence Baseline Reference:** $0.093576$ ($9.36\%$)

---

## 3. Grouped Permutation Importance (38 Original Predictors)

Grouped permutation importance was computed across 10 reproducible repeats (random seed $42$, repeat seeds $42 \dots 51$). Importance is measured as the degradation in predictive skill:

$$\Delta \text{Metric} = \text{Metric}_{\text{baseline}} - \text{Metric}_{\text{permuted}}$$

### Table 1: Top 20 Original Survey Predictors Ranked by Mean PR-AUC Decrease
*Source:* [`results/tables/xai_global_variable_importance.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv)

| Rank | Source Variable | Feature Type | Transformed Columns | Mean PR-AUC Drop | Std PR-AUC Drop | Mean ROC-AUC Drop | Std ROC-AUC Drop | Variable Construct Description |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`_AGE80`** | Numeric | 1 | **0.077214** | 0.003596 | **0.093609** | 0.003326 | Continuous age in years (imputed, top-coded at 80) |
| **2** | **`GENHLTH`** | Categorical | 8 | **0.043380** | 0.002198 | **0.020937** | 0.000788 | Self-rated general health (5 ordinal levels + DK/Ref/Missing) |
| **3** | **`_SEX`** | Categorical | 2 | **0.038017** | 0.002061 | **0.020163** | 0.000929 | Biological sex (Male vs. Female) |
| **4** | **`CVDSTRK3`** | Categorical | 5 | **0.030589** | 0.000706 | **0.012800** | 0.000366 | Diagnosed stroke history (cerebrovascular comorbidity) |
| **5** | **`PERSDOC3`** | Categorical | 6 | **0.012699** | 0.000931 | **0.007135** | 0.000232 | Personal healthcare provider continuity |
| **6** | **`DIABETE4`** | Categorical | 7 | **0.009585** | 0.000902 | **0.003729** | 0.000244 | Diagnosed diabetes comorbidity |
| **7** | **`CHCKDNY2`** | Categorical | 5 | **0.008759** | 0.000546 | **0.002115** | 0.000227 | Diagnosed chronic kidney disease comorbidity |
| **8** | **`CHCCOPD3`** | Categorical | 5 | **0.007772** | 0.000814 | **0.002993** | 0.000212 | Diagnosed COPD / emphysema comorbidity |
| **9** | **`_RACE`** | Categorical | 9 | **0.004422** | 0.000674 | **0.001554** | 0.000201 | Race / ethnicity demographic categories |
| **10** | **`_SMOKER3`** | Categorical | 5 | **0.004382** | 0.000688 | **0.002302** | 0.000232 | Smoking status (Daily, Some days, Former, Never) |
| **11** | `HAVARTH4` | Categorical | 5 | 0.002462 | 0.000980 | 0.001451 | 0.000322 | Diagnosed arthritis comorbidity |
| **12** | `EMPLOY1` | Categorical | 10 | 0.002013 | 0.000436 | 0.000637 | 0.000112 | Employment / disability status |
| **13** | `CHECKUP1` | Categorical | 7 | 0.001655 | 0.000363 | 0.001181 | 0.000259 | Time since last routine clinical checkup |
| **14** | `DIFFWALK` | Categorical | 5 | 0.001504 | 0.000408 | 0.000890 | 0.000058 | Serious mobility difficulty |
| **15** | `MARITAL` | Categorical | 8 | 0.001475 | 0.000708 | 0.000644 | 0.000179 | Marital status |
| **16** | `MEDCOST1` | Categorical | 5 | 0.001240 | 0.000576 | 0.000599 | 0.000155 | Healthcare cost barrier in past 12 months |
| **17** | `PHYSHLTH_DAYS` | Numeric | 1 | 0.001160 | 0.000490 | 0.000703 | 0.000170 | Physically unhealthy days in past 30 days (0-30) |
| **18** | `_AGEG5YR` | Categorical | 14 | 0.001123 | 0.000285 | 0.000323 | 0.000072 | 14-level 5-year age categories |
| **19** | `DIFFDRES` | Categorical | 5 | 0.001117 | 0.000252 | 0.000148 | 0.000047 | Difficulty dressing or bathing |
| **20** | `BMI_NUMERIC` | Numeric | 1 | 0.001073 | 0.000493 | 0.000242 | 0.000098 | Body Mass Index (continuous kg/m²) |

### Visual Artifact: Figure 16
[`results/figures/fig16_global_variable_importance.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig16_global_variable_importance.png) visualizes the top 20 predictors with uncertainty bars ($\pm 1\text{ SD}$) across the 10 permutation repeats.

### Scientific Discussion of Grouped Findings:
1. **Dominance of Age, Self-Rated Health, and Biological Sex:**
   - `_AGE80` is by far the single most influential predictor, causing a substantial mean PR-AUC drop of **$0.0772$** ($20.9\%$ of total model PR-AUC) and an ROC-AUC drop of **$0.0936$**.
   - `GENHLTH` (General health rating) ranks second (PR-AUC drop: **$0.0434$**), reflecting the strong epidemiological reality that poor self-rated health aggregates subclinical and chronic systemic illness.
   - `_SEX` ranks third (PR-AUC drop: **$0.0380$**), consistent with well-established sex disparities in reported CAD/MI prevalence in older cohorts.
2. **Cardiovascular & Chronic Comorbidity Cluster:**
   - Stroke history (`CVDSTRK3`, PR drop: **$0.0306$**) represents shared vascular pathology with coronary disease.
   - Cardiometabolic and chronic systemic comorbidities (`DIABETE4` [0.0096], `CHCKDNY2` [0.0088], and `CHCCOPD3` [0.0078]) form a consistent secondary cluster.
3. **Collinearity Sharing & Masking:**
   - Notice that categorical 5-year age groups (`_AGEG5YR`) exhibits a smaller incremental importance (PR drop: $0.0011$) when `_AGE80` is present in the model. This occurs because the continuous variable `_AGE80` captures the primary monotonic age gradient, partially masking the marginal impact of the categorical age bands. This illustrates the critical principle that **permutation importance evaluates marginal loss upon feature removal, and highly correlated features share or mask each other's importance.**

---

## 4. Indicator-Level Permutation Importance (190 Transformed Features)

To identify specific category-level contrast drivers, all 190 transformed features were permuted individually across 10 repeats.

### Table 2: Top 15 Individual Transformed Indicators
*Source:* [`results/tables/xai_transformed_feature_importance.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_transformed_feature_importance.csv)

| Rank | Transformed Feature Name | Parent Variable | Category Level | Mean PR-AUC Drop | Std PR-AUC Drop | Mean ROC-AUC Drop | Indicator Interpretation |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `num___AGE80` | `_AGE80` | Numeric (Z-score) | **0.077214** | 0.003596 | **0.093609** | Continuous Age in years |
| **2** | `cat___SEX_1.0` | `_SEX` | 1.0 (Male) | **0.029848** | 0.001764 | **0.015999** | Male respondent contrast indicator |
| **3** | `cat__CVDSTRK3_2.0` | `CVDSTRK3` | 2.0 (No Stroke) | **0.021378** | 0.000760 | **0.007892** | Absence of stroke history contrast |
| **4** | `cat__GENHLTH_5.0` | `GENHLTH` | 5.0 (Poor Health) | **0.014871** | 0.000909 | **0.002811** | Self-rated "Poor" health indicator |
| **5** | `cat__PERSDOC3_2.0` | `PERSDOC3` | 2.0 (More than one doctor) | **0.011223** | 0.000816 | **0.006357** | Having multiple personal healthcare providers |
| **6** | `cat__GENHLTH_1.0` | `GENHLTH` | 1.0 (Excellent) | **0.006277** | 0.001030 | **0.005279** | Self-rated "Excellent" health indicator (negative risk) |
| **7** | `cat__GENHLTH_4.0` | `GENHLTH` | 4.0 (Fair Health) | **0.006246** | 0.000565 | **0.002924** | Self-rated "Fair" health indicator |
| **8** | `cat__GENHLTH_2.0` | `GENHLTH` | 2.0 (Very Good) | **0.005395** | 0.000601 | **0.003504** | Self-rated "Very Good" health indicator |
| **9** | `cat__CHCCOPD3_2.0` | `CHCCOPD3` | 2.0 (No COPD) | **0.004692** | 0.000653 | **0.002097** | Absence of COPD comorbidity contrast |
| **10** | `cat__CHCKDNY2_2.0` | `CHCKDNY2` | 2.0 (No Kidney Disease) | **0.004553** | 0.000437 | **0.000858** | Absence of kidney disease contrast |
| **11** | `cat__DIABETE4_3.0` | `DIABETE4` | 3.0 (No Diabetes) | 0.004245 | 0.000473 | 0.000772 | Absence of diabetes contrast |
| **12** | `cat__DIABETE4_1.0` | `DIABETE4` | 1.0 (Diagnosed Diabetes) | 0.004092 | 0.000378 | 0.002235 | Confirmed diabetes diagnosis indicator |
| **13** | `cat___SEX_2.0` | `_SEX` | 2.0 (Female) | 0.003666 | 0.000468 | 0.001859 | Female respondent contrast indicator |
| **14** | `cat__CHCKDNY2_1.0` | `CHCKDNY2` | 1.0 (Yes Kidney Disease) | 0.003501 | 0.000372 | 0.001096 | Confirmed kidney disease indicator |
| **15** | `cat__CHCCOPD3_1.0` | `CHCCOPD3` | 1.0 (Yes COPD) | 0.003189 | 0.000402 | 0.001153 | Confirmed COPD indicator |

### Visual Artifact: Figure 17
[`results/figures/fig17_indicator_feature_importance.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig17_indicator_feature_importance.png) illustrates the top 25 individual one-hot and numeric features, distinguishing continuous numeric features from categorical dummy levels.

### Comparison Between Grouped vs. Indicator Importance:
This comparison empirically demonstrates the necessity of grouped permutation:
- `GENHLTH` ranks as the **#2 most important variable globally** (PR-AUC drop: **$0.0434$**).
- However, when permuted as isolated columns, its individual levels exhibit smaller drops: `cat__GENHLTH_5.0` (Poor) drop is $0.0149$, `cat__GENHLTH_1.0` (Excellent) drop is $0.0063$, `cat__GENHLTH_4.0` (Fair) is $0.0062$, and `cat__GENHLTH_2.0` is $0.0054$.
- Relying exclusively on indicator-level importance would severely understate the true aggregate contribution of self-rated health. Grouped permutation correctly restores its prominent standing.

---

## 5. Partial Dependence Profiles (Continuous Numeric Predictors)

Partial dependence was evaluated for the three continuous predictors across the validation cohort. Grid values were reliably connected back to their original survey scales using the fitted transformer parameters:

- `_AGE80`: Scaler mean $= 55.024\text{ yrs}$, scale $= 18.128\text{ yrs}$. Unscaled grid: $25.4\text{ to }78.6\text{ yrs}$.
- `PHYSHLTH_DAYS`: Scaler mean $= 4.459\text{ days}$, scale $= 8.812\text{ days}$. Unscaled grid: $0.0\text{ to }30.0\text{ days}$.
- `BMI_NUMERIC`: Scaler mean $= 28.455\text{ kg/m}^2$, scale $= 6.276\text{ kg/m}^2$. Unscaled grid: $19.6\text{ to }40.5\text{ kg/m}^2$.

### Visual Artifact: Figure 18
[`results/figures/fig18_partial_dependence_profiles.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png) contains three side-by-side panels:
- **Panel A (`_AGE80`):** Demonstrates a sharp non-linear risk gradient. Marginal predicted probability remains below the cohort prevalence ($9.36\%$) through age 55, then surges steeply, crossing the operational screening threshold ($0.080$) at approximately age 54 and exceeding $20\%$ past age 75.
- **Panel B (`PHYSHLTH_DAYS`):** Shows a progressive positive slope between 0 and 15 physically unhealthy days, followed by a plateau between 15 and 30 days.
- **Panel C (`BMI_NUMERIC`):** Demonstrates an asymmetric J-shaped curve, with lowest predicted probabilities in the healthy BMI range ($22 - 25\text{ kg/m}^2$) and gradual elevation as BMI increases into class I/II obesity.

### Methodological & Scientific Disclaimer:
Partial dependence describes statistical model predictions averaged over empirical feature distributions; **it does not reflect prospective intervention effects, causal treatment response, or clinical prognosis.** In the presence of correlated predictors, partial dependence averages over potentially unlikely feature combinations.

---

## 6. Automated Verification Checks & Test Suite Results

A dedicated automated test suite was constructed at [`tests/test_xai_global_importance.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_xai_global_importance.py) and executed via `unittest`:

```text
Ran 3 tests in 0.011s — OK
Ran 11 total tests across discovery suite in 1.265s — OK
```

### Verification Checks Passed:
1. **Grouped Partitioning Completeness & Disjointness [PASSED]:** Verified that grouping columns by `source_variable` covers all 190 transformed features with zero overlap and union equal to `set(range(190))`.
2. **Output Table Dimensions & Finiteness [PASSED]:**
   - `xai_global_variable_importance.csv`: Exactly 38 rows (all 38 original predictors present, 0 missing, 0 duplicates, all values finite, ranks 1 to 38 strictly sorted).
   - `xai_transformed_feature_importance.csv`: Exactly 190 rows (all 190 features present, all values finite, ranks 1 to 190 strictly sorted).
3. **Artifact Integrity [PASSED]:**
   - `fig16_global_variable_importance.png`: $333,520\text{ bytes}$ (> 10 KB).
   - `fig17_indicator_feature_importance.png`: $309,820\text{ bytes}$ (> 10 KB).
   - `fig18_partial_dependence_profiles.png`: $367,790\text{ bytes}$ (> 10 KB).
4. **Frozen State Preservation [PASSED]:** Timestamps and contents of `models/`, `data/processed/`, `README.md`, and `docs/PROJECT_DECISIONS.md` remain completely unmodified.

---

## 7. Artifact Registry for Milestone 7, Stage 2

| File Path | Category | Purpose | Status |
| :--- | :--- | :--- | :---: |
| [`src/xai/global_importance.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py) | Source Code | Engine for grouped and indicator permutation importance & PDP | Created |
| [`tests/test_xai_global_importance.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_xai_global_importance.py) | Test Suite | Unit tests for grouped column partitioning, table integrity, and figures | Created |
| [`results/tables/xai_global_variable_importance.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv) | Table | 38 original predictors ranked by validation PR-AUC drop (10 repeats) | Generated |
| [`results/tables/xai_transformed_feature_importance.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_transformed_feature_importance.csv) | Table | 190 transformed indicators ranked by validation PR-AUC drop (10 repeats) | Generated |
| [`results/figures/fig16_global_variable_importance.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig16_global_variable_importance.png) | Figure | Top 20 original predictors bar plot with error bars | Generated |
| [`results/figures/fig17_indicator_feature_importance.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig17_indicator_feature_importance.png) | Figure | Top 25 transformed indicators bar plot with error bars | Generated |
| [`results/figures/fig18_partial_dependence_profiles.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png) | Figure | 3-panel partial dependence profiles unscaled to survey units | Generated |
| [`reports/milestone7_global_importance_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md) | Audit Report | Authoritative audit report for Milestone 7 Stage 2 | Created |
