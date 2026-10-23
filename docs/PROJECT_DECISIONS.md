# Project Decisions & Audit Log

## Milestone 1: Environment, Archive Verification, and Repository Setup

### 1. Active Python Environment
* **Decision:** Retain and utilize the existing virtual environment without re-creation or modification.
* **Interpreter Path:** `.venv\Scripts\python.exe` (`C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project\.venv\Scripts\python.exe`)
* **Python Version:** `3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]`
* **Verified Dependencies:**
  * Core processing: `numpy` (2.5.4), `pandas` (3.0.6), `scipy` (1.18.1)
  * SAS XPT ingestion: `pyreadstat` (1.3.6)
  * Statistics & ML: `statsmodels` (0.15.0), `scikit-learn` (1.9.1), `joblib` (1.6.0)
  * Visualizations: `matplotlib` (3.11.2), `seaborn` (0.13.2), `plotly` (7.1.0)
  * Notebooks: `ipykernel` (7.4.0), `jupyterlab` (4.6.4), `notebook` (7.6.3)
* **Configuration:** Pinned verified core dependency versions into `requirements.txt`. Additional packages (e.g., gradient boosters, SHAP, dashboard frameworks) will only be introduced when justified in later milestones.

### 2. Dataset Archive Audit & Integrity
* **Archive Path:** `data/raw/LLCP2024XPT.zip`
* **Archive Size:** 83,007,956 bytes (~79.16 MB)
* **Cryptographic Hash (SHA-256):** `d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4`
* **Integrity Status (`zipfile.testzip()`):** **PASSED** (0 CRC mismatch errors, no corrupt records detected).
* **Archive Contents:**
  * File entry: `'LLCP2024.XPT '` (contains trailing space in entry header as released by CDC).
  * Uncompressed size: 1,093,874,240 bytes (~1.02 GB / 1,043.20 MB).
  * Archive timestamp: `2025-08-21 09:06:24`.
* **Storage Decision:** The ZIP archive in `data/raw/` remains untouched and unextracted in place per Rule 7 and Rule 13.

### 3. Deferral of Predictive Target Selection
* **Decision:** Target selection is explicitly deferred until Milestone 5, following codebook review and exploratory data analysis.
* **Rationale:**
  * Per **Rule 6** of `PROJECT_RULES.md`: *"Do not select the predictive target until the actual dataset and official codebook have been inspected."*
  * The CDC BRFSS questionnaire consists of Core Sections (nationwide) and Optional Modules (state-specific). Arbitrarily choosing a target variable without checking variable completeness across states, survey skip patterns, and sampling weights (`_LLCPWT`) risks severe selection bias and clinical/methodological invalidity.
  * Target selection requires an empirical evaluation of outcome prevalence, missingness mechanisms, and public health relevance.

### 4. Version Control and Data Protection
* **Decision:** Configured `.gitignore` to prevent any raw, interim, or processed datasets, binary models, notebook checkpoints, and environment secrets from being committed to Git.
* **Exclusion Scope:** Excludes `data/raw/*`, `data/interim/*`, `data/processed/*` (with exception for `.gitkeep`), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, and environment secrets.

---

## Milestone 2: Schema Inspection, Documentation & Provenance Audit

### 1. Ingestion Feasibility via In-Memory Streaming
* **Finding:** Calling `pyreadstat.read_xport()` on an in-memory stream from `zipfile.ZipFile.open()` with default arguments raises a `UnicodeDecodeError` on the SAS XPT header.
* **Resolution:** Passing `encoding='latin1'` resolves the issue entirely. `pyreadstat` reads all 301 variable names, labels, and formats directly from the compressed archive stream without extracting the 1.02 GB uncompressed `.XPT` file to disk.
* **Limitation:** In the SAS XPT specification, row counts are not stored in header metadata (`meta.number_rows` returns `None`). CDC documentation verifies 457,670 total survey records in the 2024 public file.

### 2. Discrepancy Investigation (Archive Size & Variable Count)
* **Observed Discrepancy:**
  * Local Audit: 83,007,956 bytes (~79.16 MB), 301 variables.
  * CDC Landing Page HTML Text (`annual_2024.html`): Claims `[ZIP – 64.3 MB]` and states *"This file contains 345 variables."*
* **Forensic Provenance Verification:**
  * Windows NTFS Alternate Data Stream (`data\raw\LLCP2024XPT.zip:Zone.Identifier`) confirmed the local file was directly downloaded from `https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip` (Referrer: `https://www.cdc.gov/brfss/annual_data/annual_2024.html`).
* **Official Server Verification:**
  * Direct HTTP stream check of the live CDC endpoint (`https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip`):
    * `Content-Length`: 83,007,956 bytes.
    * `Last-Modified`: `Thu, 21 Aug 2025 13:07:35 GMT`.
    * `ETag`: `"29f038909c12dc1:0"`.
    * SHA-256 of live stream: `d33de91a584e866f657d29b1d5b51b5672467286ed22d021e36ee3c7b4d712a4`.
    * **Conclusion:** The local archive is **100% bit-for-bit identical** to the official file currently distributed by the CDC.
* **Variable Count Resolution:**
  * The official CDC 2024 Variable Layout table (`https://www.cdc.gov/brfss/annual_data/2024/llcp_varlayout_24_onecolumn.html`) officially defines **301 variables** (`_STATE` to `_AIDTST4`).
  * The official CDC 2024 Codebook (`codebook24_llcp-v2-508.zip`, `USCODE24_LLCP_082125.HTML`) confirms the 2024 variable set.
  * **Root Cause:** The text *"64.3 MB"* and *"345 variables"* on `annual_2024.html` is an unupdated editorial misprint on the CDC web page (likely carried over from draft templates or prior-year metrics). The actual CDC data release file and CDC layout table contain exactly 301 variables and 83,007,956 bytes.

### 3. Local Reference Documentation Acquired
The following official CDC documents were downloaded into `docs/reference/` on October 16, 2026:
1. `docs/reference/codebook24_llcp-v2-508.zip` (71,439 bytes) from `https://www.cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip`
2. `docs/reference/2024-calculated-variables-version4-508.pdf` (1,690,211 bytes) from `https://www.cdc.gov/brfss/annual_data/2024/pdf/2024-calculated-variables-version4-508.pdf`
3. `docs/reference/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf` (440,842 bytes) from `https://www.cdc.gov/brfss/annual_data/2024/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf`
4. `docs/reference/Overview_2024-508.pdf` (267,231 bytes) from `https://www.cdc.gov/brfss/annual_data/2024/pdf/Overview_2024-508.pdf`
5. `docs/reference/2024-BRFSS-Questionnaire-508.pdf` (967,272 bytes) from `https://www.cdc.gov/brfss/questionnaires/pdf-ques/2024-BRFSS-Questionnaire-508.pdf`

---

## Milestone 3: Data Quality Assessment & Exploratory Data Analysis (EDA)

### 1. Intermediate Serialization Format Decision
* **Pre-implementation Finding:** Neither `pyarrow` nor `fastparquet` is installed in the virtual environment. Parquet export was blocked.
* **Decision:** Per user direction, intermediate data is serialized as a gzip-compressed CSV (`data/interim/brfss_2024_selected.csv.gz`) without installing external packages. Compression yielded a compact **15.63 MB** file on disk, preserving memory efficiency and reproducible execution.

### 2. Candidate Feature Selection & Streaming Ingestion
* **Implementation:** Created `src/data/make_dataset.py` to stream selected columns directly from `data/raw/LLCP2024XPT.zip` member `'LLCP2024.XPT '` with `encoding='latin1'`.
* **Cohort Dimensions:** Exactly **457,670 rows** and **61 columns** extracted in 19.02 seconds; in-memory footprint: **213.00 MB**.
* **Variable Scope:** Selected 61 candidate variables across 7 clinical/methodological categories:
  1. Survey design & weights (`_LLCPWT`, `_STSTR`, `_PSU`, `_STATE`)
  2. Demographics (`_SEX`, `_AGEG5YR`, `_AGE65YR`, `_AGE80`, `_RACE`, `_EDUCAG`, `_INCOMG1`, `MARITAL`, `EMPLOY1`, `_METSTAT`, `_URBSTAT`)
  3. General health & quality of life (`GENHLTH`, `_RFHLTH`, `PHYSHLTH`, `_PHYS14D`, `MENTHLTH`, `_MENT14D`, `POORHLTH`)
  4. Healthcare access (`PRIMINS2`, `_HLTHPL2`, `PERSDOC3`, `MEDCOST1`, `CHECKUP1`)
  5. Chronic condition indicators (`_MICHD`, `CVDINFR4`, `CVDCRHD4`, `CVDSTRK3`, `DIABETE4`, `PREDIAB2`, `ASTHMA3`, `_ASTHMS1`, `CHCCOPD3`, `ADDEPEV3`, `CHCKDNY2`, `HAVARTH4`, `CHCSCNC1`, `CHCOCNC1`)
  6. Behavioral risk factors (`_BMI5`, `_BMI5CAT`, `_RFBMI5`, `SMOKE100`, `_SMOKER3`, `_RFSMOK3`, `ECIGNOW3`, `_CURECI3`, `EXERANY2`, `_TOTINDA`, `DRNKANY6`, `_RFBING6`, `_DRNKWK3`, `_RFDRHV9`)
  7. Functional disabilities (`DEAF`, `BLIND`, `DECIDE`, `DIFFWALK`, `DIFFDRES`, `DIFFALON`)
* **Metadata Export:** Generated `data/interim/brfss_2024_dictionary.json` defining every variable's category, question role, and inferred type.

### 3. Empirical Data Quality & Missingness Findings
* **No global code replacement:** Refusal (`9`/`99`) and don't know (`7`/`77`) codes were preserved as original numeric values in interim storage to prevent conflation with true unasked/missing values (`NaN`).
* **Structural Skip Patterns:**
  * `PREDIAB2` exhibits 65.2% missingness because it is routed only to non-diabetics or asked in optional modules.
  * `POORHLTH` exhibits 41.4% missingness because it is skipped if both `PHYSHLTH` and `MENTHLTH` are 0 days (`88`).
  * Body Mass Index (`_BMI5`, `_BMI5CAT`) has 9.4% missingness due to self-reported height/weight omissions.
* **Core Completeness:** All core chronic condition questions (`_MICHD`, `CHCCOPD3`, `CHCKDNY2`, `HAVARTH4`) exhibit < 1.2% missingness nationwide.

### 4. Candidate Outcome Prevalence Analysis (Rule 12 Adherence)
Evaluation of candidate chronic disease targets revealed notable divergence between raw sample proportions and CDC survey-weighted (`_LLCPWT`) population estimates:

| Variable | Clinical Condition | Positive Cases | Negative Cases | Missing/Excluded | Unweighted Prev (%) | Weighted Prev (%) | Imbalance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `_MICHD` | Myocardial Infarction / CHD | 42,338 | 410,126 | 5,206 | 9.36% | 6.80% | 1 : 9.7 |
| `CVDINFR4` | Heart Attack (History) | 26,672 | 427,883 | 3,115 | 5.87% | 4.38% | 1 : 16.0 |
| `CVDCRHD4` | Angina / Coronary Heart Disease | 28,511 | 424,581 | 4,578 | 6.29% | 4.31% | 1 : 14.9 |
| `CVDSTRK3` | Stroke (History) | 20,661 | 435,557 | 1,452 | 4.53% | 3.53% | 1 : 21.1 |
| `DIABETE4` | Diabetes (Ever Diagnosed) | 65,809 | 387,432 | 4,429 | 14.52% | 12.63% | 1 : 5.9 |
| `ASTHMA3` | Asthma (Ever Diagnosed) | 72,037 | 383,772 | 1,861 | 15.80% | 15.73% | 1 : 5.3 |
| `CHCCOPD3` | COPD / Emphysema | 36,921 | 418,575 | 2,174 | 8.11% | 6.30% | 1 : 11.3 |
| `ADDEPEV3` | Depressive Disorder | 96,125 | 358,881 | 2,664 | 21.13% | 21.04% | 1 : 3.7 |
| `CHCKDNY2` | Kidney Disease | 23,752 | 431,939 | 1,979 | 5.21% | 4.13% | 1 : 18.2 |
| `HAVARTH4` | Arthritis | 158,319 | 296,778 | 2,573 | 34.79% | 26.55% | 1 : 1.9 |

### 5. Artifacts Created & Validated
* **Notebook:** [`notebooks/01_data_quality_and_eda.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/01_data_quality_and_eda.ipynb) (21 cells executed in full via `nbconvert` with complete figures and outputs embedded).
* **Figures Generated:**
  1. `results/figures/fig1_missingness_top_variables.png`
  2. `results/figures/fig2_risk_factor_distributions.png`
  3. `results/figures/fig3_demographic_profiles.png`
  4. `results/figures/fig4_candidate_outcomes_prevalence.png`
  5. `results/figures/fig5_chronic_comorbidity_correlation.png`
* **Tables Generated:**
  1. `results/tables/data_quality_missingness.csv`
  2. `results/tables/candidate_outcomes_prevalence.csv`
  3. `results/tables/continuous_risk_factors_stats.csv`
  4. `results/tables/variable_dictionary_summary.csv`

---

## Milestone 4: Statistical Analysis & Knowledge Discovery

### 1. Survey Weighting & Statistical Methodology
* **Software Capability Assessment:** Evaluated `statsmodels` (0.15.0) within the active Python environment.
  * **Weighted Descriptives:** Utilized `statsmodels.stats.weightstats.DescrStatsW` with `weights=df['_LLCPWT']` to compute design-weighted prevalences, proportions, weighted means, standard errors, and 95% confidence intervals.
  * **Weight Normalization Correction (GLM):** Passing raw expansion weights (_LLCPWT) directly into `statsmodels.api.GLM` with `freq_weights` treats the population expansion sum (~217.6M) as integer cell replication counts, artificially shrinking standard errors by ~24x and yielding spuriously narrow confidence intervals (e.g., [4.56, 4.58]). To correct this, weights were normalized to sample size ($w_i^* = w_i \times N / \sum w_i$ so that $\sum w_i^* = N$). Point estimates ($\hat{\beta}$ and aOR) are mathematically invariant to scaling, while standard errors and confidence intervals are restored to the appropriate sample-size variance scale ($N = 385,120$).
  * **Cluster-Robust Sensitivity Analysis:** Implemented a secondary cluster-robust covariance model clustering on primary sampling units (`_PSU`) using `cov_type='cluster'` to evaluate intra-cluster standard error inflation.
  * **Design Limitation & Honest Disclosure:** Standard `statsmodels` 0.15.0 does not natively support simultaneous multi-stage Taylor series linearization combining stratum (`_STSTR`) and primary sampling unit (`_PSU`) clustering. In accordance with user guidance, no external packages (e.g., `samplics`) were installed. Standard errors and p-values are documented as survey-weighted estimates with cluster-robust sensitivity intervals, rather than full complex-design cluster-linearized standard errors.
  * **Multiple Testing Correction:** Applied Benjamini-Hochberg False Discovery Rate (FDR) control across all bivariate hypothesis tests; all primary associations remained statistically significant at $q < 0.001$.

### 2. Candidate Outcome Feasibility Audit
Evaluated 10 chronic condition indicators according to exact CDC 2024 codebook definitions, verifying that all rows strictly account for all 457,670 survey respondents:
* `_MICHD` (Cardiovascular - MI/CHD): 42,338 positive, 410,126 negative, 5,206 missing (checksum 457,670); 98.86% complete; 9.36% unweighted, **6.80% weighted**; 1 : 9.7 imbalance.
* `CVDINFR4` (Cardiovascular - Heart Attack): 26,672 positive, 427,883 negative, 2,857 DK, 256 Ref, 2 missing (checksum 457,670); 5.87% unweighted, 4.38% weighted; 1 : 16.0 imbalance (constituent of `_MICHD`).
* `CVDCRHD4` (Cardiovascular - Angina): 28,511 positive, 424,581 negative, 4,297 DK, 278 Ref, 3 missing (checksum 457,670); 6.29% unweighted, 4.31% weighted; 1 : 14.9 imbalance (constituent of `_MICHD`).
* `CVDSTRK3` (Cerebrovascular - Stroke): 20,661 positive, 435,557 negative, 1,244 DK, 205 Ref, 3 missing (checksum 457,670); 4.53% unweighted, 3.53% weighted; 1 : 21.1 imbalance (severe class imbalance).
* `DIABETE4` (Metabolic - Diabetes): 65,809 positive, 387,432 negative, 3,395 gestational diabetes (special code 2), 798 DK, 232 Ref, 4 missing (checksum 457,670); 99.03% complete; 14.52% unweighted, **12.63% weighted**; 1 : 5.9 imbalance.
* `ASTHMA3` (Respiratory - Asthma): 72,037 positive, 383,772 negative, 1,650 DK, 208 Ref, 3 missing (checksum 457,670); 15.80% unweighted, 15.73% weighted; 1 : 5.3 imbalance (high atopic/early childhood etiology).
* `CHCCOPD3` (Respiratory - COPD): 36,921 positive, 418,575 negative, 1,921 DK, 246 Ref, 7 missing (checksum 457,670); 8.11% unweighted, 6.30% weighted; 1 : 11.3 imbalance (dominated predominantly by smoking).
* `ADDEPEV3` (Mental Health - Depression): 96,125 positive, 358,881 negative, 2,066 DK, 593 Ref, 5 missing (checksum 457,670); 21.13% unweighted, 21.04% weighted; 1 : 3.7 imbalance (subjective diagnostic criteria).
* `CHCKDNY2` (Renal - Kidney Disease): 23,752 positive, 431,939 negative, 1,715 DK, 258 Ref, 6 missing (checksum 457,670); 5.21% unweighted, 4.13% weighted; 1 : 18.2 imbalance (severe imbalance, asymptomatic early stage).
* `HAVARTH4` (Musculoskeletal - Arthritis): 158,319 positive, 296,778 negative, 2,273 DK, 295 Ref, 5 missing (checksum 457,670); 34.79% unweighted, 26.55% weighted; 1 : 1.9 imbalance (broad multi-etiology symptom umbrella).

### 3. Key Statistical Findings & Multivariable Associations (`_MICHD`)
Multivariable survey-weighted logistic regression on reported history of coronary heart disease / myocardial infarction ($N = 385,120$ complete cases across predictors) yielded the following adjusted Odds Ratios (aOR) with 95% Wald Confidence Intervals (both normalized-weights CI and cluster-robust on `_PSU` CI):
* **Age Disparity:** Older adults aged 65+ had dramatically higher odds of CHD/MI compared to adults under 65: **aOR = 4.57** (95% CI normalized: 4.45 – 4.70; cluster-robust: 4.32 – 4.84).
* **Sex Disparity:** Males exhibited substantially higher odds than females: **aOR = 1.73** (95% CI normalized: 1.68 – 1.78; cluster-robust: 1.64 – 1.83).
* **Behavioral Risk Factors:**
  * Current cigarette smoking: **aOR = 1.58** (95% CI normalized: 1.52 – 1.64; cluster-robust: 1.47 – 1.70).
  * Physical inactivity (no leisure physical activity in past 30 days): **aOR = 1.21** (95% CI normalized: 1.17 – 1.24; cluster-robust: 1.14 – 1.28).
  * Overweight / Obesity (BMI $\ge 25.0$): **aOR = 1.20** (95% CI normalized: 1.17 – 1.24; cluster-robust: 1.13 – 1.28).
* **General Health Status:** Self-reported Fair or Poor general health: **aOR = 3.10** (95% CI normalized: 3.01 – 3.19; cluster-robust: 2.93 – 3.29).
* **Comorbid Cardiometabolic Conditions:**
  * Diagnosed Diabetes (`DIABETE4`): **aOR = 1.99** (95% CI normalized: 1.92 – 2.05; cluster-robust: 1.86 – 2.12).
  * Diagnosed Kidney Disease (`CHCKDNY2`): **aOR = 2.41** (95% CI normalized: 2.31 – 2.52; cluster-robust: 2.19 – 2.66).

### 4. Target Leakage Prevention & Task Definition Rule
* **Mandatory Constraint:** The calculated variable `_MICHD` is constructed by CDC SAS algorithms directly from the survey questions `CVDINFR4` (Has a doctor ever told you that you had a heart attack?) and `CVDCRHD4` (Has a doctor ever told you that you had angina or coronary heart disease?).
* **Binding Decision:** When `_MICHD` is modeled as the prediction target, `CVDINFR4` and `CVDCRHD4` must be **strictly excluded** from all feature sets, imputation routines, and model training pipelines.
* **Task Nature:** The modeling task is defined as binary classification of reported disease history from cross-sectional survey indicators. It does NOT represent prospective prediction of incident clinical events or a clinical diagnostic substitute.

### 5. Target Selection Recommendations
* **Primary Recommendation: `_MICHD` (Coronary Heart Disease / Myocardial Infarction)**
  * Leading cause of mortality in the United States.
  * 42,338 positive cases provide immense statistical power, while a 6.80% weighted prevalence (1 : 9.69 imbalance) provides an authentic, non-trivial clinical machine learning challenge requiring PR-AUC, calibration, and threshold optimization.
  * Demonstrates coherent, actionable associations with modifiable lifestyle factors suitable for population health intervention and explainability.
* **Secondary / Backup Recommendation: `DIABETE4` (Diagnosed Diabetes)**
  * Major public health burden (65,809 cases, 12.63% weighted prevalence, 1 : 5.89 imbalance).
  * Excellent candidate if lower class imbalance is desired.

### 6. Artifacts Created & Validated
* **Notebook:** [`notebooks/02_statistical_analysis.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/02_statistical_analysis.ipynb) (18 cells fully executed via `nbconvert` with complete figures and tables embedded).
* **Source Module:** [`src/analysis/statistical_tests.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/analysis/statistical_tests.py) (reusable module for candidate outcome audit, bivariate OR/Cramer's V, survey-weighted logistic regression, and forest plotting).
* **Report:** [`reports/target_selection_assessment.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/target_selection_assessment.md) (comprehensive 10-target comparative assessment).
* **Audit Report:** [`reports/milestone4_methodological_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone4_methodological_audit.md) (comprehensive methodological audit and resolution log).
* **Figures Generated:**
  1. `results/figures/fig6_outcome_feasibility_matrix.png`
  2. `results/figures/fig7_odds_ratios_forest_plot.png`
  3. `results/figures/fig8_stratified_prevalence_demographics.png`
  4. `results/figures/fig9_behavioral_risk_gradient.png`
* **Tables Generated:**
  1. `results/tables/outcome_feasibility_analysis.csv`
  2. `results/tables/bivariate_associations_summary.csv`
  3. `results/tables/multivariable_logistic_regression_or.csv`

---

## Milestone 5: Leakage-Safe Feature Engineering and Dataset Preparation

### 1. Approved Target Lock & Verified Eligible Cohort
* **Approved Primary Target:** `_MICHD` (Reported History of Coronary Heart Disease or Myocardial Infarction).
* **Task Formulation:** Binary classification of reported disease history from cross-sectional self-reported survey indicators. Observational surveillance classification; not prospective risk prediction of future disease onset or clinical diagnosis.
* **Target Definition:**
  * $y = 1$ if `_MICHD == 1` (Reported CHD or MI: $N = 42,338$).
  * $y = 0$ if `_MICHD == 2` (Did not report CHD or MI: $N = 410,126$).
  * Excluded unasked/missing records: $N = 5,206$ (1.14%).
* **Eligible Cohort Dimensions:** Exactly **452,464 respondents** (98.86% of total 457,670 survey respondents).
  * Unweighted prevalence: **9.36%**.
  * Survey-weighted prevalence: **6.80%**.
  * Class imbalance ratio: **1 : 9.69**.

### 2. Mandatory Leakage Exclusions & Feature Accounting
Audited all 61 candidate variables from the interim dictionary (`results/tables/feature_inclusion_exclusion_audit.csv`):
* **Direct Target Leakage (Excluded):**
  * `CVDINFR4` (Heart attack history): Direct mathematical constituent (`_MICHD=1` if `CVDINFR4=1`). Banned from feature space.
  * `CVDCRHD4` (Angina history): Direct mathematical constituent (`_MICHD=1` if `CVDCRHD4=1`). Banned from feature space.
* **Survey Design Metadata (Isolated):**
  * `_LLCPWT`, `_STSTR`, `_PSU`, `_STATE` are preserved in `cohort_metadata.csv.gz` for evaluation, calibration, and subgroup audit; excluded from the ordinary predictive feature matrix.
* **Redundant / High-Missingness Exclusions (16 variables):**
  * Excluded in favor of richer continuous/ordinal equivalents (`_AGE65YR`, `_RFHLTH`, `_PHYS14D`, `_MENT14D`, `_BMI5CAT`, `_RFBMI5`, `SMOKE100`, `_RFSMOK3`, `ECIGNOW3`, `EXERANY2`, `_DRNKWK3`, `_RFBING6`, `_ASTHMS1`, `PRIMINS2`) or due to high structural skip routing (`POORHLTH` [41.5%], `PREDIAB2` [65.2%]).
* **Final Included Predictors (38 features from 38 raw source variables):**
  * 4 Continuous numeric: `_AGE80` (raw `_AGE80`), `PHYSHLTH_DAYS` (raw `PHYSHLTH`: 0-30 days, code 88 -> 0, 77/99 -> NaN), `MENTHLTH_DAYS` (raw `MENTHLTH`: 0-30 days, code 88 -> 0, 77/99 -> NaN), `BMI_NUMERIC` (raw `_BMI5`: `_BMI5 / 100.0`).
  * 34 Categorical predictors across demographics, behaviors, healthcare access, comorbidities, and functional limitations. Valid survey responses (including refusal/DK) are preserved; genuine missingness is preserved as an explicit sentinel category (`-1.0`) guaranteed not to collide with valid BRFSS codes, expanding to 186 dummy indicators.

### 3. Stratified Disjoint Partitioning Strategy
* **Split Ratios:** 70% Train ($N = 316,724$), 15% Validation ($N = 67,870$), 15% Test ($N = 67,870$).
* **Random State:** Fixed at `42`, stratified strictly on binary target $y$.
* **Class Distribution Consistency:**
  * Train: 29,637 positive cases (9.36% unweighted, **6.82% survey-weighted**).
  * Validation: 6,351 positive cases (9.36% unweighted, **6.77% survey-weighted**).
  * Test: 6,350 positive cases (9.36% unweighted, **6.76% survey-weighted**).
* **Disjointness & Coverage:** Partitions are mutually disjoint ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$) and collectively cover all 452,464 eligible records.
* **Survey Design Considerations:** Stratification ensures identical class prevalence across partitions. To preserve complex survey evaluation capabilities, individual survey weights and cluster variables are retained alongside partition labels in `cohort_metadata.csv.gz`.

### 4. Leakage-Safe Preprocessing Pipeline
* **Architecture:** Scikit-learn `ColumnTransformer` with numeric pipeline (`SimpleImputer(median)` + `StandardScaler()`) and categorical pipeline (`SimpleImputer(constant, fill_value=-1.0)` + `OneHotEncoder(handle_unknown='ignore')`).
* **Train-Only Fitting:** Fitted strictly on `X_train` ($N = 316,724$). Validation and test sets are transformed using frozen training parameters, eliminating data snooping across folds.
* **Dimensionality:** Transforms 38 raw predictors into exactly **190 numerical features** (4 numeric + 186 one-hot encoded categories: 164 valid response codes + 22 explicit -1.0 missing sentinel indicators). Zero string `"nan"` categories generated.

### 5. Artifacts Created & Validated
* **Source Modules:**
  * [`src/features/feature_config.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/features/feature_config.py) (feature definitions, leakage exclusion list, partition config).
  * [`src/features/build_features.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/features/build_features.py) (cohort filtering, feature recoding, partitioning, preprocessor fitting).
* **Notebook:** [`notebooks/03_feature_engineering_and_splits.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/03_feature_engineering_and_splits.ipynb) (15 cells executed in full via `nbconvert` with 0 errors).
* **Audit Report:** [`reports/feature_engineering_and_leakage_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/feature_engineering_and_leakage_audit.md) (comprehensive feature audit and partition documentation).
* **Tables Generated:**
  1. `results/tables/feature_inclusion_exclusion_audit.csv` (all 61 variables audited with explicit source-to-engineered mapping).
  2. `results/tables/dataset_partition_summary.csv` (sample counts, class distributions, prevalences).
* **Processed Datasets (`data/processed/`):**
  1. `train_features.csv.gz` ($316,724 \times 38$) & `train_target.csv.gz` ($316,724 \times 1$)
  2. `validation_features.csv.gz` ($67,870 \times 38$) & `validation_target.csv.gz` ($67,870 \times 1$)
  3. `test_features.csv.gz` ($67,870 \times 38$) & `test_target.csv.gz` ($67,870 \times 1$)
  4. `cohort_metadata.csv.gz` ($452,464 \times 8$, containing weights, design variables, target, and partition labels)
* **Model Preprocessor Artifact:**
  * `models/preprocessor.joblib` (fitted `ColumnTransformer`).

---

## Milestone 6: Baseline and Improved Model Development

### 1. Candidate Architectures & Training Protocol
* **Candidate Scope:** Implemented four supervised classifiers via native `scikit-learn 1.9.1` without external boosting dependencies:
  1. `logistic_regression_unweighted`: L2-regularized logistic regression ($C=1.0$, L-BFGS), natural calibration.
  2. `logistic_regression_balanced`: L2-regularized logistic regression ($C=1.0$, L-BFGS) with inverse-frequency class weights ($w_1 \approx 5.34, w_0 \approx 0.55$).
  3. `hist_gradient_boosting_unweighted`: Histogram gradient boosting classifier (150 max iter, lr=0.08, 31 leaves, min samples leaf=30, L2=1.0), natural calibration.
  4. `hist_gradient_boosting_balanced`: Same histogram GBDT with internal balanced class weighting in gradient/Hessian estimation.
* **Deterministic Training:** Fixed random state `42`, fitted strictly on `X_train_trans` ($N = 316,724$, 190 features).
* **Frozen Preprocessing Contract:** Preprocessor `models/preprocessor.joblib` was executed strictly via `.transform()` with zero refitting or parameter updating on validation or test sets.

### 2. Validation-Only Model Selection & Calibration Audit
* **Primary Selection Criterion:** **PR-AUC (Average Precision)** on validation data ($N = 67,870$). In imbalanced surveillance ($9.36\%$ prevalence), PR-AUC evaluates positive predictive power across all sensitivity regimes without distortion from true negative volume.
* **Champion Model:** **HistGradientBoosting (Unweighted GBDT)** selected as Champion:
  * Validation PR-AUC: **0.3701** (Rank 1).
  * Validation ROC-AUC: **0.8449** (Rank 2, virtual tie with weighted GBDT at 0.8450).
  * Validation Brier Score: **0.0700** (Rank 1).
* **Empirical Calibration Finding:** Calibration was empirically evaluated on the validation set. Unweighted models aligned closely with observed cohort event frequencies across decile bins ($\text{Brier} = 0.0700$), whereas class weighting distorted predicted probabilities toward $\sim 0.50$ ($\text{Brier} = 0.1652$), yielding negative Brier Skill Scores ($-0.9481$). Unweighted training does not guarantee calibration in general, but on this cohort and feature space, empirical validation favored the unweighted GBDT.

### 3. Decision Threshold Optimization & Formal Lock
* **Validation Search Grid:** Grid search across $\tau \in [0.02, 0.90]$ with step $0.01$.
* **Validation Selection Objective:** At the default $0.50$ cutoff, the model missed $89.34\%$ of positive responses. The operational decision threshold was selected by maximizing Youden's J index ($J = Sensitivity + Specificity - 1$) on the validation partition, identifying **$\tau = 0.080$**:
  * Validation Sensitivity: **83.29%** | Specificity: **70.28%** | PPV: **22.44%** | NPV: **97.60%** | F1: **0.3536** | Youden's J: **0.5357**.
* **Pre-Test Formal Lock:** The champion model identity and the validation-selected operational threshold ($0.080$) were formally locked into `models/model_selection_summary.json` prior to accessing the holdout test partition.

### 4. Final Holdout Test Evaluation
* **Quarantine Release:** Evaluated strictly after model and threshold locking in a single unbiased forward pass on the holdout test partition ($N = 67,870$, $6,350$ positive cases).
* **Final Unweighted Sample Results:**
  * PR-AUC: **0.3808** | ROC-AUC: **0.8492** | Brier Score: **0.0694**.
  * At locked threshold $\tau = 0.080$: Sensitivity = **83.91%** ($5,328 / 6,350$ cases identified), Specificity = **70.45%**, PPV = **22.67%**, NPV = **97.69%**, F1 = **0.3569**, Balanced Accuracy = **77.18%**.
* **Final Survey-Weighted Population Estimates (`_LLCPWT`):**
  * PR-AUC: **0.3594** | ROC-AUC: **0.8726** | Brier Score: **0.0514**.
  * At locked threshold $\tau = 0.080$: Sensitivity = **80.44%**, Specificity = **79.46%**, PPV = **22.12%**, NPV = **98.42%**, F1 = **0.3469**, Balanced Accuracy = **79.95%**.
  * These represent descriptive population-weighted point estimates; full complex-survey variance estimation was not conducted.

### 5. Surveillance Framing & Limitations Disclosed
* **Cross-Sectional Classification:** The model classifies self-reported historical physician diagnoses (`_MICHD`), not prospective incidence or clinical diagnoses. **Clinical utility has not been established;** the reported sensitivity and negative predictive value do not establish clinical screening validity or rule-out capability in clinical practice.
* **Survey Weighting Disclosures:** Models were trained by minimizing unweighted sample loss; weighted metrics are descriptive population point estimates without complex survey design standard errors.


### 6. Artifacts Created & Validated
* **Source Modules:**
  * [`src/models/model_config.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/models/model_config.py)
  * [`src/models/train_models.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/models/train_models.py)
  * [`src/evaluation/evaluate_models.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/evaluation/evaluate_models.py)
* **Model Artifacts:**
  * `models/baseline_logistic_regression.joblib`
  * `models/weighted_logistic_regression.joblib`
  * `models/improved_hist_gradient_boosting.joblib` (Champion)
  * `models/weighted_hist_gradient_boosting.joblib`
  * `models/model_selection_summary.json` (Formal Lock)
* **Tables Generated:**
  * `results/tables/model_validation_comparison.csv`
  * `results/tables/threshold_optimization_analysis.csv`
  * `results/tables/final_test_evaluation_summary.csv`
* **Figures Generated:**
  * `results/figures/fig10_roc_curves_validation.png`
  * `results/figures/fig11_pr_curves_validation.png`
  * `results/figures/fig12_calibration_curves_validation.png`
  * `results/figures/fig13_threshold_tradeoffs.png`
  * `results/figures/fig14_champion_test_confusion_matrices.png`
  * `results/figures/fig15_unweighted_vs_survey_weighted_test.png`
* **Notebook:** [`notebooks/04_model_development_and_evaluation.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/04_model_development_and_evaluation.ipynb) (Executed in full via `nbconvert`, 2.0 MB).
* **Audit Report:** [`reports/milestone6_model_development_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone6_model_development_audit.md).

---

## Milestone 7: Explainable Artificial Intelligence (XAI)

### 1. Feature Mapping Architecture & Bijective Audit (Stage 1)
* **Design Decision:** Built a deterministic, bijective mapping utility (`src/xai/validate_feature_mapping.py`) connecting the 38 original survey variables to the 190 preprocessed features generated by `models/preprocessor.joblib`.
* **Verification Contract:** Verified exact index alignment, unique column naming, and disjoint variable grouping without re-fitting or altering the preprocessor.
* **Audit Table:** Exported `results/tables/xai_feature_mapping_audit.csv` auditing all 4 continuous scaled columns, 186 categorical dummy indicators, and 22 missing sentinel indicators.

### 2. Grouped Permutation Importance & Partial Dependence (Stage 2)
* **Grouped Permutation Decision:** To avoid the indicator dilution flaw of independent one-hot permutation and prevent synthetic categorical collisions, all dummy columns belonging to a parent survey variable are shuffled simultaneously using a shared row permutation index across 10 repeats (seed 42).
* **Primary Metric:** Evaluated primarily by **PR-AUC decrease** due to outcome class imbalance ($9.36\%$ validation prevalence).
* **Key Global Rankings:** Continuous age (`_AGE80`, PR-AUC drop $0.0772$), self-rated general health (`GENHLTH`, drop $0.0434$), biological sex (`_SEX`, drop $0.0380$), stroke history (`CVDSTRK3`, drop $0.0306$), and provider continuity (`PERSDOC3`, drop $0.0127$).
* **Partial Dependence Unscaling:** Computed recursion PDP on continuous predictors (`_AGE80`, `PHYSHLTH_DAYS`, `BMI_NUMERIC`), dynamically unscaling standardized grids back to natural epidemiological units (years, days, $\text{kg/m}^2$) via fitted `StandardScaler` mean and scale parameters, with outputs mapped to marginal predicted probabilities via the logistic sigmoid.

### 3. Local Explainability & Case Archetypes (Stage 3)
* **Model-Agnostic Decision:** Because `shap` and `lime` are not installed in the verified environment, custom, dependency-safe local explanation methods were implemented using `scikit-learn`, `numpy`, and `scipy`:
  1. *Grouped Marginal Perturbation Attribution ($\Delta p$):* Measures directional risk shifts relative to an empirical validation background sample ($M = 200$, seed 42), isolating whether a respondent's state elevates or mitigates risk.
  2. *Local Ridge Linear Surrogates:* Fits weighted Ridge regressions on $N = 1,000$ neighborhood perturbations weighted by an exponential distance kernel, demonstrating high local fidelity ($R^2 \in [0.931, 0.969]$).
* **Objective Case Selection:** Deterministically selected 4 validation respondents representing the median predicted probability of each confusion matrix quadrant at the locked threshold ($0.080$):
  - *True Positive (`VALIDATION_ID_56065`, $\hat{p} = 0.267$):* Driven by age 76, fair health, kidney disease, and diabetes.
  - *False Positive (`VALIDATION_ID_279669`, $\hat{p} = 0.166$):* Screening alert driven by poor health and 30 days of physical limitation.
  - *True Negative (`VALIDATION_ID_253891`, $\hat{p} = 0.016$):* Protected by young age (41) and complete absence of chronic disease despite Class I obesity.
  - *False Negative (`VALIDATION_ID_44471`, $\hat{p} = 0.044$):* Favorable self-reported vitality ("Very Good" health) and normal BMI mask disease history, suppressing predicted risk below threshold despite age 73.

### 4. Validation-Only Development & Holdout Quarantine Policy
* **Decision:** All global feature importance estimations, partial dependence profiles, and local explanation analyses were conducted strictly on the **validation partition** ($N = 67,870$). The holdout test partition remained completely untouched, preserving its integrity for final project auditing.

### 5. Surveillance Framing & Non-Causal Disclosures
* **Binding Limitation:** Feature attributions describe associative properties of the gradient-boosted decision trees within cross-sectional surveillance data; they **do not** represent biological causation, prospective incidence risk, or confirmed clinical diagnoses. Clinical utility has not been established.

### 6. Artifacts Created & Validated (Milestone 7)
* **Source Modules:**
  * [`src/xai/xai_config.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/xai_config.py)
  * [`src/xai/validate_feature_mapping.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/validate_feature_mapping.py)
  * [`src/xai/global_importance.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py)
  * [`src/xai/local_explanations.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/local_explanations.py)
* **Unit Test Modules:**
  * [`tests/test_xai_feature_mapping.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_xai_feature_mapping.py) (4 tests)
  * [`tests/test_xai_global_importance.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_xai_global_importance.py) (7 tests)
  * [`tests/test_xai_local_explanations.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_xai_local_explanations.py) (8 tests)
* **Tables Generated:**
  * `results/tables/xai_feature_mapping_audit.csv` (190 transformed features mapped)
  * `results/tables/xai_global_variable_importance.csv` (38 original predictors ranked)
  * `results/tables/xai_transformed_feature_importance.csv` (190 indicators ranked)
  * `results/explanations/xai_case_archetypes_summary.csv` (4 validation archetypes)
  * `results/explanations/xai_local_case_explanations.csv` (152 itemized local attributions)
* **Figures Generated:**
  * `results/figures/fig16_global_variable_importance.png` (Top 20 predictors with $\pm 1\,\text{SD}$ bars)
  * `results/figures/fig17_indicator_feature_importance.png` (Top 25 indicators)
  * `results/figures/fig18_partial_dependence_profiles.png` (3 continuous predictor PDPs unscaled)
  * `results/figures/fig19_local_case_attributions.png` (4-panel local directional attribution bars)
* **Executed Notebook:** [`notebooks/05_explainable_ai_analysis.ipynb`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/05_explainable_ai_analysis.ipynb) (Executed in full across all 14 sections, 2.26 MB).
* **Audit Reports:**
  * [`reports/milestone7_xai_planning_and_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_xai_planning_and_audit.md)
  * [`reports/milestone7_global_importance_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md)
  * [`reports/milestone7_stage2_data_integrity_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_stage2_data_integrity_audit.md)
  * [`reports/milestone7_stage3_local_explainability_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_stage3_local_explainability_audit.md)
  * [`reports/milestone7_stage4_notebook_and_documentation_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_stage4_notebook_and_documentation_audit.md)




