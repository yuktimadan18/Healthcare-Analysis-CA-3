# Feature Engineering & Leakage Audit Report: Milestone 5
**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Milestone:** Milestone 5 (Leakage-Safe Feature Engineering & Reproducible Dataset Preparation)
**Date:** October 2026
**Primary Target:** `_MICHD` (Reported History of Coronary Heart Disease or Myocardial Infarction)

---

## 1. Executive Summary

Milestone 5 establishes the feature engineering, leakage prevention boundary, and reproducible dataset partitioning for the predictive modeling framework.

Following formal approval of `_MICHD` as the primary prediction target, this milestone:
1. **Verified the Target Coding:** Confirmed the eligible surveillance cohort of **452,464 respondents** (42,338 positive cases, 410,126 negative cases; 9.36% unweighted, 6.80% survey-weighted prevalence; 1 : 9.69 class imbalance).
2. **Enforced Strict Leakage Prevention:** Audited all 61 candidate variables. Direct constituents `CVDINFR4` (heart attack) and `CVDCRHD4` (angina) were permanently excluded from the feature space.
3. **Isolated Survey Metadata:** Retained `_LLCPWT`, `_STSTR`, `_PSU`, and `_STATE` strictly in a dedicated metadata table for audit and evaluation, preventing artificial memorization of survey clusters or geographic IDs.
4. **Engineered 38 Clean Predictors:** 4 continuous numeric features + 34 categorical risk factors across demographics, health status, lifestyle behaviors, comorbidities, healthcare access, and functional disabilities.
5. **Constructed a Leakage-Safe Preprocessor:** Built a scikit-learn `ColumnTransformer` (median imputation + standard scaling for numeric; most-frequent imputation + one-hot encoding with `handle_unknown='ignore'` for categorical) and fitted it **strictly on the training partition only**.
6. **Executed Stratified 70/15/15 Partitioning:** Created disjoint training ($N = 316,724$), validation ($N = 67,870$), and test ($N = 67,870$) partitions with identical class distributions (9.36% unweighted, ~6.80% survey-weighted) using fixed seed (42).

---

## 2. Target Definition & Eligible Cohort Verification

### Codebook Definition
Under CDC BRFSS 2024 documentation (`USCODE24_LLCP_082125.HTML`), `_MICHD` is a calculated binary variable:
$$\text{_MICHD} = \begin{cases} 1 & \text{if } \text{CVDINFR4} = 1 \text{ or } \text{CVDCRHD4} = 1 \quad \text{(Reported having MI or CHD)} \\ 2 & \text{if } \text{CVDINFR4} = 2 \text{ and } \text{CVDCRHD4} = 2 \quad \text{(Did not report MI or CHD)} \\ \text{NaN} & \text{if either question is 7 (Don't Know), 9 (Refused), or Missing} \end{cases}$$

### Cohort Accounting ($N = 457,670$)
- **Total Survey Respondents:** 457,670
- **Excluded Unasked / Missing Target:** 5,206 (1.14%)
- **Eligible Surveillance Cohort ($N$):** **452,464** (98.86%)
  - **Positive Cases ($y = 1$):** 42,338 (9.36% unweighted, **6.80% survey-weighted**)
  - **Negative Cases ($y = 0$):** 410,126 (90.64% unweighted, 93.20% survey-weighted)
  - **Imbalance Ratio:** 1 : 9.69 (1 positive case per 9.69 negative controls)

---

## 3. Comprehensive Feature Inclusion & Exclusion Audit

Every variable from `data/interim/brfss_2024_selected.csv.gz` was assigned a documented status and rationale in `results/tables/feature_inclusion_exclusion_audit.csv`:

| Status Category | Raw Variable Count | Raw Source Variables | Final Engineered Model Column(s) | Clinical & Methodological Rationale |
| :--- | :---: | :--- | :--- | :--- |
| **Primary Target** | 1 | `_MICHD` | `target` (in metadata/target) | Approved prediction outcome (Reported MI/CHD). Recoded to binary indicator (1/0). Strictly excluded from feature matrix $X$. |
| **Excluded: Direct Target Leakage** | 2 | `CVDINFR4`, `CVDCRHD4` | None (Excluded) | Direct mathematical constituents of `_MICHD` (`_MICHD=1` if `CVDINFR4=1` or `CVDCRHD4=1`). Permanently excluded from feature space to eliminate deterministic leakage. |
| **Preserved: Survey Metadata** | 4 | `_LLCPWT`, `_STSTR`, `_PSU`, `_STATE` | `_LLCPWT`, `_STSTR`, `_PSU`, `_STATE` (in metadata) | Preserved in `cohort_metadata.csv.gz` for survey weighting, clustering, and geographic audit. Excluded from predictive feature matrix $X$ to prevent shortcut learning. |
| **Excluded: Redundant / Skip Pattern** | 16 | `_AGE65YR`, `_RFHLTH`, `_PHYS14D`, `_MENT14D`, `POORHLTH`, `_BMI5CAT`, `_RFBMI5`, `SMOKE100`, `_RFSMOK3`, `ECIGNOW3`, `EXERANY2`, `_DRNKWK3`, `_RFBING6`, `PREDIAB2`, `_ASTHMS1`, `PRIMINS2` | None (Excluded) | Excluded in favor of richer continuous/ordinal equivalents (e.g., continuous age `_AGE80`, continuous BMI, 4-level smoking `_SMOKER3`, 5-level general health `GENHLTH`) or due to high structural skip routing (`POORHLTH` [41.5%], `PREDIAB2` [65.2%]). |
| **Source for Engineered Numeric Predictor** | 4 | `_AGE80`, `PHYSHLTH`, `MENTHLTH`, `_BMI5` | `_AGE80`, `PHYSHLTH_DAYS`, `MENTHLTH_DAYS`, `BMI_NUMERIC` | Clean continuous clinical and lifestyle measurements. Raw codes transformed to true physiological scales (e.g., `_BMI5` divided by 100.0; code 88 mapped to 0 days; codes 77/99 to NaN). |
| **Included: Categorical Predictor** | 34 | `_SEX`, `_AGEG5YR`, `_RACE`, `_EDUCAG`, `_INCOMG1`, `MARITAL`, `EMPLOY1`, `_METSTAT`, `_URBSTAT`, `GENHLTH`, `_SMOKER3`, `_CURECI3`, `_TOTINDA`, `DRNKANY6`, `_RFDRHV9`, `_HLTHPL2`, `PERSDOC3`, `MEDCOST1`, `CHECKUP1`, `DIABETE4`, `ASTHMA3`, `CHCCOPD3`, `ADDEPEV3`, `CHCKDNY2`, `HAVARTH4`, `CHCSCNC1`, `CHCOCNC1`, `CVDSTRK3`, `DEAF`, `BLIND`, `DECIDE`, `DIFFWALK`, `DIFFDRES`, `DIFFALON` | 34 raw columns (yielding 186 OHE categories: 164 valid response codes + 22 explicit -1.0 missing sentinel indicators) | Clinically vetted risk factors, comorbidities, healthcare access indicators, and functional limitations. Valid survey responses (including refusal/DK) are preserved; genuine missingness is preserved as an explicit sentinel category (-1.0). |

**Total Audit Accounting:** Exactly 61 raw variables accounted for (38 included model predictors [4 numeric + 34 categorical], 4 survey metadata, 1 primary target, 2 direct leakage exclusions, 16 redundant exclusions).

---

## 4. Response Code Handling & Missing-Value Strategy

In strict accordance with **Rule 8** (*"Never globally replace refusal or don't-know codes with ordinary values"*):
1. **Continuous Days (`PHYSHLTH`, `MENTHLTH` $\rightarrow$ `PHYSHLTH_DAYS`, `MENTHLTH_DAYS`):**
   - Codebook: `1–30` = Number of days; `88` = None (0 days); `77` = Don't know; `99` = Refused.
   - Code `88` was explicitly mapped to `0.0` (zero impairment days).
   - Codes `77` and `99` were mapped to `np.nan` so that missing-value imputation in the scikit-learn pipeline estimates them based on training medians without conflating non-response with 0 days or 30 days.
2. **Continuous BMI (`_BMI5` $\rightarrow$ `BMI_NUMERIC`):**
   - Rescaled by dividing by 100.0 (implied two decimals). Valid values range from 12.0 to 99.8 kg/m$^2$.
   - Missing responses (6.52% in cohort) are retained as `np.nan` for pipeline median imputation.
3. **Categorical Features & Missingness Strategy:**
   - Categorical columns are kept in their native numeric code representations (with `np.nan` for genuine missingness).
   - Valid response codes (e.g., `1` = Yes, `2` = No, `7` = Don't Know, `9` = Refused) are preserved as distinct numerical levels, ensuring that respondents who refused or did not know are modeled as separate behavioral categories from negative controls.
   - Genuine missing values (e.g., unasked questions or territorial missingness like `_METSTAT` in Guam/Puerto Rico) are retained as `np.nan` and are NOT converted into the literal string `"nan"`.
   - `SimpleImputer(strategy='constant', fill_value=-1.0)` inside the categorical pipeline replaces genuine missing values with an explicit sentinel category (`-1.0`) guaranteed not to collide with non-negative valid BRFSS codes.
   - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` expands the 34 categorical predictors into 186 dummy variables (164 for valid survey response codes + 22 for explicit missing sentinel categories). Any unseen levels encountered at inference time are safely ignored.

---

## 5. Dataset Partitioning & Survey Design Considerations

### Partition Strategy
The eligible cohort ($N = 452,464$) was partitioned into disjoint Train, Validation, and Test sets using stratified random sampling on the target:
- **Random State:** `42`
- **Stratification Variable:** Binary target ($y \in \{0, 1\}$)
- **Train Split:** **70.0%** ($N = 316,724$)
- **Validation Split:** **15.0%** ($N = 67,870$)
- **Test Split:** **15.0%** ($N = 67,870$)

### Partition Distribution Summary (`results/tables/dataset_partition_summary.csv`)

| Partition | Total Records | % of Cohort | Positive Cases | Negative Cases | Unweighted Prev (%) | Survey-Weighted Prev (%) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **316,724** | 70.0% | 29,637 | 287,087 | 9.36% | **6.82%** | 1 : 9.69 |
| **Validation** | **67,870** | 15.0% | 6,351 | 61,519 | 9.36% | **6.77%** | 1 : 9.69 |
| **Test** | **67,870** | 15.0% | 6,350 | 61,520 | 9.36% | **6.76%** | 1 : 9.69 |
| **Full Cohort** | **452,464** | 100.0% | 42,338 | 410,126 | 9.36% | **6.80%** | 1 : 9.69 |

### Disjointness & Coverage Verification
- $\text{Train} \cap \text{Val} = \emptyset$
- $\text{Train} \cap \text{Test} = \emptyset$
- $\text{Val} \cap \text{Test} = \emptyset$
- $|\text{Train}| + |\text{Val}| + |\text{Test}| = 316,724 + 67,870 + 67,870 = 452,464 \equiv |\text{Eligible Cohort}|$

### Complex Survey Design & State-Grouped Splitting Analysis
- **Limitations of Stratified Random Splitting:** Stratified random splitting generates independent partitions at the individual respondent level. However, because BRFSS employs a multi-stage clustered survey design (clustering within primary sampling units `_PSU` and stratification within `_STSTR`), observations from the same cluster or geographic stratum can be allocated to both training and test partitions.
- **Why Stratification Does Not Mean Nationally Representative:** Stratification on the target ensures equal disease prevalence across partitions, but does not alter the underlying complex sampling probabilities. True population estimates require survey weighting (`_LLCPWT`), which has been preserved in `cohort_metadata.csv.gz`.
- **State-Grouped Partitioning Trade-Off:**
  - A state-grouped split (e.g., training on 40 states, validating on 5, testing on 5) answers a fundamentally different generalization question: out-of-domain geographic transferability.
  - Because states exhibit varying sample sizes, distinct optional module inclusions, and demographic heterogeneity, a state-grouped split tests transferability across state health systems rather than national population risk estimation.
  - Therefore, the stratified random split is retained as the primary benchmark, with state-level subgroup auditing supported via the metadata table.

---

## 6. Preprocessing Pipeline Architecture

In strict adherence to **Rule 9** (*"Fit all learned transformations using training data only"*):
1. **Pipeline Structure:**
   ```text
   ColumnTransformer:
   ├── Numeric Pipeline (4 features):
   │   ├── SimpleImputer(strategy='median')
   │   └── StandardScaler()
   └── Categorical Pipeline (34 features):
       ├── SimpleImputer(strategy='constant', fill_value=-1.0)
       └── OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   ```
2. **Leakage Prevention Verification:**
   - `preprocessor.fit(X_train)` was executed strictly on the training partition ($N = 316,724$).
   - The validation and test sets were transformed using the frozen training medians, standard deviations, and category levels.
   - Target vector $y$ and survey design variables were completely detached during fitting and transformation.
3. **Transformed Dimensions:**
   - The 38 raw predictors expand into exactly **190 transformed feature columns** following one-hot encoding:
     - 4 continuous numeric features.
     - 186 one-hot encoded categorical indicator columns (164 for valid survey response levels + 22 for explicit -1.0 missingness sentinel indicators).
   - No string `"nan"` categories were generated.
4. **Serialization:**
   - Saved fitted preprocessor to `models/preprocessor.joblib`.

---

## 7. Generated Data Artifacts (`data/processed/`)

All processed datasets are saved as gzip-compressed CSV files:

| Artifact Path | Format | Dimensions | Purpose |
| :--- | :--- | :---: | :--- |
| `data/processed/train_features.csv.gz` | CSV (gzip) | 316,724 rows $\times$ 38 cols | Training predictor matrix ($X_{\text{train}}$). |
| `data/processed/train_target.csv.gz` | CSV (gzip) | 316,724 rows $\times$ 1 col | Training binary target ($y_{\text{train}} \in \{0, 1\}$). |
| `data/processed/validation_features.csv.gz` | CSV (gzip) | 67,870 rows $\times$ 38 cols | Validation predictor matrix ($X_{\text{val}}$). |
| `data/processed/validation_target.csv.gz` | CSV (gzip) | 67,870 rows $\times$ 1 col | Validation binary target ($y_{\text{val}} \in \{0, 1\}$). |
| `data/processed/test_features.csv.gz` | CSV (gzip) | 67,870 rows $\times$ 38 cols | Holdout test predictor matrix ($X_{\text{test}}$). |
| `data/processed/test_target.csv.gz` | CSV (gzip) | 67,870 rows $\times$ 1 col | Holdout test binary target ($y_{\text{test}} \in \{0, 1\}$). |
| `data/processed/cohort_metadata.csv.gz` | CSV (gzip) | 452,464 rows $\times$ 8 cols | Survey weights (`_LLCPWT`), strata (`_STSTR`), cluster (`_PSU`), state (`_STATE`), raw target (`_MICHD`), binary `target`, `record_id`, and `partition`. |
| `models/preprocessor.joblib` | Scikit-learn binary | 12.2 KB | Fitted ColumnTransformer for model inference. |

### Row Alignment & Indexing
- All partition feature matrices and target vectors preserve the original DataFrame index matching `record_id`.
- `cohort_metadata.csv.gz` contains exactly 8 named columns (`record_id`, `_MICHD`, `target`, `_LLCPWT`, `_STSTR`, `_PSU`, `_STATE`, `partition`) and is index-aligned with the feature matrices.

---

## 8. Methodological Boundaries and Readiness for Milestone 6

Milestone 5 is complete, fully executed, and verified.

### Methodological Boundaries:
1. **Cross-Sectional Observational Nature:** This modeling task is strictly defined as binary classification of reported disease history from concurrent self-reported survey indicators. It does NOT represent prospective prediction of incident clinical events, causal inference, or a clinical diagnostic tool.
2. **Holdout Set Integrity:** The test partition has been generated and validated for row alignment and dimensions, but has NOT been used for model selection, hyperparameter tuning, or threshold setting.
3. **No Model Training:** No baseline or predictive models were trained in Milestone 5.

The pipeline outputs provide clean, leakage-safe, reproducible training, validation, and test datasets ready for Milestone 6 (Baseline and Improved Model Development).
