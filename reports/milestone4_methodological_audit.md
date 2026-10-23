# Methodological Audit & Corrections Report: Milestone 4
**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Audit Scope:** Milestone 4 (Statistical Analysis & Knowledge Discovery)
**Date:** October 2026
**Auditor:** Statistical Pair Programmer / Independent Audit Pipeline

---

## 1. Audit Scope & Inspected Artifacts

A comprehensive methodological audit was conducted on all Milestone 4 analytical code, outputs, and documentation. The inspection included:
- `PROJECT_RULES.md` (Binding constraints: Rules 6, 8, 9, 10, 11, 12, 13).
- `src/analysis/statistical_tests.py` (Core statistical functions).
- `notebooks/02_statistical_analysis.ipynb` (Executed Jupyter notebook).
- `reports/target_selection_assessment.md` (Target selection report).
- `results/tables/outcome_feasibility_analysis.csv`
- `results/tables/bivariate_associations_summary.csv`
- `results/tables/multivariable_logistic_regression_or.csv`
- `data/interim/brfss_2024_dictionary.json`
- Official CDC documentation: `codebook24_llcp-v2-508.zip` (`USCODE24_LLCP_082125.HTML`), `2024-calculated-variables-version4-508.pdf`, and `Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2024-508.pdf`.

---

## 2. Confirmed Issues & Severity Classification

| Issue ID | Identified Defect | Affected Component | Severity | Resolution Status |
| :--- | :--- | :--- | :---: | :---: |
| **AUD-01** | **Weight Scale Misinterpretation in GLM:** Passing raw expansion weights `_LLCPWT` directly into `statsmodels.api.GLM` with `freq_weights` treated the population expansion total (~217.6M) as integer cell replications. This artificially shrunk standard errors by $\approx 24\times$, generating spuriously narrow Wald 95% CIs (e.g., Age 65+ aOR 4.57 with CI [4.56, 4.58]). | `src/analysis/statistical_tests.py`, `notebooks/02_statistical_analysis.ipynb`, `multivariable_logistic_regression_or.csv` | **HIGH** | **RESOLVED**: Weights normalized to sample size $N = 385,120$; cluster-robust sensitivity on `_PSU` implemented. |
| **AUD-02** | **Conflation of Unweighted and Weighted Quantities in Bivariate Table:** The bivariate association output table juxtaposed unweighted Wald CIs and unweighted Pearson $\chi^2$ test statistics with survey-weighted OR point estimates without explicit distinction. | `src/analysis/statistical_tests.py`, `bivariate_associations_summary.csv`, Figure 7 | **MEDIUM** | **RESOLVED**: Renamed columns to `unweighted_or`, `unweighted_ci_95_lower/upper`, `unweighted_chi2`, `weighted_or`; updated Figure 7 error bars and legends. |
| **AUD-03** | **Omission of Special Response Codes in Feasibility Accounting:** For `DIABETE4`, code 2 ("Yes, but female told only during pregnancy", $N = 3,395$) was neither positive, negative, DK, refused, nor NaN, causing 3,395 respondents to be omitted from the row breakdown (row sum = 454,275 instead of 457,670). | `src/analysis/statistical_tests.py`, `outcome_feasibility_analysis.csv` | **MEDIUM** | **RESOLVED**: Added `other_special_count` and strict checksum assertion requiring exact row sum of 457,670. |
| **AUD-04** | **Ambiguity in Predictive Task Definition:** Preliminary narrative in the assessment report occasionally described the task with phrasing suggesting prospective clinical risk prediction or diagnostic decision-making. | `reports/target_selection_assessment.md`, `notebooks/02_statistical_analysis.ipynb` | **MEDIUM** | **RESOLVED**: Standardized task definition as "Binary classification of reported disease history from cross-sectional survey indicators"; explicitly barred causal and prospective claims. |
| **AUD-05** | **Complete Case Sample Size Reporting Discrepancy:** Complete cases across modeled predictors in multivariable GLM were previously referenced as 405,091 instead of the exact valid complete case count of 385,120. | Documentation and table captions | **LOW** | **RESOLVED**: Corrected complete-case cohort count to $N = 385,120$. |

---

## 3. Codebook Validation Results

Every variable utilized in the feasibility audit, bivariate tests, and multivariable regressions was audited against the official CDC 2024 Codebook (`USCODE24_LLCP_082125.HTML`):

1. **`_MICHD` (Coronary Heart Disease / Myocardial Infarction):**
   - Codebook definition: Calculated variable derived from `CVDINFR4` and `CVDCRHD4`.
   - `1` = Reported having MI or CHD (`CVDINFR4=1` OR `CVDCRHD4=1`): **42,338 respondents** (9.36% unweighted, 6.80% survey-weighted).
   - `2` = Did not report having MI or CHD (`CVDINFR4=2` AND `CVDCRHD4=2`): **410,126 respondents** (90.64% unweighted, 93.20% survey-weighted).
   - `BLANK` = Missing / Don't Know / Refused in either component: **5,206 respondents**.
   - *Audit Check:* Verified that `_MICHD` has no raw numeric codes `7` or `9`; missingness is exclusively represented as `BLANK` (NaN).
2. **`CVDINFR4` (Heart Attack History) & `CVDCRHD4` (Angina History):**
   - Direct survey questions from Core Section 6 (Chronic Health Conditions).
   - `1` = Yes, `2` = No, `7` = Don't Know, `9` = Refused, `BLANK` = Missing.
   - *Audit Check:* Sums of positive, negative, DK, refused, and blank equal exactly 457,670 for both variables.
3. **`_RFSMOK3` (Current Cigarette Smoking Calculated Variable):**
   - Derived from `_SMOKER3` (which in turn is derived from `SMOKE100` and `SMOKEDAY`).
   - `1` = No (Non-smoker: `_SMOKER3` is 3 [Former smoker] or 4 [Never smoked]): **378,554**.
   - `2` = Yes (Current smoker: `_SMOKER3` is 1 [Every day] or 2 [Some days]): **47,094**.
   - `9` = Don't know / Refused / Missing (`_SMOKER3 = 9`): **32,022**.
   - *Audit Check:* Verified reference category in modeling: code `1` is unexposed (non-smoker), code `2` is exposed (current smoker).
4. **`_RFBMI5` (Overweight / Obese Calculated Variable):**
   - Derived from `_BMI5` (which has 2 implied decimal places, calculated as weight in kg / height in meters squared).
   - `1` = No ($\text{BMI} < 25.00$): **128,430**.
   - `2` = Yes ($\text{BMI} \ge 25.00$): **286,203**.
   - `9` = Don't know / Refused / Missing ($\text{BMI} = 9999$): **43,037**.
   - *Audit Check:* Verified threshold is $\ge 25.00$; reference category is normal/underweight (code 1).
5. **`_TOTINDA` (Leisure Physical Activity Calculated Variable):**
   - Derived from `EXERANY2`.
   - `1` = Had physical activity or exercise in past 30 days outside regular job: **350,061**.
   - `2` = No physical activity or exercise: **106,294**.
   - `9` = Don't know / Refused / Missing: **1,315**.
   - *Audit Check:* Verified code `2` indicates physical inactivity.
6. **`_AGE65YR`, `_SEX`, `_RFHLTH`:**
   - `_AGE65YR`: `1` = 18–64 (278,326), `2` = 65+ (171,034), `3` = DK/Ref/Missing (8,310).
   - `_SEX`: `1` = Male (217,507), `2` = Female (240,163).
   - `_RFHLTH`: `1` = Good/Better (GENHLTH 1–3, 366,237), `2` = Fair/Poor (GENHLTH 4–5, 90,123), `9` = DK/Ref/Missing (1,310).
7. **`DIABETE4` & `CHCKDNY2`:**
   - `DIABETE4`: `1` = Yes (65,809), `2` = Gestational (3,395), `3` = No (376,125), `4` = Borderline/Pre-diabetes (11,307), `7` = DK (798), `9` = Refused (232), `BLANK` = 4.
   - `CHCKDNY2`: `1` = Yes (23,752), `2` = No (431,939), `7` = DK (1,715), `9` = Refused (258), `BLANK` = 6.

---

## 4. Weighting & Variance-Estimation Assessment

### Investigation of Artificial Confidence Interval Shrinkage (AUD-01)
The reported confidence interval of $[4.56, 4.58]$ for Age 65+ was subjected to mathematical and empirical investigation:
1. **Root Cause Identification:**
   - In `statsmodels.api.GLM`, the parameter `freq_weights` designates *frequency weights*, which signify that an observation is replicated $w_i$ identical times in the dataset.
   - BRFSS `_LLCPWT` values are *sampling expansion weights* designed to expand the survey sample to the non-institutionalized US civilian adult population.
   - Across the complete-case cohort ($N = 385,120$), $\sum w_i = 217,602,166.0$, with a mean weight $\bar{w} = 565.02$.
   - The Hessian matrix in GLM estimation scales linearly with $\sum w_i$:
     $$\mathcal{I}(\beta) = X^T W X = \sum_{i=1}^N w_i \mu_i (1 - \mu_i) x_i x_i^T$$
   - Consequently, the asymptotic covariance matrix $\text{Cov}(\hat{\beta}) = \mathcal{I}^{-1}$ was divided by 217.6 million rather than the actual sample size of 385,120, artificially shrinking the standard errors by $\sqrt{\bar{w}} \approx \sqrt{565} \approx 23.77\times$.
2. **Empirical Verification:**
   - Raw expansion weights: $SE(\hat{\beta}_{\text{age65}}) = 0.0006 \implies \text{Wald } 95\%\text{ CI on aOR} = [4.564, 4.575]$.
   - Sample-size normalized weights ($w_i^* = w_i \times \frac{N}{\sum w_i}$):
     $$SE(\hat{\beta}_{\text{age65}}) = 0.0140 \implies \text{Wald } 95\%\text{ CI on aOR} = [4.446, 4.696]$$
   - Point estimates are completely identical ($\hat{\beta} = 1.5194, \text{aOR} = 4.569$) because the score equations $\sum w_i^* (y_i - \mu_i) x_i = 0$ are invariant to constant scalar multiplication.
3. **Cluster-Robust Sensitivity Analysis:**
   - In complex survey sampling, respondents clustered within primary sampling units (`_PSU`) exhibit positive intra-cluster correlation ($\rho > 0$), resulting in a design effect:
     $$\text{DEFF} \approx 1 + (m - 1)\rho$$
   - A cluster-robust covariance model clustering on `_PSU` yields:
     $$SE_{\text{cluster}}(\hat{\beta}_{\text{age65}}) = 0.0291 \implies \text{Cluster-robust } 95\%\text{ CI on aOR} = [4.316, 4.838]$$

### Multivariable Model Comparison Table ($N = 385,120$)

| Covariate | Coef ($\hat{\beta}$) | aOR | $SE$ (Normalized) | 95% CI (Normalized) | $SE$ (Cluster on `_PSU`) | 95% CI (Cluster `_PSU`) | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | -4.2951 | 0.014 | 0.0190 | [0.013, 0.014] | 0.0370 | [0.013, 0.015] | $< 10^{-300}$ |
| **`age_65plus`** | 1.5194 | **4.569** | 0.0140 | **[4.446, 4.696]** | 0.0291 | **[4.316, 4.838]** | $< 10^{-300}$ |
| **`male_sex`** | 0.5484 | **1.730** | 0.0140 | **[1.684, 1.779]** | 0.0283 | **[1.637, 1.829]** | $< 10^{-300}$ |
| **`bmi_overweight_obese`** | 0.1859 | **1.204** | 0.0158 | **[1.168, 1.242]** | 0.0314 | **[1.132, 1.281]** | $4.79 \times 10^{-32}$ |
| **`current_smoker`** | 0.4589 | **1.582** | 0.0192 | **[1.524, 1.643]** | 0.0378 | **[1.469, 1.704]** | $2.28 \times 10^{-126}$ |
| **`physical_inactivity`** | 0.1873 | **1.206** | 0.0152 | **[1.171, 1.243]** | 0.0308 | **[1.135, 1.281]** | $7.52 \times 10^{-35}$ |
| **`fair_poor_health`** | 1.1319 | **3.102** | 0.0147 | **[3.014, 3.192]** | 0.0299 | **[2.925, 3.289]** | $< 10^{-300}$ |
| **`diabetes`** | 0.6856 | **1.985** | 0.0158 | **[1.924, 2.047]** | 0.0331 | **[1.860, 2.118]** | $< 10^{-300}$ |
| **`kidney_disease`** | 0.8807 | **2.413** | 0.0217 | **[2.312, 2.517]** | 0.0492 | **[2.191, 2.657]** | $< 10^{-300}$ |

---

## 5. Target Leakage Review

A rigorous audit was performed to identify and prevent data leakage:
1. **Constituent Variables of `_MICHD`:**
   - CDC calculated variable documentation explicitly defines `_MICHD` as:
     $$\text{_MICHD} = \begin{cases} 1 & \text{if } \text{CVDINFR4} = 1 \text{ or } \text{CVDCRHD4} = 1 \\ 2 & \text{if } \text{CVDINFR4} = 2 \text{ and } \text{CVDCRHD4} = 2 \\ . & \text{otherwise} \end{cases}$$
   - Including `CVDINFR4` or `CVDCRHD4` in any predictive model for `_MICHD` constitutes complete target leakage. Both variables are strictly excluded from all feature pipelines.
2. **Post-Diagnostic & Management Leakage:**
   - Questions assessing cardiac rehabilitation or specialized post-event procedures do not exist in the 2024 core instrument.
   - For backup candidate `DIABETE4`, post-diagnostic management variables (`INSULIN1`, `CHKHEMO3`, `DIABEYE1`) were audited and designated for exclusion in Milestone 5.
3. **Cross-Sectional Nature & Scope:**
   - The predictive task is strictly defined as classification of reported disease history from survey indicators. It does NOT forecast future incidence.

---

## 6. Recomputed Feasibility Metrics & Discrepancies

All 10 candidate outcomes were recomputed directly from `data/interim/brfss_2024_selected.csv.gz` ($N = 457,670$). All row checksums strictly equal 457,670:

| Variable | Condition Name | Valid Pos | Valid Neg | Total Valid | DK | Ref | Other Special | Missing | Checksum | Unw. Prev | Wtd. Prev | Imbalance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `_MICHD` | Myocardial Infarction / CHD | 42,338 | 410,126 | 452,464 | 0 | 0 | 0 | 5,206 | **457,670** | 9.36% | **6.80%** | 1 : 9.69 |
| `CVDINFR4` | Heart Attack (History) | 26,672 | 427,883 | 454,555 | 2,857 | 256 | 0 | 2 | **457,670** | 5.87% | **4.38%** | 1 : 16.04 |
| `CVDCRHD4` | Angina / Coronary Heart Disease | 28,511 | 424,581 | 453,092 | 4,297 | 278 | 0 | 3 | **457,670** | 6.29% | **4.31%** | 1 : 14.89 |
| `CVDSTRK3` | Stroke (Ever Diagnosed) | 20,661 | 435,557 | 456,218 | 1,244 | 205 | 0 | 3 | **457,670** | 4.53% | **3.53%** | 1 : 21.08 |
| `DIABETE4` | Diabetes (Ever Diagnosed) | 65,809 | 387,432 | 453,241 | 798 | 232 | 3,395 | 4 | **457,670** | 14.52% | **12.63%** | 1 : 5.89 |
| `ASTHMA3` | Asthma (Ever Diagnosed) | 72,037 | 383,772 | 455,809 | 1,650 | 208 | 0 | 3 | **457,670** | 15.80% | **15.73%** | 1 : 5.33 |
| `CHCCOPD3` | COPD / Emphysema | 36,921 | 418,575 | 455,496 | 1,921 | 246 | 0 | 7 | **457,670** | 8.11% | **6.30%** | 1 : 11.34 |
| `ADDEPEV3` | Depressive Disorder | 96,125 | 358,881 | 455,006 | 2,066 | 593 | 0 | 5 | **457,670** | 21.13% | **21.04%** | 1 : 3.73 |
| `CHCKDNY2` | Kidney Disease | 23,752 | 431,939 | 455,691 | 1,715 | 258 | 0 | 6 | **457,670** | 5.21% | **4.13%** | 1 : 18.19 |
| `HAVARTH4` | Arthritis (Diagnosed) | 158,319 | 296,778 | 455,097 | 2,273 | 295 | 0 | 5 | **457,670** | 34.79% | **26.55%** | 1 : 1.87 |

---

## 7. Corrections Implemented

1. **`src/analysis/statistical_tests.py`:**
   - Modified `audit_outcome_feasibility` to accept and report `other_special_count` and compute `row_checksum`.
   - Updated `compute_bivariate_odds_ratios` to rename and clearly distinguish unweighted statistics (`unweighted_or`, `unweighted_ci_95_lower/upper`, `unweighted_chi2`, `unweighted_cramers_v`) from survey-weighted point estimates (`weighted_or`).
   - Refactored `fit_survey_weighted_logistic_regression` to normalize weights ($w_i^* = w_i \times N / \sum w_i$) and implement cluster-robust sensitivity modeling on `_PSU`.
2. **`notebooks/02_statistical_analysis.ipynb`:**
   - Updated cells with corrected functions, verified checksums, and added documentation on the resolution of confidence interval shrinkage.
   - Executed notebook from cell 1 to cell 18 via `nbconvert` without errors.
3. **`reports/target_selection_assessment.md`:**
   - Updated feasibility matrix with `other_special_count`.
   - Updated multivariable regression section with normalized and cluster-robust confidence intervals.
   - Refined task definition as cross-sectional surveillance classification of reported disease history.
4. **`docs/PROJECT_DECISIONS.md`:**
   - Documented weight normalization rationale, GLM frequency-weight mechanics, cluster-robust sensitivity findings, and exact codebook response categories.

---

## 8. Remaining Limitations That Remain Unresolved

1. **Full Multi-Stage Taylor Linearization:**
   - Standard `statsmodels` 0.15 does not support simultaneous multi-stage Taylor-series linearization over both strata (`_STSTR`) and primary sampling units (`_PSU`).
   - Per binding user instructions, no external survey packages (e.g. `samplics`) were installed.
   - Results are honestly documented as survey-weighted point estimates with cluster-robust sensitivity checks.
2. **Cross-Sectional Observational Limitations:**
   - Cross-sectional survey data cannot establish temporal sequencing or prospective incidence.
   - Behavioral variables (physical activity, smoking, height/weight) are subject to self-report and recall bias.

---

## 9. Recommendation on Milestone 4 Closure & Target Selection

- **Milestone 4 Status:** **READY TO CLOSE**. All methodological defects (AUD-01 through AUD-05) have been audited, resolved, and verified through successful notebook execution.
- **Recommended Primary Target:** **`_MICHD` (Coronary Heart Disease / Myocardial Infarction)**
  - *Public Health Burden:* Leading cause of death in the US.
  - *Statistical Power:* 42,338 positive cases provide vast power for machine learning.
  - *Evaluation Rigor:* 6.80% weighted prevalence (1:9.7 imbalance) provides a genuine clinical machine learning challenge requiring PR-AUC, Brier score, and threshold optimization.
  - *Predictive Signal:* Robust independent multivariable associations across modifiable lifestyle factors.
  - *Safety:* Strict exclusion of `CVDINFR4` and `CVDCRHD4` prevents data leakage.
- **Recommended Backup Target:** **`DIABETE4` (Diagnosed Diabetes)**
  - 65,809 cases, 12.63% weighted prevalence, 1:5.9 imbalance.
