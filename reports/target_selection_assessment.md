# Evidence-Based Predictive Target Selection Assessment
**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Milestone:** Milestone 4 (Knowledge Discovery & Target Selection Preparation)
**Date:** October 2026
**Status:** Formal Target Recommendation (Pending Final Approval before Milestone 5 Target Lock)

---

## 1. Executive Summary

In strict compliance with **Rule 6** of `PROJECT_RULES.md` (*"Do not select the predictive target until the actual dataset and official codebook have been inspected"*), target selection was withheld through Milestones 1–3. Following comprehensive data quality auditing (Milestone 3) and statistical hypothesis testing with survey-weighted modeling across 457,670 respondents (Milestone 4), this report presents an empirical evaluation of **10 candidate chronic condition outcomes**.

### Recommendations:
1. **Primary Recommended Target:** **Coronary Heart Disease / Myocardial Infarction (`_MICHD`)**
   - *Task Formulation:* Binary classification of reported history of coronary heart disease or myocardial infarction (`_MICHD`) from cross-sectional self-reported survey indicators. (Observational surveillance classification; not prospective risk prediction of future disease onset or individual clinical diagnosis).
   - *Case Burden:* 42,338 positive cases (9.36% unweighted, **6.80% survey-weighted**).
   - *Imbalance Ratio:* 1 : 9.69.
   - *Completeness:* 98.86% nationwide valid responses.
   - *Clinical Relevance:* Cardiovascular disease remains the #1 leading cause of death in the United States.
   - *Predictive Signal:* Strong, statistically robust associations across modifiable behavioral factors (smoking, physical inactivity, obesity) and actionable clinical indicators.
2. **Secondary / Backup Recommended Target:** **Diabetes (`DIABETE4`)**
   - *Task Formulation:* Binary classification of reported history of diagnosed diabetes from survey indicators.
   - *Case Burden:* 65,809 positive cases (14.52% unweighted, **12.63% survey-weighted**).
   - *Imbalance Ratio:* 1 : 5.89.
   - *Completeness:* 99.03% nationwide valid responses (with 3,395 gestational diabetes cases explicitly segregated).
   - *Clinical Relevance:* Highly prevalent cardiometabolic epidemic with marked lifestyle and socioeconomic disparities.

---

## 2. Multi-Criteria Evaluation Matrix of Candidate Outcomes

All metrics derived from the validated interim dataset (`data/interim/brfss_2024_selected.csv.gz`, $N = 457,670$). All candidate rows strictly sum to 457,670 across valid positive, valid negative, don't know, refused, other special, and missing/NaN categories:

| Candidate Variable | Clinical Outcome Name | Disease Category | Valid Positive | Valid Negative | Other Special / Excluded | Missing / DK / Ref | Completeness (%) | Unweighted Prev (%) | Survey-Weighted Prev (%) | Imbalance Ratio | Recommendation Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`_MICHD`** | **Myocardial Infarction / CHD** | **Cardiovascular** | **42,338** | **410,126** | **0** | **5,206** | **98.86%** | **9.36%** | **6.80%** | **1 : 9.7** | **PRIMARY RECOMMENDATION** |
| `CVDINFR4` | Heart Attack (History) | Cardiovascular | 26,672 | 427,883 | 0 | 3,115 | 99.32% | 5.87% | 4.38% | 1 : 16.0 | Subsumed by `_MICHD` |
| `CVDCRHD4` | Angina / Coronary Heart Disease | Cardiovascular | 28,511 | 424,581 | 0 | 4,578 | 99.00% | 6.29% | 4.31% | 1 : 14.9 | Subsumed by `_MICHD` |
| `CVDSTRK3` | Stroke (History) | Cerebrovascular | 20,661 | 435,557 | 0 | 1,452 | 99.68% | 4.53% | 3.53% | 1 : 21.1 | Excluded: Severe Imbalance |
| **`DIABETE4`** | **Diabetes (Ever Diagnosed)** | **Metabolic** | **65,809** | **387,432** | **3,395** | **1,034** | **99.03%** | **14.52%** | **12.63%** | **1 : 5.9** | **BACKUP RECOMMENDATION** |
| `ASTHMA3` | Asthma (Ever Diagnosed) | Respiratory | 72,037 | 383,772 | 0 | 1,861 | 99.59% | 15.80% | 15.73% | 1 : 5.3 | Excluded: Early-Onset / Atopic Noise |
| `CHCCOPD3` | COPD / Emphysema | Respiratory | 36,921 | 418,575 | 0 | 2,174 | 99.52% | 8.11% | 6.30% | 1 : 11.3 | Excluded: Smoking Confounded |
| `ADDEPEV3` | Depressive Disorder | Mental Health | 96,125 | 358,881 | 0 | 2,664 | 99.42% | 21.13% | 21.04% | 1 : 3.7 | Excluded: Subjective Diagnostic Scope |
| `CHCKDNY2` | Kidney Disease | Renal | 23,752 | 431,939 | 0 | 1,979 | 99.57% | 5.21% | 4.13% | 1 : 18.2 | Excluded: Asymptomatic Underdiagnosis |
| `HAVARTH4` | Arthritis (Diagnosed) | Musculoskeletal | 158,319 | 296,778 | 0 | 2,573 | 99.44% | 34.79% | 26.55% | 1 : 1.9 | Excluded: Broad Symptom Umbrella |

---

## 3. In-Depth Comparative Assessment Against Criteria

### Criterion 1: Public Health Relevance & Clinical Impact
- **`_MICHD` (Coronary Heart Disease / Myocardial Infarction):** Cardiovascular disease accounts for 1 in every 5 deaths in the US. An explainable predictive tool identifying individuals at high risk based on routine behavioral and demographic profiles has immense preventative screening utility.
- **`DIABETE4` (Diabetes):** Affects over 38 million Americans; leading cause of adult blindness, lower-limb amputations, and end-stage renal disease.
- *Verdict:* Both `_MICHD` and `DIABETE4` possess maximal public health relevance.

### Criterion 2: Response Completeness & Codebook Integrity
- **`_MICHD`:** Derived directly by CDC SAS algorithms from Core Section questions (`CVDINFR4` and `CVDCRHD4`). Valid response recorded for 452,464 out of 457,670 respondents (98.86% completeness).
- **`DIABETE4`:** Asked directly to all core participants. Only 4,429 respondents (0.97%) refused or answered \"don't know\".
- *Verdict:* Both candidates have negligible missingness (< 1.2%), eliminating risks of sample attrition or structural imputation artifacts.

### Criterion 3: Sample Size & Statistical Power
- **`_MICHD`:** 42,338 positive cases. In a 70/30 train/test split, the test set retains ~12,700 positive cases, ensuring extraordinarily tight confidence intervals on performance metrics (AUROC, PR-AUC, calibration error).
- **`DIABETE4`:** 65,809 positive cases. Unlocks massive statistical power for gradient boosted trees and non-linear interactions.
- *Verdict:* Both candidates vastly exceed statistical power requirements for machine learning pipelines.

### Criterion 4: Class Imbalance & Evaluation Strategy
- **`_MICHD` (1 : 9.7, 6.80% weighted prevalence):**
  - Represents a *realistic, challenging clinical class imbalance*.
  - Forces students and researchers to implement sound evaluation protocols: relying on **Precision-Recall AUC (PR-AUC)**, **Brier Score / Calibration Curves**, and **Cost-Sensitive Threshold Tuning** rather than misleading overall Accuracy or generic ROC-AUC.
- **`DIABETE4` (1 : 5.9, 12.63% weighted prevalence):**
  - Moderate imbalance; less severe class weighting required.
- **`CVDSTRK3` & `CHCKDNY2` (1 : 21.1 and 1 : 18.2):**
  - Imbalance is too severe (< 4% prevalence), leading to severe sample sparsity in sub-cohorts and high susceptibility to false positive inflation.

### Criterion 5: Availability of Modifiable Predictors & Etiological Coherence
- **`_MICHD`:** Demonstrates multi-factorial etiological alignment across distinct domains in the 2024 core:
  - *Demographic:* Age 65+ (Unweighted OR = 4.46; aOR = 4.57, 95% CI normalized [4.45, 4.70], cluster-robust [4.32, 4.84]), Male Sex (Unweighted OR = 1.59; aOR = 1.73, 95% CI normalized [1.68, 1.78], cluster-robust [1.64, 1.83]).
  - *Behavioral:* Current Cigarette Smoking (Unweighted OR = 1.46; aOR = 1.58, 95% CI normalized [1.52, 1.64], cluster-robust [1.47, 1.70]), Physical Inactivity (Unweighted OR = 2.01; aOR = 1.21, 95% CI normalized [1.17, 1.24], cluster-robust [1.14, 1.28]), Obesity / Overweight (Unweighted OR = 1.34; aOR = 1.20, 95% CI normalized [1.17, 1.24], cluster-robust [1.13, 1.28]).
  - *Comorbidity:* Diabetes (Unweighted OR = 3.61; aOR = 1.99, 95% CI normalized [1.92, 2.05], cluster-robust [1.86, 2.12]), Chronic Kidney Disease (Unweighted OR = 4.56; aOR = 2.41, 95% CI normalized [2.31, 2.52], cluster-robust [2.19, 2.66]).
  - *General Health:* Fair or Poor Self-Rated Health (Unweighted OR = 4.10; aOR = 3.10, 95% CI normalized [3.01, 3.19], cluster-robust [2.93, 3.29]).
  - *Multivariable Fit:* In survey-weighted multivariable GLM ($N = 385,120$ complete cases), all behavioral factors remain independently significant ($p < 10^{-30}$), proving that predictive models can learn authentic, non-spurious signal.
- **`DIABETE4`:** Strong predictors available (BMI, age, physical activity, race/ethnicity), but hypertension and cholesterol screening questions were removed from the 2024 even-year core instrument.

### Criterion 6: Target Leakage Prevention & Methodological Safety
Target leakage occurs when a predictor variable carries direct definitional or post-diagnostic dependency on the target:
- **For `_MICHD`:**
  - *Definitional Leakage:* CDC defines `_MICHD = 1` if `CVDINFR4 = 1` (heart attack) OR `CVDCRHD4 = 1` (angina).
  - *MANDATORY RULE:* In Milestone 5 feature pipeline, `CVDINFR4` and `CVDCRHD4` **MUST BE PERMANENTLY EXCLUDED** from the feature set. Failure to exclude them constitutes complete data leakage ($AUC = 1.0$).
  - Once those two variables are excluded, all other predictors (BMI, smoking, exercise, general health, age, sex) are true pre-morbid or independent risk indicators.
- **For `DIABETE4`:**
  - *Management Leakage:* Variables `PREDIAB2` (pre-diabetes), `INSULIN1` (insulin use), `CHKHEMO3` (A1c checks), and `DIABEYE1` (diabetic retinopathy) are post-diagnostic and **MUST BE EXCLUDED**.

### Criterion 7: Feasibility within Student Compute Environment
- Both `_MICHD` and `DIABETE4` allow fast, stable training of baseline logistic regression models and gradient boosted trees (`HistGradientBoostingClassifier` or `LightGBM`) in < 60 seconds on standard CPU hardware without out-of-memory errors.

---

## 4. Why Other Candidates Were Deprioritized

1. **`CVDSTRK3` (Stroke):** 3.53% weighted prevalence (1:21 imbalance). Stroke is an acute vascular event with substantial etiological dependence on clinical biomarkers (atrial fibrillation, carotid stenosis, acute blood pressure) that are unmeasured in self-reported BRFSS surveys.
2. **`CHCKDNY2` (Kidney Disease):** 4.13% weighted prevalence. Chronic kidney disease is notoriously under-diagnosed in early stages; self-reported survey diagnosis captures only advanced/dialysis patients, inducing severe self-reporting bias.
3. **`ADDEPEV3` (Depressive Disorder):** 21.04% weighted prevalence. While statistically balanced, depression diagnosis reflects significant regional, cultural, and healthcare access biases rather than biological/lifestyle etiologies.
4. **`CHCCOPD3` (COPD):** 6.30% weighted prevalence. Heavily dominated by smoking pack-years with limited contribution from other lifestyle factors.
5. **`HAVARTH4` (Arthritis):** 26.55% weighted prevalence. Too non-specific; combines osteoarthritis, rheumatoid arthritis, gout, and fibromyalgia into an unweighted symptom composite.
6. **`ASTHMA3` (Asthma):** 15.73% weighted prevalence. High childhood/atopic onset noise that does not correlate well with adult lifestyle interventions.

---

## 5. Formal Recommendations for Milestone 5

### Primary Recommendation: **`_MICHD`**
- **Task Formulation:** Binary classification of reported history of Coronary Heart Disease or Myocardial Infarction from cross-sectional self-reported survey indicators.
- **Observational Constraint:** This task classifies surveillance survey respondents based on reported disease history; it does not estimate prospective incidence or individual clinical diagnostic risk.
- **Target Definition:**
  - $Y = 1$ if `_MICHD == 1` (Reported CHD or MI: $N = 42,338$)
  - $Y = 0$ if `_MICHD == 2` (Did not report CHD or MI: $N = 410,126$)
  - Exclude unasked/missing records where `_MICHD` is NaN ($N = 5,206$, 1.14%).
- **Mandatory Feature Dropping (Leakage Prevention):** Exclude `CVDINFR4` and `CVDCRHD4` (direct mathematical constituents).
- **Target Evaluation Metrics:** PR-AUC (Average Precision), F1-Score at optimal threshold, Brier Score, and Balanced Accuracy.

### Backup Recommendation: **`DIABETE4`**
- **Task Formulation:** Binary classification of reported history of diagnosed diabetes from survey indicators.
- **Target Definition:**
  - $Y = 1$ if `DIABETE4 == 1` (Yes: $N = 65,809$)
  - $Y = 0$ if `DIABETE4 in [3, 4]` (No / Borderline: $N = 387,432$)
  - Exclude gestational diabetes (`DIABETE4 == 2`, $N = 3,395$) and missing/refused ($N = 1,034$).
- **Mandatory Feature Dropping:** Exclude `PREDIAB2`, `INSULIN1`, `CHKHEMO3`, `DIABEYE1`, `DIABEDU1`.

---

## 6. Next Steps for Milestone 5
1. Secure user alignment on adopting **`_MICHD`** as the primary prediction target.
2. Formulate the machine learning task specification and feature matrix preparation pipeline in `src/features/build_features.py`.
3. Design a rigorous, non-leaking validation strategy (stratified $K$-fold cross-validation accounting for class imbalance).
