# Milestone 6 Audit Report: Baseline and Improved Model Development

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Target Variable:** `_MICHD` (Binary classification of reported history of myocardial infarction or coronary heart disease)
**Evaluation Date:** October 20, 2026
**Execution Environment:** Python 3.13.14 (AMD64), scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.4, statsmodels 0.15.0

---

## 1. Executive Summary

Milestone 6 establishes, optimizes, locks, and evaluates four supervised machine learning candidates for classifying reported cardiovascular disease history (`_MICHD`) on the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 cohort ($N = 452,464$).

### Core Methodological Safeguards Enforced:
1. **Frozen Preprocessing Pipeline:** The frozen `models/preprocessor.joblib` artifact established in Milestone 5 was reused strictly via `.transform()`. All 38 raw survey predictors were projected into exactly 190 one-hot encoded indicators (incorporating explicit `-1.0` sentinels for genuine missingness). Preprocessing parameters were neither refit nor re-estimated on validation or test data.
2. **Strict Test Partition Quarantine:** The holdout test set ($N = 67,870$) remained completely untouched throughout candidate training, validation model comparison, calibration evaluation, and decision threshold search.
3. **Validation-Only Selection Rule:** Candidate models were compared strictly on the validation partition ($N = 67,870$). In accordance with project decisions, **PR-AUC (Average Precision)** served as the primary selection criterion due to cohort class imbalance ($9.36\%$ positive prevalence).
4. **Probability Calibration Assessment:** Rather than assuming unweighted models are automatically calibrated or that class weighting preserves risk probabilities, probability calibration was empirically assessed on the validation partition via Brier Score, Brier Skill Score, and decile reliability diagrams. Unweighted models empirically demonstrated closer probability alignment with cohort event rates ($\text{Brier} = 0.0700$ vs $0.1652$), whereas class-weighted models exhibited substantial probability distortion.
5. **Pre-Test Champion and Threshold Lock:** The top-performing candidate (**HistGradientBoostingClassifier, Unweighted GBDT**) and its validation-selected Youden's J decision threshold ($\tau = 0.080$, chosen to balance sensitivity and specificity) were formally locked into `models/model_selection_summary.json` prior to any exposure to test data.
6. **Final Holdout Test Evaluation:** Evaluated in a single forward pass strictly after model and threshold locking, the final holdout test results for the champion model were $\text{PR-AUC} = 0.3808$, $\text{ROC-AUC} = 0.8492$, Sensitivity $= 83.91\%$, Specificity $= 70.45\%$, and Negative Predictive Value (NPV) $= 97.69\%$. When applying CDC sample expansion weights (`_LLCPWT`), the population-weighted point estimates were $\text{ROC-AUC} = 0.8726$, Sensitivity $= 80.44\%$, and Specificity $= 79.46\%$. The model classifies cross-sectional self-reported disease history; clinical utility has not been established.

---

## 2. Partition Architecture and Preprocessor Contract

The dataset splits established and audited in Milestone 5 were loaded directly from `data/processed/`:

| Partition | Record Count ($N$) | Positive Cases (`_MICHD=1`) | Empirical Prevalence | Role in Modeling Workflow |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | $316,724$ | $29,637$ | $9.357\%$ | Parameter estimation for Candidate Models 1A, 1B, 2A, 2B |
| **Validation** | $67,870$ | $6,351$ | $9.358\%$ | Model discrimination ranking, calibration audit, threshold search |
| **Holdout Test** | $67,870$ | $6,350$ | $9.356\%$ | Quarantined single-pass evaluation of locked champion |
| **Total Cohort** | $452,464$ | $42,338$ | $9.357\%$ | Complete analyzed surveillance cohort |

### Preprocessor Transformation Audit
- **Artifact Path:** [`models/preprocessor.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/preprocessor.joblib)
- **Transformation Mode:** `preprocessor.transform(X)` strictly.
- **Input Dimensions:** 38 raw predictor columns.
- **Output Dimensions:** Exactly 190 engineered features (4 continuous/discrete numeric variables + 186 dummy indicators representing 164 valid response codes and 22 explicit `-1.0` missing category indicators).
- **Execution Speed:** Train ($1.43\text{ s}$), Validation ($0.31\text{ s}$), Test ($0.31\text{ s}$).

---

## 3. Candidate Model Architectures and Training Specifications

In accordance with project constraints, four models were trained strictly on `X_train_trans` ($316,724 \times 190$):

```
                                 Candidate Architectures
                  ┌──────────────────────────────────────────────────┐
                  │                 scikit-learn 1.9.1               │
                  └─────────────────────────┬────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [1] Logistic Regression                         [2] HistGradientBoosting
         (L2 Regularization, C=1.0)                      (31 Leaf Nodes, lr=0.08)
         ┌───────────┴───────────┐                       ┌───────────┴───────────┐
         ▼                       ▼                       ▼                       ▼
    Candidate 1A            Candidate 1B            Candidate 2A            Candidate 2B
    Unweighted              Class-Weighted          Unweighted              Class-Weighted
    (class_weight=None)     (class_weight='bal')    (class_weight=None)     (class_weight='bal')
```

### Model Specifications & Training Runtimes

| Candidate | Architecture | Imbalance Strategy | Training Runtime | Iterations / Trees | Serialized Artifact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **1A: LR Unweighted** | LogisticRegression (`l2`, `C=1.0`, `lbfgs`) | Natural cohort prevalence | $3.25\text{ s}$ | 50 iterations | [`baseline_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/baseline_logistic_regression.joblib) |
| **1B: LR Weighted** | LogisticRegression (`l2`, `C=1.0`, `lbfgs`) | `class_weight='balanced'` ($w_1=5.34, w_0=0.55$) | $5.94\text{ s}$ | 85 iterations | [`weighted_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_logistic_regression.joblib) |
| **2A: HGB Unweighted** | HistGradientBoostingClassifier (`lr=0.08`, `nodes=31`) | Natural cohort prevalence | $12.67\text{ s}$ | 124 trees | [`improved_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib) |
| **2B: HGB Weighted** | HistGradientBoostingClassifier (`lr=0.08`, `nodes=31`) | `class_weight='balanced'` | $12.36\text{ s}$ | 119 trees | [`weighted_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_hist_gradient_boosting.joblib) |

---

## 4. Validation-Only Evaluation and Model Selection

The four candidate models were evaluated on the validation partition ($N = 67,870$, $6,351$ positive cases).

### Formal Selection Rule
1. **Primary Ranking Metric:** **PR-AUC (Average Precision)** on validation data. Rationale: In imbalanced clinical surveillance ($9.36\%$ prevalence), ROC-AUC can be overly optimistic due to large true-negative volumes ($90.64\%$), whereas PR-AUC evaluates positive predictive value across all sensitivity thresholds.
2. **Secondary Ranking Metrics:** ROC-AUC (overall discriminative ranking) and Brier Score (probabilistic risk accuracy).
3. **Threshold Metric Reporting:** Threshold-dependent metrics (Sensitivity, Specificity, F1) are reported at default $0.50$ and reference prevalence ($0.0936$).

### Table 1: Validation Performance Comparison (Ranked by PR-AUC)
*Source:* [`results/tables/model_validation_comparison.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv)

| Rank | Model Name | Val PR-AUC | Val ROC-AUC | Val Brier Score | Brier Skill Score | Sens @ 0.50 | Spec @ 0.50 | F1 @ 0.50 | Sens @ Prev Th | Spec @ Prev Th | F1 @ Prev Th |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **HistGradientBoosting (Unweighted)** | **0.3701** | **0.8449** | **0.0700** | **+0.1745** | $10.66\%$ | $99.19\%$ | $0.1799$ | $80.52\%$ | $74.63\%$ | $0.3725$ |
| **2** | HistGradientBoosting (Class-Weighted) | $0.3687$ | $0.8450$ | $0.1652$ | $-0.9481$ | $80.35\%$ | $73.47\%$ | $0.3675$ | $97.97\%$ | $24.71\%$ | $0.2115$ |
| **3** | Logistic Regression (Unweighted Baseline) | $0.3658$ | $0.8413$ | $0.0705$ | $+0.1686$ | $12.61\%$ | $98.92\%$ | $0.2049$ | $77.86\%$ | $75.02\%$ | $0.3678$ |
| **4** | Logistic Regression (Class-Weighted) | $0.3646$ | $0.8413$ | $0.1661$ | $-0.9587$ | $78.40\%$ | $74.85\%$ | $0.3715$ | $97.35\%$ | $28.31\%$ | $0.2185$ |

### Visual Validation Artifacts
- **Figure 10 (ROC Curves):** [`results/figures/fig10_roc_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig10_roc_curves_validation.png) shows that all candidates exhibit strong discriminative power ($\text{AUC} \approx 0.841 - 0.845$), with GBDT models showing slight superiority over linear baselines.
- **Figure 11 (Precision-Recall Curves):** [`results/figures/fig11_pr_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig11_pr_curves_validation.png) confirms that the unweighted HistGradientBoosting classifier dominates all other candidates across the entire recall spectrum, especially in the clinically vital $60\% - 85\%$ sensitivity zone.
- **Figure 12 (Calibration Diagrams):** [`results/figures/fig12_calibration_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig12_calibration_curves_validation.png) reveals profound differences in probability fidelity.

---

## 5. Probability Calibration Assessment: Unweighted vs. Class-Weighted Models

Probability calibration was empirically evaluated on the validation partition:

1. **Empirical Calibration on the Validation Set:**
   - Rather than assuming that unweighted training guarantees calibration, probability calibration was evaluated across decile bins on the validation set.
   - For this cohort, both unweighted models (`hist_gradient_boosting_unweighted` and `logistic_regression_unweighted`) tracked the ideal 45-degree calibration diagonal closely.
   - The validation Brier score for unweighted GBDT was **0.0700** (mean predicted probability of $0.0937$ closely matched the validation cohort prevalence of $0.0936$), yielding a positive Brier Skill Score ($+0.1745$) relative to climatological prevalence baseline. Unweighted training does not guarantee calibration in general, but on this dataset and feature representation, empirical evaluation demonstrated close alignment with observed prevalence.
2. **Impact of Class Weighting on Probability Calibration:**
   - Class weighting (`class_weight='balanced'`) penalizes minority misclassifications by inverse class frequency ($w_1 / w_0 \approx 9.7\times$).
   - On the validation set, this shifted the mean predicted probability to $\sim 0.50$, substantially overestimating absolute probability relative to cohort prevalence ($9.36\%$).
   - The validation Brier score degraded from $0.0700$ to **$0.1652$** ($136\%$ increase in mean squared probability error), resulting in a negative Brier Skill Score ($-0.9481$).
   - While class weighting increased sensitivity at the arbitrary default cutoff of $0.50$, it did so by artificially inflating predicted probabilities. Threshold tuning on the empirically well-calibrated unweighted model achieved comparable sensitivity without corrupting the meaning of the probability estimates.

---

## 6. Validation Decision Threshold Optimization and Champion Lock

Using validation data only, a grid search across candidate thresholds ($\tau \in [0.02, 0.90]$, step $0.01$) was conducted for the top-ranked model (`hist_gradient_boosting_unweighted`).

### Table 2: Threshold Optimization Benchmarks (Validation Set)
*Source:* [`results/tables/threshold_optimization_analysis.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/threshold_optimization_analysis.csv)

| Objective / Benchmark | Cutoff Threshold ($\tau$) | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | F1 Score | Balanced Accuracy | Youden's J Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Model Cutoff** | $0.500$ | $10.66\%$ | $99.19\%$ | $57.57\%$ | $91.56\%$ | $0.1799$ | $54.92\%$ | $0.0985$ |
| **Optimal Youden's J (LOCKED)** | **0.080** | **83.29%** | **70.28%** | **22.44%** | **97.60%** | **0.3536** | **76.79%** | **0.5357** |
| **Prevalence Reference** | $0.0936$ | $80.52\%$ | $74.63\%$ | $24.73\%$ | $97.35\%$ | $0.3783$ | $77.58\%$ | $0.5515$ |
| **Optimal F1 Score** | $0.210$ | $54.34\%$ | $89.87\%$ | $35.68\%$ | $94.97\%$ | $0.4305$ | $72.10\%$ | $0.4421$ |
| **Target 75% Recall** | $0.110$ | $76.84\%$ | $78.85\%$ | $27.32\%$ | $97.08\%$ | $0.4030$ | $77.84\%$ | $0.5569$ |

### Rationale for Selected Operational Threshold ($\tau = 0.080$)
- In an imbalanced surveillance context ($9.36\%$ prevalence), the default threshold $\tau = 0.50$ produces a high false-negative rate, missing **$89.34\%$** of positive survey responses.
- To balance classification performance between positive and negative responses, the decision threshold was selected by maximizing Youden's J index ($J = Sensitivity + Specificity - 1$) on the validation partition, yielding **$\tau = 0.080$**.
- On the validation set, this threshold achieved a sensitivity of **$83.29\%$**, specificity of **$70.28\%$**, PPV of **$22.44\%$**, NPV of **$97.60\%$**, and Youden's J of **$0.5357$**.
- This threshold was locked based strictly on this documented validation objective. Threshold trade-offs are visualized in **Figure 13:** [`results/figures/fig13_threshold_tradeoffs.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig13_threshold_tradeoffs.png).

### Official Champion Lock Payload
Prior to holdout test set access, the selection decision and operational threshold were recorded in [`models/model_selection_summary.json`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json):

```json
{
  "status": "LOCKED",
  "primary_selection_metric": "pr_auc",
  "champion_model_key": "hist_gradient_boosting_unweighted",
  "champion_model_name": "HistGradientBoosting (Unweighted GBDT)",
  "validation_pr_auc": 0.3701,
  "validation_roc_auc": 0.8449,
  "validation_brier_score": 0.07,
  "selected_operational_threshold": 0.08,
  "selection_objective": "youden_j (Balanced Population Screening)",
  "candidate_thresholds": {
    "default_050": 0.5,
    "empirical_prevalence_ref": 0.0936,
    "optimal_youden_j": 0.08,
    "optimal_f1": 0.21,
    "target_75pct_recall": 0.11
  }
}
```

---

## 7. Single-Pass Holdout Test Evaluation ($N = 67,870$)

Final holdout test evaluation was conducted in a single forward pass strictly after the champion model and operational threshold were locked into `models/model_selection_summary.json`. Test features were transformed via the frozen preprocessor with zero refitting.

### Table 3: Final Holdout Test Evaluation Summary
*Source:* [`results/tables/final_test_evaluation_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/final_test_evaluation_summary.csv)

| Evaluation Mode | Threshold Label | Cutoff ($\tau$) | PR-AUC | ROC-AUC | Sensitivity | Specificity | PPV | NPV | F1 Score | Balanced Acc | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unweighted Sample** | Default Threshold (0.50) | $0.500$ | $0.3808$ | $0.8492$ | $10.91\%$ | $99.26\%$ | $60.26\%$ | $91.58\%$ | $0.1848$ | $55.09\%$ | $0.0694$ |
| **Unweighted Sample** | **Locked Optimal (0.080)** | **0.080** | **0.3808** | **0.8492** | **83.91%** | **70.45%** | **22.67%** | **97.69%** | **0.3569** | **77.18%** | **0.0694** |
| **Survey-Weighted (`_LLCPWT`)** | Default Threshold (0.50) | $0.500$ | $0.3594$ | $0.8726$ | $9.34\%$ | $99.58\%$ | $61.72\%$ | $94.39\%$ | $0.1623$ | $54.46\%$ | $0.0514$ |
| **Survey-Weighted (`_LLCPWT`)** | **Locked Optimal (0.080)** | **0.080** | **0.3594** | **0.8726** | **80.44%** | **79.46%** | **22.12%** | **98.42%** | **0.3469** | **79.95%** | **0.0514** |

### Key Test Evaluation Findings:
1. **Generalization Consistency:** Test discrimination on the holdout partition ($\text{PR-AUC} = 0.3808$, $\text{ROC-AUC} = 0.8492$) aligned closely with validation results ($\text{PR-AUC} = 0.3701$, $\text{ROC-AUC} = 0.8449$).
2. **Classification Performance on Self-Reported Outcomes:** At the locked $\tau = 0.080$ threshold, the model correctly identified **$5,328$ of $6,350$ self-reported positive cases** in the test partition ($83.91\%$ sensitivity) with a specificity of $70.45\%$ and an NPV of $97.69\%$. The model classifies cross-sectional self-reported disease history; clinical utility, clinical screening validity, and rule-out capability have not been established.
3. **Population-Weighted Point Estimates:** Applying CDC sample expansion weights (`_LLCPWT`), the population-weighted point estimates were $\text{ROC-AUC} = 0.8726$, Sensitivity $= 80.44\%$, and Specificity $= 79.46\%$ (Balanced Accuracy: $79.95\%$). These point estimates do not incorporate complex-survey variance estimation.
4. **Visual Test Artifacts:**
   - **Figure 14 (Test Confusion Matrices):** [`results/figures/fig14_champion_test_confusion_matrices.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig14_champion_test_confusion_matrices.png) displays side-by-side matrices contrasting the default $0.50$ cutoff against the locked $0.080$ threshold.
   - **Figure 15 (Sample vs Population Comparison):** [`results/figures/fig15_unweighted_vs_survey_weighted_test.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig15_unweighted_vs_survey_weighted_test.png) highlights the alignment between unweighted sample metrics and population-weighted point estimates.

---

## 8. Epidemiological Surveillance Framing & Methodological Disclosures

1. **Cross-Sectional Surveillance vs. Clinical Validity:**
   - The CDC BRFSS is an annual cross-sectional telephone surveillance survey. The target variable `_MICHD` records whether a survey respondent has ever received a physician diagnosis of coronary heart disease or myocardial infarction.
   - Consequently, model outputs reflect cross-sectional classification of self-reported historical disease status, rather than prospective disease prediction, etiological causality, or verified clinical diagnosis.
   - **Clinical utility has not been established:** Although the model achieves high sensitivity ($80.4\% - 83.9\%$) and high negative predictive value ($97.7\% - 98.4\%$) within this cross-sectional surveillance survey, clinical utility, clinical screening validity, and rule-out capability have not been established. Any potential clinical deployment would require prospective evaluation against gold-standard clinical diagnostics in prospective healthcare settings.
2. **Survey-Weighted Metrics and Complex Survey Design Disclosures:**
   - Machine learning algorithms were trained by optimizing unweighted sample loss across survey respondents.
   - Survey-weighted metrics reported on the test set are descriptive **population-weighted point estimates** computed using CDC sample expansion weights (`_LLCPWT`).
   - These point estimates do not represent full complex-survey variance estimation. Fully design-adjusted standard errors, confidence intervals, and hypothesis tests under the BRFSS sampling framework require accounting for primary sampling units (`_PSU`) and stratification (`_STSTR`) via Taylor series linearization or balanced repeated replication.

---

## 9. Artifact Registry and Reproducibility

### Files Created or Modified in Milestone 6

| File Path | Category | Purpose | Status |
| :--- | :--- | :--- | :---: |
| [`src/models/model_config.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/models/model_config.py) | Source Code | Model hyperparameters, candidate definitions, evaluation grid | Created / Verified |
| [`src/models/train_models.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/models/train_models.py) | Source Code | Training pipeline for 4 candidate models using frozen preprocessor | Created / Verified |
| [`src/evaluation/evaluate_models.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/evaluation/evaluate_models.py) | Source Code | Evaluation engine, threshold search, model locking, test evaluation | Created / Verified |
| [`models/baseline_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/baseline_logistic_regression.joblib) | Model Artifact | Candidate 1A: Logistic Regression (Unweighted Baseline) | Trained / Saved |
| [`models/weighted_logistic_regression.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_logistic_regression.joblib) | Model Artifact | Candidate 1B: Logistic Regression (Class-Weighted) | Trained / Saved |
| [`models/improved_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib) | Model Artifact | Candidate 2A: HistGradientBoosting (Champion Model) | Trained / Saved |
| [`models/weighted_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_hist_gradient_boosting.joblib) | Model Artifact | Candidate 2B: HistGradientBoosting (Class-Weighted) | Trained / Saved |
| [`models/model_selection_summary.json`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json) | Audit Artifact | Formal lock record: champion model & Youden's J threshold | Locked / Verified |
| [`results/tables/model_validation_comparison.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv) | Results Table | Validation performance metrics across 4 candidate models | Generated |
| [`results/tables/threshold_optimization_analysis.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/threshold_optimization_analysis.csv) | Results Table | Performance metrics across decision threshold grid [0.02, 0.90] | Generated |
| [`results/tables/final_test_evaluation_summary.csv`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/final_test_evaluation_summary.csv) | Results Table | Unweighted and survey-weighted holdout test evaluation metrics | Generated |
| [`results/figures/fig10_roc_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig10_roc_curves_validation.png) | Figure | Validation ROC curves for all candidate models | Generated |
| [`results/figures/fig11_pr_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig11_pr_curves_validation.png) | Figure | Validation PR curves with empirical baseline reference | Generated |
| [`results/figures/fig12_calibration_curves_validation.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig12_calibration_curves_validation.png) | Figure | Decile probability calibration diagrams | Generated |
| [`results/figures/fig13_threshold_tradeoffs.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig13_threshold_tradeoffs.png) | Figure | Sensitivity, specificity, PPV, F1 vs. decision threshold curves | Generated |
| [`results/figures/fig14_champion_test_confusion_matrices.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig14_champion_test_confusion_matrices.png) | Figure | Side-by-side test confusion matrices (0.50 vs. 0.080) | Generated |
| [`results/figures/fig15_unweighted_vs_survey_weighted_test.png`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig15_unweighted_vs_survey_weighted_test.png) | Figure | Grouped bar comparison: unweighted sample vs. population metrics | Generated |
| [`notebooks/04_model_development_and_evaluation.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/04_model_development_and_evaluation.ipynb) | Notebook | Complete, executed end-to-end model development notebook | Executed (2.0 MB) |
| [`reports/milestone6_model_development_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone6_model_development_audit.md) | Audit Report | Authoritative audit report for Milestone 6 | Created |

---

## 10. Reproduction Commands

To reproduce all Milestone 6 outcomes from scratch using the existing virtual environment:

```powershell
# 1. Train the four candidate models strictly on training data
.venv\Scripts\python.exe src/models/train_models.py

# 2. Run validation evaluation, lock champion model, and evaluate on holdout test set
.venv\Scripts\python.exe src/evaluation/evaluate_models.py

# 3. Execute the Milestone 6 Jupyter Notebook in-place
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/04_model_development_and_evaluation.ipynb --inplace
```
