# Milestone 7 Planning Report: Explainable AI (XAI) Strategy & Environment Audit

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Target Variable:** `_MICHD` (Cross-sectional self-reported history of coronary heart disease or myocardial infarction)
**Planning Date:** October 20, 2026
**Status:** Planning & Audit Phase Only (No model training, code modification, or explanation generation)

---

## 1. Environment & Dependency Audit

A comprehensive dependency audit of the active virtual environment (`.venv\Scripts\python.exe`, Python 3.13.14 AMD64) was executed without modifying, installing, or upgrading any packages:

| Library / Tool | Availability | Installed Version | Notes & XAI Applicability |
| :--- | :---: | :---: | :--- |
| **`shap`** | **Not Installed** | None | SHAP is not present in `.venv`. In accordance with strict constraints, no packages will be installed. |
| **`lime`** | **Not Installed** | None | LIME is not present in `.venv`. |
| **`scikit-learn`** | **Installed** | `1.9.1` | Provides native model inspection engines: `sklearn.inspection.permutation_importance` and `sklearn.inspection.partial_dependence`. |
| **`scipy`** | **Installed** | `1.18.1` | Numerical linear algebra, distance metrics, and statistical hypothesis testing. |
| **`statsmodels`** | **Installed** | `0.15.0` | Statistical and survey regression references. |
| **`numpy`** | **Installed** | `2.5.4` | Matrix computations and vector operations. |
| **`pandas`** | **Installed** | `3.0.6` | Tabular data manipulation and aggregations. |
| **`matplotlib`** | **Installed** | `3.11.2` | Publication-quality static visualization engine. |
| **`seaborn`** | **Installed** | `0.13.2` | High-level statistical visualization library. |
| **`plotly`** | **Installed** | `7.1.0` | Interactive visualization library (useful for dashboard readiness in Milestone 8). |
| **`joblib`** | **Installed** | `1.6.0` | Fast parallelization and model/pipeline artifact persistence. |

---

## 2. Frozen Milestone 6 Model & Pipeline Inspection

The artifacts generated in Milestones 5 and 6 were inspected and verified as strictly frozen:

### A. Champion Model Identity & Architecture
- **Model Key:** `hist_gradient_boosting_unweighted`
- **Class / Estimator:** `sklearn.ensemble.HistGradientBoostingClassifier`
- **Artifact Path:** [`models/improved_hist_gradient_boosting.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib)
- **Hyperparameters:** `learning_rate=0.08`, `max_iter=150` (fitted trees: 124), `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=1.0`, `loss='log_loss'`, `class_weight=None`, `random_state=42`.
- **Native Attribute Check:** `HistGradientBoostingClassifier` in scikit-learn does not expose a tree-based Gini/impurity `feature_importances_` array (as it operates on binned integer histograms). Official scikit-learn design prescribes **Permutation Importance** (`sklearn.inspection.permutation_importance`) as the primary feature importance mechanism.

### B. Input Feature Representation & Mapping Contract
- **Preprocessor Artifact:** [`models/preprocessor.joblib`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/preprocessor.joblib) (Fitted `ColumnTransformer`, frozen from Milestone 5).
- **Dimensionality:** 38 raw predictor survey variables are transformed into exactly **190 numerical features**.
- **Transformation Output Breakdown:**
  - 4 continuous / discrete numeric features (`num___AGE80`, `num__PHYSHLTH_DAYS`, `num__MENTHLTH_DAYS`, `num__BMI_NUMERIC`).
  - 186 dummy indicators (`cat__<VAR>_<VAL>`) representing 164 valid categorical codes and 22 explicit `-1.0` missing category indicators.
- **Mapping Contract:** In Milestone 7, explainability will be evaluated at two distinct levels:
  1. **Transformed Indicator Level (190 features):** Identifies specific high-risk response levels (e.g., `cat__DIABETE4_1.0` vs. `cat__DIABETE4_3.0` vs. `cat__DIABETE4_-1.0`).
  2. **Grouped Original Survey Variable Level (38 variables):** Groups all dummy indicators back to their parent survey construct (e.g., all 14 indicators of `_AGEG5YR`, all indicators of `DIABETE4`, `SMOKDAY2`, `_BMI5`) by permuting variable blocks simultaneously, avoiding the known pitfall of artificially deflating multi-category feature importances.

### C. Locked Operational Threshold
- **Selected Cutoff:** **$\tau = 0.080$** (Maximizing Youden's J index $Sensitivity + Specificity - 1$ on validation data).
- **Lock Verification:** Serialized in [`models/model_selection_summary.json`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json).
- **Holdout Test Benchmark:** Sensitivity $= 83.91\%$, Specificity $= 70.45\%$, PPV $= 22.67\%$, NPV $= 97.69\%$ (unweighted); Sensitivity $= 80.44\%$, Specificity $= 79.46\%$ (population-weighted point estimates).

---

## 3. Recommended XAI Methodology (Using Existing Dependencies)

Because `shap` is not installed and packages cannot be added, we recommend a methodologically sound, model-agnostic XAI architecture built entirely on native `scikit-learn 1.9.1`, `numpy`, and `scipy`:

```
                           Proposed Milestone 7 XAI Architecture
             ┌───────────────────────────────────────────────────────────────┐
             │            Native Scikit-Learn 1.9.1 & Scientific Stack       │
             └───────────────────────────────┬───────────────────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
   [1] Global Explanations                                         [2] Local Explanations
   ┌─────────────────────────────────────────┐                     ┌─────────────────────────────────────────┐
   │ • Grouped Variable Permutation          │                     │ • Individual Perturbation Attributions │
   │   Importance (38 original predictors)   │                     │   (Δp relative to population reference) │
   │ • Transformed Feature Permutation       │                     │ • Local Linear Surrogates (Sparse LIME) │
   │   Importance (Top 25 of 190 indicators) │                     │ • Case Archetypes: TP, FP, FN, TN       │
   │ • Partial Dependence Profiles (PDP)     │                     │ • Risk Factor Directional Waterfalls    │
   │   (Non-linear risk gradients)           │                     │                                         │
   └─────────────────────────────────────────┘                     └─────────────────────────────────────────┘
```

### 1. Global Explainability Strategy
1. **Grouped Original-Variable Permutation Importance:**
   - Permuting individual one-hot columns dilutes the perceived importance of variables with many categories (e.g., age groups, income, education).
   - We will implement grouped permutation: for each of the 38 original survey variables, all associated transformed columns are simultaneously shuffled.
   - Evaluated on validation and holdout test sets using PR-AUC (Average Precision) drops and ROC-AUC drops, with $n\_repeats=10$ to establish confidence intervals and stability.
2. **Indicator-Level Permutation Importance:**
   - Standard column-wise permutation on `X_test_trans` to isolate which specific category levels drive maximum predictive divergence (e.g., age $\ge 80$, diagnosed diabetes, kidney disease, physical inactivity).
3. **Partial Dependence & Risk Gradient Profiles:**
   - Using native `sklearn.inspection.partial_dependence` on key continuous risk factors (`_AGE80`, `PHYSHLTH_DAYS`, `BMI_NUMERIC`) and top ordinal predictors.
   - Shows non-linear risk transitions (e.g., threshold effects where CAD/MI history reporting sharply rises past age 60 or $>14$ physically unhealthy days).

### 2. Local Explainability Strategy (Selected Test Respondents)
1. **Individual Perturbation Attribution (Marginal Risk Shifts):**
   - For an individual respondent $x$, evaluate the change in predicted probability $\Delta p_j = f(x) - f(x_{\setminus j})$, where $x_{\setminus j}$ replaces predictor $j$ (or its categorical dummy block) with a neutral population reference (e.g., median/mode or baseline absence category).
   - Produces clear, interpretable directional attribution: $+12.4\%$ risk contributed by age group $75-79$, $+8.2\%$ contributed by diabetes, $-3.1\%$ mitigated by high physical activity.
2. **Local Sparse Linear Surrogate (Pure Scikit-Learn LIME Equivalent):**
   - Generate perturbed samples in the local Gaussian/multivariate neighborhood of respondent $x$, weighted by exponential kernel distance $D(x, x')$.
   - Fit an L1-penalized or ridge linear regression on local probabilities to extract local linear weights without external dependencies.
3. **Archetypal Clinical Audit Profiles:**
   - Select 4 transparent case studies from the held-out test partition:
     - **Case 1: True Positive (High Confidence)** — Correctly flagged individual with reported CAD/MI ($\hat{p} = 0.42 > 0.080$).
     - **Case 2: False Positive (Screening Alert)** — Individual without reported CAD/MI flagged above threshold ($\hat{p} = 0.16 > 0.080$) due to high multimorbidity (elderly, diabetes, hypertension).
     - **Case 3: False Negative (Missed Case)** — Individual with reported CAD/MI predicted below threshold ($\hat{p} = 0.05 < 0.080$) due to young age and low traditional risk factor burden.
     - **Case 4: True Negative (Low Risk Baseline)** — Healthy individual predicted far below threshold ($\hat{p} = 0.01 < 0.080$).

---

## 4. Proposed Files and Outputs for Milestone 7

### Source Modules (`src/xai/`)
1. [`src/xai/xai_config.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/xai_config.py):
   - Configuration of permutation repetitions ($n=10$), random seed ($42$), top $K$ features ($K=20$), local neighborhood sample size ($N=1,000$), and test archetype criteria.
2. [`src/xai/global_importance.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py):
   - Grouped variable permutation importance engine.
   - Transformed indicator permutation importance engine.
   - Partial dependence calculation routines.
3. [`src/xai/local_explanations.py`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/local_explanations.py):
   - Local perturbation attribution calculator.
   - Local linear surrogate engine for individual risk decomposition.
   - Case archetype selector and waterfall formatting routines.

### Tables (`results/tables/` or `results/explanations/`)
1. `results/tables/xai_global_variable_importance.csv`:
   - 38 original survey variables ranked by mean PR-AUC and ROC-AUC degradation with standard deviations.
2. `results/tables/xai_transformed_feature_importance.csv`:
   - Top 30 transformed one-hot/numeric features ranked by importance.
3. `results/tables/xai_local_case_explanations.csv`:
   - Itemized local factor attributions for the selected test archetypes.

### Figures (`results/figures/`)
1. `results/figures/fig16_global_variable_importance.png`:
   - Horizontal bar plot with error bars showing the top 20 original survey predictors.
2. `results/figures/fig17_indicator_feature_importance.png`:
   - Permutation importance of specific one-hot category levels.
3. `results/figures/fig18_partial_dependence_profiles.png`:
   - Multi-panel PDP showing non-linear marginal risk curves for continuous/discrete variables (`_AGE80`, `PHYSHLTH_DAYS`, `BMI_NUMERIC`).
4. `results/figures/fig19_local_case_attributions.png`:
   - Multi-panel directional attribution plots for True Positive, False Positive, False Negative, and True Negative test respondents.

### Interactive Notebook & Audit Report
1. [`notebooks/05_explainable_ai.ipynb`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/notebooks/05_explainable_ai.ipynb):
   - Dedicated, self-contained, reproducible notebook documenting global and local XAI workflows.
2. [`reports/milestone7_xai_audit.md`](file:///c:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_xai_audit.md):
   - Comprehensive methodological audit detailing feature mapping, collinearity effects, explanation stability, and public health interpretation.

---

## 5. Methodological Risks and Scientific Safeguards

| Methodological Risk | Technical Mechanism | Enforced Scientific Safeguard |
| :--- | :--- | :--- |
| **Causal Misinterpretation** | End users assume top features are causal levers for preventing CAD/MI. | **Explicit Framing:** All documentation and figures will explicitly state that importance reflects associative predictive value in cross-sectional surveillance, not causal etiology or clinical intervention targets. |
| **Collinearity Distortion** | Highly correlated predictors (e.g., `_AGE80` and `_AGEG5YR`; `GENHLTH` and `PHYSHLTH_DAYS`) split importance. | Grouped variable permutation will be evaluated; collinearity clusters will be explicitly highlighted and cross-referenced with Milestone 4 correlation matrices. |
| **Multi-Category Dilution** | One-hot dummy encoding artificially divides a single categorical construct across 14 dummy columns. | Both grouped (38 original variables) and individual (190 indicator columns) permutation importance will be reported side-by-side. |
| **False Claims of Clinical Utility** | Local explanations interpreted as diagnostic explanations. | Explicit disclosure stating that explanations reflect statistical perturbations of self-reported survey responses, with no clinical validity or diagnostic certainty established. |
| **Test Set snooping / Leakage** | Using test explanations to alter models or tune thresholds. | Test split ($N=67,870$) will be used strictly for post-hoc descriptive explanation of the frozen champion model. No hyperparameters, thresholds, or features will be modified. |

---

## 6. Implementation Sequence for Next Phase (Milestone 7 Execution)

When approved to proceed, Milestone 7 will execute in the following structured sequence:

1. **Stage 1 (Configuration & Architecture):** Implement `src/xai/xai_config.py` defining variables, grouping dictionaries, and test sample parameters.
2. **Stage 2 (Global Feature Importance Engine):** Implement and run `src/xai/global_importance.py` to calculate grouped and indicator permutation importances; save tables and Figures 16, 17, and 18.
3. **Stage 3 (Local Explanation Engine):** Implement and run `src/xai/local_explanations.py` to identify test archetypes, compute directional perturbations and local surrogates; save explanation tables and Figure 19.
4. **Stage 4 (Notebook Construction & Verification):** Assemble and execute `notebooks/05_explainable_ai.ipynb` end-to-end via Jupyter `nbconvert` in-place.
5. **Stage 5 (Audit Reporting & Decision Logging):** Author `reports/milestone7_xai_audit.md` and update `docs/PROJECT_DECISIONS.md` and `README.md`.
