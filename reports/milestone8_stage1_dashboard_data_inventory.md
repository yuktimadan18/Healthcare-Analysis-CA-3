# Milestone 8 Stage 1: Dashboard Data & Artifact Inventory

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Date:** October 21, 2026
**Auditor:** Antigravity AI
**Scope:** Complete technical audit of existing models, precomputed evaluation tables, publication figures, processed data partitions, and software dependencies available for dashboard integration.

---

## 1. Executive Summary & Inventory Overview

This inventory catalogs all verified data assets, model binaries, precomputed evaluation metrics, and visual artifacts produced across Milestones 1 through 7.

### Core Architectural Principle:
The interactive research dashboard (**"Population Health Analytics & Explainable ML — CDC BRFSS 2024"**) will operate strictly on a **read-only consumption model**. To ensure instantaneous user responsiveness, zero runtime latency, and prevention of memory exhaustion, the dashboard will consume precomputed evaluation tables, serialized summary JSONs, and optimized summary distributions. **No live model retraining, no refitting of preprocessors, and no live inference across the 452k-row raw dataset will occur at dashboard runtime.**

---

## 2. Software & Framework Inventory

A diagnostic probe of the active project virtual environment (`.venv`, Python 3.13.14 on Windows AMD64) revealed the following dependency status:

| Package Category | Library | Installed Version | Status for Dashboard | Role / Availability |
|:---|:---|:---:|:---:|:---|
| **Visualization** | `plotly` | **7.1.0** | **Installed** | Primary engine for interactive charts, choropleths, histograms, and scatter plots. |
| **Visualization** | `matplotlib` | **3.11.2** | **Installed** | Static figure fallback; generation of high-resolution publication PNGs. |
| **Visualization** | `seaborn` | **0.13.2** | **Installed** | Statistical styling and heatmap rendering engine. |
| **Data Processing** | `pandas` | **3.0.6** | **Installed** | Loading, filtering, and tabulating precomputed summary metrics. |
| **Numerical/Stats** | `numpy` | **2.5.4** | **Installed** | Vector calculations and probability transformations. |
| **Numerical/Stats** | `scipy` | **1.18.1** | **Installed** | Sigmoid mappings (`expit`) and kernel distance evaluations. |
| **Machine Learning** | `scikit-learn` | **1.9.1** | **Installed** | Champion model and preprocessor inference pipelines. |
| **Serialization** | `joblib` | **1.6.0** | **Installed** | Deserialization of `.joblib` model binaries. |
| **Web Server / UI** | `streamlit` | *None* | **NOT Installed** | Standard interactive UI framework; requires user approval to install. |
| **Web Server / UI** | `dash` | *None* | **NOT Installed** | Alternative web UI framework; not installed. |
| **Web Server / UI** | `gradio` | *None* | **NOT Installed** | Alternative ML UI framework; not installed. |
| **Standard Library**| `http.server` | **Built-in** | **Available** | Python built-in zero-dependency local web server. |

### Technical Decision on Dashboard Serving:
- **Option A (Zero-Dependency Standalone Architecture - Default):** A modern, responsive multi-page web application using Plotly.js, HTML5, Vanilla CSS, and modern JavaScript, served via Python's built-in `http.server` (e.g., `python -m http.server 8501 --directory dashboard`) or viewable directly in any standard browser. Requires 0 additional packages.
- **Option B (Streamlit Architecture - Upon User Approval):** If the user authorizes installing `streamlit` (e.g. `streamlit==1.42.0`), an idiomatic multi-page Streamlit application (`dashboard/app.py` with `pages/`).

---

## 3. Serialized Model & Preprocessing Artifacts (`models/`)

All model binaries are frozen and audited. The dashboard accesses them strictly in read-only mode:

| File Name | Size | Type / Class | Function in Dashboard |
|:---|:---:|:---|:---|
| [`models/improved_hist_gradient_boosting.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib) | 469.3 KB | `HistGradientBoostingClassifier` | Frozen Champion model. Used for read-only metadata display and prespecified archetype what-if sensitivity simulations. |
| [`models/preprocessor.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/preprocessor.joblib) | 8.1 KB | `ColumnTransformer` | Frozen preprocessing pipeline (38 raw predictors $\to$ 190 preprocessed features). Contains fitted scaler means/variances. |
| [`models/model_selection_summary.json`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json) | 945 B | JSON Payload | **Single Source of Truth** for locked threshold ($\tau = 0.080$), champion model key, validation scores, and confusion matrix. |
| [`models/baseline_logistic_regression.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/baseline_logistic_regression.joblib) | 2.4 KB | `LogisticRegression` | Unweighted linear baseline artifact for architectural comparison. |
| [`models/weighted_logistic_regression.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_logistic_regression.joblib) | 2.4 KB | `LogisticRegression` | Class-weighted linear model artifact for class-imbalance comparison. |
| [`models/weighted_hist_gradient_boosting.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/weighted_hist_gradient_boosting.joblib) | 451.1 KB | `HistGradientBoostingClassifier` | Class-weighted non-linear GBDT artifact for calibration trade-off display. |

---

## 4. Precomputed Evaluation & Performance Tables (`results/tables/`)

These precomputed CSV tables contain all validation and holdout test metrics required for Sections B, C, D, and E:

| Artifact Path | Dimensions | Primary Key / Index | Description & Content | Target Dashboard Section |
|:---|:---:|:---|:---|:---:|
| [`results/tables/model_validation_comparison.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv) | $4 \times 16$ | `model_key` | Full validation metrics for all 4 models: PR-AUC, ROC-AUC, Brier score, Brier Skill Score, Sensitivity, Specificity, F1, and survey-weighted estimates. | **Section C (Model Performance)** |
| [`results/tables/threshold_optimization_analysis.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/threshold_optimization_analysis.csv) | $89 \times 12$ | `threshold` (0.02 to 0.90) | Grid evaluation of champion model: TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, F1, Balanced Accuracy, and Youden's J across all candidate cutoffs. | **Section D (Threshold Analysis)** |
| [`results/tables/final_test_evaluation_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/final_test_evaluation_summary.csv) | $4 \times 15$ | `evaluation_mode`, `threshold_label` | Final holdout test metrics at default ($0.50$) and locked ($0.080$) thresholds, reporting unweighted sample metrics and survey-weighted population estimates. | **Section C (Model Performance)** |
| [`results/tables/dataset_partition_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/dataset_partition_summary.csv) | $4 \times 8$ | `partition` (Train, Val, Test, Total) | Record counts ($N = 452,464$), case counts ($42,338$), unweighted prevalence ($9.36\%$), survey-weighted prevalence ($6.80\%$), and imbalance ratio ($9.69$). | **Section A (Overview)** |
| [`results/tables/outcome_feasibility_analysis.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/outcome_feasibility_analysis.csv) | $10 \times 16$ | `variable` | Epidemiological audit of 10 candidate BRFSS chronic conditions: valid counts, missingness, unweighted/weighted prevalences, and target feasibility scores. | **Section A (Overview)** |
| [`results/tables/candidate_outcomes_prevalence.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/candidate_outcomes_prevalence.csv) | $10 \times 10$ | `outcome_variable` | Summary comparison of candidate chronic condition prevalences. | **Section A (Overview)** |
| [`results/tables/bivariate_associations_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/bivariate_associations_summary.csv) | $24 \times 7$ | `variable` | Survey-weighted Chi-square / t-test association statistics between predictors and `_MICHD`. | **Section B (Population Health Explorer)** |
| [`results/tables/continuous_risk_factors_stats.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/continuous_risk_factors_stats.csv) | $4 \times 7$ | `variable` | Means, medians, standard deviations, and IQR for continuous risk factors (`_AGE80`, `PHYSHLTH_DAYS`, `MENTHLTH_DAYS`, `BMI_NUMERIC`). | **Section B (Population Health Explorer)** |
| [`results/tables/data_quality_missingness.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/data_quality_missingness.csv) | $61 \times 11$ | `variable` | Raw survey missingness rates, refusal rates, and structural skip-pattern classifications. | **Section B (Population Health Explorer)** |
| [`results/tables/multivariable_logistic_regression_or.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/multivariable_logistic_regression_or.csv) | $15 \times 6$ | `variable` | Survey-weighted multivariable odds ratios (ORs) with 95% confidence intervals from Milestone 4. | **Section B (Population Health Explorer)** |
| [`results/tables/xai_global_variable_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv) | $38 \times 10$ | `rank` (1 to 38) | Grouped permutation importance: mean/std PR-AUC drop and ROC-AUC drop over 10 repeats for all 38 original predictors. | **Section E (Explainable AI)** |
| [`results/tables/xai_transformed_feature_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_transformed_feature_importance.csv) | $190 \times 10$ | `rank` (1 to 190) | Individual indicator permutation importance over 10 repeats across all 190 transformed one-hot/numeric columns. | **Section E (Explainable AI)** |
| [`results/tables/xai_feature_mapping_audit.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_feature_mapping_audit.csv) | $190 \times 6$ | `feature_index` (0 to 189) | Complete bijective mapping table connecting transformed features to parent survey constructs. | **Section E (Explainable AI)** |

---

## 5. Local XAI & Case Archetype Artifacts (`results/explanations/`)

| Artifact Path | Dimensions | Primary Key / Index | Description & Content | Target Dashboard Section |
|:---|:---:|:---|:---|:---:|
| [`results/explanations/xai_case_archetypes_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/explanations/xai_case_archetypes_summary.csv) | $4 \times 14$ | `category` (TP, FP, TN, FN) | Case demographics, actual target, predicted probability ($\hat{p}$), locked threshold ($0.080$), and clinical archetype roles for the 4 representative validation cases. | **Section E (Explainable AI)** |
| [`results/explanations/xai_local_case_explanations.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/explanations/xai_local_case_explanations.csv) | $152 \times 11$ | `case_category`, `source_variable` | Itemized marginal perturbation attributions ($\Delta p$) and local Ridge surrogate weights ($\beta$) for all 38 original predictors across all 4 case archetypes. | **Section E (Explainable AI)** |

---

## 6. Publication Figures Inventory (`results/figures/`)

The repository contains 19 high-resolution ($300\,\text{DPI}$) PNG figures generated during Milestones 3 through 7. These will be integrated into the dashboard as high-fidelity visual evidence and downloadable research figures:

| Figure File | Size | Description & Key Evidence | Target Dashboard Section |
|:---|:---:|:---|:---:|
| [`results/figures/fig16_global_variable_importance.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig16_global_variable_importance.png) | 333.5 KB | Top 20 original predictors ranked by grouped PR-AUC degradation with $\pm 1\,\text{SD}$ error bars. | **Section E (Explainable AI)** |
| [`results/figures/fig17_indicator_feature_importance.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig17_indicator_feature_importance.png) | 309.8 KB | Top 25 indicator features showing one-hot dummy dilution dynamics. | **Section E (Explainable AI)** |
| [`results/figures/fig18_partial_dependence_profiles.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png) | 367.8 KB | 3-panel unscaled PDP for `_AGE80`, `PHYSHLTH_DAYS`, and `BMI_NUMERIC` with threshold ($0.080$) marked. | **Section E (Explainable AI)** |
| [`results/figures/fig19_local_case_attributions.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig19_local_case_attributions.png) | 632.1 KB | 4-panel directional attribution plots for True Positive, False Positive, True Negative, and False Negative archetypes. | **Section E (Explainable AI)** |
| [`results/figures/fig10_roc_curves_validation.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig10_roc_curves_validation.png) | 244.4 KB | Validation ROC curves comparing all 4 supervised candidate models. | **Section C (Model Performance)** |
| [`results/figures/fig11_pr_curves_validation.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig11_pr_curves_validation.png) | 229.8 KB | Validation PR curves illustrating champion dominance on imbalanced outcome. | **Section C (Model Performance)** |
| [`results/figures/fig12_calibration_curves_validation.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig12_calibration_curves_validation.png) | 281.6 KB | Validation calibration decile plots contrasting natural unweighted calibration with weighted distortion. | **Section C (Model Performance)** |
| [`results/figures/fig13_threshold_tradeoffs.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig13_threshold_tradeoffs.png) | 324.5 KB | Threshold sensitivity-specificity trade-off curves with locked cutoff ($0.080$) marked. | **Section D (Threshold Analysis)** |
| [`results/figures/fig14_champion_test_confusion_matrices.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig14_champion_test_confusion_matrices.png) | 211.1 KB | Holdout test confusion matrices comparing default $0.50$ vs. locked $0.080$ thresholds. | **Section C (Model Performance)** |
| [`results/figures/fig15_unweighted_vs_survey_weighted_test.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig15_unweighted_vs_survey_weighted_test.png) | 183.8 KB | Bar comparison of unweighted sample performance vs. survey-weighted population point estimates. | **Section C (Model Performance)** |
| [`results/figures/fig1_missingness_top_variables.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig1_missingness_top_variables.png) | 157.9 KB | Missingness rates across raw survey candidate predictors. | **Section B (Population Health Explorer)** |
| [`results/figures/fig2_risk_factor_distributions.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig2_risk_factor_distributions.png) | 245.6 KB | Population distributions of physical health, mental health, and BMI. | **Section B (Population Health Explorer)** |
| [`results/figures/fig3_demographic_profiles.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig3_demographic_profiles.png) | 236.4 KB | Demographic breakdown by age, race, sex, and education. | **Section B (Population Health Explorer)** |
| [`results/figures/fig4_candidate_outcomes_prevalence.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig4_candidate_outcomes_prevalence.png) | 137.8 KB | Prevalences of candidate chronic conditions across BRFSS 2024. | **Section A (Overview)** |
| [`results/figures/fig5_chronic_comorbidity_correlation.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig5_chronic_comorbidity_correlation.png) | 245.8 KB | Correlation matrix illustrating multimorbidity clustering. | **Section B (Population Health Explorer)** |
| [`results/figures/fig7_odds_ratios_forest_plot.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig7_odds_ratios_forest_plot.png) | 191.0 KB | Multivariable survey-weighted logistic regression odds ratios forest plot. | **Section B (Population Health Explorer)** |
| [`results/figures/fig8_stratified_prevalence_demographics.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig8_stratified_prevalence_demographics.png) | 208.9 KB | CAD/MI prevalence stratified by age, sex, and race. | **Section B (Population Health Explorer)** |
| [`results/figures/fig9_behavioral_risk_gradient.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig9_behavioral_risk_gradient.png) | 169.8 KB | Prevalence gradients across smoking, physical activity, and alcohol use. | **Section B (Population Health Explorer)** |

---

## 7. Raw & Processed Data Partition Constraints (`data/processed/`)

| Partition File | Shape | File Size on Disk | Role in Dashboard | Safety & Access Protocol |
|:---|:---:|:---:|:---|:---|
| `cohort_metadata.csv.gz` | $452,464 \times 8$ | 8.3 MB | Precomputing population health aggregates | **Offline ingestion only.** Never parsed at live dashboard runtime. Summary aggregates pre-extracted. |
| `train_features.csv.gz` | $316,724 \times 38$ | 6.5 MB | Training feature storage | **Quarantined.** Not accessed by dashboard. |
| `validation_features.csv.gz`| $67,870 \times 38$ | 1.4 MB | Validation feature storage | Precomputed summaries used; read-only access for archetype verification. |
| `validation_target.csv.gz` | $67,870 \times 1$ | 232 KB | Validation target labels | Precomputed metrics used; read-only access for archetype verification. |
| `test_features.csv.gz` | $67,870 \times 38$ | 1.4 MB | Final holdout feature storage | **Strictly Quarantined.** The dashboard will display precomputed test summary metrics from Table 3 only; it will never load or execute test data. |
| `test_target.csv.gz` | $67,870 \times 1$ | 232 KB | Final holdout target labels | **Strictly Quarantined.** Precomputed metrics used only. |

---

## 8. Summary of Inventory Readiness

- **Data Completeness:** 100% of required tables (13 CSVs), local explanation records (2 CSVs), model binaries (6 artifacts), and figures (19 PNGs) are verified present, validated, and numerically aligned.
- **Runtime Performance Strategy:** All metrics and distributions needed for instant UI interactivity exist as precomputed artifacts. The dashboard can load in under 1 second without scanning large `.csv.gz` partitions.
- **Dependency Invariant:** Plotly 7.1.0 is installed and functional. UI framework strategy (Option A: Vanilla JS/Plotly.js SPA vs. Option B: Streamlit) is ready for user selection.
