# Milestone 8 Stage 3: Interactive Dashboard UI Implementation Audit

**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Project Root:** `C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project`
**Date:** October 21, 2026
**Auditor:** Antigravity AI
**Scope:** Complete implementation and audit of the interactive research dashboard web interface, component architecture, client-side routing, automated testing, and local serving verification for Milestone 8 Stage 3.

---

## 1. Executive Summary

Milestone 8 Stage 3 implements the interactive web dashboard titled:

> **Population Health Analytics & Explainable ML — CDC BRFSS 2024**

Built upon the user-approved **Option A: Zero-Dependency Standalone Plotly.js SPA**, the dashboard operates entirely on standard web technologies (HTML5, Vanilla CSS3, modern modular JavaScript) and consumes the precomputed JSON bundles ($172\,\text{KB}$ total) produced in Stage 2.

### Core Safeguards Maintained:
- **Zero Third-Party Python Package Installation:** No web frameworks (`streamlit`, `dash`, `gradio`, `flask`, `fastapi`) were installed. The virtual environment (`.venv`, Python 3.13.14) remains untouched.
- **Strict Read-Only Architecture:** No model retraining, preprocessor refitting, or partition alteration was performed.
- **Locked Threshold Invariant:** The operational screening cutoff remains permanently fixed at $\tau = 0.080$ (derived from validation Youden's J criterion). Interactive threshold controls function strictly as read-only visual trade-off exploration tools.
- **Public Health Boundary:** Free-form individual patient symptom checkers, clinical prognostic calculators, and raw respondent microdata are strictly excluded.

---

## 2. Inventory of Files Created, Modified, and Preserved

| File Path | Nature of Work | Role & Technical Description | Size / Line Count |
|:---|:---:|:---|:---:|
| [`dashboard/index.html`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/index.html) | **Created** | Single-page application entry point, semantic DOM structure, header/sidebar, and section containers | 6.55 KB (118 lines) |
| [`dashboard/styles.css`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/styles.css) | **Created** | Professional clinical visual language: CSS tokens (Navy `#0b2545`, Teal `#0e8388`, Locked Purple `#8e44ad`), responsive grid, tables, cards, and animations | 14.15 KB (448 lines) |
| [`dashboard/app.js`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/app.js) | **Created** | Application coordinator: parallel preloading, hash routing, lazy rendering, resize management, and offline fallback | 7.84 KB (168 lines) |
| [`dashboard/components/data_loader.js`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/components/data_loader.js) | **Enhanced** | Client fetch layer with multi-path resolution (`./data/`, `../data/`, `./dashboard/data/`) and in-memory cache | 2.52 KB (80 lines) |
| [`dashboard/components/charts.js`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/components/charts.js) | **Enhanced** | Plotly.js chart engine: candidate outcomes, population dual-axis, ROC/PR curves, calibration deciles, threshold curves, and XAI plots | 19.53 KB (430 lines) |
| [`dashboard/components/views.js`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/components/views.js) | **Enhanced** | UI view renderers for Sections A through F, controls binding, dynamic confusion matrix, and case profile auditor | 47.43 KB (650 lines) |
| [`tests/test_dashboard_ui.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/tests/test_dashboard_ui.py) | **Created** | 9 automated unit tests verifying DOM layout, script order, threshold invariance, disclaimers, and symptom-checker exclusion | 4.88 KB (125 lines) |
| [`README.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/README.md) | **Modified** | Updated Milestone Roadmap and documented local dashboard launch commands | 10.92 KB (165 lines) |
| [`dashboard/data/*.json`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/dashboard/data/) | **Preserved** | 5 precomputed JSON bundles generated in Stage 2 | 171.9 KB (5 files) |

---

## 3. Features Implemented by Section

### Section A: Overview & Project Scope
- **Cohort Dimensions Grid:** Displays analytical sample size ($N = 452,464$), raw records ($457,670$), missing exclusions ($5,206$), target CAD/MI prevalence ($9.36\%$ sample, $6.80\%$ survey-weighted US civilian population), and class imbalance ($9.69:1$).
- **Model Champion & Locked Cutoff:** Highlights the champion GBDT architecture (`HistGradientBoostingClassifier`), validation PR-AUC ($0.3701$), ROC-AUC ($0.8449$), and locked threshold ($\tau = 0.080$).
- **Methodology Pipeline Flow:** Visual sequential workflow showing the 6 project phases from raw CDC survey ingestion to explainability auditing.
- **Candidate Outcomes Feasibility Assessment:** Interactive Plotly horizontal bar chart and comprehensive table comparing 10 chronic condition outcomes from `outcome_feasibility_analysis.csv`.
- **Prominent Disclaimers:** Visible callout stating that the model classifies historical disease reporting in cross-sectional surveillance and does not establish clinical diagnostic utility.

### Section B: Population Health Explorer
- **Interactive Variable Selector:** Allows dynamic inspection across 12 demographic and chronic condition constructs (`_SEX`, `_AGEG5YR`, `_RACE`, `_EDUCAG`, `_INCOMG1`, `GENHLTH`, `DIABETE4`, `CVDSTRK3`, `CHCKDNY2`, `_SMOKER3`, `CHECKUP1`, `PERSDOC3`).
- **Weighting Mode Toggle:** Switches instantaneously between Survey-Weighted US Adult Population % and Unweighted Sample Counts.
- **Dual-Axis Visualization:** Plotly chart combining population stratum proportions (bars) and reported CAD/MI prevalence (line) on a secondary axis.
- **Dynamic Strata Table:** Tabular breakdown of category labels, unweighted counts, unweighted percentages, weighted population percentages, and stratum CAD/MI prevalences.
- **Continuous Covariates Table:** Reports distribution statistics (mean, std dev, median, min, max) for `_AGE80`, `PHYSHLTH_DAYS`, `MENTHLTH_DAYS`, and `BMI_NUMERIC`.
- **Multivariable Odds Ratios Table:** Displays adjusted odds ratios (aOR), 95% confidence intervals, and p-values from Milestone 4 survey-weighted logistic regression.

### Section C: Model Performance & Comparative Evaluation
- **Supervised Benchmark Table:** Cross-architecture comparison of all 4 candidate models evaluated on the validation partition ($N = 67,870$), reporting PR-AUC, ROC-AUC, Brier score, sensitivity, specificity, and survey-weighted metrics.
- **Discrimination Curves:** Interactive Plotly ROC curves (with random chance diagonal) and Precision-Recall curves (with $9.36\%$ prevalence baseline).
- **Calibration Deciles:** Interactive Plotly calibration curve comparing predicted risk deciles against observed event fractions for both champion and weighted models.
- **Educational Metric Callouts:** Concise explainers defining PR-AUC under class imbalance, ROC-AUC discrimination, Brier score calibration, and the distinction between continuous probabilities and threshold-dependent binary decisions.
- **Holdout Test Summary Table:** Reports independent holdout test partition ($N = 67,870$) results at default ($0.50$) and locked operational ($0.080$) cutoffs for unweighted sample metrics and survey-weighted population estimates.

### Section D: Decision Threshold Optimization & Trade-Off Analysis
- **Interactive Cutoff Exploration Slider:** Smooth range slider spanning $\tau \in [0.02, 0.65]$ (step 0.01) linked to the 89-point precomputed threshold grid.
- **Dynamic 2x2 Confusion Matrix:** Real-time calculation and display of TP, FP, TN, FN, Sensitivity, Specificity, Precision (PPV), NPV, and Youden's J for the explored threshold.
- **Locked Benchmark Side-by-Side:** Permanent comparison card displaying the immutable locked operational cutoff ($\tau = 0.080$: TP 5,290, FP 18,283, TN 43,236, FN 1,061; Sensitivity 83.29%, Specificity 70.28%, NPV 97.60%).
- **Interactive Threshold Curve:** Plotly chart displaying Sensitivity, Specificity, PPV, NPV, and Youden's J curves with a locked threshold line ($\tau = 0.080$) and a dynamic marker tracking the user's slider position.
- **Epidemiological Trade-Off Rationale:** Educational callout explaining why missing cardiac cases (False Negatives at $\tau = 0.50$) is unacceptable in population screening and why accepting a manageable false-positive burden at $\tau = 0.080$ is clinically justified.

### Section E: Explainable AI (XAI) Audits
- **Dual Granularity Feature Importance:** Toggle between original survey constructs (38 variables, grouped permutation importance) and transformed dummy indicators (top 25 of 190 columns showing indicator dilution).
- **Interactive Partial Dependence Profiles:** Variable selector for unscaled risk trajectories (`_AGE80` 18–80 years, `PHYSHLTH_DAYS` 0–30 days, `BMI_NUMERIC` $12.0–70.0\,\text{kg/m}^2$) with reference lines for the screening cutoff ($0.080$) and population prevalence ($0.0936$).
- **Four Anonymized Case Archetypes:** Interactive selector for True Positive (`VALIDATION_ID_044235`), False Positive (`VALIDATION_ID_029011`), True Negative (`VALIDATION_ID_051398`), and False Negative (`VALIDATION_ID_048866`).
- **Clinical Profile Card:** Complete demographic and health status summary for each case archetype.
- **Directional Local Attribution Plot:** Horizontal bar chart displaying marginal risk shifts ($\Delta p$) with color-coded positive (risk-increasing, red) and negative (protective, teal) contributions.
- **Methodology Notice:** Prominent statement clarifying that grouped marginal perturbation ($\Delta p$) and local Ridge surrogate models ($R^2 > 0.85$) are custom model-agnostic methods, NOT SHAP or LIME, and describe statistical associations rather than causal clinical mechanisms.

### Section F: Limitations & Responsible Use Guidelines
- **Surveillance Limitations Grid:** Details cross-sectional surveillance boundaries, absence of clinical screening trials, complex survey weighting point estimates, and demographic equity disclosures.
- **Governance Framework:** Side-by-side breakdown of Permissible Public Health Uses vs. Strictly Prohibited Clinical & Commercial Uses.
- **Mandatory Final Disclaimer:** Clear warning that the tool is an academic demonstration and not an FDA-cleared Software as a Medical Device (SaMD).

---

## 4. Verification & Testing

### 4.1 Automated Unit Tests
The complete automated test suite was executed via `.venv\Scripts\python.exe -m unittest discover tests -v`:

```
test_all_json_files_exist (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_exclusion_of_raw_respondent_records (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_file_size_constraints (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_locked_threshold_invariant (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_missing_artifact_and_schema_rejection (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_numerical_consistency_model_performance (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_numerical_consistency_overview (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_numerical_consistency_threshold_analysis (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_numerical_consistency_xai (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_schema_validations (test_dashboard_precomputation.TestDashboardPrecomputation) ... ok
test_all_six_sections_present_in_dom (test_dashboard_ui.TestDashboardUI) ... ok
test_charts_methods_coverage (test_dashboard_ui.TestDashboardUI) ... ok
test_css_design_system_tokens (test_dashboard_ui.TestDashboardUI) ... ok
test_disclaimers_prominently_rendered (test_dashboard_ui.TestDashboardUI) ... ok
test_exclusion_of_clinical_prediction_calculator (test_dashboard_ui.TestDashboardUI) ... ok
test_html_structure_and_title (test_dashboard_ui.TestDashboardUI) ... ok
test_locked_threshold_displayed_and_immutable (test_dashboard_ui.TestDashboardUI) ... ok
test_script_loading_sequence (test_dashboard_ui.TestDashboardUI) ... ok
test_views_methods_coverage (test_dashboard_ui.TestDashboardUI) ... ok
test_bijective_alignment_with_artifact (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_feature_count_dynamic_matching (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_locked_threshold_configuration (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_mapping_uniqueness (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_missing_sentinels_identified (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_numeric_feature_mapping (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_preprocessor_artifact_exists_and_fitted (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_source_variable_coverage (test_xai_feature_mapping.TestXAIFeatureMapping) ... ok
test_grouped_column_membership_completeness_and_disjointness (test_xai_global_importance.TestXAIGlobalImportance) ... ok
test_output_figures_integrity_if_present (test_xai_global_importance.TestXAIGlobalImportance) ... ok
test_output_tables_integrity_if_present (test_xai_global_importance.TestXAIGlobalImportance) ... ok
test_case_archetypes_summary_table_integrity (test_xai_local_explanations.TestLocalExplainability) ... ok
test_deterministic_case_selection_coverage (test_xai_local_explanations.TestLocalExplainability) ... ok
test_fig19_exists_and_valid (test_xai_local_explanations.TestLocalExplainability) ... ok
test_local_explanations_table_integrity (test_xai_local_explanations.TestLocalExplainability) ... ok
test_local_surrogate_fidelity (test_xai_local_explanations.TestLocalExplainability) ... ok
test_locked_threshold_value (test_xai_local_explanations.TestLocalExplainability) ... ok
test_quadrant_classification_logic (test_xai_local_explanations.TestLocalExplainability) ... ok
test_test_partition_untouched (test_xai_local_explanations.TestLocalExplainability) ... ok

----------------------------------------------------------------------
Ran 38 tests in 1.872s

OK
```
**Result:** **38 tests passed, 0 failures, 0 errors (100% pass rate).**

### 4.2 JavaScript Syntax Verification
Executed Node.js static syntax verification (`node --check`):
```powershell
node --check dashboard/components/data_loader.js dashboard/components/charts.js dashboard/components/views.js dashboard/app.js
```
**Result:** Exit code `0` with zero syntax errors across all 4 scripts.

### 4.3 HTTP Resource Serving Verification
An integration test script verified resource delivery under both Python HTTP server configurations:
1. **Serving directly from `dashboard/` directory:**
   - Command: `.venv\Scripts\python.exe -m http.server 8501 --directory dashboard`
   - All 11 resources (`index.html`, `styles.css`, `app.js`, 3 components, 5 JSON bundles) returned **HTTP 200 OK**.
2. **Serving from repository root:**
   - Command: `.venv\Scripts\python.exe -m http.server 8501`
   - All 11 resources returned **HTTP 200 OK**.

---

## 5. Technical Limitations & External Network Dependencies

1. **Plotly.js CDN Dependency:**
   - Plotly.js is loaded via CDN (`https://cdn.plot.ly/plotly-2.35.2.min.js`).
   - An active internet connection is required to download Plotly.js into the browser.
   - If offline, `app.js` gracefully intercepts missing Plotly, displays a prominent user warning banner, and renders all data tables and metadata cards without crashing.
2. **Cross-Sectional Inference Boundary:**
   - The dashboard classifies historical chronic disease reporting (`_MICHD`). It does not predict future disease onset or offer personal medical diagnoses.
3. **Survey Weight Variance Limitations:**
   - Weighted population percentages are point estimates; formal Taylor-series variance linearization is not computed dynamically on the client.

---

## 6. How to Launch the Dashboard Locally

To inspect and interact with the completed dashboard locally:

```powershell
# Open terminal in project root:
cd "C:\Users\Asus\Desktop\7 SEM\7-HA\CA-3\Project"

# Launch local HTTP server (Option 1 - Recommended):
.venv\Scripts\python.exe -m http.server 8501 --directory dashboard
# Navigate in browser to: http://localhost:8501/

# Alternatively (Option 2 - Root Serving):
.venv\Scripts\python.exe -m http.server 8501
# Navigate in browser to: http://localhost:8501/dashboard/
```

---

## 7. Stage 3 Conclusion & Stop Notice

Milestone 8 Stage 3 is **100% complete**:
- The standalone interactive dashboard UI is implemented and verified.
- All 6 sections are fully operational with real precomputed data and zero fabricated values.
- All 38 automated unit tests pass cleanly.
- `README.md` and project documentation have been updated with verified launch commands.

In accordance with the project prompt instructions, work has stopped here for your review before proceeding to Milestone 8 Stage 4 (Dashboard Integration Audit & Packaging).
