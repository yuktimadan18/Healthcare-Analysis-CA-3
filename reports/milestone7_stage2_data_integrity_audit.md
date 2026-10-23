# Milestone 7 Stage 2: Validation Data Integrity and Numerical Discrepancy Audit

**Project:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024
**Date:** October 20, 2026
**Audit Target:** Investigation of reported validation positive cases ($5,744$ vs. $6,351$) and prevalence ($8.46\%$ vs. $9.36\%$).
**Scope:** Strict inspection of data partitions, code, saved models, evaluation artifacts, and audit reports without altering any models, preprocessors, partitions, tables, or figures.

---

## 1. Executive Summary & Core Finding

A thorough, reproducible forensic audit was conducted to resolve the apparent discrepancy between:
- **Milestone 6 Documentation & Datasets:** Validation $N = 67,870$, positive cases = **$6,351$**, prevalence = **$9.358\%$** ($9.36\%$), PR-AUC = **$0.3701$**, ROC-AUC = **$0.8449$**.
- **Stage 2 Turn Summary Statement:** Stated positive cases = **$5,744$**, prevalence = **$8.46\%$**, baseline PR-AUC = **$0.370088$**, baseline ROC-AUC = **$0.844895$**.

### Core Determination:
1. **The physical validation datasets and pipeline code are 100% correct and uncorrupted.**
   - `data/processed/validation_features.csv.gz` contains exactly $67,870$ rows and $38$ survey predictors.
   - `data/processed/validation_target.csv.gz` contains exactly $67,870$ rows, strictly aligned by row index, with exactly **$6,351$ positive cases** ($9.3576\%$) and $61,519$ negative cases ($90.6424\%$). There are $0$ missing values.
2. **The Stage 2 computation and audit report were executed on the true data.**
   - In [`reports/milestone7_global_importance_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md), lines 7 and 34 explicitly state:
     - `Analysis Partition: Validation Partition Strictly ($N = 67,870$, $6,351$ positive cases, prevalence: $9.358\%$)`
     - `Prevalence Baseline Reference: 0.093576 (9.36%)`
   - In [`src/xai/global_importance.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py), the code loaded `y_val` directly from `data/processed/validation_target.csv.gz` without any subsetting or filtering.
   - The unpermuted baseline PR-AUC (**$0.370088$**) and ROC-AUC (**$0.844895$**) match Milestone 6's rounded figures ($0.3701$ and $0.8449$) to 4 decimal places. Mathematically, achieving a PR-AUC of $0.370088$ on this model's probability predictions requires the true $6,351$ positive labels; if the true positives were $5,744$, the PR-AUC would differ substantially due to the class-imbalance sensitivity of average precision.
3. **Root Cause:**
   - The numbers **$5,744$** and **$8.46\%$** were an **isolated conversational drafting error** in Section 3 of the assistant's previous turn response text.
   - A workspace-wide automated regex search (`5744`, `8.46`, `0.0846`) across all source code, CSV tables, JSON summaries, and markdown reports confirmed that **neither $5,744$ nor $8.46\%$ exists anywhere in any file within the repository**.
   - No underlying data corruption, silent filtering, or misaligned target vector occurred.

---

## 2. Item-by-Item Verification

### Check 1: Physical Validation Features and Target Inspection
The physical validation files were inspected directly using `pandas`:
- **Validation Features:** `data/processed/validation_features.csv.gz`
  - Shape: `(67870, 38)`
  - Columns: 38 original survey variables
  - Index: `int64` ranging from 7 to 457,664 (disjoint from train/test)
  - Null count: 29,624 raw cells (handled as documented by frozen preprocessor)
- **Validation Target:** `data/processed/validation_target.csv.gz`
  - Shape: `(67870, 1)`
  - Column name: `target`
  - Positive cases (`target == 1`): **$6,351$**
  - Negative cases (`target == 0`): **$61,519$**
  - Missing values: **$0$**
  - Unweighted prevalence: **$0.0935759540$** ($9.3576\% \approx 9.36\%$)
- **Index Alignment:**
  - `(df_feat.index == df_tgt.index).all()` evaluated to `True` across all $67,870$ rows.

### Check 2: Milestone 6 Validation Baseline Artifacts
The frozen Milestone 6 artifacts were cross-checked:
- [`models/model_selection_summary.json`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json):
  - Confusion matrix at locked threshold ($0.08$):
    - $\text{TP} = 5,290$, $\text{FN} = 1,061 \implies \text{Total Positives} = 5,290 + 1,061 = \mathbf{6,351}$.
    - $\text{FP} = 18,283$, $\text{TN} = 43,236 \implies \text{Total Negatives} = 18,283 + 43,236 = \mathbf{61,519}$.
    - Total validation cohort: $6,351 + 61,519 = \mathbf{67,870}$.
  - Validation PR-AUC: $0.3701$, Validation ROC-AUC: $0.8449$.
- [`results/tables/dataset_partition_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/dataset_partition_summary.csv):
  - Row 3 (Validation): Total = $67,870$, Positives = $6,351$, Negatives = $61,519$, Unweighted prevalence = $9.36\%$, Survey-weighted prevalence = $6.77\%$.
- [`results/tables/model_validation_comparison.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv):
  - Champion `hist_gradient_boosting_unweighted`: `val_pr_auc`: $0.3701$, `val_roc_auc`: $0.8449$, `val_brier_score`: $0.07$.

### Check 3 & 4: Data-Loading Logic in Milestone 7 XAI vs. Milestone 6
Comparison of data loading implementations:
- **Milestone 6 (`src/evaluation/evaluate_models.py`, lines 619–620):**
  ```python
  X_val = pd.read_csv(data_dir / "validation_features.csv.gz", index_col=0)
  y_val = pd.read_csv(data_dir / "validation_target.csv.gz", index_col=0).squeeze("columns")
  ```
- **Milestone 7 XAI (`src/xai/global_importance.py`, lines 40–41):**
  ```python
  X_val_raw = pd.read_csv(xai_config.VALIDATION_FEATURES_PATH, index_col=0)
  y_val = pd.read_csv(xai_config.VALIDATION_TARGET_PATH, index_col=0).squeeze("columns").values
  ```
- **Verification:** Both modules read the exact same path (`data/processed/validation_target.csv.gz`) with `index_col=0`. The loaded vector in `global_importance.py` was directly verified:
  ```python
  len(y_val) == 67870
  (y_val == 1).sum() == 6351
  (y_val == 0).sum() == 61519
  ```
  Target alignment is strictly 1-to-1 with zero misalignment or row omissions.

### Check 5: Recalculation of Unpermuted Baseline Metrics
Without altering any model or preprocessor artifact, the champion model and frozen preprocessor were loaded in Python, the validation features were transformed, and baseline discrimination and calibration metrics were recalculated:
- **Baseline PR-AUC (full double-precision float):** `0.3700880405524041`
- **Baseline ROC-AUC (full double-precision float):** `0.8448948066269037`
- **Baseline Brier Score (full double-precision float):** `0.0700116084282583`
- **Comparison to Milestone 6:**
  - PR-AUC: $0.370088 \implies \mathbf{0.3701}$ (exact match)
  - ROC-AUC: $0.844895 \implies \mathbf{0.8449}$ (exact match)
  - Brier score: $0.070012 \implies \mathbf{0.0700}$ (exact match)

### Check 6: Permutation Importance Target Vector Integrity
In [`src/xai/global_importance.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py), `compute_grouped_permutation_importance()` receives `y_val` from `load_validation_data_and_model()`. The target vector was never altered during permutation; only rows of `worker_X[:, cols]` were shuffled. In [`results/tables/xai_global_variable_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv), column `baseline_pr_auc` records `0.3700880405524041`, confirming that all feature drops were calculated against the true $6,351$-positive baseline.

### Check 7: Grouped Permutation Multi-Column Integrity
The implementation in `src/xai/global_importance.py` (lines 95–105) was verified:
```python
rng = np.random.RandomState(random_state + repeat_idx)
perm = rng.permutation(n_samples)
...
for var, cols in grouped_indices.items():
    orig_slice = worker_X[:, cols].copy()
    worker_X[:, cols] = X_val_trans[perm][:, cols]
    probs_perm = model.predict_proba(worker_X)[:, 1]
    worker_X[:, cols] = orig_slice
```
- `perm` is a single $67,870$-length row permutation generated once per repeat.
- `worker_X[:, cols] = X_val_trans[perm][:, cols]` assigns the exact same row permutation simultaneously to all column indices `cols` belonging to `var`.
- Dummy indicators within multi-category groups (e.g., `GENHLTH`, `_SEX`, `CVDSTRK3`) were never permuted independently.

### Check 8: Partial Dependence Reference Lines & Description Accuracy
Inspection of [`results/figures/fig18_partial_dependence_profiles.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png) and `src/xai/global_importance.py` (lines 253–256):
- Reference line 1: `Cohort Prevalence (9.36%)` at $y = 0.0936$. This corresponds exactly to the empirical validation prevalence of $6,351 / 67,870 = 0.093576$.
- Reference line 2: `Screening Threshold (0.080)` at $y = 0.0800$. This corresponds to the locked Youden's J threshold from Milestone 6.
- The title and caption explicitly state:
  `[Note: Curves describe statistical model behavior across empirical ranges; they do not establish causal intervention effects.]`

---

## 3. Discrepancy Forensic Matrix

| Dimension | Milestone 6 Audit Artifacts | Stage 2 Audit Report (`reports/milestone7_global_importance_audit.md`) | Assistant Turn Text Response (Chat Only) | Actual Physical Validation Files (`data/processed/`) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sample Size ($N$)** | $67,870$ | $67,870$ | $67,870$ | $67,870$ | Fully Consistent |
| **Positive Cases** | **$6,351$** | **$6,351$** | **$5,744$** | **$6,351$** | Chat Typo Only; Files & Report Correct |
| **Negative Cases** | $61,519$ | $61,519$ | Unstated ($62,126$) | $61,519$ | Fully Consistent in Code & Artifacts |
| **Prevalence** | $9.36\%$ ($9.358\%$) | $9.358\%$ ($9.36\%$) | $8.46\%$ | $9.3576\%$ ($9.36\%$) | Chat Typo Only; Files & Report Correct |
| **Baseline PR-AUC** | $0.3701$ | $0.370088$ | $0.370088$ | $0.370088$ | Mathematically Identical |
| **Baseline ROC-AUC** | $0.8449$ | $0.844895$ | $0.844895$ | $0.844895$ | Mathematically Identical |
| **Grouped Permutation** | N/A | Grouped (simultaneous) | Grouped (simultaneous) | Verified in code | Valid |
| **Threshold Annotations** | $0.080$ | $0.080$ & $9.36\%$ | $0.080$ | N/A | Valid |

---

## 4. Exact Diagnostic Commands and Files Inspected

### Commands Executed:
1. `data/processed` partition check:
   ```powershell
   .venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/processed/validation_target.csv.gz', index_col=0); print(df.value_counts(dropna=False))"
   ```
2. Full workspace regex search for `5744`, `8.46`, and `0.0846`:
   ```powershell
   .venv\Scripts\python.exe -c "import os, re; pattern = re.compile(r'5744|8\.46|0\.0846'); ..."
   # Result: 0 matches found across entire repository
   ```
3. End-to-end baseline recalculation:
   ```powershell
   .venv\Scripts\python.exe -c "import joblib, pandas as pd; from sklearn.metrics import average_precision_score, roc_auc_score; ..."
   # Result: PR-AUC = 0.3700880406, ROC-AUC = 0.8448948066 on 6,351 positive targets
   ```

### Files Inspected:
- [`data/processed/validation_features.csv.gz`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/data/processed/validation_features.csv.gz)
- [`data/processed/validation_target.csv.gz`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/data/processed/validation_target.csv.gz)
- [`models/improved_hist_gradient_boosting.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/improved_hist_gradient_boosting.joblib)
- [`models/preprocessor.joblib`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/preprocessor.joblib)
- [`models/model_selection_summary.json`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/models/model_selection_summary.json)
- [`results/tables/dataset_partition_summary.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/dataset_partition_summary.csv)
- [`results/tables/model_validation_comparison.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/model_validation_comparison.csv)
- [`results/tables/threshold_optimization_analysis.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/threshold_optimization_analysis.csv)
- [`results/tables/xai_global_variable_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv)
- [`results/tables/xai_transformed_feature_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_transformed_feature_importance.csv)
- [`results/figures/fig16_global_variable_importance.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig16_global_variable_importance.png)
- [`results/figures/fig17_indicator_feature_importance.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig17_indicator_feature_importance.png)
- [`results/figures/fig18_partial_dependence_profiles.png`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/figures/fig18_partial_dependence_profiles.png)
- [`reports/milestone7_global_importance_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md)
- [`src/xai/global_importance.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/xai/global_importance.py)
- [`src/evaluation/evaluate_models.py`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/src/evaluation/evaluate_models.py)

---

## 5. Recommended Next Action

Because:
1. No data corruption, partition contamination, or code bug exists,
2. The Stage 2 importance results in [`results/tables/xai_global_variable_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_global_variable_importance.csv) and [`results/tables/xai_transformed_feature_importance.csv`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/results/tables/xai_transformed_feature_importance.csv) were computed on the verified $6,351$-positive validation partition,
3. The formal report [`reports/milestone7_global_importance_audit.md`](file:///C:/Users/Asus/Desktop/7%20SEM/7-HA/CA-3/Project/reports/milestone7_global_importance_audit.md) already correctly documents $N = 67,870$, $6,351$ positives, and $9.358\%$ prevalence:

**Recommended Action:**
- No file overwrite, code modification, or recomputation of Stage 2 importance tables or figures is necessary.
- Formally record the resolution of this inquiry in this data integrity report.
- Proceed with user authorization to **Milestone 7, Stage 3: Local Explainable AI (XAI)**.
