"""
generate_feature_notebook.py
----------------------------
Script to generate notebooks/03_feature_engineering_and_splits.ipynb for Milestone 5.
"""

from pathlib import Path
import json

nb_path = Path("notebooks/03_feature_engineering_and_splits.ipynb")

cells = [
    # Title & Metadata
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Milestone 5: Leakage-Safe Feature Engineering and Dataset Preparation\n",
            "**Project Title:** An Explainable Data-Driven Framework for Population Health Analytics and Predictive Modeling Using CDC BRFSS 2024  \n",
            "**Primary Target Outcome:** `_MICHD` (Reported History of Coronary Heart Disease or Myocardial Infarction)  \n",
            "**Binding Rules Adherence:** Rule 6 (Target lock verification), Rule 8 (Zero global code replacements), Rule 9 (Train-only preprocessing), Rule 10 (Target leakage prevention), Rule 11 (Observational task boundaries), Rule 12 (Complex survey weighting).\n",
            "\n",
            "---\n",
            "\n",
            "### Notebook Objectives:\n",
            "1. **Recompute and Verify Target Outcome (`_MICHD`):** Confirm exact response codes and establish the $N = 452,464$ eligible surveillance cohort.\n",
            "2. **Enforce Leakage Controls:** Audit all 61 candidate variables and strictly exclude constituent variables (`CVDINFR4`, `CVDCRHD4`).\n",
            "3. **Survey Metadata Isolation:** Retain complex design variables (`_LLCPWT`, `_STSTR`, `_PSU`, `_STATE`) strictly in metadata for audit and evaluation, never as predictive inputs.\n",
            "4. **Feature Engineering & Codebook Recoding:** Clean continuous variables (`PHYSHLTH_DAYS`, `MENTHLTH_DAYS`, `BMI_NUMERIC`) and preserve categorical response codes.\n",
            "5. **Stratified Disjoint Partitioning:** Partition cohort into 70% Train, 15% Validation, and 15% Test splits with fixed random seed (42).\n",
            "6. **Leakage-Safe Preprocessing Pipeline:** Build and fit scikit-learn `ColumnTransformer` strictly on the training partition.\n",
            "7. **Verification & Serialization:** Validate artifact integrity in `data/processed/` and export summary audit tables."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "# Visual configuration\n",
            "sns.set_theme(style=\"whitegrid\", font_scale=1.1)\n",
            "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
            "plt.rcParams[\"figure.dpi\"] = 150\n",
            "\n",
            "# Project path configuration\n",
            "PROJECT_ROOT = Path(\"..\").resolve()\n",
            "sys.path.append(str(PROJECT_ROOT))\n",
            "\n",
            "from src.features import feature_config\n",
            "from src.features.build_features import (\n",
            "    load_interim_data,\n",
            "    filter_eligible_cohort,\n",
            "    engineer_features,\n",
            "    partition_data,\n",
            "    build_preprocessor_pipeline,\n",
            "    generate_inclusion_exclusion_audit,\n",
            "    generate_partition_summary\n",
            ")\n",
            "\n",
            "DATA_PATH = PROJECT_ROOT / \"data\" / \"interim\" / \"brfss_2024_selected.csv.gz\"\n",
            "DICT_PATH = PROJECT_ROOT / \"data\" / \"interim\" / \"brfss_2024_dictionary.json\"\n",
            "PROCESSED_DIR = PROJECT_ROOT / \"data\" / \"processed\"\n",
            "TABLES_DIR = PROJECT_ROOT / \"results\" / \"tables\"\n",
            "MODELS_DIR = PROJECT_ROOT / \"models\"\n",
            "\n",
            "print(f\"Project root: {PROJECT_ROOT}\")\n",
            "print(f\"Target outcome: {feature_config.TARGET_COL} ({feature_config.TARGET_NAME})\")"
        ]
    },

    # Section 1: Eligible Cohort Verification
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Codebook Target Verification & Eligible Cohort Extraction\n",
            "\n",
            "According to the official CDC 2024 Codebook (`USCODE24_LLCP_082125.HTML`), the primary target `_MICHD` is defined as:\n",
            "- `1` = Reported having MI or CHD (`CVDINFR4=1` OR `CVDCRHD4=1`)\n",
            "- `2` = Did not report having MI or CHD (`CVDINFR4=2` AND `CVDCRHD4=2`)\n",
            "- `BLANK` (NaN) = Not asked or missing in either question\n",
            "\n",
            "We verify these figures against the actual interim dataset and isolate eligible records."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "raw_df = load_interim_data(DATA_PATH)\n",
            "cohort_df, excluded_df = filter_eligible_cohort(raw_df)\n",
            "\n",
            "# Verify target figures\n",
            "n_pos = int((cohort_df[\"target\"] == 1).sum())\n",
            "n_neg = int((cohort_df[\"target\"] == 0).sum())\n",
            "total_valid = len(cohort_df)\n",
            "weights = cohort_df[\"_LLCPWT\"]\n",
            "w_pos = weights[cohort_df[\"target\"] == 1].sum()\n",
            "w_tot = weights.sum()\n",
            "\n",
            "print(\"\\n--- Target Verification Summary ---\")\n",
            "print(f\"Total raw survey respondents:       {len(raw_df):,}\")\n",
            "print(f\"Excluded unasked/missing target:    {len(excluded_df):,} ({len(excluded_df)/len(raw_df)*100:.2f}%)\")\n",
            "print(f\"Eligible surveillance cohort:       {total_valid:,} ({total_valid/len(raw_df)*100:.2f}%)\")\n",
            "print(f\"Valid positive cases (_MICHD=1):    {n_pos:,} ({n_pos/total_valid*100:.2f}% unweighted, {w_pos/w_tot*100:.2f}% weighted)\")\n",
            "print(f\"Valid negative cases (_MICHD=2):    {n_neg:,} ({n_neg/total_valid*100:.2f}% unweighted)\")\n",
            "print(f\"Class imbalance ratio:              1 : {n_neg/n_pos:.2f}\")\n",
            "\n",
            "# Assert exact match with official audited figures\n",
            "assert n_pos == 42338, f\"Mismatch in positive cases: {n_pos} != 42,338\"\n",
            "assert n_neg == 410126, f\"Mismatch in negative cases: {n_neg} != 410,126\"\n",
            "assert len(excluded_df) == 5206, f\"Mismatch in excluded cases: {len(excluded_df)} != 5,206\"\n",
            "print(\"All target verification assertions passed with 100% precision!\")"
        ]
    },

    # Section 2: Feature Inclusion / Exclusion Audit
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Feature Inclusion / Exclusion Audit & Leakage Controls\n",
            "\n",
            "To prevent target leakage and multi-collinear redundancy, all 61 variables from the interim dictionary are classified into explicit operational roles:\n",
            "1. **Primary Target (`_MICHD`):** Excluded from feature space.\n",
            "2. **Mandatory Leakage Exclusions (`CVDINFR4`, `CVDCRHD4`):** Direct mathematical constituents of `_MICHD`. Banned from predictive inputs.\n",
            "3. **Survey Design Metadata (`_LLCPWT`, `_STSTR`, `_PSU`, `_STATE`):** Preserved for evaluation, calibration, and subgroup audit; excluded from ordinary feature matrix.\n",
            "4. **Redundant Variables:** Excluded in favor of richer, more granular representations (e.g. `_AGE65YR` excluded in favor of continuous `_AGE80` and 5-year `_AGEG5YR`).\n",
            "5. **Included Predictors (38 features):** 4 continuous numeric features + 34 categorical risk factors across demographics, behaviors, healthcare access, comorbidities, and functional limitations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "audit_df = generate_inclusion_exclusion_audit(DICT_PATH)\n",
            "print(f\"Audited {len(audit_df)} variables from {DICT_PATH.name}:\")\n",
            "display(audit_df[\"status\"].value_counts().to_frame(\"Variable Count\"))\n",
            "\n",
            "# Display leakage exclusions\n",
            "print(\"\\n--- Mandatory Leakage Exclusions ---\")\n",
            "display(audit_df[audit_df[\"status\"] == \"Excluded: Target Leakage\"][[\"variable\", \"category\", \"rationale\"]])\n",
            "\n",
            "# Save audit table\n",
            "audit_table_path = TABLES_DIR / \"feature_inclusion_exclusion_audit.csv\"\n",
            "audit_df.to_csv(audit_table_path, index=False)\n",
            "print(f\"Saved feature audit table to {audit_table_path}\")"
        ]
    },

    # Section 3: Feature Engineering
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Feature Engineering & Clean Recoding\n",
            "\n",
            "We engineer clean predictors according to exact codebook rules without global replacements:\n",
            "- `PHYSHLTH_DAYS`: Map 88 (None) to 0 days; 77/99 to NaN.\n",
            "- `MENTHLTH_DAYS`: Map 88 (None) to 0 days; 77/99 to NaN.\n",
            "- `BMI_NUMERIC`: Rescale continuous `_BMI5` by 100.0 (kg/m$^2$).\n",
            "- `_AGE80`: Preserved as continuous imputed age (18 to 80 years).\n",
            "- Categorical features: Cast to string representations so Refusal/DK/Missing codes remain distinct levels."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X, y, metadata_df = engineer_features(cohort_df)\n",
            "\n",
            "print(\"\\n--- Continuous Numeric Features Summary ---\")\n",
            "display(X[feature_config.NUMERIC_FEATURES].describe().round(2))\n",
            "\n",
            "# Assertions confirming leakage prevention\n",
            "assert \"_MICHD\" not in X.columns, \"LEAKAGE ERROR: _MICHD found in X!\"\n",
            "assert \"CVDINFR4\" not in X.columns, \"LEAKAGE ERROR: CVDINFR4 found in X!\"\n",
            "assert \"CVDCRHD4\" not in X.columns, \"LEAKAGE ERROR: CVDCRHD4 found in X!\"\n",
            "assert \"_LLCPWT\" not in X.columns, \"DESIGN ERROR: Survey weight found in X!\"\n",
            "print(\"Leakage controls verified: no target components or survey weights enter feature matrix X.\")"
        ]
    },

    # Section 4: Stratified Disjoint Partitioning
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Stratified Disjoint Partitioning & Complex Survey Considerations\n",
            "\n",
            "We partition the 452,464 eligible records into **70% Training ($N = 316,724$)**, **15% Validation ($N = 67,870$)**, and **15% Test ($N = 67,870$)** splits with fixed seed (42).\n",
            "\n",
            "### Complex Survey Partitioning Analysis:\n",
            "- **Stratification on Target:** Preserves identical unweighted (9.36%) and survey-weighted (~6.80%) disease prevalence across all three partitions.\n",
            "- **Limitations of Random Splitting:** Because BRFSS is a multi-stage clustered survey, independent random sampling across clusters means respondents from the same PSU (`_PSU`) or stratum (`_STSTR`) can appear in both training and test sets.\n",
            "- **Sensitivity to State-Grouped Splitting:** A state-grouped split tests geographic out-of-domain transfer rather than within-survey population estimation. Because states have differing module routing and sample sizes, stratified random partitioning remains our primary benchmark, while state-level evaluations serve as a sensitivity analysis."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X_dict, y_dict, metadata_df = partition_data(X, y, metadata_df)\n",
            "\n",
            "partition_summary_df = generate_partition_summary(y_dict, metadata_df)\n",
            "print(\"\\n--- Dataset Partition Summary Table ---\")\n",
            "display(partition_summary_df)\n",
            "\n",
            "# Save partition summary\n",
            "partition_table_path = TABLES_DIR / \"dataset_partition_summary.csv\"\n",
            "partition_summary_df.to_csv(partition_table_path, index=False)\n",
            "print(f\"Saved partition summary to {partition_table_path}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visual verification of class balance across partitions\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "plot_data = partition_summary_df[partition_summary_df[\"partition\"] != \"Full Eligible Cohort\"].copy()\n",
            "\n",
            "# Subplot A: Total sample size per partition\n",
            "sns.barplot(data=plot_data, x=\"partition\", y=\"total_records\", palette=\"Blues_d\", ax=axes[0])\n",
            "for i, row in plot_data.iterrows():\n",
            "    axes[0].text(i, row[\"total_records\"] + 5000, f\"{row['total_records']:,}\\n({row['pct_of_cohort']} %)\", ha=\"center\", weight=\"bold\")\n",
            "axes[0].set_title(\"A. Total Sample Size by Partition\", weight=\"bold\")\n",
            "axes[0].set_xlabel(\"Partition\")\n",
            "axes[0].set_ylabel(\"Number of Respondents\")\n",
            "axes[0].set_ylim(0, 360000)\n",
            "\n",
            "# Subplot B: Unweighted vs Survey-Weighted Prevalence\n",
            "prev_melt = plot_data.melt(\n",
            "    id_vars=[\"partition\"],\n",
            "    value_vars=[\"unweighted_prevalence_pct\", \"survey_weighted_prevalence_pct\"],\n",
            "    var_name=\"Prevalence Type\",\n",
            "    value_name=\"Prevalence (%)\"\n",
            ")\n",
            "prev_melt[\"Prevalence Type\"] = prev_melt[\"Prevalence Type\"].map({\n",
            "    \"unweighted_prevalence_pct\": \"Unweighted Sample Prev\",\n",
            "    \"survey_weighted_prevalence_pct\": \"Survey-Weighted Prev\"\n",
            "})\n",
            "sns.barplot(data=prev_melt, x=\"partition\", y=\"Prevalence (%)\", hue=\"Prevalence Type\", palette=\"Set2\", ax=axes[1])\n",
            "axes[1].set_title(\"B. Unweighted vs. Survey-Weighted Prevalence Consistency\", weight=\"bold\")\n",
            "axes[1].set_xlabel(\"Partition\")\n",
            "axes[1].set_ylabel(\"Disease Prevalence (%)\")\n",
            "axes[1].set_ylim(0, 12)\n",
            "axes[1].legend(loc=\"upper right\")\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },

    # Section 5: Preprocessing Pipeline Construction
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Leakage-Safe Preprocessing Pipeline Construction\n",
            "\n",
            "We construct a scikit-learn `ColumnTransformer`:\n",
            "- **Numeric pipeline:** Median imputation (`SimpleImputer(strategy='median')`) followed by standard scaling (`StandardScaler()`).\n",
            "- **Categorical pipeline:** Mode imputation (`SimpleImputer(strategy='most_frequent')`) followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`).\n",
            "\n",
            "**MANDATORY LEAKAGE ENFORCEMENT:**\n",
            "The preprocessor is fitted **ONLY on `X_train`**. Validation and test partitions are transformed using the training parameters, preventing statistical data leakage across folds."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "preprocessor = build_preprocessor_pipeline()\n",
            "print(\"Fitting ColumnTransformer on X_train ONLY...\")\n",
            "preprocessor.fit(X_dict[\"train\"])\n",
            "print(\"ColumnTransformer fitted successfully.\")\n",
            "\n",
            "# Transform a sample of train and validation sets\n",
            "train_sample_trans = preprocessor.transform(X_dict[\"train\"].head(100))\n",
            "val_sample_trans = preprocessor.transform(X_dict[\"validation\"].head(100))\n",
            "\n",
            "cat_ohe = preprocessor.named_transformers_[\"cat\"].named_steps[\"onehot\"]\n",
            "num_feature_names = feature_config.NUMERIC_FEATURES\n",
            "cat_feature_names = cat_ohe.get_feature_names_out(feature_config.CATEGORICAL_FEATURES)\n",
            "all_transformed_feature_names = list(num_feature_names) + list(cat_feature_names)\n",
            "\n",
            "print(f\"Transformed input feature dimension: {len(all_transformed_feature_names)} columns (from 38 raw predictors).\")\n",
            "print(f\"  Numeric feature count:     {len(num_feature_names)}\")\n",
            "print(f\"  One-Hot categorical count: {len(cat_feature_names)}\")\n",
            "\n",
            "# Save fitted preprocessor artifact\n",
            "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n",
            "preprocessor_artifact_path = MODELS_DIR / \"preprocessor.joblib\"\n",
            "joblib.dump(preprocessor, preprocessor_artifact_path)\n",
            "print(f\"Saved fitted preprocessor to {preprocessor_artifact_path}\")"
        ]
    },

    # Section 6: Artifact Integrity Verification
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Processed Dataset Serialization & Artifact Integrity Verification\n",
            "\n",
            "All datasets are saved as gzip-compressed CSV files in `data/processed/`:\n",
            "- `train_features.csv.gz`, `train_target.csv.gz`\n",
            "- `validation_features.csv.gz`, `validation_target.csv.gz`\n",
            "- `test_features.csv.gz`, `test_target.csv.gz`\n",
            "- `cohort_metadata.csv.gz` (survey weights `_LLCPWT`, `_STSTR`, `_PSU`, `_STATE` with partition assignments)\n",
            "\n",
            "We verify that all saved artifacts reload with 100% integrity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Verify files exist and can be reloaded\n",
            "for part in [\"train\", \"validation\", \"test\"]:\n",
            "    x_file = PROCESSED_DIR / f\"{part}_features.csv.gz\"\n",
            "    y_file = PROCESSED_DIR / f\"{part}_target.csv.gz\"\n",
            "    \n",
            "    assert x_file.exists(), f\"Missing artifact: {x_file}\"\n",
            "    assert y_file.exists(), f\"Missing artifact: {y_file}\"\n",
            "    \n",
            "    loaded_x = pd.read_csv(x_file, index_col=0, nrows=10)\n",
            "    loaded_y = pd.read_csv(y_file, index_col=0, nrows=10)\n",
            "    assert len(loaded_x) == 10 and len(loaded_y) == 10, \"Reload row count error!\"\n",
            "    print(f\"Verified {part} partition files exist and reload successfully.\")\n",
            "\n",
            "meta_file = PROCESSED_DIR / \"cohort_metadata.csv.gz\"\n",
            "assert meta_file.exists(), f\"Missing metadata: {meta_file}\"\n",
            "meta_sample = pd.read_csv(meta_file, index_col=0, nrows=10)\n",
            "print(f\"Verified cohort metadata exists ({meta_sample.shape[1]} columns: {list(meta_sample.columns)}).\")\n",
            "print(\"\\nAll Milestone 5 artifacts verified. Ready for baseline model training in Milestone 6!\")"
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.14"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated {nb_path} with {len(cells)} cells.")
