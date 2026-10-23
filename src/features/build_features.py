"""
build_features.py
-----------------
Leakage-safe feature engineering, codebook-aligned response recoding,
and reproducible train/validation/test dataset preparation for CDC BRFSS 2024.

Strictly enforces:
  - Exclusion of target components (CVDINFR4, CVDCRHD4) to prevent target leakage.
  - Retention of survey design variables (_LLCPWT, _STSTR, _PSU, _STATE) in metadata only.
  - Fitting of preprocessing transformers ONLY on the training partition.
  - Reproducible 70/15/15 stratified partitioning with documented random seed.
"""

from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from src.features import feature_config


def load_interim_data(data_path: Path) -> pd.DataFrame:
    """Loads interim selected dataset."""
    print(f"Loading interim data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.")
    return df


def filter_eligible_cohort(
    df: pd.DataFrame,
    target_col: str = feature_config.TARGET_COL,
    pos_codes: List[int] = feature_config.TARGET_POS_CODES,
    neg_codes: List[int] = feature_config.TARGET_NEG_CODES
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters respondents with valid positive or negative target codes.
    Separates unasked, missing, or indeterminate records into an audit DataFrame.

    Returns:
        Tuple of (eligible_cohort_df, excluded_records_df).
    """
    valid_mask = df[target_col].isin(pos_codes + neg_codes)
    cohort_df = df[valid_mask].copy()
    excluded_df = df[~valid_mask].copy()

    # Create stable unique identifier mapping to original row index
    cohort_df["record_id"] = cohort_df.index
    excluded_df["record_id"] = excluded_df.index

    # Create binary target indicator (1 = positive, 0 = negative)
    cohort_df["target"] = (cohort_df[target_col].isin(pos_codes)).astype(int)

    pos_count = int(cohort_df["target"].sum())
    neg_count = int((cohort_df["target"] == 0).sum())
    total_valid = len(cohort_df)
    unw_prev = (pos_count / total_valid) * 100.0

    weights = cohort_df["_LLCPWT"]
    weighted_prev = (weights[cohort_df["target"] == 1].sum() / weights.sum()) * 100.0

    print(f"Eligible Cohort: {total_valid:,} records ({total_valid / len(df) * 100:.2f}% of survey).")
    print(f"  Target Positive (1): {pos_count:,} ({unw_prev:.2f}% unweighted, {weighted_prev:.2f}% weighted).")
    print(f"  Target Negative (0): {neg_count:,} ({100 - unw_prev:.2f}% unweighted).")
    print(f"Excluded Records: {len(excluded_df):,} ({len(excluded_df) / len(df) * 100:.2f}% missing target).")

    return cohort_df, excluded_df


def engineer_features(
    cohort_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Performs codebook-guided recoding and feature engineering.

    Operations:
      - PHYSHLTH: Recodes 88 (None) to 0 days; 77/99 to NaN.
      - MENTHLTH: Recodes 88 (None) to 0 days; 77/99 to NaN.
      - _BMI5: Divides by 100.0 to convert to standard BMI (kg/m^2).
      - _AGE80: Kept as continuous imputed age.
      - Categorical variables: Cast to string representations so missing/refusal/DK
        codes are modeled as explicit distinct categories.
      - Separates survey metadata (_LLCPWT, _STSTR, _PSU, _STATE) from predictors.

    Returns:
        Tuple of (X_features, y_target, metadata_df).
    """
    work_df = cohort_df.copy()

    # 1. Continuous numeric transformations
    work_df["PHYSHLTH_DAYS"] = work_df["PHYSHLTH"].replace({88: 0, 77: np.nan, 99: np.nan})
    work_df["MENTHLTH_DAYS"] = work_df["MENTHLTH"].replace({88: 0, 77: np.nan, 99: np.nan})
    work_df["BMI_NUMERIC"] = work_df["_BMI5"] / 100.0

    # 2. Extract metadata
    metadata_cols = ["record_id", feature_config.TARGET_COL, "target"] + list(feature_config.SURVEY_DESIGN_METADATA.keys())
    metadata_df = work_df[metadata_cols].copy()

    # 3. Extract feature matrix X
    predictor_cols = feature_config.NUMERIC_FEATURES + feature_config.CATEGORICAL_FEATURES
    X = work_df[predictor_cols].copy()

    # Note: Categorical columns are kept in their native numeric code representations (with np.nan
    # for genuine missingness). This preserves legitimate response codes (such as 7 for Don't Know
    # and 9 for Refused) as distinct categories while avoiding accidental conversion of NaNs into
    # literal 'nan' strings. SimpleImputer(strategy='most_frequent') in the preprocessor pipeline
    # will impute genuine NaNs strictly based on training partition modes.

    y = work_df["target"].copy()

    # Assert no target leakage in X
    assert feature_config.TARGET_COL not in X.columns, "FATAL: Target column present in feature matrix!"
    for leak_col in feature_config.MANDATORY_LEAKAGE_EXCLUSIONS.keys():
        assert leak_col not in X.columns, f"FATAL: Leakage column {leak_col} present in feature matrix!"

    print(f"Engineered feature matrix X: {X.shape[0]:,} rows x {X.shape[1]} predictors.")
    print(f"  Numeric predictors ({len(feature_config.NUMERIC_FEATURES)}): {feature_config.NUMERIC_FEATURES}")
    print(f"  Categorical predictors ({len(feature_config.CATEGORICAL_FEATURES)}): {len(feature_config.CATEGORICAL_FEATURES)} features.")

    return X, y, metadata_df


def partition_data(
    X: pd.DataFrame,
    y: pd.Series,
    metadata_df: pd.DataFrame,
    train_ratio: float = feature_config.TRAIN_RATIO,
    val_ratio: float = feature_config.VAL_RATIO,
    test_ratio: float = feature_config.TEST_RATIO,
    random_state: int = feature_config.RANDOM_STATE
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], pd.DataFrame]:
    """
    Partitions the eligible cohort into disjoint Train (70%), Validation (15%),
    and Test (15%) splits, stratified on the binary target.

    Returns:
        Tuple of (X_dict, y_dict, updated_metadata_df_with_partitions).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0!"

    # First split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # Second split: Val vs Test (50/50 of temp)
    val_sub_ratio = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_sub_ratio),
        random_state=random_state,
        stratify=y_temp
    )

    # Assign partition tags to metadata
    meta = metadata_df.copy()
    meta["partition"] = "unknown"
    meta.loc[X_train.index, "partition"] = "train"
    meta.loc[X_val.index, "partition"] = "validation"
    meta.loc[X_test.index, "partition"] = "test"

    # Verify disjointness and completeness
    train_idx_set = set(X_train.index)
    val_idx_set = set(X_val.index)
    test_idx_set = set(X_test.index)

    assert len(train_idx_set.intersection(val_idx_set)) == 0, "Train and Validation overlap!"
    assert len(train_idx_set.intersection(test_idx_set)) == 0, "Train and Test overlap!"
    assert len(val_idx_set.intersection(test_idx_set)) == 0, "Validation and Test overlap!"
    assert len(train_idx_set) + len(val_idx_set) + len(test_idx_set) == len(X), "Partitions do not cover all rows!"

    X_dict = {"train": X_train, "validation": X_val, "test": X_test}
    y_dict = {"train": y_train, "validation": y_val, "test": y_test}

    print(f"Stratified Partitioning Complete (random_state={random_state}):")
    for part in ["train", "validation", "test"]:
        part_y = y_dict[part]
        part_meta = meta[meta["partition"] == part]
        w = part_meta["_LLCPWT"]
        w_prev = (w[part_y == 1].sum() / w.sum()) * 100.0
        print(f"  {part.capitalize():10s}: {len(part_y):,} rows ({len(part_y)/len(X)*100:.1f}%), "
              f"pos={part_y.sum():,} ({part_y.mean()*100:.2f}% unw, {w_prev:.2f}% wtd).")

    return X_dict, y_dict, meta


def build_preprocessor_pipeline() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for leakage-safe preprocessing.

    Components:
      - Numeric Pipeline: SimpleImputer(median) -> StandardScaler()
      - Categorical Pipeline: SimpleImputer(constant, fill_value=-1.0) -> OneHotEncoder(handle_unknown='ignore')
        Preserves genuine missingness as an explicit, separate category (-1.0)
        guaranteed not to collide with non-negative valid BRFSS response codes.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=feature_config.CATEGORICAL_MISSING_SENTINEL)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_config.NUMERIC_FEATURES),
            ("cat", cat_pipeline, feature_config.CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def generate_inclusion_exclusion_audit(dictionary_path: Path) -> pd.DataFrame:
    """
    Audits all 61 variables from the interim dictionary, recording the explicit
    raw-to-engineered mapping, status, and technical rationale for every variable.
    """
    with open(dictionary_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)

    variables = meta_dict["variables"]
    audit_rows = []

    # Explicit mapping for raw source variables transformed into numeric predictors
    RAW_NUMERIC_MAPPING = {
        "_AGE80": (
            "_AGE80",
            "Source for Engineered Numeric Predictor",
            "Imputed age collapsed at 80 years. Continuous clinical predictor (18-80 years); imputed via median and scaled via StandardScaler (fit on train only)."
        ),
        "PHYSHLTH": (
            "PHYSHLTH_DAYS",
            "Source for Engineered Numeric Predictor",
            "Cleaned days physical health not good in past 30 days. Code 88 recoded to 0 days; codes 77 (Don't Know) and 99 (Refused) recoded to NaN; imputed via median and scaled via StandardScaler (fit on train only)."
        ),
        "MENTHLTH": (
            "MENTHLTH_DAYS",
            "Source for Engineered Numeric Predictor",
            "Cleaned days mental health not good in past 30 days. Code 88 recoded to 0 days; codes 77 (Don't Know) and 99 (Refused) recoded to NaN; imputed via median and scaled via StandardScaler (fit on train only)."
        ),
        "_BMI5": (
            "BMI_NUMERIC",
            "Source for Engineered Numeric Predictor",
            "Continuous Body Mass Index (kg/m^2). Computed by dividing raw _BMI5 by 100.0 (implied two decimals); genuine missingness imputed via median and scaled via StandardScaler (fit on train only)."
        ),
    }

    for var_name, var_info in variables.items():
        role_desc = var_info.get("role", "")
        category = var_info.get("category", "")

        if var_name == feature_config.TARGET_COL:
            status = "Primary Prediction Target"
            eng_col = "target"
            rationale = (
                "Approved primary prediction target (Reported MI/CHD). Recoded to binary target "
                "(1 = reported history, 0 = no reported history). Strictly excluded from feature matrix X."
            )
        elif var_name in feature_config.MANDATORY_LEAKAGE_EXCLUSIONS:
            status = "Excluded: Direct Target Leakage"
            eng_col = "None (Excluded)"
            rationale = feature_config.MANDATORY_LEAKAGE_EXCLUSIONS[var_name]
        elif var_name in feature_config.SURVEY_DESIGN_METADATA:
            status = "Preserved: Survey Metadata"
            eng_col = var_name
            rationale = (
                f"{feature_config.SURVEY_DESIGN_METADATA[var_name]} Preserved in cohort_metadata.csv.gz; "
                "excluded from predictive feature matrix X to prevent shortcut learning."
            )
        elif var_name in feature_config.REDUNDANT_FEATURE_EXCLUSIONS:
            status = "Excluded: Redundant / Skip Pattern"
            eng_col = "None (Excluded)"
            rationale = feature_config.REDUNDANT_FEATURE_EXCLUSIONS[var_name]
        elif var_name in RAW_NUMERIC_MAPPING:
            eng_col, status, rationale = RAW_NUMERIC_MAPPING[var_name]
        elif var_name in feature_config.CATEGORICAL_FEATURES:
            status = "Included: Categorical Predictor"
            eng_col = var_name
            rationale = (
                f"Categorical survey predictor ({role_desc}). Valid codes (including refusal/DK) "
                "preserved as distinct categories; genuine missingness preserved as explicit sentinel "
                f"category ({feature_config.CATEGORICAL_MISSING_SENTINEL}) and one-hot encoded (186 OHE columns)."
            )
        else:
            status = "Excluded: Unselected"
            eng_col = "None (Excluded)"
            rationale = f"Not selected for core predictive model ({role_desc})."

        audit_rows.append({
            "raw_variable": var_name,
            "category": category,
            "cdc_description": role_desc,
            "variable_status": status,
            "engineered_column": eng_col,
            "transformation_and_leakage_notes": rationale
        })

    audit_df = pd.DataFrame(audit_rows)
    return audit_df


def generate_partition_summary(
    y_dict: Dict[str, pd.Series],
    metadata_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates summary table describing sample sizes, positive case counts,
    unweighted and survey-weighted prevalences, and imbalance ratios across partitions.
    """
    rows = []
    total_cohort_size = len(metadata_df)

    for part in ["train", "validation", "test"]:
        y_part = y_dict[part]
        meta_part = metadata_df[metadata_df["partition"] == part]
        weights = meta_part["_LLCPWT"]

        n_obs = len(y_part)
        n_pos = int(y_part.sum())
        n_neg = n_obs - n_pos
        pct_of_total = (n_obs / total_cohort_size) * 100.0
        unw_prev = (n_pos / n_obs) * 100.0
        weighted_prev = (weights[y_part == 1].sum() / weights.sum()) * 100.0
        imbalance = n_neg / n_pos if n_pos > 0 else np.nan

        rows.append({
            "partition": part.capitalize(),
            "total_records": n_obs,
            "pct_of_cohort": round(pct_of_total, 2),
            "positive_cases": n_pos,
            "negative_cases": n_neg,
            "unweighted_prevalence_pct": round(unw_prev, 2),
            "survey_weighted_prevalence_pct": round(weighted_prev, 2),
            "imbalance_ratio": round(imbalance, 2)
        })

    # Add Full Cohort Row
    full_weights = metadata_df["_LLCPWT"]
    full_pos = int(metadata_df["target"].sum())
    full_neg = total_cohort_size - full_pos
    full_unw = (full_pos / total_cohort_size) * 100.0
    full_wtd = (full_weights[metadata_df["target"] == 1].sum() / full_weights.sum()) * 100.0
    rows.append({
        "partition": "Full Eligible Cohort",
        "total_records": total_cohort_size,
        "pct_of_cohort": 100.00,
        "positive_cases": full_pos,
        "negative_cases": full_neg,
        "unweighted_prevalence_pct": round(full_unw, 2),
        "survey_weighted_prevalence_pct": round(full_wtd, 2),
        "imbalance_ratio": round(full_neg / full_pos, 2)
    })

    return pd.DataFrame(rows)


def run_pipeline(
    data_path: Path,
    dict_path: Path,
    output_processed_dir: Path,
    output_tables_dir: Path,
    models_dir: Path
):
    """Executes the complete Milestone 5 feature engineering and dataset preparation pipeline."""
    output_processed_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load interim dataset
    df = load_interim_data(data_path)

    # 2. Filter eligible cohort
    cohort_df, excluded_df = filter_eligible_cohort(df)

    # 3. Engineer features and extract metadata
    X, y, metadata_df = engineer_features(cohort_df)

    # 4. Partition data
    X_dict, y_dict, metadata_df = partition_data(X, y, metadata_df)

    # 5. Fit Preprocessing Pipeline ONLY on Training Data
    print("Fitting ColumnTransformer on training partition ONLY (preventing data leakage)...")
    preprocessor = build_preprocessor_pipeline()
    preprocessor.fit(X_dict["train"])
    print("Successfully fitted ColumnTransformer on X_train.")

    # Save fitted preprocessor
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved fitted preprocessor to {preprocessor_path}")

    # 6. Save Processed Datasets (gzip compressed CSVs)
    print("Saving processed partition datasets...")
    for part in ["train", "validation", "test"]:
        x_path = output_processed_dir / f"{part}_features.csv.gz"
        y_path = output_processed_dir / f"{part}_target.csv.gz"
        X_dict[part].to_csv(x_path, compression="gzip", index=True)
        y_dict[part].to_csv(y_path, compression="gzip", index=True)
        print(f"  Saved {x_path} ({X_dict[part].shape[0]:,} rows x {X_dict[part].shape[1]} cols)")
        print(f"  Saved {y_path}")

    # Save metadata table
    meta_path = output_processed_dir / "cohort_metadata.csv.gz"
    metadata_df.to_csv(meta_path, compression="gzip", index=True)
    print(f"Saved cohort metadata to {meta_path}")

    # 7. Generate and Save Summary Tables
    audit_table = generate_inclusion_exclusion_audit(dict_path)
    audit_path = output_tables_dir / "feature_inclusion_exclusion_audit.csv"
    audit_table.to_csv(audit_path, index=False)
    print(f"Saved feature audit to {audit_path} ({len(audit_table)} variables audited).")

    summary_table = generate_partition_summary(y_dict, metadata_df)
    summary_path = output_tables_dir / "dataset_partition_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    print(f"Saved partition summary to {summary_path}")

    print("\nMilestone 5 feature pipeline execution completed successfully!")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" / "interim" / "brfss_2024_selected.csv.gz"
    DICT_PATH = PROJECT_ROOT / "data" / "interim" / "brfss_2024_dictionary.json"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TABLES_DIR = PROJECT_ROOT / "results" / "tables"
    MODELS_DIR = PROJECT_ROOT / "models"

    run_pipeline(
        data_path=DATA_PATH,
        dict_path=DICT_PATH,
        output_processed_dir=PROCESSED_DIR,
        output_tables_dir=TABLES_DIR,
        models_dir=MODELS_DIR
    )
