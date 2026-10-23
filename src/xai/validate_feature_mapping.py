"""
validate_feature_mapping.py
---------------------------
Feature-mapping validation utility for Milestone 7: Explainable AI (XAI).
Inspects the frozen fitted preprocessor artifact and deterministically maps
each transformed output feature back to its original survey variable.
Validates uniqueness, completeness, and exact 1-to-1 correspondence.
"""

from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import feature_config
from src.xai import xai_config


def extract_and_validate_feature_mapping(
    preprocessor_path: Path = xai_config.PREPROCESSOR_PATH,
    audit_output_path: Path = xai_config.FEATURE_MAPPING_AUDIT_PATH,
    save_if_valid: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Extracts the feature mapping directly from the fitted preprocessor artifact,
    validates that every transformed feature maps to exactly one source survey variable,
    and saves the audit table if validation succeeds.
    """
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")

    print(f"Loading preprocessor artifact from: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)

    # 1. Obtain actual transformed feature names dynamically from the fitted transformer
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError("Preprocessor artifact does not implement get_feature_names_out()")

    actual_feature_names = list(preprocessor.get_feature_names_out())
    actual_feature_count = len(actual_feature_names)
    print(f"Actual transformed features in artifact: {actual_feature_count}")

    # 2. Extract transformer structure
    named_transformers = getattr(preprocessor, "named_transformers_", None)
    if named_transformers is None:
        raise AttributeError("Preprocessor does not have named_transformers_ attribute.")

    mapping_records: List[Dict[str, Any]] = []
    mapped_feature_names: List[str] = []

    # Process Numeric Pipeline
    if "num" in named_transformers:
        num_cols = None
        for name, transformer, cols in preprocessor.transformers_:
            if name == "num":
                num_cols = cols
                break

        if num_cols is None:
            raise ValueError("Numeric transformer 'num' found but could not resolve column list.")

        for col in num_cols:
            transformed_name = f"num__{col}"
            mapping_records.append({
                "transformed_feature_name": transformed_name,
                "source_variable": col,
                "feature_type": "numeric",
                "category_value": "N/A",
                "is_missing_sentinel": False
            })
            mapped_feature_names.append(transformed_name)
    else:
        raise ValueError("Preprocessor missing expected 'num' transformer.")

    # Process Categorical Pipeline
    if "cat" in named_transformers:
        cat_pipeline = named_transformers["cat"]
        if not hasattr(cat_pipeline, "named_steps") or "onehot" not in cat_pipeline.named_steps:
            raise ValueError("Categorical pipeline missing 'onehot' step.")

        onehot = cat_pipeline.named_steps["onehot"]
        categories = getattr(onehot, "categories_", None)
        if categories is None:
            raise ValueError("OneHotEncoder in preprocessor has not been fitted or missing categories_.")

        cat_cols = None
        for name, transformer, cols in preprocessor.transformers_:
            if name == "cat":
                cat_cols = cols
                break

        if cat_cols is None:
            raise ValueError("Categorical transformer 'cat' found but could not resolve column list.")

        if len(cat_cols) != len(categories):
            raise ValueError(f"Mismatch between cat_cols count ({len(cat_cols)}) and categories_ arrays ({len(categories)}).")

        for col, cat_vals in zip(cat_cols, categories):
            for val in cat_vals:
                transformed_name = f"cat__{col}_{val}"
                is_missing = bool(val == feature_config.CATEGORICAL_MISSING_SENTINEL)
                mapping_records.append({
                    "transformed_feature_name": transformed_name,
                    "source_variable": col,
                    "feature_type": "categorical",
                    "category_value": str(val),
                    "is_missing_sentinel": is_missing
                })
                mapped_feature_names.append(transformed_name)
    else:
        raise ValueError("Preprocessor missing expected 'cat' transformer.")

    # 3. Rigorous Validation Checks
    print("\n--- Executing Rigorous Feature Mapping Validation ---")

    # Check A: Count Validation
    mapped_count = len(mapping_records)
    if mapped_count != actual_feature_count:
        raise ValueError(
            f"VALIDATION ERROR: Mapped feature count ({mapped_count}) does not match "
            f"actual transformed feature count ({actual_feature_count})!"
        )
    print(f"[PASS] Feature count matches: {mapped_count} == {actual_feature_count}")

    # Check B: Uniqueness of Transformed Features
    if len(set(mapped_feature_names)) != len(mapped_feature_names):
        duplicates = [x for x in mapped_feature_names if mapped_feature_names.count(x) > 1]
        raise ValueError(f"VALIDATION ERROR: Duplicate feature names detected in mapping: {set(duplicates)}")
    print(f"[PASS] Transformed feature names are strictly unique ({len(set(mapped_feature_names))} unique).")

    # Check C: Exact Bijective Set Equality
    unmapped_in_actual = set(actual_feature_names) - set(mapped_feature_names)
    unmapped_in_mapping = set(mapped_feature_names) - set(actual_feature_names)

    if unmapped_in_actual:
        raise ValueError(f"VALIDATION ERROR: Unmapped features present in preprocessor: {unmapped_in_actual}")
    if unmapped_in_mapping:
        raise ValueError(f"VALIDATION ERROR: Mapping contains unexpected features not in preprocessor: {unmapped_in_mapping}")
    print("[PASS] Exact bijective set equality verified (0 unmapped, 0 ambiguous).")

    # Check D: Exact Order Alignment
    if actual_feature_names != mapped_feature_names:
        raise ValueError("VALIDATION ERROR: Order of mapped features does not exactly match preprocessor get_feature_names_out().")
    print("[PASS] Exact index-by-index order alignment verified.")

    # Check E: Source Variable Coverage
    expected_sources = set(feature_config.NUMERIC_FEATURES + feature_config.CATEGORICAL_FEATURES)
    mapped_sources = set(rec["source_variable"] for rec in mapping_records)

    missing_sources = expected_sources - mapped_sources
    unexpected_sources = mapped_sources - expected_sources

    if missing_sources:
        raise ValueError(f"VALIDATION ERROR: Expected source variables missing from mapping: {missing_sources}")
    if unexpected_sources:
        raise ValueError(f"VALIDATION ERROR: Unexpected source variables found in mapping: {unexpected_sources}")
    print(f"[PASS] Source variable coverage verified: All {len(expected_sources)} original predictors accounted for.")

    # 4. Construct Mapping DataFrame
    df_mapping = pd.DataFrame(mapping_records)
    df_mapping.insert(0, "feature_index", range(len(df_mapping)))

    # Construct Mapping Dictionary
    mapping_dict = {
        row["transformed_feature_name"]: {
            "feature_index": row["feature_index"],
            "source_variable": row["source_variable"],
            "feature_type": row["feature_type"],
            "category_value": row["category_value"],
            "is_missing_sentinel": row["is_missing_sentinel"]
        }
        for _, row in df_mapping.iterrows()
    }

    # 5. Summary Statistics
    num_count = (df_mapping["feature_type"] == "numeric").sum()
    cat_count = (df_mapping["feature_type"] == "categorical").sum()
    missing_sentinel_count = df_mapping["is_missing_sentinel"].sum()
    unique_sources = df_mapping["source_variable"].nunique()

    print("\n--- Feature Mapping Summary ---")
    print(f"Total Transformed Features:      {len(df_mapping)}")
    print(f"Unique Original Predictors:      {unique_sources} (4 Numeric + {unique_sources - num_count} Categorical)")
    print(f"Continuous Numeric Features:     {num_count}")
    print(f"Categorical Dummy Indicators:    {cat_count}")
    print(f"Missing Sentinel Indicators:     {missing_sentinel_count} (value == -1.0)")

    # 6. Save Audit Table Only If Validation Passes
    if save_if_valid:
        audit_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_mapping.to_csv(audit_output_path, index=False)
        print(f"\n[SAVED] Mapping audit table successfully saved to: {audit_output_path}")

    return mapping_dict, df_mapping


if __name__ == "__main__":
    extract_and_validate_feature_mapping()
