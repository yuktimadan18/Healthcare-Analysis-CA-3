"""
xai_config.py
-------------
Configuration and path specifications for Milestone 7: Explainable AI (XAI)
for the CDC BRFSS 2024 Cardiovascular Disease Surveillance Model.
"""

from pathlib import Path
import json
from typing import Dict, Any, List

# Project Root Resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Artifact Paths
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL_PATH = MODELS_DIR / "improved_hist_gradient_boosting.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_SELECTION_SUMMARY_PATH = MODELS_DIR / "model_selection_summary.json"

# Read Locked Operational Threshold from Frozen Milestone 6 Model Selection Summary
def get_locked_threshold() -> float:
    """Reads the locked operational threshold from model_selection_summary.json."""
    if MODEL_SELECTION_SUMMARY_PATH.exists():
        with open(MODEL_SELECTION_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
            th = summary.get("selected_operational_threshold", 0.080)
            return float(th)
    return 0.080

LOCKED_THRESHOLD: float = get_locked_threshold()

# Reproducibility and Computational Settings
RANDOM_STATE: int = 42
PERMUTATION_REPEATS: int = 10
TOP_K_GLOBAL_FEATURES: int = 25
LOCAL_NEIGHBORHOOD_SAMPLES: int = 1000

# Explicit Dataset Partition Paths
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_FEATURES_PATH = DATA_PROCESSED_DIR / "train_features.csv.gz"
TRAIN_TARGET_PATH = DATA_PROCESSED_DIR / "train_target.csv.gz"
VALIDATION_FEATURES_PATH = DATA_PROCESSED_DIR / "validation_features.csv.gz"
VALIDATION_TARGET_PATH = DATA_PROCESSED_DIR / "validation_target.csv.gz"
TEST_FEATURES_PATH = DATA_PROCESSED_DIR / "test_features.csv.gz"
TEST_TARGET_PATH = DATA_PROCESSED_DIR / "test_target.csv.gz"
COHORT_METADATA_PATH = DATA_PROCESSED_DIR / "cohort_metadata.csv.gz"

# Output Directories
RESULTS_DIR = PROJECT_ROOT / "results"
XAI_TABLES_DIR = RESULTS_DIR / "tables"
XAI_FIGURES_DIR = RESULTS_DIR / "figures"
XAI_EXPLANATIONS_DIR = RESULTS_DIR / "explanations"

# Output File Paths for Milestone 7
FEATURE_MAPPING_AUDIT_PATH = XAI_TABLES_DIR / "xai_feature_mapping_audit.csv"
GLOBAL_VARIABLE_IMPORTANCE_PATH = XAI_TABLES_DIR / "xai_global_variable_importance.csv"
TRANSFORMED_FEATURE_IMPORTANCE_PATH = XAI_TABLES_DIR / "xai_transformed_feature_importance.csv"
LOCAL_CASE_EXPLANATIONS_PATH = XAI_EXPLANATIONS_DIR / "xai_local_case_explanations.csv"

FIG_GLOBAL_VARIABLE_IMPORTANCE = XAI_FIGURES_DIR / "fig16_global_variable_importance.png"
FIG_INDICATOR_FEATURE_IMPORTANCE = XAI_FIGURES_DIR / "fig17_indicator_feature_importance.png"
FIG_PARTIAL_DEPENDENCE_PROFILES = XAI_FIGURES_DIR / "fig18_partial_dependence_profiles.png"
FIG_LOCAL_CASE_ATTRIBUTIONS = XAI_FIGURES_DIR / "fig19_local_case_attributions.png"
