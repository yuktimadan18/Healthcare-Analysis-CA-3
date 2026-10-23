"""
model_config.py
---------------
Model hyperparameters, evaluation configurations, and random state settings
for Milestone 6: Baseline and Improved Model Development (CDC BRFSS 2024).
"""

from typing import Dict, Any, List
import numpy as np

# Global deterministic random state
RANDOM_STATE: int = 42

# Primary Model Selection Metric on Validation Set
PRIMARY_SELECTION_METRIC: str = "pr_auc"  # Average Precision (PR-AUC)
SECONDARY_SELECTION_METRIC: str = "roc_auc"
CALIBRATION_METRIC: str = "brier_score"

# Empirical cohort prevalence reference (for baseline comparison)
EMPIRICAL_PREVALENCE_REF: float = 0.09357

# Candidate Model Configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "logistic_regression_unweighted": {
        "model_name": "Logistic Regression (Unweighted Baseline)",
        "estimator_type": "linear",
        "params": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "class_weight": None,
        },
        "description": "Standard L2-regularized logistic regression maintaining natural 9.36% cohort calibration."
    },
    "logistic_regression_balanced": {
        "model_name": "Logistic Regression (Class-Weighted)",
        "estimator_type": "linear",
        "params": {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
        },
        "description": "L2-regularized logistic regression with inverse class-frequency loss penalty."
    },
    "hist_gradient_boosting_unweighted": {
        "model_name": "HistGradientBoosting (Unweighted GBDT)",
        "estimator_type": "tree_ensemble",
        "params": {
            "loss": "log_loss",
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "random_state": RANDOM_STATE,
            "class_weight": None,
            "early_stopping": "auto",
            "n_iter_no_change": 10,
        },
        "description": "Native scikit-learn histogram GBDT modeling non-linearities under natural calibration."
    },
    "hist_gradient_boosting_balanced": {
        "model_name": "HistGradientBoosting (Class-Weighted GBDT)",
        "estimator_type": "tree_ensemble",
        "params": {
            "loss": "log_loss",
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "early_stopping": "auto",
            "n_iter_no_change": 10,
        },
        "description": "Histogram GBDT with internal balanced class weighting in gradient/Hessian calculation."
    },
}

# Decision Threshold Optimization Grid
THRESHOLD_SEARCH_MIN: float = 0.02
THRESHOLD_SEARCH_MAX: float = 0.90
THRESHOLD_SEARCH_STEP: float = 0.01

# Documented Validation Objectives for Threshold Selection
THRESHOLD_OBJECTIVES: List[str] = [
    "youden_j",           # Maximize Sensitivity + Specificity - 1 (balanced population screening)
    "f1_optimal",         # Maximize harmonic mean of Precision and Recall
    "high_capture_recall" # Target >= 75% Sensitivity for broad surveillance capture
]
