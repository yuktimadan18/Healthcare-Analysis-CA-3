"""
train_models.py
---------------
Training engine for Milestone 6: Baseline and Improved Model Development.
Fits Candidate Models 1A, 1B, 2A, and 2B strictly on the training partition
using features transformed by the frozen preprocessor artifact.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models import model_config


def load_training_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads preprocessed training features and target."""
    train_x_path = data_dir / "train_features.csv.gz"
    train_y_path = data_dir / "train_target.csv.gz"

    print(f"Loading training data from {train_x_path}...")
    X_train = pd.read_csv(train_x_path, index_col=0)
    y_train = pd.read_csv(train_y_path, index_col=0).squeeze("columns")

    print(f"Loaded X_train: {X_train.shape[0]:,} rows x {X_train.shape[1]} features.")
    print(f"Loaded y_train: {len(y_train):,} labels (positive count: {int(y_train.sum()):,}, {y_train.mean()*100:.2f}%).")
    return X_train, y_train


def transform_features(X: pd.DataFrame, preprocessor_path: Path) -> np.ndarray:
    """Transforms raw features using the frozen preprocessor artifact without refitting."""
    print(f"Loading frozen preprocessor from {preprocessor_path}...")
    preprocessor = joblib.load(preprocessor_path)

    print("Transforming feature matrix (calling transform ONLY)...")
    t0 = time.time()
    X_trans = preprocessor.transform(X)
    elapsed = time.time() - t0

    print(f"Transformation complete in {elapsed:.2f}s: shape = {X_trans.shape}.")
    assert X_trans.shape[1] == 190, f"Expected 190 features, got {X_trans.shape[1]}!"
    return X_trans


def train_candidate_models(
    X_train_trans: np.ndarray,
    y_train: pd.Series,
    output_models_dir: Path
) -> Dict[str, Any]:
    """
    Fits the four candidate models strictly on X_train_trans and saves them to disk.

    Candidates:
      1A. Logistic Regression (Unweighted Baseline)
      1B. Logistic Regression (Class-Weighted Comparison)
      2A. HistGradientBoostingClassifier (Unweighted GBDT)
      2B. HistGradientBoostingClassifier (Class-Weighted GBDT)
    """
    output_models_dir.mkdir(parents=True, exist_ok=True)
    trained_models = {}
    training_log = {}

    configs = model_config.MODEL_CONFIGS

    # 1. Candidate 1A: Logistic Regression (Unweighted)
    name_1a = "logistic_regression_unweighted"
    cfg_1a = configs[name_1a]
    print(f"\n[1/4] Training {cfg_1a['model_name']}...")
    t0 = time.time()
    clf_1a = LogisticRegression(**cfg_1a["params"])
    clf_1a.fit(X_train_trans, y_train)
    time_1a = time.time() - t0
    path_1a = output_models_dir / "baseline_logistic_regression.joblib"
    joblib.dump(clf_1a, path_1a)
    print(f"  Completed in {time_1a:.2f}s (iterations: {clf_1a.n_iter_[0]}). Saved to {path_1a.name}.")
    trained_models[name_1a] = clf_1a
    training_log[name_1a] = {"train_time_sec": round(time_1a, 2), "n_iter": int(clf_1a.n_iter_[0])}

    # 2. Candidate 1B: Logistic Regression (Class-Weighted)
    name_1b = "logistic_regression_balanced"
    cfg_1b = configs[name_1b]
    print(f"\n[2/4] Training {cfg_1b['model_name']}...")
    t0 = time.time()
    clf_1b = LogisticRegression(**cfg_1b["params"])
    clf_1b.fit(X_train_trans, y_train)
    time_1b = time.time() - t0
    path_1b = output_models_dir / "weighted_logistic_regression.joblib"
    joblib.dump(clf_1b, path_1b)
    print(f"  Completed in {time_1b:.2f}s (iterations: {clf_1b.n_iter_[0]}). Saved to {path_1b.name}.")
    trained_models[name_1b] = clf_1b
    training_log[name_1b] = {"train_time_sec": round(time_1b, 2), "n_iter": int(clf_1b.n_iter_[0])}

    # 3. Candidate 2A: HistGradientBoostingClassifier (Unweighted)
    name_2a = "hist_gradient_boosting_unweighted"
    cfg_2a = configs[name_2a]
    print(f"\n[3/4] Training {cfg_2a['model_name']}...")
    t0 = time.time()
    clf_2a = HistGradientBoostingClassifier(**cfg_2a["params"])
    clf_2a.fit(X_train_trans, y_train)
    time_2a = time.time() - t0
    path_2a = output_models_dir / "improved_hist_gradient_boosting.joblib"
    joblib.dump(clf_2a, path_2a)
    print(f"  Completed in {time_2a:.2f}s (trees: {clf_2a.n_iter_}). Saved to {path_2a.name}.")
    trained_models[name_2a] = clf_2a
    training_log[name_2a] = {"train_time_sec": round(time_2a, 2), "n_iter": int(clf_2a.n_iter_)}

    # 4. Candidate 2B: HistGradientBoostingClassifier (Class-Weighted)
    name_2b = "hist_gradient_boosting_balanced"
    cfg_2b = configs[name_2b]
    print(f"\n[4/4] Training {cfg_2b['model_name']}...")
    t0 = time.time()
    clf_2b = HistGradientBoostingClassifier(**cfg_2b["params"])
    clf_2b.fit(X_train_trans, y_train)
    time_2b = time.time() - t0
    path_2b = output_models_dir / "weighted_hist_gradient_boosting.joblib"
    joblib.dump(clf_2b, path_2b)
    print(f"  Completed in {time_2b:.2f}s (trees: {clf_2b.n_iter_}). Saved to {path_2b.name}.")
    trained_models[name_2b] = clf_2b
    training_log[name_2b] = {"train_time_sec": round(time_2b, 2), "n_iter": int(clf_2b.n_iter_)}

    print("\nAll 4 candidate models trained and serialized successfully!")
    return {"models": trained_models, "log": training_log}


if __name__ == "__main__":
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"
    OUTPUT_MODELS_DIR = PROJECT_ROOT / "models"

    X_train, y_train = load_training_data(DATA_DIR)
    X_train_trans = transform_features(X_train, PREPROCESSOR_PATH)
    results = train_candidate_models(X_train_trans, y_train, OUTPUT_MODELS_DIR)
