"""
evaluate_models.py
------------------
Pure evaluation engine for Milestone 6: Baseline and Improved Model Development.
Implements:
  - Discrimination metrics: PR-AUC (Average Precision), ROC-AUC
  - Probabilistic calibration metrics: Brier score, Brier Skill Score, ECE, calibration curves
  - Threshold optimization on validation data across multiple public health objectives
  - Strict model locking mechanism prior to holdout test evaluation
  - Single-pass test evaluation comparing unweighted sample performance against
    survey-weighted population estimates (using _LLCPWT)
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from sklearn.calibration import calibration_curve

from src.models import model_config


def compute_discrimination_and_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes threshold-independent discrimination and calibration metrics."""
    # 1. Discrimination
    roc_auc = float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
    pr_auc = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))

    # 2. Calibration
    brier = float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight))

    # Brier Skill Score relative to climatological baseline prevalence
    if sample_weight is not None:
        p_ref = float(np.average(y_true, weights=sample_weight))
    else:
        p_ref = float(np.mean(y_true))
    brier_ref = p_ref * (1.0 - p_ref)
    bss = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "brier_skill_score": round(bss, 4),
        "mean_predicted_prob": round(float(np.average(y_prob, weights=sample_weight)), 4),
        "observed_prevalence": round(p_ref, 4)
    }


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Computes confusion matrix and threshold-dependent classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    if sample_weight is not None:
        # Weighted confusion matrix counts
        tp = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 1)]))
        fp = float(np.sum(sample_weight[(y_true == 0) & (y_pred == 1)]))
        tn = float(np.sum(sample_weight[(y_true == 0) & (y_pred == 0)]))
        fn = float(np.sum(sample_weight[(y_true == 1) & (y_pred == 0)]))
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tp, fp, tn, fn = float(tp), float(fp), float(tn), float(fn)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": round(threshold, 3),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision_ppv": round(precision, 4),
        "npv": round(npv, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "youden_j": round(youden_j, 4),
        "tp": round(tp, 1),
        "fp": round(fp, 1),
        "tn": round(tn, 1),
        "fn": round(fn, 1)
    }


def evaluate_validation_models(
    models: Dict[str, Any],
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    meta_val: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates all candidate models on the validation set strictly.
    Primary ranking is determined by PR-AUC (Average Precision).
    """
    print("\nEvaluating Candidate Models on Validation Partition (N = 67,870)...")
    results = []

    y_val_arr = y_val.values
    weights_val = meta_val["_LLCPWT"].values

    for model_key, clf in models.items():
        cfg = model_config.MODEL_CONFIGS.get(model_key, {"model_name": model_key, "description": ""})
        prob_pos = clf.predict_proba(X_val_trans)[:, 1]

        # 1. Unweighted validation metrics
        unw_metrics = compute_discrimination_and_calibration(y_val_arr, prob_pos, sample_weight=None)

        # Default 0.50 threshold metrics
        th_050 = compute_threshold_metrics(y_val_arr, prob_pos, threshold=0.50, sample_weight=None)
        # Reference prevalence threshold metrics
        th_prev = compute_threshold_metrics(y_val_arr, prob_pos, threshold=model_config.EMPIRICAL_PREVALENCE_REF, sample_weight=None)

        # 2. Survey-weighted validation metrics (for reference)
        wtd_metrics = compute_discrimination_and_calibration(y_val_arr, prob_pos, sample_weight=weights_val)

        results.append({
            "model_key": model_key,
            "model_name": cfg["model_name"],
            "val_pr_auc": unw_metrics["pr_auc"],
            "val_roc_auc": unw_metrics["roc_auc"],
            "val_brier_score": unw_metrics["brier_score"],
            "val_brier_skill_score": unw_metrics["brier_skill_score"],
            "val_mean_pred_prob": unw_metrics["mean_predicted_prob"],
            "sens_at_050": th_050["sensitivity"],
            "spec_at_050": th_050["specificity"],
            "f1_at_050": th_050["f1_score"],
            "sens_at_prev_th": th_prev["sensitivity"],
            "spec_at_prev_th": th_prev["specificity"],
            "f1_at_prev_th": th_prev["f1_score"],
            "survey_weighted_pr_auc": wtd_metrics["pr_auc"],
            "survey_weighted_roc_auc": wtd_metrics["roc_auc"],
            "survey_weighted_brier": wtd_metrics["brier_score"],
        })

    df_res = pd.DataFrame(results)
    # Sort by Primary Selection Metric (val_pr_auc descending)
    df_res = df_res.sort_values(by="val_pr_auc", ascending=False).reset_index(drop=True)
    return df_res



def optimize_thresholds(
    clf: Any,
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    threshold_grid: np.ndarray = np.arange(model_config.THRESHOLD_SEARCH_MIN, model_config.THRESHOLD_SEARCH_MAX + 0.001, model_config.THRESHOLD_SEARCH_STEP)
) -> pd.DataFrame:
    """Evaluates classification performance across candidate decision thresholds on validation data."""
    y_val_arr = y_val.values
    prob_pos = clf.predict_proba(X_val_trans)[:, 1]

    rows = []
    for th in threshold_grid:
        m = compute_threshold_metrics(y_val_arr, prob_pos, threshold=float(th), sample_weight=None)
        rows.append(m)

    df_th = pd.DataFrame(rows)
    return df_th


def lock_champion_model_decision(
    val_summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    output_path: Path
) -> Dict[str, Any]:
    """
    Selects and locks the champion model and operational threshold based on validation data.
    Saves a formal JSON lock before holdout test data is accessed.
    """
    # 1. Select champion by primary metric (PR-AUC)
    top_model = val_summary_df.iloc[0]
    champion_key = top_model["model_key"]
    champion_name = top_model["model_name"]
    best_pr_auc = float(top_model["val_pr_auc"])
    best_roc_auc = float(top_model["val_roc_auc"])

    # 2. Select optimal threshold on validation set
    # Objective A: Maximum Youden's J
    best_youden_idx = threshold_df["youden_j"].idxmax()
    th_youden = float(threshold_df.loc[best_youden_idx, "threshold"])

    # Objective B: Maximum F1-score
    best_f1_idx = threshold_df["f1_score"].idxmax()
    th_f1 = float(threshold_df.loc[best_f1_idx, "threshold"])

    # Objective C: High-capture recall (first threshold reaching >= 75% sensitivity)
    high_rec_candidates = threshold_df[threshold_df["sensitivity"] >= 0.75]
    th_high_rec = float(high_rec_candidates["threshold"].max()) if len(high_rec_candidates) > 0 else th_youden

    # Documented choice: We select the Youden's J threshold as the primary balanced operational threshold,
    # as it balances clinical sensitivity against false alarm rates for general population screening.
    selected_threshold = th_youden
    selected_objective = "youden_j (Balanced Population Screening)"

    lock_payload = {
        "status": "LOCKED",
        "primary_selection_metric": model_config.PRIMARY_SELECTION_METRIC,
        "champion_model_key": champion_key,
        "champion_model_name": champion_name,
        "validation_pr_auc": best_pr_auc,
        "validation_roc_auc": best_roc_auc,
        "validation_brier_score": float(top_model["val_brier_score"]),
        "selected_operational_threshold": selected_threshold,
        "selection_objective": selected_objective,
        "candidate_thresholds": {
            "default_050": 0.50,
            "empirical_prevalence_ref": round(model_config.EMPIRICAL_PREVALENCE_REF, 4),
            "optimal_youden_j": th_youden,
            "optimal_f1": th_f1,
            "target_75pct_recall": th_high_rec
        },
        "validation_metrics_at_selected_threshold": threshold_df.loc[best_youden_idx].to_dict()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lock_payload, f, indent=2)

    print(f"\nCHAMPION MODEL LOCKED: {champion_name}")
    print(f"  Primary Metric (Validation PR-AUC): {best_pr_auc:.4f} (ROC-AUC: {best_roc_auc:.4f})")
    print(f"  Selected Operational Threshold:     {selected_threshold:.3f} ({selected_objective})")
    print(f"  Lock saved to {output_path}")
    return lock_payload


def evaluate_locked_model_on_test(
    champion_clf: Any,
    lock_payload: Dict[str, Any],
    X_test_trans: np.ndarray,
    y_test: pd.Series,
    meta_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Executes a SINGLE, UNBIASED forward evaluation pass on the holdout test set (N = 67,870).
    Evaluates both unweighted sample performance and survey-weighted population estimates.
    """
    print("\n--- EXECUTING FINAL UNBIASED HOLDOUT TEST EVALUATION (N = 67,870) ---")
    y_test_arr = y_test.values
    weights_test = meta_test["_LLCPWT"].values
    prob_test = champion_clf.predict_proba(X_test_trans)[:, 1]

    sel_th = float(lock_payload["selected_operational_threshold"])

    # 1. Unweighted test metrics
    unw_disc = compute_discrimination_and_calibration(y_test_arr, prob_test, sample_weight=None)
    unw_th_050 = compute_threshold_metrics(y_test_arr, prob_test, threshold=0.50, sample_weight=None)
    unw_th_opt = compute_threshold_metrics(y_test_arr, prob_test, threshold=sel_th, sample_weight=None)

    # 2. Survey-weighted test metrics
    wtd_disc = compute_discrimination_and_calibration(y_test_arr, prob_test, sample_weight=weights_test)
    wtd_th_050 = compute_threshold_metrics(y_test_arr, prob_test, threshold=0.50, sample_weight=weights_test)
    wtd_th_opt = compute_threshold_metrics(y_test_arr, prob_test, threshold=sel_th, sample_weight=weights_test)

    test_summary = [
        {
            "evaluation_mode": "Unweighted Sample Performance",
            "threshold": 0.50,
            "threshold_label": "Default Threshold (0.50)",
            "pr_auc": unw_disc["pr_auc"],
            "roc_auc": unw_disc["roc_auc"],
            "brier_score": unw_disc["brier_score"],
            "brier_skill_score": unw_disc["brier_skill_score"],
            "sensitivity": unw_th_050["sensitivity"],
            "specificity": unw_th_050["specificity"],
            "precision_ppv": unw_th_050["precision_ppv"],
            "npv": unw_th_050["npv"],
            "f1_score": unw_th_050["f1_score"],
            "balanced_accuracy": unw_th_050["balanced_accuracy"],
            "youden_j": unw_th_050["youden_j"],
        },
        {
            "evaluation_mode": "Unweighted Sample Performance",
            "threshold": sel_th,
            "threshold_label": f"Locked Optimal Threshold ({sel_th:.3f})",
            "pr_auc": unw_disc["pr_auc"],
            "roc_auc": unw_disc["roc_auc"],
            "brier_score": unw_disc["brier_score"],
            "brier_skill_score": unw_disc["brier_skill_score"],
            "sensitivity": unw_th_opt["sensitivity"],
            "specificity": unw_th_opt["specificity"],
            "precision_ppv": unw_th_opt["precision_ppv"],
            "npv": unw_th_opt["npv"],
            "f1_score": unw_th_opt["f1_score"],
            "balanced_accuracy": unw_th_opt["balanced_accuracy"],
            "youden_j": unw_th_opt["youden_j"],
        },
        {
            "evaluation_mode": "Survey-Weighted Population Estimates (_LLCPWT)",
            "threshold": 0.50,
            "threshold_label": "Default Threshold (0.50)",
            "pr_auc": wtd_disc["pr_auc"],
            "roc_auc": wtd_disc["roc_auc"],
            "brier_score": wtd_disc["brier_score"],
            "brier_skill_score": wtd_disc["brier_skill_score"],
            "sensitivity": wtd_th_050["sensitivity"],
            "specificity": wtd_th_050["specificity"],
            "precision_ppv": wtd_th_050["precision_ppv"],
            "npv": wtd_th_050["npv"],
            "f1_score": wtd_th_050["f1_score"],
            "balanced_accuracy": wtd_th_050["balanced_accuracy"],
            "youden_j": wtd_th_050["youden_j"],
        },
        {
            "evaluation_mode": "Survey-Weighted Population Estimates (_LLCPWT)",
            "threshold": sel_th,
            "threshold_label": f"Locked Optimal Threshold ({sel_th:.3f})",
            "pr_auc": wtd_disc["pr_auc"],
            "roc_auc": wtd_disc["roc_auc"],
            "brier_score": wtd_disc["brier_score"],
            "brier_skill_score": wtd_disc["brier_skill_score"],
            "sensitivity": wtd_th_opt["sensitivity"],
            "specificity": wtd_th_opt["specificity"],
            "precision_ppv": wtd_th_opt["precision_ppv"],
            "npv": wtd_th_opt["npv"],
            "f1_score": wtd_th_opt["f1_score"],
            "balanced_accuracy": wtd_th_opt["balanced_accuracy"],
            "youden_j": wtd_th_opt["youden_j"],
        }
    ]

    df_test = pd.DataFrame(test_summary)
    return df_test


def plot_validation_roc_curves(
    models_dict: Dict[str, Any],
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    output_path: Path
) -> None:
    """Generates Figure 10: Validation ROC Curves for all candidate models."""
    plt.figure(figsize=(8, 6), dpi=300)
    y_val_arr = y_val.values

    colors = {
        "logistic_regression_unweighted": "#1f77b4",
        "logistic_regression_balanced": "#aec7e8",
        "hist_gradient_boosting_unweighted": "#2ca02c",
        "hist_gradient_boosting_balanced": "#98df8a"
    }

    for model_key, clf in models_dict.items():
        prob = clf.predict_proba(X_val_trans)[:, 1]
        fpr, tpr, _ = roc_curve(y_val_arr, prob)
        auc = roc_auc_score(y_val_arr, prob)
        name = model_config.MODEL_CONFIGS[model_key]["model_name"]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=colors.get(model_key, None), lw=2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance Line (AUC = 0.5000)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Figure 10: Receiver Operating Characteristic (ROC) — Validation Set (N=67,870)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 10 to {output_path.name}")


def plot_validation_pr_curves(
    models_dict: Dict[str, Any],
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    output_path: Path
) -> None:
    """Generates Figure 11: Validation Precision-Recall Curves with Empirical Baseline."""
    plt.figure(figsize=(8, 6), dpi=300)
    y_val_arr = y_val.values
    prevalence = float(np.mean(y_val_arr))

    colors = {
        "logistic_regression_unweighted": "#1f77b4",
        "logistic_regression_balanced": "#aec7e8",
        "hist_gradient_boosting_unweighted": "#2ca02c",
        "hist_gradient_boosting_balanced": "#98df8a"
    }

    for model_key, clf in models_dict.items():
        prob = clf.predict_proba(X_val_trans)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val_arr, prob)
        ap = average_precision_score(y_val_arr, prob)
        name = model_config.MODEL_CONFIGS[model_key]["model_name"]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {ap:.4f})", color=colors.get(model_key, None), lw=2)

    plt.axhline(prevalence, color="red", linestyle=":", lw=1.5, label=f"Empirical Prevalence Reference ({prevalence*100:.2f}%)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    plt.title("Figure 11: Precision-Recall (PR) Curves — Validation Set (N=67,870)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 11 to {output_path.name}")


def plot_validation_calibration_curves(
    models_dict: Dict[str, Any],
    X_val_trans: np.ndarray,
    y_val: pd.Series,
    output_path: Path
) -> None:
    """Generates Figure 12: Validation Reliability / Calibration Diagrams."""
    plt.figure(figsize=(8, 6), dpi=300)
    y_val_arr = y_val.values

    colors = {
        "logistic_regression_unweighted": "#1f77b4",
        "logistic_regression_balanced": "#aec7e8",
        "hist_gradient_boosting_unweighted": "#2ca02c",
        "hist_gradient_boosting_balanced": "#98df8a"
    }

    for model_key, clf in models_dict.items():
        prob = clf.predict_proba(X_val_trans)[:, 1]
        brier = brier_score_loss(y_val_arr, prob)
        frac_pos, mean_pred = calibration_curve(y_val_arr, prob, n_bins=10, strategy="uniform")
        name = model_config.MODEL_CONFIGS[model_key]["model_name"]
        plt.plot(mean_pred, frac_pos, "s-", label=f"{name} (Brier = {brier:.4f})", color=colors.get(model_key, None), lw=2, markersize=5)

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect Calibration Diagonal")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Mean Predicted Probability (Decile Bin)", fontsize=11, fontweight="bold")
    plt.ylabel("Fraction of Positives (Empirical Risk)", fontsize=11, fontweight="bold")
    plt.title("Figure 12: Probability Calibration Curves — Validation Set (N=67,870)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 12 to {output_path.name}")


def plot_threshold_tradeoffs(
    threshold_df: pd.DataFrame,
    lock_payload: Dict[str, Any],
    output_path: Path
) -> None:
    """Generates Figure 13: Decision Threshold Trade-off Curves for the Champion Model."""
    plt.figure(figsize=(9, 6), dpi=300)
    th = threshold_df["threshold"]

    plt.plot(th, threshold_df["sensitivity"], label="Sensitivity (Recall)", color="#d62728", lw=2)
    plt.plot(th, threshold_df["specificity"], label="Specificity", color="#1f77b4", lw=2)
    plt.plot(th, threshold_df["f1_score"], label="F1 Score", color="#2ca02c", lw=2)
    plt.plot(th, threshold_df["precision_ppv"], label="Precision (PPV)", color="#ff7f0e", lw=2)
    plt.plot(th, threshold_df["youden_j"], label="Youden's J Index", color="#9467bd", linestyle="--", lw=1.8)

    # Highlight key thresholds
    opt_youden = lock_payload["candidate_thresholds"]["optimal_youden_j"]
    opt_f1 = lock_payload["candidate_thresholds"]["optimal_f1"]
    ref_prev = lock_payload["candidate_thresholds"]["empirical_prevalence_ref"]

    plt.axvline(opt_youden, color="#9467bd", linestyle=":", lw=2, label=f"Locked Youden Threshold ({opt_youden:.3f})")
    plt.axvline(opt_f1, color="#2ca02c", linestyle=":", lw=1.5, alpha=0.8, label=f"Optimal F1 ({opt_f1:.3f})")
    plt.axvline(0.50, color="gray", linestyle="-.", lw=1.2, alpha=0.7, label="Default Threshold (0.50)")
    plt.axvline(ref_prev, color="red", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Ref ({ref_prev:.3f})")

    plt.xlim([0.0, 0.90])
    plt.ylim([0.0, 1.02])
    plt.xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.title(f"Figure 13: Decision Threshold Trade-offs — {lock_payload['champion_model_name']}", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="center right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 13 to {output_path.name}")


def plot_test_confusion_matrices(
    y_test: pd.Series,
    prob_test: np.ndarray,
    lock_payload: Dict[str, Any],
    output_path: Path
) -> None:
    """Generates Figure 14: Side-by-side Confusion Matrices on Holdout Test Set."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    y_test_arr = y_test.values
    sel_th = float(lock_payload["selected_operational_threshold"])

    # 1. Default 0.50 Threshold
    cm_050 = confusion_matrix(y_test_arr, (prob_test >= 0.50).astype(int))
    sns.heatmap(cm_050, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Negative (No CAD)", "Positive (CAD)"],
                yticklabels=["Negative (No CAD)", "Positive (CAD)"])
    axes[0].set_title("Test Confusion Matrix: Default Threshold (0.50)\nSensitivity: {:.1f}% | Specificity: {:.1f}%".format(
        cm_050[1, 1] / (cm_050[1, 1] + cm_050[1, 0]) * 100,
        cm_050[0, 0] / (cm_050[0, 0] + cm_050[0, 1]) * 100
    ), fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontweight="bold")
    axes[0].set_ylabel("True Label", fontweight="bold")

    # 2. Locked Optimal Threshold
    cm_opt = confusion_matrix(y_test_arr, (prob_test >= sel_th).astype(int))
    sns.heatmap(cm_opt, annot=True, fmt=",d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Negative (No CAD)", "Positive (CAD)"],
                yticklabels=["Negative (No CAD)", "Positive (CAD)"])
    axes[1].set_title(f"Test Confusion Matrix: Locked Optimal Threshold ({sel_th:.3f})\nSensitivity: " +
                      "{:.1f}% | Specificity: {:.1f}%".format(
                          cm_opt[1, 1] / (cm_opt[1, 1] + cm_opt[1, 0]) * 100,
                          cm_opt[0, 0] / (cm_opt[0, 0] + cm_opt[0, 1]) * 100
                      ), fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontweight="bold")
    axes[1].set_ylabel("True Label", fontweight="bold")

    plt.suptitle(f"Figure 14: Holdout Test Set Confusion Matrices (N = 67,870)\n{lock_payload['champion_model_name']}",
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 14 to {output_path.name}")


def plot_unweighted_vs_survey_weighted(
    test_summary_df: pd.DataFrame,
    output_path: Path
) -> None:
    """Generates Figure 15: Comparison of Unweighted Sample vs Survey-Weighted Test Performance."""
    plt.figure(figsize=(10, 6), dpi=300)

    # Filter for the optimal threshold rows
    df_opt = test_summary_df[test_summary_df["threshold_label"].str.contains("Locked Optimal")].copy()

    metrics = ["sensitivity", "specificity", "precision_ppv", "npv", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc"]
    metric_labels = ["Sensitivity", "Specificity", "Precision", "NPV", "F1 Score", "Balanced Acc", "ROC-AUC", "PR-AUC"]

    unw_row = df_opt[df_opt["evaluation_mode"].str.contains("Unweighted")].iloc[0]
    wtd_row = df_opt[df_opt["evaluation_mode"].str.contains("Survey-Weighted")].iloc[0]

    unw_vals = [unw_row[m] for m in metrics]
    wtd_vals = [wtd_row[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, unw_vals, width, label="Unweighted Sample (N=67,870)", color="#1f77b4", alpha=0.85)
    plt.bar(x + width/2, wtd_vals, width, label="Survey-Weighted Population (_LLCPWT)", color="#ff7f0e", alpha=0.85)

    plt.xticks(x, metric_labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.title("Figure 15: Unweighted Sample vs. Survey-Weighted Population Estimates (Holdout Test)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=9.5)

    for i in range(len(metrics)):
        plt.text(x[i] - width/2, unw_vals[i] + 0.02, f"{unw_vals[i]:.2f}", ha="center", fontsize=8, color="#1f77b4")
        plt.text(x[i] + width/2, wtd_vals[i] + 0.02, f"{wtd_vals[i]:.2f}", ha="center", fontsize=8, color="#d95f02")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 15 to {output_path.name}")


def run_milestone6_evaluation():
    """
    Main evaluation workflow:
    1. Load validation data and transform via frozen preprocessor.
    2. Load trained candidate models.
    3. Evaluate models on validation set strictly and rank by PR-AUC.
    4. Save model_validation_comparison.csv and validation figures (Fig 10, 11, 12).
    5. Evaluate decision thresholds on champion and save threshold_optimization_analysis.csv and Fig 13.
    6. Formally lock champion model decision to models/model_selection_summary.json.
    7. ONLY AFTER LOCK: Load holdout test data, transform via frozen preprocessor.
    8. Execute single-pass test evaluation, save final_test_evaluation_summary.csv and Fig 14, 15.
    """
    data_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    preprocessor_path = models_dir / "preprocessor.joblib"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation data and cohort metadata
    print("Loading validation data and cohort metadata...")
    cohort_meta = pd.read_csv(data_dir / "cohort_metadata.csv.gz", index_col=0)
    X_val = pd.read_csv(data_dir / "validation_features.csv.gz", index_col=0)
    y_val = pd.read_csv(data_dir / "validation_target.csv.gz", index_col=0).squeeze("columns")
    meta_val = cohort_meta.loc[X_val.index]

    # 2. Transform validation data using frozen preprocessor
    print("Transforming validation data via frozen preprocessor...")
    preprocessor = joblib.load(preprocessor_path)
    X_val_trans = preprocessor.transform(X_val)
    assert X_val_trans.shape[1] == 190, f"Expected 190 features, got {X_val_trans.shape[1]}"

    # 3. Load trained models
    print("Loading candidate models...")
    models_dict = {
        "logistic_regression_unweighted": joblib.load(models_dir / "baseline_logistic_regression.joblib"),
        "logistic_regression_balanced": joblib.load(models_dir / "weighted_logistic_regression.joblib"),
        "hist_gradient_boosting_unweighted": joblib.load(models_dir / "improved_hist_gradient_boosting.joblib"),
        "hist_gradient_boosting_balanced": joblib.load(models_dir / "weighted_hist_gradient_boosting.joblib"),
    }

    # 4. Evaluate models on validation data
    val_summary_df = evaluate_validation_models(models_dict, X_val_trans, y_val, meta_val)
    val_table_path = tables_dir / "model_validation_comparison.csv"
    val_summary_df.to_csv(val_table_path, index=False)
    print(f"Saved validation comparison table to {val_table_path.name}")
    print("\n--- VALIDATION MODEL COMPARISON (Ranked by PR-AUC) ---")
    print(val_summary_df[["model_key", "val_pr_auc", "val_roc_auc", "val_brier_score", "sens_at_050", "spec_at_050", "f1_at_050"]].to_string(index=False))

    # 5. Generate validation figures
    plot_validation_roc_curves(models_dict, X_val_trans, y_val, figures_dir / "fig10_roc_curves_validation.png")
    plot_validation_pr_curves(models_dict, X_val_trans, y_val, figures_dir / "fig11_pr_curves_validation.png")
    plot_validation_calibration_curves(models_dict, X_val_trans, y_val, figures_dir / "fig12_calibration_curves_validation.png")

    # 6. Threshold optimization on champion model (top ranked by PR-AUC)
    champion_key = val_summary_df.iloc[0]["model_key"]
    champion_clf = models_dict[champion_key]
    print(f"\nAnalyzing decision thresholds on validation set for Champion: {champion_key}...")
    threshold_df = optimize_thresholds(champion_clf, X_val_trans, y_val)
    th_table_path = tables_dir / "threshold_optimization_analysis.csv"
    threshold_df.to_csv(th_table_path, index=False)
    print(f"Saved threshold optimization table to {th_table_path.name}")

    # 7. Lock champion model and operational threshold
    lock_path = models_dir / "model_selection_summary.json"
    lock_payload = lock_champion_model_decision(val_summary_df, threshold_df, lock_path)

    # Plot threshold tradeoffs
    plot_threshold_tradeoffs(threshold_df, lock_payload, figures_dir / "fig13_threshold_tradeoffs.png")

    # 8. NOW AND ONLY NOW: Holdout Test Set Evaluation
    print("\nQuarantine released for test partition. Proceeding with single-pass test evaluation...")
    X_test = pd.read_csv(data_dir / "test_features.csv.gz", index_col=0)
    y_test = pd.read_csv(data_dir / "test_target.csv.gz", index_col=0).squeeze("columns")
    meta_test = cohort_meta.loc[X_test.index]

    print("Transforming test data via frozen preprocessor...")
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape[1] == 190, f"Expected 190 features, got {X_test_trans.shape[1]}"

    test_summary_df = evaluate_locked_model_on_test(champion_clf, lock_payload, X_test_trans, y_test, meta_test)
    test_table_path = tables_dir / "final_test_evaluation_summary.csv"
    test_summary_df.to_csv(test_table_path, index=False)
    print(f"Saved final test evaluation summary to {test_table_path.name}")
    print("\n--- FINAL TEST EVALUATION SUMMARY ---")
    print(test_summary_df[["evaluation_mode", "threshold_label", "pr_auc", "roc_auc", "sensitivity", "specificity", "precision_ppv", "f1_score", "brier_score"]].to_string(index=False))

    # 9. Test figures
    prob_test = champion_clf.predict_proba(X_test_trans)[:, 1]
    plot_test_confusion_matrices(y_test, prob_test, lock_payload, figures_dir / "fig14_champion_test_confusion_matrices.png")
    plot_unweighted_vs_survey_weighted(test_summary_df, figures_dir / "fig15_unweighted_vs_survey_weighted_test.png")

    print("\nMilestone 6 Evaluation Pipeline completed successfully!")


if __name__ == "__main__":
    import joblib
    run_milestone6_evaluation()
