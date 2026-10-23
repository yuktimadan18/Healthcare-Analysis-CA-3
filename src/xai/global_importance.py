"""
global_importance.py
--------------------
Global Explainable AI (XAI) engine for Milestone 7:
1. Grouped permutation importance across all 38 original survey predictors.
2. Transformed-indicator permutation importance across all 190 one-hot/numeric features.
3. Partial dependence profiles for continuous predictors unscaled to natural survey units.
4. Publication-grade visualization generation.
"""

from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.inspection import partial_dependence
from scipy.special import expit

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import xai_config
from src.xai.validate_feature_mapping import extract_and_validate_feature_mapping


def load_validation_data_and_model() -> Tuple[Any, Any, np.ndarray, np.ndarray, pd.DataFrame]:
    """Loads frozen champion model, preprocessor, validation partition, and mapping."""
    print("Loading champion model and preprocessor...")
    model = joblib.load(xai_config.CHAMPION_MODEL_PATH)
    preprocessor = joblib.load(xai_config.PREPROCESSOR_PATH)

    print(f"Loading validation partition from: {xai_config.VALIDATION_FEATURES_PATH}")
    X_val_raw = pd.read_csv(xai_config.VALIDATION_FEATURES_PATH, index_col=0)
    y_val = pd.read_csv(xai_config.VALIDATION_TARGET_PATH, index_col=0).squeeze("columns").values

    print("Transforming validation features via frozen preprocessor (transform ONLY)...")
    X_val_trans = preprocessor.transform(X_val_raw)
    assert X_val_trans.shape[1] == 190, f"Expected 190 features, got {X_val_trans.shape[1]}"

    # Load validated mapping table
    if xai_config.FEATURE_MAPPING_AUDIT_PATH.exists():
        df_mapping = pd.read_csv(xai_config.FEATURE_MAPPING_AUDIT_PATH)
    else:
        _, df_mapping = extract_and_validate_feature_mapping()

    return model, preprocessor, X_val_trans, y_val, df_mapping


def compute_baseline_metrics(model: Any, X_val_trans: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    """Computes baseline PR-AUC and ROC-AUC on unpermuted validation data."""
    probs = model.predict_proba(X_val_trans)[:, 1]
    baseline_pr = float(average_precision_score(y_val, probs))
    baseline_roc = float(roc_auc_score(y_val, probs))
    print(f"Validation Baseline PR-AUC:  {baseline_pr:.6f}")
    print(f"Validation Baseline ROC-AUC: {baseline_roc:.6f}")
    return baseline_pr, baseline_roc


def compute_grouped_permutation_importance(
    model: Any,
    X_val_trans: np.ndarray,
    y_val: np.ndarray,
    df_mapping: pd.DataFrame,
    baseline_pr: float,
    baseline_roc: float,
    n_repeats: int = xai_config.PERMUTATION_REPEATS,
    random_state: int = xai_config.RANDOM_STATE,
    n_jobs: int = 6
) -> pd.DataFrame:
    """
    Computes grouped permutation importance for all 38 original predictors.
    For each original predictor, simultaneously permutes all associated dummy columns.
    """
    print(f"\n--- Computing Grouped Permutation Importance (38 Original Predictors, {n_repeats} repeats) ---")
    t0 = time.time()

    # Build group dictionary: source_variable -> list of column indices
    grouped_indices: Dict[str, List[int]] = {}
    var_types: Dict[str, str] = {}
    for var, group in df_mapping.groupby("source_variable"):
        grouped_indices[var] = group["feature_index"].tolist()
        var_types[var] = group["feature_type"].iloc[0]

    n_samples = len(X_val_trans)

    def evaluate_worker(repeat_idx: int) -> Dict[str, Tuple[float, float]]:
        """Worker function for repeat_idx."""
        rng = np.random.RandomState(random_state + repeat_idx)
        perm = rng.permutation(n_samples)
        worker_X = X_val_trans.copy()
        repeat_drops = {}

        for var, cols in grouped_indices.items():
            orig_slice = worker_X[:, cols].copy()
            worker_X[:, cols] = X_val_trans[perm][:, cols]
            probs_perm = model.predict_proba(worker_X)[:, 1]
            worker_X[:, cols] = orig_slice

            drop_pr = baseline_pr - average_precision_score(y_val, probs_perm)
            drop_roc = baseline_roc - roc_auc_score(y_val, probs_perm)
            repeat_drops[var] = (drop_pr, drop_roc)

        return repeat_drops

    # Execute repeats in parallel
    results_list = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(evaluate_worker)(r) for r in range(n_repeats)
    )

    # Aggregate across repeats
    rows = []
    for var in grouped_indices.keys():
        pr_drops = [res[var][0] for res in results_list]
        roc_drops = [res[var][1] for res in results_list]

        rows.append({
            "source_variable": var,
            "feature_type": var_types[var],
            "num_transformed_columns": len(grouped_indices[var]),
            "mean_pr_auc_drop": float(np.mean(pr_drops)),
            "std_pr_auc_drop": float(np.std(pr_drops, ddof=1)),
            "mean_roc_auc_drop": float(np.mean(roc_drops)),
            "std_roc_auc_drop": float(np.std(roc_drops, ddof=1)),
            "baseline_pr_auc": baseline_pr,
            "baseline_roc_auc": baseline_roc
        })

    df_grouped = pd.DataFrame(rows)
    # Primary sort by PR-AUC drop descending
    df_grouped = df_grouped.sort_values(by="mean_pr_auc_drop", ascending=False).reset_index(drop=True)
    df_grouped.insert(0, "rank", range(1, len(df_grouped) + 1))

    elapsed = time.time() - t0
    print(f"Grouped permutation importance computed in {elapsed:.2f} s.")
    return df_grouped


def compute_indicator_permutation_importance(
    model: Any,
    X_val_trans: np.ndarray,
    y_val: np.ndarray,
    df_mapping: pd.DataFrame,
    baseline_pr: float,
    baseline_roc: float,
    n_repeats: int = xai_config.PERMUTATION_REPEATS,
    random_state: int = xai_config.RANDOM_STATE,
    n_jobs: int = 6
) -> pd.DataFrame:
    """
    Computes individual indicator-level permutation importance across all 190 transformed features.
    """
    print(f"\n--- Computing Indicator-Level Permutation Importance (190 Transformed Features, {n_repeats} repeats) ---")
    t0 = time.time()
    n_samples = len(X_val_trans)

    def evaluate_worker(repeat_idx: int) -> Dict[int, Tuple[float, float]]:
        rng = np.random.RandomState(random_state + repeat_idx)
        perm = rng.permutation(n_samples)
        worker_X = X_val_trans.copy()
        repeat_drops = {}

        for col_idx in range(X_val_trans.shape[1]):
            orig_col = worker_X[:, col_idx].copy()
            worker_X[:, col_idx] = X_val_trans[perm, col_idx]
            probs_perm = model.predict_proba(worker_X)[:, 1]
            worker_X[:, col_idx] = orig_col

            drop_pr = baseline_pr - average_precision_score(y_val, probs_perm)
            drop_roc = baseline_roc - roc_auc_score(y_val, probs_perm)
            repeat_drops[col_idx] = (drop_pr, drop_roc)

        return repeat_drops

    results_list = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(evaluate_worker)(r) for r in range(n_repeats)
    )

    rows = []
    for _, mapping_row in df_mapping.iterrows():
        col_idx = int(mapping_row["feature_index"])
        pr_drops = [res[col_idx][0] for res in results_list]
        roc_drops = [res[col_idx][1] for res in results_list]

        rows.append({
            "transformed_feature_name": mapping_row["transformed_feature_name"],
            "source_variable": mapping_row["source_variable"],
            "feature_type": mapping_row["feature_type"],
            "category_value": mapping_row["category_value"],
            "is_missing_sentinel": mapping_row["is_missing_sentinel"],
            "mean_pr_auc_drop": float(np.mean(pr_drops)),
            "std_pr_auc_drop": float(np.std(pr_drops, ddof=1)),
            "mean_roc_auc_drop": float(np.mean(roc_drops)),
            "std_roc_auc_drop": float(np.std(roc_drops, ddof=1)),
            "baseline_pr_auc": baseline_pr,
            "baseline_roc_auc": baseline_roc
        })

    df_indicators = pd.DataFrame(rows)
    df_indicators = df_indicators.sort_values(by="mean_pr_auc_drop", ascending=False).reset_index(drop=True)
    df_indicators.insert(0, "rank", range(1, len(df_indicators) + 1))

    elapsed = time.time() - t0
    print(f"Indicator-level permutation importance computed in {elapsed:.2f} s.")
    return df_indicators


def compute_and_plot_partial_dependence(
    model: Any,
    X_val_trans: np.ndarray,
    preprocessor: Any,
    output_path: Path
) -> None:
    """
    Generates partial-dependence profiles for _AGE80, PHYSHLTH_DAYS, and BMI_NUMERIC,
    unscaling grid values to their natural survey units.
    """
    print("\n--- Computing Partial Dependence Profiles for Numeric Predictors ---")
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    # Target continuous features in preprocessor: (col_idx_in_transformed, variable_name, scaler_idx, unit_label)
    numeric_specs = [
        (0, "_AGE80", 0, "Age in Years (18 - 80+)", "Panel A: Age Profile (_AGE80)"),
        (1, "PHYSHLTH_DAYS", 1, "Physically Unhealthy Days (0 - 30)", "Panel B: Physical Health Limitation (PHYSHLTH_DAYS)"),
        (3, "BMI_NUMERIC", 3, "Body Mass Index (kg/m²)", "Panel C: Body Mass Index (BMI_NUMERIC)")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    for i, (col_idx, name, scaler_idx, unit_label, title) in enumerate(numeric_specs):
        pd_res = partial_dependence(
            model,
            X_val_trans,
            features=[col_idx],
            response_method="decision_function",
            grid_resolution=50,
            method="recursion"
        )

        grid_scaled = pd_res["grid_values"][0]
        grid_unscaled = grid_scaled * scaler.scale_[scaler_idx] + scaler.mean_[scaler_idx]

        # Map decision function log-odds to predicted probability via logistic sigmoid
        marginal_prob = expit(pd_res["average"][0])

        ax = axes[i]
        ax.plot(grid_unscaled, marginal_prob, color="#1f77b4", lw=2.5, label="Partial Dependence (Marginal Prob)")
        ax.axhline(0.0936, color="red", linestyle=":", lw=1.5, alpha=0.8, label="Cohort Prevalence (9.36%)")
        ax.axhline(0.0800, color="purple", linestyle="--", lw=1.5, alpha=0.8, label="Screening Threshold (0.080)")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel(unit_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Marginal Predicted Probability", fontsize=10, fontweight="bold")
        ax.set_ylim([0.0, max(0.25, marginal_prob.max() * 1.15)])
        ax.legend(loc="upper left" if i != 0 else "lower right", fontsize=8.5, frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Figure 18: Partial Dependence Profiles for Key Numeric Predictors (Validation Set, N=67,870)\n"
        "[Note: Curves describe statistical model behavior across empirical ranges; they do not establish causal intervention effects.]",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 18 to: {output_path.name}")


def plot_global_variable_importance(
    df_grouped: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
) -> None:
    """Generates Figure 16: Top original survey variables ranked by PR-AUC degradation with error bars."""
    plt.figure(figsize=(10, 8), dpi=300)
    top_df = df_grouped.head(top_n).iloc[::-1]  # Invert so rank 1 is at top

    y_pos = np.arange(len(top_df))
    bars = plt.barh(
        y_pos,
        top_df["mean_pr_auc_drop"],
        xerr=top_df["std_pr_auc_drop"],
        color="#2ca02c",
        alpha=0.85,
        edgecolor="#1b611b",
        capsize=3.5,
        height=0.65
    )

    labels = [
        f"{row['source_variable']} ({row['num_transformed_columns']} cols)"
        for _, row in top_df.iterrows()
    ]
    plt.yticks(y_pos, labels, fontsize=9.5, fontweight="bold")
    plt.xlabel("Mean Validation PR-AUC Decrease (± 1 SD across 10 repeats)", fontsize=10.5, fontweight="bold")
    plt.title(
        f"Figure 16: Top {top_n} Original Survey Predictors Ranked by Grouped Permutation Importance\n"
        "(Validation Partition, N=67,870 | Primary Metric: PR-AUC Decrease | Seed: 42)",
        fontsize=11.5, fontweight="bold", pad=12
    )

    # Add numeric drop labels next to bars
    for i, bar in enumerate(bars):
        val = top_df.iloc[i]["mean_pr_auc_drop"]
        plt.text(val + 0.001, y_pos[i], f"{val:.4f}", va="center", fontsize=8.5, color="#1b611b")

    plt.xlim(left=min(0.0, top_df["mean_pr_auc_drop"].min() - 0.005))
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 16 to: {output_path.name}")


def plot_indicator_importance(
    df_indicators: pd.DataFrame,
    output_path: Path,
    top_n: int = 25
) -> None:
    """Generates Figure 17: Top individual transformed one-hot/numeric features ranked by PR-AUC drop."""
    plt.figure(figsize=(11, 9), dpi=300)
    top_df = df_indicators.head(top_n).iloc[::-1]

    y_pos = np.arange(len(top_df))
    colors = ["#1f77b4" if row["feature_type"] == "numeric" else "#ff7f0e" for _, row in top_df.iterrows()]

    plt.barh(
        y_pos,
        top_df["mean_pr_auc_drop"],
        xerr=top_df["std_pr_auc_drop"],
        color=colors,
        alpha=0.85,
        capsize=3.0,
        height=0.65
    )

    labels = [
        f"{row['transformed_feature_name']} [{row['source_variable']}]"
        for _, row in top_df.iterrows()
    ]
    plt.yticks(y_pos, labels, fontsize=8.5)
    plt.xlabel("Mean Validation PR-AUC Decrease (± 1 SD across 10 repeats)", fontsize=10.5, fontweight="bold")
    plt.title(
        f"Figure 17: Top {top_n} Transformed Indicator Features by Permutation Importance\n"
        "(Blue: Continuous Numeric | Orange: Categorical Dummy | Validation Partition, N=67,870)",
        fontsize=11.5, fontweight="bold", pad=12
    )

    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 17 to: {output_path.name}")


def run_global_importance_pipeline():
    """Main pipeline executing Stage 2 global explainable AI."""
    print("=================================================================")
    print("  Milestone 7 — Stage 2: Global Explainable AI Execution Pipeline")
    print("=================================================================")

    # 1. Load data, model, and mapping
    model, preprocessor, X_val_trans, y_val, df_mapping = load_validation_data_and_model()

    # 2. Compute baseline performance
    baseline_pr, baseline_roc = compute_baseline_metrics(model, X_val_trans, y_val)

    # 3. Compute grouped permutation importance (38 original predictors)
    df_grouped = compute_grouped_permutation_importance(
        model, X_val_trans, y_val, df_mapping, baseline_pr, baseline_roc,
        n_repeats=xai_config.PERMUTATION_REPEATS, random_state=xai_config.RANDOM_STATE
    )
    xai_config.GLOBAL_VARIABLE_IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_grouped.to_csv(xai_config.GLOBAL_VARIABLE_IMPORTANCE_PATH, index=False)
    print(f"Saved grouped variable importance to: {xai_config.GLOBAL_VARIABLE_IMPORTANCE_PATH.name}")

    print("\n--- Top 10 Original Predictors by Mean PR-AUC Decrease ---")
    print(df_grouped[["rank", "source_variable", "feature_type", "num_transformed_columns", "mean_pr_auc_drop", "std_pr_auc_drop", "mean_roc_auc_drop"]].head(10).to_string(index=False))

    # 4. Compute indicator-level permutation importance (190 transformed features)
    df_indicators = compute_indicator_permutation_importance(
        model, X_val_trans, y_val, df_mapping, baseline_pr, baseline_roc,
        n_repeats=xai_config.PERMUTATION_REPEATS, random_state=xai_config.RANDOM_STATE
    )
    df_indicators.to_csv(xai_config.TRANSFORMED_FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Saved indicator feature importance to: {xai_config.TRANSFORMED_FEATURE_IMPORTANCE_PATH.name}")

    print("\n--- Top 10 Transformed Indicators by Mean PR-AUC Decrease ---")
    print(df_indicators[["rank", "transformed_feature_name", "source_variable", "category_value", "mean_pr_auc_drop", "std_pr_auc_drop", "mean_roc_auc_drop"]].head(10).to_string(index=False))

    # 5. Generate Visualizations
    plot_global_variable_importance(df_grouped, xai_config.FIG_GLOBAL_VARIABLE_IMPORTANCE, top_n=20)
    plot_indicator_importance(df_indicators, xai_config.FIG_INDICATOR_FEATURE_IMPORTANCE, top_n=25)
    compute_and_plot_partial_dependence(model, X_val_trans, preprocessor, xai_config.FIG_PARTIAL_DEPENDENCE_PROFILES)

    print("\nStage 2 Global XAI Pipeline completed successfully!")


if __name__ == "__main__":
    run_global_importance_pipeline()
