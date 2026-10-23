"""
local_explanations.py
---------------------
Local Explainable AI (XAI) engine for Milestone 7:
1. Deterministic, objective selection of validation case archetypes (TP, FP, TN, FN)
   at the locked operational threshold (0.080).
2. Transparent, model-agnostic local marginal perturbation attribution (Delta p)
   evaluating feature contributions relative to an empirical validation background.
3. Local sparse linear surrogate regression (Ridge LIME-equivalent) on perturbed
   neighborhoods with exponential kernel weighting and local fidelity (R^2) reporting.
4. Mapping of transformed dummy features back to the 38 original survey variables.
5. Generation of structured CSV explanation tables and 4-panel publication visualization.
"""

from pathlib import Path
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import xai_config
from src.xai.validate_feature_mapping import extract_and_validate_feature_mapping


def load_validation_data_and_champion() -> Tuple[Any, Any, float, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Loads frozen champion model, preprocessor, locked threshold, validation partition,
    and feature mapping audit table.
    """
    print("Loading champion model and preprocessor...")
    model = joblib.load(xai_config.CHAMPION_MODEL_PATH)
    preprocessor = joblib.load(xai_config.PREPROCESSOR_PATH)

    # Read locked threshold dynamically
    threshold = xai_config.get_locked_threshold()
    print(f"Loaded locked operational threshold: {threshold:.4f}")

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

    return model, preprocessor, threshold, X_val_trans, y_val, X_val_raw, df_mapping


def select_representative_validation_cases(
    model: Any,
    X_val_trans: np.ndarray,
    y_val: np.ndarray,
    X_val_raw: pd.DataFrame,
    threshold: float = 0.080
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Selects representative validation cases for all 4 confusion matrix quadrants:
    True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN).

    Selection rule: For each quadrant, deterministically selects the case closest
    to the median predicted probability of that quadrant, avoiding arbitrary cherry-picking.
    """
    probs = model.predict_proba(X_val_trans)[:, 1]
    preds = (probs >= threshold).astype(int)

    quadrants = {
        "True Positive (TP)": {
            "indices": np.where((y_val == 1) & (preds == 1))[0],
            "archetype_role": "High-risk respondent with reported CAD/MI history correctly flagged by screening model."
        },
        "False Positive (FP)": {
            "indices": np.where((y_val == 0) & (preds == 1))[0],
            "archetype_role": "Screening alert: Elevated risk profile above threshold without reported CAD/MI diagnosis."
        },
        "True Negative (TN)": {
            "indices": np.where((y_val == 0) & (preds == 0))[0],
            "archetype_role": "Low-risk baseline respondent correctly classified below screening threshold."
        },
        "False Negative (FN)": {
            "indices": np.where((y_val == 1) & (preds == 0))[0],
            "archetype_role": "Under-detected respondent with reported CAD/MI history presenting low conventional risk scores."
        }
    }

    selected_cases: Dict[str, int] = {}
    summary_rows = []

    for name, data in quadrants.items():
        idxs = data["indices"]
        assert len(idxs) > 0, f"No cases found for quadrant: {name}"
        cat_probs = probs[idxs]
        median_prob = float(np.median(cat_probs))

        # Pick index closest to median
        best_local_idx = int(np.argmin(np.abs(cat_probs - median_prob)))
        best_val_idx = int(idxs[best_local_idx])
        selected_cases[name] = best_val_idx

        row_id = int(X_val_raw.index[best_val_idx])
        p_val = float(probs[best_val_idx])
        actual_y = int(y_val[best_val_idx])
        pred_y = int(preds[best_val_idx])
        raw_row = X_val_raw.iloc[best_val_idx]

        summary_rows.append({
            "category": name,
            "case_id": f"VALIDATION_ID_{row_id}",
            "validation_row_index": best_val_idx,
            "actual_target": actual_y,
            "predicted_probability": round(p_val, 4),
            "locked_threshold": round(threshold, 4),
            "predicted_class": pred_y,
            "age_years": float(raw_row["_AGE80"]),
            "general_health_code": float(raw_row["GENHLTH"]),
            "sex_code": float(raw_row["_SEX"]),
            "stroke_code": float(raw_row["CVDSTRK3"]),
            "physical_unhealthy_days": float(raw_row["PHYSHLTH_DAYS"]),
            "bmi_numeric": float(raw_row["BMI_NUMERIC"]),
            "archetype_description": data["archetype_role"]
        })

    df_cases_summary = pd.DataFrame(summary_rows)
    return selected_cases, df_cases_summary


def compute_case_explanations(
    model: Any,
    X_val_trans: np.ndarray,
    selected_cases: Dict[str, int],
    df_mapping: pd.DataFrame,
    threshold: float = 0.080,
    n_background: int = 200,
    n_surrogate_samples: int = 1000,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Computes local explanations for each selected case using:
    1. Grouped marginal perturbation attribution (Delta p) relative to an empirical
       validation background sample.
    2. Local Ridge linear surrogate weights on perturbed neighborhood with exponential kernel.
    """
    rng = np.random.RandomState(random_state)
    n_val = len(X_val_trans)

    # Sample background reference partition from validation set
    bg_indices = rng.choice(n_val, size=n_background, replace=False)
    X_bg = X_val_trans[bg_indices]

    # Map source variables to transformed column indices
    grouped_indices: Dict[str, List[int]] = {}
    var_types: Dict[str, str] = {}
    for var, grp in df_mapping.groupby("source_variable"):
        grouped_indices[var] = grp["feature_index"].tolist()
        var_types[var] = grp["feature_type"].iloc[0]

    all_explanation_records = []
    case_details: Dict[str, Dict[str, Any]] = {}

    for cat_name, row_idx in selected_cases.items():
        x_target = X_val_trans[row_idx:row_idx + 1]
        p_target = float(model.predict_proba(x_target)[0, 1])

        # -------------------------------------------------------------
        # 1. Grouped Marginal Perturbation Attribution (Delta p)
        # -------------------------------------------------------------
        var_marginal_deltas: Dict[str, float] = {}
        for var, cols in grouped_indices.items():
            # Tile target respondent N_bg times and substitute cols with background sample
            X_pert = np.repeat(x_target, n_background, axis=0)
            X_pert[:, cols] = X_bg[:, cols]
            expected_p = float(np.mean(model.predict_proba(X_pert)[:, 1]))
            # Delta p: positive means feature elevates risk above population background expectation
            var_marginal_deltas[var] = p_target - expected_p

        # -------------------------------------------------------------
        # 2. Local Linear Surrogate (Ridge LIME-Equivalent)
        # -------------------------------------------------------------
        # Perturb respondent by probabilistically replacing features with background values
        surrogate_rng = np.random.RandomState(random_state + row_idx % 1000)
        rand_bg_idx = surrogate_rng.choice(n_val, size=n_surrogate_samples, replace=True)
        mask = surrogate_rng.binomial(1, p=0.15, size=(n_surrogate_samples, X_val_trans.shape[1]))
        Z = np.where(mask == 1, X_val_trans[rand_bg_idx], x_target)

        probs_z = model.predict_proba(Z)[:, 1]
        dists = np.sqrt(np.sum((Z - x_target) ** 2, axis=1))
        kernel_width = float(np.median(dists))
        if kernel_width == 0:
            kernel_width = 1.0
        weights = np.exp(-(dists ** 2) / (2 * kernel_width ** 2))

        ridge = Ridge(alpha=1.0, random_state=random_state)
        ridge.fit(Z, probs_z, sample_weight=weights)
        surrogate_r2 = float(ridge.score(Z, probs_z, sample_weight=weights))

        # Map surrogate coefficients to source variables (sum of absolute weights or max)
        surrogate_coefs = ridge.coef_
        var_surrogate_weights: Dict[str, float] = {}
        for var, cols in grouped_indices.items():
            # Sum signed coefficients for that group
            var_surrogate_weights[var] = float(np.sum(surrogate_coefs[cols]))

        # Store case details for plotting
        case_details[cat_name] = {
            "validation_row_idx": row_idx,
            "predicted_prob": p_target,
            "threshold": threshold,
            "surrogate_r2": surrogate_r2,
            "marginal_deltas": var_marginal_deltas,
            "surrogate_weights": var_surrogate_weights
        }

        # Format records for structured CSV export
        for var in grouped_indices.keys():
            delta = var_marginal_deltas[var]
            weight = var_surrogate_weights[var]
            impact = "Risk-Elevating" if delta > 0.001 else ("Risk-Mitigating" if delta < -0.001 else "Neutral")

            all_explanation_records.append({
                "case_category": cat_name,
                "validation_row_index": row_idx,
                "predicted_probability": round(p_target, 6),
                "locked_threshold": round(threshold, 4),
                "source_variable": var,
                "feature_type": var_types[var],
                "num_transformed_columns": len(grouped_indices[var]),
                "marginal_risk_delta_dp": round(delta, 6),
                "local_surrogate_weight": round(weight, 6),
                "directional_impact": impact,
                "surrogate_fidelity_r2": round(surrogate_r2, 4)
            })

    df_explanations = pd.DataFrame(all_explanation_records)
    return df_explanations, case_details


def plot_local_case_attributions(
    case_details: Dict[str, Dict[str, Any]],
    df_cases_summary: pd.DataFrame,
    output_path: Path,
    top_k: int = 8
) -> None:
    """
    Generates Figure 19: 4-panel publication visualization comparing local feature
    attributions across the True Positive, False Positive, True Negative, and False Negative
    representative archetypes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    axes_flat = axes.flatten()

    category_keys = [
        "True Positive (TP)",
        "False Positive (FP)",
        "False Negative (FN)",
        "True Negative (TN)"
    ]

    palette_elevating = "#d62728"  # Crimson red for risk-elevating
    palette_mitigating = "#1f77b4" # Steel blue for risk-mitigating

    for i, cat_name in enumerate(category_keys):
        ax = axes_flat[i]
        details = case_details[cat_name]
        case_info = df_cases_summary[df_cases_summary["category"] == cat_name].iloc[0]

        deltas = pd.Series(details["marginal_deltas"])
        # Select top k by absolute magnitude
        top_vars = deltas.abs().sort_values(ascending=False).head(top_k).index
        sub_deltas = deltas[top_vars].sort_values(ascending=True)  # ascending for barh

        y_pos = np.arange(len(sub_deltas))
        colors = [palette_elevating if val > 0 else palette_mitigating for val in sub_deltas.values]

        bars = ax.barh(y_pos, sub_deltas.values, color=colors, alpha=0.85, height=0.6, edgecolor="#333333", lw=0.5)

        ax.axvline(0, color="black", linestyle="-", lw=1.0, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_deltas.index, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Marginal Risk Contribution (Δp relative to population background)", fontsize=9, fontweight="bold")

        # Value annotations
        for j, bar in enumerate(bars):
            val = sub_deltas.values[j]
            offset = 0.002 if val >= 0 else -0.002
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, y_pos[j], f"{val:+.3f}", va="center", ha=ha, fontsize=8, fontweight="bold", color="#222222")

        # Title box
        p_val = case_info["predicted_probability"]
        th_val = case_info["locked_threshold"]
        y_val = case_info["actual_target"]
        case_id = case_info["case_id"]
        age = case_info["age_years"]
        gen = int(case_info["general_health_code"])

        title_text = (
            f"{cat_name} | {case_id}\n"
            f"Actual CAD/MI: {y_val} | Predicted Prob: {p_val:.3f} (Threshold: {th_val:.3f})\n"
            f"Profile: Age {age:.0f}y, General Health Code {gen} | Local Surrogate R²: {details['surrogate_r2']:.2f}"
        )
        ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8, loc="left",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", lw=0.8))

        ax.grid(True, linestyle="--", alpha=0.4, axis="x")

        # Add legend on first subplot
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=palette_elevating, edgecolor="#333333", label="Risk-Elevating Factor (Δp > 0)"),
                Patch(facecolor=palette_mitigating, edgecolor="#333333", label="Risk-Mitigating Factor (Δp < 0)")
            ]
            ax.legend(handles=legend_elements, loc="lower right", fontsize=8, frameon=True)

    plt.suptitle(
        "Figure 19: Local Explainable AI (XAI) Directional Attributions for Validation Case Archetypes\n"
        "[Note: Model classifies self-reported history from cross-sectional data; attributions describe associative model behavior, not causal clinical interventions.]",
        fontsize=11.5, fontweight="bold", y=0.99
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 19 to: {output_path.name}")


def main() -> None:
    """Executes Milestone 7 Stage 3 Local Explainability Pipeline."""
    print("=" * 70)
    print("MILESTONE 7 — STAGE 3: LOCAL EXPLAINABLE AI (XAI) PIPELINE")
    print("=" * 70)
    t_start = time.time()

    # 1. Load frozen model, preprocessor, locked threshold, and validation data
    model, preprocessor, threshold, X_val_trans, y_val, X_val_raw, df_mapping = load_validation_data_and_champion()

    # 2. Select representative case archetypes
    print("\n--- Selecting Representative Validation Case Archetypes ---")
    selected_cases, df_cases_summary = select_representative_validation_cases(
        model=model,
        X_val_trans=X_val_trans,
        y_val=y_val,
        X_val_raw=X_val_raw,
        threshold=threshold
    )

    summary_path = xai_config.XAI_EXPLANATIONS_DIR / "xai_case_archetypes_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_cases_summary.to_csv(summary_path, index=False)
    print(f"Saved case archetypes summary table to: {summary_path.name}")
    print(df_cases_summary[["category", "case_id", "actual_target", "predicted_probability", "age_years", "general_health_code"]].to_string(index=False))

    # 3. Compute local explanations (marginal perturbation + local surrogate)
    print("\n--- Computing Local Explanations (Delta p and Local Ridge Surrogates) ---")
    df_explanations, case_details = compute_case_explanations(
        model=model,
        X_val_trans=X_val_trans,
        selected_cases=selected_cases,
        df_mapping=df_mapping,
        threshold=threshold,
        n_background=200,
        n_surrogate_samples=1000,
        random_state=xai_config.RANDOM_STATE
    )

    explanations_path = xai_config.LOCAL_CASE_EXPLANATIONS_PATH
    explanations_path.parent.mkdir(parents=True, exist_ok=True)
    df_explanations.to_csv(explanations_path, index=False)
    print(f"Saved local case explanations to: {explanations_path.name} ({len(df_explanations)} rows)")

    # 4. Generate visualization
    print("\n--- Generating Figure 19: Local Case Directional Attributions ---")
    fig_path = xai_config.FIG_LOCAL_CASE_ATTRIBUTIONS
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_local_case_attributions(
        case_details=case_details,
        df_cases_summary=df_cases_summary,
        output_path=fig_path,
        top_k=8
    )

    t_elapsed = time.time() - t_start
    print(f"\nStage 3 Local XAI Pipeline completed successfully in {t_elapsed:.2f} s.")


if __name__ == "__main__":
    main()
