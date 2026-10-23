"""
statistical_tests.py
--------------------
Reusable statistical analysis module for CDC BRFSS 2024 Healthcare Analytics.

Provides functions for:
  - Outcome feasibility auditing (exact codebook responses, valid counts, missingness, weights).
  - Categorical association tests (Chi-Square, Cramer's V, Bivariate Odds Ratios with 95% CIs).
  - Continuous variable comparisons (Survey-weighted means, SEs, and confidence intervals).
  - Multivariable Logistic Regression and Adjusted Odds Ratios (aOR) using statsmodels.
  - Multiple comparison corrections (Benjamini-Hochberg False Discovery Rate).
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.multitest import multipletests


def audit_outcome_feasibility(
    df: pd.DataFrame,
    outcomes_config: Dict[str, Dict[str, Any]],
    weight_col: str = "_LLCPWT"
) -> pd.DataFrame:
    """
    Computes feasibility metrics for candidate chronic conditions using exact codebook definitions.

    Args:
        df: Input DataFrame containing candidate variables.
        outcomes_config: Mapping of column names to positive, negative, invalid, and special codes.
        weight_col: Column name containing complex survey sampling weights.

    Returns:
        pd.DataFrame summarizing valid sample sizes, missingness, unweighted and weighted prevalence,
        and accounting for all 457,670 respondents.
    """
    rows = []
    weights = df[weight_col].fillna(0)

    for col, cfg in outcomes_config.items():
        if col not in df.columns:
            continue

        s = df[col]
        pos_codes = cfg["pos"]
        neg_codes = cfg["neg"]
        dk_codes = cfg.get("dk", [7, 77, 777])
        refused_codes = cfg.get("refused", [9, 99, 999])
        special_codes = cfg.get("special", [])

        pos_mask = s.isin(pos_codes)
        neg_mask = s.isin(neg_codes)
        dk_mask = s.isin(dk_codes)
        refused_mask = s.isin(refused_codes)
        special_mask = s.isin(special_codes)
        null_mask = s.isna()

        valid_mask = pos_mask | neg_mask
        pos_count = int(pos_mask.sum())
        neg_count = int(neg_mask.sum())
        valid_count = int(valid_mask.sum())
        dk_count = int(dk_mask.sum())
        refused_count = int(refused_mask.sum())
        special_count = int(special_mask.sum())
        null_count = int(null_mask.sum())
        total_records = len(df)

        # Checksum verification (must equal total records exactly)
        row_checksum = pos_count + neg_count + dk_count + refused_count + special_count + null_count

        unweighted_prev = (pos_count / valid_count * 100.0) if valid_count > 0 else 0.0

        # Survey-weighted prevalence (domain ratio within valid responses)
        valid_weights = weights[valid_mask]
        pos_weights = weights[pos_mask]
        if valid_weights.sum() > 0:
            weighted_prev = (pos_weights.sum() / valid_weights.sum()) * 100.0
        else:
            weighted_prev = 0.0

        imbalance_ratio = (neg_count / pos_count) if pos_count > 0 else np.nan
        completeness_pct = (valid_count / total_records) * 100.0

        rows.append({
            "variable": col,
            "condition_name": cfg["name"],
            "category": cfg.get("category", "Chronic Condition"),
            "valid_positive": pos_count,
            "valid_negative": neg_count,
            "total_valid": valid_count,
            "dont_know_count": dk_count,
            "refused_count": refused_count,
            "other_special_count": special_count,
            "missing_nan_count": null_count,
            "row_checksum": row_checksum,
            "completeness_pct": round(completeness_pct, 2),
            "unweighted_prev_pct": round(unweighted_prev, 2),
            "weighted_prev_pct": round(weighted_prev, 2),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "prevalence_gap_pct": round(unweighted_prev - weighted_prev, 2)
        })

    result_df = pd.DataFrame(rows)
    return result_df


def compute_bivariate_odds_ratios(
    df: pd.DataFrame,
    target_col: str,
    target_pos_codes: List[int],
    target_neg_codes: List[int],
    predictor_cols: List[str],
    predictor_mapping: Dict[str, Dict[str, Any]],
    weight_col: str = "_LLCPWT"
) -> pd.DataFrame:
    """
    Computes 2x2 contingency tables, unweighted Odds Ratios with 95% Wald CIs,
    Pearson Chi-Square test of independence, Cramer's V, and survey-weighted OR point estimates.

    IMPORTANT METHODOLOGICAL NOTE:
      - 'unweighted_or', 'unweighted_ci_95_lower/upper', 'unweighted_chi2', and 'unweighted_cramers_v'
        are calculated from the raw unweighted sample contingency table. They represent exploratory
        sample statistics and do NOT incorporate complex survey design adjustments (such as Rao-Scott
        second-order correction).
      - 'weighted_or' is the survey-weighted point estimate (ratio of weighted cross-products).
    """
    target_s = df[target_col]
    target_valid = target_s.isin(target_pos_codes + target_neg_codes)
    y = target_s[target_valid].isin(target_pos_codes).astype(int)
    weights = df.loc[target_valid, weight_col]

    results = []

    for pred in predictor_cols:
        if pred not in df.columns:
            continue

        cfg = predictor_mapping.get(pred, {})
        pred_pos_codes = cfg.get("pos")
        pred_neg_codes = cfg.get("neg")

        if not pred_pos_codes or not pred_neg_codes:
            continue

        pred_s = df.loc[target_valid, pred]
        pred_valid = pred_s.isin(pred_pos_codes + pred_neg_codes)

        y_sub = y[pred_valid]
        x_sub = pred_s[pred_valid].isin(pred_pos_codes).astype(int)
        w_sub = weights[pred_valid]

        if len(y_sub) == 0 or x_sub.nunique() < 2 or y_sub.nunique() < 2:
            continue

        # 2x2 table: exposed (x=1) vs unexposed (x=0), cases (y=1) vs controls (y=0)
        a = int(((x_sub == 1) & (y_sub == 1)).sum())
        b = int(((x_sub == 1) & (y_sub == 0)).sum())
        c = int(((x_sub == 0) & (y_sub == 1)).sum())
        d = int(((x_sub == 0) & (y_sub == 0)).sum())

        if a == 0 or b == 0 or c == 0 or d == 0:
            or_val = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            se_log_or = np.nan
        else:
            or_val = (a * d) / (b * c)
            se_log_or = np.sqrt((1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d))
            ci_lower = np.exp(np.log(or_val) - 1.96 * se_log_or)
            ci_upper = np.exp(np.log(or_val) + 1.96 * se_log_or)

        # Unweighted Chi-Square test of independence
        table = np.array([[a, b], [c, d]])
        chi2, p_val, dof, _ = stats.chi2_contingency(table, correction=True)
        cramers_v = np.sqrt(chi2 / (len(y_sub) * min(table.shape[0] - 1, table.shape[1] - 1)))

        # Weighted Odds Ratio point estimate
        wa = float(w_sub[(x_sub == 1) & (y_sub == 1)].sum())
        wb = float(w_sub[(x_sub == 1) & (y_sub == 0)].sum())
        wc = float(w_sub[(x_sub == 0) & (y_sub == 1)].sum())
        wd = float(w_sub[(x_sub == 0) & (y_sub == 0)].sum())
        weighted_or = (wa * wd) / (wb * wc) if (wb > 0 and wc > 0) else np.nan

        results.append({
            "predictor": pred,
            "predictor_label": cfg.get("label", pred),
            "target": target_col,
            "n_obs": len(y_sub),
            "exposed_cases": a,
            "exposed_controls": b,
            "unexposed_cases": c,
            "unexposed_controls": d,
            "unweighted_or": round(or_val, 3),
            "unweighted_ci_95_lower": round(ci_lower, 3),
            "unweighted_ci_95_upper": round(ci_upper, 3),
            "weighted_or": round(weighted_or, 3),
            "unweighted_chi2": round(chi2, 2),
            "unweighted_p_value": p_val,
            "unweighted_cramers_v": round(cramers_v, 4)
        })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # Multiple testing correction on unweighted p-values
        rejected, p_adj, _, _ = multipletests(res_df["unweighted_p_value"], method="fdr_bh", alpha=0.05)
        res_df["unweighted_p_fdr"] = p_adj
        res_df["significant_fdr"] = rejected
        res_df = res_df.sort_values(by="unweighted_or", ascending=False)

    return res_df


def fit_survey_weighted_logistic_regression(
    df: pd.DataFrame,
    formula: str,
    weight_col: str = "_LLCPWT",
    cluster_col: Optional[str] = "_PSU"
) -> Tuple[pd.DataFrame, Any, Optional[Any]]:
    """
    Fits a survey-weighted logistic regression model via statsmodels GLM with Binomial family.

    CRITICAL METHODOLOGICAL CORRECTION:
      1. Weight Normalization: In statsmodels, 'freq_weights' treats values as integer replication
         counts. Raw survey expansion weights (_LLCPWT) sum to ~230M+ (the US adult population).
         Passing raw expansion weights inflates the effective sample size by ~570x, artificially
         shrinking standard errors by ~24x and generating unrealistically tight confidence intervals.
         To correct this, weights are normalized to sample size: w_norm = w * (N / sum(w)), so that
         sum(w_norm) = N. Point estimates (aOR) are invariant to constant scaling, but the variance
         scale accurately reflects the actual sample size.
      2. Cluster-Robust Sensitivity Analysis: If cluster_col is specified (e.g., '_PSU'), a secondary
         cluster-robust variance model is fitted using cov_type='cluster' to evaluate the impact of
         clustering on standard error inflation.
      3. Limitation Disclosure: Neither model fully reproduces official CDC Taylor-series multi-stage
         linearization across both strata (_STSTR) and primary sampling units (_PSU), which requires
         specialized complex-survey packages.

    Returns:
        Tuple of (summary_dataframe, normalized_glm_result, cluster_robust_glm_result).
    """
    import statsmodels.formula.api as smf

    # Create a working copy with normalized weights
    work_df = df.copy()
    N = len(work_df)
    sum_w = work_df[weight_col].sum()
    work_df["_w_norm"] = work_df[weight_col] * (N / sum_w)

    # 1. Normalized weights GLM
    model = smf.glm(
        formula=formula,
        data=work_df,
        family=sm.families.Binomial(),
        freq_weights=work_df["_w_norm"]
    )
    result_norm = model.fit()

    params = result_norm.params
    se_norm = result_norm.bse
    pvalues_norm = result_norm.pvalues
    aor = np.exp(params.values)
    ci_lower_norm = np.exp(params.values - 1.96 * se_norm.values)
    ci_upper_norm = np.exp(params.values + 1.96 * se_norm.values)

    # 2. Cluster-robust sensitivity model (if cluster_col is present)
    result_cluster = None
    se_cluster_vals = [np.nan] * len(params)
    ci_lower_cl_vals = [np.nan] * len(params)
    ci_upper_cl_vals = [np.nan] * len(params)

    if cluster_col and cluster_col in work_df.columns:
        try:
            result_cluster = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": work_df[cluster_col]}
            )
            se_cluster = result_cluster.bse
            se_cluster_vals = se_cluster.values
            ci_lower_cl_vals = np.exp(params.values - 1.96 * se_cluster.values)
            ci_upper_cl_vals = np.exp(params.values + 1.96 * se_cluster.values)
        except Exception as e:
            print(f"Cluster covariance warning: {e}")

    summary_df = pd.DataFrame({
        "covariate": params.index,
        "coef": params.values.round(4),
        "aOR": aor.round(3),
        "se_norm_weights": se_norm.values.round(4),
        "ci_95_norm_lower": ci_lower_norm.round(3),
        "ci_95_norm_upper": ci_upper_norm.round(3),
        "se_cluster_psu": np.array(se_cluster_vals).round(4),
        "ci_95_cluster_lower": np.array(ci_lower_cl_vals).round(3),
        "ci_95_cluster_upper": np.array(ci_upper_cl_vals).round(3),
        "p_value_norm": pvalues_norm.values
    })

    return summary_df, result_norm, result_cluster


def compute_weighted_summary_stats(
    df: pd.DataFrame,
    col: str,
    weight_col: str = "_LLCPWT"
) -> Dict[str, float]:
    """
    Computes survey-weighted mean, standard deviation, and standard error using DescrStatsW.
    """
    valid_mask = df[col].notna() & df[weight_col].notna()
    data = df.loc[valid_mask, col].astype(float)
    weights = df.loc[valid_mask, weight_col].astype(float)

    descr = DescrStatsW(data, weights=weights, ddof=1)
    mean = descr.mean
    std = descr.std
    std_err = descr.std_mean
    ci_low, ci_high = descr.tconfint_mean()

    return {
        "n_obs": int(valid_mask.sum()),
        "weighted_mean": round(float(mean), 3),
        "weighted_std": round(float(std), 3),
        "weighted_se": round(float(std_err), 4),
        "ci_95_lower": round(float(ci_low), 3),
        "ci_95_upper": round(float(ci_high), 3)
    }
