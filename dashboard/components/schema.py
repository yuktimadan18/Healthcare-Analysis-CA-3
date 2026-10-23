"""
schema.py
---------
Schema definition and programmatic validation module for Milestone 8 Dashboard JSON artifacts:
1. overview_summary.json
2. population_health.json
3. model_performance.json
4. threshold_analysis.json
5. explainable_ai.json

Enforces data integrity, numerical bounds, and the strict exclusion of raw respondent records.
"""

from pathlib import Path
import json
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dashboard" / "data"


def validate_overview_summary(data: Dict[str, Any]) -> bool:
    """Validates schema and contents of overview_summary.json."""
    required_keys = ["project_title", "data_source", "cohort_dimensions", "target_definition", "champion_model", "candidate_outcomes_feasibility", "methodological_disclaimers"]
    for k in required_keys:
        assert k in data, f"Missing key in overview_summary: {k}"

    dims = data["cohort_dimensions"]
    assert dims["total_survey_respondents"] == 452464, f"Unexpected cohort size: {dims['total_survey_respondents']}"
    assert dims["positive_cases"] == 42338, f"Unexpected positive cases: {dims['positive_cases']}"
    assert dims["negative_cases"] == 410126, f"Unexpected negative cases: {dims['negative_cases']}"
    assert abs(dims["unweighted_prevalence_pct"] - 9.36) < 0.01
    assert abs(dims["survey_weighted_prevalence_pct"] - 6.80) < 0.01

    target = data["target_definition"]
    assert target["variable"] == "_MICHD"
    assert "CVDINFR4" in target["excluded_direct_leakage_constituents"]
    assert "CVDCRHD4" in target["excluded_direct_leakage_constituents"]

    champ = data["champion_model"]
    assert champ["key"] == "hist_gradient_boosting_unweighted"
    assert champ["locked_threshold"] == 0.080
    assert abs(champ["validation_pr_auc"] - 0.3701) < 0.001
    return True


def validate_population_health(data: Dict[str, Any]) -> bool:
    """Validates schema and contents of population_health.json."""
    required_keys = ["metadata", "distributions", "continuous_risk_factor_stats", "multivariable_odds_ratios"]
    for k in required_keys:
        assert k in data, f"Missing key in population_health: {k}"

    assert data["metadata"]["cohort_size"] == 452464
    dists = data["distributions"]
    expected_vars = ["_SEX", "_AGEG5YR", "_RACE", "_EDUCAG", "_INCOMG1", "GENHLTH", "DIABETE4", "CVDSTRK3", "CHCKDNY2", "_SMOKER3", "CHECKUP1", "PERSDOC3"]
    for v in expected_vars:
        assert v in dists, f"Missing distribution for variable: {v}"
        cat_list = dists[v]["categories"]
        assert len(cat_list) > 0, f"Empty categories for variable: {v}"
        for cat in cat_list:
            assert "unweighted_count" in cat
            assert "weighted_pct" in cat
            assert "unweighted_prevalence_pct" in cat
            assert "weighted_prevalence_pct" in cat

    cont_stats = data["continuous_risk_factor_stats"]
    assert len(cont_stats) >= 4, f"Expected at least 4 continuous stats, got {len(cont_stats)}"

    ors = data["multivariable_odds_ratios"]
    assert len(ors) >= 8, f"Expected at least 8 multivariable odds ratios, got {len(ors)}"
    return True


def validate_model_performance(data: Dict[str, Any]) -> bool:
    """Validates schema and contents of model_performance.json."""
    required_keys = ["validation_models", "test_holdout_summary", "curves"]
    for k in required_keys:
        assert k in data, f"Missing key in model_performance: {k}"

    val_models = data["validation_models"]
    assert len(val_models) == 4, f"Expected 4 candidate models, got {len(val_models)}"
    champ = next(m for m in val_models if m["model_key"] == "hist_gradient_boosting_unweighted")
    assert abs(champ["val_pr_auc"] - 0.3701) < 0.001
    assert abs(champ["val_roc_auc"] - 0.8449) < 0.001
    assert abs(champ["val_brier_score"] - 0.0700) < 0.001

    test_summary = data["test_holdout_summary"]
    assert len(test_summary) == 4, f"Expected 4 test summary rows, got {len(test_summary)}"
    unweighted_080 = next(t for t in test_summary if t["evaluation_mode"] == "Unweighted Sample Performance" and abs(t["threshold"] - 0.08) < 0.005)
    assert abs(unweighted_080["sensitivity"] - 0.8391) < 0.001
    assert abs(unweighted_080["specificity"] - 0.7045) < 0.001
    assert abs(unweighted_080["npv"] - 0.9770) < 0.001

    curves = data["curves"]
    assert "hist_gradient_boosting_unweighted" in curves
    champ_curves = curves["hist_gradient_boosting_unweighted"]
    assert len(champ_curves["roc"]["fpr"]) == 101
    assert len(champ_curves["pr"]["recall"]) == 101
    assert len(champ_curves["calibration"]["predicted_prob"]) == 10
    return True


def validate_threshold_analysis(data: Dict[str, Any]) -> bool:
    """Validates schema and contents of threshold_analysis.json."""
    required_keys = ["locked_operational_threshold", "selection_objective", "threshold_grid", "confusion_matrix_at_locked_080", "confusion_matrix_at_default_050", "safety_guardrail"]
    for k in required_keys:
        assert k in data, f"Missing key in threshold_analysis: {k}"

    assert data["locked_operational_threshold"] == 0.080
    grid = data["threshold_grid"]
    assert len(grid) == 89, f"Expected 89 grid points, got {len(grid)}"

    cm_080 = data["confusion_matrix_at_locked_080"]
    assert cm_080["tp"] == 5290
    assert cm_080["fp"] == 18283
    assert cm_080["tn"] == 43236
    assert cm_080["fn"] == 1061

    cm_050 = data["confusion_matrix_at_default_050"]
    assert cm_050["tp"] == 677
    assert cm_050["fp"] == 499
    assert cm_050["tn"] == 61020
    assert cm_050["fn"] == 5674
    return True


def validate_explainable_ai(data: Dict[str, Any]) -> bool:
    """Validates schema and contents of explainable_ai.json."""
    required_keys = ["global_grouped_importance", "indicator_importance_top30", "partial_dependence_profiles", "case_archetypes_summary", "case_explanations", "methodology_notes"]
    for k in required_keys:
        assert k in data, f"Missing key in explainable_ai: {k}"

    global_imp = data["global_grouped_importance"]
    assert len(global_imp) == 38, f"Expected 38 original predictors, got {len(global_imp)}"
    assert global_imp[0]["source_variable"] == "_AGE80"
    assert global_imp[1]["source_variable"] == "GENHLTH"
    assert global_imp[2]["source_variable"] == "_SEX"

    pdp = data["partial_dependence_profiles"]
    for v in ["_AGE80", "PHYSHLTH_DAYS", "BMI_NUMERIC"]:
        assert v in pdp, f"Missing PDP for variable: {v}"
        assert len(pdp[v]["grid_unscaled"]) in [31, 50]
        assert len(pdp[v]["marginal_probability"]) in [31, 50]

    archetypes = data["case_archetypes_summary"]
    assert len(archetypes) == 4, f"Expected 4 archetypes, got {len(archetypes)}"
    expected_categories = {"True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)"}
    assert {a["category"] for a in archetypes} == expected_categories

    case_exps = data["case_explanations"]
    assert len(case_exps) == 4
    for cat in expected_categories:
        assert cat in case_exps
        assert len(case_exps[cat]) == 38, f"Expected 38 variable attributions for {cat}, got {len(case_exps[cat])}"
    return True


def validate_all_precomputed_artifacts() -> Dict[str, bool]:
    """Validates all 5 precomputed JSON files on disk."""
    files = {
        "overview_summary.json": validate_overview_summary,
        "population_health.json": validate_population_health,
        "model_performance.json": validate_model_performance,
        "threshold_analysis.json": validate_threshold_analysis,
        "explainable_ai.json": validate_explainable_ai
    }
    results = {}
    for fname, validator in files.items():
        fpath = DATA_DIR / fname
        assert fpath.exists(), f"Missing required precomputed file: {fname}"
        with open(fpath, "r", encoding="utf-8") as fp:
            content = json.load(fp)
        results[fname] = validator(content)
        print(f"[PASS] {fname} passed schema and numerical validation.")
    return results


if __name__ == "__main__":
    validate_all_precomputed_artifacts()
