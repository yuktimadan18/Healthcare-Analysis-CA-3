"""
precompute_dashboard_data.py
----------------------------
Milestone 8, Stage 2: Dashboard Data Precomputation Pipeline.
Extracts, structures, and serializes lightweight JSON bundles for the
"Population Health Analytics & Explainable ML — CDC BRFSS 2024" dashboard:
1. overview_summary.json
2. population_health.json
3. model_performance.json
4. threshold_analysis.json
5. explainable_ai.json

All data are derived strictly from existing verified project artifacts.
No raw respondent-level records are exposed.
"""

from pathlib import Path
import json
import time
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence
from scipy.special import expit

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_overview_summary() -> Dict[str, Any]:
    """Generates overview_summary.json summarizing cohort scope, target, and champion model."""
    print("Generating overview_summary.json...")

    # 1. Dataset partition summary
    part_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "dataset_partition_summary.csv")
    full_cohort = part_df[part_df["partition"] == "Full Eligible Cohort"].iloc[0]

    # 2. Model selection summary
    with open(PROJECT_ROOT / "models" / "model_selection_summary.json", "r", encoding="utf-8") as f:
        model_summary = json.load(f)

    # 3. Outcome feasibility analysis
    outcomes_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "outcome_feasibility_analysis.csv")
    outcomes_list = []
    for _, row in outcomes_df.iterrows():
        outcomes_list.append({
            "variable": str(row["variable"]),
            "condition_name": str(row["condition_name"]),
            "category": str(row["category"]),
            "valid_positive": int(row["valid_positive"]),
            "valid_negative": int(row["valid_negative"]),
            "unweighted_prev_pct": float(row["unweighted_prev_pct"]),
            "weighted_prev_pct": float(row["weighted_prev_pct"]),
            "imbalance_ratio": float(row["imbalance_ratio"])
        })

    payload = {
        "project_title": "Population Health Analytics & Explainable ML — CDC BRFSS 2024",
        "data_source": "CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024",
        "cohort_dimensions": {
            "total_survey_respondents": int(full_cohort["total_records"]),
            "positive_cases": int(full_cohort["positive_cases"]),
            "negative_cases": int(full_cohort["negative_cases"]),
            "missing_excluded_cases": 5206,
            "raw_total_records": 457670,
            "unweighted_prevalence_pct": float(full_cohort["unweighted_prevalence_pct"]),
            "survey_weighted_prevalence_pct": float(full_cohort["survey_weighted_prevalence_pct"]),
            "imbalance_ratio": float(full_cohort["imbalance_ratio"])
        },
        "target_definition": {
            "variable": "_MICHD",
            "full_name": "Reported History of Myocardial Infarction or Coronary Heart Disease",
            "construct_description": (
                "Binary indicator recording whether a respondent ever reported receiving a diagnosis "
                "of heart attack (myocardial infarction) or angina / coronary heart disease from a health professional."
            ),
            "excluded_direct_leakage_constituents": ["CVDINFR4", "CVDCRHD4"]
        },
        "champion_model": {
            "key": model_summary["champion_model_key"],
            "name": model_summary["champion_model_name"],
            "status": model_summary["status"],
            "validation_pr_auc": float(model_summary["validation_pr_auc"]),
            "validation_roc_auc": float(model_summary["validation_roc_auc"]),
            "validation_brier_score": float(model_summary["validation_brier_score"]),
            "locked_threshold": float(model_summary["selected_operational_threshold"]),
            "selection_objective": model_summary["selection_objective"]
        },
        "candidate_outcomes_feasibility": outcomes_list,
        "methodological_disclaimers": [
            "Cross-sectional surveillance classification model; not a prospective disease onset predictor.",
            "Self-reported chronic disease history; clinical diagnostic utility has not been established.",
            "Zero target leakage: _MICHD, CVDINFR4, and CVDCRHD4 are strictly excluded from predictors."
        ]
    }

    out_file = DASHBOARD_DATA_DIR / "overview_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_file.name} ({out_file.stat().st_size:,} bytes)")
    return payload


def generate_population_health() -> Dict[str, Any]:
    """Generates population_health.json containing stratified distributions and risk statistics."""
    print("Generating population_health.json...")

    # Load cohort metadata
    meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "cohort_metadata.csv.gz", index_col=0)

    # Load raw interim dataset for rich demographic categories
    interim = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "brfss_2024_selected.csv.gz")
    interim = interim.loc[meta.index]  # Align strictly to eligible cohort (452,464)
    interim["_MICHD_TARGET"] = meta["target"].values
    interim["_WEIGHT"] = meta["_LLCPWT"].values

    variable_codebooks = {
        "_SEX": {
            "name": "Biological Sex",
            "domain": "Demographics",
            "levels": {1.0: "Male", 2.0: "Female"}
        },
        "_AGEG5YR": {
            "name": "Age Category (5-Year Bands)",
            "domain": "Demographics",
            "levels": {
                1.0: "18-24", 2.0: "25-29", 3.0: "30-34", 4.0: "35-39",
                5.0: "40-44", 6.0: "45-49", 7.0: "50-54", 8.0: "55-59",
                9.0: "60-64", 10.0: "65-69", 11.0: "70-74", 12.0: "75-79",
                13.0: "80+"
            }
        },
        "_RACE": {
            "name": "Race / Ethnicity",
            "domain": "Demographics",
            "levels": {
                1.0: "White Non-Hispanic", 2.0: "Black Non-Hispanic",
                3.0: "American Indian / AK Native", 4.0: "Asian Non-Hispanic",
                5.0: "Native Hawaiian / Pacific Is.", 6.0: "Other Non-Hispanic",
                7.0: "Multiracial Non-Hispanic", 8.0: "Hispanic"
            }
        },
        "_EDUCAG": {
            "name": "Education Completed",
            "domain": "Demographics",
            "levels": {
                1.0: "Did not graduate High School", 2.0: "High School Graduate",
                3.0: "Attended College / Technical", 4.0: "College Graduate"
            }
        },
        "_INCOMG1": {
            "name": "Annual Household Income",
            "domain": "Demographics",
            "levels": {
                1.0: "< $15,000", 2.0: "$15,000 to <$25,000",
                3.0: "$25,000 to <$35,000", 4.0: "$35,000 to <$50,000",
                5.0: "$50,000 to <$100,000", 6.0: "$100,000 to <$200,000",
                7.0: ">= $200,000"
            }
        },
        "GENHLTH": {
            "name": "Self-Rated General Health",
            "domain": "Health Status",
            "levels": {
                1.0: "Excellent", 2.0: "Very Good", 3.0: "Good",
                4.0: "Fair", 5.0: "Poor"
            }
        },
        "DIABETE4": {
            "name": "Diagnosed Diabetes History",
            "domain": "Comorbidities",
            "levels": {
                1.0: "Yes (Diabetes)", 2.0: "Yes (Pregnancy only)",
                3.0: "No (Never diagnosed)", 4.0: "No (Pre-diabetes / Borderline)"
            }
        },
        "CVDSTRK3": {
            "name": "Diagnosed Stroke History",
            "domain": "Comorbidities",
            "levels": {
                1.0: "Yes (Ever diagnosed)", 2.0: "No"
            }
        },
        "CHCKDNY2": {
            "name": "Chronic Kidney Disease",
            "domain": "Comorbidities",
            "levels": {
                1.0: "Yes (Ever diagnosed)", 2.0: "No"
            }
        },
        "_SMOKER3": {
            "name": "Tobacco Smoking Status",
            "domain": "Behaviors & Access",
            "levels": {
                1.0: "Current Smoker (Daily)", 2.0: "Current Smoker (Some days)",
                3.0: "Former Smoker", 4.0: "Never Smoked"
            }
        },
        "CHECKUP1": {
            "name": "Time Since Last Routine Checkup",
            "domain": "Behaviors & Access",
            "levels": {
                1.0: "Within past year (<12 mo)", 2.0: "1 to <2 years ago",
                3.0: "2 to <5 years ago", 4.0: "5+ years ago", 8.0: "Never"
            }
        },
        "PERSDOC3": {
            "name": "Personal Healthcare Provider Continuity",
            "domain": "Behaviors & Access",
            "levels": {
                1.0: "Yes, only one", 2.0: "More than one", 3.0: "No"
            }
        }
    }

    total_unweighted = len(interim)
    total_weighted = float(interim["_WEIGHT"].sum())

    distributions = {}
    for var, cfg in variable_codebooks.items():
        sub_series = interim[var]
        categories = []
        for code, label in cfg["levels"].items():
            mask = (sub_series == code)
            count = int(mask.sum())
            if count == 0:
                continue

            w_sum = float(interim.loc[mask, "_WEIGHT"].sum())
            unweighted_pct = (count / total_unweighted) * 100.0
            weighted_pct = (w_sum / total_weighted) * 100.0

            # Stratified prevalence of _MICHD
            cases = int((mask & (interim["_MICHD_TARGET"] == 1)).sum())
            w_cases = float(interim.loc[mask & (interim["_MICHD_TARGET"] == 1), "_WEIGHT"].sum())

            strat_unweighted_prev = (cases / count * 100.0) if count > 0 else 0.0
            strat_weighted_prev = (w_cases / w_sum * 100.0) if w_sum > 0 else 0.0

            categories.append({
                "code": float(code),
                "label": label,
                "unweighted_count": count,
                "unweighted_pct": round(unweighted_pct, 2),
                "weighted_population_est": round(w_sum, 1),
                "weighted_pct": round(weighted_pct, 2),
                "target_cases": cases,
                "unweighted_prevalence_pct": round(strat_unweighted_prev, 2),
                "weighted_prevalence_pct": round(strat_weighted_prev, 2)
            })

        distributions[var] = {
            "variable_name": var,
            "display_name": cfg["name"],
            "domain": cfg["domain"],
            "categories": categories
        }

    # Load continuous risk factor stats (transposed table from Milestone 3)
    cont_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "continuous_risk_factors_stats.csv", index_col=0)
    cont_stats = []
    for col in cont_df.columns:
        cont_stats.append({
            "variable_label": col,
            "count": int(cont_df.loc["count", col]),
            "mean": round(float(cont_df.loc["mean", col]), 2),
            "std": round(float(cont_df.loc["std", col]), 2),
            "median": round(float(cont_df.loc["50%", col]), 2),
            "iqr": round(float(cont_df.loc["75%", col] - cont_df.loc["25%", col]), 2),
            "min": round(float(cont_df.loc["min", col]), 2),
            "max": round(float(cont_df.loc["max", col]), 2)
        })

    # Load multivariable logistic regression odds ratios (Milestone 4)
    or_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "multivariable_logistic_regression_or.csv")
    odds_ratios = []
    for _, row in or_df.iterrows():
        if row["covariate"] == "Intercept":
            continue
        odds_ratios.append({
            "covariate": str(row["covariate"]),
            "odds_ratio": float(row["aOR"]),
            "ci_lower_95": float(row["ci_95_cluster_lower"]),
            "ci_upper_95": float(row["ci_95_cluster_upper"]),
            "p_value": float(row["p_value_norm"])
        })

    payload = {
        "metadata": {
            "cohort_size": total_unweighted,
            "total_weighted_population": round(total_weighted, 1),
            "disclaimer": "Survey-weighted metrics represent point estimates; complex-survey standard errors require PSU/Strata adjustments."
        },
        "distributions": distributions,
        "continuous_risk_factor_stats": cont_stats,
        "multivariable_odds_ratios": odds_ratios
    }

    out_file = DASHBOARD_DATA_DIR / "population_health.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_file.name} ({out_file.stat().st_size:,} bytes)")
    return payload


def generate_model_performance() -> Dict[str, Any]:
    """Generates model_performance.json with comparison metrics and downsampled ROC/PR curves."""
    print("Generating model_performance.json...")

    val_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "model_validation_comparison.csv")
    val_models = []
    for _, row in val_df.iterrows():
        val_models.append({
            "model_key": str(row["model_key"]),
            "model_name": str(row["model_name"]),
            "val_pr_auc": float(row["val_pr_auc"]),
            "val_roc_auc": float(row["val_roc_auc"]),
            "val_brier_score": float(row["val_brier_score"]),
            "val_brier_skill_score": float(row["val_brier_skill_score"]),
            "val_mean_pred_prob": float(row["val_mean_pred_prob"]),
            "sens_at_050": float(row["sens_at_050"]),
            "spec_at_050": float(row["spec_at_050"]),
            "f1_at_050": float(row["f1_at_050"]),
            "sens_at_prev_th": float(row["sens_at_prev_th"]),
            "spec_at_prev_th": float(row["spec_at_prev_th"]),
            "f1_at_prev_th": float(row["f1_at_prev_th"]),
            "survey_weighted_pr_auc": float(row["survey_weighted_pr_auc"]),
            "survey_weighted_roc_auc": float(row["survey_weighted_roc_auc"]),
            "survey_weighted_brier": float(row["survey_weighted_brier"])
        })

    test_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "final_test_evaluation_summary.csv")
    test_eval = []
    for _, row in test_df.iterrows():
        test_eval.append({
            "evaluation_mode": str(row["evaluation_mode"]),
            "threshold_label": str(row["threshold_label"]),
            "threshold": float(row["threshold"]),
            "pr_auc": float(row["pr_auc"]),
            "roc_auc": float(row["roc_auc"]),
            "brier_score": float(row["brier_score"]),
            "sensitivity": float(row["sensitivity"]),
            "specificity": float(row["specificity"]),
            "precision_ppv": float(row["precision_ppv"]),
            "npv": float(row["npv"]),
            "f1_score": float(row["f1_score"]),
            "balanced_accuracy": float(row["balanced_accuracy"])
        })

    # Compute downsampled ROC & PR curves for validation partition
    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.joblib")
    X_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "validation_features.csv.gz", index_col=0)
    y_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "validation_target.csv.gz", index_col=0).squeeze("columns").values
    X_val_trans = preprocessor.transform(X_val)

    models_dict = {
        "hist_gradient_boosting_unweighted": joblib.load(PROJECT_ROOT / "models" / "improved_hist_gradient_boosting.joblib"),
        "hist_gradient_boosting_balanced": joblib.load(PROJECT_ROOT / "models" / "weighted_hist_gradient_boosting.joblib"),
        "logistic_regression_unweighted": joblib.load(PROJECT_ROOT / "models" / "baseline_logistic_regression.joblib"),
        "logistic_regression_balanced": joblib.load(PROJECT_ROOT / "models" / "weighted_logistic_regression.joblib")
    }

    curves = {}
    for m_key, m_obj in models_dict.items():
        probs = m_obj.predict_proba(X_val_trans)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, probs)
        prec, rec, _ = precision_recall_curve(y_val, probs)
        prob_true, prob_pred = calibration_curve(y_val, probs, n_bins=10, strategy="quantile")

        # Downsample ROC to 101 points
        x_grid = np.linspace(0, 1, 101)
        tpr_interp = np.interp(x_grid, fpr, tpr)
        tpr_interp[0] = 0.0

        # Downsample PR curve: interpolate precision across recall grid
        rec_rev = rec[::-1]
        prec_rev = prec[::-1]
        r_grid = np.linspace(0, 1, 101)
        prec_interp = np.interp(r_grid, rec_rev, prec_rev)

        curves[m_key] = {
            "roc": {
                "fpr": [round(float(v), 4) for v in x_grid],
                "tpr": [round(float(v), 4) for v in tpr_interp]
            },
            "pr": {
                "recall": [round(float(v), 4) for v in r_grid],
                "precision": [round(float(v), 4) for v in prec_interp]
            },
            "calibration": {
                "predicted_prob": [round(float(v), 4) for v in prob_pred],
                "observed_fraction": [round(float(v), 4) for v in prob_true]
            }
        }

    payload = {
        "validation_models": val_models,
        "test_holdout_summary": test_eval,
        "curves": curves
    }

    out_file = DASHBOARD_DATA_DIR / "model_performance.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_file.name} ({out_file.stat().st_size:,} bytes)")
    return payload


def generate_threshold_analysis() -> Dict[str, Any]:
    """Generates threshold_analysis.json with grid coordinates and confusion matrix payloads."""
    print("Generating threshold_analysis.json...")

    th_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "threshold_optimization_analysis.csv")
    grid_points = []
    for _, row in th_df.iterrows():
        grid_points.append({
            "threshold": round(float(row["threshold"]), 2),
            "sensitivity": round(float(row["sensitivity"]), 4),
            "specificity": round(float(row["specificity"]), 4),
            "precision_ppv": round(float(row["precision_ppv"]), 4),
            "npv": round(float(row["npv"]), 4),
            "f1_score": round(float(row["f1_score"]), 4),
            "balanced_accuracy": round(float(row["balanced_accuracy"]), 4),
            "youden_j": round(float(row["youden_j"]), 4),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "tn": int(row["tn"]),
            "fn": int(row["fn"])
        })

    with open(PROJECT_ROOT / "models" / "model_selection_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    locked_th = float(summary["selected_operational_threshold"])

    th_080 = th_df[np.isclose(th_df["threshold"], 0.08, atol=0.005)].iloc[0]
    th_050 = th_df[np.isclose(th_df["threshold"], 0.50, atol=0.005)].iloc[0]

    payload = {
        "locked_operational_threshold": locked_th,
        "selection_objective": summary["selection_objective"],
        "threshold_grid": grid_points,
        "confusion_matrix_at_locked_080": {
            "threshold": 0.08,
            "tp": int(th_080["tp"]),
            "fp": int(th_080["fp"]),
            "tn": int(th_080["tn"]),
            "fn": int(th_080["fn"]),
            "sensitivity": float(th_080["sensitivity"]),
            "specificity": float(th_080["specificity"]),
            "precision_ppv": float(th_080["precision_ppv"]),
            "npv": float(th_080["npv"]),
            "f1_score": float(th_080["f1_score"]),
            "balanced_accuracy": float(th_080["balanced_accuracy"]),
            "youden_j": float(th_080["youden_j"])
        },
        "confusion_matrix_at_default_050": {
            "threshold": 0.50,
            "tp": int(th_050["tp"]),
            "fp": int(th_050["fp"]),
            "tn": int(th_050["tn"]),
            "fn": int(th_050["fn"]),
            "sensitivity": float(th_050["sensitivity"]),
            "specificity": float(th_050["specificity"]),
            "precision_ppv": float(th_050["precision_ppv"]),
            "npv": float(th_050["npv"]),
            "f1_score": float(th_050["f1_score"]),
            "balanced_accuracy": float(th_050["balanced_accuracy"]),
            "youden_j": float(th_050["youden_j"])
        },
        "safety_guardrail": (
            "Interactive exploration of alternative cutoffs does not modify the locked threshold of 0.080, "
            "which was established prior to test set evaluation."
        )
    }

    out_file = DASHBOARD_DATA_DIR / "threshold_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_file.name} ({out_file.stat().st_size:,} bytes)")
    return payload


def generate_explainable_ai() -> Dict[str, Any]:
    """Generates explainable_ai.json with global importance, PDP profiles, and case archetypes."""
    print("Generating explainable_ai.json...")

    # 1. Global grouped variable importance
    global_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "xai_global_variable_importance.csv")
    global_list = []
    for _, row in global_df.iterrows():
        global_list.append({
            "rank": int(row["rank"]),
            "source_variable": str(row["source_variable"]),
            "feature_type": str(row["feature_type"]),
            "num_transformed_columns": int(row["num_transformed_columns"]),
            "mean_pr_auc_drop": round(float(row["mean_pr_auc_drop"]), 6),
            "std_pr_auc_drop": round(float(row["std_pr_auc_drop"]), 6),
            "mean_roc_auc_drop": round(float(row["mean_roc_auc_drop"]), 6),
            "std_roc_auc_drop": round(float(row["std_roc_auc_drop"]), 6)
        })

    # 2. Transformed indicator importance (Top 30)
    ind_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "xai_transformed_feature_importance.csv")
    ind_list = []
    for _, row in ind_df.head(30).iterrows():
        ind_list.append({
            "rank": int(row["rank"]),
            "transformed_feature_name": str(row["transformed_feature_name"]),
            "source_variable": str(row["source_variable"]),
            "feature_type": str(row["feature_type"]),
            "category_value": str(row["category_value"]),
            "mean_pr_auc_drop": round(float(row["mean_pr_auc_drop"]), 6),
            "std_pr_auc_drop": round(float(row["std_pr_auc_drop"]), 6),
            "mean_roc_auc_drop": round(float(row["mean_roc_auc_drop"]), 6)
        })

    # 3. Partial Dependence Curves (Unscaled)
    model = joblib.load(PROJECT_ROOT / "models" / "improved_hist_gradient_boosting.joblib")
    preprocessor = joblib.load(PROJECT_ROOT / "models" / "preprocessor.joblib")
    df_mapping = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "xai_feature_mapping_audit.csv")

    X_val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "validation_features.csv.gz", index_col=0)
    X_val_trans = preprocessor.transform(X_val)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]

    numeric_vars = {
        "_AGE80": {"name": "Continuous Age", "unit": "Years"},
        "PHYSHLTH_DAYS": {"name": "Poor Physical Health Days", "unit": "Days in past 30 days"},
        "BMI_NUMERIC": {"name": "Body Mass Index (BMI)", "unit": "kg/m²"}
    }

    pdp_curves = {}
    for var, v_info in numeric_vars.items():
        col_idx = int(df_mapping[df_mapping["source_variable"] == var]["feature_index"].iloc[0])
        scaler_idx = list(preprocessor.transformers_[0][2]).index(var)

        res = partial_dependence(
            model,
            X_val_trans,
            features=[col_idx],
            response_method="decision_function",
            grid_resolution=50,
            method="recursion"
        )
        grid_scaled = res["grid_values"][0]
        grid_unscaled = grid_scaled * scaler.scale_[scaler_idx] + scaler.mean_[scaler_idx]
        marginal_prob = expit(res["average"][0])

        pdp_curves[var] = {
            "display_name": v_info["name"],
            "unit": v_info["unit"],
            "grid_unscaled": [round(float(v), 2) for v in grid_unscaled],
            "marginal_probability": [round(float(v), 4) for v in marginal_prob]
        }

    # 4. Case Archetypes Summary
    cases_df = pd.read_csv(PROJECT_ROOT / "results" / "explanations" / "xai_case_archetypes_summary.csv")
    archetypes_list = []
    for _, row in cases_df.iterrows():
        archetypes_list.append({
            "category": str(row["category"]),
            "case_id": str(row["case_id"]),
            "validation_row_index": int(row["validation_row_index"]),
            "actual_target": int(row["actual_target"]),
            "predicted_probability": float(row["predicted_probability"]),
            "locked_threshold": float(row["locked_threshold"]),
            "predicted_class": int(row["predicted_class"]),
            "age_years": float(row["age_years"]),
            "general_health_code": float(row["general_health_code"]),
            "sex_code": float(row["sex_code"]),
            "stroke_code": float(row["stroke_code"]),
            "physical_unhealthy_days": float(row["physical_unhealthy_days"]),
            "bmi_numeric": float(row["bmi_numeric"]),
            "archetype_description": str(row["archetype_description"])
        })

    # 5. Local Case Explanations (38 variables * 4 archetypes)
    exp_df = pd.read_csv(PROJECT_ROOT / "results" / "explanations" / "xai_local_case_explanations.csv")
    case_explanations = {}
    for cat in exp_df["case_category"].unique():
        sub = exp_df[exp_df["case_category"] == cat]
        records = []
        for _, row in sub.iterrows():
            records.append({
                "source_variable": str(row["source_variable"]),
                "feature_type": str(row["feature_type"]),
                "num_transformed_columns": int(row["num_transformed_columns"]),
                "marginal_risk_delta_dp": round(float(row["marginal_risk_delta_dp"]), 6),
                "local_surrogate_weight": round(float(row["local_surrogate_weight"]), 6),
                "directional_impact": str(row["directional_impact"]),
                "surrogate_fidelity_r2": float(row["surrogate_fidelity_r2"])
            })
        case_explanations[cat] = records

    payload = {
        "global_grouped_importance": global_list,
        "indicator_importance_top30": ind_list,
        "partial_dependence_profiles": pdp_curves,
        "case_archetypes_summary": archetypes_list,
        "case_explanations": case_explanations,
        "methodology_notes": {
            "attribution_method": "Grouped Marginal Perturbation Attribution (Delta p) relative to empirical validation background.",
            "surrogate_method": "Local Ridge Linear Surrogate with exponential kernel distance weighting.",
            "disclaimer": "Attributions reflect statistical model behavior, not causal biological effects or confirmed diagnoses."
        }
    }

    out_file = DASHBOARD_DATA_DIR / "explainable_ai.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_file.name} ({out_file.stat().st_size:,} bytes)")
    return payload


def main() -> None:
    """Runs the full dashboard data precomputation pipeline."""
    print("=" * 70)
    print("MILESTONE 8 — STAGE 2: DASHBOARD DATA PRECOMPUTATION PIPELINE")
    print("=" * 70)
    t0 = time.time()

    generate_overview_summary()
    generate_population_health()
    generate_model_performance()
    generate_threshold_analysis()
    generate_explainable_ai()

    elapsed = time.time() - t0
    print(f"\nAll 5 dashboard JSON bundles generated successfully in {elapsed:.2f} s.")
    print(f"Data directory: {DASHBOARD_DATA_DIR}")


if __name__ == "__main__":
    main()
