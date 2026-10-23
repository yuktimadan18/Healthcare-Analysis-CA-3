"""
test_dashboard_precomputation.py
--------------------------------
Unit test suite for Milestone 8 Stage 2: Dashboard Data Precomputation.
Validates:
1. Presence of all 5 precomputed JSON files under dashboard/data/.
2. Schema validity and typing via dashboard.components.schema.
3. Source-to-JSON numerical consistency against CSV tables.
4. Exclusion of raw respondent-level data (only 4 validated archetypes).
5. Invariant preservation of the locked threshold (0.080).
6. File size constraints for sub-second web performance (< 250 KB total).
"""

import unittest
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"

from dashboard.components.schema import (
    validate_overview_summary,
    validate_population_health,
    validate_model_performance,
    validate_threshold_analysis,
    validate_explainable_ai,
    validate_all_precomputed_artifacts
)


class TestDashboardPrecomputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads all 5 JSON bundles once for testing."""
        cls.files = {
            "overview": DASHBOARD_DATA_DIR / "overview_summary.json",
            "population": DASHBOARD_DATA_DIR / "population_health.json",
            "performance": DASHBOARD_DATA_DIR / "model_performance.json",
            "threshold": DASHBOARD_DATA_DIR / "threshold_analysis.json",
            "xai": DASHBOARD_DATA_DIR / "explainable_ai.json"
        }
        cls.data = {}
        for k, p in cls.files.items():
            cls.assertTrue(p.exists(), f"Missing required JSON bundle: {p.name}")
            with open(p, "r", encoding="utf-8") as fp:
                cls.data[k] = json.load(fp)

    def test_all_json_files_exist(self):
        """Verifies that all 5 required JSON artifacts exist on disk."""
        for name, p in self.files.items():
            self.assertTrue(p.exists(), f"File {name} does not exist at {p}")
            self.assertGreater(p.stat().st_size, 1000, f"File {name} is unusually small ({p.stat().st_size} bytes)")

    def test_schema_validations(self):
        """Executes all programmatic schema validators."""
        results = validate_all_precomputed_artifacts()
        for fname, passed in results.items():
            self.assertTrue(passed, f"Validation failed for {fname}")

    def test_missing_artifact_and_schema_rejection(self):
        """Verifies that schema validators strictly reject missing keys or incomplete schemas."""
        with self.assertRaises(AssertionError):
            validate_overview_summary({})
        with self.assertRaises(AssertionError):
            validate_population_health({})
        with self.assertRaises(AssertionError):
            validate_model_performance({})
        with self.assertRaises(AssertionError):
            validate_threshold_analysis({})
        with self.assertRaises(AssertionError):
            validate_explainable_ai({})

    def test_numerical_consistency_overview(self):
        """Verifies overview metrics against dataset_partition_summary.csv."""
        overview = self.data["overview"]
        dims = overview["cohort_dimensions"]

        part_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "dataset_partition_summary.csv")
        full_row = part_df[part_df["partition"] == "Full Eligible Cohort"].iloc[0]

        self.assertEqual(dims["total_survey_respondents"], int(full_row["total_records"]))
        self.assertEqual(dims["positive_cases"], int(full_row["positive_cases"]))
        self.assertEqual(dims["negative_cases"], int(full_row["negative_cases"]))
        self.assertAlmostEqual(dims["unweighted_prevalence_pct"], float(full_row["unweighted_prevalence_pct"]), places=2)
        self.assertAlmostEqual(dims["survey_weighted_prevalence_pct"], float(full_row["survey_weighted_prevalence_pct"]), places=2)

    def test_numerical_consistency_model_performance(self):
        """Verifies champion metrics against model_validation_comparison.csv."""
        perf = self.data["performance"]
        val_models = perf["validation_models"]

        val_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "model_validation_comparison.csv")
        champ_row = val_df[val_df["model_key"] == "hist_gradient_boosting_unweighted"].iloc[0]

        champ_json = next(m for m in val_models if m["model_key"] == "hist_gradient_boosting_unweighted")
        self.assertAlmostEqual(champ_json["val_pr_auc"], float(champ_row["val_pr_auc"]), places=4)
        self.assertAlmostEqual(champ_json["val_roc_auc"], float(champ_row["val_roc_auc"]), places=4)
        self.assertAlmostEqual(champ_json["val_brier_score"], float(champ_row["val_brier_score"]), places=4)

    def test_numerical_consistency_threshold_analysis(self):
        """Verifies confusion matrix counts against threshold_optimization_analysis.csv."""
        th_data = self.data["threshold"]
        cm_080 = th_data["confusion_matrix_at_locked_080"]

        th_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "threshold_optimization_analysis.csv")
        row_080 = th_df[np.isclose(th_df["threshold"], 0.08, atol=0.005)].iloc[0]

        self.assertEqual(cm_080["tp"], int(row_080["tp"]))
        self.assertEqual(cm_080["fp"], int(row_080["fp"]))
        self.assertEqual(cm_080["tn"], int(row_080["tn"]))
        self.assertEqual(cm_080["fn"], int(row_080["fn"]))
        self.assertAlmostEqual(cm_080["sensitivity"], float(row_080["sensitivity"]), places=4)
        self.assertAlmostEqual(cm_080["specificity"], float(row_080["specificity"]), places=4)
        self.assertAlmostEqual(cm_080["npv"], float(row_080["npv"]), places=4)

    def test_numerical_consistency_xai(self):
        """Verifies top global predictors against xai_global_variable_importance.csv."""
        xai_data = self.data["xai"]
        global_imp = xai_data["global_grouped_importance"]

        global_df = pd.read_csv(PROJECT_ROOT / "results" / "tables" / "xai_global_variable_importance.csv")

        # Verify top 3 predictors match exactly
        self.assertEqual(global_imp[0]["source_variable"], global_df.iloc[0]["source_variable"])
        self.assertEqual(global_imp[1]["source_variable"], global_df.iloc[1]["source_variable"])
        self.assertEqual(global_imp[2]["source_variable"], global_df.iloc[2]["source_variable"])
        self.assertAlmostEqual(global_imp[0]["mean_pr_auc_drop"], float(global_df.iloc[0]["mean_pr_auc_drop"]), places=6)

    def test_locked_threshold_invariant(self):
        """Verifies that the threshold is recorded as 0.080 across all relevant files."""
        self.assertEqual(self.data["overview"]["champion_model"]["locked_threshold"], 0.080)
        self.assertEqual(self.data["threshold"]["locked_operational_threshold"], 0.080)
        self.assertEqual(self.data["threshold"]["confusion_matrix_at_locked_080"]["threshold"], 0.080)

    def test_exclusion_of_raw_respondent_records(self):
        """Verifies that no raw microdata rows or respondent-level feature vectors are exposed."""
        # 1. population_health should contain summary statistics, not individual rows
        pop = self.data["population"]
        self.assertNotIn("records", pop)
        self.assertNotIn("rows", pop)

        # 2. explainable_ai should only contain the 4 approved validation archetypes
        xai = self.data["xai"]
        archetypes = xai["case_archetypes_summary"]
        self.assertEqual(len(archetypes), 4)
        for a in archetypes:
            self.assertTrue(a["case_id"].startswith("VALIDATION_ID_"))
            self.assertIn(a["category"], ["True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)"])

    def test_file_size_constraints(self):
        """Verifies that each file is lightweight and the total bundle is < 250 KB for instant load."""
        total_size = 0
        for name, p in self.files.items():
            sz = p.stat().st_size
            total_size += sz
            self.assertLess(sz, 150_000, f"File {name} exceeds 150 KB ({sz} bytes)")
        self.assertLess(total_size, 250_000, f"Total data bundle exceeds 250 KB ({total_size} bytes)")


if __name__ == "__main__":
    unittest.main()
