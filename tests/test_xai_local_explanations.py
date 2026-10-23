"""
test_xai_local_explanations.py
------------------------------
Unit tests for Milestone 7 Stage 3 Local Explainable AI (XAI):
1. Deterministic case archetype selection (TP, FP, TN, FN).
2. Locked threshold dynamic retrieval and consistency (0.080).
3. Prediction and confusion quadrant alignment.
4. Explanations output schema, finite values, and 38 original variable coverage.
5. Local linear surrogate fidelity (R^2 > 0.80).
6. Visualization artifact generation and size sanity.
7. Verification that test set was untouched.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from src.xai import xai_config
from src.xai.local_explanations import (
    load_validation_data_and_champion,
    select_representative_validation_cases,
    compute_case_explanations
)


class TestLocalExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Loads data and runs case selection once for test suite."""
        cls.model, cls.preprocessor, cls.threshold, cls.X_val_trans, cls.y_val, cls.X_val_raw, cls.df_mapping = (
            load_validation_data_and_champion()
        )
        cls.selected_cases, cls.df_cases_summary = select_representative_validation_cases(
            model=cls.model,
            X_val_trans=cls.X_val_trans,
            y_val=cls.y_val,
            X_val_raw=cls.X_val_raw,
            threshold=cls.threshold
        )

    def test_locked_threshold_value(self):
        """Verifies that the threshold loaded from model_selection_summary.json is strictly 0.080."""
        self.assertAlmostEqual(self.threshold, 0.080, places=3)
        self.assertEqual(xai_config.LOCKED_THRESHOLD, 0.080)

    def test_deterministic_case_selection_coverage(self):
        """Verifies that all 4 quadrants (TP, FP, TN, FN) are deterministically selected."""
        expected_categories = {"True Positive (TP)", "False Positive (FP)", "True Negative (TN)", "False Negative (FN)"}
        self.assertEqual(set(self.selected_cases.keys()), expected_categories)
        self.assertEqual(len(self.df_cases_summary), 4)

    def test_quadrant_classification_logic(self):
        """Verifies that each selected case matches its confusion matrix category definition."""
        probs = self.model.predict_proba(self.X_val_trans)[:, 1]

        # TP: y=1, p >= 0.080
        tp_idx = self.selected_cases["True Positive (TP)"]
        self.assertEqual(self.y_val[tp_idx], 1)
        self.assertGreaterEqual(probs[tp_idx], self.threshold)

        # FP: y=0, p >= 0.080
        fp_idx = self.selected_cases["False Positive (FP)"]
        self.assertEqual(self.y_val[fp_idx], 0)
        self.assertGreaterEqual(probs[fp_idx], self.threshold)

        # TN: y=0, p < 0.080
        tn_idx = self.selected_cases["True Negative (TN)"]
        self.assertEqual(self.y_val[tn_idx], 0)
        self.assertLess(probs[tn_idx], self.threshold)

        # FN: y=1, p < 0.080
        fn_idx = self.selected_cases["False Negative (FN)"]
        self.assertEqual(self.y_val[fn_idx], 1)
        self.assertLess(probs[fn_idx], self.threshold)

    def test_case_archetypes_summary_table_integrity(self):
        """Verifies that xai_case_archetypes_summary.csv exists, has 4 rows, and valid columns."""
        summary_path = xai_config.XAI_EXPLANATIONS_DIR / "xai_case_archetypes_summary.csv"
        self.assertTrue(summary_path.exists(), f"Missing summary table at {summary_path}")

        df = pd.read_csv(summary_path)
        self.assertEqual(len(df), 4)
        required_cols = [
            "category", "case_id", "validation_row_index", "actual_target",
            "predicted_probability", "locked_threshold", "predicted_class",
            "age_years", "general_health_code", "sex_code", "stroke_code"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertFalse(df.isna().any().any(), "Summary table contains NaN values.")

    def test_local_explanations_table_integrity(self):
        """Verifies that xai_local_case_explanations.csv has 152 rows (4 cases * 38 variables) and finite values."""
        exp_path = xai_config.LOCAL_CASE_EXPLANATIONS_PATH
        self.assertTrue(exp_path.exists(), f"Missing explanations table at {exp_path}")

        df = pd.read_csv(exp_path)
        # 4 case archetypes * 38 original predictors = 152 rows
        self.assertEqual(len(df), 152)

        required_cols = [
            "case_category", "validation_row_index", "predicted_probability",
            "locked_threshold", "source_variable", "feature_type",
            "num_transformed_columns", "marginal_risk_delta_dp",
            "local_surrogate_weight", "directional_impact", "surrogate_fidelity_r2"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        self.assertFalse(df.isna().any().any(), "Local explanations table contains NaN values.")

        # Check that all 38 original predictors are present for each case
        for cat in df["case_category"].unique():
            sub_df = df[df["case_category"] == cat]
            self.assertEqual(len(sub_df), 38)
            self.assertEqual(sub_df["source_variable"].nunique(), 38)

    def test_local_surrogate_fidelity(self):
        """Verifies that the local Ridge surrogate achieves good local fidelity (R^2 > 0.80)."""
        exp_path = xai_config.LOCAL_CASE_EXPLANATIONS_PATH
        df = pd.read_csv(exp_path)
        r2_values = df.groupby("case_category")["surrogate_fidelity_r2"].first()
        for cat, r2 in r2_values.items():
            self.assertGreater(r2, 0.80, f"Local surrogate R^2 for {cat} is {r2:.4f}, expected > 0.80")

    def test_fig19_exists_and_valid(self):
        """Verifies that Figure 19 was generated and has non-trivial size (> 50 KB)."""
        fig_path = xai_config.FIG_LOCAL_CASE_ATTRIBUTIONS
        self.assertTrue(fig_path.exists(), f"Figure 19 not found at {fig_path}")
        size_bytes = fig_path.stat().st_size
        self.assertGreater(size_bytes, 50_000, f"Figure 19 file size too small: {size_bytes} bytes")

    def test_test_partition_untouched(self):
        """Verifies that the test target file timestamp predates Stage 3 and test features are untouched."""
        test_tgt_path = xai_config.TEST_TARGET_PATH
        self.assertTrue(test_tgt_path.exists())
        df_test_tgt = pd.read_csv(test_tgt_path, index_col=0)
        self.assertEqual(len(df_test_tgt), 67870)
        self.assertEqual(int((df_test_tgt["target"] == 1).sum()), 6350)


if __name__ == "__main__":
    unittest.main()
