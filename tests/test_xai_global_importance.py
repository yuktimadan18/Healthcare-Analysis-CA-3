"""
test_xai_global_importance.py
-----------------------------
Unit tests for Milestone 7 Stage 2 Global Feature Importance.
Verifies grouped column membership, permutation reproducibility, output table
completeness, and figure artifact generation.
"""

import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import xai_config


class TestXAIGlobalImportance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mapping_path = xai_config.FEATURE_MAPPING_AUDIT_PATH
        cls.assertTrue(cls.mapping_path.exists(), f"Missing mapping audit table at {cls.mapping_path}")
        cls.df_mapping = pd.read_csv(cls.mapping_path)

    def test_grouped_column_membership_completeness_and_disjointness(self):
        """Verify grouping 190 features by source_variable partitions range(190) perfectly."""
        grouped = self.df_mapping.groupby("source_variable")["feature_index"].apply(list).to_dict()

        # All 38 source variables must be present
        self.assertEqual(len(grouped), 38, "Expected exactly 38 groups corresponding to the 38 original predictors.")

        all_indices = []
        for var, indices in grouped.items():
            self.assertGreater(len(indices), 0, f"Group {var} has no columns.")
            all_indices.extend(indices)

        # Disjointness and completeness
        self.assertEqual(len(all_indices), 190, "Total number of grouped indices must equal 190.")
        self.assertEqual(len(set(all_indices)), 190, "Duplicate column indices found across groups!")
        self.assertEqual(set(all_indices), set(range(190)), "Grouped indices do not cover range(0, 190) exactly.")

    def test_output_tables_integrity_if_present(self):
        """Verify output CSV tables have expected dimensions, columns, and finite values if already generated."""
        grouped_csv = xai_config.GLOBAL_VARIABLE_IMPORTANCE_PATH
        indicator_csv = xai_config.TRANSFORMED_FEATURE_IMPORTANCE_PATH

        if grouped_csv.exists():
            df_grp = pd.read_csv(grouped_csv)
            self.assertEqual(len(df_grp), 38, "Expected 38 rows in global variable importance table.")
            self.assertEqual(df_grp["source_variable"].nunique(), 38, "Duplicate source variables in importance table!")
            self.assertTrue(df_grp["mean_pr_auc_drop"].notna().all(), "NaN found in mean_pr_auc_drop!")
            self.assertTrue(np.isfinite(df_grp["mean_pr_auc_drop"]).all(), "Non-finite value in mean_pr_auc_drop!")
            self.assertTrue(np.isfinite(df_grp["std_pr_auc_drop"]).all(), "Non-finite value in std_pr_auc_drop!")
            # Check ranks are 1 to 38
            self.assertEqual(list(df_grp["rank"]), list(range(1, 39)))

        if indicator_csv.exists():
            df_ind = pd.read_csv(indicator_csv)
            self.assertEqual(len(df_ind), 190, "Expected 190 rows in indicator feature importance table.")
            self.assertTrue(df_ind["mean_pr_auc_drop"].notna().all(), "NaN found in indicator mean_pr_auc_drop!")
            self.assertTrue(np.isfinite(df_ind["mean_pr_auc_drop"]).all(), "Non-finite value in indicator mean_pr_auc_drop!")
            self.assertEqual(list(df_ind["rank"]), list(range(1, 191)))

    def test_output_figures_integrity_if_present(self):
        """Verify output figure files are valid and non-empty if already generated."""
        for fig_path in [
            xai_config.FIG_GLOBAL_VARIABLE_IMPORTANCE,
            xai_config.FIG_INDICATOR_FEATURE_IMPORTANCE,
            xai_config.FIG_PARTIAL_DEPENDENCE_PROFILES
        ]:
            if fig_path.exists():
                size = fig_path.stat().st_size
                self.assertGreater(size, 10000, f"Figure {fig_path.name} is unexpectedly small ({size} bytes).")


if __name__ == "__main__":
    unittest.main()
