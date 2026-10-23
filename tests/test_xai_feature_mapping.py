"""
test_xai_feature_mapping.py
---------------------------
Unit and validation tests for Milestone 7 XAI feature mapping.
Verifies preprocessor feature counts, mapping uniqueness, bijective alignment,
and source-variable coverage.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from src.features import feature_config
from src.xai import xai_config
from src.xai.validate_feature_mapping import extract_and_validate_feature_mapping


class TestXAIFeatureMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads preprocessor artifact and extracts feature names dynamically."""
        cls.preprocessor_path = xai_config.PREPROCESSOR_PATH
        cls.assertTrue(cls.preprocessor_path.exists(), f"Missing preprocessor at {cls.preprocessor_path}")
        cls.preprocessor = joblib.load(cls.preprocessor_path)
        cls.actual_feature_names = list(cls.preprocessor.get_feature_names_out())
        cls.actual_count = len(cls.actual_feature_names)
        cls.mapping_dict, cls.df_mapping = extract_and_validate_feature_mapping(save_if_valid=False)

    def test_preprocessor_artifact_exists_and_fitted(self):
        """Verify preprocessor artifact exists and has fitted attributes."""
        self.assertTrue(hasattr(self.preprocessor, "transformers_"))
        self.assertTrue(hasattr(self.preprocessor, "named_transformers_"))
        self.assertGreater(self.actual_count, 0)

    def test_feature_count_dynamic_matching(self):
        """Verify mapped feature count matches preprocessor output count dynamically."""
        self.assertEqual(len(self.df_mapping), self.actual_count)
        self.assertEqual(len(self.mapping_dict), self.actual_count)

    def test_mapping_uniqueness(self):
        """Verify that every mapped feature name is strictly unique."""
        names = self.df_mapping["transformed_feature_name"].tolist()
        self.assertEqual(len(names), len(set(names)), "Duplicate transformed feature names detected in mapping!")

    def test_bijective_alignment_with_artifact(self):
        """Verify exact bijective set equality and order alignment with preprocessor output."""
        mapped_names = list(self.mapping_dict.keys())
        self.assertEqual(set(mapped_names), set(self.actual_feature_names))
        self.assertEqual(mapped_names, self.actual_feature_names, "Transformed feature order does not match artifact output!")

    def test_source_variable_coverage(self):
        """Verify all 38 source predictors from feature_config are completely covered."""
        expected_sources = set(feature_config.NUMERIC_FEATURES + feature_config.CATEGORICAL_FEATURES)
        mapped_sources = set(self.df_mapping["source_variable"].unique())
        
        self.assertEqual(len(expected_sources), 38, "Expected exactly 38 source variables in feature_config.")
        self.assertEqual(mapped_sources, expected_sources, "Mapped source variables do not match feature_config!")

    def test_numeric_feature_mapping(self):
        """Verify 4 continuous numeric features are mapped correctly with prefix num__."""
        numeric_rows = self.df_mapping[self.df_mapping["feature_type"] == "numeric"]
        self.assertEqual(len(numeric_rows), 4)
        for col in feature_config.NUMERIC_FEATURES:
            expected_name = f"num__{col}"
            self.assertIn(expected_name, self.mapping_dict)
            self.assertEqual(self.mapping_dict[expected_name]["source_variable"], col)
            self.assertEqual(self.mapping_dict[expected_name]["category_value"], "N/A")

    def test_missing_sentinels_identified(self):
        """Verify explicit -1.0 missing sentinels are flagged accurately."""
        sentinel_rows = self.df_mapping[self.df_mapping["is_missing_sentinel"]]
        self.assertGreater(len(sentinel_rows), 0, "Missing sentinel indicators should be detected.")
        for _, row in sentinel_rows.iterrows():
            self.assertTrue(
                "-1.0" in row["transformed_feature_name"] or "-1" in row["transformed_feature_name"],
                f"Sentinel row name {row['transformed_feature_name']} does not contain -1"
            )

    def test_locked_threshold_configuration(self):
        """Verify xai_config reads the exact locked threshold (0.080) from Milestone 6."""
        self.assertAlmostEqual(xai_config.LOCKED_THRESHOLD, 0.080, places=3)
        self.assertEqual(xai_config.RANDOM_STATE, 42)
        self.assertEqual(xai_config.PERMUTATION_REPEATS, 10)


if __name__ == "__main__":
    unittest.main()
