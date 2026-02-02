import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xgb_matcher.routing_risk import BASE_FEATURES, build_feature_row, default_feature_columns


class TestRoutingRiskFeatures(unittest.TestCase):
    def test_build_feature_row(self):
        top1 = {
            "score": 0.9,
            "name_jaro_max": 0.8,
            "addr_jaro": 0.7,
        }
        top2 = {
            "score": 0.6,
            "name_jaro_max": 0.5,
            "addr_jaro": 0.4,
        }

        feature_names = default_feature_columns(BASE_FEATURES)
        row = build_feature_row(top1, top2, feature_names)

        col_index = {name: idx for idx, name in enumerate(feature_names)}
        self.assertAlmostEqual(row[col_index["score_top1"]], 0.9, places=6)
        self.assertAlmostEqual(row[col_index["score_top2"]], 0.6, places=6)
        self.assertAlmostEqual(row[col_index["score_gap"]], 0.3, places=6)
        self.assertAlmostEqual(row[col_index["score_ratio"]], 1.5, places=6)

        self.assertAlmostEqual(row[col_index["top1_name_jaro_max"]], 0.8, places=6)
        self.assertAlmostEqual(row[col_index["top2_name_jaro_max"]], 0.5, places=6)
        self.assertAlmostEqual(row[col_index["delta_name_jaro_max"]], 0.3, places=6)

    def test_score_ratio_floor(self):
        top1 = {
            "score": 0.5,
            "name_jaro_max": 0.1,
            "addr_jaro": 0.2,
        }
        top2 = {
            "score": 0.0,
            "name_jaro_max": 0.05,
            "addr_jaro": 0.1,
        }

        feature_names = default_feature_columns(BASE_FEATURES)
        row = build_feature_row(top1, top2, feature_names)

        col_index = {name: idx for idx, name in enumerate(feature_names)}
        self.assertAlmostEqual(row[col_index["score_ratio"]], 500.0, places=6)


if __name__ == "__main__":
    unittest.main()
