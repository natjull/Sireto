import unittest

import pandas as pd

from src.xgb_matcher import selective


class TestCalibrateRoutingThresholdsTop1(unittest.TestCase):
    def test_prepare_top1_rank_and_gap(self):
        infer = pd.DataFrame(
            {
                "crm_id": ["A", "A", "B", "B"],
                "rank": [1, 2, 1, 2],
                "score": [0.90, 0.80, 0.70, 0.60],
                "siret_candidate": [
                    "12345678900011",
                    "12345678900022",
                    "98765432100011",
                    "98765432100022",
                ],
            }
        )

        top1 = selective.prepare_top1(infer)

        self.assertEqual(len(top1), 2)
        a_row = top1[top1["crm_id"] == "A"].iloc[0]
        b_row = top1[top1["crm_id"] == "B"].iloc[0]

        self.assertEqual(a_row["siret_candidate"], "12345678900011")
        self.assertEqual(b_row["siret_candidate"], "98765432100011")
        self.assertAlmostEqual(float(a_row["score_gap"]), 0.10, places=6)
        self.assertAlmostEqual(float(b_row["score_gap"]), 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
