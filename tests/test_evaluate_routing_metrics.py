import unittest

import pandas as pd

from scripts import evaluate_routing as er


class TestEvaluateRoutingMetrics(unittest.TestCase):
    def test_merge_data_strict_and_siren(self):
        routed = pd.DataFrame(
            {
                "crm_id": ["1", "2", "3"],
                "chosen_siret_final": [
                    "12345678900011",
                    "12345678900022",
                    "99999999900011",
                ],
                "final_status": ["AUTO", "AUTO", "REVIEW"],
            }
        )

        gt = pd.DataFrame(
            {
                "crm_id": ["1", "2", "2", "3"],
                "siret_gt": [
                    "12345678900011",
                    "12345678900033",
                    "12345678900044",
                    "88888888800011",
                ],
            }
        )

        merged = er.merge_data(routed, gt)
        row1 = merged[merged["crm_id"] == "1"].iloc[0]
        row2 = merged[merged["crm_id"] == "2"].iloc[0]
        row3 = merged[merged["crm_id"] == "3"].iloc[0]

        self.assertTrue(row1["is_correct_strict"])
        self.assertFalse(row2["is_correct_strict"])
        self.assertTrue(row2["is_correct_siren"])
        self.assertFalse(row3["is_correct_siren"])


if __name__ == "__main__":
    unittest.main()
