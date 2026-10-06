import unittest
from scripts.embeddinggemma2_night import grouped_split, clean_text


class NightIsolation(unittest.TestCase):
    def test_siren_group_disjoint_and_order_independent(self):
        rows=[{'human_ground_truth_siren':str(100000000+i//3)} for i in range(300)]
        forward=grouped_split(rows); reverse=grouped_split(rows[::-1])
        self.assertEqual(forward,reverse)
        self.assertEqual(set(forward.values()),{'train','validation','test'})
        self.assertEqual(len(forward),100)

    def test_identifiers_not_features(self):
        text=clean_text(['ETS 123456789'],['SIRET 12345678900012 18 RUE DU PORT'],'75001')
        self.assertNotIn('123456789',text)
        self.assertIn('18 RUE DU PORT',text)
        self.assertIn('75001',text)


if __name__=='__main__': unittest.main()
