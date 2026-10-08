import unittest
from scripts.embeddinggemma2_v10_representative import family_fusion, project_population, wilson


def candidate(key): return dict(siret=key, siren='p')


class RepresentativeFusion(unittest.TestCase):
    def test_correlated_duplicates_are_not_double_votes(self):
        rows = [candidate('a'), candidate('b')]
        self.assertEqual(family_fusion(rows, [], []), family_fusion(rows, rows, []))

    def test_both_dense_channels_contribute_and_final_cap(self):
        rows = family_fusion([candidate(str(i)) for i in range(100)],
            [candidate(str(i)) for i in range(100, 200)], [])
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({r['siret'] for r in rows}), 100)
        self.assertIn('0', {r['siret'] for r in rows})
        self.assertIn('100', {r['siret'] for r in rows})

    def test_invalid_channel_cap_duplicate_and_parent_rejected(self):
        for rows in ([candidate('a')]*2, [candidate(str(i)) for i in range(101)]):
            with self.assertRaises(ValueError): family_fusion(rows, [], [])
        with self.assertRaises(ValueError):
            family_fusion([candidate('a')], [dict(siret='a', siren='other')], [])

    def test_projection_keeps_missing_geo_and_strips_labels(self):
        row = dict(query_id='q', fold=2, crm_name='', crm_number='', crm_suffix='',
            crm_street='', crm_postcode='', crm_insee='', siret='SECRET', siren='SECRET')
        result = project_population([row])
        self.assertEqual(len(result), 1)
        self.assertNotIn('siret', result[0]); self.assertNotIn('siren', result[0])
        with self.assertRaises(ValueError): project_population([{**row, 'fold':1}])

    def test_uncertainty_not_certainty(self):
        self.assertLess(wilson(80, 80)[0], .992)
        self.assertLess(wilson(2194, 2212)[0], .99)
        self.assertEqual(wilson(79,80), wilson(79,80))


if __name__ == '__main__': unittest.main()
