import unittest
from scripts.embeddinggemma2_v3_compare import fused100,summarize,views_for_site
from tests.test_embeddinggemma2_v2_data import WordTokenizer


class Initial100Comparison(unittest.TestCase):
    def test_rrf_cap_dedup_and_agreement(self):
        a=[dict(siret=str(i),siren='a') for i in range(100)]
        b=[dict(siret=str(i),siren='a') for i in range(50,150)]
        rows=fused100([a,b]); self.assertEqual(len(rows),100)
        self.assertEqual(len({r['siret'] for r in rows}),100)
        self.assertEqual(rows[0]['siret'],'50')
        with self.assertRaises(ValueError): fused100([a+[a[0]]])

    def test_all_human_denominator_and_exact_operational_separate(self):
        rows=[dict(exact=True,siren=True,operational=True),dict(exact=False,siren=True,operational=True)]
        result=summarize(rows)
        self.assertEqual(result['queries'],2); self.assertEqual(result['exact_recall'],.5)
        self.assertEqual(result['operational_hits'],2)
        self.assertLess(result['wilson95_lower'],.5)
        self.assertGreater(result['wilson95_upper'],.5)

    def test_sibling_alias_not_copied_to_site(self):
        doc=dict(own_names=['SITE A'],all_addresses=['4 RUE DE PARIS'],postcode='75000')
        views=views_for_site(doc,dict(legal=['ENTREPRISE'],own_site=['ANCIEN SITE A'],sites=['SIBLING B']),WordTokenizer())
        self.assertIn('ANCIEN SITE A',' '.join(views)); self.assertNotIn('SIBLING B',' '.join(views))
