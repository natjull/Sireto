import unittest
import numpy as np
from scripts.embeddinggemma2_compare import top100,query_text


class Comparison(unittest.TestCase):
    def test_streaming_exact100_matches_whole_corpus(self):
        docs=[dict(siret=f'{i:014d}',siren=f'{i//2:09d}') for i in range(537)]
        scores=np.random.default_rng(4).integers(0,10,len(docs)).astype(float)
        best=[]
        for i in range(0,len(docs),37): best=top100(best,scores[i:i+37],docs[i:i+37])
        self.assertEqual(best,top100([],scores,docs))
        self.assertEqual(len(best),100)

    def test_identifiers_not_model_text(self):
        q=dict(crm_name='CLIENT 123456789',crm_number='4',crm_suffix='',crm_street='RUE 12345678900012',crm_postcode='75000')
        text=query_text(q)
        self.assertNotIn('123456789',text)
        self.assertIn('75000',text)
