import unittest
from unittest.mock import patch
import numpy as np
from scripts import perplexity_late_comparison as run


class ComparisonTests(unittest.TestCase):
    def test_complete_query_scope_unknown_retained_and_label_blind(self):
        zones=[dict(zone=z,queries=n,indexed_sites=s) for z,n,s in
            [('A',3,400),('B',4,700),('C',5,2000),('D',3,6000),('E',6,29000),('',3,0)]]
        panel=[dict(query_id=f'{z["zone"]}{i}',fold=2,crm_name='N',crm_number='',crm_suffix='',
            crm_street='S',crm_postcode='',crm_insee=z['zone'],truth='SECRET')
            for z in zones for i in range(z['queries'])]
        selected,queries=run.select(zones,panel)
        self.assertEqual({z['zone'] for z in selected},{'B','C','E',''})
        self.assertEqual(len(queries),18)
        self.assertFalse(any('truth' in q for q in queries))
        panel[3]['fold']=1
        with self.assertRaises(ValueError): run.select(zones,panel)

    def test_ragged_cache_is_lossless_float32(self):
        values=[np.arange(256,dtype=np.float32).reshape(2,128),np.ones((3,128),dtype=np.float32)]
        packed,offsets=run.pack_shard(values)
        self.assertEqual(offsets.tolist(),[0,2,5])
        for i,value in enumerate(values):
            np.testing.assert_array_equal(packed[offsets[i]:offsets[i+1]],value)

    def test_missing_queries_duplicate_or_overcap_refused(self):
        with self.assertRaises(ValueError): run.validate_candidates([],{'q'})
        with self.assertRaises(ValueError):
            run.validate_candidates([dict(query_id='q',candidates=[dict(siret='1')]*2)],{'q'})
        with self.assertRaises(ValueError):
            run.validate_candidates([dict(query_id='q',candidates=[dict(siret=str(i)) for i in range(101)])],{'q'})

    def test_labels_cannot_open_without_all_generations(self):
        with patch.object(run,'seal_all',side_effect=ValueError('Incomplete generation')):
            with patch.object(run.io,'read',side_effect=AssertionError('Label read')):
                with self.assertRaisesRegex(ValueError,'Incomplete generation'): run.evaluate()
