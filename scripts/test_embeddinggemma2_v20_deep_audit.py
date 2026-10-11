import unittest
import numpy as np
from scripts import embeddinggemma2_v20_deep_audit as audit


class AttributionTest(unittest.TestCase):
    def test_rank_with_deterministic_id_ties(self):
        keys=np.array([20,10,30],dtype=np.uint64)
        values=np.array([.5,.5,.7],dtype=np.float32)
        self.assertEqual([audit.rank(values,keys,i) for i in range(3)],[3,2,1])

    def test_raw_number_reconstructed_only_if_absent(self):
        row=dict(street_number='12',street_number_suffix='BIS',raw_value='RUE ALPHA')
        self.assertEqual(audit.source_address(row),'12 BIS RUE ALPHA')
        row['raw_value']='12 BIS RUE ALPHA'
        self.assertEqual(audit.source_address(row),'12 BIS RUE ALPHA')

    def test_heuristic_address_can_override_dense(self):
        f={k:np.array([.1],dtype=np.float32) for k in audit.FIELDS}
        f['site'][0]=.6;f['identity'][0]=.5;f['address'][0]=.9
        self.assertEqual(int(audit.native_branches(f).argmax(axis=0)[0]),2)

    def test_weighting_preserves_all_rows(self):
        self.assertAlmostEqual(audit.weighted([dict(weight=1,hit=True),dict(weight=9,hit=False)],'hit'),.1)

    def test_identifier_normalization(self):
        self.assertEqual(audit.normalized(['Nom 123456789','NOM']),{'NOM'})


if __name__=='__main__':unittest.main()
