import ast
from pathlib import Path
import unittest
import numpy as np
from scripts import embeddinggemma2_v19_proof_retrieval as r
from scripts import embeddinggemma2_v19_official_audit as audit


class NativeProofTests(unittest.TestCase):
    def test_embedded_number_suffix(self):
        self.assertEqual(r.address('661B RUE BENOIT MULSANT'),('661','B','RUE BENOIT MULSANT'))
        self.assertEqual(r.address('4 BIS AV DU GENERAL'),('4','BIS','AVENUE DU GENERAL'))

    def test_unknown_number_is_not_an_exclusion(self):
        self.assertEqual(r.address('ROUTE DE TILLY'),('','','ROUTE DE TILLY'))
        proof,street=r.address_proof(r.address('ROUTE DE TILLY'),[r.address('19 ROUTE DE TILLY')],0.)
        self.assertEqual(street,1.);self.assertAlmostEqual(proof,.8)

    def test_suffix_uncertainty_and_conflict(self):
        self.assertEqual(r.number_grade('4','','4','BIS'),.8)
        self.assertEqual(r.number_grade('4','BIS','4','TER'),0.)
        self.assertEqual(r.number_grade('4','BIS','4','BIS'),1.)

    def test_no_address_stitching(self):
        query=r.address('4 RUE ALPHA')
        value,_=r.address_proof(query,[r.address('4 RUE BETA'),r.address('8 RUE ALPHA')],0.)
        self.assertLess(value,.85)

    def test_identifiers_and_masked_values_not_features(self):
        self.assertEqual(r.address('39922665300935'),('','',''))
        self.assertEqual(r.address('ND'),('','',''))

    def test_generic_street_has_no_distinctive_proof(self):
        self.assertEqual(r.street_similarity('RUE','RUE DU PORT'),0.)

    def test_parent_cannot_decrease_site_proof(self):
        s=np.array([[.75]],dtype=np.float32);z=np.zeros_like(s)
        self.assertEqual(float(r.native_score(s,z,z,z,z,z)[0,0]),.75)

    def test_explicit_geo_uncertainty_is_bounded(self):
        z=np.zeros((1,1),dtype=np.float32);g=np.ones_like(z)
        self.assertAlmostEqual(float(r.native_score(z,z,z,z,z,g)[0,0]),.05)
        self.assertAlmostEqual(float(r.native_score(z,z,z,z,z,.5*g)[0,0]),.025)

    def test_generation_does_not_read_audit_or_labels(self):
        tree=ast.parse(Path(r.__file__).read_text())
        for f in tree.body:
            if isinstance(f,ast.FunctionDef) and f.name=='generate':
                strings=[n.value for n in ast.walk(f) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                self.assertFalse(any(bad in v for v in strings for bad in ('outcomes.json','findings.json','validation_membership','label_side','cases.json')))

    def test_recipe_not_claimed_learned_or_full_national(self):
        self.assertTrue(r.RECIPE['no_grid']);self.assertTrue(r.RECIPE['no_fixed_prepool'])
        self.assertIn('UNTRAINED',r.RECIPE['weights'])
        self.assertIn('NOT_NEW_NATIONAL',r.RECIPE['source_coverage'])

    def test_audit_categories_are_disjoint(self):
        channels={k:{'q':[]} for k in ('joint','lexical','oldgeo','site','newgeo','family')}
        self.assertEqual(audit.categories('q','00000000100001',channels)[0],'ALL_CHANNELS_MISS')
        channels['site']['q']=[{'siret':'00000000100001'}]
        self.assertEqual(audit.categories('q','00000000100001',channels)[0],'OTHER_CHANNEL_ONLY')
        channels['oldgeo']['q']=[{'siret':'00000000100001'}]
        self.assertEqual(audit.categories('q','00000000100001',channels)[0],'FUSION_EVICTION')


if __name__=='__main__':unittest.main()
