import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.embeddinggemma2_v17_structural_retrieval import normalize, choose_street, fuse, push100, finish, similarity, evaluate


def candidate(i, score=1.):
    key=f'{i:014d}'
    return dict(siret=key,siren=key[:9],score=score)


class StructuralRetrieval(unittest.TestCase):
    def test_identifiers_never_features(self):
        self.assertEqual(normalize('SARL École & Cie 123456789 12345678900012'),'ECOLE ET CIE')

    def test_street_abbreviation_and_missing_number(self):
        self.assertEqual(normalize('87 AV. du Général',street=True),'AVENUE DU GENERAL')
        q=dict(crm_street='AV du General',crm_insee='44109')
        g=choose_street(q,{('44109','AVENUE DU GENERAL'):[(350000,6700000),(352000,6700000)]})
        self.assertEqual(g['precision'],'STREET_FOOTPRINT_NOT_HOUSE')
        self.assertTrue(g['ban_declared_weak'])
        self.assertEqual(g['footprint_radius_m'],1300)

    def test_street_ambiguity_refused(self):
        q=dict(crm_street='RUE DU GENERAL',crm_insee='44109')
        g=choose_street(q,{('44109','RUE DU GENERAL A'):[(350000,6700000)],('44109','RUE DU GENERAL B'):[(350010,6700000)]})
        self.assertIsNone(g)

    def test_unknown_dom_and_no_street_remain_unresolved(self):
        for zone,street in [('', 'RUE A'),('97105','RUE A'),('44109','')]:
            self.assertIsNone(choose_street(dict(crm_street=street,crm_insee=zone),{}))

    def test_normalized_street_variants_merge_points(self):
        g=choose_street(dict(crm_street='AV GENERAL',crm_insee='44109'),{
            ('44109','AV GENERAL'):[(350000,6700000)],('44109','AVENUE GENERAL'):[(352000,6700000)]})
        self.assertEqual(g['source_points'],2)

    def test_max_not_consensus_double_vote(self):
        first=[candidate(1),candidate(2)]; second=[candidate(2)]
        self.assertEqual(fuse([(first,1),(second,1)],'sum')[0]['siret'],candidate(2)['siret'])
        self.assertEqual(fuse([(first,1),(second,1)],'max')[0]['siret'],candidate(1)['siret'])

    def test_cap_and_all_eligible_not_nearest_pruning(self):
        heap=[]
        for i in range(150): push100(heap,candidate(i,i/150))
        self.assertEqual(len(finish(heap)),100)
        self.assertEqual(finish(heap)[0]['siret'],candidate(149)['siret'])
        self.assertEqual(finish(heap)[-1]['siret'],candidate(50)['siret'])
        self.assertEqual(len(fuse([([candidate(i) for i in range(100)],1),([candidate(i) for i in range(100,200)],1)],'max')),100)

    def test_blank_names_no_fake_similarity(self):
        self.assertEqual(similarity('',('MAIRIE',)),0)
        self.assertEqual(similarity('MAIRIE',()),0)

    def test_incomplete_or_unowned_generation_cannot_open_labels(self):
        for exit_code,owned,status in [(1,True,'COMPLETE_SEALED_ALL526'),
                                        (0,False,'COMPLETE_SEALED_ALL526'),
                                        (0,True,'PARTIAL')]:
            opened=[]
            def read(path):
                opened.append(Path(path).name)
                if opened[-1]=='launch.json': return dict(implementation={},recipe={},helpers=[])
                if opened[-1]=='generate_exit.json': return dict(exit_code=exit_code,own_child_exit_confirmed=owned)
                if opened[-1]=='generation.json': return dict(status=status)
                raise AssertionError('Unsealed generation must not open any other file')
            with patch('scripts.embeddinggemma2_v17_structural_retrieval.io.read',side_effect=read), \
                 patch('scripts.embeddinggemma2_v17_structural_retrieval.verify'):
                with self.assertRaisesRegex(ValueError,'BEFORE labels'): evaluate(Path('/tmp/v17_unit_only'))
            self.assertEqual(opened,['launch.json','generate_exit.json','generation.json'])

    def test_fusion_rejects_fixed_larger_source_pool(self):
        with self.assertRaises(ValueError): fuse([([candidate(i) for i in range(101)],1)],'max')
        with self.assertRaises(ValueError): fuse([([candidate(1),candidate(1)],1)],'sum')


if __name__=='__main__': unittest.main()
