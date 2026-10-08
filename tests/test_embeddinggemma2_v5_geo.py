import unittest
import numpy as np
from scripts.embeddinggemma2_v5_geo import geo_radius,nearest100,weighted_fusion


class IndependentGeography(unittest.TestCase):
    def test_ban_missing_and_confidence_expands_radius(self):
        g=dict(spatial_crs_compatible=False)
        self.assertIsNone(geo_radius(g))
        g=dict(spatial_crs_compatible=True,ban_uncertainty_m=40,ban_declared_weak=False)
        self.assertEqual(geo_radius(g),300)
        self.assertEqual(geo_radius({**g,'ban_declared_weak':True}),1000)
        self.assertEqual(geo_radius({**g,'ban_uncertainty_m':600}),1800)

    def test_streaming_exact_nearest100_and_boundary_ties(self):
        keys=np.arange(513,dtype=np.uint64); distances=np.random.default_rng(3).integers(0,30,len(keys))
        best=[]
        for i in range(0,len(keys),31): best=nearest100(best,distances[i:i+31],keys[i:i+31])
        self.assertEqual(best,nearest100([],distances,keys)); self.assertEqual(len(best),100)

    def test_confidence_weight_is_used_and_output100(self):
        a=[dict(siret=str(i),siren='p') for i in range(100)]; b=[dict(siret=str(i),siren='p') for i in range(100,200)]
        rows=weighted_fusion([(a,1.),(b,.5)])
        self.assertEqual(rows[0]['siret'],'0'); self.assertEqual(len(rows),100)
        self.assertEqual(weighted_fusion([(a,1.),(a,.5)])[0]['score'],1.5/61)
