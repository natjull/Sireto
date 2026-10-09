import unittest
from scripts.embeddinggemma2_v16_failure_audit import overlap, fusion_trace, coordinate_status


class AuditTests(unittest.TestCase):
    def test_overlap_descriptive_not_alias_expansion(self):
        self.assertEqual(overlap('SAS École du Parc',['ECOLE PARC']),1.)
        self.assertEqual(overlap('ATOL',['MODERNE 66']),0.)

    def test_single_channel_high_rank_can_be_evicted(self):
        outputs={k:{'q':[]} for k in ('gemma2','lexical','geo')}
        outputs['gemma2']['q']=[{'siret':str(i)} for i in range(50)]+[{'siret':'GT'}]+[{'siret':str(i)} for i in range(50,99)]
        outputs['lexical']['q']=[{'siret':str(i)} for i in range(100)]
        outputs['geo']['q']=[{'siret':str(i)} for i in range(100)]
        result=fusion_trace(outputs,'q','GT',1.)
        self.assertEqual(result['channel_ranks'],{'gemma2':51})
        self.assertGreater(result['fusion_rank'],100)

    def test_geometry_failure_and_cap_separated(self):
        g=dict(spatial_crs_compatible=False)
        self.assertEqual(coordinate_status(g,None,None,None)['status'],'QUERY_BAN_UNRESOLVED_OR_INCOMPATIBLE')
        g=dict(spatial_crs_compatible=True,ban_x_l93=700000,ban_y_l93=6500000)
        self.assertEqual(coordinate_status(g,[700500,6500000],300,100)['status'],'OUTSIDE_RADIUS')
        self.assertEqual(coordinate_status(g,[700200,6500000],300,100)['status'],'WITHIN_RADIUS_BEYOND_NEAREST100')


if __name__=='__main__': unittest.main()
