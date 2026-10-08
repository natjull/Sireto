import copy
import unittest
from unittest.mock import patch
from scripts import embeddinggemma2_v15_probability_pilot as pilot


class ProbabilityPilot(unittest.TestCase):
    def fixture(self):
        zones=[]
        for h,sites,n in (('s',500,145),('m',5000,120),('d',20000,50),('l',100000,10),('v',300000,4)):
            zones.extend(dict(zone=f'{h}{i}',indexed_sites=sites,queries=1) for i in range(n))
        zones[0]['zone']=''
        # All 2212 rows are query-only; repeated sites of a commune are retained.
        for i in range(2212-len(zones)): zones[i%len(zones)]['queries']+=1
        panel=[]
        for z in zones:
            for i in range(z['queries']):
                panel.append(dict(query_id=f'{z["zone"]}:{i}',fold=2,crm_name='n',crm_number='',
                    crm_suffix='',crm_street='r',crm_postcode='',crm_insee=z['zone']))
        return dict(zones=zones),panel

    def test_probability_sample_is_deterministic_and_complete_communes(self):
        census,panel=self.fixture(); a=pilot.sample(census,panel); b=pilot.sample(census,list(reversed(panel)))
        self.assertEqual({q['query_id'] for q in a[1]},{q['query_id'] for q in b[1]})
        self.assertEqual(len(a[0]),295)
        self.assertIn('',{q['crm_insee'] for q in a[1]})
        self.assertEqual(sum(z['queries'] for z in a[0]),len(a[1]))
        self.assertEqual(sum(s['sampled_zones'] for s in a[2] if s['name']=='VERY_LARGE'),2)
        for z,w in a[3].items():
            self.assertGreater(w['zone_inclusion_probability'],0)
            expected={q['query_id'] for q in panel if q['crm_insee']==z}
            self.assertEqual(expected,{q['query_id'] for q in a[1] if q['crm_insee']==z})

    def test_sampling_ignores_names_and_address_difficulty(self):
        census,panel=self.fixture(); changed=copy.deepcopy(panel)
        for q in changed: q['crm_name']='IMPOSSIBLE'; q['crm_number']='99'; q['crm_street']='OTHER'
        self.assertEqual([z['zone'] for z in pilot.sample(census,panel)[0]],
            [z['zone'] for z in pilot.sample(census,changed)[0]])

    def test_weighted_rate_uses_population_shares_not_raw_counts(self):
        sampling=dict(source_queries=100,zone_weights={'a':{'stratum':'S'},'b':{'stratum':'L'},'c':{'stratum':'L'}},
            strata=[dict(name='S',population_queries=20,population_zones=1),
                dict(name='L',population_queries=80,population_zones=10)])
        rows=[dict(zone='a',exact=True)]*20+[dict(zone='b',exact=False),dict(zone='c',exact=False)]
        result=pilot.weighted_report(rows,sampling,'exact',draws=10)
        self.assertAlmostEqual(result['poststratified_hajek_estimate'],.2)
        self.assertTrue(result['not_a_99_2pct_certification'])
        self.assertTrue(result['bootstrap_can_be_degenerate_when_no_observed_misses'])

    def test_incomplete_outputs_gate_precedes_labels(self):
        with patch.object(pilot,'generation',side_effect=ValueError('Missing dense exit')):
            with patch.object(pilot.io,'read',side_effect=AssertionError('Labels opened')):
                with self.assertRaisesRegex(ValueError,'Missing dense exit'): pilot.evaluate(None)


if __name__=='__main__': unittest.main()
