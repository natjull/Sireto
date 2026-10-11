import ast
from datetime import date
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from scripts import embeddinggemma2_v21_relation_views as retrieval


class RelationViewsTests(unittest.TestCase):
    def test_duckdb_parameterized_read_stream_to_parquet(self):
        import duckdb
        import pyarrow as pa
        import pyarrow.parquet as pq
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.parquet';dest=Path(tmp)/'selected.parquet'
            pq.write_table(pa.table({'siret':['10000000000001','10000000000002']}),source)
            con=duckdb.connect()
            try:
                retrieval.write_selection(con,'SELECT * FROM read_parquet(?) WHERE siret = ?',
                    [str(source),'10000000000002'],dest)
            finally:con.close()
            self.assertEqual(pq.read_table(dest).to_pylist(),[{'siret':'10000000000002'}])

    def test_original_ban_and_fallback_footprints(self):
        strong={'ban_declared_weak':False,'ban_uncertainty_m':25}
        self.assertEqual(retrieval.footprint_radius(strong),300)
        self.assertEqual(retrieval.footprint_radius({**strong,'ban_declared_weak':True}),1000)
        self.assertEqual(retrieval.footprint_radius({**strong,'ban_uncertainty_m':200}),600)
        self.assertEqual(retrieval.footprint_radius({**strong,'footprint_radius_m':450}),450)

    def test_all_views_are_visited_not_only_first100(self):
        keys=np.arange(10000000000000,10000000000101,dtype=np.uint64)
        scores=np.linspace(1.,.5,101);calls=set()
        class ObservedEdges(dict):
            def get(self,key,default):
                calls.add(key);return super().get(key,default)
        retrieval.collect(keys,scores,ObservedEdges())
        self.assertEqual(calls,set(map(int,keys)))

    def test_context_has_provenance_not_own_role(self):
        rows=retrieval.collect([10000000000001],[.9],{10000000000001:[(99999999999999,{'date':'2025-01-01'})]})
        row=next(r for r in rows if r['siret']=='99999999999999')
        self.assertEqual(row['view_role'],'DIRECT_SUCCESSION_CONTEXT')
        self.assertEqual(row['context_endpoint'],'10000000000001')
        self.assertEqual(row['official_link']['date'],'2025-01-01')

    def test_deduplicate_and_own_role_wins_equal_score(self):
        rows=retrieval.collect([10000000000001,10000000000002],[.9,.9],{10000000000001:[(10000000000002,{'x':1})]})
        self.assertEqual(len(rows),2);self.assertTrue(all(r['view_role']=='OWN_VIEW' for r in rows))

    def test_one_hop_not_recursive(self):
        rows=retrieval.collect([10000000000001],[.8],{
            10000000000001:[(10000000000002,{})],10000000000002:[(10000000000003,{})]})
        self.assertNotIn('10000000000003',{r['siret'] for r in rows})

    def test_identifier_never_changes_score(self):
        a=retrieval.collect([10000000000001],[.7],{10000000000001:[(99999999999999,{})]})
        self.assertTrue(all(abs(r['score']-.7)<1e-12 for r in a))

    def test_typed_dated_continuity_only(self):
        self.assertTrue(retrieval.valid_link(True,date(2025,1,1)))
        for flag,when in [(None,date(2025,1,1)),(False,date(2025,1,1)),('true',date(2025,1,1)),(True,None),(True,date(2027,1,1))]:
            self.assertFalse(retrieval.valid_link(flag,when))

    def test_unique_absolute_cap_and_finite_scores(self):
        rows=retrieval.collect(list(range(200)),np.ones(200),{})
        self.assertEqual(len(rows),100)
        with self.assertRaises(ValueError):retrieval.collect([1,1],[.5,.6],{})
        with self.assertRaises(ValueError):retrieval.collect([1],[np.nan],{})

    def test_context_not_own_alias_or_same_site(self):
        self.assertTrue(retrieval.RECIPE['context_is_not_own_alias'])
        self.assertTrue(retrieval.RECIPE['moving_successions_not_same_site'])
        self.assertTrue(retrieval.RECIPE['no_new_fusion_of_capped_channels'])

    def test_generation_has_no_labels_or_audit_sources(self):
        tree=ast.parse(Path(retrieval.__file__).read_text())
        for f in tree.body:
            if isinstance(f,ast.FunctionDef) and f.name not in ('evaluate',):
                literals=[n.value for n in ast.walk(f) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                self.assertFalse(any(s in v for v in literals for s in ('outcomes.json','validation_membership.json','cases.json','label_ledger.parquet','acceptable_operational.parquet')))


if __name__=='__main__':unittest.main()
