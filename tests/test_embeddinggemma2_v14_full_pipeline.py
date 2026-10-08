import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import embeddinggemma2_v14_full_pipeline as run
from src.xgb_matcher.retrieval_v20_workbench import record


class FullPipeline(unittest.TestCase):
    def test_all_official_owner_names_no_sibling_alias_transfer_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); run.io.write(out/'launch.json',{})
            facts=out/'facts.parquet'; names=out/'names.parquet'
            pq.write_table(pa.Table.from_pylist([
                dict(name_id=1,parent_key=111,owner_key=111,subject_code=1,scope_code=1),
                dict(name_id=2,parent_key=111,owner_key=11100001,subject_code=2,scope_code=3),
                dict(name_id=3,parent_key=111,owner_key=11100002,subject_code=2,scope_code=3)]),facts)
            pq.write_table(pa.Table.from_pylist([dict(name_id=1,name_text='ＲＡＩＳＯＮ 123456789'),
                dict(name_id=2,name_text='ENSEIGNE A'),dict(name_id=3,name_text='ENSEIGNE B')]),names)
            manifest=out/'manifest.json'
            run.io.write(manifest,dict(status='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY',
                buckets=[dict(bucket=i,facts=3,artifacts={'facts.parquet':record(facts),'names.parquet':record(names)}) for i in range(64)]))
            with patch.object(run,'authority',return_value={'dictionary':record(manifest)}),patch.object(run,'guard'):
                run.compile_aliases(out)
                database=out/'names/owners.sqlite'; original=record(database)
                con=sqlite3.connect(database)
                self.assertEqual(run.lookup(con,1,[111]),{111:{'RAISON'}})
                own=run.lookup(con,2,[11100001,11100002])
                self.assertEqual(own[11100001],{'ENSEIGNE A'})
                self.assertNotIn('ENSEIGNE B',own[11100001])
                self.assertEqual(con.execute('SELECT count(*) FROM buckets').fetchone()[0],64)
                con.close(); run.compile_aliases(out)
                self.assertEqual(record(database),original)
                self.assertEqual(run.durable_progress(out,out,'aliases',None),64)

    def test_max_site_view_score_not_parent_or_average(self):
        vectors={'a':np.array([1.,0.],dtype=np.float32),'b':np.array([0.,1.],dtype=np.float32),
            'c':np.array([.6,.8],dtype=np.float32)}
        queries=np.eye(2,dtype=np.float32)
        actual=run.score_views(queries,[['a','b'],['c']],vectors.__getitem__)
        np.testing.assert_array_equal(actual,np.array([[1.,.6],[1.,.8]],dtype=np.float32))
        native=run.score_views(queries,[['a'],['c']],vectors.__getitem__,True)
        np.testing.assert_array_equal(native,queries@np.stack([vectors['a'],vectors['c']]).T)

    def test_text_order_and_boundary_binding(self):
        self.assertNotEqual(run.text_hash(['a','bc']),run.text_hash(['ab','c']))
        self.assertNotEqual(run.text_hash(['a','b']),run.text_hash(['b','a']))
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'binding.json'; run.bind(target,{'x':1}); run.bind(target,{'x':1})
            with self.assertRaises(ValueError): run.bind(target,{'x':2})

    def test_incomplete_generation_cannot_open_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(run,'complete_channels',side_effect=ValueError('Missing owned exit')):
                with patch.object(run.io,'read',side_effect=AssertionError('Labels opened')):
                    with self.assertRaisesRegex(ValueError,'Missing owned exit'): run.evaluate(out)

    def test_durable_progress_ignores_unsealed_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); cache=out/'v3_official512'; cache.mkdir()
            (cache/'vectors_00000000.npy').write_bytes(b'not durable until sealed')
            self.assertEqual(run.durable_progress(out,out,'dense','v3_official512'),0)
            run.seal(cache/'vectors_00000000.npy')
            self.assertEqual(run.durable_progress(out,out,'dense','v3_official512'),1)


if __name__=='__main__': unittest.main()
