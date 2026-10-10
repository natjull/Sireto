import ast
import unittest
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from scripts import embeddinggemma2_v18_identity_retrieval as retrieval
from scripts import embeddinggemma2_v2_data as data


class IdentityRetrievalTests(unittest.TestCase):
    def test_joint_is_original_frozen_hypothesis(self):
        self.assertAlmostEqual(retrieval.joint_scores(.8,.4),.7)
        self.assertIn('EVERY_COMPLETE_COMMUNE_SITE',retrieval.RECIPE['joint'])

    def test_sibling_names_are_explicit_parent_context(self):
        self.assertEqual(data.identity_atoms(dict(legal=['PARENT'],sites=['FOYER'])),
            ['nom entreprise: PARENT','nom de site appartenant a cette entreprise: FOYER'])

    def test_complete_output_required(self):
        values={v:{'a':[]} for v in retrieval.RECIPE['variants']}
        retrieval.validate_outputs(values,{'a'})
        with self.assertRaises(ValueError): retrieval.validate_outputs(values,{'a','b'})

    def test_cap_and_unique(self):
        values={v:{'a':[]} for v in retrieval.RECIPE['variants']}
        values['joint']['a']=[dict(siret=str(i)) for i in range(101)]
        with self.assertRaises(ValueError): retrieval.validate_outputs(values,{'a'})
        values['joint']['a']=[dict(siret='1'),dict(siret='1')]
        with self.assertRaises(ValueError): retrieval.validate_outputs(values,{'a'})

    def test_no_tuned_grid_or_frozen_prepool(self):
        self.assertTrue(retrieval.RECIPE['no_grid'])
        self.assertTrue(retrieval.RECIPE['no_fixed_prepool'])
        self.assertEqual(retrieval.RECIPE['inference_batch'],8)
        self.assertEqual(retrieval.RECIPE['max_tokens'],512)
        self.assertEqual(retrieval.RECIPE['adapter_step'],1000)

    def test_ignored_duplicate_text_has_no_rowid_hole(self):
        con=sqlite3.connect(':memory:')
        con.execute('CREATE TABLE texts(id INTEGER PRIMARY KEY,text TEXT UNIQUE NOT NULL)')
        for t in ['one','two','one','three','two','four']:
            con.execute('INSERT OR IGNORE INTO texts(text) VALUES(?)',(t,))
        self.assertEqual(con.execute('SELECT id FROM texts ORDER BY id').fetchall(),[(1,),(2,),(3,),(4,)])
        con.close()

    def test_generation_does_not_read_label_artifacts(self):
        tree=ast.parse(Path(retrieval.__file__).read_text())
        for function in tree.body:
            if isinstance(function,ast.FunctionDef) and function.name in ('initialize','prepare','generate'):
                literals=[n.value for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
                self.assertFalse(any(s in value for value in literals for s in
                    ('outcomes.json','validation_membership.json','cases.json','acceptable_operational.parquet')))

    def test_vector_cache_copies_before_eviction(self):
        import numpy as np
        from scripts import embeddinggemma2_v14_full_pipeline as engine
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'vectors_00000000.npy'
            np.save(path,np.ones((2,768),dtype=np.float32)); engine.seal(path)
            cache=retrieval.VectorCache(Path(tmp)); a=cache.get(0); cache.close()
            self.assertEqual(float(a.sum()),768.)


if __name__=='__main__': unittest.main()
