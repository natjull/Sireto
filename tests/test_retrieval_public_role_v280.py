import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
from src.xgb_matcher import retrieval_public_role_v280 as role
from scripts import retrieval_v280_public_role_training as producer


def fixture():
 rows=[dict(query_id=i,fold=2+i%3,positive_siren=f'{100000000+i:09d}',positive_siret=f'{100000000+i:09d}00001') for i in range(512)]
 registry={str(f):[r['positive_siren'] for r in rows if r['fold']==f] for f in (2,3,4)}
 x=np.random.default_rng(280).normal(size=(512,384))
 parents=dict(commune={r['positive_siren'] for r in rows if r['query_id']%5==0},epci={r['positive_siren'] for r in rows if r['query_id']%7==0})
 safe=dict(query_ids=[r['query_id'] for r in rows if r['fold']!=2],heldout_fold=2,training_folds=[3,4])
 return x,rows,registry,parents,safe


class RoleTests(unittest.TestCase):
 def test_whole_fixed_fit(self):
  x,r,g,p,s=fixture();a,m=role.fit(x,r,g,s,p,2)
  self.assertEqual(a.shape,(385,2));self.assertEqual(m['query_ids'],s['query_ids'])
  self.assertEqual(m['ridge'],1);self.assertFalse(m['identifiers_as_features'])
  self.assertFalse(m['probabilities_calibrated']);self.assertTrue(m['official_list_missing_is_not_a_business_label_error'])
  v=role.scores(x,a);self.assertTrue(np.isfinite(v).all());self.assertTrue(((0<=v)&(v<=1)).all())
 def test_family_protected(self):
  x,r,g,p,s=fixture();g['2'].append(r[1]['positive_siren'])
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_held_fold_excluded(self):
  x,r,g,p,s=fixture();s['query_ids'].append(0)
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_duplicate_safe(self):
  x,r,g,p,s=fixture();s['query_ids'].append(s['query_ids'][0])
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_incomplete_denom(self):
  x,r,g,p,s=fixture()
  with self.assertRaises(ValueError):role.fit(x[:-1],r[:-1],g,s,p,2)
 def test_closed_fold(self):
  x,r,g,p,s=fixture();r[0]['fold']=1
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_wrong_dimension(self):
  x,r,g,p,s=fixture()
  with self.assertRaises(ValueError):role.fit(np.ones((512,768)),r,g,s,p,2)
 def test_parent_id_not_numeric_feature(self):
  x,r,g,p,s=fixture();p['commune'].add(100000000)
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_source_missing_preserves_rows(self):
  x,r,g,p,s=fixture();p={k:set() for k in p};a,m=role.fit(x,r,g,s,p,2)
  self.assertEqual(m['training_queries'],len(s['query_ids']));self.assertEqual(m['positive_training_memberships'],[0,0])
  self.assertTrue(np.array_equal(a,np.zeros((385,2))))
 def test_original_safe_authority(self):
  x,r,g,p,s=fixture();s['heldout_fold']=3
  with self.assertRaises(ValueError):role.fit(x,r,g,s,p,2)
 def test_score_rejects_nan(self):
  with self.assertRaises(ValueError):role.scores(np.ones((1,384)),np.full((385,2),np.nan))
 def test_own_stop_refused_before_parse(self):
  with tempfile.TemporaryDirectory() as d,patch.object(producer,'OUTPUT',Path(d)):
   root=Path(d)/'stop';root.mkdir();(root/'status.json').touch()
   with self.assertRaises(ValueError):producer.load(root)
 def test_fresh_only(self):
  with tempfile.TemporaryDirectory() as d,patch.object(producer,'OUTPUT',Path(d)):
   with self.assertRaises(ValueError):producer.run(Path(d))
 def test_serving_no_training_open(self):
  import inspect
  text=inspect.getsource(producer.admission)+inspect.getsource(producer.sealed)
  self.assertNotIn('training(files)',text);self.assertNotIn("['training_inputs']",text)
  self.assertIn('check_epochs',text)
 def test_fixed_constants(self):
  self.assertEqual(producer.INPUTS['municipal']['sha256'],'2c0f9b6c924ea2833f1a372a329d05e082f780fd55f88246ae9f5be0ed06a8f1')
  self.assertEqual(producer.LIMITS['max_rss_bytes'],2147483648)
  self.assertEqual(role.ROLES,('commune','epci'))


if __name__=='__main__':unittest.main()
