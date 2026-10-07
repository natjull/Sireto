import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import embeddinggemma2_v3_fit as fit
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record
from tests.test_embeddinggemma2_v2_fit import DualProof


class NightMath(DualProof):
    """Same tested loss, adapted namespace; no GPU needed for numerical parity."""
    def test_v3_cached_gradient(self):
        import torch
        torch.manual_seed(12)
        model=torch.nn.Linear(4,7,bias=False).double()
        x=torch.randn(9,4,dtype=torch.float64)
        def encode(ids): return torch.nn.functional.normalize(model(x[ids]),dim=-1)
        q=[0,1,2]; d=[3,4,5,6,7,8]; mask=[[False]*6 for _ in q]; weights=[1,.5,.25]
        value=fit.objective(encode(q),encode(d),mask,weights); value.backward()
        expected=model.weight.grad.clone(); model.zero_grad()
        measured=fit.cached_objective(encode,q,d,mask,weights,micro=2)
        self.assertAlmostEqual(float(value.detach()),measured,places=12)
        torch.testing.assert_close(model.weight.grad,expected,rtol=1e-11,atol=1e-11)


class RestartableMining(unittest.TestCase):
    def test_resource_yield_retains_hash_bound_shard_and_restarts_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; source.mkdir(); out=root/'mining'
            io.write(source/'manifest.json',{'toy':True}); initial=root/'initial.pt'; initial.write_bytes(b'fixture')
            docs={str(i):dict(key=str(i),siren=str(i),insee='zone',site_views=[f'name {i}'],related_parents=[]) for i in range(129)}
            rows=[dict(query_id='q',positive='0',siren='0',crm=dict(crm_insee='zone'),site_query='query',negatives=[])]
            for name,value in [('train.json',rows),('validation.json',[]),('train_bank.json',list(docs))]: io.write(source/name,value)
            artifacts={name:record(source/name) for name in ('train.json','validation.json','train_bank.json')}
            cfg=dict(max_rss_bytes=8*1024**3); calls=[]
            def inference(texts):
                calls.append(len(texts))
                if len(calls)==2: raise fit.ResourceYield('toy resource pause')
                a=np.zeros((len(texts),768),dtype=np.float32); a[:,0]=1; return a
            with patch.object(fit,'inputs',return_value=({},cfg,docs,{},artifacts)),patch.object(fit,'INITIAL',initial),patch.object(fit,'resource_guard'),patch.object(fit,'load_model',return_value=(None,None,None,inference,None)):
                with self.assertRaises(fit.ResourceYield): fit.mine(source,out,time.time()+3600)
                self.assertTrue((out/'vectors_0000000.json').exists())
                fit.mine(source,out,time.time()+3600)
                self.assertEqual(calls,[128,1,1,1])  # remaining document then CRM query
                self.assertEqual(io.read(out/'manifest.json')['rows'],1)
                (out/'vectors_0000000.npy').write_bytes(b'corrupted')
                with self.assertRaises(Exception): fit.mine(source,out,time.time()+3600)

    def test_deadline_and_resource_failure_are_distinct(self):
        with patch.object(io,'pressure',return_value=(79,1)):
            with self.assertRaises(TimeoutError): fit.resource_guard(time.time()-1)
        with patch.object(io,'pressure',return_value=(10,1)):
            with self.assertRaises(fit.ResourceYield): fit.resource_guard(time.time()+3600)
