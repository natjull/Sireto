import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import embeddinggemma2_v4_fit as fit
from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v4_compare import fused100
from src.xgb_matcher.retrieval_v20_workbench import record


class ExecutionContinuation(unittest.TestCase):
    def test_warning_not_critical_but_low_free_memory_still_yields(self):
        with patch.object(io,'pressure',return_value=(38,2)): fit.resource_guard(time.time()+60)
        for level in (3,4,99):
            with patch.object(io,'pressure',return_value=(38,level)):
                with self.assertRaises(fit.ResourceYield): fit.resource_guard(time.time()+60)
        with patch.object(io,'pressure',return_value=(10,1)):
            with self.assertRaises(fit.ResourceYield): fit.resource_guard(time.time()+60)

    def test_exact_checkpoint_migration_and_deadline_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'original'; source.mkdir(); previous=source/'fit'; previous.mkdir(); out=Path(tmp)/'new'
            cfg=dict(admit_free_percent=45,yield_free_percent=25,supervisor_free_percent=20)
            io.write(source/'config.json',cfg)
            io.write(source/'data.json',dict(config=record(source/'config.json'),artifacts={'test':'unchanged'}))
            io.write(source/'train.json',[{'fixture':True}])
            io.write(source/'mining.json',dict(data=record(source/'data.json'),train=record(source/'train.json')))
            io.write(previous/'launch.json',dict(data=record(source/'data.json'),mining=record(source/'mining.json'),deadline_unix=1234))
            state=dict(adapter={'a':torch.ones(2)},optimizer={'step':torch.tensor(3)},step=3,epoch=0,next_offset=48,
                launch_sha256=io.digest(previous/'launch.json'))
            for name in ('checkpoint.pt','best_adapter.pt'): torch.save(state,previous/name)
            io.write(previous/'preflight.json',{'fixture':True}); io.write(source/'night.json',dict(deadline_unix=5678))
            def supervise(f):
                io.write(f/'supervisor.json',dict(status='COMPLETE')); io.write(f/'state.json',dict(status='FIT_COMPLETE'))
            with patch.object(fit,'preflight'),patch.object(fit,'supervise',side_effect=supervise): fit.continuation(source,out)
            resumed=torch.load(out/'fit/checkpoint.pt',weights_only=True)
            self.assertEqual(resumed['step'],3); self.assertEqual(resumed['next_offset'],48)
            torch.testing.assert_close(resumed['adapter']['a'],state['adapter']['a'])
            torch.testing.assert_close(resumed['optimizer']['step'],state['optimizer']['step'])
            self.assertEqual(resumed['launch_sha256'],io.digest(out/'fit/launch.json'))
            self.assertEqual(io.read(out/'fit/launch.json')['deadline_unix'],1234)
            self.assertEqual(io.read(out/'night.json')['deadline_unix'],5678)
            self.assertEqual(torch.load(previous/'checkpoint.pt',weights_only=True)['launch_sha256'],state['launch_sha256'])

    def test_v4_loss_and_cached_gradients_still_match(self):
        torch.manual_seed(4); model=torch.nn.Linear(3,5,bias=False).double(); x=torch.randn(6,3,dtype=torch.float64)
        def encode(ids): return torch.nn.functional.normalize(model(x[ids]),dim=-1)
        q=[0,1]; d=[2,3,4,5]; mask=[[False]*4 for _ in q]; weights=[1.,.5]
        loss=fit.objective(encode(q),encode(d),mask,weights); loss.backward(); gradient=model.weight.grad.clone(); model.zero_grad()
        measured=fit.cached_objective(encode,q,d,mask,weights,micro=2)
        self.assertAlmostEqual(measured,float(loss.detach()),places=12)
        torch.testing.assert_close(model.weight.grad,gradient,rtol=1e-11,atol=1e-11)

    def test_v4_union_retains_initial100_cap(self):
        rows=[dict(siret=str(i),siren='p') for i in range(100)]
        self.assertEqual(len(fused100([rows,rows])),100)
