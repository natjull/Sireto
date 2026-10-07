import unittest
import torch
from scripts.embeddinggemma2_v2_fit import cached_objective,objective,training_texts,plan_rows


class DualProof(unittest.TestCase):
    def test_weighted_cached_gradient_matches_direct(self):
        torch.manual_seed(9); model=torch.nn.Linear(4,7,bias=False).double(); x=torch.randn(9,4,dtype=torch.float64)
        def encode(ids): return torch.nn.functional.normalize(model(x[ids]),dim=-1)
        q=[0,1,2]; d=[3,4,5,6,7,8]; mask=[[False,True,False,False,False,False],[True,False,False,False,False,False],[False,False,False,False,False,True]]
        weights=[1.,.5,.25]
        value=objective(encode(q),encode(d),mask,weights); value.backward(); wanted=model.weight.grad.clone(); model.zero_grad()
        measured=cached_objective(encode,q,d,mask,weights,micro=2)
        self.assertAlmostEqual(float(value.detach()),measured,places=12)
        torch.testing.assert_close(model.weight.grad,wanted,rtol=1e-11,atol=1e-11)

    def test_site_negative_allowed_only_with_proof_identity_always_masks(self):
        crm=dict(crm_number='4',crm_suffix='',crm_street='RUE DE PARIS',crm_insee='75056')
        base=dict(siren='a',insee='75056',related_parents=[])
        docs={'p':dict(base,key='p',site_views=['site p'],all_addresses=['4 RUE DE PARIS']),
              'n':dict(base,key='n',site_views=['site n'],all_addresses=['6 RUE DE PARIS'])}
        row=dict(positive='p',siren='a',site_query='site',identity_query='name',crm=crm,negatives=[dict(key='n')])
        q,d,mask,w=training_texts([(row,0,1)],docs,{'a':['name']},'site',0)
        self.assertEqual(mask,[[False,False]])
        # Sibling is NOT an identity hard negative at all.
        q,d,mask,w=training_texts([(row,0,1)],docs,{'a':['name']},'identity',0)
        self.assertEqual(len(d),1); self.assertEqual(mask,[[False]])
        docs['n']['all_addresses']=['4 RUE DE PARIS']
        self.assertEqual(training_texts([(row,0,1)],docs,{'a':['name']},'site',0)[2],[[False,True]])

    def test_plan_covers_all_views_without_entity_oversampling_weight(self):
        rows=[dict(positive='p',siren='a'),dict(positive='q',siren='b')]
        docs={'p':dict(site_views=['x','y']),'q':dict(site_views=['z'])}; parents={'a':['1','2','3'],'b':['4']}
        plan=plan_rows(rows,docs,parents); self.assertEqual(len(plan),4)
        for i in (0,1): self.assertAlmostEqual(sum(1/n for row,v,n in plan if row==i),1.)
