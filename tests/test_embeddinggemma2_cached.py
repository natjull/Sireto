import unittest
import torch
from scripts.embeddinggemma2_cached import cached_backward,contrastive


class GradientParity(unittest.TestCase):
    def test_cache_keeps_global_loss_and_gradients(self):
        torch.manual_seed(7)
        model=torch.nn.Linear(3,5,bias=False).double()
        x=torch.randn(7,3,dtype=torch.float64)
        def encode(ids): return torch.nn.functional.normalize(model(x[ids]),dim=-1)
        q=[0,1,2]; d=[3,4,5,6]; sirens=['a','a','c']; owners=['a','a','c','z']
        direct=contrastive(encode(q),encode(d),sirens,owners); direct.backward()
        expected=model.weight.grad.clone(); model.zero_grad()
        cached=cached_backward(encode,q,d,sirens,owners,micro=2)
        self.assertAlmostEqual(float(direct.detach()),cached,places=12)
        torch.testing.assert_close(model.weight.grad,expected,rtol=1e-11,atol=1e-11)


if __name__=='__main__': unittest.main()
