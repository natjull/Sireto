import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts import embeddinggemma2_v7_batched_comparison as comparison


class BucketedInference(unittest.TestCase):
    def test_restores_order_and_preserves_all_texts(self):
        texts = ['aaaaa', 'b', 'cc', 'dddd']; calls = []
        class Model:
            def eval(self): pass
        def tokenize(values, **kwargs):
            return {'input_ids': [[0] * len(v) for v in values]}
        def encode(values):
            calls.append(values)
            return torch.tensor([[len(v), ord(v[0])] for v in values], dtype=torch.float32)
        with patch.object(torch.mps, 'empty_cache'):
            got = comparison.bucketed_inference(encode, Model(), tokenize, texts, batch_size=2)
        np.testing.assert_array_equal(got, [[5, 97], [1, 98], [2, 99], [4, 100]])
        self.assertEqual(calls, [['b', 'cc'], ['dddd', 'aaaaa']])
        self.assertEqual(sum(map(len, calls)), len(texts))

    def test_empty_input_is_not_silently_removed(self):
        with self.assertRaises(ValueError):
            comparison.bucketed_inference(None, None, None, [])


if __name__ == '__main__': unittest.main()
