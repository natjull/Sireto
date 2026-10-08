import unittest
from scripts import embeddinggemma2_v9_postseal_audit as audit


class PostsealAudit(unittest.TestCase):
    def test_complementary_gains_and_losses_are_not_hidden(self):
        self.assertEqual(audit.changes({'a': True, 'b': False}, {'a': False, 'b': True}),
            {'gained': ['b'], 'lost': ['a']})
        with self.assertRaises(ValueError): audit.changes({'a': True}, {})

    def test_rank_does_not_inject_absent_target(self):
        candidates = [{'siret': 'a'}, {'siret': 'b'}]
        self.assertEqual(audit.rank(candidates, 'b'), 2)
        self.assertIsNone(audit.rank(candidates, 'c'))
        self.assertEqual(len(candidates), 2)


if __name__ == '__main__': unittest.main()
