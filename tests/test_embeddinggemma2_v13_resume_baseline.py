import unittest
from scripts.embeddinggemma2_v13_resume_baseline import retriable, RESOURCE_ERROR
from scripts import embeddinggemma2_night as io


class ResourceResume(unittest.TestCase):
    def test_only_known_resource_error_retry(self):
        self.assertTrue(retriable(1,RESOURCE_ERROR))
        self.assertFalse(retriable(0,RESOURCE_ERROR))
        self.assertFalse(retriable(1,'Index drift'))
        self.assertFalse(retriable(-9,''))

    def test_test_authorization_is_conditional_and_not_fold1(self):
        a=io.read(io.ROOT/'config/embeddinggemma2_reserved_test_authorization_20261008.json')
        self.assertEqual(a['expected_queries'],4323)
        self.assertEqual(a['maximum_openings'],1)
        self.assertEqual(a['status'],'AUTHORIZED_CONDITIONAL_NOT_CONSUMED')
        self.assertFalse(a['test_opened']); self.assertFalse(a['fold1_open_authorized'])


if __name__=='__main__': unittest.main()
