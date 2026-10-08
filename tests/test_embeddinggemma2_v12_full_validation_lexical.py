import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import embeddinggemma2_v12_full_validation_lexical as experiment


class CompleteBeforeLabels(unittest.TestCase):
    def test_incomplete_or_failed_child_cannot_open_labels(self):
        for exit_code, queries in ((-9,2212),(0,80)):
            with patch.object(experiment.io,'read',side_effect=[
                    {'exit_code':exit_code,'own_child_exit_confirmed':True}, {'queries':queries}]) as read:
                with self.assertRaises(ValueError): experiment.evaluate(Path('/unused'))
                self.assertEqual(read.call_count,2)


if __name__ == '__main__': unittest.main()
