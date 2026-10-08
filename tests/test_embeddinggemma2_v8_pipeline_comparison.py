import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v8_pipeline_comparison as comparison
from src.xgb_matcher.retrieval_v20_workbench import record


class PipelineSeals(unittest.TestCase):
    def fixture(self, root):
        io.write(root / 'panel.json', [{'query_id': 'q'}])
        rows = [{'query_id': 'q', 'candidates': [{'siret': '123', 'siren': '1'}]}]
        io.write(root / 'lexical.json', rows)
        io.write(root / 'launch.json', {'panel': record(root / 'panel.json'), 'lexical': record(root / 'lexical.json')})
        for variant in comparison.VARIANTS:
            io.write(root / (variant + '_outputs.json'), rows)
            io.write(root / (variant + '_generation.json'), {'status': 'SEALED_COMPLETE_PAIRED_INITIAL100', 'outputs': record(root / (variant + '_outputs.json'))})
            io.write(root / (variant + '_exit.json'), {'exit_code': 0, 'own_child_exit_confirmed': True})

    def test_two_complete_pipelines_explicitly_omit_intermediate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            with patch.object(comparison.paired, 'verify_launch'):
                scope, candidates, launch = comparison.seal(root)
            self.assertEqual(scope, {'q'})
            self.assertIn('v3_official512_rrf', candidates)
            self.assertNotIn('v1_official512', candidates)
            self.assertFalse(io.read(root / 'all_outputs_sealed.json')['truth_opened'])

    def test_failed_exit_prevents_seal_before_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root)
            io.write(root / 'v3_official512_exit.json', {'exit_code': -9, 'own_child_exit_confirmed': True})
            with patch.object(comparison.paired, 'verify_launch'):
                with self.assertRaises(ValueError): comparison.seal(root)
            self.assertFalse((root / 'all_outputs_sealed.json').exists())


if __name__ == '__main__': unittest.main()
