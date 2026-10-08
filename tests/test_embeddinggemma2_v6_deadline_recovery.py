import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v6_deadline_recovery as recovery
from src.xgb_matcher.retrieval_v20_workbench import record


class DeadlineRecovery(unittest.TestCase):
    def test_preserves_cursor_optimizer_weights_deadlines_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'old'; old = source / 'fit'; old.mkdir(parents=True)
            out = root / 'new'
            io.write(source / 'config.json', {'learning_rate': .00003})
            io.write(source / 'data.json', {'config': record(source / 'config.json')})
            launch = {'data': record(source / 'data.json'), 'deadline_unix': 1234}
            io.write(old / 'launch.json', launch)
            io.write(old / 'supervisor.json', {'status': 'BUDGET_EXHAUSTED'})
            io.write(source / 'night.json', {'deadline_unix': 5678})
            for name in ('preflight.json', 'preflight_exit.json'):
                io.write(old / name, {'fixture': True})
            state = dict(step=1340, epoch=0, next_offset=21440,
                adapter={'a': torch.ones(2)}, optimizer={'x': torch.tensor(4)},
                launch_sha256=io.digest(old / 'launch.json'))
            for name in ('checkpoint.pt', 'best_adapter.pt'): torch.save(state, old / name)
            hashes = {p.name: io.digest(p) for p in old.iterdir()}
            with patch.object(recovery, 'require_deadline_exit', return_value=launch):
                cursor = recovery.migrate_checkpoint(source, out)
            self.assertEqual(cursor, (1340, 0, 21440))
            self.assertEqual(io.read(out / 'fit/launch.json')['deadline_unix'], 1234)
            self.assertEqual(io.read(out / 'night.json')['deadline_unix'], 5678)
            new = torch.load(out / 'fit/checkpoint.pt', weights_only=True)
            torch.testing.assert_close(new['adapter']['a'], state['adapter']['a'])
            torch.testing.assert_close(new['optimizer']['x'], state['optimizer']['x'])
            self.assertEqual(new['launch_sha256'], io.digest(out / 'fit/launch.json'))
            self.assertEqual(hashes, {p.name: io.digest(p) for p in old.iterdir()})

    def test_refuses_recovery_of_running_or_non_deadline_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp); (source / 'fit').mkdir()
            io.write(source / 'fit/launch.json', {'deadline_unix': time.time() - 1})
            for status, history in [('CHILD_RUNNING', []), ('BUDGET_EXHAUSTED',
                    [{'exit_code': -9, 'reason': {'deadline_reached': False}}])]:
                io.write(source / 'fit/supervisor.json', {'status': status, 'history': history})
                with self.assertRaises(ValueError): recovery.require_deadline_exit(source)


if __name__ == '__main__': unittest.main()
