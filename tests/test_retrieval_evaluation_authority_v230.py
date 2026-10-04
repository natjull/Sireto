"""Real tiny producer files, simulated process authorities, NOT national proof."""
import json

import pytest

from src.xgb_matcher import retrieval_evaluation_authority_v230 as work
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_corroborated_population_v229 import fixture


def setup(tmp_path, monkeypatch):
    queries, epochs = fixture(tmp_path, monkeypatch)
    value = work.generation.admission(tmp_path)
    write_json(tmp_path/'outputs.json', dict(rows=value['rows'], source_epochs=epochs, **work.generation.FALSE))
    write_json(tmp_path/'manifest.json', dict(status=work.generation.STATUS, queries=128, population=128,
        all128_outputs_sealed=True, initial100_emitted=True, launch=record(tmp_path/'launch.json'),
        full_appended_leaf_universe=False, outputs=record(tmp_path/'outputs.json'), pairs=value['pairs'],
        peak_worker_rss_bytes=1000, **work.generation.FALSE))
    monkeypatch.setattr(work, 'source_scope', lambda: ({'SIMULATED_AVAILABLE': True}, {'SIMULATED_NATIVE': True}))
    return tmp_path


def test_authority_binds_every_output_and_epoch_before_any_target(tmp_path, monkeypatch):
    root = setup(tmp_path, monkeypatch)
    permission = work.authorize(root)
    assert len(permission['controls']) == 128 and len(permission['producers']) == 16
    assert permission['all128_before_targets'] and not permission['truth_opened']
    assert permission['evaluation_process_supervision_still_required']
    work.recheck(permission)
    (root/'source.txt').write_text('CHANGED_FIXTURE')
    with pytest.raises(ValueError): work.recheck(permission)


@pytest.mark.parametrize('change', ['partial', 'fold0', 'fold1', 'producer', 'rss', 'ready', 'truth'])
def test_never_authorizes_partial_or_unsafe_sources(tmp_path, monkeypatch, change):
    root = setup(tmp_path, monkeypatch)
    path = root/'manifest.json'
    manifest = json.loads(path.read_text())
    if change == 'partial': manifest['queries'] = 127
    elif change == 'rss': manifest['peak_worker_rss_bytes'] = 2**31+1
    elif change == 'ready': manifest['ready_for_fold1'] = True
    elif change == 'truth': manifest['truth_opened'] = True
    elif change == 'producer': (root/'batch015'/'manifest.json').unlink()
    else:
        launch = json.loads((root/'launch.json').read_text())
        launch['queries'][-1]['fold'] = 0 if change == 'fold0' else 1
        # The real validator rejects the contaminated128 input even though
        # this fixture's load callback is explicitly simulated.
        monkeypatch.setattr(work.generation, 'load', lambda root: launch)
    write_json(path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)): work.authorize(root)


def test_real_stop217_refused_without_opening_target_or_official_namespace(monkeypatch):
    from pathlib import Path
    def forbidden(): raise AssertionError('Source scope called before full128 seal')
    monkeypatch.setattr(work, 'source_scope', forbidden)
    # Empty/missing root is sufficient for the source-independent test.
    with pytest.raises((ValueError, FileNotFoundError)):
        work.authorize(Path('/private/tmp/NONEXISTENT_V230_TEST_ROOT'))

