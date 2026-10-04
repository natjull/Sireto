"""Real tiny producer bytes/epochs; source ledger/process evidence explicitly simulated."""
from copy import deepcopy
import json

import pytest

from src.xgb_matcher import retrieval_evaluation_authority_v203 as authority
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_run_retrieval_v202_population_initial import ledger_fixture


def fixture(tmp_path, monkeypatch):
    out, cfg = ledger_fixture(tmp_path, monkeypatch)
    checked, plan = authority.generation.admission(out)
    catalog = {'path': str(tmp_path / 'catalog.json'), 'sha256': 'a'*64}
    monkeypatch.setattr(authority.generation.original, 'config', lambda: {'catalog': catalog})
    manifest = dict(status=authority.generation.STATUS, launch=record(out / 'launch.json'),
        ledger=checked['_reference'], outputs=checked['artifacts']['draft'], plan=record(out / 'plan/manifest.json'),
        initial100_emitted=True, full_query_population=True, complete_national_source=True,
        old_candidates_opened=False, ready_for_fold1=False, slo_validated=False, downstream_models_changed=False,
        queries=128, population=128, peak_worker_rss_bytes=1000, phases=450+len(plan['chunks']),
        source_and_derived_files_full_hashed=plan['files'], full_bytes_hashed=plan['full_bytes'],
        **authority.generation.original.FLAGS)
    write_json(out / 'manifest.json', manifest)
    return out


def test_full_actual_tiny_admission_then_epoch_recheck_not_a_metric_or_process_permission(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch)
    permit = authority.authorize(out)
    assert permit['status'] == authority.STATUS and permit['all128_before_targets']
    assert not permit['truth_opened'] and not permit['recall_measured'] and not permit['ready_for_fold1']
    assert permit['evaluation_process_supervision_still_required'] and len(permit['controls']) == 128
    authority.recheck(permit)
    write_json(tmp_path / 'facts.json', {'source_changed': True})
    with pytest.raises(ValueError): authority.recheck(permit)


@pytest.mark.parametrize('change', ['partial', 'not_initial', 'ready', 'truth', 'fold0', 'fold1', 'rss', 'old_candidates', 'count', 'bytes', 'output_ref', 'phases', 'producer_removed'])
def test_header_and_complete_producers_required_before_authority_can_return(tmp_path, monkeypatch, change):
    out = fixture(tmp_path, monkeypatch); path = out / 'manifest.json'; m = json.loads(path.read_text())
    if change == 'partial': m['queries'] = 127
    elif change == 'not_initial': m['initial100_emitted'] = False
    elif change == 'ready': m['ready_for_fold1'] = True
    elif change == 'truth': m['truth_opened'] = True
    elif change == 'fold0': m['fold0_opened'] = True
    elif change == 'fold1': m['fold1_opened'] = True
    elif change == 'rss': m['peak_worker_rss_bytes'] = 2**30+1
    elif change == 'old_candidates': m['old_candidates_opened'] = True
    elif change == 'count': m['source_and_derived_files_full_hashed'] -= 1
    elif change == 'bytes': m['full_bytes_hashed'] -= 1
    elif change == 'output_ref': m['outputs'] = m['launch']
    elif change == 'phases': m['phases'] -= 1
    else: (out / 'score063-003/manifest.json').unlink()  # Own fixture only.
    write_json(path, m)
    with pytest.raises((ValueError, FileNotFoundError)): authority.authorize(out)


def test_no_generation_manifest_does_not_trigger_admission_or_any_target_callback(tmp_path, monkeypatch):
    root = tmp_path / 'not_sealed'; root.mkdir()
    monkeypatch.setattr(authority.generation, 'config', lambda: {'output_root': str(tmp_path)})
    def forbidden(unused): raise AssertionError('admission called before full seal')
    monkeypatch.setattr(authority.generation, 'admission', forbidden)
    with pytest.raises(FileNotFoundError): authority.authorize(root)
