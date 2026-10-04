"""Full real tiny hashes, simulated supervision/population, no national metric."""
import json
from types import SimpleNamespace

import pytest

from scripts import probe_retrieval_v193_names_pair as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def control_fixture(tmp_path, monkeypatch):
    refs, epochs, packets = fixture(tmp_path)
    refs = workbench.hashes.references(refs.values())
    epochs = workbench.hashes.canonical_epochs(epochs)
    out = tmp_path / 'own193'; out.mkdir()
    write_json(out / 'launch.json', {'fixture': 'SIMULATED_CONTROL_NOT_NATIONAL'})
    cfg = dict(positions=[0, 1], query0_expected_gather_digest='a'*64)
    monkeypatch.setattr(workbench, 'config', lambda: cfg)
    monkeypatch.setattr(workbench, 'load', lambda output, dest: {})
    class TinyPopulation:
        def select(self, i):
            return dict(query_id='Q'+str(i), fold=3-i), None
    monkeypatch.setattr(workbench, 'inputs', lambda c: ({'sites': 9}, TinyPopulation()))
    def put(phase, values):
        dest = out / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        artifacts = {}
        for k, v in values.items():
            p = dest / (k+'.json'); write_json(p, v); artifacts[k] = record(p)
        m = dict(status='QUARANTINED_V193_'+phase.upper(), pending_source_hash=phase == 'names',
                 launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'),
                 resources=resource_report(), artifacts=artifacts, old_candidates_opened=False,
                 initial100_emitted=False, **workbench.original.FLAGS)
        write_json(dest / 'manifest.json', m)
    proofs = dict(rows=[dict(position=i, query_id='Q'+str(i), fold=3-i,
                    artifacts={'coverage': refs[0], 'exact': refs[1]},
                    gather=dict(sites=9, every_original_site_gathered_once=True, candidates_emitted=False,
                                original_float64_gather_digest='a'*64)) for i in (0, 1)],
                  initial100_emitted=False, truth_opened=False)
    put('names', dict(proofs=proofs, targets=refs, epochs=epochs))
    receipt = workbench.audit(out, lambda p: None)  # Every tiny file is REALLY hashed.
    put('audit', {'receipt': receipt})
    return out, cfg


def mutate_artifact(out, phase, name, change):
    dest = out / phase
    p = dest / (name+'.json'); v = json.loads(p.read_text()); change(v); write_json(p, v)
    p = dest / 'manifest.json'; m = json.loads(p.read_text())
    m['artifacts'][name] = record(dest / (name+'.json')); write_json(p, m)


def test_full_hash_actual_control_and_exact_pair_required_before_name_proofs_admitted(tmp_path, monkeypatch):
    out, cfg = control_fixture(tmp_path, monkeypatch)
    named, checked = workbench.admission(out)
    assert named['pending_source_hash'] and not checked['pending_source_hash']
    assert not named['initial100_emitted'] and not checked['truth_opened']


@pytest.mark.parametrize('change', ['missing_audit', 'parent_nonzero', 'parent_unconfirmed', 'rss', 'stop',
                                   'truth', 'wrong_targets', 'wrong_epoch_ref', 'unhashed', 'wrong_files',
                                   'wrong_bytes', 'different_first_query', 'missing_query', 'source_epoch'])
def test_no_partial_source_hash_or_query_pair_or_supervision_accepted(tmp_path, monkeypatch, change):
    out, cfg = control_fixture(tmp_path, monkeypatch)
    if change == 'missing_audit':
        (out / 'audit/manifest.json').unlink()  # Own pytest fixture only.
    elif change in ('parent_nonzero', 'parent_unconfirmed'):
        s = supervision(); s.update(exit_code=1 if change == 'parent_nonzero' else 0,
                                    own_child_exit_confirmed=change != 'parent_unconfirmed')
        write_json(out / 'audit/supervision.json', s)
    elif change == 'rss':
        p = out / 'audit/manifest.json'; m = json.loads(p.read_text())
        m['resources']['peak_rss_bytes'] = 2**30+1; write_json(p, m)
    elif change == 'stop':
        write_json(out / 'audit/worker_stop.json', {'status': 'STOP'})
    elif change == 'source_epoch':
        write_json(tmp_path / 'facts.json', {'changed': True})
    elif change in ('different_first_query', 'missing_query'):
        def mutate(p):
            if change == 'missing_query': p['rows'].pop()
            else: p['rows'][0]['gather']['original_float64_gather_digest'] = 'b'*64
        mutate_artifact(out, 'names', 'proofs', mutate)
    else:
        def mutate(r):
            if change == 'truth': r['truth_opened'] = True
            elif change == 'wrong_targets': r['exact_targets_digest'] = 'b'*64
            elif change == 'wrong_epoch_ref': r['epochs']['sha256'] = 'b'*64
            elif change == 'unhashed': r['every_expected_source_full_hash_verified'] = False
            elif change == 'wrong_files': r['files'] -= 1
            else: r['full_bytes_hashed'] -= 1
        mutate_artifact(out, 'audit', 'receipt', mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.admission(out)


@pytest.mark.parametrize('change', ['rss', 'wall', 'subset', 'selected_pair', 'bool_position'])
def test_physical_batch_never_shrinks128_population_or_expands_budget(tmp_path, monkeypatch, change):
    cfg = json.loads(workbench.CONFIG.read_text())
    if change == 'rss': cfg['resources']['max_rss_bytes'] += 1
    elif change == 'wall': cfg['resources']['max_seconds'] += 1
    elif change == 'subset': cfg['queries'] = 2
    elif change == 'selected_pair': cfg['positions'] = [2, 3]
    else: cfg['positions'][0] = False
    p = tmp_path / 'config.json'; write_json(p, cfg); monkeypatch.setattr(workbench, 'CONFIG', p)
    with pytest.raises(ValueError): workbench.config()
