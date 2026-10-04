"""Actual tiny full hashes; explicitly simulated supervision/population only."""
import json

import pytest

from scripts import probe_retrieval_v201_names_ram as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def control_fixture(tmp_path, monkeypatch):
    refs, epochs, unused = fixture(tmp_path)
    refs = workbench.hashes.references(refs.values())
    epochs = workbench.hashes.canonical_epochs(epochs)
    out = tmp_path / 'own201'; out.mkdir()
    write_json(out / 'launch.json', {'fixture': 'SIMULATED_CONTROL_NOT_NATIONAL'})
    cfg = dict(positions=[8])
    monkeypatch.setattr(workbench, 'config', lambda: cfg)
    monkeypatch.setattr(workbench, 'load', lambda output, dest: {})
    class TinyPopulation:
        def select(self, i):
            return dict(query_id='Q'+str(i), fold=3), None
    monkeypatch.setattr(workbench, 'inputs', lambda c: ({'sites': 9}, TinyPopulation()))
    def put(phase, values):
        dest = out / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        artifacts = {}
        for k, v in values.items():
            p = dest / (k+'.json'); write_json(p, v); artifacts[k] = record(p)
        m = dict(status='QUARANTINED_V201_'+phase.upper(), pending_source_hash=phase == 'names',
                 launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'),
                 resources=resource_report(), artifacts=artifacts, old_candidates_opened=False,
                 initial100_emitted=False, **workbench.original.FLAGS)
        write_json(dest / 'manifest.json', m)
    proofs = dict(rows=[dict(position=8, query_id='Q8', fold=3,
                    artifacts={'coverage': record(refs[0]['path']), 'exact': record(refs[1]['path'])},
                    gather=dict(sites=9, every_original_site_gathered_once=True, candidates_emitted=False))],
                  population=128, recall_measured=False, initial100_emitted=False, truth_opened=False)
    put('names', dict(proofs=proofs, targets=refs, epochs=epochs))
    receipt = workbench.audit(out, lambda p: None)
    put('audit', {'receipt': receipt})
    return out, cfg


def mutate_artifact(out, phase, name, change):
    dest = out / phase
    p = dest / (name+'.json'); v = json.loads(p.read_text()); change(v); write_json(p, v)
    p = dest / 'manifest.json'; m = json.loads(p.read_text())
    m['artifacts'][name] = record(dest / (name+'.json')); write_json(p, m)
    if phase == 'names':
        # Rebind simulated audit envelope so semantic guards are actually
        # reached; do not pass solely because an outer checksum became stale.
        receipt_path = out / 'audit/receipt.json'; r = json.loads(receipt_path.read_text())
        r['names'] = record(dest / 'manifest.json'); r[name] = record(dest / (name+'.json'))
        write_json(receipt_path, r)
        audit_path = out / 'audit/manifest.json'; a = json.loads(audit_path.read_text())
        a['artifacts']['receipt'] = record(receipt_path); write_json(audit_path, a)


def test_actual_full_hash_query4_not_recall_or_initial100(tmp_path, monkeypatch):
    out, unused = control_fixture(tmp_path, monkeypatch)
    named, checked = workbench.admission(out)
    assert named['pending_source_hash'] and not checked['pending_source_hash']
    assert not named['initial100_emitted'] and not checked['truth_opened']


@pytest.mark.parametrize('change', ['missing_audit', 'parent_nonzero', 'parent_unconfirmed', 'rss', 'stop',
    'truth', 'wrong_targets', 'wrong_epoch_ref', 'unhashed', 'wrong_files', 'wrong_bytes', 'wrong_query',
    'missing_query', 'source_epoch', 'fold', 'population', 'measured', 'proof_truth', 'sites', 'gather',
    'candidates', 'extra_artifact', 'wrong_proof_keys', 'unaudited_proof', 'old_candidates', 'bool_position', 'proof_size'])
def test_reject_partial_sources_proofs_or_processes(tmp_path, monkeypatch, change):
    out, unused = control_fixture(tmp_path, monkeypatch)
    if change == 'missing_audit':
        (out / 'audit/manifest.json').unlink()  # Own pytest fixture only.
    elif change in ('parent_nonzero', 'parent_unconfirmed'):
        s = supervision(); s.update(exit_code=1 if change == 'parent_nonzero' else 0,
                                    own_child_exit_confirmed=change != 'parent_unconfirmed')
        write_json(out / 'audit/supervision.json', s)
    elif change in ('rss', 'old_candidates', 'extra_artifact'):
        p = out / 'names/manifest.json'; m = json.loads(p.read_text())
        if change == 'rss': m['resources']['peak_rss_bytes'] = 2**30+1
        elif change == 'old_candidates': m['old_candidates_opened'] = True
        else: m['artifacts']['extra'] = m['artifacts']['proofs']
        write_json(p, m)
    elif change == 'stop': write_json(out / 'audit/worker_stop.json', {'status': 'STOP'})
    elif change == 'source_epoch': write_json(tmp_path / 'facts.json', {'changed': True})
    elif change in ('wrong_query', 'missing_query', 'fold', 'population', 'measured', 'proof_truth', 'sites',
                    'gather', 'candidates', 'wrong_proof_keys', 'unaudited_proof', 'bool_position', 'proof_size'):
        def mutate(p):
            if change == 'missing_query': p['rows'].pop(); return
            row = p['rows'][0]
            if change == 'wrong_query': row['query_id'] = 'Q0'
            elif change == 'fold': row['fold'] = 1
            elif change == 'population': p['population'] = 1
            elif change == 'measured': p['recall_measured'] = True
            elif change == 'proof_truth': p['truth_opened'] = True
            elif change == 'sites': row['gather']['sites'] -= 1
            elif change == 'gather': row['gather']['every_original_site_gathered_once'] = False
            elif change == 'candidates': row['gather']['candidates_emitted'] = True
            elif change == 'wrong_proof_keys': row['artifacts']['new'] = row['artifacts'].pop('exact')
            elif change == 'bool_position': row['position'] = True
            elif change == 'proof_size': row['artifacts']['coverage']['size_bytes'] += 1
            else:
                q = tmp_path / 'unaudited.json'; write_json(q, {'proof': 'outside audited inventory'})
                row['artifacts']['coverage'] = record(q)
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


@pytest.mark.parametrize('change', ['rss', 'wall', 'subset', 'selected_query', 'bool_position', 'ram', 'relation'])
def test_no_budget_expansion_or_new_quality_selection(tmp_path, monkeypatch, change):
    cfg = json.loads(workbench.CONFIG.read_text())
    if change == 'rss': cfg['resources']['max_rss_bytes'] += 1
    elif change == 'wall': cfg['resources']['max_seconds'] += 1
    elif change == 'subset': cfg['queries'] = 1
    elif change == 'selected_query': cfg['positions'] = [0]
    elif change == 'ram': cfg['max_counter_ram_bytes'] += 1
    elif change == 'relation': cfg['max_relation_chunk'] += 1
    else: cfg['positions'][0] = False
    p = tmp_path / 'config.json'; write_json(p, cfg); monkeypatch.setattr(workbench, 'CONFIG', p)
    with pytest.raises(ValueError): workbench.config()


def test_real_config_keeps_actual20second_source_bound():
    workbench.config()
    assert workbench.original.config()['max_proof_query_seconds'] == 20
