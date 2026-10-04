"""Real tiny files/hashes, simulated control/backend: not a national measure."""
import json
from pathlib import Path

import pytest

from scripts import probe_retrieval_v190_staged_merge as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report
from test_probe_retrieval_v188_compact_merge import compact_fixture


def test_same_full_world_reduction_quarantined_without_ledger_inline(tmp_path, monkeypatch):
    out, cfg, events = compact_fixture(tmp_path, monkeypatch)
    source_cfg = dict(partitions=2, sites=9, facts=17, catalog={'fixture': True},
                      addresses={'path': str(tmp_path / 'unused' / 'manifest.json')})
    monkeypatch.setattr(workbench, 'origin', lambda c: (out, source_cfg))
    monkeypatch.setattr(workbench, 'NAMES', ('score000', 'score001', 'audit-facts', 'audit-postings', 'audit-context'))
    def forbidden(path):
        raise AssertionError('Ledger must be in its independent later worker')
    monkeypatch.setattr(workbench.backend, 'load', forbidden)
    result, captured = workbench.compact_merge(out, cfg, events.append)
    assert result['trace']['sites_completed'] == 9 and result['trace']['address_facts'] == 17
    assert len(captured) == 3 and result['candidates'] == []
    assert not result['truth_opened'] and not result['recall_measured']
    assert 'keys_closed' in events


def control_fixture(tmp_path, monkeypatch):
    refs, captured, packets = fixture(tmp_path)
    captured = workbench.original.post_audit.canonical_epochs(captured)
    out = tmp_path / 'own190'
    out.mkdir()
    write_json(out / 'launch.json', {'fixture': 'SIMULATED_NOT_NATIONAL'})
    cfg = dict(stages={'fixture': True}, source_epoch_count=3)
    root = tmp_path / 'source'
    root.mkdir()
    write_json(root / 'manifest.json', {'fixture': 'source'})
    source_cfg = dict(sites=9, facts=17, addresses=record(root / 'manifest.json'))
    write_json(root / 'plan.json', dict(sites=9, facts=17))
    write_json(root / 'epochs.json', dict(epochs=captured))
    launch = dict(plan=record(root / 'plan.json'), epochs=record(root / 'epochs.json'))
    write_json(root / 'launch.json', launch)
    monkeypatch.setattr(workbench, 'config', lambda: cfg)
    monkeypatch.setattr(workbench, 'load', lambda output, dest: {})  # Own UI/process control is simulated.
    monkeypatch.setattr(workbench.original, 'config', lambda: source_cfg)
    def full_load(path):
        assert path == root
        # Actual stage validator requires confirmed successful merge first.
        workbench.stage(out, 'merge')
        return launch, json.loads(verify(launch['plan']).read_text()), json.loads(verify(launch['epochs']).read_text())
    monkeypatch.setattr(workbench.backend, 'load', full_load)
    def check_live(l, p, e):
        assert p['sites'] == 9 and p['facts'] == 17
        workbench.original.check_epochs(e['epochs'])
    monkeypatch.setattr(workbench.backend, 'check_live', check_live)
    def put(phase, artifacts, trace):
        dest = out / phase
        dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        stored = {}
        for k, v in artifacts.items():
            p = dest / (k+'.json'); write_json(p, v); stored[k] = record(p)
        m = dict(status='QUARANTINED_V190_'+phase.upper(), launch=record(out / 'launch.json'),
                 inherited_stages=cfg['stages'], pending_postmerge_ledger=phase == 'merge',
                 preflight=record(dest / 'preflight.json'), artifacts=stored, trace=trace,
                 resources=resource_report(), initial100_emitted=False,
                 old_candidates_opened=False, **workbench.original.FLAGS)
        write_json(dest / 'manifest.json', m)
    put('merge', dict(draft={'query_id': 'CRM_ONLY', 'candidates': [], 'truth_opened': False}, epochs=captured),
        {'signature': 'a'*64})
    receipt = workbench.ledger(out, cfg, lambda step: None)
    put('ledger', {'receipt': receipt}, {'signature': receipt['signature'], 'source_epoch_count': 3})
    return out, cfg, receipt


def replace_receipt(out, change):
    path = out / 'ledger/receipt.json'
    r = json.loads(path.read_text()); change(r); write_json(path, r)
    p = out / 'ledger/manifest.json'
    m = json.loads(p.read_text()); m['artifacts']['receipt'] = record(path); write_json(p, m)


def test_only_two_actual_successful_controls_and_exact_postmerge_receipt_admit(tmp_path, monkeypatch):
    out, cfg, receipt = control_fixture(tmp_path, monkeypatch)
    merged, checked = workbench.admission(out)
    assert not merged['initial100_emitted'] and not checked['initial100_emitted']
    assert receipt['source_epoch_count'] == 3 and receipt['sites'] == 9 and receipt['facts'] == 17


@pytest.mark.parametrize('change', ['no_ledger', 'unconfirmed', 'nonzero', 'rss', 'stop', 'truth',
                                   'wrong_draft', 'wrong_merge', 'wrong_signature', 'missing_epoch',
                                   'different_source', 'epoch_changed', 'emitted_early'])
def test_no_release_without_entire_current_successful_postmerge_source_receipt(tmp_path, monkeypatch, change):
    out, cfg, receipt = control_fixture(tmp_path, monkeypatch)
    if change == 'no_ledger':
        (out / 'ledger/manifest.json').unlink()  # Own test fixture only.
    elif change in ('unconfirmed', 'nonzero'):
        s = supervision()
        s.update(own_child_exit_confirmed=change != 'unconfirmed', exit_code=1 if change == 'nonzero' else 0)
        write_json(out / 'ledger/supervision.json', s)
    elif change in ('rss', 'emitted_early'):
        p = out / 'ledger/manifest.json'
        m = json.loads(p.read_text())
        if change == 'rss':
            m['resources']['peak_rss_bytes'] = 2**30+1
        else:
            m['initial100_emitted'] = True
        write_json(p, m)
    elif change == 'stop':
        write_json(out / 'ledger/worker_stop.json', {'status': 'STOP'})
    elif change == 'epoch_changed':
        write_json(tmp_path / 'facts.json', {'drift': True})
    else:
        def mutate(r):
            if change == 'truth':
                r['truth_opened'] = True
            elif change == 'wrong_draft':
                r['draft']['sha256'] = 'b'*64
            elif change == 'wrong_merge':
                r['merge']['sha256'] = 'b'*64
            elif change == 'wrong_signature':
                r['signature'] = 'b'*64
            elif change == 'missing_epoch':
                r['source_epoch_count'] = 2
            else:
                r['authorities']['source']['sha256'] = 'b'*64
        replace_receipt(out, mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.admission(out)


@pytest.mark.parametrize('key', ['source_epoch_count', 'max_rss_bytes', 'max_seconds'])
def test_no_ledger_or_budget_shrink_or_relaxation(tmp_path, monkeypatch, key):
    cfg = json.loads(workbench.CONFIG.read_text())
    if key == 'source_epoch_count':
        cfg[key] -= 1
    else:
        cfg['resources'][key] += 1
    p = tmp_path / 'config.json'; write_json(p, cfg)
    monkeypatch.setattr(workbench, 'CONFIG', p)
    with pytest.raises(ValueError):
        workbench.config()
