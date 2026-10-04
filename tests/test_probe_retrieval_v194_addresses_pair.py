"""Full real tiny hashes, simulated supervision/population, no national metric."""
import json
from types import SimpleNamespace

import pytest

from scripts import probe_retrieval_v194_addresses_pair as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from test_retrieval_postscore_audit_v187 import fixture
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def control_fixture(tmp_path, monkeypatch):
    refs, epochs, packets = fixture(tmp_path)
    refs = workbench.hashes.references(refs.values())
    epochs = workbench.hashes.canonical_epochs(epochs)
    out = tmp_path / 'own194'; out.mkdir()
    write_json(out / 'launch.json', {'fixture': 'SIMULATED_CONTROL_NOT_NATIONAL'})
    spans = [dict(start=0, stop=9, facts=13)]
    cfg = dict(positions=[0, 1], expected_spans=spans,
               query0_expected_proof_hashes={'coverage': refs[0]['sha256'], 'exact': refs[1]['sha256']})
    monkeypatch.setattr(workbench, 'config', lambda: cfg)
    monkeypatch.setattr(workbench, 'load', lambda output, dest: {})
    class TinyPopulation:
        def select(self, i):
            return dict(query_id='Q'+str(i), fold=3-i), None
    monkeypatch.setattr(workbench, 'inputs', lambda c: (
        {'sites': 9, 'aliases': 4, 'facts': 13}, TinyPopulation(), {}))
    monkeypatch.setattr(workbench.original, 'query',
        lambda row, geo: Query(name=row['query_id'], address='ADDR'+row['query_id'],
                              number=row['query_id'][-1], xy=(float(row['query_id'][-1]), 3.)))
    monkeypatch.setattr(workbench, 'address_authority', lambda c: 'SIMULATED_FULL_AUTHORITY')
    monkeypatch.setattr(workbench, 'check_address_authority', lambda a: None)
    def put(phase, values):
        dest = out / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        artifacts = {}
        for k, v in values.items():
            p = dest / (k+'.json'); write_json(p, v); artifacts[k] = record(p)
        m = dict(status='QUARANTINED_V194_'+phase.upper(), pending_source_hash=phase == 'addresses',
                 launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'),
                 resources=resource_report(), artifacts=artifacts, old_candidates_opened=False,
                 initial100_emitted=False, **workbench.original.FLAGS)
        write_json(dest / 'manifest.json', m)
    proofs = dict(rows=[dict(position=i, query_id='Q'+str(i), fold=3-i,
                    query=workbench.query_control({'query_id': 'Q'+str(i)}, {}),
                    artifacts={'coverage': refs[0], 'exact': refs[1]}, aliases=4,
                    trace=dict(sites=9, facts=13, aliases=4, spans=spans,
                               every_original_alias_checked=True, query_snapshot_file_backed=True,
                               candidates_emitted=False, recall_measured=False,
                               slo_validated=False, ready_for_fold1=False)) for i in (0, 1)],
                  population=128, initial100_emitted=False, truth_opened=False, recall_measured=False)
    put('addresses', dict(proofs=proofs, targets=refs, epochs=epochs))
    receipt = workbench.audit(out, lambda p: None)  # Every tiny file is REALLY hashed.
    put('audit', {'receipt': receipt})
    return out, cfg


def mutate_artifact(out, phase, name, change):
    dest = out / phase
    p = dest / (name+'.json'); v = json.loads(p.read_text()); change(v); write_json(p, v)
    p = dest / 'manifest.json'; m = json.loads(p.read_text())
    m['artifacts'][name] = record(dest / (name+'.json')); write_json(p, m)


def test_full_hash_actual_control_and_exact_pair_required_before_address_proofs_admitted(tmp_path, monkeypatch):
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
            else: p['rows'][0]['artifacts']['coverage']['sha256'] = 'b'*64
        mutate_artifact(out, 'addresses', 'proofs', mutate)
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


@pytest.mark.parametrize('change', ['query_mixed', 'xy_mixed', 'aliases', 'facts', 'sites', 'spans',
    'partial_aliases', 'candidate', 'recall', 'population', 'missing_exact', 'unaudited_proof'])
def test_each_query_owns_whole_address_and_geo_and_all_proofs_were_audited(tmp_path, monkeypatch, change):
    out, cfg = control_fixture(tmp_path, monkeypatch)
    def mutate(p):
        row = p['rows'][1]
        if change == 'query_mixed': row['query'] = p['rows'][0]['query'].copy()
        elif change == 'xy_mixed': row['query']['xy'] = p['rows'][0]['query']['xy'].copy()
        elif change in ('aliases', 'facts', 'sites'): row['trace'][change] -= 1
        elif change == 'spans': row['trace']['spans'][0]['stop'] -= 1
        elif change == 'partial_aliases': row['trace']['every_original_alias_checked'] = False
        elif change == 'candidate': row['trace']['candidates_emitted'] = True
        elif change == 'recall': p['recall_measured'] = True
        elif change == 'population': p['population'] = 2
        elif change == 'missing_exact': row['artifacts'].pop('exact')
        elif change == 'unaudited_proof':
            path = tmp_path / 'unaudited.json'; write_json(path, {'new': 'not full audited'})
            row['artifacts']['coverage'] = record(path)
    mutate_artifact(out, 'addresses', 'proofs', mutate)
    # Even if another control process binds this changed proof ledger, the
    # admission must reject query/source incompleteness; no national claim.
    receipt = workbench.audit(out, lambda p: None)
    mutate_artifact(out, 'audit', 'receipt', lambda r: r.update(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.admission(out)


def test_original_full_source_authority_is_checked_before_and_after_hash_and_at_admission(tmp_path, monkeypatch):
    out, cfg = control_fixture(tmp_path, monkeypatch)
    events = []
    monkeypatch.setattr(workbench, 'address_authority', lambda c: events.append('full93k_before') or 'AUTH')
    monkeypatch.setattr(workbench, 'check_address_authority', lambda a: events.append('full93k_after'))
    workbench.audit(out, lambda p: events.append('actual_full_hash'))
    assert events[0] == 'full93k_before' and events[-1] == 'full93k_after'
    assert 'actual_full_hash' in events
    events.clear()
    workbench.admission(out)
    assert events == ['full93k_before']


def test_changed_original_authority_blocks_even_when_read_target_hashes_match(tmp_path, monkeypatch):
    out, cfg = control_fixture(tmp_path, monkeypatch)
    def fail(*args): raise ValueError('SIMULATED original93507 drift or STOP')
    monkeypatch.setattr(workbench, 'check_address_authority', fail)
    with pytest.raises(ValueError): workbench.audit(out, lambda p: None)
    monkeypatch.setattr(workbench, 'address_authority', fail)
    with pytest.raises(ValueError): workbench.admission(out)


def test_query_xy_control_roundtrips_without_changing_coordinates(monkeypatch):
    q = Query(name='OWN', address='OWN ADDRESS', xy=(123.5, 456.25), reliable_geo=True)
    monkeypatch.setattr(workbench.original, 'query', lambda r, g: q)
    actual = workbench.query_control({}, {})
    assert actual == json.loads(json.dumps(actual))
    assert actual['xy'] == [123.5, 456.25]
