"""Real tiny file hashes and score oracle; simulated process controls only."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import probe_retrieval_v195_pair_initial as workbench
from src.xgb_matcher import retrieval_pair_audit_v195 as pair
from src.xgb_matcher import retrieval_staged_initial_v183 as kernel
from src.xgb_matcher.retrieval_certified_address_facts_v181 import CertifiedAddressFacts
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import search
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_retrieval_staged_initial_v183 import keys_catalog, Names
from test_retrieval_certified_address_facts_v181 import fixture as facts_fixture, snapshot
from test_retrieval_initial_collector_v146 import Attributes
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def controls():
    return [dict(position=i, query_id='OWN'+str(i), fold=3-i, signature=('a' if i == 0 else 'b')*64)
            for i in (0, 1)]


def setup(tmp_path, monkeypatch):
    refs, epochs, old = fixture(tmp_path)
    out = tmp_path / 'own195'; out.mkdir()
    write_json(out / 'launch.json', {'SIMULATED_CONTROL_NOT_NATIONAL': True})
    spans = [p['span'] for p in old]
    cfg = dict(sites=9, facts=17, expected_spans=spans, partitions=2, positions=[0, 1])
    monkeypatch.setattr(workbench, 'config', lambda: cfg)
    monkeypatch.setattr(workbench, 'load', lambda *args: None)
    def put(phase, artifacts, trace):
        dest = out / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate())
        write_json(dest / 'supervision.json', supervision())
        values = {}
        for k, v in artifacts.items():
            write_json(dest / (k+'.json'), v); values[k] = record(dest / (k+'.json'))
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V195_'+phase.upper(),
            launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'),
            pending_source_hash=phase.startswith('score'), resources=resource_report(),
            artifacts=values, trace=trace, initial100_emitted=False,
            old_candidates_opened=False, **workbench.original.FLAGS))
    for i, span in enumerate(spans):
        p = pair.packet({k: [v] for k, v in refs.items()}, epochs, controls(), span, 9, 17)
        artifacts = {'packet': p}
        for row in controls():
            artifacts['part'+str(row['position'])] = dict(signature=row['signature'], span=span,
                candidates=[], partition_only_not_initial_output=True, same_scores_never_rescored=True,
                stats=dict(sites_completed=span['stop']-span['start'], address_facts=span['facts'],
                    every_site_scored_once=True, fixed_reservoir=None, score_after_fixed_pool=False, initial_cap=100))
        put(f'score{i:03d}', artifacts, {'rows': controls()})
    plan, producers, packets = pair.score_plan(out, cfg, workbench.stage)
    for group in pair.audit.GROUPS:
        receipt = pair.audit.audit_group(plan, group)  # Every tiny byte REALLY hashed.
        receipt.update(score_stages=producers, audit_packets=packets)
        put('audit-'+group, {'receipt': receipt}, {'group': group})
    return out, cfg


def mutate(out, phase, artifact, change):
    path = out / phase / (artifact+'.json')
    v = json.loads(path.read_text()); change(v); write_json(path, v)
    path = out / phase / 'manifest.json'
    m = json.loads(path.read_text()); m['artifacts'][artifact] = record(out / phase / (artifact+'.json'))
    write_json(path, m)


def test_two_independent_recipes_every_national_span_and_full_hash_once(tmp_path, monkeypatch):
    out, cfg = setup(tmp_path, monkeypatch)
    p = pair.audited_plan(out, cfg, workbench.stage)
    assert p['rows'] == controls()
    assert p['signature'] == pair.signature(controls())
    assert len(p['source_epochs']) == 3 and sum(len(v) for v in p['groups'].values()) == 3
    assert not p.get('initial100_emitted', False)


@pytest.mark.parametrize('change', ['missing_tail', 'query_mixed', 'recipe_mixed', 'source_subset', 'truth',
    'sites', 'facts', 'span', 'part_recipe', 'part_sites', 'part_facts', 'part_cap', 'rerank',
    'receipt_missing', 'receipt_producers', 'receipt_signature', 'receipt_bytes', 'receipt_full_hash',
    'parent_nonzero', 'parent_unconfirmed', 'rss', 'stop'])
def test_no_initial_release_for_any_partial_query_source_score_or_actual_control(tmp_path, monkeypatch, change):
    out, cfg = setup(tmp_path, monkeypatch)
    phase = 'score001'
    if change == 'missing_tail': (out / phase / 'manifest.json').unlink()
    elif change == 'receipt_missing': (out / 'audit-context/manifest.json').unlink()
    elif change.startswith('receipt_'):
        def alter(r):
            if change == 'receipt_producers': r['score_stages'].pop()
            elif change == 'receipt_signature': r['signature'] = 'c'*64
            elif change == 'receipt_bytes': r['full_bytes_hashed'] -= 1
            else: r['every_expected_source_full_hash_verified'] = False
        mutate(out, 'audit-context', 'receipt', alter)
    elif change in ('parent_nonzero', 'parent_unconfirmed'):
        s = supervision(); s.update(exit_code=int(change == 'parent_nonzero'),
                                    own_child_exit_confirmed=change != 'parent_unconfirmed')
        write_json(out / phase / 'supervision.json', s)
    elif change == 'rss':
        path = out / phase / 'manifest.json'; m = json.loads(path.read_text())
        m['resources']['peak_rss_bytes'] = 2**30+1; write_json(path, m)
    elif change == 'stop': write_json(out / phase / 'worker_stop.json', {'status': 'STOP'})
    elif change.startswith('part_') or change == 'rerank':
        def alter(p):
            if change == 'part_recipe': p['signature'] = 'a'*64
            elif change == 'part_sites': p['stats']['sites_completed'] -= 1
            elif change == 'part_facts': p['stats']['address_facts'] -= 1
            elif change == 'part_cap': p['stats']['initial_cap'] = 101
            else: p['same_scores_never_rescored'] = False
        mutate(out, phase, 'part1', alter)
    else:
        def alter(p):
            if change == 'query_mixed': p['rows'][1]['query_id'] = p['rows'][0]['query_id']
            elif change == 'recipe_mixed': p['rows'][1]['signature'] = 'c'*64
            elif change == 'source_subset': p['groups']['facts'] = []
            elif change == 'truth': p['truth_opened'] = True
            elif change in ('sites', 'facts'): p[change] -= 1
            else: p['span']['start'] -= 1
        mutate(out, phase, 'packet', alter)
    with pytest.raises((ValueError, FileNotFoundError)):
        pair.audited_plan(out, cfg, workbench.stage)


def test_physical_audit_dedup_never_drops_a_target_or_checksum_conflict(tmp_path):
    refs, epochs, old = fixture(tmp_path)
    g = {k: [v] for k, v in refs.items()}
    g['context'].extend(refs.values())
    p = pair.packet(g, epochs, controls(), old[0]['span'], 9, 17)
    assert sum(len(v) for v in p['groups'].values()) == 3
    g['context'].append({**refs['facts'], 'sha256': 'f'*64})
    with pytest.raises(ValueError): pair.packet(g, epochs, controls(), old[0]['span'], 9, 17)


@pytest.mark.parametrize('change', ['one_query', 'bool_position', 'fold0', 'fold1', 'bool_fold', 'id_duplicate', 'bad_recipe'])
def test_safe_own_pair_controls_before_source_plan(change):
    rows = controls()
    if change == 'one_query': rows.pop()
    elif change == 'bool_position': rows[0]['position'] = False
    elif change == 'fold0': rows[1]['fold'] = 0
    elif change == 'fold1': rows[1]['fold'] = 1
    elif change == 'bool_fold': rows[1]['fold'] = True
    elif change == 'id_duplicate': rows[1]['query_id'] = rows[0]['query_id']
    else: rows[1]['signature'] = 'X'*64
    with pytest.raises(ValueError): pair.records(rows)


@pytest.mark.parametrize('change', ['rss', 'wall', 'subset', 'bool_position', 'partition_subset', 'span_gap'])
def test_same_business_world_budget_and_population_before_national_run(tmp_path, monkeypatch, change):
    c = json.loads(workbench.CONFIG.read_text())
    if change == 'rss': c['resources']['max_rss_bytes'] += 1
    elif change == 'wall': c['resources']['max_seconds'] += 1
    elif change == 'subset': c['queries'] = 2
    elif change == 'bool_position': c['positions'][0] = False
    elif change == 'partition_subset': c['partitions'] = 3
    else: c['expected_spans'][1]['start'] += 1
    path = tmp_path / 'config.json'; write_json(path, c); monkeypatch.setattr(workbench, 'CONFIG', path)
    with pytest.raises(ValueError): workbench.config()


def test_shared_real_tiny_fact_source_two_own_queries_equals_independent_unsplit146_oracle(tmp_path, monkeypatch):
    index, _, _ = facts_fixture(tmp_path, monkeypatch)
    source = CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    cat = keys_catalog(np.arange(100001, 100001+source.sites, dtype=np.uint64)[::-1])
    flat, unused = kernel.flatten_names(Names(source.sites), tmp_path / 'flat', lambda: None, 3)
    queries = [Query(postcode='75001', number='7'), Query(postcode='69001', number='9')]
    outputs = []
    try:
        for position, q in enumerate(queries):
            w = np.linspace(.01, 3., 30) * (position+1)
            bound = source.bind(snap, q, 1)
            expected, unused = search(q, w, cat, flat, Attributes(source.sites), bound, chunk=2)
            bound.finish()
            parts = []
            for span in kernel.spans(source, 3):
                own = kernel.PartitionFacts(source, snap, q, span, 1)
                parts.append(kernel.partition_search(q, w, cat, flat, Attributes(source.sites), own,
                    kernel.recipe(q, w, {'tiny_only': True}, 3-position), chunk=2))
                assert own.cursor == span['stop'] and own.consumed == span['facts'] and source.active is None
            actual, trace = kernel.merge_partitions(cat, source.facts, parts)
            assert actual == expected and trace['sites_completed'] == source.sites
            assert trace['address_facts'] == source.facts and trace['fixed_reservoir'] is None
            outputs.append(actual)
        assert outputs[0] != outputs[1]
        source.verify_after()
    finally:
        flat.close(); snap.close(); source.close()


def merged_setup(tmp_path, monkeypatch):
    out, c = setup(tmp_path, monkeypatch)
    plan = pair.audited_plan(out, c, workbench.stage)
    epoch_path = out / 'merge/epochs.json'
    def put(phase, values, trace):
        dest = out / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate()); write_json(dest / 'supervision.json', supervision())
        refs = {}
        for k, v in values.items():
            write_json(dest / (k+'.json'), v); refs[k] = record(dest / (k+'.json'))
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V195_'+phase.upper(),
            launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'), resources=resource_report(),
            pending_source_hash=False, artifacts=refs, trace=trace,
            initial100_emitted=False, old_candidates_opened=False, **workbench.original.FLAGS))
    rows = []
    for control in controls():
        rows.append(dict(position=control['position'], query_id=control['query_id'], fold=control['fold'], candidates=[],
            trace=dict(signature=control['signature'], sites_completed=9, address_facts=17, physical_partitions=2,
                initial_cap=100, fixed_reservoir=None, score_after_fixed_pool=False,
                every_site_scored_once=True, same_original_score_never_rescored=True)))
    trace = dict(rows=controls(), signature=plan['signature'], producers=plan['producers'],
                 packets=plan['packets'], audits=plan['audits'])
    put('merge', {'draft': dict(rows=rows, population=128, truth_opened=False, recall_measured=False,
        full_query_population=False, ready_for_fold1=False), 'epochs': plan['source_epochs']}, trace)
    merged = workbench.stage(out, 'merge')
    receipt = dict(merge=merged['_reference'], draft=merged['artifacts']['draft'], epochs=merged['artifacts']['epochs'],
        rows=controls(), signature=plan['signature'], source_epoch_count=93507, sites=9, facts=17,
        initial100_emitted=False, **workbench.original.FLAGS)
    put('ledger', {'receipt': receipt}, {'rows': controls(), 'signature': plan['signature'], 'source_epoch_count': 93507})
    monkeypatch.setattr(workbench, 'prefix', lambda c: ('SIMULATED_SEALED_NO_CANDIDATE', None, None))
    return out, c


def test_independent_complete_postmerge_ledger_before_pair_admitted(tmp_path, monkeypatch):
    out, c = merged_setup(tmp_path, monkeypatch)
    m, l = workbench.admission(out)
    assert not m['initial100_emitted'] and not l['truth_opened']


@pytest.mark.parametrize('change', ['ledger_missing', 'ledger_parent_nonzero', 'ledger_source_subset',
    'ledger_truth', 'ledger_old_draft', 'postmerge_source_epoch', 'audit_deleted_after_merge',
    'part_changed_after_merge', 'output_cap101', 'output_population2', 'output_partial_world', 'output_recipe_mixed'])
def test_never_admit_partial_or_unaudited_or_modified_or_overcap_pair(tmp_path, monkeypatch, change):
    out, c = merged_setup(tmp_path, monkeypatch)
    if change == 'ledger_missing': (out / 'ledger/manifest.json').unlink()
    elif change == 'ledger_parent_nonzero':
        s = supervision(); s['exit_code'] = 1; write_json(out / 'ledger/supervision.json', s)
    elif change == 'postmerge_source_epoch': write_json(tmp_path / 'facts.json', {'drift': True})
    elif change == 'audit_deleted_after_merge': (out / 'audit-context/manifest.json').unlink()
    elif change == 'part_changed_after_merge': mutate(out, 'score001', 'part1', lambda p: p.update(signature='c'*64))
    elif change.startswith('ledger_'):
        def alter(r):
            if change == 'ledger_source_subset': r['source_epoch_count'] -= 1
            elif change == 'ledger_truth': r['truth_opened'] = True
            else: r['draft']['sha256'] = 'c'*64
        mutate(out, 'ledger', 'receipt', alter)
    else:
        def alter(p):
            if change == 'output_cap101': p['rows'][1]['candidates'] = [{'siret': str(i)} for i in range(101)]
            elif change == 'output_population2': p['population'] = 2
            elif change == 'output_partial_world': p['rows'][1]['trace']['sites_completed'] -= 1
            else: p['rows'][1]['trace']['signature'] = 'a'*64
        mutate(out, 'merge', 'draft', alter)
        # Bind the altered draft to a simulated ledger to test output guards,
        # rather than relying only on the earlier reference mismatch guard.
        merged = workbench.stage(out, 'merge')
        mutate(out, 'ledger', 'receipt', lambda r: r.update(merge=merged['_reference'], draft=merged['artifacts']['draft']))
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.admission(out)
