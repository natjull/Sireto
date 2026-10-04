"""ALL128 protocol on tiny REAL hashes; process/resource evidence simulated explicitly."""
from copy import deepcopy
import json

import pytest

from scripts import run_retrieval_v197_population_initial as work
from src.xgb_matcher import retrieval_population_audit_v197 as whole
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json
from test_retrieval_postscore_audit_v187 import fixture
from test_retrieval_population_audit_v197 import controls, outputs, population
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def setup(tmp_path, monkeypatch):
    refs, epochs, unused = fixture(tmp_path)
    out = tmp_path / 'own197'; out.mkdir()
    spans = [dict(start=a, stop=b, facts=f) for a, b, f in ((0, 2, 4), (2, 4, 4), (4, 6, 4), (6, 9, 5))]
    cfg = dict(output_root=str(tmp_path), queries=128, sites=9, aliases=3, facts=17, partitions=4,
               expected_spans=spans, max_audit_bytes=whole.MAX_AUDIT_BYTES, max_audit_files=whole.MAX_AUDIT_FILES)
    monkeypatch.setattr(work, 'config', lambda: cfg)
    monkeypatch.setattr(work, 'inputs', lambda c: (cfg, population(), {}, {}, {}))
    code = tmp_path / 'simulated_process_fixture_code.json'; write_json(code, {'simulation_not_national': True})
    write_json(out / 'launch.json', dict(config=cfg, implementation={'unit_code': record(code)},
        old_candidates_opened=False, **work.original.FLAGS))
    capsule = whole.capsule({k: [r] for k, r in refs.items()}, epochs, out)
    dest = out / 'score000-000'; dest.mkdir()
    write_json(dest / 'source_capsule.json', capsule); capsule_ref = record(dest / 'source_capsule.json')
    shared = dict(source_capsule=capsule_ref, shared_source_targets_digest=capsule['targets_digest'],
        shared_source_epochs_digest=capsule['epochs_digest'], all_own_source_references_checked=True,
        exact_full_source_set_checked=True)
    def put(phase, artifacts, trace):
        dest = out / phase; dest.mkdir(exist_ok=True)
        write_json(dest / 'preflight.json', gate()); write_json(dest / 'supervision.json', supervision())
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V197_'+phase.upper(),
            launch=record(out / 'launch.json'), preflight=record(dest / 'preflight.json'),
            artifacts=artifacts, trace=trace, resources=resource_report(),
            pending_source_hash=work.phase_info(phase)[0] in ('names', 'addresses', 'score'),
            initial100_emitted=False, old_candidates_opened=False, ready_for_fold1=False, **work.original.FLAGS))
    for phase in work.generation_phases():
        kind, batch, part = work.phase_info(phase); dest = out / phase; dest.mkdir(exist_ok=True)
        own = controls()[2*batch:2*batch+2]
        if kind in ('names', 'addresses'):
            rows, derived = [], []
            for row in own:
                a = {}
                for side in ('coverage', 'exact'):
                    file = dest / f'{row["position"]}_{side}.json'; write_json(file, {'unit_proof_only': row['position']})
                    a[side] = record(file); derived.extend(work.original.epochs([a[side]]))
                rows.append(dict(position=row['position'], query_id=row['query_id'], fold=row['fold'], artifacts=a))
            write_json(dest / 'proofs.json', dict(rows=rows, population=128, truth_opened=False, recall_measured=False,
                initial100_emitted=False, derived_epochs=derived))
            write_json(dest / 'targets.json', list(refs.values()))
            write_json(dest / 'epochs.json', epochs)
            a = {k: record(dest / (k+'.json')) for k in ('proofs', 'targets', 'epochs')}
            trace = shared if batch else {'source_capsule': None}
        else:
            a = {}
            for row in own:
                p = dict(signature=row['signature'], span=spans[part], candidates=[],
                    partition_only_not_initial_output=True, same_scores_never_rescored=True,
                    stats=dict(sites_completed=spans[part]['stop']-spans[part]['start'], address_facts=spans[part]['facts'],
                        every_site_scored_once=True, fixed_reservoir=None, score_after_fixed_pool=False, initial_cap=100))
                file = dest / f'part{row["position"]}.json'; write_json(file, p); a[f'part{row["position"]}'] = record(file)
            if phase == 'score000-000': a['source_capsule'] = capsule_ref
            trace = dict(rows=own, span=spans[part], **shared)
        put(phase, a, trace)
    return out, cfg, put


def complete_fixture(out, cfg, put):
    (out / 'plan').mkdir(); a, t = work.make_plan(out, cfg, lambda x: None); put('plan', a, t)
    unused, plan = work.read_plan(out)
    for i in range(len(plan['chunks'])):
        (out / f'audit{i:04d}').mkdir()
        a, t = work.audit_chunk(out, i, lambda x: None)  # Every tiny source AND derived byte REALLY hashed.
        put(f'audit{i:04d}', a, t)
    unused, unused_p, audit_refs = work.receipts(out)
    result = outputs()
    for batch in range(64):
        dest = out / f'merge{batch:03d}'; dest.mkdir()
        write_json(dest / 'draft.json', dict(rows=result[2*batch:2*batch+2], population=128,
            truth_opened=False, recall_measured=False, initial100_emitted=False))
        put(f'merge{batch:03d}', dict(draft=record(dest / 'draft.json')),
            dict(rows=plan['rows'][2*batch:2*batch+2], audits=audit_refs, plan=record(out / 'plan/plan.json'),
                producers=[plan['producers'][6*batch+2+i]['reference'] for i in range(4)]))
    return plan


def test_whole128_shared_full_audit_includes_every_external_and_owned_feature(tmp_path, monkeypatch):
    out, cfg, put = setup(tmp_path, monkeypatch); p = complete_fixture(out, cfg, put)
    rows, merges, unused, unused_p, refs = work.collect(out, cfg)
    assert len(rows) == 128 and len(merges) == 64 and len(p['producers']) == 384
    targets = hashes.references(r for chunk in p['chunks'] for r in chunk)
    assert sum('unit_proof' in verify(r).read_text() for r in targets) == 512
    assert p['files'] == len(targets) and len(refs) == len(p['chunks'])
    assert not p['initial100_emitted'] and not p['truth_opened']


@pytest.mark.parametrize('phase', ['names064', 'addresses128', 'score001-004', 'score-1-001', 'merge064', 'audit1', 'recall'])
def test_no_partial_ad_hoc_population_or_out_of_range_phase(phase):
    with pytest.raises(ValueError): work.phase_info(phase)


@pytest.mark.parametrize('change', ['missing_last', 'mixed_query', 'partial_source_span', 'no_parent', 'parent_nonzero', 'rss', 'truth', 'stop'])
def test_partial_or_contaminated_actual_producer_stops_plan(tmp_path, monkeypatch, change):
    out, cfg, put = setup(tmp_path, monkeypatch)
    phase = 'score063-003'; dest = out / phase
    path = dest / 'manifest.json'; m = json.loads(path.read_text())
    if change == 'missing_last': path.unlink()
    elif change == 'mixed_query': m['trace']['rows'][1]['query_id'] = 'OTHER_QUERY'; write_json(path, m)
    elif change == 'partial_source_span': m['trace']['span']['stop'] -= 1; write_json(path, m)
    elif change == 'no_parent': (dest / 'supervision.json').unlink()
    elif change == 'parent_nonzero':
        s = supervision(); s['exit_code'] = 1; write_json(dest / 'supervision.json', s)
    elif change == 'rss': m['resources']['peak_rss_bytes'] = 2**30+1; write_json(path, m)
    elif change == 'truth': m['truth_opened'] = True; write_json(path, m)
    else: write_json(dest / 'worker_stop.json', {'status': 'STOP'})
    (out / 'plan').mkdir()
    with pytest.raises((ValueError, FileNotFoundError)): work.make_plan(out, cfg, lambda x: None)


def test_missing_owned_feature_from_self_rehashed_plan_is_rejected(tmp_path, monkeypatch):
    out, cfg, put = setup(tmp_path, monkeypatch); p = complete_fixture(out, cfg, put)
    path = out / 'plan/plan.json'; plan = json.loads(path.read_text())
    owned = next(r for r in plan['chunks'][0] if '127_coverage' in r['path'])
    plan['chunks'][0].remove(owned); plan['files'] -= 1
    plan['full_bytes'] -= verify(owned).stat().st_size
    plan['signature'] = whole.digest(dict(producers=plan['producers'], rows=plan['rows'],
        targets=hashes.targets_digest(r for chunk in plan['chunks'] for r in chunk)))
    write_json(path, plan)
    mpath = out / 'plan/manifest.json'; m = json.loads(mpath.read_text()); m['artifacts']['plan'] = record(path); write_json(mpath, m)
    # Old actual FULL receipts do not validate a new self-rehashed/smaller target plan.
    with pytest.raises(ValueError): work.receipts(out)


def test_real_frozen_config_does_not_follow_labels_or_old_predictions():
    c = work.config()
    assert c['queries'] == 128 and c['batch_width'] == 2
    assert len(work.generation_phases()) == len(set(work.generation_phases())) == 384
    assert not any(k in c for k in ('names_root', 'addresses_root', 'truth', 'candidates', 'positions'))


def ledger_fixture(tmp_path, monkeypatch):
    out, cfg, put = setup(tmp_path, monkeypatch); complete_fixture(out, cfg, put)
    # Explicitly SIMULATED original-source authority/process evidence only.
    # Full tiny source/proof bytes and their hash/epoch checks above are real.
    monkeypatch.setattr(work.addresses, 'address_authority', lambda c: (None, None, {'epochs': range(93507)}))
    monkeypatch.setattr(work.addresses, 'check_address_authority', lambda a: None)
    (out / 'ledger').mkdir()
    a, t = work.ledger(out, cfg, lambda x: None); put('ledger', a, t)
    return out, cfg


def test_whole128_only_after_all_reductions_and_actual_postmerge_ledger(tmp_path, monkeypatch):
    out, cfg = ledger_fixture(tmp_path, monkeypatch)
    checked, plan = work.admission(out)
    p = json.loads(verify(checked['artifacts']['draft']).read_text())
    assert len(p['rows']) == 128 and p['full_query_population'] and not p['truth_opened']
    assert [r['query_id'] for r in p['rows']] == [r['query_id'] for r in controls()]
    assert len(plan['producers']) == 384 and not checked['initial100_emitted']


@pytest.mark.parametrize('change', ['ledger_missing', 'parent_missing', 'ledger_truth', 'partial_output',
    'cap101', 'mixed_recipe', 'source_epoch', 'part_after_merge', 'audit_after_merge'])
def test_final_admission_refuses_partial_source_query_output_or_ledger(tmp_path, monkeypatch, change):
    out, cfg = ledger_fixture(tmp_path, monkeypatch)
    if change == 'ledger_missing': (out / 'ledger/manifest.json').unlink()
    elif change == 'parent_missing': (out / 'ledger/supervision.json').unlink()
    elif change == 'source_epoch': write_json(tmp_path / 'facts.json', {'source_drift': True})
    elif change == 'part_after_merge': write_json(out / 'score063-003/part127.json', {'changed': True})
    elif change == 'audit_after_merge': (out / 'audit0000/manifest.json').unlink()
    else:
        path = out / 'ledger/draft_initial100.json'; p = json.loads(path.read_text())
        if change == 'ledger_truth': p['truth_opened'] = True
        elif change == 'partial_output': p['rows'].pop()
        elif change == 'cap101': p['rows'][-1]['candidates'] = [{'siret': f'{i:014d}'} for i in range(101)]
        else: p['rows'][-1]['trace']['signature'] = p['rows'][0]['trace']['signature']
        write_json(path, p)
        # Do not rely solely on stale outer checksums: rebind a SIMULATED
        # ledger to the changed output, and exercise the actual output guard.
        receipt_path = out / 'ledger/receipt.json'; r = json.loads(receipt_path.read_text())
        r['draft'] = record(path); write_json(receipt_path, r)
        manifest = out / 'ledger/manifest.json'; m = json.loads(manifest.read_text())
        m['artifacts'] = dict(draft=record(path), receipt=record(receipt_path)); write_json(manifest, m)
    with pytest.raises((ValueError, FileNotFoundError)): work.admission(out)


@pytest.mark.parametrize('change', ['rss', 'wall', 'partial', 'bool_count', 'batch_width', 'physical_gap', 'truth_key'])
def test_no_resource_budget_or_full_population_or_world_relaxation(tmp_path, monkeypatch, change):
    c = json.loads(work.CONFIG.read_text())
    if change == 'rss': c['resources']['max_rss_bytes'] += 1
    elif change == 'wall': c['resources']['max_seconds'] += 1
    elif change == 'partial': c['queries'] = 126
    elif change == 'bool_count': c['queries'] = True
    elif change == 'batch_width': c['batch_width'] = 4
    elif change == 'physical_gap': c['expected_spans'][-1]['start'] += 1
    else: c['truth_key'] = 'NEVER_FOLLOW'
    path = tmp_path / 'config.json'; write_json(path, c); monkeypatch.setattr(work, 'CONFIG', path)
    with pytest.raises(ValueError): work.config()


def test_last_pair_uses_own_proofs_and_fold_not_first_pair(tmp_path):
    import numpy as np
    pop = population(); rows = controls()[126:128]
    geo = {r['query_id']: {} for r in rows}; p = dict(names=[], addresses=[])
    for control in rows:
        raw, unused = pop.select(control['position'])
        file = tmp_path / f'own{control["position"]}.json'; write_json(file, {'only_control_fixture': control['position']})
        r = dict(position=control['position'], query_id=control['query_id'], fold=control['fold'],
                 artifacts=dict(coverage=record(file), exact=record(file)))
        p['names'].append(deepcopy(r)); p['addresses'].append(dict(r, query=work.addresses.query_control(raw, geo)))
    s = dict(catalog={'fixture_catalog': True}, structured={'fixture_attributes': True})
    old = dict(models={'fixture_models': True}, geo_queries={'fixture_geo': True}, query_vectors={'fixture_vectors': True})
    models = {str(f): {'weights': (np.linspace(.01, 3., 30)*f).tolist()} for f in (2, 3, 4)}
    actual = work.row_controls(p, 63, pop, old, models, geo, s)
    assert [r['position'] for r in actual] == [126, 127] and actual[0]['signature'] != actual[1]['signature']
    p['addresses'][1]['query_id'] = p['addresses'][0]['query_id']
    with pytest.raises(ValueError, match='OWN query'): work.row_controls(p, 63, pop, old, models, geo, s)


def test_all_own_queries_scalar_oracle_not_only_first_pair(tmp_path, monkeypatch):
    import numpy as np
    from src.xgb_matcher.retrieval_certified_address_facts_v181 import CertifiedAddressFacts
    from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
    from src.xgb_matcher.retrieval_initial_collector_v146 import search
    from test_retrieval_certified_address_facts_v181 import fixture as native, snapshot
    from test_retrieval_initial_collector_v146 import Attributes
    from test_retrieval_staged_initial_v183 import Names, keys_catalog
    index, unused, unused_root = native(tmp_path, monkeypatch); source = CertifiedAddressFacts(index)
    snap = snapshot(tmp_path); k = work.original
    cat = keys_catalog(np.arange(100001, 100001+source.sites, dtype=np.uint64)[::-1])
    flat, unused = k.flatten_names(Names(source.sites), tmp_path / 'flat', lambda: None, 3)
    try:
        for position in (0, 1, 64, 65, 126, 127):
            q = Query(postcode='75001' if position%2 == 0 else '69001', number='7' if position%2 == 0 else '9')
            w = np.linspace(.01, 3., 30)*(1+position%3)
            bound = source.bind(snap, q, 1)
            expected, unused = search(q, w, cat, flat, Attributes(source.sites), bound, chunk=2); bound.finish()
            signature = k.recipe(q, w, {'tiny_own_control': position}, 2+position%3)
            parts = [k.partition_search(q, w, cat, flat, Attributes(source.sites),
                k.PartitionFacts(source, snap, q, span, 1), signature, chunk=2) for span in k.spans(source, 3)]
            actual, trace = k.merge_partitions(cat, source.facts, parts)
            assert actual == expected and trace['signature'] == signature and source.active is None
        source.verify_after()
    finally: flat.close(); snap.close(); source.close()
