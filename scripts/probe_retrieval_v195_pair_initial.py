"""Two own-query initial100s over the complete national corpus; closed GT."""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import psutil

from scripts import probe_retrieval_v187_staged_initial as original
from scripts import probe_retrieval_v193_names_pair as names
from scripts import probe_retrieval_v194_addresses_pair as addresses
from src.xgb_matcher import retrieval_pair_audit_v195 as paired
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher import retrieval_query_population_v192 as queries
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v195_pair_initial.json'
STATUS = 'SEALED_V195_TWO_COMPLETE_NATIONAL_INITIAL100_COST_ONLY'
SPANS = [dict(start=0, stop=11141120, facts=25049175),
         dict(start=11141120, stop=22282240, facts=25956701),
         dict(start=22282240, stop=33357824, facts=25357414),
         dict(start=33357824, stop=44403034, facts=22441582)]
PHASES = ('score000', 'score001', 'score002', 'score003',
          'audit-facts', 'audit-postings', 'audit-context', 'merge', 'ledger')


def config():
    c = json.loads(CONFIG.read_text())
    s = original.config()
    if (c['resources'] != LIMITS or c['quiet_seconds'] != 30 or c['positions'] != [0, 1]
            or any(type(i) is not int for i in c['positions'])
            or type(c['queries']) is not int or c['queries'] != 128
            or type(c['partitions']) is not int or c['partitions'] != 4
            or c['expected_spans'] != SPANS
            or any(c[k] != s[k] for k in ('sites', 'aliases', 'facts'))
            or json.loads(verify(c['base_config']).read_text()) != s):
        raise ValueError('Same complete world/full128/physical pair and unchanged budget required')
    return c


def inputs(c):
    s, population, geo = addresses.inputs(c)
    old, unused, models, geo = original.inputs(s)
    return s, population, old, models, geo


def prefix(c, deep=False):
    rows, refs, epochs = {}, [], []
    for kind, module in (('names', names), ('addresses', addresses)):
        root_ref = c[kind+'_root']
        path = verify(root_ref); root = path.parent
        m = json.loads(path.read_text())
        launch = json.loads(verify(m['launch']).read_text())
        if (m['status'] != module.STATUS or m['queries'] != 2 or m['population'] != 128
                or m['initial100_emitted'] is not False or m['full_query_population'] is not False
                or m['complete_retrieval_index'] is not False or m['slo_validated'] is not False
                or m['ready_for_fold1'] is not False or m['old_candidates_opened'] is not False
                or any(m.get(k) is not v for k, v in original.FLAGS.items())
                or launch['config']['base_config'] != c['base_config']
                or launch['config']['population_config'] != c['population_config']
                or launch['config']['positions'] != c['positions']):
            raise ValueError('Actual complete sealed two-query proof prefix, no candidates/GT required')
        if deep: module.admission(root)
        own, audit = module.stage(root, kind), module.stage(root, 'audit')
        if (m[kind] != record(root / kind / 'manifest.json')
                or m['audit'] != record(root / 'audit/manifest.json')
                or m['outputs'] != own['artifacts']['proofs']):
            raise ValueError('Exact final prefix and actual own producer/auditor required')
        p = json.loads(verify(own['artifacts']['proofs']).read_text())
        if ([r['position'] for r in p['rows']] != [0, 1] or p['population'] != 128
                or p['truth_opened'] is not False or p['recall_measured'] is not False
                or p['initial100_emitted'] is not False):
            raise ValueError('Same two complete own proof rows inside whole128 required')
        rows[kind] = p['rows']
        refs.extend([root_ref, m['launch'], m[kind], m['audit'],
                     *own['artifacts'].values(), *audit['artifacts'].values()])
        epochs.append(own['artifacts']['epochs'])
        # The original full target lists are pinned, never inferred from only
        # touched files or a winning candidate; loaded once after scoring.
        refs.append(own['artifacts']['targets'])
    for a, b in zip(rows['names'], rows['addresses']):
        if any(a[k] != b[k] for k in ('position', 'query_id', 'fold')):
            raise ValueError('No name/address proof exchange between queries permitted')
    return rows, hashes.references(refs), epochs


def controls(c, row_proofs, population, old, models, geo):
    result = []
    for position in c['positions']:
        row, unused = population.select(position)
        n, a = row_proofs['names'][position], row_proofs['addresses'][position]
        if (any(r['query_id'] != row['query_id'] or r['fold'] != row['fold'] for r in (n, a))
                or a['query'] != addresses.query_control(row, geo)):
            raise ValueError('Own complete query proofs and query-only BAN geography required')
        q = original.query(row, geo)
        sig = original.recipe(q, models[str(row['fold'])]['weights'],
            dict(names=n['artifacts'], addresses=a['artifacts'], catalog=original.config()['catalog'],
                 structured=original.config()['structured'], model=old['models'],
                 geography=old['geo_queries'], query_vectors=old['query_vectors']), row['fold'])
        result.append(dict(position=position, query_id=row['query_id'], fold=row['fold'], signature=sig))
    return paired.records(result)


def implementation():
    refs = {**original.implementation(), **names.implementation(), **addresses.implementation()}
    for p in (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v195_pair_initial_contract.md',
              ROOT / 'src/xgb_matcher/retrieval_pair_audit_v195.py'):
        refs[str(p)] = record(p)
    return refs


def load(output, dest):
    launch = json.loads((output / 'launch.json').read_text())
    c = config()
    if (launch['config'] != c or any(launch.get(k) is not v for k, v in original.FLAGS.items())
            or launch['old_candidates_opened'] is not False or (output / 'status.json').exists()
            or any(output.glob('*/worker_stop.json')) or any(output.glob('*/supervisor_stop.json'))
            or not output.resolve().is_relative_to(Path(c['output_root']).resolve())):
        raise ValueError('Fresh exact195 source-bound scope, no STOP retry')
    for ref in launch['implementation'].values(): verify(ref)
    gate = json.loads((dest / 'preflight.json').read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual own authorized195 pressure preflight required')
    return launch


def stage(output, phase):
    if phase not in PHASES: raise ValueError('Explicit whole physical195 phase required')
    dest = output / phase
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP195_STAGE_DO_NOT_USE')
    parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text())
    validate_report(m['resources'])
    if (m['status'] != 'QUARANTINED_V195_'+phase.upper() or m['launch'] != record(output / 'launch.json')
            or m['pending_source_hash'] is not phase.startswith('score')
            or m['initial100_emitted'] is not False or m['old_candidates_opened'] is not False
            or any(m.get(k) is not v for k, v in original.FLAGS.items())
            or verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve()):
        raise ValueError('Actual own supervised quarantined195 phase required')
    load(output, dest)
    for ref in m['artifacts'].values(): verify(ref)
    m['_reference'] = record(dest / 'manifest.json')
    return m


def shared_targets(s, c, old, models, keys, attrs, index, facts, proof_rows, prefix_refs, prefix_epochs):
    base = dict(facts=list(facts.used.values()),
        postings=[*index.references, *index.authorities.values(), index.launch['plan'], index.launch['epochs'],
                  *index.launch['implementation'].values()],
        context=[*attrs.references, keys.reference, keys.metadata['launch'], *keys.keys.references,
                 *[s[k] for k in ('names', 'joins', 'catalog', 'structured', 'addresses', 'query_only_launch')],
                 c['base_config'], c['population_config'], old['models'], old['model_manifest'],
                 old['geo_queries'], old['query_vectors'], old['activity_inputs']['descriptions'],
                 *[m['decoder_parameter'] for m in models.values()], *prefix_refs,
                 *[r for kind in proof_rows for row in proof_rows[kind] for r in row['artifacts'].values()]])
    captured = []
    for ref in prefix_epochs:
        e = hashes.canonical_epochs(json.loads(verify(ref).read_text()))
        captured = hashes.canonical_epochs([*captured, *e])
        base['context'].extend(item['reference'] for item in e)
        del e; gc.collect()
    clean = paired.groups(base)
    captured = hashes.canonical_epochs([*captured, *original.epochs(
        r for group in hashes.GROUPS for r in clean[group])])
    return clean, captured


def score(output, phase, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    proofs, prefix_refs, prefix_epochs = prefix(c)
    rows = controls(c, proofs, population, old, models, geo)
    gc.collect()
    mark(dict(checkpoint='EARLY_CATALOG_BEFORE_LARGE_SOURCE_METADATA'))
    keys = original.RotatedKeys(s['catalog']); owned.append(keys)
    catalog = SimpleNamespace(sites=keys.sites, keys=keys)
    attrs = original.PagedStructuredSites(s['structured']); owned.append(attrs)
    attributes = original.AppendedAttributes(attrs, keys.keys)
    mark(dict(checkpoint='LOAD_COMPLETE_SHARED_FACTS'))
    index = original.PagedAddresses(s['addresses']); owned.append(index)
    facts = original.CertifiedAddressFacts(index); owned.append(facts)
    if (keys.sites != c['sites'] or facts.catalog_reference != s['catalog']
            or any(getattr(facts, k) != c[k] for k in ('sites', 'aliases', 'facts'))
            or original.spans(facts, c['partitions']) != c['expected_spans']):
        raise ValueError('ALL unchanged official sites/facts and whole physical4 boundaries required')
    span = c['expected_spans'][int(phase.removeprefix('score'))]
    artifacts, timings = {}, []
    for control in rows:
        position = control['position']
        row, unused = population.select(position)
        q = original.query(row, geo)
        mark(dict(checkpoint='OPEN_OWN_COMPLETE_QUERY_PROOFS', position=position))
        flat = original.reopen_names(proofs['names'][position], s); owned.append(flat)
        snap = original.reopen_address(proofs['addresses'][position], s); owned.append(snap)
        grades = queries.activity(population, old, models, attrs, position)
        bound = original.PartitionFacts(facts, snap, q, span, s['chunk'])
        mark(dict(checkpoint='SCORE_ALL_PHYSICAL_SITES_AND_FACTS', position=position))
        began = time.monotonic()
        part = original.partition_search(q, models[str(row['fold'])]['weights'], catalog, flat, attributes,
            bound, control['signature'], grades, s['chunk'], s['max_partition_score_seconds'])
        paired.part_control(part, control, span)
        timings.append(dict(position=position, score_seconds=time.monotonic()-began, **part['stats']))
        write_json(output / phase / ('part'+str(position)+'.json'), part)
        artifacts['part'+str(position)] = record(output / phase / ('part'+str(position)+'.json'))
        flat.verify_after(); snap.verify_after()
        flat.close(); snap.close(); owned.remove(flat); owned.remove(snap)
        del flat, snap, bound, part, grades
        gc.collect()
        mark(dict(checkpoint='OWN_QUERY_COMPLETE_NO_OTHER_QUERY_FEATURE_REUSE', position=position))
    mark(dict(checkpoint='CAPTURE_SHARED_COMPLETE_FULL_HASH_TARGETS_ONCE'))
    groups, captured = shared_targets(s, c, old, models, keys, attrs, index, facts,
                                      proofs, prefix_refs, prefix_epochs)
    p = paired.packet(groups, captured, rows, span, c['sites'], c['facts'])
    del groups, captured; gc.collect()
    facts.assert_live(); index.check_live(); attrs.check_live(); keys.verify_after()
    write_json(output / phase / 'packet.json', p)
    artifacts['packet'] = record(output / phase / 'packet.json')
    hashes.check_epochs(p['source_epochs'])
    return artifacts, dict(rows=rows, span=span, timings=timings, targets=sum(len(v) for v in p['groups'].values()))


def merge(output, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    proofs, unused, unused_epochs = prefix(c)
    expected = controls(c, proofs, population, old, models, geo)
    gc.collect()
    mark(dict(checkpoint='EARLY_CATALOG_BEFORE_COMPLETE_AUDIT_METADATA'))
    keys = original.RotatedKeys(s['catalog']); owned.append(keys)
    value = paired.audited_plan(output, c, stage)
    if value['rows'] != expected: raise ValueError('Every OWN query recipe and full hash before reduction required')
    results = []
    for row in value['rows']:
        parts = []
        for i in range(c['partitions']):
            m = stage(output, f'score{i:03d}')
            parts.append(json.loads(verify(m['artifacts']['part'+str(row['position'])]).read_text()))
        candidates, trace = original.merge_partitions(SimpleNamespace(sites=keys.sites, keys=keys), c['facts'], parts)
        if trace['signature'] != row['signature']: raise ValueError('Own query score unchanged in exact reduction')
        results.append(dict(position=row['position'], query_id=row['query_id'], fold=row['fold'],
                            candidates=candidates, trace=trace))
        del parts; gc.collect()
    keys.verify_after(); keys.close(); owned.remove(keys)
    hashes.check_epochs(value['source_epochs'])
    write_json(output / 'merge/draft_initial100.json', dict(rows=results, population=128,
        truth_opened=False, recall_measured=False, full_query_population=False, ready_for_fold1=False))
    write_json(output / 'merge/epochs.json', value['source_epochs'])
    return (dict(draft=record(output / 'merge/draft_initial100.json'), epochs=record(output / 'merge/epochs.json')),
            dict(rows=expected, signature=value['signature'], queries=2, initial_cap=100,
                 producers=value['producers'], packets=value['packets'], audits=value['audits']))


def ledger(output, c, mark):
    m = stage(output, 'merge')
    captured = hashes.canonical_epochs(json.loads(verify(m['artifacts']['epochs']).read_text()))
    hashes.check_epochs(captured)
    mark(dict(checkpoint='COMPLETE_ORIGINAL_SOURCE_LEDGER_AFTER_ACTUAL_TWO_QUERY_MERGE'))
    authority = addresses.address_authority(c)
    if len(authority[2]['epochs']) != 93507: raise ValueError('Whole original93507 ledger after BOTH merges required')
    addresses.check_address_authority(authority)
    hashes.check_epochs(captured); verify(m['artifacts']['draft'])
    return dict(merge=record(output / 'merge/manifest.json'), draft=m['artifacts']['draft'],
        epochs=m['artifacts']['epochs'], rows=m['trace']['rows'], signature=m['trace']['signature'],
        source_epoch_count=93507, sites=c['sites'], facts=c['facts'],
        initial100_emitted=False, **original.FLAGS)


def admission(output):
    c = config()
    m, l = stage(output, 'merge'), stage(output, 'ledger')
    r = json.loads(verify(l['artifacts']['receipt']).read_text())
    if (r['merge'] != record(output / 'merge/manifest.json') or r['draft'] != m['artifacts']['draft']
            or r['epochs'] != m['artifacts']['epochs'] or r['rows'] != m['trace']['rows']
            or r['signature'] != m['trace']['signature'] or r['source_epoch_count'] != 93507
            or any(r[k] != c[k] for k in ('sites', 'facts')) or r['initial100_emitted'] is not False
            or any(r.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Independent complete ledger after BOTH exact national reductions required')
    for i in range(c['partitions']):
        produced = stage(output, f'score{i:03d}')
        if (produced['_reference'] != m['trace']['producers'][i]
                or produced['artifacts']['packet'] != m['trace']['packets'][i]):
            raise ValueError('Every exact original producer/packet still valid after merge/ledger')
    for i, group in enumerate(hashes.GROUPS):
        audited = stage(output, 'audit-'+group)
        if audited['_reference'] != m['trace']['audits'][i]:
            raise ValueError('Every exact successful full audit remains bound to admitted output')
    p = json.loads(verify(r['draft']).read_text())
    if (p['population'] != 128 or p['truth_opened'] is not False or p['recall_measured'] is not False
            or p['full_query_population'] is not False or p['ready_for_fold1'] is not False
            or [row['position'] for row in p['rows']] != c['positions']):
        raise ValueError('Whole own physical pair only, never full128 or recall claim')
    for row, control in zip(p['rows'], paired.records(r['rows'])):
        t = row['trace']
        if (any(row[k] != control[k] for k in ('position', 'query_id', 'fold'))
                or t['signature'] != control['signature'] or t['sites_completed'] != c['sites']
                or t['address_facts'] != c['facts'] or t['physical_partitions'] != c['partitions']
                or t['initial_cap'] != 100 or t['fixed_reservoir'] is not None
                or t['score_after_fixed_pool'] is not False or t['every_site_scored_once'] is not True
                or t['same_original_score_never_rescored'] is not True
                or len(row['candidates']) > 100 or len({v['siret'] for v in row['candidates']}) != len(row['candidates'])):
            raise ValueError('Every national leaf/fact, unchanged score, absolute100 cap per query required')
    hashes.check_epochs(json.loads(verify(r['epochs']).read_text()))
    prefix(c)
    return m, l


def worker(output, phase):
    dest = output / phase
    if phase not in PHASES or any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Explicit fresh immutable195 worker required')
    owned, budget = [], None
    began = time.monotonic()
    def mark(event):
        print(json.dumps(dict(event, elapsed_seconds=time.monotonic()-began,
            rss_bytes=psutil.Process().memory_info().rss,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest); c = launch['config']
            if phase.startswith('score'):
                artifacts, trace = score(output, phase, c, mark, owned)
            elif phase.startswith('audit-'):
                value, producers, packets = paired.score_plan(output, c, stage)
                group = phase.removeprefix('audit-')
                receipt = hashes.audit_group(value, group, mark)
                receipt.update(score_stages=producers, audit_packets=packets)
                write_json(dest / 'receipt.json', receipt)
                artifacts = dict(receipt=record(dest / 'receipt.json'))
                trace = dict(group=group, files=receipt['files'], full_bytes_hashed=receipt['full_bytes_hashed'])
            elif phase == 'merge': artifacts, trace = merge(output, c, mark, owned)
            else:
                receipt = ledger(output, c, mark)
                write_json(dest / 'receipt.json', receipt)
                artifacts = dict(receipt=record(dest / 'receipt.json'))
                trace = dict(rows=receipt['rows'], signature=receipt['signature'], source_epoch_count=93507)
            load(output, dest)
            for owner in reversed(owned): owner.close()
            owned.clear()
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V195_'+phase.upper(),
            launch=record(output / 'launch.json'), preflight=record(dest / 'preflight.json'),
            artifacts=artifacts, trace=trace, resources=budget.report(), pending_source_hash=phase.startswith('score'),
            initial100_emitted=False, old_candidates_opened=False, ready_for_fold1=False, **original.FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            old_candidates_opened=False, **original.FLAGS))
        raise
    finally:
        for owner in reversed(owned): owner.close()


def run(output):
    began = time.monotonic()
    c = config(); inputs(c); prefix(c, deep=True)
    controller_seconds = time.monotonic()-began
    gc.collect()
    if output.exists() or not output.resolve().is_relative_to(Path(c['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only immutable195 national pair root required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=c, implementation=implementation(),
                                          old_candidates_opened=False, **original.FLAGS))
    try:
        wall = 0
        for phase in PHASES:
            gate = quiet(c['quiet_seconds'], 50, 55)
            dest = output / phase; dest.mkdir()
            write_json(dest / 'preflight.json', gate)
            with (dest / 'worker.log').open('x') as log:
                supervision = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v195_pair_initial',
                    'worker', '--output', str(output), '--phase', phase], log,
                    {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
            write_json(dest / 'supervision.json', supervision)
            m = stage(output, phase); wall += m['resources']['wall_seconds']
            print(json.dumps(dict(completed=phase, resources=m['resources'], trace=m['trace'])), flush=True)
        merged, checked = admission(output)
        prefix_cost = sum(json.loads(verify(c[k+'_root']).read_text())[metric] for k, metric in
            (('names', 'all_load_query_flatten_rotation_and_full_hash_worker_seconds'),
             ('addresses', 'all_load_query_and_full_hash_worker_seconds')))
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            merge=record(output / 'merge/manifest.json'), ledger=record(output / 'ledger/manifest.json'),
            outputs=merged['artifacts']['draft'], queries=2, population=128, initial100_emitted=True,
            complete_national_source=True, full_query_population=False,
            scoring_audits_reduction_and_ledger_worker_seconds=wall,
            prior_complete_name_and_address_proof_worker_seconds=prefix_cost,
            complete_retrieval_worker_seconds_per_query=(wall+prefix_cost)/2,
            controller_prelaunch_source_validation_seconds=controller_seconds,
            elapsed_current_run_including_quiet_parent_seconds=time.monotonic()-began,
            disk_cache_state='UNCONTROLLED_NOT_COLD_DISK', slo_validated=False, ready_for_fold1=False,
            downstream_models_changed=False, old_candidates_opened=False, **original.FLAGS))
        print(STATUS, flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            old_candidates_opened=False, **original.FLAGS))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=PHASES)
    a = p.parse_args()
    if a.mode == 'worker' and a.phase is None: p.error('Explicit complete physical phase required')
    run(a.output) if a.mode == 'run' else worker(a.output, a.phase)
