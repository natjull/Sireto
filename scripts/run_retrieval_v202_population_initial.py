"""ALL128 initial100 outputs; shared complete after-hash, closed labels."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from types import SimpleNamespace

import psutil

from scripts import probe_retrieval_v187_staged_initial as original
from scripts import probe_retrieval_v193_names_pair as names
from scripts import probe_retrieval_v194_addresses_pair as addresses
from scripts import probe_retrieval_v196_pair_initial as integration
from src.xgb_matcher import retrieval_population_audit_v197 as whole
from src.xgb_matcher import retrieval_postscore_audit_v187 as hashes
from src.xgb_matcher import retrieval_query_population_v192 as queries
from src.xgb_matcher.retrieval_deferred_addresses_v196 import DeferredHashAddresses
from src.xgb_matcher.retrieval_bounded_name_relations_v201 import BoundedRelationNames, MAX_COUNTER_RAM_BYTES, MAX_RELATION_CHUNK
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v202_population_initial.json'
STATUS = 'SEALED_V202_ALL128_COMPLETE_NATIONAL_INITIAL100_NO_LABELS'


def config():
    c = json.loads(CONFIG.read_text()); s = original.config()
    if (set(c) != {'base_config', 'population_config', 'queries', 'batch_width', 'resources',
                    'quiet_seconds', 'max_audit_bytes', 'max_audit_files', 'output_root',
                    'sites', 'aliases', 'facts', 'partitions', 'expected_spans',
                    'max_counter_ram_bytes', 'max_relation_chunk', 'execution_order'}
            or c['resources'] != LIMITS or c['quiet_seconds'] != 30
            or c['max_counter_ram_bytes'] != MAX_COUNTER_RAM_BYTES
            or c['max_relation_chunk'] != MAX_RELATION_CHUNK
            or c['execution_order'] != 'NAMES_BEFORE_REMAINING_SCORES_TRUTH_FREE'
            or s['max_proof_query_seconds'] != 20
            or type(c['queries']) is not int or c['queries'] != 128
            or type(c['batch_width']) is not int or c['batch_width'] != 2
            or type(c['partitions']) is not int or c['partitions'] != 4
            or c['expected_spans'] != integration.SPANS
            or c['max_audit_bytes'] != whole.MAX_AUDIT_BYTES or c['max_audit_files'] != whole.MAX_AUDIT_FILES
            or any(c[k] != s[k] for k in ('sites', 'aliases', 'facts'))
            or json.loads(verify(c['base_config']).read_text()) != s):
        raise ValueError('ALL128 frozen population, whole source and unchanged resource policy required')
    return c


def inputs(c):
    return integration.inputs(c)  # All128 typed vectors/folds/heads validated before any selection.


def generation_phases():
    return tuple(p for b in range(64) for p in
        (f'names{b:03d}', f'addresses{b:03d}', *[f'score{b:03d}-{i:03d}' for i in range(4)]))


def execution_phases():
    """Same complete384 phases; finish all name proofs before remaining scores."""
    first = ('names000', 'addresses000', 'score000-000')
    order = (*first, *[f'names{b:03d}' for b in range(1, 64)],
             *[f'addresses{b:03d}' for b in range(1, 64)],
             *[f'score{b:03d}-{i:03d}' for b in range(64) for i in range(4)
               if (b, i) != (0, 0)])
    if len(order) != 384 or len(set(order)) != 384 or set(order) != set(generation_phases()):
        raise ValueError('Exact384 once-only producers; physical order never changes population')
    return order


def phase_info(phase):
    if phase in ('plan', 'ledger'): return phase, None, None
    m = re.fullmatch(r'(names|addresses|merge)(\d{3})', phase)
    if m and 0 <= int(m[2]) < 64: return m[1], int(m[2]), None
    m = re.fullmatch(r'score(\d{3})-(\d{3})', phase)
    if m and 0 <= int(m[1]) < 64 and 0 <= int(m[2]) < 4: return 'score', int(m[1]), int(m[2])
    m = re.fullmatch(r'audit(\d{4})', phase)
    if m: return 'audit', None, int(m[1])
    raise ValueError('Exact declared physical full128 phase required')


def implementation():
    refs = integration.implementation()
    for p in (Path(__file__), CONFIG, ROOT / 'docs/retrieval_v202_population_initial_contract.md',
              ROOT / 'src/xgb_matcher/retrieval_population_audit_v197.py',
              ROOT / 'src/xgb_matcher/retrieval_ram_name_counters_v198.py',
              ROOT / 'src/xgb_matcher/retrieval_bounded_name_relations_v201.py'):
        refs[str(p)] = record(p)
    return refs


def load(output, dest=None, *, full=True):
    launch = json.loads((output / 'launch.json').read_text())
    if (launch['config'] != config() or any(launch.get(k) is not v for k, v in original.FLAGS.items())
            or launch['old_candidates_opened'] is not False or (output / 'status.json').exists()
            or any(output.glob('*/worker_stop.json')) or any(output.glob('*/supervisor_stop.json'))
            or not output.resolve().is_relative_to(Path(launch['config']['output_root']).resolve())):
        raise ValueError('Exact immutable202 closed-label launch, no STOP retry')
    if full:
        for ref in launch['implementation'].values(): verify(ref)
    if dest is not None:
        gate = json.loads((dest / 'preflight.json').read_text())
        if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
            raise ValueError('Actual own authorized pressure preflight required')
    return launch


def stage(output, phase):
    kind, unused, unused_i = phase_info(phase); dest = output / phase
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP202_STAGE_DO_NOT_USE')
    parent_exit(dest)
    m = json.loads((dest / 'manifest.json').read_text()); validate_report(m['resources'])
    if (m['status'] != 'QUARANTINED_V202_'+phase.upper() or m['launch'] != record(output / 'launch.json')
            or m['initial100_emitted'] is not False or m['old_candidates_opened'] is not False
            or m['pending_source_hash'] is not (kind in ('names', 'addresses', 'score'))
            or any(m.get(k) is not v for k, v in original.FLAGS.items())
            or verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve()):
        raise ValueError('Actual own successful quarantined producer, never partial initial100')
    load(output, dest, full=False)  # Common implementation FULL-checked at each worker entry/exit and caller boundary.
    for ref in m['artifacts'].values(): verify(ref)
    m['_reference'] = record(dest / 'manifest.json')
    return m


def anchor(output):
    m = stage(output, 'score000-000')
    ref = m['artifacts']['source_capsule']
    value = whole.validate_capsule(json.loads(verify(ref).read_text()))
    if (m['trace']['source_capsule'] != ref
            or m['trace']['shared_source_targets_digest'] != value['targets_digest']
            or m['trace']['shared_source_epochs_digest'] != value['epochs_digest']
            or m['trace']['all_own_source_references_checked'] is not True
            or m['trace']['exact_full_source_set_checked'] is not True):
        raise ValueError('Actual initial complete scorer bound to exact shared-source capsule required')
    return value, ref


def bind_sources(output, phase, refs, captured, *, exact=False):
    if phase in ('names000', 'addresses000', 'score000-000'):
        return dict(source_capsule=None, original_source_epochs_retained=True)
    value, ref = anchor(output)
    control = whole.require_sources(value, refs, captured, output, exact=exact)
    return dict(source_capsule=ref, **control)


def release(owned):
    for owner in reversed(owned): owner.close()
    owned.clear(); gc.collect()


def name_proofs(output, phase, batch, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    index = original.ScopedNamePages(s['names']); owned.append(index)
    reader = BoundedRelationNames(s['joins']); owned.append(reader)
    if (reader.sites != s['sites'] or reader.metadata['sources']['catalog'] != s['catalog']
            or index.names != reader.names or index.source != reader.source):
        raise ValueError('Same complete official alias/parent/site namespace required')
    pc = json.loads(verify(c['population_config']).read_text())
    refs = [index.source, *reader.refs, c['base_config'], c['population_config'],
        pc['encoder_launch'], pc['encoded_reference'],
        *[s[k] for k in ('names', 'joins', 'catalog', 'query_only_launch')],
        old['query_vectors'], old['models'], old['model_manifest'], old['geo_queries'],
        *[m['decoder_parameter'] for m in models.values()]]
    for access in index.metadata['access']: refs.extend(access['artifacts'].values())
    refs = hashes.references(refs); captured = hashes.canonical_epochs(original.epochs(refs))
    bind_sources(output, phase, refs, captured)
    rows = []
    for position in whole.positions(batch):
        row, unused = population.select(position); hashes.check_epochs(captured)
        directory = output / phase / f'query{position}'
        mark(dict(checkpoint='COMPLETE_OWN_NAME_QUERY', position=position))
        snap, trace = reader.query(index, row['crm_name'], directory / 'scratch',
                                  s['max_proof_query_seconds'], s['lexical_chunk'], progress=mark)
        owned.append(snap); began = time.monotonic()
        def guard():
            if time.monotonic()-began > s['max_proof_query_seconds']:
                raise TimeoutError('STOP202_WHOLE_NAME_FLATTEN_COST')
        flat, gather = original.flatten_names(snap, directory / 'flat', guard, s['chunk']); owned.append(flat)
        artifacts = dict(coverage=flat.coverage.reference, exact=flat.exact.reference)
        rows.append(dict(position=position, query_id=row['query_id'], fold=row['fold'],
            artifacts=artifacts, query=trace, gather=gather, flatten_seconds=time.monotonic()-began))
        refs = hashes.references([*refs, *artifacts.values()])
        captured = hashes.canonical_epochs([*captured, *original.epochs(artifacts.values())])
        flat.verify_after(); flat.close(); owned.remove(flat)
        snap.close(remove=True); owned.remove(snap)  # Only this worker's temporary scratch; retained proofs never removed.
        index.rotate(); reader.rotate_keys(); hashes.check_epochs(captured)
        del flat, snap; gc.collect()
    release(owned)
    control = bind_sources(output, phase, refs, captured)
    return save_proofs(output / phase, rows, refs, captured, control), dict(
        positions=whole.positions(batch), sites_per_query=s['sites'], original_aliases=index.names, **control)


def address_proofs(output, phase, batch, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    index = DeferredHashAddresses(s['addresses']); owned.append(index)
    facts = original.CertifiedAddressFacts(index); owned.append(facts)
    if (any(getattr(facts, k) != c[k] for k in ('sites', 'aliases', 'facts'))
            or facts.catalog_reference != s['catalog']):
        raise ValueError('ALL unchanged original address facts/aliases/site identities required')
    pc = json.loads(verify(c['population_config']).read_text())
    refs = [*facts.used.values(), *index.references, *index.authorities.values(),
        *index.launch['implementation'].values(), index.launch['plan'], index.launch['epochs'],
        c['base_config'], c['population_config'], pc['encoder_launch'], pc['encoded_reference'],
        *[s[k] for k in ('addresses', 'catalog', 'query_only_launch')],
        old['query_vectors'], old['models'], old['model_manifest'], old['geo_queries'],
        *[m['decoder_parameter'] for m in models.values()]]
    refs = hashes.references(refs); captured = hashes.canonical_epochs(original.epochs(refs))
    bind_sources(output, phase, refs, captured)
    rows = []
    for position in whole.positions(batch):
        row, unused = population.select(position); q = original.query(row, geo)
        facts.assert_live(); hashes.check_epochs(captured)
        mark(dict(checkpoint='COMPLETE_OWN_ADDRESS_QUERY', position=position))
        snap, trace = index.snapshot(q.address, output / phase / f'query{position}/query',
                                    s['max_proof_query_seconds'], s['lexical_chunk']); owned.append(snap)
        if snap.aliases != s['aliases']: raise ValueError('Entire original address alias denominator required')
        snap.verify_after(); artifacts = dict(coverage=snap.coverage.reference, exact=snap.exact.reference)
        trace = dict(trace, spans=c['expected_spans'], sites=facts.sites, facts=facts.facts, aliases=facts.aliases)
        rows.append(dict(position=position, query_id=row['query_id'], fold=row['fold'],
            query=addresses.query_control(row, geo), artifacts=artifacts, aliases=snap.aliases, trace=trace))
        refs = hashes.references([*refs, *artifacts.values()])
        captured = hashes.canonical_epochs([*captured, *original.epochs(artifacts.values())])
        snap.close(); owned.remove(snap); facts.assert_live(); hashes.check_epochs(captured)
        del snap; gc.collect()
    release(owned)
    control = bind_sources(output, phase, refs, captured)
    return save_proofs(output / phase, rows, refs, captured, control), dict(
        positions=whole.positions(batch), sites_per_query=s['sites'], original_aliases=c['aliases'], **control)


def save_proofs(dest, rows, refs, captured, control):
    inside = [e for e in captured if Path(e['reference']['path']).resolve().is_relative_to(dest.resolve())]
    write_json(dest / 'proofs.json', dict(rows=rows, population=128, initial100_emitted=False,
        truth_opened=False, recall_measured=False, derived_epochs=inside, **control))
    write_json(dest / 'targets.json', refs); write_json(dest / 'epochs.json', captured)
    return {k: record(dest / (k+'.json')) for k in ('proofs', 'targets', 'epochs')}


def require_proof_inventory(artifacts, refs, epochs):
    """Compare canonical path/SHA, retain FULL verification and original sizes."""
    if set(artifacts) != {'coverage', 'exact'}:
        raise ValueError('Both and only original proof vectors required')
    targets = {r['path']: r for r in hashes.references(refs)}
    captured = {e['reference']['path']: e for e in hashes.canonical_epochs(epochs)}
    for original in artifacts.values():
        canonical = hashes.canonical(original)
        epoch = captured.get(canonical['path'])
        if (targets.get(canonical['path']) != canonical or epoch is None
                or epoch['reference'] != canonical
                or ('size_bytes' in original and original['size_bytes'] != epoch['fingerprint'][2])):
            raise ValueError('Every exact proof in full source inventory and original epochs required')
        hashes.check_epochs([epoch]); verify(original)


def proofs(output, batch, c, population, geo):
    result, refs, epochs = {}, [], []
    for kind in ('names', 'addresses'):
        m = stage(output, kind+f'{batch:03d}')
        p = json.loads(verify(m['artifacts']['proofs']).read_text())
        if (p['population'] != 128 or p['truth_opened'] is not False or p['recall_measured'] is not False
                or p['initial100_emitted'] is not False
                or [r['position'] for r in p['rows']] != whole.positions(batch)):
            raise ValueError('Every complete own physical pair proof, still closed-label, required')
        inventory = json.loads(verify(m['artifacts']['targets']).read_text())
        captured = json.loads(verify(m['artifacts']['epochs']).read_text())
        for r in p['rows']:
            require_proof_inventory(r['artifacts'], inventory, captured)
            row, unused = population.select(r['position'])
            if r['query_id'] != row['query_id'] or r['fold'] != row['fold'] or set(r['artifacts']) != {'coverage', 'exact'}:
                raise ValueError('Own original query and both complete proof vectors required')
            if kind == 'names':
                if (r['gather']['sites'] != c['sites'] or r['gather']['every_original_site_gathered_once'] is not True
                        or r['gather']['candidates_emitted'] is not False):
                    raise ValueError('ALL name sites once, no candidates, required')
            else:
                t = r['trace']
                if (r['query'] != addresses.query_control(row, geo) or r['aliases'] != c['aliases']
                        or any(t[k] != c[k] for k in ('sites', 'facts', 'aliases'))
                        or t['spans'] != c['expected_spans'] or t['every_original_alias_checked'] is not True
                        or t['query_snapshot_file_backed'] is not True or t['candidates_emitted'] is not False
                        or t['recall_measured'] is not False or t['slo_validated'] is not False):
                    raise ValueError('Complete original OWN CRM/BAN address proof required')
        result[kind] = p['rows']; refs.extend([m['_reference'], *m['artifacts'].values()])
        epochs.append(m['artifacts']['epochs'])
    return result, hashes.references(refs), epochs


def row_controls(proof_rows, batch, population, old, models, geo, s):
    by_kind = {k: {r['position']: r for r in values} for k, values in proof_rows.items()}
    rows = []
    for position in whole.positions(batch):
        row, unused = population.select(position); q = original.query(row, geo)
        n, a = by_kind['names'][position], by_kind['addresses'][position]
        if (any(r['query_id'] != row['query_id'] or r['fold'] != row['fold'] for r in (n, a))
                or a['query'] != addresses.query_control(row, geo)):
            raise ValueError('OWN query, head, name/address proof and BAN controls required')
        sig = original.recipe(q, models[str(row['fold'])]['weights'], dict(names=n['artifacts'],
            addresses=a['artifacts'], catalog=s['catalog'], structured=s['structured'], model=old['models'],
            geography=old['geo_queries'], query_vectors=old['query_vectors']), row['fold'])
        rows.append(dict(position=position, query_id=row['query_id'], fold=row['fold'], signature=sig))
    return whole.controls(rows, whole.positions(batch))


def score(output, phase, batch, partition, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    p, prefix_refs, prefix_epochs = proofs(output, batch, c, population, geo)
    rows = row_controls(p, batch, population, old, models, geo, s)
    by_kind = {k: {r['position']: r for r in v} for k, v in p.items()}
    keys = original.RotatedKeys(s['catalog']); owned.append(keys)
    catalog = SimpleNamespace(sites=keys.sites, keys=keys)
    attrs = original.PagedStructuredSites(s['structured']); owned.append(attrs)
    attributes = original.AppendedAttributes(attrs, keys.keys)
    index = DeferredHashAddresses(s['addresses']); owned.append(index)
    facts = original.CertifiedAddressFacts(index); owned.append(facts)
    if (keys.sites != c['sites'] or facts.catalog_reference != s['catalog']
            or any(getattr(facts, k) != c[k] for k in ('sites', 'aliases', 'facts'))
            or original.spans(facts, c['partitions']) != c['expected_spans']):
        raise ValueError('ALL unchanged national sites/facts and physical4 spans required')
    span = c['expected_spans'][partition]; artifacts, timings = {}, []
    for row in rows:
        position = row['position']; query_row, unused = population.select(position); q = original.query(query_row, geo)
        mark(dict(checkpoint='OWN_WHOLE_PARTITION_SCORE', position=position, span=span))
        flat = original.reopen_names(by_kind['names'][position], s); owned.append(flat)
        snap = original.reopen_address(by_kind['addresses'][position], s); owned.append(snap)
        grades = queries.activity(population, old, models, attrs, position)
        bound = original.PartitionFacts(facts, snap, q, span, s['chunk'])
        part = original.partition_search(q, models[str(row['fold'])]['weights'], catalog, flat, attributes,
            bound, row['signature'], grades, s['chunk'], s['max_partition_score_seconds'])
        integration.paired.part_control(part, row, span)
        timings.append(dict(position=position, **part['stats']))
        file = output / phase / f'part{position}.json'; write_json(file, part)
        artifacts[f'part{position}'] = record(file)
        flat.verify_after(); snap.verify_after(); flat.close(); snap.close()
        owned.remove(flat); owned.remove(snap)
        del flat, snap, bound, part, grades; gc.collect()
    groups, captured = integration.shared_targets(s, c, old, models, keys, attrs, index, facts,
                                                   p, prefix_refs, prefix_epochs)
    facts.assert_live(); index.check_live(); attrs.check_live(); keys.verify_after()
    release(owned)  # Release only our readonly pages BEFORE additional capsule metadata allocation.
    code_refs = list(json.loads((output / 'launch.json').read_text())['implementation'].values())
    groups['context'].extend(code_refs); groups = integration.paired.groups(groups)
    captured = hashes.canonical_epochs([*captured, *original.epochs(code_refs)])
    refs = [r for k in hashes.GROUPS for r in groups[k]]
    if phase == 'score000-000':
        value = whole.capsule(groups, captured, output)
        file = output / phase / 'source_capsule.json'; write_json(file, value)
        artifacts['source_capsule'] = record(file)
        control = dict(source_capsule=artifacts['source_capsule'],
            shared_source_targets_digest=value['targets_digest'], shared_source_epochs_digest=value['epochs_digest'],
            all_own_source_references_checked=True, exact_full_source_set_checked=True)
    else: control = bind_sources(output, phase, refs, captured, exact=True)
    hashes.check_epochs(captured)
    return artifacts, dict(rows=rows, span=span, timings=timings, **control)


def make_plan(output, c, mark):
    value, capsule_ref = anchor(output)
    refs, captured, producers, controls = list(value['references']), list(value['source_epochs']), [], []
    for phase in generation_phases():
        kind, batch, partition = phase_info(phase); m = stage(output, phase)
        producers.append(dict(phase=phase, reference=m['_reference']))
        refs.extend([m['_reference'], m['preflight'], parent_exit(output / phase), *m['artifacts'].values()])
        if kind in ('names', 'addresses'):
            p = json.loads(verify(m['artifacts']['proofs']).read_text())
            owned_refs = hashes.references(r for row in p['rows'] for r in row['artifacts'].values())
            e = hashes.canonical_epochs(p['derived_epochs'])
            if (owned_refs != hashes.references(x['reference'] for x in e)
                    or [r['position'] for r in p['rows']] != whole.positions(batch)
                    or p['truth_opened'] is not False or p['initial100_emitted'] is not False):
                raise ValueError('Every own proof and ORIGINAL creation epoch required')
            refs.extend(owned_refs); captured.extend(e)
        if phase not in ('names000', 'addresses000'):
            t = m['trace']
            if (t['source_capsule'] != capsule_ref or t['shared_source_targets_digest'] != value['targets_digest']
                    or t['shared_source_epochs_digest'] != value['epochs_digest']
                    or t['all_own_source_references_checked'] is not True):
                raise ValueError('Every producer bound to original common source hashes AND epochs')
        if kind == 'score':
            rows = whole.controls(m['trace']['rows'], whole.positions(batch))
            span = c['expected_spans'][partition]
            if m['trace']['span'] != span or m['trace']['exact_full_source_set_checked'] is not True:
                raise ValueError('Every physical span uses ALL same external source targets')
            for row in rows:
                part = json.loads(verify(m['artifacts'][f'part{row["position"]}']).read_text())
                integration.paired.part_control(part, row, span)
            if partition == 0: controls.extend(rows)
            else:
                if rows != controls[-2:]: raise ValueError('No own-query recipe exchange between physical spans')
        if len(producers) % 32 == 0: mark(dict(producers_checked=len(producers), expected=384))
    whole.controls(controls, list(range(128)))
    refs = hashes.references(refs)
    # Original query-proof epochs take precedence, not fresh stat recapture.
    old = {e['reference']['path']: e for e in hashes.canonical_epochs(captured)}
    captured = hashes.canonical_epochs([*old.values(), *original.epochs(r for r in refs if r['path'] not in old)])
    if {r['path']: r for r in refs} != {e['reference']['path']: e['reference'] for e in captured}:
        raise ValueError('All and only external AND derived targets with original epochs required')
    hashes.check_epochs(captured)
    physical = whole.chunks(refs, c['max_audit_bytes'], c['max_audit_files'])
    plan = dict(status='PENDING_V202_ALL128_FULL_HASH', source_capsule=capsule_ref, producers=producers,
        rows=controls, chunks=physical, files=len(refs), full_bytes=sum(Path(r['path']).stat().st_size for r in refs),
        signature=whole.digest(dict(producers=producers, rows=controls, targets=hashes.targets_digest(refs))),
        initial100_emitted=False, truth_opened=False, recall_measured=False)
    write_json(output / 'plan/plan.json', plan); write_json(output / 'plan/epochs.json', captured)
    return dict(plan=record(output / 'plan/plan.json'), epochs=record(output / 'plan/epochs.json')), dict(
        queries=128, producers=len(producers), files=len(refs), full_bytes=plan['full_bytes'], chunks=len(physical))


def read_plan(output):
    m = stage(output, 'plan'); plan = json.loads(verify(m['artifacts']['plan']).read_text())
    if (plan['status'] != 'PENDING_V202_ALL128_FULL_HASH'
            or plan['truth_opened'] is not False or plan['recall_measured'] is not False
            or plan['initial100_emitted'] is not False
            or [p['phase'] for p in plan['producers']] != list(generation_phases())):
        raise ValueError('Actual ALL384 generators and whole128 before any audit/admission required')
    whole.controls(plan['rows'], list(range(128)))
    refs = hashes.references(r for chunk in plan['chunks'] for r in chunk)
    if (sum(len(x) for x in plan['chunks']) != len(refs) or plan['files'] != len(refs)
            or plan['full_bytes'] != sum(Path(r['path']).stat().st_size for r in refs)
            or any(not chunk or len(chunk) > whole.MAX_AUDIT_FILES
                or sum(Path(r['path']).stat().st_size for r in chunk) > whole.MAX_AUDIT_BYTES
                for chunk in plan['chunks'])
            or plan['signature'] != whole.digest(dict(producers=plan['producers'], rows=plan['rows'],
                                                     targets=hashes.targets_digest(refs)))):
        raise ValueError('No duplicate/missing/sampled physical hash targets')
    value, capsule_ref = anchor(output)
    if plan['source_capsule'] != capsule_ref:
        raise ValueError('Exact actual initial producer capsule required')
    by_path = {r['path']: r for r in refs}
    if any(by_path.get(r['path']) != r for r in value['references']):
        raise ValueError('Every original shared source remains in FULL hash plan')
    return m, plan


def audit_chunk(output, index, mark):
    m, plan = read_plan(output)
    if not 0 <= index < len(plan['chunks']): raise ValueError('Declared whole-file audit chunk required')
    epochs = hashes.canonical_epochs(json.loads(verify(m['artifacts']['epochs']).read_text()))
    hashes.check_epochs(epochs)
    refs = plan['chunks'][index]; size = 0
    for i, ref in enumerate(refs):
        path = verify(ref); size += path.stat().st_size
        if (i+1) % 1024 == 0 or i+1 == len(refs): mark(dict(files_hashed=i+1, full_bytes_hashed=size))
    hashes.check_epochs(epochs)
    receipt = dict(chunk=index, plan=m['artifacts']['plan'], epochs=m['artifacts']['epochs'],
        signature=plan['signature'], files=len(refs), full_bytes_hashed=size,
        exact_target_digest=hashes.targets_digest(refs), every_expected_source_full_hash_verified=True,
        initial100_emitted=False, truth_opened=False, recall_measured=False)
    dest = output / f'audit{index:04d}/receipt.json'; write_json(dest, receipt)
    return dict(receipt=record(dest)), dict(chunk=index, files=len(refs), full_bytes_hashed=size)


def receipts(output):
    m, plan = read_plan(output); result, refs = [], []
    for i in range(len(plan['chunks'])):
        checked = stage(output, f'audit{i:04d}'); r = json.loads(verify(checked['artifacts']['receipt']).read_text())
        if r['plan'] != m['artifacts']['plan'] or r['epochs'] != m['artifacts']['epochs']:
            raise ValueError('Every full receipt bound to exact ALL128 producer plan required')
        result.append(r); refs.append(checked['_reference'])
    whole.require_receipts(plan, result)
    hashes.check_epochs(json.loads(verify(m['artifacts']['epochs']).read_text()))
    return m, plan, refs


def merge(output, batch, c, mark, owned):
    s, population, old, models, geo = inputs(c)
    p, unused, unused_epochs = proofs(output, batch, c, population, geo)
    expected = row_controls(p, batch, population, old, models, geo, s)
    keys = original.RotatedKeys(s['catalog']); owned.append(keys)
    unused_m, plan, audit_refs = receipts(output)
    if expected != plan['rows'][2*batch:2*batch+2]: raise ValueError('Same OWN original initial score before reduction')
    rows, producers = [], []
    for row in expected:
        parts = []
        for i in range(c['partitions']):
            m = stage(output, f'score{batch:03d}-{i:03d}')
            if m['_reference'] != plan['producers'][6*batch+2+i]['reference']:
                raise ValueError('Every original actually successful scorer still unchanged')
            parts.append(json.loads(verify(m['artifacts'][f'part{row["position"]}']).read_text()))
            if len(producers) < 4: producers.append(m['_reference'])
        candidates, trace = original.merge_partitions(SimpleNamespace(sites=keys.sites, keys=keys), c['facts'], parts)
        if trace['signature'] != row['signature']: raise ValueError('No score change during physical exact reduction')
        rows.append(dict(position=row['position'], query_id=row['query_id'], fold=row['fold'],
                         candidates=candidates, trace=trace))
        del parts; gc.collect()
    keys.verify_after(); release(owned)
    dest = output / f'merge{batch:03d}/draft.json'
    write_json(dest, dict(rows=rows, population=128, initial100_emitted=False, truth_opened=False, recall_measured=False))
    return dict(draft=record(dest)), dict(rows=expected, producers=producers, audits=audit_refs,
        plan=record(output / 'plan/plan.json'), queries=2, initial_cap=100)


def collect(output, c):
    m, plan, audit_refs = receipts(output)
    s, population, old, models, geo = inputs(c)
    rows, merge_refs = [], []
    for batch in range(64):
        merged = stage(output, f'merge{batch:03d}')
        if (merged['trace']['audits'] != audit_refs or merged['trace']['plan'] != m['artifacts']['plan']
                or merged['trace']['rows'] != plan['rows'][2*batch:2*batch+2]
                or merged['trace']['producers'] != [plan['producers'][6*batch+2+i]['reference'] for i in range(4)]):
            raise ValueError('Every whole128 reduction bound to ALL audits and original scorers required')
        p = json.loads(verify(merged['artifacts']['draft']).read_text())
        if (p['population'] != 128 or p['truth_opened'] is not False or p['recall_measured'] is not False
                or p['initial100_emitted'] is not False): raise ValueError('No partial or label-opened draft admission')
        rows.extend(p['rows']); merge_refs.append(merged['_reference'])
    whole.outputs(rows, population, plan['rows'], c['sites'], c['facts'], c['partitions'])
    capsule, unused = anchor(output)
    expected_refs = list(capsule['references'])
    original_epochs = list(capsule['source_epochs'])
    for p in plan['producers']:
        produced = stage(output, p['phase'])
        if produced['_reference'] != p['reference']: raise ValueError('Every original producer remains actual and unchanged')
        expected_refs.extend([produced['_reference'], produced['preflight'], parent_exit(output / p['phase']),
                              *produced['artifacts'].values()])
        kind, unused_b, unused_i = phase_info(p['phase'])
        if kind in ('names', 'addresses'):
            proof = json.loads(verify(produced['artifacts']['proofs']).read_text())
            expected_refs.extend(ref for row in proof['rows'] for ref in row['artifacts'].values())
            original_epochs.extend(proof['derived_epochs'])
    if hashes.references(expected_refs) != hashes.references(r for chunk in plan['chunks'] for r in chunk):
        raise ValueError('No external OR own derived feature/producer file may disappear from complete audit')
    stored_epochs = {e['reference']['path']: e for e in hashes.canonical_epochs(
        json.loads(verify(m['artifacts']['epochs']).read_text()))}
    for e in hashes.canonical_epochs(original_epochs):
        if stored_epochs.get(e['reference']['path']) != e:
            raise ValueError('Original pre-read source/proof epochs, never fresh recapture, required')
    return rows, merge_refs, m, plan, audit_refs


def ledger(output, c, mark):
    rows, merge_refs, m, plan, audit_refs = collect(output, c)
    captured = hashes.canonical_epochs(json.loads(verify(m['artifacts']['epochs']).read_text()))
    hashes.check_epochs(captured)
    mark(dict(checkpoint='ORIGINAL_FULL93507_LEDGER_AFTER_ALL128_REDUCTIONS'))
    authority = addresses.address_authority(c)
    if len(authority[2]['epochs']) != 93507: raise ValueError('Entire original source ledger required')
    addresses.check_address_authority(authority); hashes.check_epochs(captured)
    dest = output / 'ledger/draft_initial100.json'
    write_json(dest, dict(rows=rows, population=128, full_query_population=True,
        truth_opened=False, recall_measured=False, initial100_emitted=False, ready_for_fold1=False))
    receipt = dict(draft=record(dest), merges=merge_refs, plan=m['artifacts']['plan'],
        epochs=m['artifacts']['epochs'], audits=audit_refs, signature=plan['signature'],
        source_epoch_count=93507, queries=128, initial100_emitted=False, **original.FLAGS)
    write_json(output / 'ledger/receipt.json', receipt)
    return dict(draft=record(dest), receipt=record(output / 'ledger/receipt.json')), dict(
        queries=128, source_epoch_count=93507, full_query_population=True)


def admission(output):
    load(output)  # FULL implementation before whole-manifest checks.
    c = config(); rows, merges, m, plan, audit_refs = collect(output, c)
    checked = stage(output, 'ledger'); r = json.loads(verify(checked['artifacts']['receipt']).read_text())
    if (r['draft'] != checked['artifacts']['draft'] or r['merges'] != merges or r['plan'] != m['artifacts']['plan']
            or r['epochs'] != m['artifacts']['epochs'] or r['audits'] != audit_refs
            or r['signature'] != plan['signature'] or r['source_epoch_count'] != 93507 or r['queries'] != 128
            or r['initial100_emitted'] is not False or any(r.get(k) is not v for k, v in original.FLAGS.items())):
        raise ValueError('Actual final ledger after ALL128 reductions required')
    p = json.loads(verify(r['draft']).read_text())
    if (p['rows'] != rows or p['population'] != 128 or p['full_query_population'] is not True
            or p['truth_opened'] is not False or p['recall_measured'] is not False
            or p['initial100_emitted'] is not False or p['ready_for_fold1'] is not False):
        raise ValueError('Exact full128 final output before any labels or claim required')
    hashes.check_epochs(json.loads(verify(r['epochs']).read_text()))
    addresses.address_authority(c); load(output)
    return checked, plan


def worker(output, phase):
    kind, batch, index = phase_info(phase); dest = output / phase
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Fresh immutable worker only, no STOP retry')
    owned, budget = [], None; began = time.monotonic()
    def mark(event):
        print(json.dumps(dict(event, elapsed_seconds=time.monotonic()-began,
            rss_bytes=psutil.Process().memory_info().rss,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
    try:
        with Budget(LIMITS) as budget:
            launch = load(output, dest); c = launch['config']
            if kind == 'names': artifacts, trace = name_proofs(output, phase, batch, c, mark, owned)
            elif kind == 'addresses': artifacts, trace = address_proofs(output, phase, batch, c, mark, owned)
            elif kind == 'score': artifacts, trace = score(output, phase, batch, index, c, mark, owned)
            elif kind == 'plan': artifacts, trace = make_plan(output, c, mark)
            elif kind == 'audit': artifacts, trace = audit_chunk(output, index, mark)
            elif kind == 'merge': artifacts, trace = merge(output, batch, c, mark, owned)
            else: artifacts, trace = ledger(output, c, mark)
            release(owned); load(output, dest)
        write_json(dest / 'manifest.json', dict(status='QUARANTINED_V202_'+phase.upper(),
            launch=record(output / 'launch.json'), preflight=record(dest / 'preflight.json'),
            artifacts=artifacts, trace=trace, resources=budget.report(),
            pending_source_hash=kind in ('names', 'addresses', 'score'),
            initial100_emitted=False, old_candidates_opened=False, ready_for_fold1=False, **original.FLAGS))
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            old_candidates_opened=False, **original.FLAGS)); raise
    finally: release(owned)


def run(output):
    began = time.monotonic(); c = config(); inputs(c)
    if output.exists() or not output.resolve().is_relative_to(Path(c['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only immutable whole128 generation root required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=c, implementation=implementation(),
        old_candidates_opened=False, **original.FLAGS))
    wall, completed, max_rss = 0., [], 0
    def execute(phase):
        nonlocal wall, max_rss
        gate = quiet(c['quiet_seconds'], 50, 55); dest = output / phase; dest.mkdir()
        write_json(dest / 'preflight.json', gate)
        with (dest / 'worker.log').open('x') as log:
            s = supervise([sys.executable, '-u', '-m', 'scripts.run_retrieval_v202_population_initial',
                'worker', '--output', str(output), '--phase', phase], log,
                {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
        write_json(dest / 'supervision.json', s)
        m = stage(output, phase); wall += m['resources']['wall_seconds']; completed.append(phase)
        max_rss = max(max_rss, m['resources']['peak_rss_bytes'])
        print(json.dumps(dict(completed=phase, phase_count=len(completed), resources=m['resources'])), flush=True)
    try:
        for phase in execution_phases(): execute(phase)
        execute('plan'); unused, plan = read_plan(output)
        for i in range(len(plan['chunks'])): execute(f'audit{i:04d}')
        for b in range(64): execute(f'merge{b:03d}')
        execute('ledger'); checked, plan = admission(output)
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'),
            ledger=checked['_reference'], plan=record(output / 'plan/manifest.json'), outputs=checked['artifacts']['draft'],
            queries=128, population=128, initial100_emitted=True, full_query_population=True,
            complete_national_source=True, worker_seconds=wall, worker_seconds_per_query=wall/128,
            elapsed_including_quiet_and_parent_seconds=time.monotonic()-began,
            quiet_seconds_per_phase=30, phases=len(completed), peak_worker_rss_bytes=max_rss,
            source_and_derived_files_full_hashed=plan['files'], full_bytes_hashed=plan['full_bytes'],
            disk_cache_state='UNCONTROLLED_NOT_COLD_DISK', slo_validated=False, ready_for_fold1=False,
            downstream_models_changed=False, old_candidates_opened=False, **original.FLAGS))
        print(STATUS, flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), completed=completed,
            initial100_emitted=False, old_candidates_opened=False, **original.FLAGS)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('run', 'worker')); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase'); a = p.parse_args()
    if a.mode == 'worker' and a.phase is None: p.error('Exact declared phase required')
    run(a.output) if a.mode == 'run' else worker(a.output, a.phase)


