"""Entire national source, one frozen query, exact staged initial100; no GT."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import duckdb
import numpy as np
import psutil

from scripts.probe_retrieval_v138_national_names import query_rows
from scripts.probe_retrieval_v125_mapped_blocks import query
from scripts.probe_retrieval_v179_paged_addresses import implementation as address_implementation
from scripts.compile_retrieval_v149_name_rebind import implementation as names_implementation
from src.xgb_matcher.retrieval_activity_decoder_v111 import query_design
from src.xgb_matcher.retrieval_activity_joint_v114 import FEATURES
from src.xgb_matcher.retrieval_activity_semantic_v113 import normalized, scores
from src.xgb_matcher.retrieval_address_authority_v168 import parent_exit
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_certified_address_facts_v181 import CertifiedAddressFacts
from src.xgb_matcher.retrieval_coupled_blocks_v121 import weights
from src.xgb_matcher.retrieval_initial_collector_v146 import AppendedAttributes
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, quiet, LIMITS, POLICY, validate_report
from src.xgb_matcher.retrieval_name_pages_v151 import ScopedNamePages, fingerprint
from src.xgb_matcher.retrieval_paged_addresses_v178 import PagedAddresses, PagedAliasSnapshot
from src.xgb_matcher.retrieval_paged_attributes_v180 import PagedStructuredSites
from src.xgb_matcher.retrieval_paged_names_v170 import PagedReboundNames
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_staged_initial_v183 import (
    FlatNames, PartitionFacts, flatten_names, spans, recipe, reopen_vector, partition_search, merge_partitions)
from src.xgb_matcher.retrieval_supervisor_v164 import supervise
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v184_staged_initial.json'
STATUS = 'SEALED_V184_ONE_COMPLETE_NATIONAL_INITIAL100_COST_ONLY'
FLAGS = {k: k == 'crm_opened' for k in PROTECTED}


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != LIMITS or cfg['numpy_version'] != np.__version__ or cfg['duckdb_version'] != duckdb.__version__
            or cfg['queries'] != 1 or type(cfg['queries']) is not int
            or any(type(cfg[k]) is not int or cfg[k] <= 0 for k in ('sites', 'aliases', 'facts', 'partitions', 'chunk', 'lexical_chunk'))):
        raise ValueError('Exact authorized source-complete single-query integration required')
    return cfg


def inputs(cfg):
    old = json.loads(verify(cfg['query_only_launch']).read_text())
    rows = query_rows(old, cfg['queries'])  # Validate ALL128 before fixed prefix.
    fields = ('query_id', 'crm_name', 'crm_number', 'crm_suffix', 'crm_street', 'crm_postcode', 'crm_insee')
    if any(type(q['fold']) is not int or any(not isinstance(q[k], str) for k in fields) for q in old['queries']):
        raise ValueError('All frozen typed open-fold CRM fields required')
    models = json.loads(verify(old['models']).read_text())
    manifest = json.loads(verify(old['model_manifest']).read_text())
    if (set(models) != {'2', '3', '4'} or manifest['models'] != old['models']
            or manifest['status'] != 'SEALED_TRAINING_ONLY_CROSSFIT_ACTIVITY_SITE_PROBE'
            or manifest['recall_measured'] or manifest['fold0_opened'] or manifest['fold1_opened']
            or manifest['output_candidates'] or not manifest['outer_decoder_parity']
            or not manifest['activity_head_training_cross_fitted'] or not manifest['resources']['pass_resource_policy']):
        raise ValueError('Frozen safe own-fold initial model manifest required')
    ids = {q['query_id'] for q in old['queries']}
    for f, model in models.items():
        if (model['heldout_fold'] != int(f) or set(model['training_folds']) != {2, 3, 4}-{int(f)}
                or ids & set(model['query_ids']) or model['features'] != list(FEATURES)
                or model['identifiers_as_features'] or model['raw_activity_without_site_proof']
                or model['descriptions'] != old['activity_inputs']['descriptions']):
            raise ValueError('Exact frozen no-query-overlap initial30 model required')
        weights(model['weights'])
        verify(model['decoder_parameter'])
    geo = json.loads(verify(old['geo_queries']).read_text())
    if len(geo) != len(ids) or {g['query_id'] for g in geo} != ids:
        raise ValueError('Complete query-only BAN population required')
    return old, rows, models, {g['query_id']: g for g in geo}


def implementation():
    refs = {**address_implementation(), **names_implementation()}
    names = ('retrieval_staged_initial_v183', 'retrieval_coded_leaf_v182', 'retrieval_certified_address_facts_v181',
             'retrieval_paged_attributes_v180', 'retrieval_initial_collector_v146', 'retrieval_initial_engine_v158',
             'retrieval_site_attributes_v139', 'retrieval_append_catalog_v141', 'retrieval_activity_semantic_v113',
             'retrieval_activity_decoder_v111', 'retrieval_activity_joint_v114', 'retrieval_joint_v46')
    files = [Path(__file__), CONFIG, ROOT / 'docs/retrieval_v184_staged_initial_contract.md',
             ROOT / 'scripts/probe_retrieval_v125_mapped_blocks.py',
             *[ROOT / ('src/xgb_matcher/'+n+'.py') for n in names]]
    refs.update({str(p): record(p) for p in files})
    return refs


def load(output, dest):
    launch = json.loads((output / 'launch.json').read_text())
    if (launch['config'] != config() or any(launch.get(k) is not v for k, v in FLAGS.items())
            or launch['old_candidates_opened'] or not output.resolve().is_relative_to(Path(config()['output_root']).resolve())
            or (output / 'status.json').exists() or any(output.glob('*/worker_stop.json'))
            or any(output.glob('*/supervisor_stop.json')) or any(output.glob('*/status.json'))):
        raise ValueError('Unchanged safe immutable staged launch required; no STOP retry')
    for ref in launch['implementation'].values():
        verify(ref)
    gate = json.loads((dest / 'preflight.json').read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual own exact authorized pressure preflight required')
    return launch, inputs(launch['config'])


def check_epochs(entries):
    for entry in entries:
        if list(fingerprint(entry['reference']['path'])) != entry['fingerprint']:
            raise ValueError('Official source epoch changed between physical query phases')


def epochs(refs):
    unique = {r['path']: r for r in refs}
    return [dict(reference=r, fingerprint=list(fingerprint(r['path']))) for r in unique.values()]


def stage(output, name):
    dest = output / name
    if any((dest / n).exists() for n in ('worker_stop.json', 'supervisor_stop.json', 'status.json')):
        raise ValueError('STOP_STAGE_DO_NOT_USE')
    parent_exit(dest)
    ref = record(dest / 'manifest.json')
    m = json.loads(verify(ref).read_text())
    validate_report(m['resources'])
    if (m['status'] != 'SEALED_V184_'+name.upper() or m['launch'] != record(output / 'launch.json')
            or any(m.get(k) is not v for k, v in FLAGS.items()) or m['old_candidates_opened']):
        raise ValueError('Complete actual supervised source-bound stage required')
    if verify(m['preflight']).resolve() != (dest / 'preflight.json').resolve():
        raise ValueError('Actual own immutable phase preflight required')
    gate = json.loads(verify(m['preflight']).read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Same authorized actual phase preflight required')
    check_epochs(m['source_epochs'])
    for artifact in m['artifacts'].values():
        verify(artifact)
    return m, ref


class RotatedKeys:
    def __init__(self, reference):
        self.reference = reference
        self.keys = AppendKeys(reference)
        self.metadata, self.previous, self.sites = self.keys.metadata, self.keys.previous, self.keys.sites
        PagedReboundNames.rotate_keys(self)
    def reconstruct(self, ids, grain):
        out = self.keys.reconstruct(ids, grain)
        PagedReboundNames.rotate_keys(self)
        return out
    def site_positions(self, ids):
        out = self.keys.site_positions(ids)
        PagedReboundNames.rotate_keys(self)
        return out
    def verify_after(self):
        for ref in (self.reference, self.metadata['launch'], *self.keys.references):
            verify(ref)
    def close(self):
        for owner in (self.keys, self.keys.previous):
            for name in ('site', 'parent', 'extra_site', 'extra_parent', 'new_site', 'new_parent'):
                a = getattr(owner, name, None)
                if isinstance(a, np.memmap):
                    a._mmap.close()


def reopen_names(m, cfg):
    return FlatNames(reopen_vector(m['artifacts']['coverage'], cfg['sites'], np.float64),
                     reopen_vector(m['artifacts']['exact'], cfg['sites'], np.bool_), lambda: None, cfg['chunk'])


def reopen_address(m, cfg):
    return PagedAliasSnapshot(reopen_vector(m['artifacts']['coverage'], cfg['aliases'], np.float64),
                              reopen_vector(m['artifacts']['exact'], cfg['aliases'], np.bool_), lambda: None, cfg['chunk'])


def activity(old, row, models, attributes):
    model = models[str(row['fold'])]
    refs = old['activity_inputs']
    cat = json.loads(verify(attributes.metadata['activity_catalog']).read_text())
    if (attributes.metadata['activity_catalog'] != model['official_catalog']
            or cat['official_labels'] != refs['semantic_refs']['labels']):
        raise ValueError('Same exact official activity description/class namespace required')
    vectors = np.load(verify(old['query_vectors']), allow_pickle=False)
    if vectors.shape != (128, 384) or not np.isfinite(vectors).all():
        raise ValueError('All original query-only encoder vectors required')
    descriptions = normalized(np.load(verify(refs['descriptions']), allow_pickle=False))
    parameter = np.load(verify(model['decoder_parameter']), allow_pickle=False)
    values = scores(query_design(vectors[:1]), descriptions, parameter, model['decoder']['family'])[0]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Original finite bounded activity grades required')
    return {i: float(v) for i, v in enumerate(values)}


def worker(output, name):
    dest = output / name
    if any(p.name not in ('preflight.json', 'worker.log') for p in dest.iterdir()):
        raise ValueError('Immutable staged worker already started; no retry')
    budget = None
    owned = []
    step = 'INPUTS'
    try:
        with Budget(LIMITS) as budget:
            launch, (old, rows, models, geo) = load(output, dest)
            cfg, row = launch['config'], rows[0]
            raw_sources = [cfg[k] for k in ('names', 'joins', 'catalog', 'structured', 'addresses', 'query_only_launch')]
            for ref in raw_sources:
                verify(ref)
            source_epochs = epochs(raw_sources)
            artifacts = {}
            trace = {}
            if name == 'names':
                step = 'LOAD_NAMES'
                index = ScopedNamePages(cfg['names']); owned.append(index)
                reader = PagedReboundNames(cfg['joins']); owned.append(reader)
                if reader.sites != cfg['sites'] or reader.metadata['sources']['catalog'] != cfg['catalog']:
                    raise ValueError('Entire original national name/catalog namespace required')
                step = 'WHOLE_NAME_QUERY'
                snap, name_trace = reader.query(index, row['crm_name'], dest / 'query',
                    cfg['max_proof_query_seconds'], cfg['lexical_chunk'])
                owned.append(snap)
                began = time.monotonic()
                def guard():
                    if time.monotonic()-began > cfg['max_proof_query_seconds']:
                        raise TimeoutError('STOP_ALL_SITE_NAME_FLATTEN_COST')
                step = 'FLATTEN_EVERY_SITE'
                flat, gather = flatten_names(snap, dest / 'flat', guard, cfg['chunk']); owned.append(flat)
                artifacts = dict(coverage=flat.coverage.reference, exact=flat.exact.reference)
                trace = dict(name_query=name_trace, gather=gather, flatten_seconds=time.monotonic()-began)
                access_refs = [index.source]
                access_refs.extend(ref for item in index.metadata['access'] for ref in item['artifacts'].values())
                refs = [*reader.refs, *access_refs]
                source_epochs += epochs(refs)
                step = 'NAME_SOURCE_HASH_AFTER'
                for ref in access_refs:
                    verify(ref)
                reader.verify_after(); flat.verify_after(); snap.close(remove=True)
            elif name == 'addresses':
                step = 'LOAD_ADDRESSES'
                index = PagedAddresses(cfg['addresses']); owned.append(index)
                facts = CertifiedAddressFacts(index); owned.append(facts)
                if any(getattr(facts, k) != cfg[k] for k in ('sites', 'facts', 'aliases')) or facts.catalog_reference != cfg['catalog']:
                    raise ValueError('Entire unchanged original address namespace required')
                q = query(row, geo)
                step = 'WHOLE_ADDRESS_QUERY'
                snap, trace = index.snapshot(q.address, dest / 'query', cfg['max_proof_query_seconds'], cfg['lexical_chunk'])
                owned.append(snap)
                snap.verify_after()
                artifacts = dict(coverage=snap.coverage.reference, exact=snap.exact.reference)
                trace = dict(trace, spans=spans(facts, cfg['partitions']), facts=facts.facts, aliases=facts.aliases)
                # Pin the already sealed complete epoch ledger without copying
                # its 93k entries into every query phase. Each scoring child
                # independently loads/checks it via the real V178 constructor;
                # merge checks that full ledger again below, no fake reader.
                source_epochs += epochs([index.launch['epochs']])
                source_epochs += epochs([*index.references, *index.authorities.values(), *facts.used.values()])
                step = 'ADDRESS_SOURCE_HASH_AFTER'
                index.verify_after(); facts.assert_live()
            else:
                nm, nr = stage(output, 'names')
                am, ar = stage(output, 'addresses')
                source_epochs += nm['source_epochs']+am['source_epochs']
                step = 'LOAD_CATALOG'
                keys = RotatedKeys(cfg['catalog']); owned.append(keys)
                catalog = SimpleNamespace(sites=keys.sites, keys=keys)
                if catalog.sites != cfg['sites']:
                    raise ValueError('Entire original site identity denominator required')
                if name == 'merge':
                    step = 'VERIFY_ALL_PHYSICAL_PARTITIONS'
                    parts = []
                    for i in range(cfg['partitions']):
                        part, ref = stage(output, f'score{i:03d}')
                        source_epochs += part['source_epochs']
                        parts.append(json.loads(verify(part['artifacts']['partition']).read_text()))
                    result, trace = merge_partitions(catalog, cfg['facts'], parts)
                    from scripts import compile_retrieval_v177_packed_addresses as address_backend
                    step = 'CHECK_COMPLETE_SOURCE_EPOCH_LEDGER'
                    address_backend.load(Path(cfg['addresses']['path']).parent)
                    write_json(dest / 'initial100.json', dict(query_id=row['query_id'], fold=row['fold'], candidates=result,
                        trace=trace, truth_opened=False, recall_measured=False, ready_for_fold1=False))
                    artifacts = dict(initial100=record(dest / 'initial100.json'))
                else:
                    i = int(name.removeprefix('score'))
                    step = 'LOAD_READONLY_QUERY_PROOFS'
                    flat = reopen_names(nm, cfg); owned.append(flat)
                    snap = reopen_address(am, cfg); owned.append(snap)
                    attrs = PagedStructuredSites(cfg['structured']); owned.append(attrs)
                    attributes = AppendedAttributes(attrs, keys.keys)
                    q = query(row, geo)
                    grades = activity(old, row, models, attrs)
                    signature = recipe(q, models[str(row['fold'])]['weights'], dict(names=nr, addresses=ar,
                        catalog=cfg['catalog'], structured=cfg['structured'], model=old['models'],
                        geography=old['geo_queries'], query_vectors=old['query_vectors']), row['fold'])
                    step = 'LOAD_COMPLETE_ORIGINAL_FACTS'
                    index = PagedAddresses(cfg['addresses']); owned.append(index)
                    facts = CertifiedAddressFacts(index); owned.append(facts)
                    if spans(facts, cfg['partitions']) != am['trace']['spans']:
                        raise ValueError('Same whole original physical fact partition plan required')
                    bound = PartitionFacts(facts, snap, q, am['trace']['spans'][i], cfg['chunk'])
                    step = 'SCORE_EVERY_PHYSICAL_LEAF'
                    began = time.monotonic()
                    part = partition_search(q, models[str(row['fold'])]['weights'], catalog, flat, attributes,
                        bound, signature, grades, cfg['chunk'], cfg['max_partition_score_seconds'])
                    trace = dict(score_seconds=time.monotonic()-began, **part['stats'])
                    source_epochs += epochs([*attrs.references, *facts.used.values(), *index.references])
                    step = 'FULL_USED_SOURCE_HASH_AFTER'
                    facts.verify_after(); index.verify_after(); attrs.verify_after(); flat.verify_after(); snap.verify_after()
                    write_json(dest / 'partition.json', part)
                    artifacts = dict(partition=record(dest / 'partition.json'))
                keys.verify_after()
            step = 'FINAL_SOURCE_AND_INPUT_CHECK'
            unique = {}
            for entry in source_epochs:
                key = entry['reference']['path']
                if key in unique and unique[key] != entry:
                    raise ValueError('Same source reference/epoch required across ALL physical phases')
                unique[key] = entry
            source_epochs = list(unique.values())
            check_epochs(source_epochs)
            load(output, dest)
            size = sum(p.stat().st_size for p in dest.rglob('*.npy'))
            if size > cfg['max_query_disk_bytes']:
                raise ValueError('STOP_OWN_STAGE_PROOF_DISK_BUDGET')
            for value in reversed(owned):
                value.close()
            owned.clear()
        write_json(dest / 'manifest.json', dict(status='SEALED_V184_'+name.upper(), launch=record(output / 'launch.json'),
            preflight=record(dest / 'preflight.json'), artifacts=artifacts, source_epochs=source_epochs,
            trace=trace, resources=budget.report(), old_candidates_opened=False, ready_for_fold1=False,
            initial100_emitted=name == 'merge', disk_cache_state='UNCONTROLLED_NOT_COLD_DISK', **FLAGS))
        print(json.dumps(dict(stage=name, resources=budget.report(), trace=trace, artifacts=artifacts)), flush=True)
    except BaseException as exc:
        write_json(dest / 'worker_stop.json', dict(status='STOP', reason=str(exc), step=step,
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            initial100_emitted=False, old_candidates_opened=False, **FLAGS))
        raise
    finally:
        for value in reversed(owned):
            value.close()


def run(output):
    cfg = config()
    inputs(cfg)
    if output.exists() or not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Fresh CATNAT-only immutable national integration root required')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'launch.json', dict(config=cfg, implementation=implementation(), old_candidates_opened=False, **FLAGS))
    began = time.monotonic()
    names = ['names', 'addresses', *[f'score{i:03d}' for i in range(cfg['partitions'])], 'merge']
    try:
        completed = []
        for name in names:
            gate = quiet(cfg['quiet_seconds'], 50, 55)
            dest = output / name
            dest.mkdir(exist_ok=False)
            write_json(dest / 'preflight.json', gate)
            with (dest / 'worker.log').open('x') as log:
                s = supervise([sys.executable, '-u', '-m', 'scripts.probe_retrieval_v184_staged_initial',
                    'worker', '--output', str(output), '--phase', name], log,
                    {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, dest / 'supervisor_stop.json')
            write_json(dest / 'supervision.json', s)
            m, ref = stage(output, name)
            completed.append(ref)
            print(json.dumps(dict(completed=name, resources=m['resources'], trace=m['trace'])), flush=True)
        merged, ref = stage(output, 'merge')
        total_worker = sum(json.loads(verify(r).read_text())['resources']['wall_seconds'] for r in completed)
        write_json(output / 'manifest.json', dict(status=STATUS, launch=record(output / 'launch.json'), stages=completed,
            outputs=merged['artifacts']['initial100'], sites=cfg['sites'], facts=cfg['facts'], queries=1,
            elapsed_including_quiet_and_all_phases_seconds=time.monotonic()-began,
            all_worker_load_proof_scoring_and_hash_seconds=total_worker, complete_retrieval_index=False,
            initial100_emitted=True, old_candidates_opened=False, slo_validated=False,
            development_panel_reused=True, full_query_population=False, ready_for_fold1=False,
            downstream_models_changed=False, **FLAGS))
        print(STATUS, flush=True)
    except BaseException as exc:
        write_json(output / 'status.json', dict(status='STOP', reason=str(exc), initial100_emitted=False,
            old_candidates_opened=False, **FLAGS))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'worker'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--phase', choices=['names', 'addresses', 'score000', 'score001', 'score002', 'merge'])
    a = p.parse_args()
    if a.mode == 'worker' and a.phase is None:
        p.error('Explicit physical worker phase required')
    (run(a.output) if a.mode == 'run' else worker(a.output, a.phase))
