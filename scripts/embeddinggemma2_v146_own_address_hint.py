"""One predeclared candidate OWN-address geo-hint ablation versus V134.

V145 points are weak community-estimated observations, never canonical XY.
CRM geometry, encoder,51 numeric fields,58 weights and all526 are unchanged.
No additional query-side BANO, parent propagation, fit, fixed pool or reranker.

Night decision: NOT EXECUTED. The postseal V145 coverage audit found zero
source keys among18 public structural misses; nine covered targets already
passed MAXV140. This schedules compute only, never gates source eligibility or
score features. Possible retention of those nine after a future V134 loss is
not denied. Quantization/tie microeffects would not prove new information.
"""
import argparse
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v115_streamed_activity_ablation as baseline
from scripts import embeddinggemma2_v143_empty_reference_continuation as parent
from scripts import embeddinggemma2_v134_supplier_native_ablation as upstream
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v146/own_address_hint_20261014_night'
SOURCE = io.HOME/'v145/candidate_own_address_source_20261014_night'
MANIFEST_SHA = '6dfe63e9a6343f3ae0405b84a9a24588a5922ff007786eb6096f8a4deccb7126'
VARIANT = 'site_own_address_estimated_geo_hint'
RECIPE = dict(hypothesis='SITE_OWN_ADDRESS_ESTIMATED_GEO_HINT', variants=[VARIANT],
    baseline='EXACT_V134_ALL_NEW_SOURCES_MAX', source='EXACT_SEALED_V145_MANIFEST',
    source_role='WEAK_COMMUNITY_ESTIMATED_OWN_ADDRESS_NOT_CANONICAL_XY_OR_IDENTITY',
    source_confidence='.5_FIXED_WEAK_EVIDENCE_NOT_PROBABILITY',
    crm_geometry='EXACT_FROZEN_ORIGINAL_V23_GEOMETRY_NO_NEW_BANO_OR_OSM_QUERY_GEOMETRY',
    qrel='.5_IF_BAN_DECLARED_WEAK_ELSE1',
    ghint='.5*QREL/(1+DISTANCE_M/(300+MAX(0,BAN_UNCERTAINTY_M)))',
    delta='MAX(0,GHINT-MAX(NUMERIC5,NUMERIC42))',
    site_only_bonus='DELTA*(W12+NUMERIC0*W13+NUMERIC2*W14)',
    no_parent16_17_19_increment=True, no_address_or_xy_or_alias_inheritance=True,
    fusion='MAX_HINT_AND_ALL_V134_LEAF_PROOFS_PLUS_IDENTICAL_MAX_RNA_BANATIC_OWNER',
    missing_geometry_or_source_or_four_empty_numeric='ZERO_HINT_ALL_QUERIES_RETAINED',
    public_diffusion='BOTH_EO_AND_UO_AND_NO_SUPPORTED_ORIGINAL_OWN_XY',
    no_canonical_coordinate_write=True, no_fit=True, no_grid=True,
    frozen_encoder='GEMMA_V6_BEST1000', frozen_coefficients=58, frozen_numeric_fields=51,
    initial_cap=100, no_prepool=True, no_downstream_reranking=True, no_positive_injected=True,
    technical_reference_joins_not_identifier_features=True,
    control_reuse='ONLY_EXACT_EQUAL_ALL_LEAF_SCORE_INPUTS_AND_IDENTICAL_OWNER_CONTEXTS',
    all526_and_observed_new_generation_exit0_before_labels=True,
    quantization_bits=baseline.RECIPE['quantization_bits'], full_encoder_oof=False,
    closed_sets_opened=False, production_promotion_allowed=False,
    eligibility='DIAGNOSTIC_CURRENT_COMMUNITY_GEOMETRY_AND_INHERITED_SOURCES_NOT_INDEPENDENT_CONFIRMATION',
    deadline_unix=parent.DEADLINE, resources=dict(prepare_rss_gib=2, native_rss_gib=10,
        free_percent=25, pressure_levels=[0, 1, 2], disk_reserve_gib=40, threads=1))
RSS_LIMIT = 2
NIGHT_ORDERING = dict(status='NON_MEASURED_NO_INIT_PREPARE_OR_NATIVE_RUN_THIS_NIGHT',
    diagnostic='v145/target_coverage_diagnostic_20261014_night/report.json',
    comparison_scope='CONSUMED_MAXV140_ALL526_NOT_FUTURE_V134_OR_INDEPENDENT_POPULATION',
    remaining_public_source_keys=0, remaining_public_queries=18, protected_two_skipped=True,
    covered_target_keys_all526=9, all_nine_already_exact_v140=True,
    scheduling_only_not_feature_or_source_admission=True, source_not_rejected_generally=True,
    future_retention_of_existing_hits_not_ruled_out=True,
    quantization_or_tie_microeffects_not_informative_source_progress=True,
    closed_populations_opened=False)


def guard():
    import psutil
    engine.guard = guard
    if time.time() >= parent.DEADLINE:
        raise TimeoutError('Absolute night deadline; preserve artifacts')
    free, level = io.pressure()
    rss = psutil.Process().memory_info().rss
    if engine.INTERRUPTED or free < 25 or level not in (0, 1, 2) or rss > RSS_LIMIT*1024**3:
        raise engine.ResourceYield(f'V146 cooperative resource yield free={free} level={level} rss={rss}')
    if psutil.disk_usage(io.HOME).free < 40*1024**3:
        raise engine.ResourceYield('CATNAT reserve40GiB; preserve all artifacts')


def authority(out, control=False):
    launch = io.read(out/'launch.json')
    for field in ('implementation', 'recipe', 'parent_launch', 'parent_prepared', 'baseline_launch',
                  'baseline_generation', 'baseline_exit', 'baseline_features', 'baseline_calibration',
                  'osm_generation', 'osm_exit', 'rna_generation', 'rna_exit', 'banatic_generation', 'banatic_exit',
                  'supplier_generation', 'supplier_exit', 'address_generation', 'address_exit',
                  'hint_manifest', 'hint_generation', 'hint_launch', 'hint_exit', 'public_source', 'public_mask'):
        verify(launch[field])
    for helper in launch['helpers']:
        verify(helper)
    guard()
    if control:
        receipt = io.read(out/'control.json')
        for field in ('upstream_launch', 'upstream_generation', 'upstream_exit', 'upstream_parent_exit'):
            verify(receipt[field])
        if receipt['status'] != 'OWNED_COMPLETE_FROZEN_V134_CONTROL_BOUND':
            raise ValueError('Real complete V134 control before new native generation')
    return launch


def initialize(out):
    guard()
    if out.exists():
        raise ValueError('Fresh single-hypothesis namespace')
    old = parent.authority(parent.ROOT)
    guard()
    prepared = io.read(parent.ROOT/'prepared.json')
    px = io.read(parent.ROOT/'prepare_exit.json')
    if prepared['status'] != 'COMPLETE_ALL526_LABEL_FREE_SOURCE_SCORES' or px['exit_code'] or not px.get('own_child_exit_confirmed'):
        raise ValueError('Actual complete V143 preparation before source projection')
    panel = io.read(verify(old['panel']))
    scope = {q['query_id'] for q in panel}
    if len(scope) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel) or set(prepared['shards']) != scope:
        raise ValueError('Same all526 safe input population')
    manifest_ref = record(SOURCE/'manifest.json')
    if manifest_ref['sha256'] != MANIFEST_SHA:
        raise ValueError('Exact predeclared V145 source manifest')
    manifest = io.read(verify(manifest_ref))
    generation = io.read(verify(manifest['generation']))
    hx = io.read(verify(manifest['own_exit']))
    if (manifest['status'] != 'READY_HASHED_SOURCE_ONLY_FOR_LATER_DIAGNOSTIC_PROJECTION_NOT_RECALL'
            or generation['status'] != 'COMPLETE_HASHED_SPARSE_COMMUNITY_OWN_ADDRESS_SOURCE_NOT_RETRIEVAL'
            or hx['exit_code'] or not hx['own_child_exit_confirmed'] or not generation['public_source_checks_passed']
            or generation['labels_read'] or generation['canonical_coordinates_written'] or not generation['sparse_sorted_unique']):
        raise ValueError('Actual public query-only source, not inferred successful source')
    sl = io.read(verify(generation['launch']))
    for source_field, original_field in (('endpoint_keys', 'keys'), ('official_supported_own_xy', 'xy'),
                                      ('own_site', 'stock_source'), ('official_owners', 'public_owners')):
        if sl['inputs'][source_field] != old[original_field]:
            raise ValueError('Same canonical snapshot/public source binding')
    if sl['inputs']['v143_launch'] != record(parent.ROOT/'launch.json'):
        raise ValueError('V145 exact query-only reference-universe parent binding')
    public_source = io.read(verify(sl['inputs']['public_source']))
    if public_source['public_mask'] != sl['inputs']['both_public_mask']:
        raise ValueError('Exact both-public mask')
    receipts = {}
    for name, directory, status in (
            ('supplier', upstream.SOURCE, 'COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT'),
            ('address', upstream.ADDRESS, 'COMPLETE_QUERY_ONLY_ALL526_CONTINUITY_ADDRESS_FEASIBILITY_NOT_RECALL')):
        g, x = io.read(directory/'generation.json'), io.read(directory/'owned_tool_exit.json')
        if g['status'] != status or g['query_panel'] != old['panel'] or x['exit_code'] or not x['own_tool_exit_confirmed']:
            raise ValueError('Complete same-panel frozen leaf-context source')
        verify(g['extras'])
        receipts[name] = (record(directory/'generation.json'), record(directory/'owned_tool_exit.json'))
    cal = io.read(verify(old['baseline_calibration']))
    for ref in cal['models'].values():
        model = io.read(verify(ref))
        if len(model['coefficients']) != 58 or len(model['feature_fields']) != 58:
            raise ValueError('Exact frozen58 model')
    out.mkdir(parents=True)
    (out/'spill').mkdir()
    (out/'calibration').mkdir()
    io.write(out/'recipe.json', RECIPE)
    launch = dict(old, implementation=record(Path(__file__)), recipe=record(out/'recipe.json'),
        parent_launch=record(parent.ROOT/'launch.json'), parent_prepared=record(parent.ROOT/'prepared.json'),
        supplier_generation=receipts['supplier'][0], supplier_exit=receipts['supplier'][1],
        address_generation=receipts['address'][0], address_exit=receipts['address'][1],
        hint_manifest=manifest_ref, hint_generation=manifest['generation'], hint_launch=generation['launch'], hint_exit=manifest['own_exit'],
        public_source=sl['inputs']['public_source'], public_mask=sl['inputs']['both_public_mask'],
        nonpredictive_night_ordering_note=NIGHT_ORDERING, created_unix=time.time())
    launch['helpers'].extend([record(Path(parent.__file__)), record(Path(upstream.__file__)),
                             record(io.ROOT/'scripts/embeddinggemma2_v145_candidate_own_address_source.py')])
    io.write(out/'launch.json', launch)
    io.write(out/'features_generation.json', io.read(parent.ROOT/'features_generation.json'))
    io.write(out/'calibration/generation.json', cal)


def source_points(launch):
    generation = io.read(verify(launch['hint_generation']))
    keys = np.load(verify(generation['artifacts']['keys.npy']))
    points = np.load(verify(generation['artifacts']['xy_points.npy']))
    verify(generation['artifacts']['provenance.jsonl'])
    if (keys.dtype != np.uint64 or points.shape != (len(keys), 2) or np.any(keys[1:] <= keys[:-1])
            or not np.isfinite(points).all() or not np.all((points[:, 0] >= 0) & (points[:, 0] <= 1300000)
            & (points[:, 1] >= 6000000) & (points[:, 1] <= 7300000))):
        raise ValueError('Unique bounded finite projected hint points')
    canonical_keys = np.load(verify(launch['keys']), mmap_mode='r')
    canonical_xy = np.load(verify(launch['xy']), mmap_mode='r')
    public = np.load(verify(launch['public_mask']), mmap_mode='r')
    if public.dtype != bool or public.shape != (len(canonical_keys),) or canonical_xy.shape != (len(canonical_keys), 2):
        raise ValueError('Exact both-public canonical snapshot schema')
    at = np.searchsorted(canonical_keys, keys)
    if np.any(at >= len(canonical_keys)) or not np.array_equal(canonical_keys[at], keys) or not np.all(public[at]):
        raise ValueError('Every hint key truly original and both-public')
    coords = np.asarray(canonical_xy[at], dtype=float)
    supported = np.isfinite(coords).all(axis=1) & (coords[:, 0] >= 0) & (coords[:, 0] <= 1300000) & (coords[:, 1] >= 6000000) & (coords[:, 1] <= 7300000)
    if np.any(supported):
        raise ValueError('Never replace a supported original own coordinate')
    del canonical_keys, canonical_xy, public, coords, at, supported
    gc.collect()
    guard()
    return keys, points


def prepare(out):
    launch = authority(out)
    panel = io.read(verify(launch['panel']))
    fg, pg = io.read(verify(launch['baseline_features'])), io.read(verify(launch['parent_prepared']))
    geometry = {r['query_id']: r['geometry'] for r in io.read(verify(launch['geometry']))}
    if set(geometry) != {q['query_id'] for q in panel}:
        raise ValueError('Whole original fixed CRM geometries')
    cal = io.read(out/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    supplier = io.read(verify(io.read(verify(launch['supplier_generation']))['extras']))
    address = io.read(verify(io.read(verify(launch['address_generation']))['extras']))
    point_keys, points = source_points(launch)
    root = out/'extras'
    root.mkdir(exist_ok=True)
    shards, metadata = {}, {}
    for q in panel:
        guard()
        qid = q['query_id']
        path = root/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
        info_path = path.with_suffix('.metadata.json')
        if engine.sealed(path):
            info = io.read(info_path)
            if info['query'] != q or info['model'] != cal['models'][str(assignment[qid])]:
                raise ValueError('Exact prepared resume model/query binding')
            shards[qid], metadata[qid] = record(path), info
            continue
        model = models[str(assignment[qid])]
        if qid in model['train_queries'] or qid not in model['heldout_queries']:
            raise ValueError('Frozen heldout calibrator')
        started = time.monotonic()
        w = np.asarray(model['coefficients'], dtype=float)
        with np.load(verify(fg['features'][qid]), allow_pickle=False) as f, np.load(verify(pg['shards'][qid]), allow_pickle=False) as p:
            keys, features = f['keys'], f['features']
            if features.shape != (len(keys), 51) or not np.array_equal(keys, p['keys']):
                raise ValueError('All original full51 references, including empty queries')
            own, temporal = upstream.projected(keys, supplier[qid]['own']), upstream.projected(keys, supplier[qid]['temporal'])
            continuity = upstream.projected(keys, address[qid])
            previous = np.maximum(np.maximum(p['geometry'], p['named']), float(w[6])*np.maximum(np.maximum(own, temporal), continuity))
            hint = np.zeros(len(keys), dtype=float)
            at = np.searchsorted(point_keys, keys)
            eligible = at < len(point_keys)
            eligible[eligible] &= point_keys[at[eligible]] == keys[eligible]
            positions = np.flatnonzero(eligible)
            g = geometry[qid]
            valid_geometry = bool(g['spatial_crs_compatible'])
            if valid_geometry:
                anchor = np.asarray([g['ban_x_l93'], g['ban_y_l93']], dtype=float)
                uncertainty = float(g['ban_uncertainty_m'])
                valid_geometry = bool(np.isfinite(anchor).all() and np.isfinite(uncertainty)
                    and 0 <= anchor[0] <= 1300000 and 6000000 <= anchor[1] <= 7300000)
            if valid_geometry and len(positions):
                coords = points[at[positions]]
                distance = np.hypot(coords[:, 0]-anchor[0], coords[:, 1]-anchor[1])
                qrel = .5 if g['ban_declared_weak'] else 1.
                ghint = .5*qrel/(1+distance/(300+max(0., uncertainty)))
                delta = np.maximum(0., ghint-np.maximum(features[positions, 5], features[positions, 42]))
                hint[positions] = delta*(w[12]+features[positions, 0].astype(float)*w[13]+features[positions, 2].astype(float)*w[14])
            score = np.maximum(previous, hint)
            if not np.isfinite(score).all() or np.any(hint < 0):
                raise ValueError('Finite nonnegative weak site-only score hint')
            info = dict(query=q, model=cal['models'][str(assignment[qid])], geometry=launch['geometry'],
                full_numeric_references=len(keys), eligible_hint_refs=int(eligible.sum()), crm_geometry_usable=valid_geometry,
                positive_hint_refs=int(np.count_nonzero(hint)), changed_max_leaf_refs=int(np.count_nonzero(score != previous)),
                exact_v134_leaf_input_equivalence=bool(np.array_equal(score, previous)),
                query_only_no_labels=True, projection_seconds=time.monotonic()-started)
            np.savez_compressed(path, keys=keys, hint=hint, previous=previous)
            io.write(info_path, info)
            engine.seal(path)
            shards[qid], metadata[qid] = record(path), info
        gc.collect()
        io.write(out/'prepare_state.json', dict(status='ALL526_QUERY_ONLY_WEAK_OWN_ADDRESS_HINT',
            queries=len(shards), total=526, updated_unix=time.time()))
    authority(out)
    io.write(out/'prepared.json', dict(status='COMPLETE_ALL526_LABEL_FREE_OWN_ADDRESS_HINT_SCORES',
        shards=shards, metadata=metadata, queries=526,
        queries_changed_max=sum(not m['exact_v134_leaf_input_equivalence'] for m in metadata.values()),
        queries_zero_or_exact_reuse=sum(m['exact_v134_leaf_input_equivalence'] for m in metadata.values()),
        total_changed_max_refs=sum(m['changed_max_leaf_refs'] for m in metadata.values()),
        source_manifest=launch['hint_manifest'], no_canonical_xy_written=True,
        no_new_query_geometry=True, source_public_and_missing_original_xy_validated=True,
        labels_read=False, no_recall=True))


def bind_control(out):
    launch = authority(out)
    upstream_launch = upstream.authority(upstream.ROOT)
    for field in ('panel', 'baseline_features', 'baseline_calibration', 'parent_prepared',
                  'supplier_generation', 'address_generation', 'rna_generation', 'banatic_generation', 'index_meta', 'index_manifest'):
        if launch[field] != upstream_launch[field]:
            raise ValueError('Exact frozen V134 input baseline, not approximate control')
    for variant in upstream.VARIANTS:
        g, x = io.read(upstream.ROOT/variant/'generation.json'), io.read(upstream.ROOT/variant/'generate_exit.json')
        if g['status'] != 'COMPLETE_SEALED_ALL526' or x['exit_code'] or not x['own_child_exit_confirmed']:
            raise ValueError('All four complete actual V134 owned outputs before binding control')
        verify(g['outputs'])
    parent_exit = io.read(upstream.ROOT/'owned_tool_exit.json')
    if parent_exit['exit_code'] or not parent_exit['own_tool_exit_confirmed']:
        raise ValueError('V134 real observed parent completion')
    primary = upstream.ROOT/'all_new_sources_max'
    receipt = dict(status='OWNED_COMPLETE_FROZEN_V134_CONTROL_BOUND',
        upstream_launch=record(upstream.ROOT/'launch.json'), upstream_generation=record(primary/'generation.json'),
        upstream_exit=record(primary/'generate_exit.json'), upstream_parent_exit=record(upstream.ROOT/'owned_tool_exit.json'),
        all_leaf_source_model_and_owner_context_inputs_equal=True, no_outcomes_read=True)
    path = out/'control.json'
    if path.exists() and io.read(path) != receipt:
        raise ValueError('Never overwrite another control binding')
    io.write(path, receipt)


def generate(out):
    launch = authority(out, control=True)
    prepared = io.read(out/'prepared.json')
    px = io.read(out/'prepare_exit.json')
    if prepared['status'] != 'COMPLETE_ALL526_LABEL_FREE_OWN_ADDRESS_HINT_SCORES' or px['exit_code'] or not px.get('own_child_exit_confirmed', px.get('own_tool_exit_confirmed', False)):
        raise ValueError('Actual complete observed preparation before national native query')
    baseline.bind_runtime(out)
    panel = io.read(verify(launch['panel']))
    fg, cal = io.read(out/'features_generation.json'), io.read(out/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    rna = io.read(verify(io.read(verify(launch['rna_generation']))['context']))
    banatic = io.read(verify(io.read(verify(launch['banatic_generation']))['context']))
    control = io.read(out/'control.json')
    cg = io.read(verify(control['upstream_generation']))
    previous = io.read(verify(cg['outputs']))
    old_cost = io.read(verify(cg['timings']))
    backend = TantivyBackend(io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Full unchanged national corpus')
    root = out/VARIANT
    (root/'native').mkdir(parents=True, exist_ok=True)
    rows, costs, changed = {}, {}, 0
    for q in panel:
        guard()
        qid, fold = q['query_id'], str(assignment[q['query_id']])
        path = root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):
            part = io.read(engine.sealed(path))
        else:
            with np.load(verify(prepared['shards'][qid]), allow_pickle=False) as p:
                keys, score = p['keys'], np.maximum(p['hint'], p['previous'])
                identical = np.array_equal(score, p['previous'])
            if identical:
                candidates = previous['learned_native_ooc'][qid]
                cost = dict(old_cost[qid], unchanged_outputs_reused=True, reused_exact_v134_leaf_and_owner_inputs=True)
            else:
                merged = {r['siren_ref']: r['confidence'] for r in rna[qid]}
                for r in banatic[qid]:
                    merged[r['siren_ref']] = max(merged.get(r['siren_ref'], 0.), r['confidence'])
                owners = [dict(siren_ref=r, confidence=c) for r, c in sorted(merged.items())]
                model = models[fold]
                if qid in model['train_queries'] or qid not in model['heldout_queries']:
                    raise ValueError('Frozen heldout model')
                with np.load(verify(fg['features'][qid]), allow_pickle=False) as f:
                    if not np.array_equal(keys, f['keys']) or f['features'].shape != (len(keys), 51):
                        raise ValueError('Unchanged full51 references')
                    proxy = baseline.baseline.BackendProxy(backend, parent.EvidenceCollector(
                        backend, q, keys, score, owners, float(model['coefficients'][6])))
                    candidates, cost = baseline.native_search(proxy, q, keys, f['features'], model['coefficients'])
                cost.update(unchanged_outputs_reused=False, weak_own_address_hint_site_only=True)
            part = dict(query=q, model=cal['models'][fold], prepared=prepared['shards'][qid],
                candidates=candidates, cost=cost, labels_read=False, reuse_not_new_full_slo=identical)
            io.write(path, part)
            engine.seal(path)
            gc.collect()
        if part['query'] != q or part['model'] != cal['models'][fold] or part['prepared'] != prepared['shards'][qid] or len(part['candidates']) > 100:
            raise ValueError('Exact model/query/source resume and absolute initial100')
        rows[qid], costs[qid] = part['candidates'], part['cost']
        changed += not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json', dict(status='ALL526_NATIVE_OWN_ADDRESS_HINT_NO_LABELS',
            queries=len(rows), total=526, recomputed=changed, updated_unix=time.time()))
    if len(rows) != 526 or any(len({r['siret'] for r in cs}) != len(cs) for cs in rows.values()):
        raise ValueError('All526 and unique capped100 output')
    authority(out, control=True)
    io.write(root/'outputs.json', dict(previous, learned_native_ooc=rows))
    io.write(root/'native_timings.json', costs)
    io.write(root/'generation.json', dict(status='COMPLETE_SEALED_ALL526', queries=526,
        outputs=record(root/'outputs.json'), timings=record(root/'native_timings.json'),
        recomputed_queries=changed, no_positive_injected=True, no_prepool=True,
        no_downstream_reranking=True, labels_opened_in_native_worker=False, full_encoder_oof=False,
        runtime_reuse_not_full_new_cold_warm_slo=True))


def run(out):
    import psutil
    authority(out, control=True)
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = out/VARIANT
        root.mkdir(exist_ok=True)
        while True:
            if engine.INTERRUPTED or time.time() >= parent.DEADLINE-30:
                raise TimeoutError('Hard deadline/user stop before new native admission')
            try:
                authority(out, control=True)
            except engine.ResourceYield as error:
                io.write(out/'supervisor_state.json', dict(status='WAITING_REAL_RESOURCE_ADMISSION', reason=str(error), updated_unix=time.time()))
                time.sleep(30)
                continue
            command = [str(engine.CPU), '-u', '-m', 'scripts.embeddinggemma2_v146_own_address_hint', 'generate', '--output', str(out)]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(out/'spill'), RAYON_NUM_THREADS='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            reason = None
            with (root/'generate.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                io.write(root/'owned_process.json', dict(pid=child.pid, command=command, owned=True, started_unix=time.time()))
                while child.poll() is None:
                    free, level = io.pressure()
                    try:
                        rss = psutil.Process(child.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        rss = 0
                    if engine.INTERRUPTED or time.time() >= parent.DEADLINE-30 or free < 25 or level not in (0, 1, 2) or rss > 10*1024**3:
                        reason = dict(user_stop=engine.INTERRUPTED, deadline=time.time() >= parent.DEADLINE-30, rss=rss, free=free, level=level)
                        child.terminate()
                        try:
                            child.wait(timeout=25)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        break
                    time.sleep(5)
                rc = child.wait()
            io.write(root/'generate_exit.json', dict(exit_code=rc, pid=child.pid, own_child_exit_confirmed=True,
                resource_or_deadline_reason=reason, finished_unix=time.time()))
            if reason and (reason['user_stop'] or reason['deadline']):
                raise TimeoutError('Deadline/user stop; no partial recall')
            if rc == 75 or reason:
                time.sleep(30)
                continue
            if rc:
                raise RuntimeError(f'Actual owned generation exit{rc}; no partial labels')
            break
        evaluate(out)


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out, control=True)
    root = out/VARIANT
    g, x = io.read(root/'generation.json'), io.read(root/'generate_exit.json')
    if g['status'] != 'COMPLETE_SEALED_ALL526' or x['exit_code'] or not x['own_child_exit_confirmed']:
        raise ValueError('Whole new526 actual owned exit0 before labels')
    verify(g['outputs'])
    io.write(root/'launch.json', io.read(out/'launch.json'))
    (root/'calibration').mkdir(exist_ok=True)
    io.write(root/'calibration/generation.json', io.read(out/'calibration/generation.json'))
    evaluator.proof = sys.modules[__name__]
    evaluator.evaluate(root)
    fresh = {r['query_id']: r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
    paired = {}
    for name, control_root in [('v134_all_new_sources_max', upstream.ROOT/'all_new_sources_max'), ('v115', baseline.OUTPUT)]:
        old = {r['query_id']: r for r in io.read(control_root/'outcomes.json')['learned_native_ooc']}
        if len(fresh) != 526 or set(fresh) != set(old) or any(fresh[q][f] != old[q][f]
                for q in old for f in ('fold', 'zone', 'siren_group', 'calibrated_query_weight')):
            raise ValueError('Exact all-human evaluation units and weights')
        paired[name] = {m: dict(gains=[q for q in old if fresh[q][m] and not old[q][m]],
                               losses=[q for q in old if old[q][m] and not fresh[q][m]])
                        for m in ('exact', 'siren', 'operational')}
    report = io.read(root/'report.json')
    report.update(paired_comparisons=paired, eligibility=RECIPE['eligibility'], production_promotion_allowed=False,
        encoder_train_sirens_disjoint=None, full_encoder_oof=False, independent_encoder_generalization_not_validated=True,
        runtime_contains_equivalent_output_reuse=True, no_full_new_cold_warm_slo=True)
    io.write(root/'report.json', report)
    io.write(out/'report.json', dict(status='COMPLETE_SINGLE_PREDECLARED_OWN_ADDRESS_HINT_ALL526',
        hypothesis=RECIPE['hypothesis'], report=record(root/'report.json'), paired_comparisons=paired,
        metrics=report['variants']['learned_native_ooc'], no_holdout_or_national_certification=True, finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'prepare', 'bind-control', 'generate', 'run', 'evaluate'])
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    RSS_LIMIT = 10 if args.action in ('generate', 'run', 'evaluate') else 2
    signal.signal(signal.SIGTERM, engine.interrupted)
    signal.signal(signal.SIGINT, engine.interrupted)
    try:
        globals()[args.action.replace('-', '_')](args.output)
    except engine.ResourceYield as error:
        print(str(error), flush=True)
        sys.exit(75)
    except Exception as error:
        if args.output.exists():
            io.write(args.output/(args.action+'_error.json'), dict(error=repr(error), updated_unix=time.time()))
        raise
