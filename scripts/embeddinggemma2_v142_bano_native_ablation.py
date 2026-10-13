"""Predeclared BANO address-cloud proof before one national initial100.

No BAN seed, centroid-as-house, fitting, target-directed rescue or closed data.
Two complete observed generations precede labels; input-equivalent upstream
outputs may be reused, never selected for their labelled retrieval outcomes.
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
from scripts import embeddinggemma2_v134_supplier_native_ablation as upstream
from scripts import embeddinggemma2_v132_local_evidence_ablation as parent
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v115_streamed_activity_ablation as baseline
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v142/bano_native_20261013_night'
BANO = io.HOME/'v141/bano_osm_20261013_night_d'
VARIANTS = ('bano_geo_only', 'all_new_sources_max_bano')
RECIPE = dict(variants=VARIANTS, primary_predeclared='all_new_sources_max_bano',
    scope='SAME_ALL526_CONSUMED_DEVELOPMENT_NOT_CONFIRMATION',
    bano_source='WHOLE_NATIONAL_QUARANTINED_CSV_EXPLICIT_OSM_ONLY_NOT_BAN_SEEDED',
    bano_points='ALL_COMPATIBLE_EXACT_HOUSE_POINTS_ELSE_ALL_COMPATIBLE_STREET_POINTS',
    house_precision='PUBLISHED_ADDRESS_POINTS_NOT_CERTIFIED_ENTRANCE_OR_SINGLE_BUILDING',
    street_precision='WEAK_PUBLISHED_ADDRESS_CLOUD_NOT_CENTROID_HOUSE_OR_INTERPOLATED_AXIS',
    bano_geo='MAX_BASELINE_OWN_GEO_AND_0.5/(1+MIN_POINT_DISTANCE_M/300)',
    changed_fields=[5, 6, 7, 16, 17, 19], frozen_features=51, frozen_calibrator=58,
    numerical_increment='NONNEGATIVE_DELTA_WITH_UNCHANGED_58_COEFFICIENTS',
    primary='MAX_BANO_GEOM_OSM_GEOM_OSM_NAME_SUPPLIER_OWN_TEMPORAL_ADDRESS_PER_LEAF_PLUS_MAX_RNA_BANATIC_OWNER',
    no_new_source_double_vote=True, no_statistical_independence_claim=True,
    original_full_numeric_keys_and_public_own_coordinates_only=True,
    public_diffusion='EXACT_V132_eO_AND_uO_MASK', original_national_lexical_unrestricted=True,
    initial_cap=100, no_prepool=True, no_fit=True, no_grid=True, no_reranking=True,
    no_positive_injected=True, no_identifier_features=True,
    reuse='ONLY_EXACT_EQUAL_LEAF_SCORES_AND_OWNER_CONTEXTS_OR_EXACT_ZERO_NEW_SOURCE',
    labels_only_after_two_whole_generations_and_actual_owned_exits=True,
    exact_operational_separate=True, full_encoder_oof=False, closed_sets_opened=False,
    eligibility='DIAGNOSTIC_CURRENT_COMMUNITY_PARTIAL_PUBLISHER_AND_INHERITED_POSTCUTOFF_SOURCES',
    production_promotion_allowed=False, quantization_bits=baseline.RECIPE['quantization_bits'],
    deadline_unix=parent.DEADLINE, resources=dict(threads=1, rss_gib=10,
        free_percent=25, pressure_levels=[0, 1, 2], disk_reserve_gib=40))


def guard():
    parent.guard()
    engine.guard = guard  # Process local; no immutable helper changes.


def authority(out):
    launch = upstream.authority(out)
    for field in ('bano_generation', 'bano_owned_exit', 'bano_launch', 'public_source',
                  'public_mask', 'upstream_launch', 'upstream_primary_generation', 'upstream_primary_exit'):
        verify(launch[field])
    guard()
    return launch


def complete_family(root, variants):
    for variant in variants:
        gen = io.read(root/variant/'generation.json')
        ex = io.read(root/variant/'generate_exit.json')
        if gen['status'] != 'COMPLETE_SEALED_ALL526' or gen['queries'] != 526:
            raise ValueError('Whole upstream generations, not partial branches')
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:
            raise ValueError('Observed owned exit0 for every upstream generation')
        verify(gen['outputs'])
        verify(gen['timings'])


def initialize(out):
    guard()
    if out.exists():
        raise ValueError('Fresh predeclared namespace')
    complete_family(parent.ROOT, parent.VARIANTS)
    complete_family(upstream.ROOT, upstream.VARIANTS)
    old = upstream.authority(upstream.ROOT)
    panel = io.read(verify(old['panel']))
    scope = {q['query_id'] for q in panel}
    if len(scope) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel):
        raise ValueError('Same all526 safe development population')
    bg = io.read(BANO/'generation.json')
    bx = io.read(BANO/'owned_tool_exit.json')
    if (bg['status'] != 'COMPLETE_SCAN_WITH_QUARANTINE_NATIONAL_BANO_OSM_ALL526_CONTEXTS'
            or bg['all_queries'] != 526 or bx['exit_code'] or not bx['own_tool_exit_confirmed']):
        raise ValueError('Complete national BANO source with actual observed exit0')
    bl = io.read(verify(bg['launch']))
    if (bl['query_only'] != old['panel'] or bl['retained_source_codes'] != ['OSM']
            or not bg['no_BAN_seed_used'] or not bl['no_BAN_geometry_input']):
        raise ValueError('Exact query binding and explicit independent-input provenance')
    context = io.read(verify(bg['query_context']))
    if set(context) != scope:
        raise ValueError('No source-selected denominator')
    verify(bg['quarantine'])
    prepared = io.read(verify(old['parent_prepared']))
    if prepared['status'] != 'COMPLETE_ALL526_LABEL_FREE_SOURCE_SCORES' or set(prepared['shards']) != scope:
        raise ValueError('All upstream query-only scores')
    public_source_path = verify(prepared['public'])
    public = io.read(public_source_path)
    if public['site_source'] != old['stock_source'] or public['owner_source'] != old['public_owners']:
        raise ValueError('Both frozen public diffusion sources')
    verify(public['public_mask'])
    cal = io.read(verify(old['baseline_calibration']))
    for ref in cal['models'].values():
        model = io.read(verify(ref))
        if len(model['coefficients']) != 58 or len(model['feature_fields']) != 58:
            raise ValueError('Frozen58=7+51 model, no fitting')
    out.mkdir(parents=True)
    (out/'spill').mkdir()
    (out/'calibration').mkdir()
    io.write(out/'recipe.json', RECIPE)
    primary = upstream.ROOT/'all_new_sources_max'
    launch = dict(old, implementation=record(Path(__file__)), recipe=record(out/'recipe.json'),
        bano_generation=record(BANO/'generation.json'), bano_owned_exit=record(BANO/'owned_tool_exit.json'),
        bano_launch=bg['launch'], public_source=record(public_source_path), public_mask=public['public_mask'],
        upstream_launch=record(upstream.ROOT/'launch.json'),
        upstream_primary_generation=record(primary/'generation.json'),
        upstream_primary_exit=record(primary/'generate_exit.json'), created_unix=time.time())
    launch['helpers'].extend([record(Path(upstream.__file__)), record(Path(parent.__file__)),
        record(io.ROOT/'scripts/embeddinggemma2_v141_bano_quarantined_source.py')])
    io.write(out/'launch.json', launch)
    io.write(out/'features_generation.json', io.read(upstream.ROOT/'features_generation.json'))
    io.write(out/'calibration/generation.json', io.read(upstream.ROOT/'calibration/generation.json'))


def compatible_points(context):
    cloud = context['published_street_address_cloud']
    indices = context['exact_house_point_indices']
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(cloud) for i in indices):
        raise ValueError('Exact source house subset')
    def allowed(point):
        if point['source'] != 'OSM':
            raise ValueError('Only explicitly admitted OSM address observations')
        if not point['l93_compatible']:
            return False
        p = np.asarray(point['point_l93'], dtype=float)
        return bool(p.shape == (2,) and np.isfinite(p).all() and 0 <= p[0] <= 1300000
                    and 6000000 <= p[1] <= 7300000)
    houses = [cloud[i]['point_l93'] for i in indices if allowed(cloud[i])]
    if houses:
        return houses, 'ALL_COMPATIBLE_EXACT_HOUSE_POINTS'
    return [p['point_l93'] for p in cloud if allowed(p)], 'ALL_COMPATIBLE_STREET_POINTS_WEAK'


def prepare(out):
    launch = authority(out)
    panel = io.read(verify(launch['panel']))
    fg = io.read(verify(launch['baseline_features']))
    contexts = io.read(verify(io.read(verify(launch['bano_generation']))['query_context']))
    keys_all = np.load(verify(launch['keys']), mmap_mode='r')
    xy = np.load(verify(launch['xy']), mmap_mode='r')
    public = np.load(verify(launch['public_mask']), mmap_mode='r')
    if public.dtype != bool or public.shape != (len(keys_all),) or xy.shape != (len(keys_all), 2):
        raise ValueError('Same public coordinate universe')
    cal = io.read(out/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    root = out/'extras'
    root.mkdir(exist_ok=True)
    shards = {}
    for q in panel:
        guard()
        qid = q['query_id']
        path = root/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')
        if engine.sealed(path):
            shards[qid] = record(path)
            continue
        model = models[str(assignment[qid])]
        if qid in model['train_queries'] or qid not in model['heldout_queries']:
            raise ValueError('Frozen heldout calibrator')
        points, precision = compatible_points(contexts[qid])
        with np.load(verify(fg['features'][qid]), allow_pickle=False) as f:
            keys, features = f['keys'], f['features']
            if features.shape != (len(keys), 51):
                raise ValueError('Frozen numerical51, no pool pruning')
            at = np.searchsorted(keys_all, keys)
            valid = at < len(keys_all)
            valid[valid] &= keys_all[at[valid]] == keys[valid]
            valid[valid] &= public[at[valid]]
            pos = np.flatnonzero(valid)
            coords = np.asarray(xy[at[pos]], dtype=float)
            good = (np.isfinite(coords).all(axis=1) & (coords[:, 0] >= 0) & (coords[:, 0] <= 1300000)
                    & (coords[:, 1] >= 6000000) & (coords[:, 1] <= 7300000))
            pos, coords = pos[good], coords[good]
            delta = np.zeros(len(keys), dtype=float)
            if points:
                distance = np.full(len(coords), np.inf)
                for p in points:
                    distance = np.minimum(distance, np.hypot(coords[:, 0]-p[0], coords[:, 1]-p[1]))
                changed = features.copy()
                changed[pos, 5] = np.maximum(changed[pos, 5], .5/(1+distance/300.))
                changed[:, 6] = changed[:, 0]*changed[:, 5]
                changed[:, 7] = changed[:, 2]*changed[:, 5]
                _, group = np.unique(keys//100000, return_inverse=True)
                parents = np.zeros(int(group.max())+1, dtype=np.float32)
                np.maximum.at(parents, group, changed[:, 5])
                changed[:, 16] = np.maximum(changed[:, 16], parents[group])
                changed[:, 17] = changed[:, 1]*changed[:, 16]
                changed[:, 19] = changed[:, 18]*changed[:, 16]
                weights = np.asarray(model['coefficients'][7:], dtype=float)
                delta = np.maximum(0., (changed.astype(float)-features.astype(float)) @ weights)
                delta[~valid] = 0.
                del changed, group, parents
            if not np.isfinite(delta).all():
                raise ValueError('Finite BANO geometric score increments')
            np.savez_compressed(path, keys=keys, bano_geometry=delta)
            engine.seal(path)
            shards[qid] = record(path)
            io.write(root/(path.stem+'.metadata.json'), dict(query=q, precision=precision,
                source_points=len(points), same_numeric_scope=len(keys), changed_sites=int(np.count_nonzero(delta)),
                no_coordinate_or_address_copied_to_other_endpoint=True, labels_read=False))
        gc.collect()
        io.write(out/'prepare_state.json', dict(status='ALL526_QUERY_ONLY_BANO_SCORE_DELTA',
            queries=len(shards), total=526, updated_unix=time.time()))
    authority(out)
    io.write(out/'prepared.json', dict(status='COMPLETE_ALL526_LABEL_FREE_BANO_SCORES',
        shards=shards, public_mask=launch['public_mask'], context=record(BANO/'query_context.json')))


def generate(out, variant):
    launch = authority(out)
    baseline.bind_runtime(out)
    panel = io.read(verify(launch['panel']))
    prepared, pg = io.read(out/'prepared.json'), io.read(verify(launch['parent_prepared']))
    fg = io.read(out/'features_generation.json')
    cal = io.read(out/'calibration/generation.json')
    assignment = io.read(verify(cal['assignment']))
    models = {k: io.read(verify(v)) for k, v in cal['models'].items()}
    supplier = io.read(verify(io.read(verify(launch['supplier_generation']))['extras']))
    address = io.read(verify(io.read(verify(launch['address_generation']))['extras']))
    rna = io.read(verify(io.read(verify(launch['rna_generation']))['context']))
    banatic = io.read(verify(io.read(verify(launch['banatic_generation']))['context']))
    bg = io.read(verify(launch['baseline_generation']))
    old, old_cost = io.read(verify(bg['outputs'])), io.read(verify(bg['timings']))
    ug = io.read(verify(launch['upstream_primary_generation']))
    previous = io.read(verify(ug['outputs']))['learned_native_ooc']
    previous_cost = io.read(verify(ug['timings']))
    backend = TantivyBackend(io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Same unrestricted national corpus')
    root = out/variant
    (root/'native').mkdir(parents=True, exist_ok=True)
    rows, costs, recomputed = {}, {}, 0
    for q in panel:
        guard()
        qid, fold = q['query_id'], str(assignment[q['query_id']])
        model = models[fold]
        if qid in model['train_queries'] or qid not in model['heldout_queries']:
            raise ValueError('Frozen heldout calibrator')
        path = root/'native'/(hashlib.sha256(qid.encode()).hexdigest()+'.json')
        if engine.sealed(path):
            part = io.read(engine.sealed(path))
        else:
            with np.load(verify(prepared['shards'][qid]), allow_pickle=False) as f:
                keys, score = f['keys'], f['bano_geometry']
            entries, equivalent_upstream = [], False
            if variant == 'all_new_sources_max_bano':
                with np.load(verify(pg['shards'][qid]), allow_pickle=False) as f:
                    if not np.array_equal(keys, f['keys']):
                        raise ValueError('Exactly same full reference keys')
                    previous_score = np.maximum(f['geometry'], f['named'])
                own = upstream.projected(keys, supplier[qid]['own'])
                temporal = upstream.projected(keys, supplier[qid]['temporal'])
                continuity = upstream.projected(keys, address[qid])
                coefficient = float(model['coefficients'][6])
                previous_score = np.maximum(previous_score, coefficient*np.maximum(np.maximum(own, temporal), continuity))
                score = np.maximum(score, previous_score)
                merged = {r['siren_ref']: r['confidence'] for r in rna[qid]}
                for r in banatic[qid]:
                    merged[r['siren_ref']] = max(merged.get(r['siren_ref'], 0.), r['confidence'])
                entries = [dict(siren_ref=r, confidence=c) for r, c in sorted(merged.items())]
                equivalent_upstream = np.array_equal(score, previous_score)
            if equivalent_upstream:
                candidates, cost = previous[qid], dict(previous_cost[qid], unchanged_outputs_reused=True,
                    reused_exact_upstream_inputs=True)
                equivalence = 'EXACT_ALL_LEAF_SCORES_AND_SAME_OWNER_CONTEXTS_V134'
            elif not np.any(score) and not entries:
                candidates, cost = old['learned_native_ooc'][qid], dict(old_cost[qid], unchanged_outputs_reused=True)
                equivalence = 'EXACT_ZERO_ADDED_SCORE_V115'
            else:
                with np.load(verify(fg['features'][qid]), allow_pickle=False) as f:
                    if not np.array_equal(keys, f['keys']):
                        raise ValueError('Unchanged full numerical proof universe')
                    proxy = baseline.baseline.BackendProxy(backend, parent.EvidenceCollector(
                        backend, q, keys, score, entries, float(model['coefficients'][6])))
                    candidates, cost = baseline.native_search(proxy, q, keys, f['features'], model['coefficients'])
                cost.update(unchanged_outputs_reused=False, new_max_leaf_scores=int(np.count_nonzero(score)))
                equivalence = None
            part = dict(query=q, model=cal['models'][fold], candidates=candidates, cost=cost,
                        equivalence=equivalence, labels_read=False)
            io.write(path, part)
            engine.seal(path)
            gc.collect()
        if part['query'] != q or part['model'] != cal['models'][fold] or len(part['candidates']) > 100:
            raise ValueError('Exact resume binding and absolute initial100 cap')
        rows[qid], costs[qid] = part['candidates'], part['cost']
        recomputed += not part['cost']['unchanged_outputs_reused']
        io.write(root/'state.json', dict(status='ALL526_NATIVE_BANO_NO_LABELS', queries=len(rows),
            total=526, recomputed=recomputed, updated_unix=time.time()))
    if len(rows) != 526 or any(len({p['siret'] for p in cs}) != len(cs) for cs in rows.values()):
        raise ValueError('Same all-human population and unique initial100')
    authority(out)
    io.write(root/'outputs.json', dict(old, learned_native_ooc=rows))
    io.write(root/'native_timings.json', costs)
    io.write(root/'generation.json', dict(status='COMPLETE_SEALED_ALL526', queries=526,
        outputs=record(root/'outputs.json'), timings=record(root/'native_timings.json'),
        no_positive_injected=True, no_prepool=True, no_downstream_reranking=True,
        labels_opened_in_native_worker=False, closed_test_opened=False, full_encoder_oof=False,
        recomputed_queries=recomputed, runtime_reuse_not_full_new_slo=True))


def run(out):
    import psutil
    with (out/'owned.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for action, variant in [('prepare', None)]+[('generate', v) for v in VARIANTS]:
            target = out/variant if variant else out
            target.mkdir(exist_ok=True)
            ex_path = target/(action+'_exit.json')
            if ex_path.exists():
                ex = io.read(ex_path)
                if ex['exit_code'] == 0 and ex['own_child_exit_confirmed']:
                    continue
            while True:
                if engine.INTERRUPTED or time.time() >= parent.DEADLINE-30:
                    raise TimeoutError('Hard mandate deadline/user stop; preserve all shards')
                try:
                    authority(out)
                except engine.ResourceYield as error:
                    io.write(out/'supervisor_state.json', dict(status='WAITING_REAL_RESOURCE_ADMISSION',
                        action=action, variant=variant, reason=str(error), updated_unix=time.time()))
                    time.sleep(30)
                    continue
                command = [str(engine.CPU), '-u', '-m', 'scripts.embeddinggemma2_v142_bano_native_ablation',
                           action, '--output', str(out)]
                if variant:
                    command += ['--variant', variant]
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(out/'spill'),
                    RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                reason = None
                with (target/(action+'.log')).open('ab', buffering=0) as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    io.write(target/'owned_process.json', dict(pid=child.pid, command=command,
                        owned=True, started_unix=time.time()))
                    while child.poll() is None:
                        free, level = io.pressure()
                        try:
                            rss = psutil.Process(child.pid).memory_info().rss
                        except psutil.NoSuchProcess:
                            rss = 0
                        if (engine.INTERRUPTED or time.time() >= parent.DEADLINE-30 or free < 25
                                or level not in (0, 1, 2) or rss > 10*1024**3):
                            reason = dict(user_stop=engine.INTERRUPTED,
                                deadline=time.time() >= parent.DEADLINE-30, rss=rss, free=free, level=level)
                            child.terminate()
                            try:
                                child.wait(timeout=25)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait()
                            break
                        time.sleep(5)
                    rc = child.wait()
                io.write(ex_path, dict(exit_code=rc, pid=child.pid, own_child_exit_confirmed=True,
                    resource_or_deadline_reason=reason, finished_unix=time.time()))
                if reason and (reason['user_stop'] or reason['deadline']):
                    raise TimeoutError('Mandate ended/user stop; durable shards preserved')
                if rc == 75 or reason:
                    time.sleep(30)
                    continue
                if rc:
                    raise RuntimeError(f'{action}/{variant} actual owned exit{rc}; no partial labels')
                break
        evaluate(out)


def evaluate(out):
    from scripts import embeddinggemma2_v32_learned_query as evaluator
    authority(out)
    complete_family(out, VARIANTS)
    summary = {}
    old = {r['query_id']: r for r in io.read(baseline.OUTPUT/'outcomes.json')['learned_native_ooc']}
    for variant in VARIANTS:
        root = out/variant
        io.write(root/'launch.json', io.read(out/'launch.json'))
        (root/'calibration').mkdir(exist_ok=True)
        io.write(root/'calibration/generation.json', io.read(out/'calibration/generation.json'))
        evaluator.proof = sys.modules[__name__]
        evaluator.evaluate(root)
        fresh = {r['query_id']: r for r in io.read(root/'outcomes.json')['learned_native_ooc']}
        if set(fresh) != set(old) or any(old[q][f] != fresh[q][f] for q in old
                for f in ('fold', 'zone', 'siren_group', 'calibrated_query_weight')):
            raise ValueError('Same all-human population and evaluation units')
        paired = dict(gains=[q for q in old if fresh[q]['exact'] and not old[q]['exact']],
                      losses=[q for q in old if old[q]['exact'] and not fresh[q]['exact']])
        report = io.read(root/'report.json')
        report.update(paired_vs_v115=paired, eligibility=RECIPE['eligibility'], production_promotion_allowed=False,
            full_encoder_oof=False, no_full_encoder_oof=True, encoder_train_sirens_disjoint=None,
            independent_encoder_generalization_not_validated=True, runtime_contains_equivalent_output_reuse=True,
            no_full_new_cold_warm_slo=True)
        io.write(root/'report.json', report)
        summary[variant] = dict(metrics=report['variants']['learned_native_ooc'], paired=paired,
                               report=record(root/'report.json'))
    io.write(out/'report.json', dict(status='COMPLETE_TWO_PREDECLARED_BANO_ABLATIONS_ALL526',
        primary_predeclared=RECIPE['primary_predeclared'], variants=summary,
        no_holdout_or_national_certification=True, finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'prepare', 'generate', 'run', 'evaluate'])
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, engine.interrupted)
    signal.signal(signal.SIGINT, engine.interrupted)
    try:
        if args.action == 'generate':
            generate(args.output, args.variant)
        else:
            globals()[args.action](args.output)
    except engine.ResourceYield as error:
        print(str(error), flush=True)
        sys.exit(75)
    except Exception as error:
        if args.output.exists():
            io.write(args.output/(args.action+'_error.json'), dict(error=repr(error), updated_unix=time.time()))
        raise
