"""Owned, resumable ALL526 site-target run; ledger remains closed until sealing."""
import argparse
import fcntl
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from scripts import retrieval_site_target_timing as timing
from scripts.retrieval_site_operational_metric import read, record, verified, fingerprint, index
from scripts.retrieval_site_target_mapping import contract_commit, RESULT as MAPPING

H = timing.H
OUT = H/'v173/site_target_full_20261015'
TIMING = timing.OUT
PREPARED = timing.PREPARED
RESULT = ROOT/'docs/retrieval_site_target_full_result_20261016.json'
STAGES = ('v32', 'v71')
STOP = False
GIB = 1024**3


def atomic_json(path, obj):
    """Immutable publication; interrupted temporary files are retained, never replaced."""
    path = Path(path)
    if path.exists():
        if read(path) != obj:
            raise ValueError('Refusing to replace '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)  # Atomic no-clobber link. The pending hardlink remains for provenance.


def stamp():
    return time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]


def storage():
    stat = shutil.disk_usage(H)
    allocated = 0
    if OUT.exists():
        seen = set()
        for p in OUT.rglob('*'):
            if p.is_file():
                s = p.stat()
                if (s.st_dev, s.st_ino) not in seen:
                    allocated += s.st_blocks*512
                    seen.add((s.st_dev, s.st_ino))
    return dict(free_bytes=stat.free, namespace_allocated_bytes=allocated)


def hydraulic():
    env = dict(os.environ, LIMA_HOME='/Volumes/CATNAT_DATA/catnat-lima')
    proc = None
    observed = None
    try:
        proc = subprocess.run(['limactl', 'list', '--json'], env=env, capture_output=True,
                              text=True, check=True, timeout=10)
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        row = next((r for r in rows if isinstance(r, dict) and r.get('name') == 'hydraulic'), None)
        observed = row.get('status') if row else None
        if observed not in ('Running', 'Stopped'):
            raise ValueError(f'hydraulic state unknown: {observed!r}')
        return dict(name='hydraulic', status=observed, lima_home=env['LIMA_HOME'])
    except Exception as error:
        # An uncertain health probe must yield compute, never terminate the run.
        print(json.dumps(dict(phase='HYDRAULIC_PROBE_UNCERTAIN_ASSUME_RUNNING',
            observed_unix=time.time(), observed_status=observed, error=repr(error),
            stdout=(proc.stdout or '')[:1024] if proc else None,
            stderr=(proc.stderr or '')[:1024] if proc else None,
            retry_seconds=60)), flush=True)
        return dict(name='hydraulic', status='Running', observed_status=observed,
                    uncertain=True, lima_home=env['LIMA_HOME'])


def free_inactive_memory():
    """Admission evidence only: free + inactive pages, excluding speculative pages."""
    try:
        proc = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True, timeout=10)
        page_size = int(re.search(r'page size of (\d+) bytes', proc.stdout)[1])
        free = int(re.search(r'^Pages free:\s+(\d+)\.', proc.stdout, re.M)[1])
        inactive = int(re.search(r'^Pages inactive:\s+(\d+)\.', proc.stdout, re.M)[1])
        return dict(observed_unix=time.time(), free_inactive_bytes=(free+inactive)*page_size,
                    page_size_bytes=page_size, free_pages=free, inactive_pages=inactive, vm_stat=proc.stdout)
    except Exception as error:
        return dict(observed_unix=time.time(), free_inactive_bytes=None, error=repr(error))


def generation_memory_limit(qid):
    """Persist the single OOM escalation across controller restarts, including old receipts."""
    first_failure = None
    for path in sorted((OUT/'attempts').glob('*/receipt.json')):
        receipt = read(path)
        command = receipt.get('command', [])
        if '--worker' not in command or '--query-id' not in command:
            continue
        if command[command.index('--worker')+1] != 'generate' or command[command.index('--query-id')+1] != qid:
            continue
        if receipt.get('stop_reason') == 'STOP_RSS_13_GIB':
            raise RuntimeError(f'Generate {qid}: single 13 GiB retry already exceeded its cap; preserve all outputs and STOP')
        if receipt.get('stop_reason') == 'STOP_RSS_8_GIB' and first_failure is None:
            first_failure = record(path)
    return (13*GIB, first_failure) if first_failure else (8*GIB, None)


def snapshots(phase, **extra):
    obj = dict(phase=phase, observed_unix=time.time(), storage=storage(), **extra)
    atomic_json(OUT/'progress'/(stamp()+'.json'), obj)
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def plan_run():
    if (OUT/'plan.json').exists():
        plan = read(OUT/'plan.json')
        if record(Path(__file__))['sha256'] == plan['implementation']['sha256']:
            verified(plan['implementation'])
        else:
            # Preserve the original plan and accept only the explicitly sealed
            # VM-guard correction authorized by the follow-up execution brief.
            revision_path = OUT/'supervisor_vm_guard_revision.json'
            revision = read(revision_path)
            if revision['previous_implementation'] != plan['implementation'] or Path(revision['implementation']['path']).resolve() != Path(__file__).resolve():
                raise ValueError('Unauthorized supervisor revision')
            record(revision['original_source']['path'], plan['implementation']['sha256'])
            implementation = revision['implementation']
            if record(Path(__file__))['sha256'] != implementation['sha256']:
                memory_revision_path = OUT/'supervisor_generate_memory_revision.json'
                memory_revision = read(memory_revision_path)
                if memory_revision['previous_implementation'] != implementation or Path(memory_revision['implementation']['path']).resolve() != Path(__file__).resolve():
                    raise ValueError('Unauthorized generate memory revision')
                record(memory_revision['original_source']['path'], implementation['sha256'])
                implementation = memory_revision['implementation']
                snapshots('RESUME_WITH_AUTHORIZED_GENERATE_MEMORY_REVISION', revision=record(memory_revision_path))
            verified(implementation)
            snapshots('RESUME_WITH_AUTHORIZED_VM_GUARD_REVISION', revision=record(revision_path))
        for ref in (plan['timing_implementation'], plan['evaluation_implementation'], plan['mapping'], plan['assignment'], plan['panel']):
            verified(ref)
        return plan
    if shutil.disk_usage(H).free < 10*GIB:
        raise OSError('STOP_DISK_BELOW_10_GIB')
    contract = contract_commit()
    inputs = timing.Inputs()
    launch = inputs.json(PREPARED/'launch.json')
    panel = index(inputs.json(launch['panel']))
    cal = inputs.json(timing.V71/'calibration/generation.json')
    assignment = inputs.json(cal['assignment'])
    old_models = {str(k): inputs.json(cal['models'][str(k)]) for k in range(5)}
    # Presealed training supervision is necessary for learning. No raw ledger,
    # acceptable sidecar, outcomes or site/evaluation results are opened here.
    mapping = index(inputs.json(MAPPING)['per_query'])
    if len(panel) != 526 or set(panel) != set(mapping) or set(panel) != set(assignment):
        raise ValueError('ALL526 scope drift')
    if any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('Closed fold')
    for k, m in old_models.items():
        train, held = m['train_queries'], m['heldout_queries']
        if any(assignment[q] == int(k) for q in train) or any(assignment[q] != int(k) for q in held):
            raise ValueError('Historical internal assignment drift')
        if {mapping[q]['training_target_siret'][:9] for q in train} & {mapping[q]['training_target_siret'][:9] for q in held}:
            raise ValueError('Training company overlap')
    eligible = sorted(set().union(*(set(m['train_queries']) for m in old_models.values())))
    pairs = {stage: {} for stage in STAGES}
    for stage in STAGES:
        part = inputs.json(TIMING/stage/'result.json')
        receipt = inputs.json(TIMING/(stage+'_receipt.json'))
        model = inputs.json(TIMING/stage/'model_0.json')
        if receipt['exit_code'] != 0 or not receipt['owned_child_exit_confirmed'] or model['train_queries'] != old_models['0']['train_queries']:
            raise ValueError('Timing model0 provenance')
        verified(model['training_target_mapping'])
        if model['training_target_mapping']['sha256'] != record(MAPPING)['sha256']:
            raise ValueError('Training target mapping differs')
        for ref in part['inputs']:
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('Timing source mutated')
        for qid in model['train_queries']:
            path = TIMING/stage/'pairs'/(timing.stem(qid)+'.npz')
            with np.load(path, allow_pickle=False) as f:
                if int(f['candidates'][0]) != int(mapping[qid]['training_target_siret']) or not np.isfinite(f['difference']).all():
                    raise ValueError('Timing pair target mismatch')
            pairs[stage][qid] = record(path)
    index_root = Path(launch['index_meta']['path']).parent
    inventory = sorted([p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in index_root.iterdir() if p.is_file())
    if inventory != launch['index_inventory']:
        raise ValueError('Index inventory drift')
    atomic_json(OUT/'panel.json', list(panel.values()))
    atomic_json(OUT/'assignment.json', assignment)
    atomic_json(OUT/'old_partitions.json', old_models)
    atomic_json(OUT/'pair_sources.json', pairs)
    initial = dict(started_unix=time.time(), storage=storage(), hydraulic=hydraulic())
    plan = dict(contract_commit=contract, authorization='Nathan, user brief sireto-site-target-2, evening 2026-10-15',
        implementation=record(Path(__file__)), evaluation_implementation=record(ROOT/'scripts/retrieval_site_target_full_evaluate.py'),
        timing_implementation=record(Path(timing.__file__)), mapping=record(MAPPING),
        panel=record(OUT/'panel.json'), assignment=record(OUT/'assignment.json'),
        partitions=record(OUT/'old_partitions.json'), pair_sources=record(OUT/'pair_sources.json'), eligible=eligible,
        initial=initial, index_root=str(index_root), index_inventory=inventory,
        memory_cap_bytes=8*GIB, minimum_free_disk_bytes=10*GIB, no_artificial_deadline=True,
        no_raw_evaluation_labels_before_complete_generation=True, sources=list(inputs.refs.values()),
        closed_sets_opened=False, generation_model_selection='UNCHANGED_V142_PRESEALED_SIREN_BASED_ASSIGNMENT_NOT_PRODUCTION_ROUTING')
    inputs.unchanged()
    atomic_json(OUT/'plan.json', plan)
    return plan


def check_inventory(plan):
    root = Path(plan['index_root'])
    if sorted([p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in root.iterdir() if p.is_file()) != plan['index_inventory']:
        raise ValueError('Index inventory changed')


def pair_file(stage, qid, plan):
    old = read(verified(plan['pair_sources']))[stage]
    if qid in old:
        return verified(old[qid])
    done = read(OUT/'pairs'/stage/(timing.stem(qid)+'.json'))
    return verified(done['pair'])


def mine_pair(stage, qid, check=False):
    """Same mining arithmetic as timing.training; process isolation only changes RSS."""
    from scripts import embeddinggemma2_v32_learned_query as learned
    from scripts import embeddinggemma2_v32_native_features as base
    from scripts import embeddinggemma2_v71_frozen_residual as residual
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    started = time.monotonic()
    plan = read(OUT/'plan.json')
    completed = (OUT/'checks'/(stage+'_'+timing.stem(qid)+'.json') if check else OUT/'pairs'/stage/(timing.stem(qid)+'.json'))
    if completed.exists():
        saved = read(completed)
        if not check:
            verified(saved['pair'])
        for ref in saved['inputs']:
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('Completed mining checkpoint source changed')
        return
    inputs = timing.Inputs()
    mapping = index(inputs.json(plan['mapping'])['per_query'])
    panel = index(inputs.json(plan['panel']))
    source, proof = (timing.V32, base) if stage == 'v32' else (timing.V71, residual)
    launch = inputs.json(source/'launch.json')
    for ref in [launch['implementation'], *launch['helpers']]:
        inputs.path(ref)
    gen = inputs.json(source/'features_generation.json')
    for name in ('published_parent_names', 'corporate_names'):
        if name in gen:
            inputs.path(gen[name])
    proof.bind_runtime(source)
    proof.authority = lambda out: launch
    if stage == 'v71':
        residual.corporate.base.authority = lambda out: launch
    learned.proof = proof
    old = inputs.json(inputs.json(proof.CACHE/'launch.json')['baseline_outputs'])
    sampling = inputs.json(inputs.json(launch['source_launch'])['sampling'])
    q = panel[qid]
    if q['fold'] not in (2, 3, 4) or qid not in plan['eligible']:
        raise ValueError('Unsafe training scope')
    backend = TantivyBackend(learned.io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Index count drift')
    with np.load(inputs.path(gen['features'][qid]), allow_pickle=False) as f:
        keys, x = f['keys'], f['features']
    gt = int(mapping[qid]['training_target_siret'])
    selected = set()
    parts = proof.lexical_parts(backend, q)
    tv = backend._tantivy
    if parts[-1] is not None:
        leaf = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
        components = [(tv.Occur.Should, parts[-1])]
        if parts[4] is not None:
            components.append((tv.Occur.Should, parts[4]))
        hard = tv.Query.boolean_query([(tv.Occur.Must, leaf), (tv.Occur.Must, tv.Query.boolean_query(components))])
        for _, loc in backend.searcher.search(hard, limit=101, count=False).hits:
            selected.add(int(backend._first(backend.searcher.doc(loc).to_dict(), 'siret')))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(('training-negatives|'+qid).encode()).digest()[:8], 'big'))
    if len(keys):
        k = min(12, len(keys))
        for col in range(x.shape[1]):
            if x[:, col].max() > 0:
                selected.update(map(int, keys[np.argpartition(x[:, col], len(keys)-k)[-k:]]))
        selected.update(map(int, keys[rng.choice(len(keys), size=min(32, len(keys)), replace=False)]))
    selected.update(int(r['siret']) for r in old['lexical'][qid])
    selected.discard(gt)
    candidates = np.array([gt]+sorted(selected), dtype=np.uint64)
    numeric = np.zeros((len(candidates), len(proof.PROOF_FIELDS)), dtype=np.float32)
    loc = np.searchsorted(keys, candidates)
    inside = loc < len(keys)
    inside[inside] &= keys[loc[inside]] == candidates[inside]
    numeric[inside] = x[loc[inside]]
    lex = learned.masked_lexical_features(backend, q, candidates)
    matrix = np.column_stack((lex, numeric))
    difference = matrix[0:1]-matrix[1:]
    if not len(difference) or not np.isfinite(matrix).all():
        raise ValueError('Invalid training pairs')
    weight = sampling['zone_weights'][q['crm_insee']]['calibrated_query_weight']
    pair_weights = np.full(len(difference), weight/len(difference), dtype=np.float64)
    inputs.unchanged()
    obj = dict(query_id=qid, stage=stage, numeric_target_present=bool(inside[0]),
        pair_count=len(difference), elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, inputs=list(inputs.refs.values()))
    if check:
        with np.load(pair_file(stage, qid, plan), allow_pickle=False) as f:
            if not all(np.array_equal(f[k], v) for k, v in dict(difference=difference, weights=pair_weights, candidates=candidates).items()):
                raise ValueError('Timing mining reproduction differs')
        obj['bitwise_equal_to_timing'] = True
        atomic_json(OUT/'checks'/(stage+'_'+timing.stem(qid)+'.json'), obj)
    else:
        target = OUT/'pairs'/stage/(timing.stem(qid)+'.npz')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with np.load(target, allow_pickle=False) as f:
                if not all(np.array_equal(f[k], v) for k, v in dict(difference=difference, weights=pair_weights, candidates=candidates).items()):
                    raise ValueError('Existing pair differs; preserve and STOP')
        else:
            temp = target.with_name(target.name+'.pending.'+uuid.uuid4().hex)
            with temp.open('xb') as stream:
                np.savez(stream, difference=difference, weights=pair_weights, candidates=candidates)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temp, target)
        obj['pair'] = record(target)
        atomic_json(target.with_suffix('.json'), obj)


def fit_models(fold):
    from scripts import embeddinggemma2_v32_learned_query as learned
    from scripts import embeddinggemma2_v32_native_features as base
    from scripts import embeddinggemma2_v71_frozen_residual as residual
    plan = read(OUT/'plan.json')
    completed = OUT/'models'/f'fold_{fold}_done.json'
    if completed.exists():
        for ref in read(completed)['model_refs'].values():
            verified(ref)
        return
    old = read(verified(plan['partitions']))[str(fold)]
    train = old['train_queries']
    times = {}
    for stage, proof in [('v32', base), ('v71', residual)]:
        started = time.monotonic()
        ds, ws, pair_refs = [], [], []
        for qid in train:
            path = pair_file(stage, qid, plan)
            pair_refs.append(record(path))
            with np.load(path, allow_pickle=False) as f:
                ds.append(f['difference'])
                ws.append(f['weights'])
        d, w = np.concatenate(ds), np.concatenate(ws)
        if stage == 'v32':
            coefficients, info = learned.fit_pairwise(d, w, .01)
        else:
            # Invoke the original residual objective, changing only its implicit
            # "smallest missing fold" selector to the explicit requested fold.
            source = inspect.getsource(residual.fit_residual)
            if source.count('fold=min(missing)') != 1:
                raise ValueError('Residual selector source drift')
            source = source.replace('fold=min(missing)', 'fold=FOLD_ID')
            refs = {str(fold): record(OUT/'models/v32'/f'model_{fold}.json')}
            ns = dict(vars(residual), FOLD_ID=fold, ACTIVE=OUT/'unused_residual_runtime', authority=lambda out: dict(frozen_models=refs))
            exec(compile(source, str(Path(__file__))+':original_residual_explicit_fold', 'exec'), ns)
            coefficients, info = ns['fit_residual'](d, w, .01)
            baseline = read(OUT/'models/v32'/f'model_{fold}.json')['coefficients']
            if not np.array_equal(coefficients[residual.BASE_AT], baseline):
                raise ValueError('V32 baseline changed during residual fit')
        reused = None
        if fold == 0:
            previous = TIMING/stage/'model_0.json'
            m = read(previous)
            if not np.array_equal(coefficients, m['coefficients']):
                raise ValueError('Model0 not strictly identical; STOP before reuse')
            reused = record(previous)
        model = dict(feature_fields=proof.FEATURE_FIELDS, coefficients=coefficients.tolist(), **info,
            train_queries=train, heldout_queries=old['heldout_queries'], heldout_fold=fold, no_group_overlap=True,
            training_target_mapping=plan['mapping'], original_partition=record(timing.V71/'calibration'/f'model_{fold}.json'),
            timing_model0_bitwise_identical=reused, pair_sources=pair_refs)
        path = OUT/'models'/stage/f'model_{fold}.json'
        if path.exists():
            saved = read(path)
            if saved != model:
                raise ValueError('Model differs on resume')
        else:
            atomic_json(path, model)
        times[stage] = time.monotonic()-started
        del d, w, ds, ws
        gc.collect()
    atomic_json(OUT/'models'/f'fold_{fold}_done.json', dict(fold=fold, elapsed_seconds=times,
        model_refs={s: record(OUT/'models'/s/f'model_{fold}.json') for s in STAGES},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def prepare_queries():
    import ijson
    if (OUT/'generation_environment.json').exists():
        for ref in read(OUT/'generation_environment.json')['inputs']:
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('Prepared generation source changed')
        return
    inputs = timing.Inputs()
    plan = read(OUT/'plan.json')
    launch = inputs.json(PREPARED/'launch.json')
    fg = inputs.json(PREPARED/'features_generation.json')
    prepared = inputs.json(PREPARED/'prepared.json')
    pg = inputs.json(launch['parent_prepared'])
    panel = index(inputs.json(plan['panel']))
    assignment = inputs.json(plan['assignment'])
    refs = dict(context=fg['context_expansion'], initials=fg['initial_alignment'],
        supplier=inputs.json(launch['supplier_generation'])['extras'], address=inputs.json(launch['address_generation'])['extras'],
        rna=inputs.json(launch['rna_generation'])['context'], banatic=inputs.json(launch['banatic_generation'])['context'],
        bano_context=prepared['context'], osm_context=pg['source_context'], shapes=pg['shapes'])
    for name, ref in refs.items():
        found = set()
        with inputs.path(ref).open('rb') as stream:
            for qid, value in ijson.kvitems(stream, '', use_float=True):
                if qid not in panel or qid in found:
                    raise ValueError('Unexpected context scope')
                atomic_json(OUT/'queries'/timing.stem(qid)/(name+'.json'), value)
                found.add(qid)
        if found != set(panel):
            raise ValueError('Incomplete context '+name)
        print('PREPARED', name, len(found), flush=True)
    geometries = index(inputs.json(launch['geometry']))
    for qid, q in panel.items():
        atomic_json(OUT/'queries'/timing.stem(qid)/'binding.json', dict(query=q, model_id=assignment[qid],
            features=fg['features'][qid], geometry=geometries[qid]['geometry']))
    inputs.unchanged()
    atomic_json(OUT/'generation_environment.json', dict(launch=launch, feature_generation=fg, inputs=list(inputs.refs.values())))


def generate_one(qid):
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    from scripts.retrieval_miss_score_explain import preparer_kernel
    started = time.monotonic()
    inputs = timing.Inputs()
    directory = OUT/'queries'/timing.stem(qid)
    binding = inputs.json(directory/'binding.json')
    if (directory/'result.json').exists():
        saved = read(directory/'result.json')
        verified(saved['model'])
        if saved['query'] != binding['query'] or saved['model_id'] != binding['model_id'] or saved['raw_labels_read']:
            raise ValueError('Generation checkpoint mismatch')
        for ref in saved['inputs']:
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('Generation checkpoint input mutated')
        return
    environment = inputs.json(OUT/'generation_environment.json')
    q, launch, fg = binding['query'], environment['launch'], environment['feature_generation']
    if q['fold'] not in (2, 3, 4):
        raise ValueError('Forbidden fold')
    model_ref = record(OUT/'models/v71'/f'model_{binding["model_id"]}.json')
    model = inputs.json(model_ref)
    if qid not in model['heldout_queries'] or qid in model['train_queries']:
        raise ValueError('Must use historical heldout model assignment')
    v115 = v142.baseline
    v99 = v115.baseline
    v81 = v99.source
    base = v81.pipeline.base
    v115.bind_runtime(PREPARED)
    base.authority = lambda out: launch
    v99.CONTEXT = {qid: read(directory/'context.json')}
    v99.INITIAL_DATA = {qid: read(directory/'initials.json')}
    v81.INDIRECT = inputs.json(fg['indirect_names'])
    for name in ('corporate_names', 'published_parent_names'):
        inputs.path(fg[name])
    inputs.path(fg['activity_shards'][qid])
    for ref in [launch['implementation'], *launch['helpers'], launch['index_manifest'], launch['index_meta']]:
        inputs.path(ref)
    w = np.asarray(model['coefficients'], dtype=float)
    backend = v142.TantivyBackend(v142.io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('National index count drift')
    with np.load(inputs.path(binding['features']), allow_pickle=False) as f:
        keys, features = f['keys'], f['features']
    all_keys = np.load(inputs.path(launch['keys']), mmap_mode='r')
    xy = np.load(inputs.path(launch['xy']), mmap_mode='r')
    public = np.load(inputs.path(launch['public_mask']), mmap_mode='r')
    bano, bsha = preparer_kernel(v142, 'bano')
    osm, osha = preparer_kernel(v142, 'osm')
    points, _ = v142.compatible_points(read(directory/'bano_context.json'))
    delta = bano(keys, features, all_keys, xy, public, points, model)
    geom, named = osm(keys, features, all_keys, xy, public, w, {qid: binding['geometry']},
        {qid: read(directory/'shapes.json')}, qid, q, read(directory/'osm_context.json'))
    supplier = read(directory/'supplier.json')
    own = v142.upstream.projected(keys, supplier['own'])
    temporal = v142.upstream.projected(keys, supplier['temporal'])
    continuity = v142.upstream.projected(keys, read(directory/'address.json'))
    scores = np.maximum(delta, np.maximum(np.maximum(geom, named), w[6]*np.maximum(np.maximum(own, temporal), continuity)))
    merged = {r['siren_ref']: r['confidence'] for r in read(directory/'rna.json')}
    for r in read(directory/'banatic.json'):
        merged[r['siren_ref']] = max(merged.get(r['siren_ref'], 0.), r['confidence'])
    entries = [dict(siren_ref=k, confidence=v) for k, v in sorted(merged.items())]
    proxy = v99.BackendProxy(backend, v142.parent.EvidenceCollector(backend, q, keys, scores, entries, float(w[6])))
    rows, cost = v115.native_search(proxy, q, keys, features, w)
    if len(rows) > 100 or len({r['siret'] for r in rows}) != len(rows):
        raise ValueError('Cap/uniqueness violation')
    inputs.unchanged()
    obj = dict(query=q, candidates=rows, cost=cost, model=model_ref, model_id=binding['model_id'],
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        raw_labels_read=False, no_recall=True, source_kernels=dict(bano=bsha, osm=osha), inputs=list(inputs.refs.values()))
    path = directory/'result.json'
    if path.exists():
        old = read(path)
        if old['candidates'] != rows or old['model'] != model_ref or old['query'] != q:
            raise ValueError('Incomplete prior attempt differs on replay; preserve and STOP')
    else:
        atomic_json(path, obj)


def job_path(action, stage=None, qid=None, fold=None):
    tag = action+('_'+stage if stage else '')+('_'+timing.stem(qid) if qid else '')+('_'+str(fold) if fold is not None else '')
    return OUT/'receipts'/(tag+'.json')


def work_ref(action, stage=None, qid=None, fold=None):
    if action == 'pair':
        return OUT/'pairs'/stage/(timing.stem(qid)+'.json')
    if action == 'check_pair':
        return OUT/'checks'/(stage+'_'+timing.stem(qid)+'.json')
    if action == 'fit':
        return OUT/'models'/f'fold_{fold}_done.json'
    if action == 'prepare':
        return OUT/'generation_environment.json'
    if action == 'generate':
        return OUT/'queries'/timing.stem(qid)/'result.json'
    if action == 'evaluate':
        return OUT/'evaluation_done.json'
    raise ValueError(action)


def signal_stop(signum, frame):
    global STOP
    STOP = True


def terminate(child):
    child.terminate()
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def supervised(action, stage=None, qid=None, fold=None):
    dest = job_path(action, stage, qid, fold)
    artifact = work_ref(action, stage, qid, fold)
    if dest.exists():
        done = read(dest)
        if done['exit_code'] != 0 or not done['owned_child_exit_confirmed']:
            raise ValueError('Invalid completed receipt')
        verified(done['artifact'])
        return done
    memory_cap, first_8gib_failure = generation_memory_limit(qid) if action == 'generate' else (8*GIB, None)
    while True:
        if STOP:
            raise InterruptedError('USER_INTERRUPT_CHECKPOINTS_RETAINED')
        if shutil.disk_usage(H).free < 10*GIB:
            raise OSError('STOP_DISK_BELOW_10_GIB')
        vm = hydraulic()
        if vm['status'] == 'Running':
            snapshots('WAITING_HYDRAULIC', pending_action=action, query_id=qid)
            time.sleep(60)
            continue
        admission = None
        if memory_cap == 13*GIB:
            admission = dict(memory=free_inactive_memory(), hydraulic=vm, required_free_inactive_bytes=15*GIB)
            available = admission['memory']['free_inactive_bytes']
            if available is None or available < 15*GIB:
                snapshots('WAITING_GENERATE_13_GIB_ADMISSION', query_id=qid, admission=admission,
                          first_8gib_failure=first_8gib_failure, retry_seconds=60)
                time.sleep(60)
                continue
            snapshots('ADMITTED_GENERATE_SINGLE_13_GIB_RETRY', query_id=qid, admission=admission,
                      first_8gib_failure=first_8gib_failure)
        attempt = OUT/'attempts'/stamp()
        attempt.mkdir(parents=True)
        command = [sys.executable, '-u', str(Path(__file__)), '--worker', action]
        for key, value in (('--stage', stage), ('--query-id', qid), ('--fold', fold)):
            if value is not None:
                command += [key, str(value)]
        started = time.monotonic()
        peak = 0
        reason = None
        next_check = 0.
        min_free = shutil.disk_usage(H).free
        with (attempt/'worker.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    try:
                        peak = max(peak, psutil.Process(child.pid).memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    if peak >= memory_cap:
                        reason = 'STOP_RSS_13_GIB' if memory_cap == 13*GIB else 'STOP_RSS_8_GIB'
                        terminate(child)
                        break
                    if STOP:
                        reason = 'USER_INTERRUPT'
                    if time.monotonic() >= next_check:
                        min_free = min(min_free, shutil.disk_usage(H).free)
                        if min_free < 10*GIB:
                            reason = 'STOP_DISK_BELOW_10_GIB'
                        elif hydraulic()['status'] == 'Running':
                            reason = 'YIELD_TO_HYDRAULIC'
                        next_check = time.monotonic()+3
                    if reason:
                        terminate(child)
                        break
                    time.sleep(.1)
                rc = child.wait()
            finally:
                if child.poll() is None:
                    terminate(child)
        receipt = dict(command=command, pid=child.pid, exit_code=rc, owned_child_exit_confirmed=True,
            started_unix=time.time()-(time.monotonic()-started), elapsed_seconds=time.monotonic()-started,
            peak_sampled_rss_bytes=peak, minimum_free_bytes=min_free, stop_reason=reason, log=str(attempt/'worker.log'),
            memory_cap_bytes=memory_cap, memory_admission=admission)
        atomic_json(attempt/'receipt.json', receipt)
        if action == 'generate' and reason == 'STOP_RSS_8_GIB':
            first_8gib_failure = record(attempt/'receipt.json')
            memory_cap = 13*GIB
            snapshots('GENERATE_RSS8_SINGLE_ESCALATION_PENDING', query_id=qid, first_8gib_failure=first_8gib_failure)
            continue
        if reason == 'YIELD_TO_HYDRAULIC':
            snapshots('YIELDED_HYDRAULIC_CHILD_MEMORY_RELEASED', action=action, query_id=qid, receipt=record(attempt/'receipt.json'))
            time.sleep(60)
            continue
        if rc or reason:
            raise RuntimeError(str(receipt))
        if not artifact.exists():
            raise ValueError('Exit0 without expected artifact')
        receipt['artifact'] = record(artifact)
        atomic_json(dest, receipt)
        return receipt


def seal_generation(plan):
    panel = index(read(verified(plan['panel'])))
    assignment = read(verified(plan['assignment']))
    outputs, receipts = {}, {}
    for qid, q in panel.items():
        rc = read(job_path('generate', qid=qid))
        if rc['exit_code'] != 0 or not rc['owned_child_exit_confirmed']:
            raise ValueError('All526 owned exits0 before labels')
        part = read(verified(rc['artifact']))
        if part['query'] != q or part['model_id'] != assignment[qid] or part['raw_labels_read']:
            raise ValueError('Query/model binding drift')
        verified(part['model'])
        cs = part['candidates']
        if len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs):
            raise ValueError('Invalid output cap')
        for c in cs:
            if len(c['siret']) != 14 or not c['siret'].isdigit() or c['siren'] != c['siret'][:9] or not np.isfinite(c['score']):
                raise ValueError('Malformed output')
        outputs[qid] = cs
        receipts[qid] = record(job_path('generate', qid=qid))
    if len(outputs) != 526:
        raise ValueError('No partial evaluation')
    check_inventory(plan)
    atomic_json(OUT/'outputs.json', outputs)
    atomic_json(OUT/'generation.json', dict(status='COMPLETE_SEALED_ALL526_EXIT0_BEFORE_LABELS', queries=526,
        outputs=record(OUT/'outputs.json'), assignment=plan['assignment'], panel=plan['panel'], receipts=receipts,
        sealed_unix=time.time(), no_positive_injected=True, no_downstream_reranking=True, closed_sets_opened=False))


def progress_status():
    plan = read(OUT/'plan.json')
    paths = list((OUT/'receipts').glob('generate_*.json'))
    seconds = [read(p)['elapsed_seconds'] for p in paths]
    n = len(paths)
    age = time.time()-plan['initial']['started_unix']
    pairs = read(verified(plan['pair_sources']))
    pending_pairs = sum(not job_path('pair', stage=s, qid=q).exists()
        for s in STAGES for q in plan['eligible'] if q not in pairs[s])
    pair_receipts = [read(p) for p in (OUT/'receipts').glob('pair_*.json')]
    fit_done = len(list((OUT/'receipts').glob('fit_*.json')))
    eta = (526-n)*sum(seconds)/n if n else None
    mining_eta = pending_pairs*sum(r['elapsed_seconds'] for r in pair_receipts)/len(pair_receipts) if pair_receipts else None
    return dict(generated_queries=n, total_queries=526, learned_models=fit_done, remaining_pair_jobs=pending_pairs,
        elapsed_seconds=age, generation_eta_seconds=eta, mining_eta_seconds=mining_eta,
        total_eta_seconds=eta+mining_eta if eta is not None and mining_eta is not None else None,
        eta_excludes_future_hydraulic_waits=True, eta_excludes_memory_admission_waits=True,
        storage=storage(), hydraulic=hydraulic(),
        evaluation_complete=(OUT/'evaluation_done.json').exists(), commit_complete=(OUT/'commit.json').exists())


def commit_results():
    if (OUT/'commit.json').exists():
        return
    # Never include foreign staged changes in the owned commit.
    paths = ['scripts/retrieval_site_target_full.py', 'scripts/retrieval_site_target_full_evaluate.py',
             str(RESULT.relative_to(ROOT)), str(RESULT.with_suffix('.md').relative_to(ROOT)), 'handover.md']
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
    if staged:
        raise ValueError('Commit blocked: preexisting staged changes; no index mutation')
    result = read(RESULT)
    exact = result['totals']['exact']
    marker = '- 2026-10-16 — Recalcul complet cible site V173'
    handover = ROOT/'handover.md'
    if marker not in handover.read_text():
        if subprocess.run(['git', 'diff', '--quiet', '--', 'handover.md'], cwd=ROOT).returncode:
            raise ValueError('Commit blocked: foreign handover changes, results preserved')
        line = (f'{marker} : contrat `996ce66b`, autorisation Nathan 15/10 soir ; '
            f'`scripts/retrieval_site_target_full{{,_evaluate}}.py`, `docs/retrieval_site_target_full_result_20261016.{{json,md}}` ; '
            f'commit isolé auto-référencé `git log -1 --format=%H -- scripts/retrieval_site_target_full.py` ; '
            f'5 V32 + 5 résiduels V71, modèle0 identique bit à bit, mêmes partitions/affectation V142, '
            f'526/526 reçus exit0 avant évaluation séparée ; exact {exact["new"]["hits"]}/526 vs {exact["v142"]["hits"]}, '
            f'+{exact["paired"]["gained"]}/−{exact["paired"]["lost"]}, site {result["totals"]["site_operational"]["new"]["hits"]}, '
            f'union {result["totals"]["union"]["new"]["hits"]} ; coût {result["resources"]["wall_seconds"]:.1f}s, '
            f'pic {result["resources"]["peak_rss_bytes"]/GIB:.3f}Gio, espace propre {result["resources"]["namespace_allocated_bytes"]/GIB:.3f}Gio ; '
            'checkpoints par requête, attente hydraulic sans action VM, garde8Gio/disque10Gio, aucun délai09h artificiel ; '
            'labels immuables, jeux fermés/web/aval/push absents, résultats développement non certification, travaux étrangers préservés.\n')
        with handover.open('a') as stream:
            stream.write(line)
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'feat: complete site-target five-model retrieval and paired evaluation'], cwd=ROOT, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    atomic_json(OUT/'commit.json', dict(commit=sha, paths=paths, push=False))


def run(commit=False):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT/'owned.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = plan_run()
        snapshots('START_OR_RESUME', generated_queries=len(list((OUT/'receipts').glob('generate_*.json'))))
        pairs = read(verified(plan['pair_sources']))
        mapping = index(read(verified(plan['mapping']))['per_query'])
        # Actual smoke: unchanged and changed target, both feature schemas.
        smoke = [sorted(pairs['v32'])[0], next(q for q in sorted(pairs['v32']) if mapping[q]['changed'])]
        for stage in STAGES:
            for qid in smoke:
                supervised('check_pair', stage=stage, qid=qid)
        supervised('fit', fold=0)  # Refit proves strict identity before model0 reuse.
        supervised('prepare')
        partition0 = read(verified(plan['partitions']))['0']['heldout_queries']
        supervised('generate', qid=sorted(partition0)[0])  # Real native smoke, no labels or metric.
        snapshots('SMOKE_COMPLETE', generated_queries=1, model0_bitwise_identity=True, mining_checks=4)
        for stage in STAGES:
            for qid in plan['eligible']:
                if qid not in pairs[stage]:
                    supervised('pair', stage=stage, qid=qid)
                    snapshots('MINING', stage=stage, query_id=qid)
        for fold in range(1, 5):
            supervised('fit', fold=fold)
            snapshots('MODELS', internal_models=fold+1)
        panel = read(verified(plan['panel']))
        for q in panel:
            supervised('generate', qid=q['query_id'])
            n = len(list((OUT/'receipts').glob('generate_*.json')))
            snapshots('GENERATING', generated_queries=n, total_queries=526)
        if not (OUT/'generation.json').exists():
            seal_generation(plan)
        snapshots('ALL526_SEALED_BEFORE_LABELS', generated_queries=526)
        supervised('evaluate')
        snapshots('COMPLETE', generated_queries=526)
        if commit:
            commit_results()
    except BaseException as error:
        snapshots('STOP_CHECKPOINTS_RETAINED_NO_PARTIAL_EVALUATION', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['pair', 'check_pair', 'fit', 'prepare', 'generate', 'evaluate'])
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--query-id')
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--commit-on-complete', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(progress_status(), indent=2))
    elif args.worker in ('pair', 'check_pair'):
        mine_pair(args.stage, args.query_id, args.worker == 'check_pair')
    elif args.worker == 'fit':
        fit_models(args.fold)
    elif args.worker == 'prepare':
        prepare_queries()
    elif args.worker == 'generate':
        generate_one(args.query_id)
    elif args.worker == 'evaluate':
        from scripts.retrieval_site_target_full_evaluate import evaluate
        evaluate()
    else:
        signal.signal(signal.SIGTERM, signal_stop)
        signal.signal(signal.SIGINT, signal_stop)
        run(args.commit_on_complete)
