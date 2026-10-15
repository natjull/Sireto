"""Bounded site-target timing: one internal model, twenty label-free searches."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from scripts.retrieval_site_operational_metric import read, record, verified, fingerprint, index
from scripts.retrieval_site_target_mapping import contract_commit, RESULT as MAPPING

H = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
OUT = H/'v173/site_target_timing_20261015'
V32 = H/'v32/global_public_parent_names_20261012_a'
V71 = H/'v71/frozen_residual_20261013_a'
PREPARED = H/'v142/bano_native_20261013_night'
SEED = 20261015


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def stem(q):
    return hashlib.sha256(q.encode()).hexdigest()


class Inputs:
    def __init__(self):
        self.refs = {}

    def path(self, ref):
        if isinstance(ref, (str, Path)):
            ref = dict(path=str(ref))
        path = Path(ref['path'])
        if str(path) not in self.refs:
            sig = fingerprint(path)
            if path.stat().st_size < 256*1024**2 and 'sha256' in ref:
                verified(ref)
            self.refs[str(path)] = dict(signature=sig, inherited=ref)
        return path

    def json(self, ref):
        return read(self.path(ref))

    def unchanged(self):
        for ref in self.refs.values():
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('STOP: input changed')


def training(stage):
    from scripts import embeddinggemma2_v32_learned_query as learned
    from scripts import embeddinggemma2_v32_native_features as base
    from scripts import embeddinggemma2_v71_frozen_residual as residual
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    inputs = Inputs()
    start = time.monotonic()
    source = V32 if stage == 'v32' else V71
    proof = base if stage == 'v32' else residual
    launch = inputs.json(source/'launch.json')
    for ref in [launch['implementation'], *launch['helpers']]:
        inputs.path(ref)
    gen = inputs.json(source/'features_generation.json')
    for name in ('published_parent_names', 'corporate_names'):
        if name in gen:
            inputs.path(gen[name])
    proof.bind_runtime(source)
    proof.authority = lambda out: launch  # verified immutable source, expired historical deadline not reused
    if stage == 'v71':
        residual.corporate.base.authority = lambda out: launch
    learned.proof = proof
    model_old = inputs.json(V71/'calibration/model_0.json')
    mapping = index(inputs.json(MAPPING)['per_query'])
    panel = index(inputs.json(launch['panel']))
    train = model_old['train_queries']
    held = model_old['heldout_queries']
    assignment = inputs.json(inputs.json(V71/'calibration/generation.json')['assignment'])
    if any(assignment[q] == 0 for q in train) or any(assignment[q] != 0 for q in held):
        raise ValueError('Internal partition drift')
    if {mapping[q]['label_siret'][:9] for q in train} & {mapping[q]['label_siret'][:9] for q in held}:
        raise ValueError('Company overlap')
    if any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('Forbidden fold')
    old = inputs.json(inputs.json(proof.CACHE/'launch.json')['baseline_outputs'])
    sampling = inputs.json(inputs.json(launch['source_launch'])['sampling'])
    backend = TantivyBackend(learned.io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Index drift')
    values, weights, missing = [], [], []
    for i, qid in enumerate(train):
        q = panel[qid]
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
        if not inside[0]:
            missing.append(qid)
        lex = learned.masked_lexical_features(backend, q, candidates)
        matrix = np.column_stack((lex, numeric))
        difference = matrix[0:1]-matrix[1:]
        if not len(difference) or not np.isfinite(matrix).all():
            raise ValueError('Invalid training pairs')
        weight = sampling['zone_weights'][q['crm_insee']]['calibrated_query_weight']
        pair_weights = np.full(len(difference), weight/len(difference), dtype=np.float64)
        path = OUT/stage/'pairs'/(stem(qid)+'.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez(stream, difference=difference, weights=pair_weights, candidates=candidates)
        values.append(difference)
        weights.append(pair_weights)
        print(stage, 'PAIRS', i+1, '/', len(train), qid, flush=True)
        del keys, x, matrix, numeric, lex
        gc.collect()
    pair_seconds = time.monotonic()-start
    d, w = np.concatenate(values), np.concatenate(weights)
    fit_start = time.monotonic()
    if stage == 'v32':
        coefficients, info = learned.fit_pairwise(d, w, .01)
    else:
        residual.ACTIVE = OUT/'v71'
        baseline_ref = record(OUT/'v32/model_0.json')
        residual.authority = lambda out: dict(frozen_models={'0': baseline_ref})
        coefficients, info = residual.fit_residual(d, w, .01)
    fit_seconds = time.monotonic()-fit_start
    model = dict(feature_fields=proof.FEATURE_FIELDS, coefficients=coefficients.tolist(), **info,
        train_queries=train, heldout_queries=held, heldout_fold=0, no_group_overlap=True,
        training_target_mapping=record(MAPPING), original_partition=record(V71/'calibration/model_0.json'))
    write(OUT/stage/'model_0.json', model)
    inputs.unchanged()
    write(OUT/stage/'result.json', dict(pair_preparation_seconds=pair_seconds, optimizer_seconds=fit_seconds,
        elapsed_seconds=time.monotonic()-start, training_queries=len(train), training_pairs=len(d),
        changed_training_targets=sum(mapping[q]['changed'] for q in train), missing_numeric_targets=missing,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, inputs=list(inputs.refs.values())))


def prepare_generation():
    import ijson
    inputs = Inputs()
    launch = inputs.json(PREPARED/'launch.json')
    fg = inputs.json(PREPARED/'features_generation.json')
    prepared = inputs.json(PREPARED/'prepared.json')
    pg = inputs.json(launch['parent_prepared'])
    panel = index(inputs.json(launch['panel']))
    source = inputs.json(launch['source_launch'])
    census = inputs.json(source['census'])
    large = [r['zone'] for r in sorted(census['zones'], key=lambda r: (-r['indexed_sites'], r['zone']))[:10]]
    rng = random.Random(SEED)
    eligible = sorted(q for q, r in panel.items() if r['crm_insee'] in large)
    # The brief requires three large communes, not three rows of one commune.
    for attempt in range(100):
        first = rng.sample(eligible, 3)
        if len({panel[q]['crm_insee'] for q in first}) == 3:
            break
    else:
        raise ValueError('Cannot sample three distinct large communes')
    chosen = first+rng.sample(sorted(set(panel)-set(first)), 17)
    if len(set(chosen)) != 20 or any(panel[q]['fold'] not in (2, 3, 4) for q in chosen):
        raise ValueError('Invalid timing selection')
    write(OUT/'selection.json', dict(seed=SEED, query_ids=chosen, large_communes=large,
        large_commune_queries=sum(panel[q]['crm_insee'] in large for q in chosen),
        distinct_large_communes=sorted({panel[q]['crm_insee'] for q in chosen if panel[q]['crm_insee'] in large}),
        reserved_draw_attempts=attempt+1, census=source['census'], model_id=0))
    refs = dict(context=fg['context_expansion'], initials=fg['initial_alignment'],
        supplier=inputs.json(launch['supplier_generation'])['extras'],
        address=inputs.json(launch['address_generation'])['extras'],
        rna=inputs.json(launch['rna_generation'])['context'], banatic=inputs.json(launch['banatic_generation'])['context'],
        bano_context=prepared['context'], osm_context=pg['source_context'], shapes=pg['shapes'])
    for name, ref in refs.items():
        found = set()
        with inputs.path(ref).open('rb') as stream:
            for qid, value in ijson.kvitems(stream, '', use_float=True):
                if qid in chosen:
                    write(OUT/'queries'/stem(qid)/(name+'.json'), value)
                    found.add(qid)
        if found != set(chosen):
            raise ValueError('Missing query context '+name)
    geometries = index(inputs.json(launch['geometry']))
    for qid in chosen:
        write(OUT/'queries'/stem(qid)/'binding.json', dict(query=panel[qid], launch=launch,
            features=fg['features'][qid], geometry=geometries[qid]['geometry'], feature_generation=fg))
    inputs.unchanged()
    write(OUT/'generation_inputs.json', dict(inputs=list(inputs.refs.values())))


def generate_one(qid):
    # No labels, mapping, outcomes, old candidates or assignment are opened here.
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    from scripts.retrieval_miss_score_explain import preparer_kernel
    inputs = Inputs()
    directory = OUT/'queries'/stem(qid)
    binding = inputs.json(directory/'binding.json')
    q, launch, fg = binding['query'], binding['launch'], binding['feature_generation']
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
    model = inputs.json(OUT/'v71/model_0.json')
    w = np.asarray(model['coefficients'], dtype=float)
    backend = v142.TantivyBackend(v142.io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('National index drift')
    start = time.monotonic()
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
    write(directory/'result.json', dict(query=q, candidates=rows, cost=cost,
        elapsed_seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        labels_read=False, model_id=0, no_recall=True, source_kernels=dict(bano=bsha, osm=osha), inputs=list(inputs.refs.values())))


def supervised(action, qid=None):
    tag = action+('_'+stem(qid) if qid else '')
    command = [sys.executable, '-u', str(Path(__file__)), '--worker', action]
    if qid:
        command += ['--query-id', qid]
    started = time.monotonic()
    peak = 0
    reason = None
    with (OUT/(tag+'.log')).open('xb') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                peak = max(peak, psutil.Process(child.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak >= 8*1024**3 or time.monotonic()-started > 7200:
                reason = 'RSS_8_GIB' if peak >= 8*1024**3 else 'TWO_HOUR_LIMIT'
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(.05)
        rc = child.wait()
    receipt = dict(command=command, pid=child.pid, exit_code=rc, elapsed_seconds=time.monotonic()-started,
        peak_sampled_rss_bytes=peak, stop_reason=reason, owned_child_exit_confirmed=True)
    write(OUT/(tag+'_receipt.json'), receipt)
    if rc or reason:
        raise RuntimeError(str(receipt))
    return receipt


def report():
    """Publish measured timing only after owned completion of the bounded run."""
    if (OUT/'stop.json').exists():
        stopped = read(OUT/'stop.json')
        receipts = [read(p) for p in sorted(OUT.glob('*_receipt.json'))]
        peak = max((r['peak_sampled_rss_bytes'] for r in receipts), default=0)
        counts = {stage: len(list((OUT/stage/'pairs').glob('*.npz'))) for stage in ('v32', 'v71')}
        completed_training = {stage: read(OUT/stage/'result.json') for stage in ('v32', 'v71') if (OUT/stage/'result.json').exists()}
        result = dict(status='STOP_NO_COMPLETE_TRAINING_OR_EXTRAPOLATION', contract_commit=contract_commit(),
            namespace=str(OUT), stop=stopped, receipts=receipts, completed_pair_shards=counts,
            completed_training=completed_training, peak_rss_bytes=peak, training_complete=(OUT/'v71/model_0.json').exists(),
            generation_queries=len(list((OUT/'queries').glob('*/result.json'))),
            measured_training_seconds=None, measured_generation_seconds=None,
            full_recalculation_low_seconds=None, full_recalculation_high_seconds=None,
            extrapolation_reason='No completed internal model and no twenty-query search sample; numerical estimate would be unsupported',
            labels_modified=False, no_recall=True, closed_sets_opened=False)
        path = ROOT/'docs/retrieval_site_target_timing_20261015.json'
        write(path, result)
        lines = ['# Chronométrage site-target — STOP du 15 octobre 2026', '',
            f'Contrat préalable `{result["contract_commit"]}` ; artefacts `{OUT}`.',
            '**Étape 3 bloquée, pas de chronométrage complet ni extrapolation chiffrée défendable.**', '',
            f'- Paires V32 terminées : {counts["v32"]}/284 ; V71 : {counts["v71"]}/284.',
            f'- Recherches top100 terminées : {result["generation_queries"]}/20 ; aucun recall.',
            f'- Pic RSS supervisé : **{peak/1024**3:.3f} Gio** ; garde enfant seul 8 Gio.',
            f'- Blocage : `{stopped["error"]}`.',
            '- Arrêt du seul enfant possédé confirmé par reçu ; aucune relance automatique, aucun autre calcul lourd lancé.',
            '- Durée complète (a), durée (b), fourchette complète basse/haute et mémoire attendue : **non mesurées / inconnues**.',
            '- Réutilisables : caches Gemma/adaptateur/vecteurs, index national, univers complets et signaux publics.',
            '- Restent à recalculer après résolution du blocage et accord : paires avec nouvelle cible, cinq V32 + cinq résiduels V71, '
            'bonus OSM/BANO dépendant des poids et 526 recherches Tantivy. Anciennes sorties non réutilisables sous de nouveaux poids.',
            '- Ne pas extrapoler la préparation partielle des paires à un apprentissage complet ni aux recherches non exécutées.',
            '- Partition0 interne reprise sans changement (pas fold0), mêmes concurrents re-minés/procédure/graines ; L2=0,01 prévue inchangée.',
            '- VM hydraulic arrêtée selon inventaire CatNat en lecture seule ; aucune commande de contrôle VM ; un thread.',
            '- Commande réelle détachée : `.venv/bin/python -u scripts/retrieval_site_target_timing.py` ; publication `--report`.',
            '- Suivi : `tail -n 20 '+str(OUT/'timing.log')+'` ; ne pas relancer sur ce namespace existant.',
            '- Aucun label modifié, jeu fermé, web, modèle aval, run/evaluate complet ou mesure de recall.']
        for stage, measured in completed_training.items():
            lines.append(f'- Phase terminée {stage} : {measured["elapsed_seconds"]:.2f} s '
                f'(préparation {measured["pair_preparation_seconds"]:.2f} s, optimiseur {measured["optimizer_seconds"]:.3f} s), '
                f'{measured["training_pairs"]} paires ; ne constitue pas à elle seule le chronométrage complet (a).')
        for r in receipts:
            lines.append(f'- Reçu : {r["command"][-1]} ; '
                         f'{r["elapsed_seconds"]:.2f} s, code {r["exit_code"]}, '
                         f'pic {r["peak_sampled_rss_bytes"]/1024**3:.3f} Gio, arrêt {r["stop_reason"]}.')
        with path.with_suffix('.md').open('x') as stream:
            stream.write('\n'.join(lines)+'\n')
        print(json.dumps(dict(status=result['status'], counts=counts, peak_rss_bytes=peak)), flush=True)
        return
    result = read(OUT/'result.json')
    if result['status'] != 'COMPLETE_ONE_INTERNAL_MODEL_TWENTY_SEARCHES_NO_RECALL':
        raise ValueError('Completed bounded run required')
    selection = read(OUT/'selection.json')
    a = {stage: read(OUT/stage/'result.json') for stage in ('v32', 'v71')}
    b = [read(OUT/'queries'/stem(q)/'result.json') for q in selection['query_ids']]
    if len(b) != 20 or any(r['labels_read'] or len(r['candidates']) > 100 or
                          len({c['siret'] for c in r['candidates']}) != len(r['candidates']) for r in b):
        raise ValueError('Invalid generation proof')
    # Validate persisted input signatures again, without reopening labels to score.
    authorities = [*a.values(), *b, read(OUT/'generation_inputs.json')]
    verified_count = 0
    for authority in authorities:
        for ref in authority['inputs']:
            if fingerprint(ref['signature']['path']) != ref['signature']:
                raise ValueError('Input changed before timing publication')
            verified_count += 1
    model = read(OUT/'v71/model_0.json')
    if len(model['coefficients']) != 58 or not np.isfinite(model['coefficients']).all():
        raise ValueError('Invalid residual model')
    base = read(OUT/'v32/model_0.json')
    from scripts.embeddinggemma2_v71_frozen_residual import BASE_AT
    if not np.array_equal(np.asarray(model['coefficients'])[BASE_AT], base['coefficients']):
        raise ValueError('Residual did not preserve new V32 baseline')
    panel = read(PREPARED/'launch.json')['panel']
    panel = index(read(panel['path']))
    large = set(selection['large_communes'])
    generation_estimate = 0.
    strata = []
    for flag in (False, True):
        sample = [r for r in b if (r['query']['crm_insee'] in large) == flag]
        n = sum((q['crm_insee'] in large) == flag for q in panel.values())
        if not sample:
            raise ValueError('Unobserved timing stratum')
        # End-to-end process times include native index/dictionary startup.
        times = [r['elapsed_seconds'] for r in result['receipts'][3:] if
                 (read(OUT/'queries'/stem(r['command'][-1])/'binding.json')['query']['crm_insee'] in large) == flag]
        estimate = n*sum(times)/len(times)
        generation_estimate += estimate
        strata.append(dict(large_communes=flag, panel_queries=n, sample_queries=len(sample),
            mean_end_to_end_seconds=sum(times)/len(times), estimated_526_seconds=estimate))
    cal = read(V71/'calibration/generation.json')
    excluded = set(read(cal['encoder_control_queries']['path']))
    eligible = len(panel)-len(excluded)
    pair_prep = sum(r['pair_preparation_seconds'] for r in a.values())
    optimizers = sum(r['optimizer_seconds'] for r in a.values())
    # Bounds are engineering scenarios, not statistical confidence intervals.
    train_low = pair_prep*eligible/len(model['train_queries']) + 5*optimizers
    train_high = 5*sum(r['elapsed_seconds'] for r in a.values())*1.25
    generation_low = .75*generation_estimate
    generation_high = 2*generation_estimate
    peak = max([r['peak_rss_bytes'] for r in a.values()]+[r['peak_rss_bytes'] for r in b]+
               [r['peak_sampled_rss_bytes'] for r in result['receipts']])
    inventory = read(OUT/'index_inventory_check.json')
    if not inventory['equal_historical']:
        raise ValueError('Historical index inventory differs')
    index_root = Path(read(PREPARED/'launch.json')['index_meta']['path']).parent
    if sorted([p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in index_root.iterdir() if p.is_file()) != inventory['inventory']:
        raise ValueError('Index inventory changed')
    measured = dict(status=result['status'], contract_commit=contract_commit(), namespace=str(OUT),
        index_inventory=record(OUT/'index_inventory_check.json'), lima_inventory=record(OUT/'lima_catnat_inventory.json'),
        mapping=record(MAPPING), training=a, generation=dict(queries=20,
            candidate_count=sum(len(r['candidates']) for r in b), selection=selection,
            preparation_receipt=result['receipts'][2], search_process_seconds=sum(r['elapsed_seconds'] for r in result['receipts'][3:]),
            kernel_seconds=sum(r['elapsed_seconds'] for r in b), strata=strata,
            receipts=result['receipts'][3:]), peak_rss_bytes=peak, input_signature_checks=verified_count,
        extrapolation=dict(training_low_seconds=train_low, training_high_seconds=train_high,
            generation_low_seconds=generation_low, generation_high_seconds=generation_high,
            total_low_seconds=train_low+generation_low, total_high_seconds=train_high+generation_high,
            expected_peak_gib=[peak/1024**3, 8], confidence_interval=False,
            assumption='Same single-thread Mac/CATNAT, caches intact; lower shared mining, upper repeated mining and 2x search cost'),
        no_recall=True, labels_modified=False, closed_sets_opened=False)
    path = ROOT/'docs/retrieval_site_target_timing_20261015.json'
    write(path, measured)
    lines = ['# Chronométrage site-target — 15 octobre 2026', '',
        f'Contrat préalable `{measured["contract_commit"]}` ; namespace `{OUT}`.',
        '**Une partition interne (0), 20 recherches réelles ; aucun recall.**', '',
        '| Phase | Paires / préparation (s) | Optimiseur (s) | Total interne (s) | Pic RSS Gio |',
        '|---|---:|---:|---:|---:|']
    for stage, r in a.items():
        lines.append(f'| {stage} | {r["pair_preparation_seconds"]:.2f} | {r["optimizer_seconds"]:.3f} | {r["elapsed_seconds"]:.2f} | {r["peak_rss_bytes"]/1024**3:.3f} |')
    for stage, r in a.items():
        lines.append(f'\n{stage} : {r["training_queries"]} requêtes train, {r["training_pairs"]} paires, '
                     f'{r["changed_training_targets"]} cibles changées, {len(r["missing_numeric_targets"])} cibles sans signal numérique.')
    lines += ['', f'Génération : **20/20**, {measured["generation"]["candidate_count"]} candidats, cap100 unique vérifié ; '
        f'{selection["large_commune_queries"]} requêtes des dix plus grandes communes du census.',
        f'Préparation des contextes : {result["receipts"][2]["elapsed_seconds"]:.2f} s ; '
        f'20 processus de recherche, démarrages inclus : **{measured["generation"]["search_process_seconds"]:.2f} s** ; '
        f'kernels/préparateurs/recherche internes : {measured["generation"]["kernel_seconds"]:.2f} s.',
        f'Pic RSS toutes phases : **{peak/1024**3:.3f} Gio** (garde supervisée 8 Gio, un thread, aucun processus lourd parallèle).',
        f'Signatures d’entrée revérifiées : {verified_count}. Inventaire national : {inventory["files"]} fichiers conformes aux tailles/mtime historiques et inchangés au contrôle final. Les 28 nouveaux poids V32 sont inchangés dans le résiduel V71.',
        '', '## Extrapolation conditionnelle : cinq modèles + 526 recherches', '',
        f'- Apprentissage : {train_low/60:.1f}–{train_high/60:.1f} min.',
        f'- Génération : {generation_low/60:.1f}–{generation_high/60:.1f} min.',
        f'- **Total : {(train_low+generation_low)/60:.1f}–{(train_high+generation_high)/60:.1f} min** ; '
        f'pic attendu {peak/1024**3:.2f}–8 Gio/processus, sans garantie que les 506 autres tiennent sous ce plafond.',
        '- Basse : mining des paires partagé sur tous les éligibles, cinq optimiseurs, 0,75× coût de recherche '
        'stratifié grandes/autres communes. Haute : cinq minings complets ×1,25 et 2× coût stratifié de recherche.',
        '- Scénarios d’ingénierie, pas IC de durée ni SLO ; état froid/chaud et autres charges non contrôlés. '
        'Préparation des contextes du panel complet et validation finale non mesurées séparément : prévoir marge dans la borne haute.',
        '- Réutilisables : Gemma/adaptateur/vecteurs, index national, univers complets et 51 signaux publics, '
        'sources officielles et contextes. Aucun besoin de réencodage Gemma ou nouveau scan national de génération d’univers.',
        '- À recalculer : paires avec nouvelle cible, 5 V32 puis 5 résiduels V71, bonus OSM/BANO dépendant des poids, '
        'requêtes Tantivy et sorties526 ; anciennes sorties V142 non réutilisables sous les nouveaux poids.',
        '- Les paires déjà produites pour ce pilote peuvent être réutilisées ; estimation ci-dessus depuis les caches publics, '
        'sans déduire ce gain. Le reste des paires est indépendant du modèle interne et peut être partagé.',
        '', '## Limites / reproduction', '',
        '- Choix explicité : « au moins3 grandes communes » signifie trois communes distinctes ; rejet déterministe des tirages réservés '
        'avec commune répétée, puis 17 autres requêtes sans remise, même graine20261015, sans labels.',
        '- Modèle interne0 fixe pour les20, pas de routage label-side, pas de prétention OOF ni qualité nouvelle mesurée.',
        '- Partitions et exclusions historiques reprises ; concurrents re-minés avec la procédure et graines historiques ; L2=0,01.',
        '- Les identifiants de cible n’entrent que dans la supervision ; aucun label réécrit, aucune injection en recherche.',
        '- VM hydraulic : arrêtée selon inventaire local CatNat en lecture seule (`LIMA_HOME=/Volumes/CATNAT_DATA/catnat-lima`), aucune commande de démarrage/arrêt ; calcul conservateur un thread.',
        '- Gros fichiers : signatures taille/mtime/inode + hashes hérités ; petits bindings vérifiés SHA256 ; pas de nouveau hash complet des segments.',
        '- Commande réelle détachée : `.venv/bin/python -u scripts/retrieval_site_target_timing.py` ; '
        'publication : même commande avec `--report`. Ne pas relancer sur ce namespace existant.',
        '- Suivi : `tail -n 20 '+str(OUT/'timing.log')+'` ; résultats détaillés par requête dans `queries/*/result.json`.',
        '- Recalcul complet et toute évaluation : **attendre accord Nathan**.']
    with path.with_suffix('.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(extrapolation=measured['extrapolation'], peak_rss_bytes=peak)), flush=True)


def main():
    commit = contract_commit()
    write(OUT/'launch.json', dict(contract_commit=commit, implementation=record(Path(__file__)),
        mapping=record(MAPPING), seed=SEED, memory_cap_bytes=8*1024**3, threads=1,
        lima_list=subprocess.run(['limactl', 'list', '--json'], capture_output=True, text=True).stdout,
        lima_home=os.environ.get('LIMA_HOME'), closed_sets_opened=False, recall_measured=False))
    receipts = []
    try:
        receipts.append(supervised('v32'))
        receipts.append(supervised('v71'))
        receipts.append(supervised('prepare'))
        for qid in read(OUT/'selection.json')['query_ids']:
            receipts.append(supervised('generate', qid))
        write(OUT/'result.json', dict(status='COMPLETE_ONE_INTERNAL_MODEL_TWENTY_SEARCHES_NO_RECALL', receipts=receipts))
    except Exception as error:
        write(OUT/'stop.json', dict(status='STOP', error=repr(error), completed_receipts=receipts))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=['v32', 'v71', 'prepare', 'generate'])
    p.add_argument('--query-id')
    p.add_argument('--report', action='store_true')
    args = p.parse_args()
    if args.report:
        report()
    elif args.worker in ('v32', 'v71'):
        training(args.worker)
    elif args.worker == 'prepare':
        prepare_generation()
    elif args.worker == 'generate':
        generate_one(args.query_id)
    else:
        main()
