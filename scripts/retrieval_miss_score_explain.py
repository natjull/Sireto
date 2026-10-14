"""Diagnostic borné des 19 ratés V142 ; moteur natif inchangé, jamais de recall.

Un processus par requête et phase. Les 19 replays fidèles précèdent toute
explication. Seules les copies d'exécution instrumentées et les caches de
requête sont réduits ; ni univers, ni formule, ni collecteur ne sont remplacés.
"""
import argparse
import collections
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import textwrap
import time
import traceback

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ijson
import numpy as np
import psutil

HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
PREPARED = HOME/'v142/bano_native_20261013_night'
RUN = PREPARED/'all_new_sources_max_bano'
OUT = HOME/'v172/miss_score_explain_20261014'
ACCESS = ROOT/'docs/retrieval_miss_access_vs_ranking_20261014.json'
RESULT = ROOT/'docs/retrieval_miss_score_explain_20261014.json'
STOP_V160 = ROOT/'docs/retrieval_v160_explain_final_c_stop_20261014.json'
CAP = 6*1024**3
ATOL = 1e-5
MAXIMUM = (1 << 20)-1


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    # Tous les fichiers de cette tâche sont nouveaux, sans reprise/écrasement.
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def peak():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def guard():
    rss = psutil.Process().memory_info().rss
    if rss > CAP:
        raise MemoryError(f'STOP_RSS_6GIB: {rss}')
    return rss


class Inputs:
    def __init__(self):
        self.refs = {}

    def path(self, ref):
        path = Path(ref['path']) if isinstance(ref, dict) else Path(ref)
        key = str(path)
        if key not in self.refs:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block)
                    guard()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source modifiée pendant lecture: '+key)
            self.refs[key] = dict(path=key, sha256=digest.hexdigest(),
                                  size_bytes=after.st_size, mtime_ns=after.st_mtime_ns)
        if isinstance(ref, dict) and self.refs[key]['sha256'] != ref['sha256']:
            raise ValueError('Empreinte non conforme: '+key)
        return path

    def json(self, ref):
        return read(self.path(ref))

    def unchanged(self):
        for ref in self.refs.values():
            stat = Path(ref['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (ref['size_bytes'], ref['mtime_ns']):
                raise ValueError('Source modifiée: '+ref['path'])


def stem(qid):
    return hashlib.sha256(qid.encode()).hexdigest()


def subset_file(inputs, ref, qids, name):
    found = set()
    with inputs.path(ref).open('rb') as stream:
        for qid, value in ijson.kvitems(stream, '', use_float=True):
            if qid in qids:
                write(OUT/'inputs'/stem(qid)/(name+'.json'), value)
                found.add(qid)
            del value
            gc.collect()
            guard()
    if found != qids:
        raise ValueError('Contexte requête manquant: '+name)


def prepare_inputs(inputs, misses):
    launch = inputs.json(PREPARED/'launch.json')
    fg = inputs.json(PREPARED/'features_generation.json')
    prepared = inputs.json(PREPARED/'prepared.json')
    pg = inputs.json(launch['parent_prepared'])
    cal = inputs.json(PREPARED/'calibration/generation.json')
    assignment = inputs.json(cal['assignment'])
    panel = {q['query_id']: q for q in inputs.json(launch['panel'])}
    generation = inputs.json(RUN/'generation.json')
    outputs = inputs.json(generation['outputs'])['learned_native_ooc']
    if generation['queries'] != 526 or not generation['no_positive_injected']:
        raise ValueError('Run scellé all526 requis')
    for ref in [launch['implementation'], *launch['helpers']]:
        inputs.path(ref)
    inputs.path(launch['index_manifest']); inputs.path(launch['index_meta'])
    models = {k: inputs.json(ref) for k, ref in cal['models'].items()}
    if set(models) != set('01234') or any(len(m['coefficients']) != 58 for m in models.values()):
        raise ValueError('Exactement cinq modèles gelés de 58 coefficients')
    qids = {r['query_id'] for r in misses}
    if len(qids) != 19:
        raise ValueError('Exactement 19 ratés requis')
    for miss in misses:
        qid = miss['query_id']; q = panel[qid]; model_id = str(assignment[qid])
        if q['fold'] not in (2, 3, 4) or q['fold'] != miss['fold']:
            raise ValueError('Folds 2/3/4 seulement')
        if miss['label_siret'] in {r['siret'] for r in outputs[qid]}:
            raise ValueError('Raté absent du top100 requis')
        if qid in models[model_id]['train_queries'] or qid not in models[model_id]['heldout_queries']:
            raise ValueError('Même calibrateur hors entreprise humaine')
        directory = OUT/'inputs'/stem(qid); directory.mkdir(parents=True)
        expected = inputs.json(RUN/'native'/(stem(qid)+'.json'))
        if expected['query'] != q or expected['model'] != cal['models'][model_id] or expected['candidates'] != outputs[qid]:
            raise ValueError('Bindings natifs / outputs différents')
        write(directory/'binding.json', dict(query=q, miss=miss, model_id=model_id,
              expected=expected['candidates'], models=cal['models'], features=fg['features'][qid],
              bano=prepared['shards'][qid], osm=pg['shards'][qid],
              activity=fg['activity_shards'][qid], launch=launch,
              field_names=models[model_id]['feature_fields']))
    refs = {
        'context': fg['context_expansion'], 'initials': fg['initial_alignment'],
        'supplier': inputs.json(launch['supplier_generation'])['extras'],
        'address': inputs.json(launch['address_generation'])['extras'],
        'rna': inputs.json(launch['rna_generation'])['context'],
        'banatic': inputs.json(launch['banatic_generation'])['context'],
        'bano_context': prepared['context'], 'osm_context': pg['source_context'],
        'shapes': pg['shapes'],
    }
    for name, ref in refs.items():
        subset_file(inputs, ref, qids, name)
    geometries = inputs.json(launch['geometry'])
    for row in geometries:
        if row['query_id'] in qids:
            write(OUT/'inputs'/stem(row['query_id'])/'geometry.json', row['geometry'])
    for name in ('corporate_names', 'published_parent_names', 'indirect_names'):
        inputs.path(fg[name])
    write(OUT/'input_manifest.json', dict(inputs=list(inputs.refs.values()),
          source_feature_generation=fg, execution_only='CACHES_QUERY_SUBSET_AND_NO_QUERY_REPR',
          original_top100_unchanged=True, labels_source=str(ACCESS)))


class Capture:
    def __init__(self, searcher):
        self.searcher = searcher
        self.query = None
        self.addresses = None

    def doc(self, address):
        return self.searcher.doc(address)

    def search(self, query, **kwargs):
        if kwargs.get('limit') != 100 or self.query is not None:
            raise ValueError('Un seul collecteur natif100 par modèle')
        self.query = query
        guard()
        result = self.searcher.search(query, **kwargs)
        self.addresses = result.hits
        guard()
        return result


def instrument(module, name, replacements, additions):
    source = inspect.getsource(getattr(module, name))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Point d’instrumentation non unique: '+name)
        source = source.replace(old, new)
    namespace = dict(vars(module), **additions)
    exec(compile(source, str(Path(__file__))+':capture_original_'+name, 'exec'), namespace)
    return namespace[name]


def environment(binding, directory, inputs):
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    v115 = v142.baseline; v99 = v115.baseline; v81 = v99.source
    base = v81.pipeline.base; corp = v81.pipeline.corporate
    fg = read(OUT/'input_manifest.json')['source_feature_generation']
    v115.bind_runtime(PREPARED)
    v99.CONTEXT = {binding['query']['query_id']: read(directory/'context.json')}
    v99.INITIAL_DATA = {binding['query']['query_id']: read(directory/'initials.json')}
    v81.INDIRECT = inputs.json(fg['indirect_names'])
    # L'autorité historique contient une date expirée. Les mêmes fichiers sont
    # vérifiés en amont ; aucune garde ancienne ni préparation nationale ici.
    base.authority = lambda out: binding['launch']
    state = dict(base={}, v81={}, stages={})
    base.native_query = instrument(base, 'native_query', [
        ('    for component,weight in zip(lexical_parts(backend,q),coefficients[:len(LEXICAL_FIELDS)]):',
         '    components=lexical_parts(backend,q)\n    TRACE["lexical"]=components\n    for component,weight in zip(components,coefficients[:len(LEXICAL_FIELDS)]):')],
         dict(TRACE=state['base']))
    v81.native_search = instrument(v81, 'native_search', [
        ('    if INDIRECT is None:', '    TRACE["indirect_start"]=len(parts)\n    if INDIRECT is None:'),
        ("    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)",
         "    TRACE[\"parts\"]=parts\n    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)")],
         dict(TRACE=state['v81']))
    for cls, tag in [(v99.InitialsBeforeOnlyCollector, 'v81'),
                     (v99.BeforeOnlyCollector, 'initials'),
                     (v115.original.BeforeOnlyCollector, 'typed'),
                     (v142.parent.EvidenceCollector, 'activity')]:
        old = cls.search
        def record(self, query, _old=old, _tag=tag, **kwargs):
            state['stages'][_tag] = query
            return _old(self, query, **kwargs)
        cls.search = record
    backend = v142.TantivyBackend(v142.io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Même index national requis')
    inputs.path(fg['corporate_names']); inputs.path(fg['published_parent_names'])
    return v142, backend, state


def original_scores(v142, binding, directory, inputs, keys, coefficients):
    with np.load(inputs.path(binding['bano']), allow_pickle=False) as archive:
        if not np.array_equal(keys, archive['keys']):
            raise ValueError('Univers BANO différent')
        bano = archive['bano_geometry']
    with np.load(inputs.path(binding['osm']), allow_pickle=False) as archive:
        if not np.array_equal(keys, archive['keys']):
            raise ValueError('Univers OSM différent')
        geom, named = archive['geometry'], archive['named']
    supplier = read(directory/'supplier.json')
    own = v142.upstream.projected(keys, supplier['own'])
    temporal = v142.upstream.projected(keys, supplier['temporal'])
    continuity = v142.upstream.projected(keys, read(directory/'address.json'))
    coefficient = float(coefficients[6])
    previous = np.maximum(geom, named)
    previous = np.maximum(previous, coefficient*np.maximum(np.maximum(own, temporal), continuity))
    score = np.maximum(bano, previous)
    merged = {r['siren_ref']: r['confidence'] for r in read(directory/'rna.json')}
    for row in read(directory/'banatic.json'):
        merged[row['siren_ref']] = max(merged.get(row['siren_ref'], 0.), row['confidence'])
    entries = [dict(siren_ref=r, confidence=c) for r, c in sorted(merged.items())]
    return score, entries, dict(bano=bano, geometry=geom, named=named,
                               supplier_own=own, supplier_temporal=temporal, continuity=continuity)


def replay(v142, backend, state, q, keys, features, coefficients, scores, entries):
    state['base'].clear(); state['v81'].clear(); state['stages'].clear()
    capture = Capture(backend.index.searcher())
    backend.searcher = capture
    proxy = v142.baseline.baseline.BackendProxy(backend, v142.parent.EvidenceCollector(
        backend, q, keys, scores, entries, float(coefficients[6])))
    rows, cost = v142.baseline.native_search(proxy, q, keys, features, coefficients)
    return rows, cost, capture


def fidelity(rows, expected):
    ids_equal = [r['siret'] for r in rows] == [r['siret'] for r in expected]
    error = max((abs(r['score']-e['score']) for r, e in zip(rows, expected)), default=float('inf'))
    return dict(passed=ids_equal and len(rows) == len(expected) == 100 and error <= ATOL,
                same_ordered_100_sirets=ids_equal, max_score_absolute_error=error,
                absolute_tolerance=ATOL, relative_tolerance=0,
                missing_sirets=sorted({r['siret'] for r in expected}-{r['siret'] for r in rows}),
                added_sirets=sorted({r['siret'] for r in rows}-{r['siret'] for r in expected}))


def explained(query, searcher, address):
    if query is None:
        return 0.
    try:
        tree = json.loads(query.explain(searcher, address).to_json())
    except ValueError as exc:
        if 'does not match' in str(exc).lower() or 'doesn\'t match' in str(exc).lower():
            return 0.
        raise
    value = float(tree['value'])
    if not np.isfinite(value):
        raise ValueError('EXPLAIN non fini')
    guard()
    return value


def lookup(backend, capture, siret):
    tv = backend._tantivy
    # Localisateur technique séparé, jamais injecté dans la requête scorante.
    query = tv.Query.boolean_query([
        (tv.Occur.Must, tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'siret', siret, 'basic'), 0.)),
        (tv.Occur.Must, tv.Query.const_score_query(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.))])
    hits = capture.searcher.search(query, limit=2, count=False).hits
    if len(hits) != 1:
        raise ValueError('Document officiel unique requis: '+siret)
    return hits[0][1]


def contribution(v142, backend, state, capture, binding, keys, features, weights, scores, sources, entries, role, siret):
    tv = backend._tantivy; address = lookup(backend, capture, siret)
    total = explained(capture.query, capture.searcher, address)
    at = int(np.searchsorted(keys, np.uint64(int(siret))))
    present = at < len(keys) and keys[at] == int(siret)
    x = features[at].astype(float) if present else np.zeros(51, dtype=float)
    fields = binding['field_names']; signals = []
    lexical = state['base']['lexical']+[v142.baseline.baseline.source.pipeline.corporate.corporate_component(backend, binding['query'])]
    for i, component in enumerate(lexical):
        value = explained(component, capture.searcher, address)
        native = explained(tv.Query.boost_query(component, float(weights[i])), capture.searcher, address) if component is not None and weights[i] > 0 else 0.
        signals.append(dict(index=i, signal=fields[i], coefficient=float(weights[i]), value=value,
                            coefficient_times_value=float(weights[i])*value, native_contribution=native))
    for j, value in enumerate(x):
        i = j+7
        signals.append(dict(index=i, signal=fields[i], coefficient=float(weights[i]), value=float(value),
                            coefficient_times_value=float(weights[i])*float(value),
                            native_contribution=None, note='Contribution avant quantification du bloc, non quantifiée séparément'))
    stage = {k: explained(query, capture.searcher, address) for k, query in state['stages'].items()}
    supplier = sum(explained(query, capture.searcher, address) for query in state['v81']['parts'][state['v81']['indirect_start']:])
    doc = capture.searcher.doc(address).to_dict()
    q = binding['query']
    admin = bool((q['crm_insee'] and backend._first(doc, 'insee') == q['crm_insee']) or
                 (q['crm_postcode'] and backend._first(doc, 'postcode') == q['crm_postcode']))
    confidence = max((e['confidence'] for e in entries if e['siren_ref'] == siret[:9]), default=0.) if admin else 0.
    owner = float(weights[6])*int(np.floor(confidence*MAXIMUM))/MAXIMUM
    bound = float(scores.max(initial=0))
    leaf_raw = float(scores[at]) if present else 0.
    leaf = bound*int(np.floor(np.clip(leaf_raw/bound, 0, 1)*MAXIMUM))/MAXIMUM if bound else 0.
    raw_sources = {k: float(v[at]) if present else 0. for k, v in sources.items()}
    weighted_sources = {k: (value*float(weights[6]) if k.startswith('supplier_') or k == 'continuity' else value)
                        for k, value in raw_sources.items()}
    bonuses = [dict(signal='supplier_indirect_lexical', contribution=supplier),
               dict(signal='official_initials_admin', contribution=stage['initials']-stage['v81']),
               dict(signal='typed_relations_admin', contribution=stage['typed']-stage['initials']),
               dict(signal='bodacc_activity_admin', contribution=stage['activity']-stage['typed']),
               dict(signal='new_sources_leaf_max_quantized', contribution=leaf, raw_max=leaf_raw,
                    alternatives_not_additive=weighted_sources,
                    winners=[k for k, value in weighted_sources.items() if leaf_raw > 0 and value == leaf_raw]),
               dict(signal='rna_banatic_owner_max_admin', contribution=owner, confidence=confidence,
                    coefficient=float(weights[6]))]
    prequant_numeric = sum(s['coefficient_times_value'] for s in signals[7:])
    explicit = sum(s['native_contribution'] for s in signals[:7])+prequant_numeric+sum(b['contribution'] for b in bonuses)
    residual = total-explicit
    quant_bound = (float(np.sum(weights[7:29]))+float(np.sum(weights[29:]))+bound+float(weights[6]))/MAXIMUM
    if abs(residual) > quant_bound+5*ATOL:
        raise ValueError(f'Décomposition non refermée: {siret}, résidu={residual}, borne={quant_bound}')
    return dict(role=role, siret=siret, siren=siret[:9], score=total, numeric_universe_present=bool(present),
                signals=signals, bonuses=bonuses, quantization_and_native_float_residual=residual,
                numeric_prequant_total=prequant_numeric, residual_absolute_bound=quant_bound+5*ATOL,
                sum_reconstructed=explicit+residual)


def preparer_kernel(v142, kind):
    # Slices exactes des préparateurs réellement liés, sans E/S ni boucle panel.
    if kind == 'bano':
        source = inspect.getsource(v142.prepare)
        block = source.split('            at = np.searchsorted(keys_all, keys)\n', 1)[1]
        block = '            at = np.searchsorted(keys_all, keys)\n'+block.split('            np.savez_compressed(path, keys=keys, bano_geometry=delta)', 1)[0]
        signature = 'keys,features,keys_all,xy,public,points,model'
        ending = '    return delta\n'
        namespace = dict(vars(v142))
    else:
        module = v142.parent
        source = Path(ROOT/'scripts/embeddinggemma2_v132_local_evidence_ablation.py').read_text()
        start = "            keys=f['keys'];features=f['features'];at=np.searchsorted(keys_all,keys)\n"
        block = '            at=np.searchsorted(keys_all,keys)\n'+source.split(start, 1)[1].split('            np.savez_compressed(path,keys=keys,geometry=geom,named=named)', 1)[0]
        block = block.replace('int(group.max())+1', 'int(group.max(initial=-1))+1')  # Exact correction V143.
        signature = 'keys,features,keys_all,xy,public,w,gs,shapes,qid,q,context'
        ending = '    return geom,named\n'
        namespace = dict(vars(module))
    body = 'def kernel('+signature+'):\n'+textwrap.indent(textwrap.dedent(block), '    ')+ending
    exec(compile(body, str(Path(__file__))+':exact_'+kind+'_prepare_slice', 'exec'), namespace)
    return namespace['kernel'], hashlib.sha256(body.encode()).hexdigest()


def alternative_sources(v142, binding, directory, inputs, keys, features, models, frozen_sources):
    launch = binding['launch']
    all_keys = np.load(inputs.path(launch['keys']), mmap_mode='r')
    xy = np.load(inputs.path(launch['xy']), mmap_mode='r')
    public = np.load(inputs.path(launch['public_mask']), mmap_mode='r')
    bano, bano_sha = preparer_kernel(v142, 'bano')
    osm, osm_sha = preparer_kernel(v142, 'osm')
    points, _ = v142.compatible_points(read(directory/'bano_context.json'))
    q = binding['query']; qid = q['query_id']
    args = dict(gs={qid: read(directory/'geometry.json')}, shapes={qid: read(directory/'shapes.json')},
                qid=qid, q=q, context=read(directory/'osm_context.json'))
    result = {}
    # Ne conserver que les vecteurs de scores, jamais cinq matrices de features.
    for model_id, model in models.items():
        guard()
        delta = bano(keys, features, all_keys, xy, public, points, model)
        geom, named = osm(keys, features, all_keys, xy, public,
                          np.asarray(model['coefficients'], dtype=float), **args)
        if model_id == binding['model_id']:
            for name, value in [('bano', delta), ('geometry', geom), ('named', named)]:
                if not np.array_equal(value, frozen_sources[name]):
                    raise ValueError('STOP : préparateur '+name+' ne reproduit pas le shard enregistré')
        own = frozen_sources['supplier_own']; temporal = frozen_sources['supplier_temporal']; continuity = frozen_sources['continuity']
        coef = float(model['coefficients'][6])
        score = np.maximum(delta, np.maximum(np.maximum(geom, named), coef*np.maximum(np.maximum(own, temporal), continuity)))
        result[model_id] = dict(scores=score, sources=dict(bano=delta, geometry=geom, named=named,
                   supplier_own=own, supplier_temporal=temporal, continuity=continuity))
        gc.collect()
    return result, dict(bano=bano_sha, osm=osm_sha)


def deficits(documents):
    target, cut = documents['label'], documents['cut100']
    values = []
    for a, b in zip(target['signals'], cut['signals']):
        key = 'native_contribution' if a['index'] < 7 else 'coefficient_times_value'
        values.append(dict(signal=a['signal'], index=a['index'], deficit=b[key]-a[key],
                           comparison='native' if a['index'] < 7 else 'prequant_numeric'))
    for a, b in zip(target['bonuses'], cut['bonuses']):
        values.append(dict(signal=a['signal'], index=None, deficit=b['contribution']-a['contribution'], comparison='native_or_quantized'))
    values.append(dict(signal='quantization_and_native_float_residual', index=None,
                       deficit=cut['quantization_and_native_float_residual']-target['quantization_and_native_float_residual'], comparison='residual'))
    gap = cut['score']-target['score']
    if abs(sum(v['deficit'] for v in values)-gap) > ATOL:
        raise ValueError('Écart de décomposition non refermé')
    return sorted([v for v in values if v['deficit'] > 0], key=lambda v: (-v['deficit'], v['signal']))[:3], gap, values


def worker(qid, phase):
    started = time.monotonic(); directory = OUT/'inputs'/stem(qid); inputs = Inputs()
    binding = read(directory/'binding.json')
    v142, backend, state = environment(binding, directory, inputs)
    with np.load(inputs.path(binding['features']), allow_pickle=False) as archive:
        keys, features = archive['keys'], archive['features']
    if features.shape != (len(keys), 51) or np.any(keys[1:] <= keys[:-1]):
        raise ValueError('51 signaux et univers complet ordonné')
    models = {k: inputs.json(ref) for k, ref in binding['models'].items()}
    model_id = binding['model_id']; weights = np.asarray(models[model_id]['coefficients'], dtype=float)
    scores, entries, sources = original_scores(v142, binding, directory, inputs, keys, weights)
    rows, cost, capture = replay(v142, backend, state, binding['query'], keys, features, weights, scores, entries)
    proof = fidelity(rows, binding['expected'])
    write(OUT/phase/(stem(qid)+'_fidelity.json'), dict(query_id=qid, **proof, actual=rows))
    if not proof['passed']:
        raise ValueError('STOP_FIDELITY : top100/notes non reproduits')
    result = dict(query_id=qid, crm_name=binding['query']['crm_name'], fold=binding['query']['fold'],
                  model_id=model_id, fidelity=proof, numeric_sites=len(keys), cost=cost)
    if phase == 'explain':
        sealed = read(OUT/'fidelity_complete.json')
        if sealed['queries_reproduced'] != 19:
            raise ValueError('19/19 avant les explications')
        documents = {}
        for role, siret in [('label', binding['miss']['label_siret']), ('first', rows[0]['siret']), ('cut100', rows[-1]['siret'])]:
            documents[role] = contribution(v142, backend, state, capture, binding, keys, features,
                    weights, scores, sources, entries, role, siret)
            if role != 'label':
                expected = rows[0 if role == 'first' else -1]['score']
                if abs(documents[role]['score']-expected) > ATOL:
                    raise ValueError('STOP_EXPLAIN_SCORE_FIDELITY')
        top3, gap, all_deficits = deficits(documents)
        result.update(documents=documents, label_rank='> 100, rang non calculé',
                       top3_deficits=top3, total_deficit=gap, all_deficits=all_deficits)
        # Libérer la requête compilée et les captures avant les autres modèles.
        state['base'].clear(); state['v81'].clear(); state['stages'].clear()
        backend.searcher = backend.index.searcher()
        v142.baseline.baseline.source.pipeline.base.LAST_PART = None
        v142.baseline.baseline.source.pipeline.corporate.LAST_CORPORATE = None
        del capture; gc.collect()
        alternatives, kernels = alternative_sources(v142, binding, directory, inputs, keys, features, models, sources)
        result['exact_preparer_slice_sha256'] = kernels
        result['models'] = {}
        for alternative_id, model in models.items():
            w = np.asarray(model['coefficients'], dtype=float)
            data = alternatives.pop(alternative_id)
            alt_rows, _, captured = replay(v142, backend, state, binding['query'], keys, features, w, data['scores'], entries)
            if alternative_id == model_id and not fidelity(alt_rows, binding['expected'])['passed']:
                raise ValueError('STOP : recomposition du modèle original différente')
            label = binding['miss']['label_siret']; cut_key = documents['cut100']['siret']
            label_score = explained(captured.query, captured.searcher, lookup(backend, captured, label))
            old_cut_score = explained(captured.query, captured.searcher, lookup(backend, captured, cut_key))
            result['models'][alternative_id] = dict(label_score=label_score,
                recorded_cut100_siret=cut_key, recorded_cut100_score=old_cut_score,
                actual_model_cut100_siret=alt_rows[-1]['siret'], actual_model_cut100_score=alt_rows[-1]['score'],
                passes_actual_top100=label in {r['siret'] for r in alt_rows},
                label_rank_if_retained=next((i+1 for i, r in enumerate(alt_rows) if r['siret'] == label), None),
                is_recorded_model=alternative_id == model_id)
            state['base'].clear(); state['v81'].clear(); state['stages'].clear()
            backend.searcher = backend.index.searcher()
            v142.baseline.baseline.source.pipeline.base.LAST_PART = None
            v142.baseline.baseline.source.pipeline.corporate.LAST_CORPORATE = None
            del captured, data; gc.collect(); guard()
        result['passes_another_model'] = any(v['passes_actual_top100'] for k, v in result['models'].items() if k != model_id)
    inputs.unchanged(); guard()
    result.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak(), inputs=list(inputs.refs.values()))
    write(OUT/phase/(stem(qid)+'.json'), result)
    print(phase, qid, 'COMPLETE', result['elapsed_seconds'], result['peak_rss_bytes'], flush=True)


def family(signal):
    if signal in ('word_names', 'char_names', 'own_name_similarity'):
        return 'Nom propre et ressemblances textuelles'
    if 'cosine' in signal:
        return 'Ressemblance Gemma site/entreprise'
    if 'activity' in signal:
        return 'Activités BODACC'
    if 'initial' in signal:
        return 'Initiales officielles'
    if 'supplier' in signal:
        return 'Contexte fournisseur publié'
    if signal == 'new_sources_leaf_max_quantized':
        return 'MAX géométrie/OSM/fournisseur/continuité (non additif)'
    if 'rna_banatic' in signal:
        return 'RNA/BANATIC, contexte public administratif'
    if any(word in signal for word in ('corporate', 'related', 'typed', 'succession', 'parent')):
        return 'Entreprise, acteurs et continuité économique'
    if any(word in signal for word in ('address', 'street', 'house', 'number')):
        return 'Adresse propre et interactions'
    if 'geo' in signal:
        return 'Géographie propre et interactions'
    if 'quantization' in signal:
        return 'Quantification et arrondis natifs'
    return 'Commune/CP et autres signaux'


def publish(misses, completed, proofs, status, failure, receipts, started):
    valid = status == 'COMPLETE_FAITHFUL_19_MISS_SCORE_DIAGNOSTIC'
    counts = collections.Counter()
    if valid:
        for row in completed:
            counts.update(set(family(d['signal']) for d in row['top3_deficits']))
    result = dict(status=status, scope='POSTEVAL_DIAGNOSTIC_19_MISSES_FOLDS_2_3_4_NOT_NEW_RECALL',
        source_run=str(RUN), quantization_bits=20, fidelity_queries_reproduced=len(proofs),
        fidelity_queries_required=19, absolute_tolerance=ATOL, relative_tolerance=0,
        memory_cap_bytes=CAP, peak_resident_bytes=max([peak()]+[r['peak_resident_bytes'] for r in receipts]),
        elapsed_seconds=time.monotonic()-started, failure=failure,
        scientific_conclusions_allowed=valid, query_results=completed if valid else [],
        family_counts_in_top3=dict(counts.most_common()), worker_receipts=receipts,
        original_outputs_unchanged=True, labels_modified=False, closed_sets_opened=False,
        no_full_rank=True, no_new_recall=True, no_coefficient_choice_proposal=True,
        unknowns=['Rang exact au-delà de 100', 'Généralisation hors des 19 ratés',
                  'Causalité : les écarts sont algébriques, pas des effets de retrait de voie'],
        heavy_artifacts=str(OUT), command=f'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/{Path(__file__).name}')
    write(OUT/'result.json', result); write(RESULT, result)
    by_id = {r['query_id']: r for r in completed} if valid else {}
    lines = ['# Notes des 19 ratés V142 — 14 octobre 2026', '',
        f'**État : `{status}`.** Fidélité : **{len(proofs)}/19** requêtes. ',
        f'Pic résident observé : **{result["peak_resident_bytes"]/1024**3:.3f} Gio** ; plafond **6 Gio**. Durée : **{result["elapsed_seconds"]:.2f} s**.', '',
        '**Mesuré.** Même moteur Tantivy, mêmes univers, cinq coefficients gelés, quantification 20 bits. Un processus par requête ; aucune relance complète, aucun nouveau recall.', '',
        '| query_id | nom CRM court | note label | note 100e | écart | 3 signaux en retard | passe avec un autre modèle ? |',
        '|---|---|---:|---:|---:|---|---|']
    for miss in misses:
        row = by_id.get(miss['query_id']); name = miss['crm_name'][:45].replace('|', '/')
        if row:
            target = row['documents']['label']['score']; cut = row['documents']['cut100']['score']
            deficits_text = '; '.join(f'{d["signal"]} (+{d["deficit"]:.6f})' for d in row['top3_deficits'])
            models = ', '.join(k for k, v in row['models'].items() if k != row['model_id'] and v['passes_actual_top100'])
            answer = 'oui ('+models+')' if models else 'non'
            lines.append(f'| {miss["query_id"]} | {name} | {target:.6f} | {cut:.6f} | {row["total_deficit"]:.6f} | {deficits_text} | {answer} |')
        else:
            lines.append(f'| {miss["query_id"]} | {name} | inconnu | inconnu | inconnu | non conclu | inconnu |')
    lines += ['', '## Agrégat des familles']
    if valid:
        lines += ['Nombre de requêtes où la famille figure parmi les trois retards principaux (familles dédupliquées par requête).', '']
        lines += [f'- {name} : **{count}/19**.' for name, count in counts.most_common()]
        lines += ['', 'Les 58 contributions sont coefficient × valeur. Les 51 valeurs numériques sont avant quantification des deux blocs : elles ne sont pas arrondies séparément. Les bonus et le résidu de quantification/arithmétique native referment la note.',
            'Le bonus de nouvelles sources est un **maximum**, pas une somme : ses alternatives ne sont pas créditées plusieurs fois.',
            '« Autre modèle » est testé contre son **vrai top100 national recalculé**, pas contre la seule note du 100e enregistré. Les mêmes préparateurs BANO/OSM recalculent leurs bonus avec ces poids ; le JSON conserve aussi la note de ce 100e enregistré.',
            'Le premier candidat, le seuil, les 58 signaux, les bonus et les cinq modèles sont détaillés dans le JSON voisin. Aucune règle de choix de modèle n’est proposée.']
    else:
        lines += ['**Aucune conclusion scientifique : arrêt avant fidélité complète ou décomposition validée.**', '',
                  f'Blocage mesuré : `{failure}`. Artefacts partiels conservés sous `{OUT}`.']
    lines += ['', '**Inconnu.** Rang exact des labels hors100 : « > 100, rang non calculé ». Les écarts expliquent la compétition mesurée, pas une causalité ni un gain de recall. Aucun jeu fermé ou label modifié.', '']
    with RESULT.with_suffix('.md').open('x') as stream:
        stream.write('\n'.join(lines))
    print('REPORT', status, len(proofs), '/19', result['peak_resident_bytes'], result['elapsed_seconds'], flush=True)


def main():
    started = time.monotonic(); inputs = Inputs()
    inputs.json(STOP_V160)
    if OUT.exists() or RESULT.exists() or RESULT.with_suffix('.md').exists():
        raise ValueError('Namespace et rapports doivent être nouveaux')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid in (os.getpid(), os.getppid()):
            continue
        command = ' '.join(process.info['cmdline'] or [])
        executable = Path((process.info['cmdline'] or [''])[0]).name.lower()
        if executable.startswith('python') and ('-m scripts.embeddinggemma2_' in command or 'retrieval_miss_score_explain.py' in command):
            raise RuntimeError('Autre calcul lourd détecté: '+str(process.pid))
    OUT.mkdir(parents=True)
    for name in ('inputs', 'fidelity', 'explain', 'logs'):
        (OUT/name).mkdir()
    misses = inputs.json(ACCESS)['queries']
    write(OUT/'recipe.json', dict(cap_resident_bytes=CAP, one_process_per_query=True,
          queries=[r['query_id'] for r in misses], fidelity='ALL19_SAME_ORDERED100_AND_ABS_SCORE_1E-5_BEFORE_EXPLANATION',
          exact_original_native_search=True, no_query_debug_repr=True, no_full_rank=True,
          alternative_weights='ALL5_WITH_EXACT_SOURCE_PREPARER_KERNELS_AND_REAL_COLLECTOR100',
          diagnostic_only=True, code=inputs.refs[str(Path(__file__))] if str(Path(__file__)) in inputs.refs else str(Path(__file__))))
    proofs, completed, receipts, failure = [], [], [], None
    status = 'STOP_NO_SCIENTIFIC_CONCLUSION'
    try:
        prepare_inputs(inputs, misses)
        for phase in ('fidelity', 'explain'):
            for miss in misses:
                qid = miss['query_id']; logpath = OUT/'logs'/(phase+'_'+stem(qid)+'.log')
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', phase, '--query-id', qid]
                child_started = time.monotonic(); child_peak = 0; reason = None
                with logpath.open('xb') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                    try:
                        while child.poll() is None:
                            try:
                                rss = psutil.Process(child.pid).memory_info().rss
                                child_peak = max(child_peak, rss)
                                total_rss = rss+psutil.Process().memory_info().rss
                                if total_rss > CAP:
                                    reason = f'STOP_RSS_6GIB: child={rss}, parent+child={total_rss}'
                                    child.terminate(); child.wait(timeout=15); break
                                if time.monotonic()-child_started > 1800:
                                    reason = 'STOP_WORKER_TIMEOUT_1800S'; child.terminate(); child.wait(timeout=15); break
                            except psutil.NoSuchProcess:
                                pass
                            time.sleep(.2)
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                child.kill(); child.wait()
                path = OUT/phase/(stem(qid)+'.json')
                receipt = dict(query_id=qid, phase=phase, pid=child.pid, exit_code=child.returncode,
                               peak_resident_bytes=child_peak, elapsed_seconds=time.monotonic()-child_started,
                               owned_child_exit_confirmed=True, stop_reason=reason, log=str(logpath))
                if path.exists():
                    row = read(path); receipt['peak_resident_bytes'] = max(child_peak, row['peak_rss_bytes'])
                receipts.append(receipt); write(OUT/phase/(stem(qid)+'_exit.json'), receipt)
                if child.returncode or reason or not path.exists():
                    errorpath = OUT/phase/(stem(qid)+'_error.json')
                    detail = read(errorpath) if errorpath.exists() else None
                    raise RuntimeError(str(dict(query_id=qid, phase=phase, reason=reason, child_error=detail, log=str(logpath))))
                if phase == 'fidelity':
                    proofs.append(row['fidelity'])
                else:
                    completed.append(row)
                print('PROGRESS', phase, len(proofs) if phase == 'fidelity' else len(completed), '/19', qid, receipt['peak_resident_bytes'], flush=True)
            if phase == 'fidelity':
                write(OUT/'fidelity_complete.json', dict(queries_reproduced=19, checks=proofs))
        inputs.unchanged(); status = 'COMPLETE_FAITHFUL_19_MISS_SCORE_DIAGNOSTIC'
    except Exception as exc:
        failure = repr(exc)
        print('STOP', failure, flush=True)
    publish(misses, completed, proofs, status, failure, receipts, started)
    return 0 if status.startswith('COMPLETE') else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('fidelity', 'explain'))
    parser.add_argument('--query-id')
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.query_id, args.worker)
        except Exception as exc:
            write(OUT/args.worker/(stem(args.query_id)+'_error.json'),
                  dict(error=repr(exc), peak_rss_bytes=peak(), traceback=traceback.format_exc()))
            raise
    else:
        sys.exit(main())
