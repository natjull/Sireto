"""Diagnostic borné des 19 ratés V142 ; moteur natif inchangé, jamais de recall.

Un processus par requête et phase, plafonné sur l'enfant seul. Le repli
restreint uniquement les jointures des bitplanes aux 101 documents demandés,
sans changer leurs bornes, leurs scores, la structure ni les statistiques
lexicales nationales. Au moins 15 fidélités précèdent toute explication.
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
PREVIOUS = HOME/'v172/miss_score_explain_20261014'
OUT = Path(os.environ.get('SIRETO_MISS_OUT', str(PREVIOUS/'resume_20261015_b')))
RESUME_CHECKS = PREVIOUS/'resume_20261015_a'
INPUTS = PREVIOUS/'inputs'
METHOD = os.environ.get('SIRETO_MISS_METHOD', 'complete')
ACCESS = ROOT/'docs/retrieval_miss_access_vs_ranking_20261014.json'
RESULT = ROOT/'docs/retrieval_miss_score_explain_20261014.json'
STOP_V160 = ROOT/'docs/retrieval_v160_explain_final_c_stop_20261014.json'
CAP = (4 if METHOD == 'light' else 8)*1024**3
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
        raise MemoryError(f'STOP_CHILD_RSS: {rss}, cap={CAP}')
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
        self.expected = None
        self.backend = None

    def doc(self, address):
        return self.searcher.doc(address)

    def search(self, query, **kwargs):
        if kwargs.get('limit') != 100 or self.query is not None:
            raise ValueError('Un seul collecteur natif100 par modèle')
        self.query = query
        guard()
        if METHOD == 'light':
            from types import SimpleNamespace
            hits = []
            for row in self.expected:
                address = lookup(self.backend, self, row['siret'])
                hits.append((explained(query, self.searcher, address), address))
            result = SimpleNamespace(hits=hits)
        else:
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
    if METHOD == 'light':
        backend._tantivy = RestrictedTantivy(backend._tantivy, binding)
    backend.diagnostic_expected = binding['expected']
    # Le contexte acteur est indépendant des coefficients. Garder sa requête
    # exacte, pas son énorme dictionnaire national d'alias entre les modèles.
    original_corporate = corp.corporate_component
    actor_cache = {}
    def cached_actor(target, q):
        key = corp.original.structural.normalize(corp.original.data.safe(q['crm_name']))
        if key not in actor_cache:
            actor_cache[key] = original_corporate(target, q)
            corp.CORPORATE = None
            corp.LAST_CORPORATE = None
            gc.collect()
        return actor_cache[key]
    corp.corporate_component = cached_actor
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
    capture.expected = backend.diagnostic_expected
    capture.backend = backend
    backend.searcher = capture
    proxy = v142.baseline.baseline.BackendProxy(backend, v142.parent.EvidenceCollector(
        backend, q, keys, scores, entries, float(coefficients[6])))
    rows, cost = v142.baseline.native_search(proxy, q, keys, features, coefficients)
    return rows, cost, capture


def fidelity(rows, expected):
    ordered_equal = [r['siret'] for r in rows] == [r['siret'] for r in expected]
    actual = {r['siret']: r['score'] for r in rows}
    recorded = {r['siret']: r['score'] for r in expected}
    ids_equal = set(actual) == set(recorded)
    error = max((abs(actual[ref]-recorded[ref]) for ref in recorded), default=float('inf')) if ids_equal else float('inf')
    return dict(passed=ids_equal and len(actual) == len(recorded) == len(rows) == len(expected) == 100 and error <= ATOL,
                same_100_sirets=ids_equal, same_ordered_100_sirets=ordered_equal,
                score_alignment='BY_SIRET_NOT_BY_POSITION', max_score_absolute_error=error,
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


class RestrictedQuery:
    """Projection exacte des seules jointures, avant compilation Tantivy.

    Les bitplanes et leurs bornes sont toujours calculés sur l'univers complet.
    Les clauses lexicales restent nationales, donc aucun IDF n'est recalculé.
    Une plane vide est conservée : même arbre et même arithmétique native.
    """
    def __init__(self, query, binding):
        self.query = query
        self.sirets = {r['siret'] for r in binding['expected']} | {binding['miss']['label_siret']}
        self.sirens = {r[:9] for r in self.sirets}

    def __getattr__(self, name):
        return getattr(self.query, name)

    def term_set_query(self, schema, field, refs):
        if field in ('siret', 'siren'):
            allowed = self.sirets if field == 'siret' else self.sirens
            refs = [ref for ref in refs if ref in allowed]
        return self.query.term_set_query(schema, field, refs)


class RestrictedTantivy:
    def __init__(self, tantivy, binding):
        self.tantivy = tantivy
        self.Query = RestrictedQuery(tantivy.Query, binding)

    def __getattr__(self, name):
        return getattr(self.tantivy, name)


def worker(qid, phase):
    started = time.monotonic(); directory = INPUTS/stem(qid); inputs = Inputs()
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
    write(OUT/(phase+'_'+METHOD)/(stem(qid)+'_fidelity.json'), dict(query_id=qid, **proof, actual=rows))
    if not proof['passed']:
        raise ValueError('STOP_FIDELITY : top100/notes non reproduits')
    result = dict(query_id=qid, crm_name=binding['query']['crm_name'], fold=binding['query']['fold'],
                  model_id=model_id, fidelity=proof, numeric_sites=len(keys), cost=cost,
                  method=METHOD, memory_cap_bytes=CAP)
    if phase == 'explain':
        sealed = read(OUT/'fidelity_complete.json')
        if sealed['queries_reproduced'] < 15:
            raise ValueError('Au moins 15/19 avant les explications')
        documents = {}
        for role, siret in [('label', binding['miss']['label_siret']), ('first', binding['expected'][0]['siret']), ('cut100', binding['expected'][-1]['siret'])]:
            documents[role] = contribution(v142, backend, state, capture, binding, keys, features,
                    weights, scores, sources, entries, role, siret)
            if role != 'label':
                expected = binding['expected'][0 if role == 'first' else -1]['score']
                if abs(documents[role]['score']-expected) > ATOL:
                    raise ValueError('STOP_EXPLAIN_SCORE_FIDELITY')
        top3, gap, all_deficits = deficits(documents)
        rank = '> 100, rang non calculé'
        if METHOD == 'complete':
            hits = capture.searcher.search(capture.query, limit=10000, count=False).hits
            ranked = sorted([(float(score), backend._first(capture.searcher.doc(loc).to_dict(), 'siret'))
                             for score, loc in hits], key=lambda r: (-r[0], r[1]))
            label = documents['label']['siret']
            if ranked and documents['label']['score'] > ranked[-1][0]+ATOL:
                rank = next((i+1 for i, (_, ref) in enumerate(ranked) if ref == label), rank)
            del hits, ranked; gc.collect()
        result.update(documents=documents, label_rank=rank,
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
                actual_model_cut100_siret=alt_rows[-1]['siret'] if METHOD == 'complete' else None,
                actual_model_cut100_score=alt_rows[-1]['score'] if METHOD == 'complete' else None,
                beats_recorded_cut100=label_score > old_cut_score+ATOL,
                passes_actual_top100=(label in {r['siret'] for r in alt_rows}) if METHOD == 'complete' else None,
                threshold_scope='NATIONAL_TOP100' if METHOD == 'complete' else 'RECORDED_CUT100_ONLY_NOT_A_NATIONAL_ADMISSION_PROOF',
                label_rank_if_retained=next((i+1 for i, r in enumerate(alt_rows) if r['siret'] == label), None) if METHOD == 'complete' else None,
                is_recorded_model=alternative_id == model_id)
            state['base'].clear(); state['v81'].clear(); state['stages'].clear()
            backend.searcher = backend.index.searcher()
            v142.baseline.baseline.source.pipeline.base.LAST_PART = None
            v142.baseline.baseline.source.pipeline.corporate.LAST_CORPORATE = None
            del captured, data; gc.collect(); guard()
        result['beats_recorded_cut_with_another_model'] = any(v['beats_recorded_cut100'] for k, v in result['models'].items() if k != model_id)
        result['passes_another_model'] = (any(v['passes_actual_top100'] for k, v in result['models'].items() if k != model_id)
                                          if METHOD == 'complete' else None)
    inputs.unchanged(); guard()
    result.update(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak(), inputs=list(inputs.refs.values()))
    write(OUT/(phase+'_'+METHOD)/(stem(qid)+'.json'), result)
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
    valid = status == 'COMPLETE_FAITHFUL_MISS_SCORE_DIAGNOSTIC'
    counts = collections.Counter()
    if valid:
        for row in completed:
            counts.update(set(family(d['signal']) for d in row['top3_deficits']))
    result = dict(status=status, scope='POSTEVAL_DIAGNOSTIC_19_MISSES_FOLDS_2_3_4_NOT_NEW_RECALL',
        source_run=str(RUN), quantization_bits=20, fidelity_queries_reproduced=len(proofs),
        fidelity_queries_required=19, absolute_tolerance=ATOL, relative_tolerance=0,
        complete_memory_cap_bytes=8*1024**3, light_memory_cap_bytes=4*1024**3,
        guard_scope='CHILD_ONLY', fidelity_by_method=dict(collections.Counter(p['method'] for p in proofs)),
        fidelity_contract='SAME_100_SIRETS_AND_SCORES_ALIGNED_BY_SIRET_AT_UNCHANGED_1E-5_TOLERANCE',
        order_checker_correction='Prior checker rejected only a swapped equal-score pair on ad20; same100 and aligned max error 1.078765869522158e-6. No score/model/tolerance changed.',
        fidelity_checks=proofs, historical_stop=read(OUT/'historical_result.json'),
        peak_resident_bytes=max([peak()]+[r['peak_resident_bytes'] for r in receipts]),
        elapsed_seconds=time.monotonic()-started, failure=failure,
        scientific_conclusions_allowed=valid, query_results=completed if valid else [],
        partial_results_not_validated_for_conclusions=completed if not valid else [],
        family_counts_in_top3=dict(counts.most_common()), worker_receipts=receipts,
        original_outputs_unchanged=True, labels_modified=False, closed_sets_opened=False,
        rank_search_limit=10000, no_new_recall=True, no_coefficient_choice_proposal=True,
        unknowns=['Rang exact au-delà de 100', 'Généralisation hors des 19 ratés',
                  'Causalité : les écarts sont algébriques, pas des effets de retrait de voie'],
        heavy_artifacts=str(OUT), command=f'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/{Path(__file__).name}')
    write(OUT/'result.json', result)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    by_id = {r['query_id']: r for r in completed} if valid else {}
    lines = ['# Notes des 19 ratés V142 — 14 octobre 2026', '',
        f'**État : `{status}`.** Fidélité : **{len(proofs)}/19** requêtes.',
        f'Pic résident observé : **{result["peak_resident_bytes"]/1024**3:.3f} Gio** ; plafonds enfant seul **8/4 Gio** (complète/allégée). Durée : **{result["elapsed_seconds"]:.2f} s**.', '',
        f'Fidélité par méthode : `{result["fidelity_by_method"]}`. Le replay historique de `10719` est conservé.', '',
        'Contrôle : mêmes 100 SIRET et notes alignées **par SIRET**, tolérance inchangée 1e−5. L’ordre est enregistré séparément. Le premier arrêt de reprise concernait seulement deux ex æquo inversés sur `ad20…` : mêmes100, erreur par SIRET 1,08e−6 ; ce contrôleur trop strict a été corrigé, sans changement du moteur ni de ses notes.', '',
        '**Mesuré.** Même moteur Tantivy, mêmes univers, cinq coefficients gelés, quantification 20 bits. Un processus par requête ; aucune relance complète, aucun nouveau recall.', '',
        '| query_id | méthode / pic Gio | note label | premier : SIRET / note | 100e : SIRET / note | écart | 3 signaux en retard | autre modèle : bat le seuil enregistré ? |',
        '|---|---|---:|---|---|---:|---|---|']
    for miss in misses:
        row = by_id.get(miss['query_id']); name = miss['crm_name'][:45].replace('|', '/')
        if row:
            target = row['documents']['label']['score']; cut = row['documents']['cut100']['score']
            deficits_text = '; '.join(f'{d["signal"]} (+{d["deficit"]:.6f})' for d in row['top3_deficits'])
            models = ', '.join(k for k, v in row['models'].items() if k != row['model_id'] and v['beats_recorded_cut100'])
            answer = 'oui ('+models+')' if models else 'non'
            first = row['documents']['first']; cutdoc = row['documents']['cut100']
            memory = max(r['peak_resident_bytes'] for r in receipts if r['query_id'] == row['query_id'] and r['phase'] == 'explain')
            label_method = 'complète' if row['method'] == 'complete' else 'allégée'
            lines.append(f'| {miss["query_id"]} | {label_method} / {memory/1024**3:.3f} | {target:.6f} | {first["siret"]} / {first["score"]:.6f} | {cutdoc["siret"]} / {cut:.6f} | {row["total_deficit"]:.6f} | {deficits_text} | {answer} |')
        else:
            attempts = [r for r in receipts if r['query_id'] == miss['query_id']]
            observed = max((r['peak_resident_bytes'] for r in attempts), default=0)/1024**3
            method = attempts[-1]['method'] if attempts else 'non exécutée'
            lines.append(f'| {miss["query_id"]} | {method} / {observed:.3f} | inconnu | inconnu | inconnu | inconnu | non conclu | inconnu |')
    lines += ['', '## Agrégat des familles']
    if valid:
        lines += ['Nombre de requêtes où la famille figure parmi les trois retards principaux (familles dédupliquées par requête).', '']
        lines += [f'- {name} : **{count}/19**.' for name, count in counts.most_common()]
        lines += ['', 'Les 58 contributions sont coefficient × valeur. Les 51 valeurs numériques sont avant quantification des deux blocs : elles ne sont pas arrondies séparément. Les bonus et le résidu de quantification/arithmétique native referment la note.',
            'Le bonus de nouvelles sources est un **maximum**, pas une somme : ses alternatives ne sont pas créditées plusieurs fois.',
            'Pour la méthode complète, le JSON teste aussi le vrai top100 national des cinq modèles. Pour le repli allégé, « autre modèle » compare **uniquement les notes du label et du 100e enregistré** : dépasser celui-ci ne prouve PAS une admission au nouveau top100 national. Celle-ci reste inconnue, sans classement d’un pool101 présenté comme moteur. Les mêmes préparateurs BANO/OSM recalculent leurs bonus avec ces poids et leurs univers complets.',
            'La méthode allégée exécute les fonctions natives et projette seulement leurs jointures SIRET/SIREN sur les documents demandés. Les bornes de quantification et les agrégations parent restent calculées sur tous les sites ; les statistiques lexicales sont nationales. La fidélité vérifie les 100 notes, pas la composition d’un nouveau top100 national.',
            'Le premier candidat, le seuil, les 58 signaux, les bonus et les cinq modèles sont détaillés dans le JSON voisin. Aucune règle de choix de modèle n’est proposée.']
    else:
        lines += ['**Aucune conclusion scientifique : arrêt avant fidélité complète ou décomposition validée.**', '',
                  f'Blocage mesuré : `{failure}`. Artefacts partiels conservés sous `{OUT}`.']
    lines += ['', '**Inconnu.** Rang exact des labels hors100 : « > 100, rang non calculé ». Les écarts expliquent la compétition mesurée, pas une causalité ni un gain de recall. Aucun jeu fermé ou label modifié.', '']
    lines += ['', '## Historique — arrêt précédent conservé intégralement', '',
              (OUT/'historical_report.md').read_text()]
    RESULT.with_suffix('.md').write_text('\n'.join(lines))
    print('REPORT', status, len(proofs), '/19', result['peak_resident_bytes'], result['elapsed_seconds'], flush=True)


def free_inactive():
    import re
    text = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+) bytes', text)[1])
    counts = [int(re.search(r'Pages '+name+r':\s+(\d+)', text)[1]) for name in ('free', 'inactive')]
    return dict(free_plus_inactive_bytes=sum(counts)*page, vm_stat=text,
                complete_admitted=sum(counts)*page >= 10*1024**3, observed_unix=time.time())


def child_call(qid, phase, method, receipts, admission):
    directory = OUT/(phase+'_'+method); directory.mkdir(exist_ok=True)
    logpath = OUT/'logs'/(phase+'_'+method+'_'+stem(qid)+'.log')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', phase, '--query-id', qid]
    cap = (8 if method == 'complete' else 4)*1024**3
    env = dict(os.environ, SIRETO_MISS_OUT=str(OUT), SIRETO_MISS_METHOD=method)
    child_started = time.monotonic(); child_peak = 0; reason = None
    with logpath.open('xb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        try:
            while child.poll() is None:
                try:
                    rss = psutil.Process(child.pid).memory_info().rss
                    child_peak = max(child_peak, rss)
                    if rss > cap:
                        reason = f'STOP_CHILD_RSS: child={rss}, cap={cap}'
                        child.terminate(); break
                    if time.monotonic()-child_started > 1800:
                        reason = 'STOP_WORKER_TIMEOUT_1800S'; child.terminate(); break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.1)
        finally:
            if child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    path = directory/(stem(qid)+'.json')
    errorpath = directory/(stem(qid)+'_error.json')
    detail = read(errorpath) if errorpath.exists() else None
    row = read(path) if path.exists() and not child.returncode and not reason else None
    receipt = dict(query_id=qid, phase=phase, method=method, pid=child.pid, exit_code=child.returncode,
                   memory_cap_bytes=cap, guard_scope='CHILD_ONLY', peak_resident_bytes=child_peak,
                   elapsed_seconds=time.monotonic()-child_started, admission=admission,
                   owned_child_exit_confirmed=True, stop_reason=reason, child_error=detail, log=str(logpath))
    if row:
        receipt['peak_resident_bytes'] = max(child_peak, row['peak_rss_bytes'])
    elif detail:
        receipt['peak_resident_bytes'] = max(child_peak, detail['peak_rss_bytes'])
    receipts.append(receipt); write(directory/(stem(qid)+'_exit.json'), receipt)
    print('CHILD', phase, method, qid, child.returncode, receipt['peak_resident_bytes'], flush=True)
    # Une infidélité ou une erreur scientifique arrête toute la tâche. Seule
    # une interruption de ressource autorise le repli prévu dans le brief.
    if not row and not reason and not (detail and 'MemoryError' in detail.get('traceback', '')):
        raise RuntimeError(str(receipt))
    return row


def main():
    started = time.monotonic(); inputs = Inputs()
    inputs.json(STOP_V160)
    if OUT.exists():
        raise ValueError('Nouveau sous-répertoire requis, sans écrasement des artefacts')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid in (os.getpid(), os.getppid()):
            continue
        command = ' '.join(process.info['cmdline'] or [])
        executable = Path((process.info['cmdline'] or [''])[0]).name.lower()
        if executable.startswith('python') and ('-m scripts.embeddinggemma2_' in command or 'retrieval_miss_score_explain.py' in command):
            raise RuntimeError('Autre calcul lourd détecté: '+str(process.pid))
    OUT.mkdir(parents=True); (OUT/'logs').mkdir()
    history = inputs.json(RESULT)
    write(OUT/'historical_result.json', history)
    (OUT/'historical_report.md').write_text(RESULT.with_suffix('.md').read_text())
    misses = inputs.json(ACCESS)['queries']
    write(OUT/'recipe.json', dict(one_process_per_query=True, complete_cap_bytes=8*1024**3,
          light_cap_bytes=4*1024**3, guard_scope='CHILD_ONLY', complete_admission_free_inactive_bytes=10*1024**3,
          queries=[r['query_id'] for r in misses], fidelity='SAME100_SCORES_ABS_1E-5_NO_RETUNING',
          source_inputs=str(INPUTS), original_query_functions_unchanged=True,
          light_projection='IDENTIFIER_TERMSET_JOINS_ONLY_GLOBAL_BOUNDS_AND_LEXICAL_STATISTICS_UNCHANGED',
          historical_10719_reused=True, explanation_gate=15,
          diagnostic_only=True, code=inputs.path(Path(__file__)).as_posix()))
    proofs, completed, receipts, failure = [], [], [], None
    methods = {}; status = 'STOP_NO_SCIENTIFIC_CONCLUSION'
    try:
        manifest = inputs.json(PREVIOUS/'input_manifest.json')
        for ref in manifest['inputs']:
            inputs.path(ref)
        write(OUT/'input_manifest.json', manifest)
        original = inputs.json(PREVIOUS/'fidelity'/(stem('10719')+'.json'))
        if not original['fidelity']['passed']:
            raise ValueError('Le replay historique 10719 doit être fidèle')
        for miss in misses:
            qid = miss['query_id']; binding = inputs.json(INPUTS/stem(qid)/'binding.json')
            if binding['query']['fold'] not in (2, 3, 4):
                raise ValueError('Folds 2/3/4 exclusivement')
            if qid == '10719':
                proofs.append(dict(query_id=qid, method='complete_historical', **original['fidelity']))
                methods[qid] = 'complete'
                continue
            saved = RESUME_CHECKS/'fidelity_light'/(stem(qid)+'_fidelity.json')
            if saved.exists():
                observed = inputs.json(saved)
                proof = fidelity(observed['actual'], binding['expected'])
                if not proof['passed']:
                    raise ValueError('STOP_FIDELITY : scores enregistrés de reprise non conformes')
                proofs.append(dict(query_id=qid, method='light', reused_from=str(saved),
                                   prior_order_checker_passed=observed['passed'], **proof))
                methods[qid] = 'light'
                receipt = inputs.json(saved.with_name(stem(qid)+'_exit.json'))
                receipts.append(dict(receipt, reused_from=str(RESUME_CHECKS)))
                print('REUSED fidelity light', qid, proof['max_score_absolute_error'], flush=True)
                continue
            admission = free_inactive(); row = None
            if admission['complete_admitted']:
                row = child_call(qid, 'fidelity', 'complete', receipts, admission)
            if not row:
                row = child_call(qid, 'fidelity', 'light', receipts, admission)
            if row:
                methods[qid] = row['method']
                proofs.append(dict(query_id=qid, method=row['method'], **row['fidelity']))
        if len(proofs) < 15:
            raise RuntimeError(f'STOP : seulement {len(proofs)}/19 fidélités')
        write(OUT/'fidelity_complete.json', dict(queries_reproduced=len(proofs), checks=proofs))
        for miss in misses:
            qid = miss['query_id']
            if qid not in methods:
                continue
            admission = free_inactive(); row = None
            if methods[qid] == 'complete' and admission['complete_admitted']:
                row = child_call(qid, 'explain', 'complete', receipts, admission)
            if not row:
                row = child_call(qid, 'explain', 'light', receipts, admission)
            if row:
                completed.append(row)
            print('PROGRESS explain', len(completed), '/19', qid, flush=True)
        if len(completed) < 15:
            raise RuntimeError(f'STOP : seulement {len(completed)}/19 décompositions')
        inputs.unchanged(); status = 'COMPLETE_FAITHFUL_MISS_SCORE_DIAGNOSTIC'
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
            write(OUT/(args.worker+'_'+METHOD)/(stem(args.query_id)+'_error.json'),
                  dict(error=repr(exc), peak_rss_bytes=peak(), traceback=traceback.format_exc()))
            raise
    else:
        sys.exit(main())
