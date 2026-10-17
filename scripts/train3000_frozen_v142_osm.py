"""Historical OSM source/V143 deltas, v174-only writes; fidelity-gated.

Full local PBF reconstruction, never a replacement by the old526 object spool.
The brief10 controller supplies the resource guard and serial execution.
"""
import ast
import gc
import hashlib
from pathlib import Path
import resource
import textwrap
import time
from types import SimpleNamespace

import numpy as np
from scripts import train3000_frozen_v142_fidelity as mini

SOURCE_ROOT = mini.bano.cohort.preflight.H/'v129/osm_local_20261013_night_e'


def _body(text, start, end, signature, ending, namespace):
    if text.count(start)!=1 or text.count(end)!=1:
        raise ValueError('STOP_HISTORICAL_OSM_CONSTRUCTOR_NOT_FOUND')
    body = text[text.index(start):text.index(end)]
    exec(compile('def build('+signature+'):\n'+textwrap.indent(textwrap.dedent(body), '    ')+ending,
        'historical_OSM_body', 'exec'), namespace)
    return namespace['build'], hashlib.sha256(body.encode()).hexdigest()


def context(panel, geometries, directory):
    if not panel or any(set(q)!=set(mini.bano.cohort.old.CRM) or q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Only original query-only CRM8 folds234')
    from scripts import embeddinggemma2_v129_osm_local_source as original
    from scripts import embeddinggemma2_v129_osm_disk_source as disk
    from scripts import embeddinggemma2_v129_osm_disk_category_source as category
    generation = mini.read(SOURCE_ROOT/'generation.json')
    mini.verified(generation['implementation'])
    binding = mini.read(SOURCE_ROOT/'category_storage_binding.json')
    source_path = mini.verified(binding['source_implementation'])
    mini.verified(binding['implementation'])
    migration = mini.read(mini.verified(generation['migration']))
    mini.verified(migration['original_implementation'])
    download = mini.read(mini.verified(generation['download']))
    pbf = mini.verified(download['pbf'])
    if pbf.stat().st_size != 5097546976:
        raise ValueError('STOP_OSM_PBF_SNAPSHOT_DRIFT')
    directory.mkdir(parents=True)
    root = directory/'source'
    root.mkdir()

    def write(path, value):
        path = Path(path)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('OSM may write only within its new v174 namespace')
        if path.name == 'state.json':
            path = path.with_name('state_'+str(time.time_ns())+'.json')
        mini.save(path, value)

    text = source_path.read_text()
    tree = ast.parse(Path(category.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    literals = {n.targets[0].id:ast.literal_eval(n.value) for n in main.body
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('before','after')}
    if text.count(literals['before'])!=1 or text.count("('node/%',)")!=3:
        raise ValueError('STOP_OSM_CATEGORY_STORAGE_BINDING')
    text = text.replace(literals['before'],literals['after']).replace("('node/%',)","('%|node/%',)")
    namespace = dict(__name__='v174_frozen_category_disk_source',__file__=str(source_path))
    exec(compile(text,str(source_path),'exec'),namespace)
    # Original classes now resolve ROOT and I/O only in this private dictionary.
    namespace.update(ROOT=root, io=SimpleNamespace(**dict(vars(disk.io),write=write)),
        original=SimpleNamespace(**dict(vars(original),guard=mini.guard)))
    queries, query_sha = _body(Path(original.__file__).read_text(), '    queries = []\n',
        '    recipe=dict(all_queries=526', 'panel,geometry', '\n    return queries\n', dict(vars(original)))
    queries = queries(panel,geometries)
    mini.save(directory/'queries.json', queries)
    mini.save(root/'launch.json',dict(implementation=binding['source_implementation'],
        historical_category_policy=binding['implementation'],queries=mini.record(directory/'queries.json'),
        pbf=download['pbf'],query_only=True,raw_labels_read=False))
    build, handler_sha = _body(text, '    tracker=osmium.IdTracker();nodes=Locations(tracker,resume)\n',
        "    files={n:record(ROOT/(n+'.json'))", 'queries,pbf,resume=False',
        '\n    return shapes\n', namespace)
    # The spatial lookup itself is the original class, with the supplied query
    # geometry. No object or node source from the old526 spool is substituted.
    namespace['lookup'] = original.Neighbourhoods(queries)
    shapes = build(queries,pbf)
    projected = {}
    import ijson
    with (root/'query_context.json').open('rb') as stream:
        for qid,value in ijson.kvitems(stream,'',use_float=True):
            # The exact retrieval-only projection predeclared by V129.
            projected[qid] = {k:value[k] for k in ('spatial_crs_compatible','pois','addresses')}
            mini.guard()
    mini.save(directory/'query_context_for_retrieval.json',projected)
    mini.save(directory/'construction.json',dict(query_body_sha256=query_sha,handler_body_sha256=handler_sha,
        source=binding['source_implementation'],category_policy=binding['implementation'],pbf=download['pbf'],
        queries=len(panel),raw_labels_read=False,reused_old526_object_spool=False,
        execution_only_changes=['new v174 root','supplied query-only panel/geometries','brief10 admitted worker memory guard']))
    return projected,shapes,dict(query=query_sha,handler=handler_sha)


def fidelity(directory):
    start = time.monotonic()
    run = mini.bano.cohort.preflight.RUN
    launch = mini.read(run/'launch.json')
    panel = mini.read(mini.bano.cohort.OUT/'pilot_fidelity_queries.json')
    sample = mini.read(mini.bano.cohort.OUT/'sample.json')
    ids = set(sample['fidelity_query_ids'])
    original_panel = mini.read(mini.verified(launch['panel']))
    if len(panel)!=20 or {q['query_id']:q for q in panel}!={q['query_id']:q for q in original_panel if q['query_id'] in ids}:
        raise ValueError('STOP_SEALED_OSM_PILOT_SCOPE')
    geometries = {r['query_id']:r['geometry'] for r in mini.read(mini.verified(launch['geometry'])) if r['query_id'] in ids}
    fresh,shapes,bodies = context(panel,geometries,directory/'reconstruction')
    pg = mini.read(mini.verified(launch['parent_prepared']))
    expected = mini.subset(pg['source_context'],ids)
    # The metadata points to the full source; apply the SAME historical V129
    # retrieval projection on both sides before comparing, never compare a
    # three-field view to a five-field raw record.
    expected = {qid:{k:value[k] for k in ('spatial_crs_compatible','pois','addresses')}
        for qid,value in expected.items()}
    expected_shapes = mini.read(mini.verified(pg['shapes']))
    matches = {qid:mini.canonical(fresh[qid])==mini.canonical(expected[qid]) and
        mini.canonical(shapes[qid])==mini.canonical(expected_shapes[qid]) for qid in ids}
    if not all(matches.values()):
        mini.save(directory/'failure.json',dict(status='STOP_OSM_CONTEXT_NONFIDELITY',
            context_matches=sum(matches.values()),failed_query_ids=sorted(q for q in ids if not matches[q]),
            bodies=bodies,expected_source_context=pg['source_context'],expected_shapes=pg['shapes']))
        raise ValueError('STOP_OSM_CONTEXT_NONFIDELITY')
    del expected,expected_shapes
    gc.collect()
    from scripts import embeddinggemma2_v142_bano_native_ablation as v142
    from scripts.retrieval_miss_score_explain import preparer_kernel
    kernel,kernel_sha = preparer_kernel(v142,'osm')
    fg,cal = (mini.read(run/p) for p in ('features_generation.json','calibration/generation.json'))
    assignment = mini.read(mini.verified(cal['assignment']))
    all_keys = np.load(mini.verified(launch['keys']),mmap_mode='r',allow_pickle=False)
    xy = np.load(mini.verified(launch['xy']),mmap_mode='r',allow_pickle=False)
    public = np.load(mini.verified(launch['public_mask']),mmap_mode='r',allow_pickle=False)
    rows = []
    for q in panel:
        mini.guard();qid=q['query_id']
        model=mini.read(mini.verified(cal['models'][str(assignment[qid])]))
        if qid not in model['heldout_queries'] or qid in model['train_queries']:
            raise ValueError('STOP_OSM_MODEL_NOT_HELDOUT')
        with np.load(mini.verified(fg['features'][qid]),allow_pickle=False) as f:
            keys,features=f['keys'],f['features']
        geom,named = kernel(keys,features,all_keys,xy,public,np.asarray(model['coefficients'],dtype=float),
            geometries,shapes,qid,q,fresh[qid])
        with np.load(mini.verified(pg['shards'][qid]),allow_pickle=False) as f:
            same_keys=np.array_equal(keys,f['keys'])
            bitwise=same_keys and np.array_equal(geom,f['geometry']) and np.array_equal(named,f['named'])
            error=max(float(np.max(np.abs(geom-f['geometry']),initial=0.)),float(np.max(np.abs(named-f['named']),initial=0.))) if same_keys else None
        row=dict(query_id=qid,passed=bitwise,keys_identical=same_keys,max_absolute_error=error)
        mini.save(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.json'),row)
        rows.append(row)
        if not bitwise:
            mini.save(directory/'failure.json',dict(status='STOP_OSM_DELTA_NONFIDELITY',query=row))
            raise ValueError('STOP_OSM_DELTA_NONFIDELITY')
        np.savez_compressed(directory/(hashlib.sha256(qid.encode()).hexdigest()+'.npz'),keys=keys,geometry=geom,named=named)
    mini.save(directory/'result.json',dict(status='PASS',family='osm',queries=20,context_matches=20,
        weighted_delta_matches=20,max_absolute_error=max(r['max_absolute_error'] for r in rows),bodies=bodies,
        weighted_body_sha256=kernel_sha,upstream_numeric51='FROZEN_INPUT_FOR_ISOLATED_OSM_TEST_NOT_REBUILT_HERE',
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        raw_labels_read=False,native_queries=0))
