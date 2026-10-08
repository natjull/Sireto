"""Full 2212-query frozen pipeline; no labels until every channel is sealed.

CPU preparation uses the project runtime. Embeddings use the ORIGINAL night
runtime, original batches and original adapters. No training, test or serving.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_compare as old
from scripts import embeddinggemma2_v2_data as data
from scripts import embeddinggemma2_v10_representative as population
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v14'
CPU = io.ROOT / '.venv/bin/python'
GPU = io.HOME / 'runtime/bin/python'
CENSUS = io.HOME / 'v10/census_20261008_a'
LEXICAL = io.HOME / 'v12/lexical_validation_20261008_a'
PREVIOUS = io.HOME / 'v8/site_pipeline_20261008_a'
VARIANTS = ('v1_native160', 'v3_official512')
INTERRUPTED = False


class ResourceYield(RuntimeError):
    pass


def interrupted(*_):
    global INTERRUPTED
    INTERRUPTED = True


def guard():
    import psutil
    try: free, level = io.pressure()
    except Exception as error: raise ResourceYield('Unknown real memory counters') from error
    rss = psutil.Process().memory_info().rss
    if INTERRUPTED or free < 20 or level not in (0, 1, 2) or rss > 10*1024**3:
        raise ResourceYield(f'resource yield free={free} level={level} rss={rss}')
    if psutil.disk_usage(io.HOME).free < 80*1024**3:
        raise RuntimeError('CATNAT reserve below 80GiB; artefacts retained')


def progress(out, stage, units, **values):
    io.write(out / 'worker_state.json', dict(stage=stage, completed_units=units,
        updated_unix=time.time(), pid=os.getpid(), **values))


def authority(out):
    launch = io.read(out / 'launch.json')
    for ref in launch['implementations']: verify(ref)
    if old.inventory() != [tuple(r) for r in launch['index_inventory']]:
        raise ValueError('Official index inventory drift')
    verify(launch['panel']); verify(launch['index_manifest']); verify(launch['index_meta'])
    for item in launch['base_files']:
        path=Path(item['file']['path']); stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(item['file']['size_bytes'],item['mtime_ns']):
            raise ValueError('Base model/tokenizer epoch drift')
    return launch


def bind(path, value):
    if path.exists():
        if io.read(path) != value: raise ValueError('Resume binding drift: ' + str(path))
    else: io.write(path, value)


def seal(path):
    ref = record(path); io.write(path.with_suffix('.seal.json'), ref); return ref


def sealed(path):
    ref = path.with_suffix('.seal.json')
    return verify(io.read(ref)) if ref.exists() else None


def compile_aliases(out):
    """Disk B-tree lookup: every official owner/name fact, no CRM/GT selection.

    Transactions are one pinned V130 bucket. SQLite avoids an enormous in-RAM
    ART index. The resulting lookup changes access, not official name content.
    """
    import duckdb
    launch = authority(out); directory = out / 'names'; directory.mkdir(exist_ok=True)
    manifest = io.read(verify(launch['dictionary']))
    if manifest['status'] != 'SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY':
        raise ValueError('Official name authority')
    target = directory / 'owners.sqlite'
    if sealed(target):
        if not (directory/'generation.json').exists():
            io.write(directory/'generation.json',dict(status='SEALED_OFFICIAL_NAMES_ALL_OWNERS',
                database=io.read(target.with_suffix('.seal.json')),mtime_ns=target.stat().st_mtime_ns,
                truth_opened=False,queries_used=False))
        return
    bind(directory / 'binding.json', dict(dictionary=launch['dictionary'], launch=record(out/'launch.json'),
        recipe='LEGAL_PARENT_1_12_OWN_SITE_2_3_SAFE_SORTED_SETS'))
    sql = sqlite3.connect(target)
    sql.execute('PRAGMA cache_size=-196608'); sql.execute('PRAGMA journal_mode=WAL')
    sql.execute('CREATE TABLE IF NOT EXISTS aliases(kind INTEGER, owner_key INTEGER, name TEXT, PRIMARY KEY(kind,owner_key,name)) WITHOUT ROWID')
    sql.execute('CREATE TABLE IF NOT EXISTS buckets(bucket INTEGER PRIMARY KEY, facts_sha TEXT, names_sha TEXT)')
    sql.commit()
    con = duckdb.connect(); con.execute("SET threads=2"); con.execute("SET memory_limit='2GB'")
    tmp = directory/'duck_tmp'; tmp.mkdir(exist_ok=True); con.execute('SET temp_directory=?', [str(tmp)])
    try:
        for bucket in manifest['buckets']:
            guard(); n = bucket['bucket']; refs = bucket['artifacts']
            fact = refs['facts.parquet']; names = refs['names.parquet']
            done = sql.execute('SELECT facts_sha,names_sha FROM buckets WHERE bucket=?', (n,)).fetchone()
            if done:
                if done != (fact['sha256'], names['sha256']): raise ValueError('Bucket resume drift')
                continue
            fp, np = verify(fact), verify(names)
            bad = con.execute('SELECT count(*) FROM read_parquet(?) WHERE NOT ((subject_code=1 AND scope_code IN (1,2)) OR (subject_code=2 AND scope_code=3))', [str(fp)]).fetchone()[0]
            if bad: raise ValueError('Official name provenance drift')
            cursor = con.execute('''SELECT CASE WHEN subject_code=1 THEN 1 ELSE 2 END AS kind,
                CASE WHEN subject_code=1 THEN parent_key ELSE owner_key END AS owner, name_text
                FROM read_parquet(?) f JOIN read_parquet(?) n USING(name_id)''', [str(fp), str(np)])
            count = 0
            sql.execute('BEGIN')
            try:
                for batch in cursor.fetch_record_batch(65536):
                    guard()
                    rows = [(r['kind'], int(r['owner']), data.safe(r['name_text'])) for r in batch.to_pylist()]
                    sql.executemany('INSERT OR IGNORE INTO aliases VALUES (?,?,?)', (r for r in rows if r[2]))
                    count += len(rows)
                    progress(out, 'OFFICIAL_OWNER_NAMES', n, bucket=n, bucket_rows=count, buckets_total=64)
                if count != bucket['facts']: raise ValueError('Name join lost official facts')
                verify(fact); verify(names)
                sql.execute('INSERT INTO buckets VALUES (?,?,?)', (n, fact['sha256'], names['sha256']))
                sql.commit()
            except BaseException:
                sql.rollback(); raise
            progress(out, 'OFFICIAL_OWNER_NAMES', n+1, buckets_total=64)
        if sql.execute('SELECT count(*) FROM buckets').fetchone()[0] != 64: raise ValueError('All official buckets required')
        sql.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:
        con.close(); sql.close()
    verify(launch['dictionary']); seal(target)
    io.write(directory/'generation.json', dict(status='SEALED_OFFICIAL_NAMES_ALL_OWNERS',
        database=io.read(target.with_suffix('.seal.json')), mtime_ns=target.stat().st_mtime_ns,
        truth_opened=False, queries_used=False))


def lookup(con, kind, keys):
    result = defaultdict(set); values = sorted(set(map(int, keys)))
    for lo in range(0, len(values), 900):
        part = values[lo:lo+900]; guard()
        query = 'SELECT owner_key,name FROM aliases WHERE kind=? AND owner_key IN ('+','.join('?'*len(part))+')'
        for key, name in con.execute(query, [kind, *part]): result[key].add(name)
    return result


def zone_path(out, zone):
    return out / 'zones' / (zone or 'UNKNOWN')


def prepare_zone(out, zone):
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    launch = authority(out); spec = next(z for z in io.read(verify(launch['census']))['zones'] if z['zone'] == zone)
    directory = zone_path(out, zone); directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'documents.json'
    if sealed(target): return
    generation_path=out/'names/generation.json'
    if not generation_path.exists(): raise ValueError('Whole official lookup must be sealed')
    names_generation=io.read(generation_path); database=Path(names_generation['database']['path'])
    if (database.stat().st_size,database.stat().st_mtime_ns)!=(names_generation['database']['size_bytes'],names_generation['mtime_ns']):
        raise ValueError('Immutable official lookup epoch drift')
    binding = dict(launch=record(out/'launch.json'), names=io.read((out/'names/owners.sqlite').with_suffix('.seal.json')), zone=spec)
    bind(directory/'preparation_binding.json', binding)
    if zone in launch['prepared_corpora']:
        docs = io.read(verify(launch['prepared_corpora'][zone]))
        if len(docs) != spec['indexed_sites']: raise ValueError('Prepared complete corpus count')
        # The original prepared SITE strings are reused literally.
        io.write(target, docs); seal(target); return
    docs = []
    if spec['indexed_sites']:
        backend = TantivyBackend(io.INDEX)
        if backend.searcher.num_docs != 82772691: raise ValueError('National index document count')
        hits = backend.searcher.search(old.zone_query(backend, zone), limit=spec['indexed_sites']+1, count=True)
        if hits.count != spec['indexed_sites'] or len(hits.hits) != hits.count:
            raise ValueError('Full commune required; no label-centred/pruned corpus')
        for i, (_, address) in enumerate(hits.hits):
            if i % 1024 == 0: guard(); progress(out, 'EXTRACTING_FULL_COMMUNE', i, zone=zone, total=hits.count)
            raw = backend.searcher.doc(address).to_dict(); doc = data.native_document(backend, raw)
            doc['native_text'] = old.document_text(backend, raw); docs.append(doc)
        del hits, backend; gc.collect()
        con = sqlite3.connect(f'file:{database}?mode=ro&immutable=1', uri=True)
        try:
            legal = lookup(con, 1, [d['siren'] for d in docs]); own = lookup(con, 2, [d['key'] for d in docs])
            for d in docs:
                d['extra'] = dict(legal=sorted(legal[int(d['siren'])]), own_site=sorted(own[int(d['key'])]))
        finally: con.close()
    docs.sort(key=lambda d: d['key'])
    if len(docs) != spec['indexed_sites'] or len({d['key'] for d in docs}) != len(docs):
        raise ValueError('Complete unique official corpus')
    io.write(target, docs); authority(out); seal(target)
    progress(out, 'FULL_COMMUNE_PREPARED', len(docs), zone=zone)


def text_hash(texts):
    digest = hashlib.sha256()
    for value in texts:
        encoded = value.encode(); digest.update(len(encoded).to_bytes(8, 'big')); digest.update(encoded)
    return digest.hexdigest()


def score_views(query_vectors, views, vector, native=False):
    import numpy as np
    if native: return query_vectors @ np.stack([vector(vs[0]) for vs in views]).T
    return np.column_stack([(query_vectors @ np.stack([vector(v) for v in vs]).T).max(axis=1) for vs in views])


def dense_zone(out, zone, variant):
    import numpy as np
    import torch
    import transformers
    from scripts import embeddinggemma2_v3_fit as fit
    if torch.__version__ != '2.14.1' or transformers.__version__ != '5.19.0':
        raise ValueError('Original encoder runtime required')
    launch = authority(out); directory = zone_path(out, zone)
    target = directory / (variant + '.json')
    if sealed(target): return
    docs_path = sealed(directory/'documents.json')
    if not docs_path: raise ValueError('Complete sealed commune required')
    docs = io.read(docs_path); queries = [q for q in io.read(verify(launch['panel'])) if q['crm_insee'] == zone]
    cache = directory / variant; cache.mkdir(exist_ok=True)
    adapter = launch['old_adapter' if variant == 'v1_native160' else 'new_adapter']
    if variant=='v3_official512':
        training=io.read(verify(launch['training_state']))
        checkpoint=torch.load(verify(adapter),map_location='cpu',weights_only=True)
        if (training['status'] not in ('FIT_COMPLETE','EARLY_STOP','TIME_BUDGET_CHECKPOINTED')
                or checkpoint['step']!=1000 or checkpoint['step']>training['step']
                or checkpoint['launch_sha256']!=io.digest(Path(adapter['path']).parent/'launch.json')):
            raise ValueError('Original finalized best checkpoint binding')
    cfg = io.read(verify(launch['night_config']))
    binding = dict(launch=record(out/'launch.json'), documents=record(docs_path), adapter=adapter,
        variant=variant, queries=queries, runtime=launch['encoder_runtime'], cache_shard_texts=1024)
    bind(cache/'binding.json', binding)
    if not docs:
        io.write(target, [dict(query_id=q['query_id'], fold=q['fold'], candidates=[], variant=variant) for q in queries])
        seal(target); return
    guard()
    if variant == 'v1_native160':
        mode, inference, tokenizer, step = old.load_encoder(verify(adapter)); mode(True)
    else:
        model, params, encode, inference, tokenizer = fit.load_model(cfg, verify(adapter))
        step = torch.load(verify(adapter), map_location='cpu', weights_only=True)['step']
        for i, doc in enumerate(docs):
            if i % 256 == 0: guard(); progress(out, 'PACKING_OFFICIAL_SITE_VIEWS', i, zone=zone, total=len(docs))
            if 'site_views' not in doc:
                anchor = io.clean_text(doc['own_names'][:1], doc['all_addresses'][:1], doc['postcode'])
                doc['site_views'] = data.pack_atoms(data.site_atoms(doc, doc['extra']), tokenizer, 512, anchor)[0]
    views = [[io.DOC_PROMPT + doc['native_text']] if variant == 'v1_native160'
        else [io.DOC_PROMPT + v for v in doc['site_views']] for doc in docs]
    texts = list(dict.fromkeys(v for vs in views for v in vs))
    positions = {v: i for i, v in enumerate(texts)}
    bind(cache/'texts_binding.json', dict(order_sha256=text_hash(texts), texts=len(texts),
        model_step=step, dimension=768, batch_size=16 if variant=='v1_native160' else 8))
    query_vectors = inference([io.QUERY_PROMPT + old.query_text(q) for q in queries])
    matrices = []
    for lo in range(0, len(texts), 1024):
        path = cache/f'vectors_{lo:08d}.npy'
        if not sealed(path):
            chunks = []
            for offset in range(lo, min(lo+1024, len(texts)), 128):
                guard(); values = inference(texts[offset:min(offset+128, lo+1024, len(texts))])
                if values.dtype != np.float32 or values.shape != (min(128, lo+1024-offset, len(texts)-offset), 768) or not np.isfinite(values).all():
                    raise ValueError('Original finite float32 vector shape')
                chunks.append(values)
                progress(out, 'FULL_CORPUS_ENCODING', lo, zone=zone, variant=variant, encoded=offset+len(values), total=len(texts))
            tmp = path.with_suffix('.pending.npy'); np.save(tmp, np.concatenate(chunks), allow_pickle=False)
            os.replace(tmp, path); seal(path)
        matrix = np.load(sealed(path), mmap_mode='r', allow_pickle=False)
        if matrix.shape != (min(1024, len(texts)-lo), 768) or matrix.dtype != np.float32: raise ValueError('Vector shard binding')
        matrices.append(matrix)
        progress(out, 'FULL_CORPUS_ENCODING', lo+len(matrix), zone=zone, variant=variant, total=len(texts))
    def vector(value):
        idx = positions[value]; return matrices[idx//1024][idx%1024]
    best = [[] for _ in queries]
    for lo in range(0, len(docs), 128):
        guard(); part = docs[lo:lo+128]
        scores=score_views(query_vectors,views[lo:lo+128],vector,variant=='v1_native160')
        sites = [dict(siret=d['key'], siren=d['siren']) for d in part]
        for i in range(len(queries)): best[i] = old.top100(best[i], scores[i], sites)
        progress(out, 'EXACT_FULL_CORPUS_INITIAL100', lo+len(part), zone=zone, variant=variant, total=len(docs))
    rows = [dict(query_id=q['query_id'], fold=q['fold'], variant=variant,
        candidates=[dict(score=s, siret=t, siren=n) for s,t,n in top]) for q,top in zip(queries,best)]
    if any(len(r['candidates']) != min(100,len(docs)) for r in rows): raise ValueError('Exact initial100 count')
    verify(adapter); authority(out); io.write(target, rows); seal(target)
    for matrix in matrices: matrix._mmap.close()


def geographic(out):
    import duckdb
    import numpy as np
    from scripts import embeddinggemma2_v5_geo as geo
    from src.xgb_matcher import retrieval_query_ban_v327 as ban
    launch = authority(out); directory = out/'geo'; directory.mkdir(exist_ok=True)
    target = directory/'outputs.json'
    if sealed(target): return
    source = launch['geographic']; queries = io.read(verify(launch['panel']))
    keys = np.load(verify(source['keys']), mmap_mode='r', allow_pickle=False)
    xy = np.load(verify(source['xy']), mmap_mode='r', allow_pickle=False)
    if keys.dtype != np.uint64 or xy.dtype != np.float64 or xy.shape != (len(keys),2) or len(keys)!=43896818: raise ValueError('Official XY namespace')
    class Files:
        def file(self, ref):
            if ref != source['ban']: raise ValueError('Pinned query-only BAN')
            return verify(ref)
    con=duckdb.connect(); con.execute('SET threads=1'); con.execute("SET memory_limit='384MB'")
    tmp=directory/'duck_tmp'; tmp.mkdir(exist_ok=True); con.execute('SET temp_directory=?',[str(tmp)])
    try: geometry=ban.resolve(con, queries, Files())
    finally: con.close()
    radii=[geo.geo_radius(g) for g in geometry]; best=[[] for _ in queries]; matched=[0 for _ in queries]
    previous=None
    try:
        for lo in range(0,len(keys),262144):
            guard(); chunk=keys[lo:lo+262144]; coordinates=xy[lo:lo+262144]
            if (previous is not None and int(chunk[0])<=previous) or np.any(chunk[1:]<=chunk[:-1]): raise ValueError('Unique sorted keys')
            previous=int(chunk[-1])
            valid=np.isfinite(coordinates).all(axis=1)&(coordinates[:,0]>=0)&(coordinates[:,0]<=1300000)&(coordinates[:,1]>=6000000)&(coordinates[:,1]<=7300000)
            for i,g in enumerate(geometry):
                if radii[i] is None: continue
                delta=coordinates-np.asarray([g['ban_x_l93'],g['ban_y_l93']]); d2=np.einsum('ij,ij->i',delta,delta)
                locations=np.flatnonzero(valid&(d2<=radii[i]**2)); matched[i]+=len(locations)
                best[i]=geo.nearest100(best[i],np.sqrt(d2[locations]),chunk[locations])
            progress(out,'NATIONAL_GEO_INITIAL100',min(lo+262144,len(keys)),total=len(keys),queries=len(queries))
    finally: keys._mmap.close(); xy._mmap.close()
    for ref in source.values(): verify(ref)
    rows=[]
    for q,g,r,top,n in zip(queries,geometry,radii,best,matched):
        rows.append(dict(query_id=q['query_id'],fold=q['fold'],geometry=g,radius_m=r,official_sites_in_radius=n,
            rrf_geo_weight=.5 if g['ban_declared_weak'] else 1., candidates=[dict(siret=f'{k:014d}',siren=f'{k:014d}'[:9],distance_m=d) for d,k in top]))
    io.write(target,rows); seal(target)
    io.write(directory/'generation.json',dict(queries=len(rows),scanned_sites=len(keys),
        unresolved_queries=sum(r is None for r in radii), coordinate_namespace_appended_sites_missing=True,truth_opened=False))


def complete_channels(out):
    """Strict pre-label gate; all full communes, variants and owned exits."""
    launch=authority(out); panel=io.read(verify(launch['panel'])); scope={q['query_id'] for q in panel}
    census=io.read(verify(launch['census']))
    names_generation=io.read(out/'names/generation.json'); verify(names_generation['database'])
    if io.read(out/'aliases_exit.json').get('exit_code')!=0: raise ValueError('Complete official name lookup exit0')
    channels={}
    for variant in VARIANTS:
        rows=[]
        for zone in census['zones']:
            directory=zone_path(out,zone['zone']); path=directory/(variant+'.json')
            ref=sealed(path)
            ex=io.read(directory/(variant+'_exit.json'))
            if not ref or ex.get('exit_code')!=0 or not ex.get('own_child_exit_confirmed'): raise ValueError('Every dense zone owned exit0')
            preparation=io.read(directory/'prepare_exit.json')
            if preparation.get('exit_code')!=0 or not preparation.get('own_child_exit_confirmed'): raise ValueError('Complete commune owned exit0')
            rows.extend(io.read(ref))
        if len(rows)!=2212 or {r['query_id'] for r in rows}!=scope: raise ValueError('Full dense denominator')
        channels[variant]={r['query_id']:r['candidates'] for r in rows}
    for name,ref,exit_path in (('lexical',launch['lexical'],LEXICAL/'owned_exit.json'),
            ('geo',io.read((out/'geo/outputs.json').with_suffix('.seal.json')),out/'geo_exit.json')):
        exited=io.read(exit_path)
        if exited.get('exit_code')!=0 or not exited.get('own_child_exit_confirmed'): raise ValueError('Owned channel exit0')
        rows=io.read(verify(ref))
        if len(rows)!=2212 or {r['query_id'] for r in rows}!=scope: raise ValueError('Complete channel scope')
        channels[name]={r['query_id']:r['candidates'] for r in rows}
        if name=='geo': geometry={r['query_id']:r for r in rows}
    from scripts.embeddinggemma2_v5_geo import weighted_fusion
    channels['gemma2_lexical_geo']={q:weighted_fusion([(channels['v3_official512'][q],1.),
        (channels['lexical'][q],1.),(channels['geo'][q],geometry[q]['rrf_geo_weight'])]) for q in scope}
    channels['dense_family_lexical_geo']={q:population.family_fusion(channels['v1_native160'][q],
        channels['v3_official512'][q],channels['lexical'][q],channels['geo'][q],geometry[q]['rrf_geo_weight']) for q in scope}
    for channel in channels.values():
        for rows in channel.values():
            if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows): raise ValueError('Absolute initial100 uniqueness/cap')
    io.write(out/'all_outputs.json',channels); ref=seal(out/'all_outputs.json')
    io.write(out/'generation.json',dict(status='SEALED_ALL2212_COMPLETE_PIPELINES',outputs=ref,
        queries=2212,official_commune_sites=19097808,truth_opened=False,test_opened=False))
    for item in launch['base_files']: verify(item['file'])
    return channels,scope,launch


def evaluate(out):
    import numpy as np
    import pyarrow.parquet as pq
    from scripts.embeddinggemma2_v3_compare import summarize
    channels,scope,launch=complete_channels(out)
    manifest=io.read(verify(launch['training_manifest'])); membership=population.SOURCE/'validation_membership.json'
    if io.digest(membership)!=manifest['split_hashes']['validation']: raise ValueError('Validation label authority drift')
    labels={r['query_id']:r for r in io.read(membership)}
    if set(labels)!=scope: raise ValueError('All human labels denominator')
    cfg=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    ev=io.read(verify(cfg['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=cfg['ledger']: raise ValueError('Official label-side authority')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))]
    accepted={q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'],filters=filters,use_threads=False).to_pylist(): accepted[r['query_id']].add(r['siret'])
    meta={r['query_id']:r for r in pq.read_table(verify(cfg['ledger']),filters=filters,use_threads=False).to_pylist()}
    roles=io.read(verify(launch['roles'])); outcomes={}; reports={}
    for name,channel in channels.items():
        rows=[]
        for q in sorted(scope):
            cs=channel[q]; gt=labels[q]; keys={c['siret'] for c in cs}
            rows.append(dict(query_id=q,siren_group=gt['siren'],fold=gt['fold'],role=roles[q],
                exact=gt['siret'] in keys,siren=gt['siren'] in {c['siren'] for c in cs},
                operational=bool(keys&accepted[q]),exact_rank=next((i+1 for i,c in enumerate(cs) if c['siret']==gt['siret']),None)))
        segments={}
        for field in ('fold','role','label_kind','qualification_v2','qualification_v3','data_origin','ground_truth_state'):
            groups=defaultdict(list)
            for r in rows: groups[str(r.get(field,meta.get(r['query_id'],{}).get(field,'UNKNOWN')))].append(r)
            segments[field]={k:summarize(v) for k,v in groups.items()}
        groups=defaultdict(list)
        for r in rows: groups[r['siren_group']].append(r['exact'])
        counts=np.array([(sum(v),len(v)) for v in groups.values()]); rng=np.random.default_rng(20261008); draws=[]
        for _ in range(2000):
            sample=counts[rng.integers(0,len(counts),len(counts))].sum(axis=0); draws.append(sample[0]/sample[1])
        outcomes[name]=rows; reports[name]={**summarize(rows), 'segments':segments,
            'siren_cluster_bootstrap95':np.quantile(draws,[.025,.975]).tolist()}
    comparisons={}
    for a,b in (('lexical','gemma2_lexical_geo'),('gemma2_lexical_geo','dense_family_lexical_geo')):
        ar={r['query_id']:r for r in outcomes[a]}; br={r['query_id']:r for r in outcomes[b]}
        comparisons[a+'__vs__'+b]=dict(gained=sum(br[q]['exact'] and not ar[q]['exact'] for q in scope),lost=sum(ar[q]['exact'] and not br[q]['exact'] for q in scope))
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_FULL2212_FROZEN_COMPLETE_PIPELINES',variants=reports,
        comparisons=comparisons,all_queries_in_denominator=True,validation_used_for_checkpoint_selection=True,
        pristine_pipeline_holdout=False,full_oof_measured=False,production_slo_measured=False,
        national_dense_corpus=False,dense_full_communes=True, isolated_fine_tuning_effect_measured=False,
        bodacc_graph_expansion_measured=False,test_opened=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False))


def durable_progress(out,directory,action,variant):
    if action=='aliases':
        path=out/'names/owners.sqlite'
        if not path.exists(): return 0
        con=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        try:
            return con.execute('SELECT count(*) FROM buckets').fetchone()[0]
        except sqlite3.OperationalError: return 0
        finally: con.close()
    if action=='dense': return len(list((directory/variant).glob('vectors_*.seal.json')))+int((directory/(variant+'.seal.json')).exists())
    return int((directory/('documents.seal.json' if action=='prepare' else 'geo/outputs.seal.json')).exists())


def owned(out, action, zone=None, variant=None):
    """Only resource exits retry; no duplicate worker, immutable shard bindings."""
    import psutil
    directory=zone_path(out,zone) if zone is not None else out
    directory.mkdir(parents=True,exist_ok=True)
    tag=variant or action; receipt=directory/(tag+'_exit.json')
    if receipt.exists() and io.read(receipt).get('exit_code')==0: return
    command=[str(GPU if action=='dense' else CPU),'-u','-m','scripts.embeddinggemma2_v14_full_pipeline',action,'--output',str(out)]
    if zone is not None: command+=['--zone',zone]
    if variant: command+=['--variant',variant]
    attempts=directory/(tag+'_attempts'); attempts.mkdir(exist_ok=True)
    no_progress=0
    for attempt in range(len(list(attempts.glob('*_exit.json')))+1,1000):
        authority(out)
        while True:
            free,level=io.pressure()
            if free>=30 and level in (0,1,2): break
            io.write(out/'state.json',dict(status='WAITING_REAL_RESOURCE_ADMISSION',action=action,zone=zone,variant=variant,free=free,level=level))
            time.sleep(30)
        before=durable_progress(out,directory,action,variant)
        with (directory/(tag+'.log')).open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=io.ROOT,
                env={**os.environ,'RAYON_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','TMPDIR':str(out/'tmp')})
            io.write(out/'state.json',dict(status='RUNNING_COMPLETE_PIPELINE',action=action,zone=zone,variant=variant,worker_pid=child.pid,command=command,attempt=attempt))
            term=None
            while child.poll() is None:
                time.sleep(5)
                if child.poll() is not None: break
                try: free,level=io.pressure()
                except Exception: free,level=0,-1
                try: rss=psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess: continue
                if term is None and (free<15 or level not in (0,1,2) or rss>12*1024**3):
                    child.send_signal(signal.SIGTERM); term=time.monotonic()
                elif term is not None and time.monotonic()-term>90: child.kill()
            code=child.wait()
        result=dict(exit_code=code,own_child_exit_confirmed=True,command=command,
            ended_unix=time.time(),resource_supervisor_signalled=term is not None)
        io.write(attempts/f'{attempt:04d}_exit.json',result); io.write(receipt,result)
        if code==0: return
        if code!=75 and not (code in (-9,-15) and term is not None): raise RuntimeError(f'{action} exit{code}; inspect {directory}/{tag}.log')
        after=durable_progress(out,directory,action,variant)
        no_progress=no_progress+1 if before==after else 0
        if no_progress>=3: raise RuntimeError('Three resource retries without durable progress; STOP')
        time.sleep(30)
    raise RuntimeError('Resource retry safety exhausted')


def initialize(out):
    from src.xgb_matcher import retrieval_query_ban_v327 as ban
    from scripts import embeddinggemma2_v3_fit as fit
    from scripts import embeddinggemma2_v5_geo as geo
    if out.exists(): authority(out); return
    census=io.read(CENSUS/'census.json'); previous=io.read(PREVIOUS/'launch.json')
    from scripts.embeddinggemma2_v3_compare import verify_launch
    verify_launch(previous)
    lexical=io.read(LEXICAL/'generation.json'); exited=io.read(LEXICAL/'owned_exit.json')
    if lexical['queries']!=2212 or exited['exit_code']!=0 or not exited['own_child_exit_confirmed']: raise ValueError('Whole lexical channel required')
    panel=io.read(verify(census['sources']['panel']))
    if len(panel)!=2212 or any(set(q)!=set(old.CRM) or q['fold'] not in (2,3,4) for q in panel): raise ValueError('Full query-only safe population')
    if sum(z['indexed_sites'] for z in census['zones'])!=19097808 or len(census['zones'])!=937: raise ValueError('No geographic census filtering')
    out.mkdir(parents=True); (out/'tmp').mkdir()
    origin=io.read(io.HOME/'v5/geo_20261008_a/launch.json')
    implementations=[record(Path(p)) for p in (__file__,io.__file__,old.__file__,data.__file__,population.__file__,fit.__file__,geo.__file__,ban.__file__,
        io.ROOT/'scripts/run_retrieval_geo_v16_workbench.py',io.ROOT/'src/xgb_matcher/retrieval_geo_v16.py',io.ROOT/'scripts/embeddinggemma2_v3_compare.py')]
    launch=dict(census=record(CENSUS/'census.json'),panel=census['sources']['panel'],roles=census['sources']['roles'],
        lexical=lexical['outputs'],lexical_generation=record(LEXICAL/'generation.json'),dictionary=data.DICTIONARY,
        previous_launch=record(PREVIOUS/'launch.json'),
        old_adapter=previous['v1_adapter'],new_adapter=record(Path(previous['requested_v3_adapter_path'])),
        night_config=previous['night_config'],training_manifest=previous['training_manifest'],
        training_state=record(Path(previous['requested_v3_adapter_path']).parent/'state.json'),
        prepared_corpora={z['zone']:z['file'] for z in previous['corpora']},
        index_inventory=old.inventory(),index_manifest=record(io.INDEX/'manifest.json'),index_meta=record(io.INDEX/'tantivy/meta.json'),
        implementations=implementations, encoder_runtime=dict(torch='2.14.1',transformers='5.19.0',python=str(GPU)),
        geographic={k:origin[k] for k in ('keys','xy','ban')},
        base_revision=io.REVISION,base_files=[dict(file=record(p),mtime_ns=p.stat().st_mtime_ns) for p in sorted((io.HOME/'base').rglob('*')) if p.is_file()],
        index_segment_full_hashes_verified=False,
        new_dense_score='MAX_SITE_VIEW_COSINE_NO_PARENT_BLEND',rrf_constant=60,weak_ban_weight=.5,
        family_score='MAX_OLD_NEW_RRF_PLUS_LEXICAL_PLUS_GEO',full_communes=True,no_gradient_steps=True,
        test_opened=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False)
    for key in ('old_adapter','new_adapter','night_config','training_manifest','training_state','dictionary','lexical'): verify(launch[key])
    io.write(out/'launch.json',launch)


def run(out):
    initialize(out); authority(out)
    # Prevent accidental second controller or resumption while a live owner exists.
    import psutil
    lock=out/'controller.json'
    if lock.exists():
        owner=io.read(lock)
        if psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['created'])<1:
            raise ValueError('Active full-pipeline controller; do not duplicate')
    io.write(lock,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        owned(out,'aliases')
        owned(out,'geo')
        census=io.read(verify(io.read(out/'launch.json')['census']))
        # All communes, deterministic size order; no outcome/status exclusion.
        for spec in sorted(census['zones'],key=lambda z:(z['indexed_sites'],z['zone'])):
            zone=spec['zone']; owned(out,'prepare',zone)
            for variant in VARIANTS: owned(out,'dense',zone,variant)
        evaluate(out)
        io.write(out/'state.json',dict(status='COMPLETE_FULL2212_PIPELINES',report=record(out/'report.json'),test_opened=False))
    finally:
        if awake.poll() is None: awake.terminate(); awake.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','aliases','prepare','dense','geo'])
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--zone'); parser.add_argument('--variant',choices=VARIANTS)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated new V14 namespace')
    if args.action!='run': signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.action=='run': run(args.output)
        elif args.action=='aliases': compile_aliases(args.output)
        elif args.action=='prepare': prepare_zone(args.output,args.zone)
        elif args.action=='dense': dense_zone(args.output,args.zone,args.variant)
        else: geographic(args.output)
    except ResourceYield as error:
        io.write(args.output/'resource_yield.json',dict(reason=str(error),pid=os.getpid())); sys.exit(75)
    except Exception as error:
        if args.action!='run' and 'MPS backend out of memory' in str(error):
            io.write(args.output/'resource_yield.json',dict(reason=str(error),pid=os.getpid())); sys.exit(75)
        if args.output.exists(): io.write(args.output/'execution_error.json',dict(reason=str(error),action=args.action,pid=os.getpid()))
        raise
