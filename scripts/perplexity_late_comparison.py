"""Full-commune paired initial100 diagnostic, never a representative certificate."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v15_probability_pilot as pilot
from scripts import embeddinggemma2_v10_representative as population
from scripts import embeddinggemma2_v3_compare as paired
from scripts import embeddinggemma2_v5_geo as spatial
from scripts import perplexity_late_probe as probe
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v14/perplexity_paired_20261008_a'
RUNTIME = io.HOME / 'runtime/bin/python'
OVERLAY = probe.ROOT / 'vision_overlay'


def select(zones, panel):
    selected = []
    for lo, hi in ((1, 1000), (1000, 10000), (20000, 50000)):
        options = [z for z in zones if lo <= z['indexed_sites'] < hi and z['queries'] >= 3]
        options.sort(key=lambda z: (-z['queries'], hashlib.sha256(
            ('pplx-paired-v1|' + z['zone']).encode()).digest()))
        if not options:
            raise ValueError('Required density stratum absent')
        selected.append(options[0])
    selected.extend(z for z in zones if not z['zone'])
    scope = {z['zone'] for z in selected}
    queries = [{k: q[k] for k in paired.old.CRM} for q in panel if q['crm_insee'] in scope]
    if sum(z['queries'] for z in selected) != len(queries) or any(q['fold'] not in (2,3,4) for q in queries):
        raise ValueError('All selected queries, safe folds only')
    return selected, queries


def initialize():
    if OUTPUT.exists():
        engine.authority(OUTPUT)
        return
    source = engine.authority(probe.SOURCE)
    census = io.read(verify(source['census']))
    zones, queries = select(census['zones'], io.read(verify(source['panel'])))
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'names').mkdir()
    (OUTPUT / 'tmp').mkdir()
    io.write(OUTPUT / 'panel.json', queries)
    io.write(OUTPUT / 'census.json', {**census, 'zones': zones, 'queries': len(queries),
        'communes': len(zones), 'total_indexed_commune_sites': sum(z['indexed_sites'] for z in zones),
        'selection': 'QUERY_COUNT_FIRST_WITHIN_THREE_DENSITY_STRATA_HASH_TIES_NOT_REPRESENTATIVE'})
    io.write(OUTPUT / 'lexical.json', pilot.projected_rows(source['lexical'], queries))
    io.write(OUTPUT / 'geo.json', pilot.projected_rows(source['projected_geo'], queries))
    for name in ('generation.json', 'owners.seal.json'):
        io.write(OUTPUT / 'names' / name, io.read(probe.SOURCE / 'names' / name))
    io.write(OUTPUT / 'launch.json', {**source, 'panel': record(OUTPUT / 'panel.json'),
        'census': record(OUTPUT / 'census.json'), 'lexical': record(OUTPUT / 'lexical.json'),
        'projected_geo': record(OUTPUT / 'geo.json'), 'paired_source': record(probe.SOURCE / 'launch.json'),
        'perplexity_revision': probe.REVISION,
        'perplexity_files': [record(p) for p in sorted(probe.MODEL.rglob('*')) if p.is_file()],
        'implementations': [*source['implementations'], record(Path(__file__)), record(Path(probe.__file__))],
        'recipes': ['gemma', 'perplexity', 'lexical', 'geo', 'gemma_lexical_geo',
            'perplexity_lexical_geo', 'dense_family_lexical_geo'],
        'scope': 'FULL_COMMUNE_SMALL_PAIRED_DIAGNOSTIC_NOT_REPRESENTATIVE',
        'truth_opened': False, 'test_opened': False})
    reused = []
    for z in zones:
        src = engine.zone_path(probe.SOURCE, z['zone'])
        dst = engine.zone_path(OUTPUT, z['zone'])
        dst.mkdir(parents=True)
        for artifact, action in (('documents', 'prepare'), ('v3_official512', 'v3_official512')):
            seal = src / (artifact + '.seal.json')
            receipt = src / (action + '_exit.json')
            if not seal.exists() or not receipt.exists():
                continue
            exited = io.read(receipt)
            if exited.get('exit_code') != 0 or not exited.get('own_child_exit_confirmed'):
                continue
            ref = io.read(seal)
            verify(ref)
            if artifact != 'documents':
                expected = {q['query_id'] for q in queries if q['crm_insee'] == z['zone']}
                if {r['query_id'] for r in io.read(ref['path'])} != expected:
                    raise ValueError('Baseline full query scope')
            io.write(dst / seal.name, ref)
            io.write(dst / receipt.name, {**exited, 'original_owned_receipt': record(receipt)})
            reused.append(dict(zone=z['zone'], artifact=artifact, source=ref, owned_exit=record(receipt)))
    io.write(OUTPUT / 'reuse.json', reused)


def pack_shard(vectors):
    import numpy as np
    offsets = np.concatenate(([0], np.cumsum([len(v) for v in vectors]))).astype(np.int64)
    return np.concatenate(vectors).astype(np.float32), offsets


def validate_candidates(rows, expected):
    if len(rows) != len(expected) or {r['query_id'] for r in rows} != expected:
        raise ValueError('Missing query outputs')
    for row in rows:
        cs = row['candidates']
        if len(cs) > 100 or len({c['siret'] for c in cs}) != len(cs):
            raise ValueError('Initial100 cap/uniqueness')


def generate(zone):
    import numpy as np
    import psutil
    import torch
    from sentence_transformers import MultiVectorEncoder
    from transformers import AutoTokenizer
    from scripts import embeddinggemma2_v2_data as data
    signal.signal(signal.SIGTERM, engine.interrupted)
    launch = engine.authority(OUTPUT)
    directory = engine.zone_path(OUTPUT, zone)
    queries = [q for q in io.read(verify(launch['panel'])) if q['crm_insee'] == zone]
    docs = io.read(engine.sealed(directory / 'documents.json'))
    target = directory / 'perplexity.json'
    if engine.sealed(target):
        return
    if not docs:
        io.write(target, [dict(query_id=q['query_id'], fold=q['fold'], candidates=[]) for q in queries])
        engine.seal(target)
        return
    torch.set_num_threads(2)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise ValueError('Disable CPU fallback')
    tokenizer = AutoTokenizer.from_pretrained(io.HOME / 'base', local_files_only=True)
    for doc in docs:
        if 'site_views' not in doc:
            doc['site_views'] = paired.views_for_site(doc, doc['extra'], tokenizer)
    del tokenizer
    # Literal same SITE512 views as Gemma; native Perplexity prompts and tokenizer.
    texts = list(dict.fromkeys(v for d in docs for v in d['site_views']))
    members = defaultdict(list)
    positions = {v: i for i, v in enumerate(texts)}
    for i, doc in enumerate(docs):
        for text in doc['site_views']:
            members[positions[text]].append(i)
    cache = directory / 'perplexity_vectors'
    cache.mkdir(exist_ok=True)
    binding = dict(launch=record(OUTPUT / 'launch.json'), documents=io.read(directory / 'documents.seal.json'),
        query_texts=[paired.old.query_text(q) for q in queries], text_order_hash=engine.text_hash(texts),
        texts=len(texts), revision=probe.REVISION, dtype='float32', batch=4, attention='eager',
        score='MAX_SITE_VIEW_TOKEN_MAXSIM', no_cpu_fallback=True)
    engine.bind(cache / 'binding.json', binding)
    model = MultiVectorEncoder(str(probe.MODEL), local_files_only=True, device='mps',
        trust_remote_code=False, model_kwargs={'dtype': torch.float32, 'attn_implementation': 'eager'})
    # Stop rather than silently truncate any official view or query.
    for items, limit in ((texts, 4096), (binding['query_texts'], 1024)):
        for lo in range(0, len(items), 512):
            engine.guard()
            if any(len(t)+32 > limit for t in model.tokenizer(items[lo:lo+512],
                    add_special_tokens=True, truncation=False)['input_ids']):
                raise ValueError('Native Perplexity input limit exceeded')
    qv = [v.detach().float().cpu() for v in model.encode_query(binding['query_texts'],
        batch_size=4, show_progress_bar=False)]
    scores = np.full((len(queries), len(docs)), -np.inf, dtype=np.float32)
    began = time.monotonic()
    for lo in range(0, len(texts), 128):
        engine.guard()
        if torch.mps.driver_allocated_memory() > 6*1024**3 or psutil.Process().memory_info().rss > 8*1024**3:
            raise engine.ResourceYield('Perplexity actual memory guard')
        path = cache / f'vectors_{lo:08d}.npz'
        if not engine.sealed(path):
            vs = []
            for offset in range(lo, min(lo+128, len(texts)), 4):
                engine.guard()
                if torch.mps.driver_allocated_memory() > 6*1024**3:
                    raise engine.ResourceYield('Perplexity driver allocation guard')
                encoded = model.encode_document(texts[offset:min(offset+4, lo+128, len(texts))],
                    batch_size=4, show_progress_bar=False)
                torch.mps.synchronize()
                vs.extend(v.detach().float().cpu().numpy() for v in encoded)
                del encoded
            if any(v.ndim != 2 or v.shape[1] != 128 or not np.isfinite(v).all() for v in vs):
                raise ValueError('Nonfinite/dimension drift')
            values, offsets = pack_shard(vs)
            pending = path.with_suffix('.pending.npz')
            np.savez(pending, values=values, offsets=offsets)
            os.replace(pending, path)
            engine.seal(path)
        with np.load(engine.sealed(path), allow_pickle=False) as shard:
            values, offsets = shard['values'], shard['offsets']
            if len(offsets)-1 != min(128, len(texts)-lo) or values.shape != (offsets[-1],128):
                raise ValueError('Cache shape drift')
            vs = [torch.from_numpy(values[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
            chunk = model.similarity(qv, vs, chunk_elements=4_000_000).numpy()
            if not np.isfinite(chunk).all():
                raise ValueError('MaxSim nonfinite')
            for j in range(len(vs)):
                indices = members[lo+j]
                scores[:, indices] = np.maximum(scores[:, indices], chunk[:, j:j+1])
        engine.progress(OUTPUT, 'PERPLEXITY_EXHAUSTIVE_COMMUNE', lo+len(vs), zone=zone,
            total=len(texts), full_sites=len(docs), seconds=time.monotonic()-began)
    if not np.isfinite(scores).all():
        raise ValueError('Site missing from exhaustive scoring')
    rows = []
    for q, s in zip(queries, scores):
        order = sorted(range(len(docs)), key=lambda i: (-float(s[i]), docs[i]['key']))[:100]
        rows.append(dict(query_id=q['query_id'], fold=q['fold'], candidates=[dict(
            siret=docs[i]['key'], siren=docs[i]['siren'], score=float(s[i])) for i in order]))
    validate_candidates(rows, {q['query_id'] for q in queries})
    engine.authority(OUTPUT)
    io.write(target, rows)
    engine.seal(target)
    io.write(directory / 'perplexity_generation.json', dict(outputs=record(target),
        binding=record(cache / 'binding.json'), sites=len(docs), texts=len(texts), truth_opened=False))


def seal_all():
    launch = engine.authority(OUTPUT)
    panel = io.read(verify(launch['panel']))
    scope = {q['query_id'] for q in panel}
    channels = {}
    for tag, variant in (('gemma','v3_official512'), ('perplexity','perplexity')):
        rows = []
        for z in io.read(verify(launch['census']))['zones']:
            directory = engine.zone_path(OUTPUT, z['zone'])
            for action in ('prepare', variant):
                ex = io.read(directory / (action+'_exit.json'))
                if ex.get('exit_code') != 0 or not ex.get('own_child_exit_confirmed'):
                    raise ValueError('All actual owned exits0 before labels')
            rows.extend(io.read(engine.sealed(directory / (variant+'.json'))))
        validate_candidates(rows, scope)
        channels[tag] = {r['query_id']: r['candidates'] for r in rows}
    geo_rows = pilot.projected_rows(launch['projected_geo'], panel)
    channels['lexical'] = {r['query_id']:r['candidates'] for r in pilot.projected_rows(launch['lexical'],panel)}
    channels['geo'] = {r['query_id']:r['candidates'] for r in geo_rows}
    weights = {r['query_id']:r['rrf_geo_weight'] for r in geo_rows}
    for name in ('gemma', 'perplexity'):
        channels[name+'_lexical_geo'] = {q:spatial.weighted_fusion([(channels[name][q],1.),
            (channels['lexical'][q],1.),(channels['geo'][q],weights[q])]) for q in scope}
    channels['dense_family_lexical_geo'] = {q:population.family_fusion(channels['gemma'][q],
        channels['perplexity'][q],channels['lexical'][q],channels['geo'][q],weights[q]) for q in scope}
    for values in channels.values():
        validate_candidates([dict(query_id=q,candidates=cs) for q,cs in values.items()], scope)
    io.write(OUTPUT/'all_outputs.json', channels)
    ref = engine.seal(OUTPUT/'all_outputs.json')
    io.write(OUTPUT/'generation.json', dict(outputs=ref, truth_opened=False, queries=len(scope)))
    return channels, scope, launch


def evaluate():
    import pyarrow.parquet as pq
    channels, scope, launch = seal_all()
    for ref in launch['perplexity_files']:
        verify(ref)
    manifest = io.read(verify(launch['training_manifest']))
    membership = population.SOURCE / 'validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']:
        raise ValueError('Label membership drift')
    labels = {r['query_id']:r for r in io.read(membership) if r['query_id'] in scope}
    if set(labels) != scope:
        raise ValueError('Complete denominator')
    cfg = io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    authority = io.read(verify(cfg['evaluation']))
    side = io.read(verify(authority['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger'] != cfg['ledger']:
        raise ValueError('Official label-side authority')
    for ref in (side['policy'], *side['files'].values()):
        verify(ref)
    filters = [('fold','in',[2,3,4]),('query_id','in',sorted(scope))]
    accepted = {q:{labels[q]['siret']} for q in scope}
    for r in pq.read_table(side['files']['acceptable_operational.parquet']['path'],
            filters=filters,use_threads=False).to_pylist():
        accepted[r['query_id']].add(r['siret'])
    outcomes, reports = {}, {}
    for variant, values in channels.items():
        rows=[]
        for q in sorted(scope):
            gt=labels[q]; cs=values[q]; ids={c['siret'] for c in cs}
            rows.append(dict(query_id=q,fold=gt['fold'],zone=gt['crm_insee'],
                siren_group=gt['siren'],exact=gt['siret'] in ids,
                siren=gt['siren'] in {c['siren'] for c in cs},operational=bool(ids&accepted[q])))
        outcomes[variant]=rows
        segments={}
        for field in ('fold','zone'):
            groups=defaultdict(list)
            for r in rows: groups[str(r[field])].append(r)
            segments[field]={k:paired.summarize(v) for k,v in groups.items()}
        reports[variant]={**paired.summarize(rows),'segments':segments}
    baseline={r['query_id']:r for r in outcomes['gemma_lexical_geo']}
    contrasts={}
    for variant in ('perplexity','perplexity_lexical_geo','dense_family_lexical_geo'):
        contrasts[variant]=dict(gained=[r['query_id'] for r in outcomes[variant] if r['exact'] and not baseline[r['query_id']]['exact']],
            lost=[r['query_id'] for r in outcomes[variant] if not r['exact'] and baseline[r['query_id']]['exact']])
    io.write(OUTPUT/'outcomes.json',outcomes)
    io.write(OUTPUT/'report.json',dict(status='MEASURED_FULL_COMMUNE_PAIRED_DIAGNOSTIC',
        variants=reports,contrasts_against_gemma_pipeline=contrasts,
        all_outputs=record(OUTPUT/'all_outputs.json'),sidecar=authority['label_side'],
        full_communes=True,representative=False,national_recall_measured=False,
        wilson_is_descriptive_not_generalization_proof=True,checkpoint_validation_population=True,
        oof_complete=False,slo_measured=False,test_opened=False,fold0_opened=False,fold1_opened=False,
        downstream_changed=False,no_99_2pct_certification=True))


def owned_generate(zone):
    directory=engine.zone_path(OUTPUT,zone)
    if engine.sealed(directory/'perplexity.json') and (directory/'perplexity_exit.json').exists():
        ex=io.read(directory/'perplexity_exit.json')
        if ex.get('exit_code')==0 and ex.get('own_child_exit_confirmed'): return
    command=[str(RUNTIME),'-u','-m','scripts.perplexity_late_comparison','generate','--zone',zone]
    env={**os.environ,'PYTHONPATH':str(OVERLAY),'PYTORCH_ENABLE_MPS_FALLBACK':'0',
        'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
    stalled=0; previous=-1
    while True:
        free,level=io.pressure()
        if free<30 or level not in (0,1,2):
            io.write(OUTPUT/'state.json',dict(status='WAITING_REAL_MEMORY',free=free,level=level,zone=zone))
            time.sleep(30); continue
        with (directory/'perplexity.log').open('a') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            io.write(OUTPUT/'state.json',dict(status='RUNNING_PERPLEXITY',zone=zone,worker_pid=child.pid,command=command))
            term=None
            import psutil
            while child.poll() is None:
                time.sleep(5)
                if child.poll() is not None: break
                try:
                    free,level=io.pressure(); rss=psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess: continue
                if term is None and (free<15 or level not in (0,1,2) or rss>9*1024**3):
                    child.send_signal(signal.SIGTERM); term=time.monotonic()
                elif term is not None and time.monotonic()-term>90: child.kill()
            code=child.wait()
        io.write(directory/'perplexity_exit.json',dict(exit_code=code,command=command,
            own_child_exit_confirmed=True,resource_supervisor_signalled=term is not None,ended_unix=time.time()))
        if code==0: return
        if code!=75 and not (term is not None and code in (-9,-15)):
            raise RuntimeError(f'Perplexity generation exit {code}')
        count=len(list((directory/'perplexity_vectors').glob('*.seal.json')))
        stalled=stalled+1 if count==previous else 0; previous=count
        if stalled>=3: raise RuntimeError('Repeated real pressure without durable progress')
        time.sleep(30)


def run():
    import psutil
    initialize()
    owner=OUTPUT/'controller.json'
    if owner.exists():
        old=io.read(owner)
        if psutil.pid_exists(old['pid']) and abs(psutil.Process(old['pid']).create_time()-old['created'])<1:
            raise ValueError('Controller already active')
    io.write(owner,dict(pid=os.getpid(),created=psutil.Process().create_time()))
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        for z in io.read(OUTPUT/'census.json')['zones']:
            # Do not duplicate a currently active Gemma calculation for this zone.
            while True:
                state=io.read(probe.SOURCE/'state.json')
                pid=state.get('worker_pid')
                if state.get('zone')!=z['zone'] or not pid or not psutil.pid_exists(pid): break
                io.write(OUTPUT/'state.json',dict(status='WAITING_EXISTING_GEMMA_ZONE',zone=z['zone']))
                time.sleep(30)
            engine.owned(OUTPUT,'prepare',z['zone'])
            engine.owned(OUTPUT,'dense',z['zone'],'v3_official512')
            owned_generate(z['zone'])
        evaluate()
        io.write(OUTPUT/'state.json',dict(status='COMPLETE_PAIRED_DIAGNOSTIC',report=record(OUTPUT/'report.json')))
    finally:
        if awake.poll() is None: awake.terminate(); awake.wait()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','generate']); p.add_argument('--zone')
    args=p.parse_args()
    try:
        if args.action=='run': run()
        else: generate(args.zone)
    except engine.ResourceYield as error:
        io.write(OUTPUT/'resource_yield.json',dict(reason=str(error),pid=os.getpid()))
        sys.exit(75)
    except Exception as error:
        if OUTPUT.exists(): io.write(OUTPUT/'execution_error.json',dict(reason=str(error),pid=os.getpid()))
        raise
