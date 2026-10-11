"""First native proof-score hypothesis, not a trained/calibrated ranker.

Every complete-commune leaf is scored before initial100. Frozen old vectors,
original query encoder/batches; separate address proofs, identity and SITE.
No pool reranking, label-based exceptions, training, or serving mutation.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import gc
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import time

from scripts import embeddinggemma2_v18_identity_retrieval as previous
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v2_data as data
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE=previous.SOURCE
OUTPUT=io.HOME/'v19/proof_engine_20261011_a'
RECIPE=dict(scope='ALL526_ALREADY_SEEN_FULL_SELECTED_COMMUNES_NOT_NATIONAL_CERTIFICATION',
    score='MAX_SITE_DUAL_IDENTITY_COHERENT_ADDRESS_NAME_PROOF_PLUS_SMALL_UNCERTAINTY_GEO',
    dense='max(site_cosine,0.75*site_cosine+0.25*identity_cosine)',
    numbered_address='max_per_complete_own_alias(0.65*street+0.20*number_suffix+0.15*official_name)',
    unnumbered_address='max_per_complete_own_alias(0.80*street+0.20*official_name)',
    name_proof='0.85*official_name+0.15*own_street',
    geo_bonus='0.05*reliability/(1+distance/(300+uncertainty_m))',
    weak_geo_reliability=.5,missing_geo_reliability=0.,
    weights='UNTRAINED_PREDECLARED_HEURISTIC_NOT_ESTIMATED_FROM_MISSES',
    no_grid=True,no_fixed_prepool=True,cap=100,
    coherent_address=True,no_number_hard_exclusion=True,
    source_coverage='ALL2554848_ORIGINAL_COMMUNE_SITES_NOT_NEW_NATIONAL_LEXICAL_OR_GRAPH_CHANNEL',
    variants=['baseline','v18_reference','evidence_native'],
    no_document_encoding=True,no_gradient_steps=True,closed_folds_opened=False,
    downstream_changed=False,pristine_oof=False)


@lru_cache(maxsize=131072)
def address(value):
    value=data.canonical(value)
    value=re.sub(r'^(\d+)([A-Z])(?=\s)',r'\1 \2',value)
    parts=value.split(); number='';suffix=''
    if parts and parts[0].isdigit():
        number=str(int(parts.pop(0)))
        if parts and parts[0] in ('BIS','TER','QUATER','A','B','C','D'):
            suffix=parts.pop(0)
    return number,suffix,' '.join(parts)


def query_address(q):
    # A leading number embedded in crm_street is query input, not a new alias.
    return address(' '.join(q[k] for k in ('crm_number','crm_suffix','crm_street')))


def number_grade(qnumber,qsuffix,number,suffix):
    if not qnumber or not number or qnumber!=number: return 0.
    if qsuffix==suffix: return 1.
    if not qsuffix or not suffix: return .8
    return 0.


def street_similarity(query,street):
    from rapidfuzz import fuzz
    generic={'RUE','AVENUE','BOULEVARD','ROUTE','CHEMIN','DE','DU','DES','LA','LE','LES'}
    if not query or not street or not (set(query.split())-generic): return 0.
    return fuzz.WRatio(query,street)/100


def address_proof(query,aliases,name):
    number,suffix,street=query; best=0.; best_street=0.
    for n,s,v in aliases:
        st=street_similarity(street,v); best_street=max(best_street,st)
        score=.65*st+.20*number_grade(number,suffix,n,s)+.15*name if number else .80*st+.20*name
        best=max(best,score)
    return best,best_street


def native_score(site,identity,addr,name,street,geo):
    import numpy as np
    return np.maximum.reduce((site,.75*site+.25*identity,addr,.85*name+.15*street))+.05*geo


def authority(out):
    launch=io.read(out/'launch.json')
    for ref in [launch['implementation'],launch['recipe'],launch['source_launch'],*launch['helpers']]: verify(ref)
    source=engine.authority(SOURCE)
    if source['new_adapter']!=launch['adapter']: raise ValueError('Frozen original encoder')
    return launch,source


def initialize(out):
    from src.xgb_matcher.retrieval_query_ban_v327 import surface
    if out.exists(): raise ValueError('New namespace only')
    old_state=io.read(previous.OUTPUT/'state.json'); verify(old_state['report'])
    ex=io.read(previous.OUTPUT/'generate_exit.json')
    if old_state['status']!='COMPLETE_IDENTITY_SITE_COMPARISON' or ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete V18 owned exit0 required')
    source=engine.authority(SOURCE); panel=io.read(verify(source['panel'])); surface(panel)
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel): raise ValueError('All526 safe query-only scope')
    old_launch=io.read(previous.OUTPUT/'launch.json'); verify(old_launch['implementation'])
    corpora=old_launch['corpora']; census=io.read(engine.sealed(previous.OUTPUT/'census.json'))
    out.mkdir(parents=True);io.write(out/'recipe.json',RECIPE)
    helpers=[record(Path(m.__file__)) for m in (io,engine,structural,previous,data)]
    helpers += [record(io.ROOT/p) for p in ('scripts/embeddinggemma2_compare.py',
        'scripts/embeddinggemma2_v3_fit.py','scripts/embeddinggemma2_v15_probability_pilot.py',
        'scripts/embeddinggemma2_v3_compare.py','src/xgb_matcher/retrieval_v20_workbench.py',
        'scripts/embeddinggemma2_v19_official_audit.py')]
    io.write(out/'launch.json',dict(implementation=record(Path(__file__)),helpers=helpers,recipe=record(out/'recipe.json'),
        source_launch=record(SOURCE/'launch.json'),corpora=corpora,adapter=source['new_adapter'],
        baseline_outputs=record(SOURCE/'all_outputs.json'),v18_outputs=io.read(previous.OUTPUT/'generation.json')['outputs'],
        v18_generation=record(previous.OUTPUT/'generation.json'),identity_database=record(engine.sealed(previous.OUTPUT/'identities.sqlite')),
        names=io.read(SOURCE/'names/generation.json')['database'],
        census=census,geo=record(previous.GEO_SOURCE/'geo.json'),created_unix=time.time(),truth_opened_in_generation=False))


def generate(out):
    import numpy as np
    import torch,transformers
    from scripts import embeddinggemma2_v3_fit as fit
    from scripts import embeddinggemma2_compare as old
    if torch.__version__!='2.14.1' or transformers.__version__!='5.19.0': raise ValueError('Original encoder runtime')
    launch,source=authority(out)
    if (out/'generation.json').exists(): verify(io.read(out/'generation.json')['outputs']); return
    panel=io.read(verify(source['panel'])); original=io.read(verify(launch['baseline_outputs']))
    v18=io.read(verify(launch['v18_outputs'])); geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geo']))}
    db=sqlite3.connect(f"file:{verify(launch['identity_database'])}?mode=ro&immutable=1",uri=True)
    names_db=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    cfg=io.read(verify(source['night_config']));engine.guard()
    model,params,encode,inference,tokenizer=fit.load_model(cfg,verify(launch['adapter']))
    cache=previous.VectorCache(previous.OUTPUT/'identity_vectors')
    keys=np.load(verify(source['geographic']['keys']),mmap_mode='r',allow_pickle=False)
    xy=np.load(verify(source['geographic']['xy']),mmap_mode='r',allow_pickle=False)
    outputs={'baseline':original['gemma2_lexical_geo'],'v18_reference':v18['joint_lexical_oldgeo'],'evidence_native':{}}
    count=0
    try:
        for zone,ref in sorted(launch['corpora'].items()):
            dest=out/'zones'/(zone or 'UNKNOWN');dest.mkdir(parents=True,exist_ok=True)
            saved=engine.sealed(dest/'outputs.json')
            if saved:
                part=io.read(saved);outputs['evidence_native'].update(part['outputs']);count+=part['sites'];continue
            engine.guard();docs=io.read(verify(ref));qs=[q for q in panel if q['crm_insee']==zone]
            values={q['query_id']:[] for q in qs}
            if docs:
                directory=engine.zone_path(SOURCE,zone);site_cache=directory/'v3_official512'
                if not site_cache.exists():site_cache=Path(io.read(directory/'v3_official512.seal.json')['path']).parent/'v3_official512'
                views=[]
                for i,d in enumerate(docs):
                    if i%256==0:engine.guard()
                    if 'site_views' not in d:
                        anchor=io.clean_text(d['own_names'][:1],d['all_addresses'][:1],d['postcode'])
                        d['site_views']=data.pack_atoms(data.site_atoms(d,d['extra']),tokenizer,512,anchor)[0]
                    views.append([io.DOC_PROMPT+v for v in d['site_views']])
                texts=list(dict.fromkeys(v for vs in views for v in vs));positions={v:i for i,v in enumerate(texts)}
                binding=io.read(site_cache/'texts_binding.json')
                if engine.text_hash(texts)!=binding['order_sha256'] or len(texts)!=binding['texts'] or binding['model_step']!=1000:
                    raise ValueError('Exact original SITE cache binding')
                matrices=[]
                for lo in range(0,len(texts),1024):
                    p=engine.sealed(site_cache/f'vectors_{lo:08d}.npy')
                    if not p:raise ValueError('Every original SITE shard required')
                    matrices.append(np.load(p,mmap_mode='r',allow_pickle=False))
                def vector(v):
                    k=positions[v];return matrices[k//1024][k%1024]
                qe=inference([io.QUERY_PROMPT+old.query_text(q) for q in qs])
                qi=inference([io.QUERY_PROMPT+'noms: '+data.safe(q['crm_name']) for q in qs])
                identity={}
                legal=engine.lookup(names_db,1,[d['siren'] for d in docs])
                for i,p in enumerate(sorted({int(d['siren']) for d in docs})):
                    if i%256==0:engine.guard()
                    ids=[r[0]-1 for r in db.execute('SELECT text_id FROM views WHERE parent=? ORDER BY ordinal',(p,))]
                    if not ids:raise ValueError('Every parent official context required')
                    identity[p]=(qi@np.stack([cache.get(k) for k in ids]).T).max(axis=1)
                features={k:np.lib.format.open_memmap(dest/(k+'.pending.npy'),mode='w+',dtype=np.float32,shape=(len(qs),len(docs))) for k in ('site','identity','name','address','street','geo')}
                qnames=[structural.normalize(data.safe(q['crm_name'])) for q in qs];qaddr=[query_address(q) for q in qs]
                best=[[] for q in qs];site_replay=[[] for q in qs];joint_replay=[[] for q in qs]
                for lo in range(0,len(docs),128):
                    engine.guard();part=docs[lo:lo+128];end=lo+len(part)
                    site=engine.score_views(qe,views[lo:end],vector)
                    ident=np.column_stack([identity[int(d['siren'])] for d in part])
                    name=np.zeros_like(site);addr=np.zeros_like(site);street=np.zeros_like(site);geo=np.zeros_like(site)
                    kk=np.array([int(d['key']) for d in part],dtype=np.uint64);idx=np.searchsorted(keys,kk)
                    found=idx<len(keys);found[found] &= keys[idx[found]]==kk[found]
                    coord=np.full((len(part),2),np.nan);coord[found]=xy[idx[found]]
                    valid=np.isfinite(coord).all(axis=1)&(coord[:,0]>=0)&(coord[:,0]<=1300000)&(coord[:,1]>=6000000)&(coord[:,1]<=7300000)
                    for j,d in enumerate(part):
                        names=tuple(dict.fromkeys(structural.normalize(data.safe(n)) for n in d['own_names']+sorted(legal[int(d['siren'])]) if data.safe(n)))
                        aliases=tuple(dict.fromkeys(address(a) for a in d['all_addresses'] if data.safe(a)))
                        for i,q in enumerate(qs):
                            name[i,j]=structural.similarity(qnames[i],names)
                            addr[i,j],street[i,j]=address_proof(qaddr[i],aliases,float(name[i,j]))
                    for i,q in enumerate(qs):
                        g=geometries[q['query_id']]
                        if g['spatial_crs_compatible']:
                            reliability=.5 if g['ban_declared_weak'] else 1.
                            distance=np.hypot(coord[:,0]-g['ban_x_l93'],coord[:,1]-g['ban_y_l93'])
                            geo[i,valid]=reliability/(1+distance[valid]/(300+max(0,g['ban_uncertainty_m'])))
                    scores=native_score(site,ident,addr,name,street,geo)
                    sites=[dict(siret=d['key'],siren=d['siren']) for d in part]
                    for i in range(len(qs)):
                        best[i]=old.top100(best[i],scores[i],sites)
                        site_replay[i]=old.top100(site_replay[i],site[i],sites)
                        joint_replay[i]=old.top100(joint_replay[i],(.75*site+.25*ident)[i],sites)
                    for key,value in zip(features,(site,ident,name,addr,street,geo)):features[key][:,lo:end]=value
                    if lo%8192==0:engine.progress(out,'FULL_CORPUS_NATIVE_PROOF_SCORE',count+lo,total=2554848,zone=zone)
                for q,a,b,c in zip(qs,best,site_replay,joint_replay):
                    for fresh,frozen in ((b,original['gemma2'][q['query_id']]),(c,v18['joint'][q['query_id']])):
                        if [z[1] for z in fresh]!=[z['siret'] for z in frozen] or any(abs(z[0]-r['score'])>1e-7 for z,r in zip(fresh,frozen)):
                            raise ValueError('SITE/identity all526 replay mismatch; no metrics')
                    values[q['query_id']]=[dict(score=s,siret=t,siren=p) for s,t,p in a]
                for k,a in features.items():
                    a.flush();a._mmap.close();os.replace(dest/(k+'.pending.npy'),dest/(k+'.npy'));engine.seal(dest/(k+'.npy'))
                io.write(dest/'features_binding.json',dict(corpus=ref,queries=qs,recipe=launch['recipe'],adapter=launch['adapter'],sirets=[d['key'] for d in docs],full_unpruned_scores=True))
                engine.seal(dest/'features_binding.json')
                for a in matrices:a._mmap.close()
                del features,matrices,identity,legal,views,texts,positions
            count+=len(docs);outputs['evidence_native'].update(values)
            io.write(dest/'outputs.json',dict(sites=len(docs),outputs=values,site_and_joint_replayed=True));engine.seal(dest/'outputs.json')
            del docs;gc.collect();engine.progress(out,'COMPLETE_COMMUNE_NATIVE_PROOF_SCORE',count,total=2554848,zone=zone)
        if count!=2554848:raise ValueError('All complete original sites before labels')
        scope={q['query_id'] for q in panel}
        for v in outputs.values():
            if set(v)!=scope or any(len(a)>100 or len({r['siret'] for r in a})!=len(a) for a in v.values()):raise ValueError('Full526 unique initial100')
        authority(out);verify(launch['adapter']);io.write(out/'outputs.json',outputs)
        io.write(out/'generation.json',dict(status='COMPLETE_SEALED_ALL526',outputs=record(out/'outputs.json'),queries=526,commune_sites=count,
            no_gradient_steps=True,no_document_encoding=True,truth_opened=False,site_and_joint_replayed_all526=True))
    finally:cache.close();db.close();names_db.close();keys._mmap.close();xy._mmap.close()


def evaluate(out):
    structural.evaluate(out)
    r=io.read(out/'report.json');r['status']='MEASURED_NATIVE_PROOF_HYPOTHESIS_ALL526';r['untrained_heuristic']=True
    r['national_lexical_and_graph_not_integrated_into_new_variant']=True
    old={v['query_id']:v for v in io.read(out/'outcomes.json')['v18_reference']}
    new=io.read(out/'outcomes.json')['evidence_native']
    r['paired_vs_v18']=dict(gained=[v['query_id'] for v in new if v['exact'] and not old[v['query_id']]['exact']],
        lost=[v['query_id'] for v in new if not v['exact'] and old[v['query_id']]['exact']])
    io.write(out/'report.json',r);io.write(out/'state.json',dict(status='COMPLETE_NATIVE_PROOF_COMPARISON',report=record(out/'report.json')))


def run(out,resume=False):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if resume:authority(out)
    else:initialize(out)
    env=dict(os.environ,TMPDIR=str(out),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    audit=None
    if not (out/'audit/completion.json').exists():
        log=(out/'official_audit.log').open('ab',buffering=0)
        audit=subprocess.Popen([str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v19_official_audit','--output',str(out/'audit')],stdout=log,stderr=subprocess.STDOUT,env=env)
        io.write(out/'audit_process.json',dict(pid=audit.pid,owned=True,started_unix=time.time()))
    while True:
        if engine.INTERRUPTED:raise engine.ResourceYield('Controller interrupted; no automatic continuation')
        try:engine.guard()
        except engine.ResourceYield:
            io.write(out/'state.json',dict(status='WAITING_FOR_REAL_RESOURCES',updated_unix=time.time()));time.sleep(60);continue
        authority(out);io.write(out/'state.json',dict(status='NATIVE_PROOF_GENERATION',updated_unix=time.time()))
        child=subprocess.Popen([str(engine.GPU),'-u','-m','scripts.embeddinggemma2_v19_proof_retrieval','generate','--output',str(out)],env=env)
        while True:
            try:rc=child.wait(timeout=5);break
            except subprocess.TimeoutExpired:
                if audit is not None and audit.poll() is not None and not (out/'audit_exit.json').exists():
                    io.write(out/'audit_exit.json',dict(exit_code=audit.returncode,pid=audit.pid,own_child_exit_confirmed=True,updated_unix=time.time()))
                if engine.INTERRUPTED:
                    child.terminate()
                    if audit and audit.poll() is None:audit.terminate()
                    try:rc=child.wait(timeout=90)
                    except subprocess.TimeoutExpired:child.kill();rc=child.wait()
                    io.write(out/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,interrupted=True))
                    raise engine.ResourceYield('Owned child stopped, shards preserved')
        io.write(out/'generate_exit.json',dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,updated_unix=time.time()))
        if rc==75:time.sleep(60);continue
        if rc:raise RuntimeError(f'Generation exit{rc}; retain shards, no metrics')
        break
    evaluate(out)
    if audit:
        rc=audit.wait();io.write(out/'audit_exit.json',dict(exit_code=rc,pid=audit.pid,own_child_exit_confirmed=True))


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','resume','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action in ('run','resume'):run(a.output,a.action=='resume')
        else:globals()[a.action](a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/'execution_error.json',dict(error=repr(e),updated_unix=time.time()))
        raise
