"""Native identity+site retrieval on the complete, already-seen V15 pilot.

Frozen before outcomes. All official parents/sites are scored, never a fixed
parent beam or a reranked 500/2000 pool. SITE vectors are reused literally;
only the trained identity task is encoded. No gradients or closed-fold labels.
"""
import argparse
from collections import OrderedDict, defaultdict
import gc
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = structural.SOURCE
GEO_SOURCE = io.HOME/'v17/structural_20261010_a'
OUTPUT = io.HOME/'v18/identity_site_20261010_a'
RECIPE = dict(
    scope='ALL526_ALREADY_SEEN_PROBABILITY_PILOT_NOT_PRISTINE_OOF',
    identity='ORIGINAL_TRAINED_NAME_ONLY_QUERY_MAX_OFFICIAL_PARENT_VIEW_COSINE',
    names='LEGAL_PARENT_PLUS_ALL_NATIONAL_SIBLING_NAMES_ROLE_LABELLED',
    site='EXACT_REUSE_ORIGINAL_SITE512_MAX_VIEW_COSINE',
    joint='0.75_SITE_PLUS_0.25_IDENTITY_OVER_EVERY_COMPLETE_COMMUNE_SITE',
    weight_source='PRIOR_UNMEASURED_V4_DUAL_HYPOTHESIS_NOT_FIT_TO_V15_OUTCOMES',
    corpus='ALL_2554848_SITES_295_COMPLETE_COMMUNES_ALL526_QUERIES',
    max_tokens=512, inference_batch=8, adapter_step=1000,
    fusion='SUM_RRF60_DENSE_LEXICAL_GEO_ORIGINAL_WEAK_GEO_WEIGHT',
    variants=['baseline','joint','joint_lexical_oldgeo','joint_lexical_newgeo'],
    no_fixed_prepool=True, output_cap=100, no_new_training=True,
    no_grid=True, closed_folds_opened=False, downstream_changed=False,
    sibling_names_are_parent_context_not_site_aliases=True)


def joint_scores(site, identity):
    return .75*site+.25*identity


def validate_outputs(outputs, scope):
    if set(outputs)!=set(RECIPE['variants']): raise ValueError('Frozen variants')
    for values in outputs.values():
        if set(values)!=scope: raise ValueError('Every query before labels')
        for rows in values.values():
            if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):
                raise ValueError('Unique absolute initial100 cap')


def authority(out):
    launch=io.read(out/'launch.json')
    for ref in [launch['implementation'],launch['recipe'],launch['source_launch'],*launch['helpers']]: verify(ref)
    source=engine.authority(SOURCE)
    if source['new_adapter']!=launch['adapter']: raise ValueError('Frozen adapter')
    return launch,source


def initialize(out):
    import ijson
    from src.xgb_matcher.retrieval_query_ban_v327 import surface
    if out.exists(): raise ValueError('New namespace only; resume is explicit')
    source=engine.authority(SOURCE)
    panel=io.read(verify(source['panel'])); surface(panel)
    if len(panel)!=526 or len({q['query_id'] for q in panel})!=526 or any(q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('All526 query-only safe folds')
    if io.read(SOURCE/'state.json')['status']!='COMPLETE_PROBABILITY_PILOT': raise ValueError('Complete baseline')
    baseline=record(SOURCE/'all_outputs.json')
    if baseline['sha256']!=structural.BASELINE_SHA: raise ValueError('Immutable baseline')
    geo_gen=io.read(GEO_SOURCE/'generation.json'); geo_exit=io.read(GEO_SOURCE/'generate_exit.json')
    if geo_gen['status']!='COMPLETE_SEALED_ALL526' or geo_exit['exit_code']!=0 or not geo_exit['own_child_exit_confirmed']:
        raise ValueError('Complete query-only new geo generation')
    census=io.read(verify(source['census'])); corpora={}
    for spec in census['zones']:
        zone=spec['zone']; directory=engine.zone_path(SOURCE,zone)
        for tag in ('prepare','v3_official512'):
            ex=io.read(directory/(tag+'_exit.json'))
            if ex['exit_code']!=0 or not ex['own_child_exit_confirmed']: raise ValueError('Original owned exit0')
        corpora[zone]=io.read(directory/'documents.seal.json')
    out.mkdir(parents=True); io.write(out/'recipe.json',RECIPE)
    modules=[engine,io,structural]
    helpers=[record(Path(m.__file__)) for m in modules]+[record(io.ROOT/p) for p in (
        'scripts/embeddinggemma2_compare.py','scripts/embeddinggemma2_v2_data.py',
        'scripts/embeddinggemma2_v3_fit.py','scripts/embeddinggemma2_v15_probability_pilot.py',
        'scripts/embeddinggemma2_v3_compare.py','scripts/embeddinggemma2_v10_representative.py',
        'src/xgb_matcher/retrieval_v20_workbench.py','src/xgb_matcher/retrieval_query_ban_v327.py')]
    names=io.read(SOURCE/'names/generation.json')['database']; verify(names)
    launch=dict(source_launch=record(SOURCE/'launch.json'),baseline_outputs=baseline,
        implementation=record(Path(__file__)),helpers=helpers,recipe=record(out/'recipe.json'),
        names=names,corpora=corpora,adapter=source['new_adapter'],
        new_geo=record(GEO_SOURCE/'geo.json'),new_geo_generation=record(GEO_SOURCE/'generation.json'),
        created_unix=time.time(),labels_opened=False)
    io.write(out/'launch.json',launch)
    count=0; parents=set(); zones={}
    for zone,ref in sorted(corpora.items()):
        engine.guard(); local=set(); n=0
        with verify(ref).open('rb') as stream:
            for d in ijson.items(stream,'item'):
                local.add(int(d['siren'])); n+=1
                if n%8192==0: engine.guard()
        zones[zone]=dict(sites=n,parents=len(local)); parents.update(local); count+=n
        engine.progress(out,'QUERY_ONLY_CORPUS_CENSUS',count,total_sites=2554848,parents=len(parents),zone=zone)
    if count!=2554848 or len(zones)!=295: raise ValueError('Complete original communes')
    io.write(out/'census.json',dict(sites=count,parents=len(parents),zones=zones,queries=526,truth_opened=False))
    engine.seal(out/'census.json')


def prepare(out):
    """Tokenizer only. Transactions bind each complete parent, resume same hash."""
    from transformers import AutoTokenizer
    from scripts import embeddinggemma2_v2_data as data
    launch,_=authority(out); census=io.read(engine.sealed(out/'census.json'))
    if engine.sealed(out/'identities.sqlite'): return
    tokenizer=AutoTokenizer.from_pretrained(io.HOME/'base',local_files_only=True)
    names=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    db=sqlite3.connect(out/'identities.sqlite'); db.execute('PRAGMA journal_mode=DELETE'); db.execute('PRAGMA cache_size=-65536')
    db.executescript('''CREATE TABLE IF NOT EXISTS texts(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL);
        CREATE TABLE IF NOT EXISTS views(parent INTEGER NOT NULL, ordinal INTEGER NOT NULL, text_id INTEGER NOT NULL,
            PRIMARY KEY(parent,ordinal)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS parents(parent INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS zones(zone TEXT PRIMARY KEY);''')
    db.commit(); started=time.monotonic(); done=0
    try:
        for zone,ref in sorted(launch['corpora'].items()):
            if db.execute('SELECT 1 FROM zones WHERE zone=?',(zone,)).fetchone(): continue
            engine.guard(); docs=io.read(verify(ref)); parents=sorted({int(d['siren']) for d in docs}); del docs; gc.collect()
            for i,p in enumerate(parents):
                if db.execute('SELECT 1 FROM parents WHERE parent=?',(p,)).fetchone(): continue
                legal=[r[0] for r in names.execute('SELECT name FROM aliases WHERE kind=1 AND owner_key=? ORDER BY name',(p,))]
                siblings=sorted({r[0] for r in names.execute('SELECT name FROM aliases WHERE kind=2 AND owner_key>=? AND owner_key<?',(p*100000,(p+1)*100000))})
                atoms=data.identity_atoms(dict(legal=[data.safe(n) for n in legal if data.safe(n)],sites=[data.safe(n) for n in siblings if data.safe(n)]))
                packed,_=data.pack_atoms(atoms,tokenizer,512)
                for ordinal,body in enumerate(packed):
                    text=io.DOC_PROMPT+body
                    db.execute('INSERT OR IGNORE INTO texts(text) VALUES(?)',(text,))
                    idx=db.execute('SELECT id FROM texts WHERE text=?',(text,)).fetchone()[0]
                    db.execute('INSERT INTO views VALUES(?,?,?)',(p,ordinal,idx))
                db.execute('INSERT INTO parents VALUES(?)',(p,)); done+=1
                if done%256==0:
                    db.commit(); engine.guard()
                    engine.progress(out,'PACKING_ALL_OFFICIAL_PARENT_NAMES',done,zone=zone,
                        zone_parent=i+1,zone_total=len(parents),elapsed_s=time.monotonic()-started)
            db.execute('INSERT INTO zones VALUES(?)',(zone,)); db.commit()
        n=db.execute('SELECT COUNT(*) FROM parents').fetchone()[0]
        texts=db.execute('SELECT COUNT(*),MIN(id),MAX(id) FROM texts').fetchone()
        if n!=census['parents'] or texts[1]!=1 or texts[0]!=texts[2]: raise ValueError('Complete unique parent/text coverage')
        if db.execute('SELECT COUNT(*) FROM zones').fetchone()[0]!=295: raise ValueError('All communes packed')
        io.write(out/'identity_counts.json',dict(parents=n,unique_texts=texts[0],views=db.execute('SELECT COUNT(*) FROM views').fetchone()[0],truth_opened=False))
    finally: db.close(); names.close()
    authority(out); engine.seal(out/'identities.sqlite'); engine.seal(out/'identity_counts.json')


class VectorCache:
    """Bounded mmap descriptors; every vector shard validated, no ANN."""
    def __init__(self,directory): self.directory=directory; self.loaded=OrderedDict(); self.epochs={}
    def get(self,idx):
        import numpy as np
        shard=idx//1024; path=self.directory/f'vectors_{shard*1024:08d}.npy'
        if shard not in self.loaded:
            stamp=(path.stat().st_size,path.stat().st_mtime_ns)
            if self.epochs.get(shard)!=stamp:
                if not engine.sealed(path): raise ValueError('Unsealed identity vector')
                self.epochs[shard]=stamp
            self.loaded[shard]=np.load(path,mmap_mode='r',allow_pickle=False)
            if len(self.loaded)>32: _,matrix=self.loaded.popitem(last=False); matrix._mmap.close()
        self.loaded.move_to_end(shard)
        # Copy: the cache may evict this mapping while collecting other views.
        return self.loaded[shard][idx%1024].copy()
    def close(self):
        for matrix in self.loaded.values(): matrix._mmap.close()
        self.loaded.clear()


def generate(out):
    import numpy as np
    import torch
    import transformers
    from scripts import embeddinggemma2_v3_fit as fit
    from scripts import embeddinggemma2_v2_data as data
    from scripts import embeddinggemma2_compare as old
    if torch.__version__!='2.14.1' or transformers.__version__!='5.19.0': raise ValueError('Original numeric runtime')
    launch,source=authority(out)
    target=out/'generation.json'
    if target.exists():
        verify(io.read(target)['outputs']); return
    database=engine.sealed(out/'identities.sqlite')
    if not database: raise ValueError('All official identity views sealed before encoding')
    counts=io.read(engine.sealed(out/'identity_counts.json')); panel=io.read(verify(source['panel']))
    original=io.read(verify(launch['baseline_outputs'])); scope={q['query_id'] for q in panel}
    db=sqlite3.connect(f'file:{database}?mode=ro&immutable=1',uri=True)
    cfg=io.read(verify(source['night_config'])); checkpoint=torch.load(verify(launch['adapter']),map_location='cpu',weights_only=True)
    if checkpoint['step']!=1000 or checkpoint['launch_sha256']!=io.digest(Path(launch['adapter']['path']).parent/'launch.json'):
        raise ValueError('Original finalized best1000 checkpoint binding')
    engine.guard(); model,params,encode,inference,tokenizer=fit.load_model(cfg,verify(launch['adapter']))
    del checkpoint; vectors=out/'identity_vectors'; vectors.mkdir(exist_ok=True)
    engine.bind(vectors/'binding.json',dict(database=record(database),adapter=launch['adapter'],recipe=launch['recipe'],implementation=launch['implementation'],batch=8,dimension=768))
    started=time.monotonic(); fresh=0
    for lo in range(0,counts['unique_texts'],1024):
        engine.guard(); path=vectors/f'vectors_{lo:08d}.npy'
        if not engine.sealed(path):
            texts=[r[0] for r in db.execute('SELECT text FROM texts WHERE id>? AND id<=? ORDER BY id',(lo,lo+1024))]
            chunks=[]
            for offset in range(0,len(texts),128):
                engine.guard(); a=inference(texts[offset:offset+128])
                if a.dtype!=np.float32 or a.shape!=(len(texts[offset:offset+128]),768) or not np.isfinite(a).all(): raise ValueError('Finite original768 vectors')
                chunks.append(a); fresh+=len(a)
                engine.progress(out,'ENCODING_IDENTITY_ONLY',lo+offset+len(a),total=counts['unique_texts'],fresh_texts=fresh,elapsed_s=time.monotonic()-started)
            temporary=path.with_suffix('.pending.npy'); np.save(temporary,np.concatenate(chunks),allow_pickle=False); os.replace(temporary,path); engine.seal(path)
    cache=VectorCache(vectors); joint={}; total=0
    try:
        for zone,ref in sorted(launch['corpora'].items()):
            destination=out/'zones'/ (zone or 'UNKNOWN'); destination.mkdir(parents=True,exist_ok=True)
            existing=engine.sealed(destination/'joint.json')
            if existing:
                saved=io.read(existing); joint.update(saved['outputs']); total+=saved['sites']; continue
            engine.guard(); docs=io.read(verify(ref)); qs=[q for q in panel if q['crm_insee']==zone]
            if not docs:
                values={q['query_id']:[] for q in qs}
            else:
                directory=engine.zone_path(SOURCE,zone); site_cache=directory/'v3_official512'
                if not site_cache.exists(): site_cache=Path(io.read(directory/'v3_official512.seal.json')['path']).parent/'v3_official512'
                views=[]
                for i,d in enumerate(docs):
                    if i%256==0: engine.guard()
                    if 'site_views' not in d:
                        anchor=io.clean_text(d['own_names'][:1],d['all_addresses'][:1],d['postcode'])
                        d['site_views']=data.pack_atoms(data.site_atoms(d,d['extra']),tokenizer,512,anchor)[0]
                    views.append([io.DOC_PROMPT+v for v in d['site_views']])
                texts=list(dict.fromkeys(v for vs in views for v in vs)); positions={v:i for i,v in enumerate(texts)}
                binding=io.read(site_cache/'texts_binding.json')
                if engine.text_hash(texts)!=binding['order_sha256'] or len(texts)!=binding['texts'] or binding['model_step']!=1000 or binding['batch_size']!=8:
                    raise ValueError('Exact original SITE vector text/model binding')
                matrices=[np.load(engine.sealed(site_cache/f'vectors_{lo:08d}.npy'),mmap_mode='r',allow_pickle=False) for lo in range(0,len(texts),1024)]
                def site_vector(v):
                    idx=positions[v]; return matrices[idx//1024][idx%1024]
                qe=inference([io.QUERY_PROMPT+old.query_text(q) for q in qs])
                qi=inference([io.QUERY_PROMPT+'noms: '+data.safe(q['crm_name']) for q in qs])
                parents={}
                for i,p in enumerate(sorted({int(d['siren']) for d in docs})):
                    if i%256==0: engine.guard()
                    ids=[r[0]-1 for r in db.execute('SELECT text_id FROM views WHERE parent=? ORDER BY ordinal',(p,))]
                    if not ids: raise ValueError('Every parent has complete official context')
                    parents[p]=(qi@np.stack([cache.get(idx) for idx in ids]).T).max(axis=1)
                best=[[] for q in qs]; replay=[[] for q in qs]
                for lo in range(0,len(docs),128):
                    engine.guard(); part=docs[lo:lo+128]
                    site=engine.score_views(qe,views[lo:lo+128],site_vector)
                    parent=np.column_stack([parents[int(d['siren'])] for d in part]); scores=joint_scores(site,parent)
                    sites=[dict(siret=d['key'],siren=d['siren']) for d in part]
                    for i in range(len(qs)):
                        best[i]=old.top100(best[i],scores[i],sites); replay[i]=old.top100(replay[i],site[i],sites)
                    if lo%8192==0: engine.progress(out,'SCORING_ALL_SITES_NATIVE_JOINT100',total+lo,zone=zone,total=2554848)
                values={}
                for q,a,b in zip(qs,best,replay):
                    frozen=original['gemma2'][q['query_id']]
                    if [r['siret'] for r in frozen]!=[r[1] for r in b] or any(abs(r['score']-v[0])>1e-7 for r,v in zip(frozen,b)):
                        raise ValueError('Full-query SITE replay mismatch; no metrics')
                    values[q['query_id']]=[dict(score=s,siret=t,siren=n) for s,t,n in a]
                for matrix in matrices: matrix._mmap.close()
                del matrices,parents,views,texts,positions
            total+=len(docs); joint.update(values)
            io.write(destination/'joint.json',dict(sites=len(docs),outputs=values,site_replay=True)); engine.seal(destination/'joint.json')
            del docs; gc.collect()
        if total!=2554848 or set(joint)!=scope: raise ValueError('Complete pilot required, no partial recall')
        oldgeo={r['query_id']:r for r in io.read(verify(source['projected_geo']))}
        newgeo=io.read(verify(launch['new_geo']))
        if isinstance(newgeo,list): newgeo={r['query_id']:r for r in newgeo}
        outputs=dict(baseline=original['gemma2_lexical_geo'],joint=joint)
        for tag,geo in [('joint_lexical_oldgeo',oldgeo),('joint_lexical_newgeo',newgeo)]:
            outputs[tag]={q:structural.fuse([(joint[q],1.),(original['lexical'][q],1.),(geo[q]['candidates'],geo[q]['rrf_geo_weight'])],'sum') for q in scope}
        validate_outputs(outputs,scope); authority(out); verify(launch['adapter']); verify(launch['new_geo']); verify(launch['names'])
        io.write(out/'outputs.json',outputs)
        io.write(target,dict(status='COMPLETE_SEALED_ALL526',outputs=record(out/'outputs.json'),queries=526,commune_sites=total,
            identity_counts=record(out/'identity_counts.json'),truth_opened=False,site_replay_all526=True,no_gradient_steps=True))
    finally: cache.close(); db.close()


def evaluate(out):
    # Immutable evaluator already enforces owned exit0, full526, fold234,
    # sealed official label-side, all-human weights/exact/SIREN/operational.
    launch,_=authority(out); outputs=io.read(verify(io.read(out/'generation.json')['outputs']))
    validate_outputs(outputs,set(outputs['baseline']))
    structural.evaluate(out)
    report=io.read(out/'report.json'); report['status']='MEASURED_NATIVE_IDENTITY_SITE_DEVELOPMENT_ALL526'
    report['recipe']=launch['recipe']; report['identity_is_trained_but_not_previously_served']=True
    report['site_replay_all526']=True; report['no_attribution_to_new_training']=True
    io.write(out/'report.json',report)
    io.write(out/'state.json',dict(status='COMPLETE_IDENTITY_SITE_COMPARISON',report=record(out/'report.json')))


def run(out,resume=False):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Already owned by an active controller; no duplicate')
    if resume: authority(out)
    else: initialize(out)
    env=dict(os.environ,TMPDIR=str(out),TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    for action in ('prepare','generate'):
        while True:
            engine.guard()
            authority(out)
            io.write(out/'state.json',dict(status=action.upper(),updated_unix=time.time()))
            child=subprocess.Popen([str(engine.GPU),'-u','-m','scripts.embeddinggemma2_v18_identity_retrieval',action,'--output',str(out)],env=env)
            while True:
                try: rc=child.wait(timeout=5); break
                except subprocess.TimeoutExpired:
                    if engine.INTERRUPTED:
                        child.terminate()
                        try: rc=child.wait(timeout=90)
                        except subprocess.TimeoutExpired: child.kill(); rc=child.wait()
                        io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,interrupted=True,updated_unix=time.time()))
                        raise engine.ResourceYield('Controller interrupted; owned child stopped, artifacts preserved')
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,updated_unix=time.time()))
            if rc==75:
                io.write(out/'state.json',dict(status='RESOURCE_YIELD_RESTART_SAME_BINDING',phase=action,updated_unix=time.time()))
                time.sleep(60); continue
            if rc: raise RuntimeError(f'{action} exit{rc}; retain shards, NO labels')
            break
    evaluate(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['run','resume','prepare','generate','evaluate'])
    parser.add_argument('--output',type=Path,default=OUTPUT); args=parser.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted); signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if args.action in ('run','resume'): run(args.output,resume=args.action=='resume')
        else: globals()[args.action](args.output)
    except engine.ResourceYield as error:
        print(str(error),flush=True); sys.exit(75)
    except Exception as error:
        if args.output.exists(): io.write(args.output/'execution_error.json',dict(error=repr(error),action=args.action,updated_unix=time.time()))
        raise
