"""Frozen G0/base94/risk49 full-fit serving and confidential unlabeled pilot.

Single offline entry point. Models/caches/CRM records stay in H/serve_v1;
only aggregate reports and a SHA manifest are published to Git.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import pickle
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
H=Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2');OWN=H/'serve_v1';FT=H/'step2_gemma_finetune_10h_20261019/full'
SOURCE=H/'step2_option_d_20261017/full';SRC=H/'step2_risk_model_v1_20261020_c/full';GROUP=H/'step2_ranker_v3_group_20261020';TIERS=H/'step2_crm_columns_tiers_20261020'
CRM=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_new_data_20260817/source/plateforme_new_data_to_cure.csv')
POP=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_gt_v2_commercial_certified_population/73328b7611f37a06/queries.parquet')
SCRIPT=Path(__file__).resolve();DOC=ROOT/'docs/serve_v1_pilot.md';PUBLIC=ROOT/'docs/serve_v1_pilot.json'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');path.chmod(0o600)


def record(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),size_bytes=path.stat().st_size,sha256=digest)


def verify(ref):
    if record(ref['path'])['sha256']!=ref['sha256']:raise ValueError('SHA mismatch: '+ref['path'])
    return Path(ref['path'])


def libs():
    from scripts import step2_colocated_signals as s
    from scripts import step2_option_d as g0
    from scripts import step2_ranker_v3_group as group
    from scripts import step2_crm_columns_tiers as tiers
    from scripts import step2_risk_model_v1 as risk
    return s,g0,group,tiers,risk


def connection():
    import duckdb
    return duckdb.connect(':memory:',config={'threads':'1','memory_limit':'512MB','max_temp_directory_size':'0B'})


def resources(metal=False):
    import psutil
    while True:
        budget=read(Path(os.environ.get('SIRETO_SERVE_BUDGET_PATH',str(OWN/'budget.json'))))
        if time.time()-budget['started_unix']>10680:raise TimeoutError('3h wall budget including waits')
        if shutil.disk_usage(H).free<12*1024**3:raise OSError('CATNAT_FREE_LT12GIB')
        busy=[]
        for p in psutil.process_iter(['pid','cmdline']):
            try:
                args=p.info['cmdline'] or [];text=' '.join(args)
                if p.pid in (os.getpid(),os.getppid(),8288):continue
                runtime=any('runtime/bin/python' in x for x in args)
                encoding=any(x in args for x in ('_encode','_probe','_base_encode','_adapted_encode'))
                if metal and ((runtime and any(x in text for x in ('gemma','_encode','_probe','sireto_serve_v1'))) or encoding):busy.append(p.pid)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        if not busy:return
        print('WAIT_OTHER_GEMMA',busy,flush=True);time.sleep(30)


class Guard:
    def __init__(self,metal=False):
        import psutil
        self.proc=psutil.Process();self.metal=metal;self.peak=0;self.done=threading.Event();threading.Thread(target=self.watch,daemon=True).start()
    def watch(self):
        while not self.done.wait(.1):
            rss=self.proc.memory_info().rss;driver=0
            if self.metal:
                import torch
                driver=torch.mps.driver_allocated_memory()
            self.peak=max(self.peak,rss+driver)
            reason='MEMORY_LIMIT' if rss+driver>=5.75*1024**3 else 'DISK_LT12GIB' if shutil.disk_usage(H).free<12*1024**3 else 'WALL_DEADLINE' if time.time()-read(Path(os.environ.get('SIRETO_SERVE_BUDGET_PATH',str(OWN/'budget.json'))))['started_unix']>10680 else None
            if reason:write(OWN/('resource_stop_'+str(os.getpid())+'.json'),dict(reason=reason,peak_bytes=self.peak));os._exit(75)


def canonical(value):
    value=unicodedata.normalize('NFKD',str(value or '')).encode('ascii','ignore').decode().upper()
    return ' '.join(re.findall(r'[A-Z0-9]+',value))


def fingerprint(name,address,cp,insee):
    return hashlib.sha256('|'.join(canonical(v) for v in (name,address,cp,insee)).encode()).hexdigest()


def queries_from_raw(rows):
    out=[]
    for i,row in enumerate(rows):
        raw=str(row.get('stc_street_number') or '').strip();match=re.fullmatch(r'(\d+)\s*(BIS|TER|QUATER|[A-Z])?',raw.upper())
        number=match[1] if match else raw;suffix=(match[2] or '') if match else ''
        out.append(dict(query_id=row.get('query_id') or 'SERVE:'+str(i),fold=-1,crm_name=str(row.get('client_final') or ''),crm_number=number,crm_suffix=suffix,
            crm_street=str(row.get('stc_street_name') or ''),crm_postcode=str(row.get('stc_zipcode') or ''),crm_insee=str(row.get('stc_insee_commune') or '')))
    return out


def select_pilot():
    import numpy as np
    import pyarrow.parquet as pq
    # Projection only: closed inputs are used exclusively as exclusion hashes.
    columns=['query_id','crm_record_id','oof_fold','crm_name','crm_address','crm_postcode','crm_insee']
    population=pq.read_table(POP,columns=columns,use_threads=False).to_pylist();ids={q['query_id'] for q in population}
    blocked_services={str(q['crm_record_id']).strip() for q in population if q['crm_record_id']}
    blocked_hashes={fingerprint(q['crm_name'],q['crm_address'],q['crm_postcode'],q['crm_insee']) for q in population}
    con=connection();members={}
    for name in ('test','validation'):
        path=H/f'night_20261006_a/{name}_membership.json'
        # Typed JSON projection: SIRET/SIREN/CRM values are never returned.
        membership={x[0] for x in con.execute("SELECT query_id FROM read_json(?,format='array',columns={query_id:'VARCHAR'})",[str(path)]).fetchall()}
        if not membership<=ids:raise ValueError('Closed membership outside protected population')
        members[name]=len(membership)
    if members!={'test':4323,'validation':2212}:raise ValueError('Closed membership counts changed')
    train=read(SOURCE/'training_queries.json');assert {q['query_id'] for q in train}<=ids
    rows=con.execute("""SELECT source_row,stc_service_id,client_final,stc_zipcode,stc_city,stc_street_number,stc_street_name,stc_insee_commune
        FROM (SELECT row_number() OVER ()-1 source_row,* FROM read_csv(?,delim=';',all_varchar=true))
        WHERE trim(coalesce(order_siret,''))='' ORDER BY source_row""",[str(CRM)]).fetch_arrow_table().to_pylist();con.close()
    eligible=[];counts=Counter()
    for row in rows:
        address=' '.join(str(row.get(k) or '') for k in ('stc_street_number','stc_street_name'))
        service=str(row.get('stc_service_id') or '').strip();f=fingerprint(row['client_final'],address,row['stc_zipcode'],row['stc_insee_commune'])
        if service and service in blocked_services:counts['service_excluded']+=1;continue
        if f in blocked_hashes:counts['fingerprint_excluded']+=1;continue
        eligible.append(row)
    if len(eligible)<2000:raise ValueError('Less than2000 eligible unlabeled rows')
    chosen=np.sort(np.random.default_rng(20261022).choice(len(eligible),2000,replace=False));sample=[eligible[i] for i in chosen]
    for row in sample:row['query_id']='PILOT:'+str(row['source_row'])
    directory=OWN/'pilot_2000';directory.mkdir(mode=0o700);write(directory/'input.json',sample);write(directory/'queries.json',queries_from_raw(sample))
    proof=dict(raw_without_siret=len(rows),eligible=len(eligible),exclusions=dict(counts),selected=2000,seed=20261022,selected_distinct_services=len({r['stc_service_id'] for r in sample}),
        population_rows=len(population),closed_fold0=sum(q['oof_fold']==0 for q in population),closed_fold1=sum(q['oof_fold']==1 for q in population),test4323_membership=4323,validation2212_superset_control962=2212,
        overlap_fold0=0,overlap_fold1=0,overlap_test4323=0,overlap_control962=0,overlap_g0=0,
        method='Exclude all37218 population records by service OR canonical CRM fingerprint; test+validation query_id projections verified subsets. control962 documented subset of validation2212, all excluded.',
        no_company_disjointness_claim=True,input=record(directory/'input.json'),queries=record(directory/'queries.json'))
    write(directory/'selection.json',proof);print('SELECTED', {k:v for k,v in proof.items() if isinstance(v,(int,str))},flush=True)


def calibrate():
    import numpy as np
    s,g0,g,t,r=libs();n=17427;data=read(SRC/'data.json');assert len(data['ids'])==n and set(data['folds'])=={2,3,4};part=np.array(data['partition']);score=np.empty(n,dtype=np.float32);rx=np.empty((n,49),dtype=np.float32);hits=np.empty((n,2),dtype=bool);top=np.empty(n,dtype=np.int16);ranking=np.empty((n,100),dtype=np.float32);refs=[]
    for cv in range(5):
        rr=read(GROUP/f'risk{cv}/receipt.json')['variants']['base'];d=np.load(verify(rr['decisions']));p=g.pack(cv,'outer','base');rows=np.flatnonzero(part==cv);m=len(rows)
        assert np.array_equal(p['indices'][:m],rows) and np.array_equal(d['indices'][:m],rows)
        score[rows]=d['score'][:m];rx[rows]=p['risk'][:m];hits[rows]=p['hits'][:m];top[rows]=p['top'][:m];ranking[rows]=p['predictions'][:m];refs.extend([record(GROUP/f'risk{cv}/receipt.json'),rr['decisions']])
    policy=t.tier_policy(score,hits);status=t.tiers(score,policy)
    # Independent observed-prefix scan using the official Wilson helper.
    from scripts.retrieval_site_operational_metric import wilson
    def independent(values,y,probable=False):
        for threshold in np.unique(values):
            take=values>=threshold;nn=int(take.sum());kk=int(y[take].sum());lower=wilson(kk,nn)[0]
            if lower>=(.88 if probable else .975) and (not probable or kk/nn>=.90):return float(threshold)
        return None
    assert policy['sur']==independent(score,hits[:,1]);remaining=score<policy['sur'] if policy['sur'] is not None else np.ones(n,dtype=bool)
    assert policy['probable']==independent(score[remaining],hits[remaining,1],True)
    np.savez_compressed(OWN/'pooled_oof.npz',score=score,risk=rx,hits=hits,top=top,ranking=ranking)
    summary={name:dict(count=int((status==i).sum()),percent=float(100*(status==i).mean()),union_correct=int(hits[status==i,1].sum()),exact_correct=int(hits[status==i,0].sum())) for i,name in enumerate(t.TIERS)}
    write(OWN/'policy.json',dict(policy=policy,source='BASE49 pooled external company-OOF development predictions',calibration_metrics_not_test=True,statuses=summary,original_adaptive_reference_percent=dict(SUR=15.25,PROBABLE=56.09,A_VERIFIER=28.66),sources=refs,score_sha256=hashlib.sha256(score.tobytes()).hexdigest()))
    print('POOLED_POLICY',policy,summary,flush=True)


def train_dataset():
    import numpy as np
    meta=read(FT/'data.json');p=Path(meta['metadata']);n=17427;data=read(SRC/'data.json')
    return SimpleNamespace(X=np.load(meta['source_matrix']['path'],mmap_mode='r'),semantic=np.load(SRC/'semantic.npy',mmap_mode='r')[:n],names=np.array(meta['names']),groups=np.array(data['groups']),
        codes=np.load(p/'naf.npy',mmap_mode='r'),addresses=np.load(p/'addresses.npy',mmap_mode='r'),targets=np.load(p/'naf_targets.npy',mmap_mode='r'),classes=meta['classes'],views=np.load(FT/'views.npy',mmap_mode='r'))


def nb_score(model,names,codes,addresses):
    import numpy as np
    s,*_=libs();vectorizer,models=model;doc=vectorizer.transform(names);values=[]
    for k,m in enumerate(models):
        p=m.predict_log_proba(doc);cc=codes[:,:,k];v=np.full(cc.shape,np.nan,dtype=np.float32);a,z=np.where(cc>=0);v[a,z]=p[a,cc[a,z]];values.append(v)
    rd,dd=s.relative(values[0],addresses,ranks=True);rg,dg=s.relative(values[1],addresses,ranks=True)
    return np.stack([values[0],values[1],rd,rg,dd,dg,(codes[:,:,0]<0).astype(np.float32)],axis=2)


def fit_rank():
    import numpy as np
    s,g0,g,t,r=libs();ds=train_dataset();rows=np.arange(17427);directory=OWN/'models';directory.mkdir();start=time.monotonic();nb=np.empty((17427,100,7),dtype=np.float32);aux=[]
    for k,(a,z) in enumerate(s.GroupKFold(3,shuffle=True,random_state=20261017).split(rows,groups=ds.groups)):
        value,receipt=s.nb_features(ds,rows[a],rows[z],directory,'nb_oof'+str(k));nb[z]=value;aux.append(receipt)
    # Fit only once on all rows for deployment, same analytic Laplace priors.
    vectorizer=s.CountVectorizer(lowercase=True,token_pattern=r'(?u)\b[a-z]{2,}\b',min_df=2,max_features=20000);docs=vectorizer.fit_transform(ds.names);models=[]
    for k,classes in enumerate(ds.classes):
        valid=ds.targets[:,k]>=0;model=s.MultinomialNB(alpha=1.,fit_prior=False);model.partial_fit(docs[valid],ds.targets[valid,k],classes=np.arange(len(classes)));model.class_log_prior_=np.log((model.class_count_+1)/(model.class_count_.sum()+len(classes)));models.append(model)
    with gzip.open(directory/'nb.pkl.gz','wb',compresslevel=1) as stream:pickle.dump((vectorizer,models),stream,protocol=5)
    X=np.concatenate([ds.X,nb,ds.semantic],axis=2);target=ds.views[:,:,0].astype(np.float32);del nb,docs,value;gc.collect()
    learner=s.xgb.XGBRanker(**s.old.PARAMS);began=time.monotonic();learner.fit(X.reshape(-1,94),target.ravel(),group=np.full(17427,100));fit_seconds=time.monotonic()-began;learner.save_model(directory/'ranker.json')
    chosen=np.sort(np.random.default_rng(20261021).choice(17427,200,replace=False));live=learner.predict(X[chosen].reshape(-1,94));replay=s.xgb.XGBRanker(n_jobs=1);replay.load_model(directory/'ranker.json');assert np.array_equal(live,replay.predict(X[chosen].reshape(-1,94)))
    write(OWN/'fit_rank.json',dict(status='COMPLETE',queries=17427,companies=11483,ranker=record(directory/'ranker.json'),nb=record(directory/'nb.pkl.gz'),params=s.old.PARAMS,auxiliary_nb=aux,
        features=s.old.CONTENT+s.S1+s.S2,train_input_sha256=s.bench.sha(X),target_sha256=s.bench.sha(target),nb_target_sha256=s.bench.sha(ds.targets),seconds=time.monotonic()-start,ranker_fit_seconds=fit_seconds,serialized_rank_max_abs_error=0.))
    print('FIT_RANK_COMPLETE',fit_seconds,flush=True)


def fit_risk():
    s,g0,g,t,r=libs();np=s.np;a=np.load(OWN/'pooled_oof.npz');start=time.monotonic();model=s.xgb.train(g.RISK,s.xgb.DMatrix(a['risk'],label=a['hits'][:,1],feature_names=g.RFEATURES,nthread=1),num_boost_round=200);path=OWN/'models/risk.json';model.save_model(path);replay=s.xgb.Booster();replay.load_model(path);assert np.array_equal(model.predict(s.xgb.DMatrix(a['risk'],feature_names=g.RFEATURES)),replay.predict(s.xgb.DMatrix(a['risk'],feature_names=g.RFEATURES)))
    write(OWN/'fit_risk.json',dict(status='COMPLETE',queries=17427,model=record(path),params=g.RISK,features=g.RFEATURES,source_features=record(OWN/'pooled_oof.npz'),target='Original outer-OOF R3 top1 UNION',seconds=time.monotonic()-start,serialized_risk_max_abs_error=0.))


def prepare(directory,fidelity=False,input_csv=None):
    import numpy as np
    import pyarrow as pa
    s,g0,g,t,r=libs();start=time.monotonic();times={}
    if fidelity:
        allq=read(SOURCE/'training_queries.json');origin=np.sort(np.random.default_rng(20261021).choice(17427,200,replace=False));queries=[allq[i] for i in origin];oldgeo=read(SOURCE/'geometry.json');geometry={q['query_id']:oldgeo[q['query_id']] for q in queries};write(directory/'origin.json',origin.tolist())
    else:
        queries=read(directory/'queries.json') if input_csv is None else queries_from_raw(connection().execute("SELECT * FROM read_csv(?,delim=';',all_varchar=true)",[str(input_csv)]).fetch_arrow_table().to_pylist());write(directory/'queries.json',queries)
        from scripts.run_retrieval_geo_v16_workbench import _resolve_ban
        from src.xgb_matcher.retrieval_geo_v16 import normalize_street
        from src.xgb_matcher.retrieval_query_ban_v327 import BAN
        began=time.monotonic();con=connection();surface=pa.Table.from_pylist([dict(query_id=q['query_id'],fold=-1,crm_number=q['crm_number'],crm_street_norm=normalize_street(q['crm_street']),crm_postcode=q['crm_postcode'],crm_insee=q['crm_insee']) for q in queries]);geo=_resolve_ban(con,surface,verify(BAN)).to_pylist();con.close();geometry={}
        for q,row in zip(queries,geo):row['spatial_crs_compatible']=bool(q['crm_insee'] and not q['crm_insee'].startswith(('97','98')) and row['ban_resolved'] and 0<=row['ban_x_l93']<=1300000 and 6000000<=row['ban_y_l93']<=7300000);geometry[q['query_id']]=dict(geometry=row)
        times['ban_geocoding']=time.monotonic()-began
    n=len(queries);write(directory/'queries.json',queries);write(directory/'geometry.json',geometry);plan=read(SOURCE/'plan.json');official=plan['official'];began=time.monotonic();engine=g0.G0(plan);times['generation_initialization']=time.monotonic()-began;outputs={};generation=[]
    for i,q in enumerate(queries):
        if i and i%200==0:
            del engine;gc.collect();began=time.monotonic();engine=g0.G0(plan);times['generation_initialization']+=time.monotonic()-began
        resources();began=time.monotonic();candidates,meta=engine.retrieve(q,geometry[q['query_id']]['geometry']);generation.append(time.monotonic()-began);outputs[q['query_id']]=candidates
        if i%100==0:print('GENERATE',i,n,flush=True)
    times['generation']=sum(generation);del engine;gc.collect();write(directory/'candidates.json',outputs);keys=np.array([[c['siret'] for c in outputs[q['query_id']]] for q in queries],dtype='S14');np.save(directory/'keys.npy',keys)
    began=time.monotonic();sites,units,points=s.b.hydrate(official,outputs);con=connection();con.register('requested',pa.table({'siret':sorted(sites)}))
    closures={row['siret']:row['closure_date'] for row in con.execute("""SELECT siret,MAX(try_cast(dateDebut AS DATE)) closure_date FROM read_parquet(?) h SEMI JOIN requested r USING(siret) WHERE etatAdministratifEtablissement='F' AND try_cast(changementEtatAdministratifEtablissement AS BOOLEAN)=true AND try_cast(dateDebut AS DATE)<=DATE '2026-08-01' GROUP BY siret""",[official['history']['path']]).fetch_arrow_table().to_pylist()}
    pairs=sorted({(c['siren'],q['crm_insee']) for q in queries for c in outputs[q['query_id']] if q['crm_insee']});con.register('pairs',pa.table({'siren':[p[0] for p in pairs],'commune':[p[1] for p in pairs]}));counts={(row['siren'],row['commune']):row['active_count'] for row in con.execute("""SELECT s.siren,s.codeCommuneEtablissement commune,count(*) active_count FROM read_parquet(?) s JOIN pairs r ON s.siren=r.siren AND s.codeCommuneEtablissement=r.commune WHERE s.etatAdministratifEtablissement='A' AND s.statutDiffusionEtablissement='O' GROUP BY 1,2""",[official['stock']['path']]).fetch_arrow_table().to_pylist()}
    pmap={q['query_id']:q for q in queries};old=s.b.feature_cube(pmap,outputs,geometry,sites,units,points);new=g0.v2.feature_cube(old,pmap,outputs,geometry,sites,points,closures,counts);X=np.concatenate([old,new],axis=2)[:,:,g0.CONTENT_COLS];np.save(directory/'content.npy',X);times['content85_hydration_features']=time.monotonic()-began
    began=time.monotonic();meta=directory/'metadata';meta.mkdir();s.connection=lambda ignored:connection();s.metadata(meta,queries,keys,official,np.arange(n),truth=None);addresses=np.load(meta/'addresses.npy');neighbours=np.zeros((n,100,3),dtype=np.int16);xy=np.full((n,100,2),np.nan)
    for i,q in enumerate(queries):
        for j,c in enumerate(outputs[q['query_id']]):
            point=s.b.candidate_point(points.get(c['siret'],{}))
            if sites.get(c['siret'],{}).get('statutDiffusionEtablissement')=='O' and point['precision'] in ('housenumber_equivalent','street_imprecise'):xy[i,j]=point['point_l93']
        same=(addresses[i,:,None]==addresses[i,None,:])&(addresses[i,:,None]>=0);near=np.square(xy[i,:,None]-xy[i,None,:]).sum(2)<=2500;neighbours[i]=np.column_stack([same.sum(1),near.sum(1),(same|near).sum(1)])
    wanted=sorted({q['crm_insee'] for q in queries if q['crm_insee']});con.register('communes',pa.table({'codeCommuneEtablissement':wanted}));density_counts=dict(con.execute("SELECT s.codeCommuneEtablissement,count(*) FROM read_parquet(?) s SEMI JOIN communes r USING(codeCommuneEtablissement) WHERE s.statutDiffusionEtablissement='O' GROUP BY 1",[official['stock']['path']]).fetchall());con.close();density=np.array([math.log1p(density_counts.get(q['crm_insee'],0)) if q['crm_insee'] else np.nan for q in queries],dtype=np.float32)
    np.save(directory/'neighbours.npy',neighbours);np.save(directory/'density.npy',density);np.save(directory/'xy.npy',xy);times['candidate_metadata_neighbours_density']=time.monotonic()-began
    if fidelity:
        assert np.array_equal(keys,np.load(FT/'keys.npy',mmap_mode='r')[origin]);assert np.array_equal(X,np.load(SOURCE/'features.npy',mmap_mode='r')[origin],equal_nan=True)
    write(directory/'prepare.json',dict(queries=n,seconds=time.monotonic()-start,stages_seconds=times,generation_p95_seconds=float(np.quantile(generation,.95)),keys=record(directory/'keys.npy'),features=record(directory/'content.npy'),new_labels_read=False));print('PREPARED',n,times,flush=True)


def cache_vectors(directory):
    import numpy as np
    start=time.monotonic();texts=read(directory/'metadata/texts.json');base=read(FT/'base_encode/receipt.json');binding=read(verify(base['binding']));assert binding['adapter'] is None
    original=read(verify(binding['texts']));lookup={v:i for i,v in enumerate(original)};prior=OWN/'shared_new_texts.json';newlookup=read(prior) if prior.exists() else {};missing=[];locations=[]
    for text in texts:
        if text in lookup:locations.append(['base',lookup[text]])
        elif text in newlookup:locations.append(newlookup[text])
        else:locations.append(['new',len(missing)]);missing.append(text)
    write(directory/'new_texts.json',missing);write(directory/'vector_locations.json',locations);needed={int(idx//base['shard_size']) for kind,idx in locations if kind=='base'}
    # Existing banks immutable; copy just selected vectors into owned compact file.
    vectors=np.full((len(texts),768),np.nan,dtype=np.float32);arrays={}
    for shard in needed:arrays[shard]=np.load(verify(base['shards'][shard]),mmap_mode='r')
    shared={}
    for i,(kind,idx) in enumerate(locations):
        if kind=='base':vectors[i]=arrays[int(idx//base['shard_size'])][int(idx%base['shard_size'])]
        elif kind!='new':
            if kind not in shared:shared[kind]=np.load(kind,mmap_mode='r')
            vectors[i]=shared[kind][idx]
    np.save(directory/'vectors.npy',vectors)
    write(directory/'cache.json',dict(texts=len(texts),new_texts=len(missing),reused=len(texts)-len(missing),base_shards_verified=len(needed),seconds=time.monotonic()-start,base_binding=base['binding']))
    print('VECTOR_CACHE',len(texts),len(missing),flush=True)


def encode(directory):
    import numpy as np
    import torch
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    resources(metal=True);start=time.monotonic();texts=read(directory/'new_texts.json')
    if not texts:write(directory/'encoding.json',dict(new_vectors=0,seconds=0.,load_seconds=0.,peak_bytes=0));return
    # A lock coordinates this namespace; process metadata also guards other agents.
    with (OWN/'metal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);resources(metal=True);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(20261017);torch.mps.set_per_process_memory_fraction(.12);guard=Guard(metal=True)
        config=AutoConfig.from_pretrained(H/'base',local_files_only=True);config.audio_config=None;config.vision_config=None
        model=AutoModel.from_pretrained(H/'base',config=config,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('mps').eval()
        for p in model.parameters():p.requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(H/'base',local_files_only=True);load=time.monotonic()-start;vectors=np.empty((len(texts),768),dtype=np.float32);truncated=0;began=time.monotonic()
        with torch.inference_mode():
            for at in range(0,len(texts),4):
                batch=texts[at:at+4];raw=tokenizer(batch,padding=False,truncation=False)['input_ids'];truncated+=sum(len(x)>512 for x in raw);tokens=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt');tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')};hidden=model(**tokens).last_hidden_state.float();mask=tokens['attention_mask'].unsqueeze(-1);vectors[at:at+len(batch)]=torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1).cpu().numpy()
                if at%4000==0:print('ENCODE',at,len(texts),flush=True);resources()
        torch.mps.synchronize();seconds=time.monotonic()-began;assert np.isfinite(vectors).all();np.save(directory/'new_vectors.npy',vectors);guard.done.set();write(directory/'encoding.json',dict(new_vectors=len(texts),seconds=seconds,load_seconds=load,truncated=truncated,peak_bytes=guard.peak,vectors=record(directory/'new_vectors.npy'),adapter=None))


def semantic(directory):
    import numpy as np
    s,*_=libs();start=time.monotonic();vectors=np.load(directory/'vectors.npy',mmap_mode='r+');locations=read(directory/'vector_locations.json');new=read(directory/'new_texts.json')
    if new:
        extra=np.load(directory/'new_vectors.npy',mmap_mode='r');shared=read(OWN/'shared_new_texts.json') if (OWN/'shared_new_texts.json').exists() else {}
        for i,(kind,idx) in enumerate(locations):
            if kind=='new':vectors[i]=extra[idx]
        for i,text in enumerate(new):shared[text]=[str(directory/'new_vectors.npy'),i]
        write(OWN/'shared_new_texts.json',shared)
    vectors.flush();meta=directory/'metadata';qi=np.load(meta/'qtext.npy');ci=np.load(meta/'ctext.npy');addresses=np.load(meta/'addresses.npy');cosine=np.full(ci.shape,np.nan,dtype=np.float32)
    for i in range(len(qi)):
        valid=np.flatnonzero(ci[i]>=0)
        if qi[i]>=0:cosine[i,valid]=vectors[ci[i,valid]]@vectors[qi[i]]
    out=np.stack([cosine,s.relative(cosine,addresses)],axis=2);np.save(directory/'semantic.npy',out);write(directory/'semantic.json',dict(seconds=time.monotonic()-start,features=record(directory/'semantic.npy')))


def risk_surface(X,pred,nb,sem,addresses,xy,keys,neighbours,density):
    import numpy as np
    s,g0,g,t,r=libs();n=len(X);i=np.arange(n);order=np.argsort(-pred,axis=1,kind='stable');a,z,t=order[:,:3].T;top=X[i,a];f={name:k for k,name in enumerate(g0.CONTENT)};signals=np.concatenate([nb,sem],axis=2);p=s.d.softmax(np.take_along_axis(pred,order[:,:10],axis=1))
    vals=[pred[i,a],pred[i,a]-pred[i,z],pred[i,a]-pred[i,t],-(p*np.log(np.maximum(p,1e-30))).sum(1),a+1,top[:,f['candidate_same_siren_count']]];vals.extend(neighbours[i,a,k] for k in range(3));vals.extend([top[:,f['geo_log_distance']],top[:,f['geo_missing']]])
    for k in range(9):vals.extend([signals[i,a,k],signals[i,z,k],signals[i,a,k]-signals[i,z,k]])
    vals.extend(top[:,f[name]] for name in ('name_jaro_max','number_exact','street_jaro','postcode_match','v2_is_active','v2_is_closed','v2_is_siege'));vals.append(density)
    extras=[]
    for row in range(n):
        adj=(addresses[row,:,None]==addresses[row,None,:])&(addresses[row,:,None]>=0);adj|=np.square(xy[row,:,None]-xy[row,None,:]).sum(2)<=900;np.fill_diagonal(adj,True)
        values=[X[row,:,f['name_jaro_max']],nb[row,:,1],sem[row,:,0]];extras.append([g.relative(value,adj)[1][a[row]] for value in values])
    return np.column_stack(vals+[np.array(extras)[:,k] for k in range(3)]).astype(np.float32),a,order[:,:4]


def infer(directory):
    import numpy as np
    s,g0,g,t,r=libs();start=time.monotonic();X=np.load(directory/'content.npy',mmap_mode='r');sem=np.load(directory/'semantic.npy',mmap_mode='r');keys=np.load(directory/'keys.npy');meta=directory/'metadata';codes=np.load(meta/'naf.npy');addresses=np.load(meta/'addresses.npy');queries=read(directory/'queries.json');stages={}
    began=time.monotonic()
    with gzip.open(verify(read(OWN/'fit_rank.json')['nb']),'rb') as stream:model=pickle.load(stream)
    nb=nb_score(model,[s.b.normalize_name(s.b.text(q['crm_name'])) for q in queries],codes,addresses);stages['s1_load_and_inference']=time.monotonic()-began
    began=time.monotonic();ranker=s.xgb.Booster();ranker.load_model(verify(read(OWN/'fit_rank.json')['ranker']));pred=ranker.predict(s.xgb.DMatrix(np.concatenate([X,nb,sem],axis=2).reshape(-1,94),nthread=1)).reshape(len(X),100);stages['ranker_load_and_inference']=time.monotonic()-began
    began=time.monotonic();rx,top,top4=risk_surface(X,pred,nb,sem,addresses,np.load(directory/'xy.npy'),keys,np.load(directory/'neighbours.npy'),np.load(directory/'density.npy'));model=s.xgb.Booster();model.load_model(verify(read(OWN/'fit_risk.json')['model']));score=model.predict(s.xgb.DMatrix(rx,feature_names=g.RFEATURES,nthread=1));status=t.tiers(score,read(OWN/'policy.json')['policy']);stages['risk_features_load_inference']=time.monotonic()-began
    np.savez_compressed(directory/'scored.npz',pred=pred,risk_features=rx,score=score,tier=status,top4=top4,nb=nb)
    lines=[]
    for i,q in enumerate(queries):
        proposals=[str(keys[i,j].decode()) for j in top4[i]];lines.append(dict(query_id=q['query_id'],proposed_siret=proposals[0],siren=proposals[0][:9],status=t.TIERS[int(status[i])],risk_score=float(score[i]),alternatives=proposals[1:]))
    with (directory/'proposals.jsonl').open('w') as stream:
        for row in lines:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    import csv
    with (directory/'proposals.csv').open('w') as stream:
        writer=csv.writer(stream,delimiter=';');writer.writerow(['query_id','proposed_siret','siren','status','risk_score','alternative_1','alternative_2','alternative_3'])
        for row in lines:writer.writerow([row[k] for k in ('query_id','proposed_siret','siren','status','risk_score')]+row['alternatives'])
    summary={name:dict(count=int((status==k).sum()),percent=float(100*(status==k).mean())) for k,name in enumerate(t.TIERS)};write(directory/'inference.json',dict(queries=len(X),statuses=summary,seconds=time.monotonic()-start,stages_seconds=stages,proposals=record(directory/'proposals.jsonl')));print('INFERRED',summary,flush=True)


def fidelity():
    import numpy as np
    s,g0,g,t,r=libs();directory=OWN/'fidelity_200';origin=np.array(read(directory/'origin.json'));a=np.load(directory/'scored.npz');ds=r.dataset(SRC)
    with gzip.open(verify(read(OWN/'fit_rank.json')['nb']),'rb') as stream:model=pickle.load(stream)
    nb=nb_score(model,ds.names[origin],ds.codes[origin],ds.addresses[origin]);assert np.array_equal(nb,a['nb'],equal_nan=True)
    learner=s.xgb.XGBRanker(n_jobs=1);learner.load_model(verify(read(OWN/'fit_rank.json')['ranker']));inputs=np.concatenate([ds.X[origin],nb,ds.semantic[origin]],axis=2);pred=learner.predict(inputs.reshape(-1,94)).reshape(200,100)
    expected,_,_=g.risk_features(ds,origin,pred,nb,g.context(origin,nb));model=s.xgb.Booster();model.load_model(verify(read(OWN/'fit_risk.json')['model']));score=model.predict(s.xgb.DMatrix(expected,feature_names=g.RFEATURES,nthread=1));err=float(np.max(np.abs(pred-a['pred'])));riskerr=float(np.max(np.abs(score-a['score'])));featureerr=float(np.nanmax(np.abs(expected-a['risk_features'])))
    if err or riskerr or not np.array_equal(expected,a['risk_features'],equal_nan=True):raise ValueError('Serving/evaluation full-fit fidelity')
    old=np.load(OWN/'pooled_oof.npz');meta=Path(read(FT/'data.json')['metadata']);assert np.array_equal(np.load(directory/'semantic.npy'),ds.semantic[origin],equal_nan=True);assert np.array_equal(np.load(directory/'metadata/naf.npy'),ds.codes[origin]);
    result=dict(queries=200,candidate_keys_equal=True,content85_equal=True,nb_equal=True,semantic_equal=True,risk49_equal=True,rank_max_abs_error=err,risk_max_abs_error=riskerr,risk_feature_max_abs_error=featureerr,top1_equal=int((pred.argmax(1)==a['pred'].argmax(1)).sum()),
        fullfit_vs_old_oof_rank_max_abs_difference=float(np.max(np.abs(pred-old['ranking'][origin]))),fullfit_vs_old_oof_risk_max_abs_difference=float(np.max(np.abs(score-old['score'][origin]))),note='Fidelity uses identical NEW full-fit weights, old OOF weights deliberately different; no assertion of identical OOF/full-fit predictions.')
    write(OWN/'fidelity.json',result);print('FIDELITY',result,flush=True)


def distribution(directory):
    import numpy as np
    from scipy.stats import ks_2samp
    s,g0,g,t,r=libs();a=np.load(directory/'scored.npz');X=np.load(directory/'content.npy');f={name:k for k,name in enumerate(g0.CONTENT)};top=a['top4'][:,0];pooled=np.load(OWN/'pooled_oof.npz');refX=np.load(SOURCE/'features.npy',mmap_mode='r');reftop=pooled['top'];queries=read(directory/'queries.json');rq=read(SOURCE/'training_queries.json')
    pilot={'risk_score':a['score'],'name_similarity':X[np.arange(len(X)),top,f['name_jaro_max']],'street_similarity':X[np.arange(len(X)),top,f['street_jaro']],'log_distance':X[np.arange(len(X)),top,f['geo_log_distance']],'distance_missing':X[np.arange(len(X)),top,f['geo_missing']],'number_missing':X[np.arange(len(X)),top,f['number_missing']],'candidate_count':np.full(len(X),100.),'crm_name_length':np.array([len(q['crm_name']) for q in queries])}
    base={'risk_score':pooled['score'],'name_similarity':refX[np.arange(17427),reftop,f['name_jaro_max']],'street_similarity':refX[np.arange(17427),reftop,f['street_jaro']],'log_distance':refX[np.arange(17427),reftop,f['geo_log_distance']],'distance_missing':refX[np.arange(17427),reftop,f['geo_missing']],'number_missing':refX[np.arange(17427),reftop,f['number_missing']],'candidate_count':np.full(17427,100.),'crm_name_length':np.array([len(q['crm_name']) for q in rq])}
    result={}
    for key,p in pilot.items():
        b=np.asarray(base[key],dtype=float);p=np.asarray(p,dtype=float);bf=b[np.isfinite(b)];pf=p[np.isfinite(p)];edges=np.unique(np.quantile(bf,np.arange(.1,1.,.1))) if len(bf) else np.array([]);edges=np.r_[-np.inf,edges,np.inf]
        bc=np.r_[np.histogram(bf,edges)[0],np.sum(~np.isfinite(b))].astype(float)/len(b);pc=np.r_[np.histogram(pf,edges)[0],np.sum(~np.isfinite(p))].astype(float)/len(p);bc=np.maximum(bc,1e-6);pc=np.maximum(pc,1e-6)
        result[key]=dict(psi=float(((pc-bc)*np.log(pc/bc)).sum()),ks_statistic=float(ks_2samp(bf,pf).statistic) if len(bf) and len(pf) else None,
            bank_missing=int((~np.isfinite(b)).sum()),pilot_missing=int((~np.isfinite(p)).sum()),bank_quantiles=np.quantile(bf,[0,.1,.5,.9,1]).tolist() if len(bf) else [],pilot_quantiles=np.quantile(pf,[0,.1,.5,.9,1]).tolist() if len(pf) else [])
    write(OWN/'population_shift.json',dict(metrics=result,interpretation='Descriptive shift, no occupation precision or inference from missing SIRET; bank risk/top1 OOF vs pilot full-fit, so model transport also contributes.'))


def review(directory):
    import numpy as np
    s,*_=libs();rows=read(directory/'input.json');proposals=[json.loads(line) for line in (directory/'proposals.jsonl').read_text().splitlines()];rng=np.random.default_rng(20261023);chosen=[];counts={}
    for status in ('SUR','PROBABLE','A_VERIFIER'):
        pool=[i for i,row in enumerate(proposals) if row['status']==status];n=min(20,len(pool));counts[status]=n;chosen.extend(rng.choice(pool,n,replace=False).tolist())
    chosen=rng.permutation(chosen).tolist();out=directory/'review_60';out.mkdir();keys=[proposals[i]['proposed_siret'] for i in chosen];official=read(SOURCE/'plan.json')['official'];outputs={str(i):[dict(siret=key,siren=key[:9])] for i,key in enumerate(keys)};sites,units,points=s.b.hydrate(official,outputs);cards=[]
    for k,i in enumerate(chosen):
        raw=rows[i];p=proposals[i];site=sites.get(p['proposed_siret'],{});unit=units.get(p['siren'],{});name=unit.get('denominationUniteLegale') or site.get('enseigne1Etablissement') or site.get('denominationUsuelleEtablissement') or 'Nom officiel non disponible';address=' '.join(str(site.get(col) or '') for col in ('numeroVoieEtablissement','indiceRepetitionEtablissement','typeVoieEtablissement','libelleVoieEtablissement','codePostalEtablissement','codeCommuneEtablissement'))
        crm=' — '.join([str(raw.get('client_final') or ''),' '.join(str(raw.get(col) or '') for col in ('stc_street_number','stc_street_name','stc_zipcode','stc_city'))]);buttons=''.join(f'<label><input type="radio" name="q{k}" value="{v}" onchange="save()">{v}</label> ' for v in ('Oui','Non','Je ne sais pas'))
        cards.append(f'<section><h2>Fiche {k+1}</h2><h3>CRM</h3><p>{html.escape(crm)}</p><h3>Proposition</h3><p>{html.escape(str(name))}<br>{html.escape(address)}<br>SIRET {p["proposed_siret"]}</p><p>Cette société occupe-t-elle ce site ?</p>{buttons}</section>')
    js="""const KEY='sireto-serve-v1-review60';function save(){let d={};document.querySelectorAll('input:checked').forEach(x=>d[x.name]=x.value);localStorage.setItem(KEY,JSON.stringify(d))}const saved=JSON.parse(localStorage.getItem(KEY)||'{}');document.querySelectorAll('input').forEach(x=>{if(saved[x.name]===x.value)x.checked=true});function exp(){let d=JSON.parse(localStorage.getItem(KEY)||'{}');let a=['fiche;reponse'];for(let i=0;i<document.querySelectorAll('section').length;i++)a.push((i+1)+';'+(d['q'+i]||''));let e=document.createElement('a');e.href=URL.createObjectURL(new Blob([a.join('\\n')],{type:'text/csv;charset=utf-8'}));e.download='reponses.csv';e.click()}"""
    page='<!doctype html><html lang="fr"><meta charset="utf-8"><title>Revue occupation — pilote SIRETO</title><style>body{font:17px system-ui;max-width:950px;margin:35px auto;background:#f3f5f8}section{background:white;padding:24px;margin:25px 0;border-radius:10px}label{margin-right:20px}button{padding:12px}</style><h1>Revue aveugle d’occupation</h1><p>Aucune réponse préremplie. Les statuts et scores sont cachés. Sans preuve, choisir « Je ne sais pas ».</p><button onclick="exp()">Exporter les réponses CSV</button>'+''.join(cards)+'<script>'+js+'</script></html>'
    (out/'jugement.html').write_text(page);write(out/'private_key.json',[dict(fiche=k+1,query_id=proposals[i]['query_id'],status=proposals[i]['status']) for k,i in enumerate(chosen)]);write(out/'manifest.json',dict(prepared_not_launched=True,seed=20261023,selected=len(chosen),per_status=counts,html=record(out/'jugement.html'),responses_filled=0))


def model_manifest():
    s,g0,g,t,r=libs();refs=[record(OWN/'models'/name) for name in ('ranker.json','nb.pkl.gz','risk.json')];plan=read(OWN/'plan.json')
    manifest=dict(version='serve_v1',recipe='Frozen G0 content85 + company-crossfit S1(7) + BASE S2(2), full-fit R3 and risk49',models=refs,policy=read(OWN/'policy.json')['policy'],rank_features=s.old.CONTENT+s.S1+s.S2,risk_features=g.RFEATURES,
        implementation=record(SCRIPT),protocol_commit='b47b1390',protocol_sha256=plan['protocol']['sha256'],source_refs=plan['sources'],base_files=read(FT/'plan.json')['base_files'],rank_params=s.old.PARAMS,risk_params=g.RISK,fit_queries=17427,validation_on_unlabeled_target=False)
    write(OWN/'manifest.json',manifest);return manifest


def publish(success,error=None):
    import numpy as np
    from scripts.retrieval_site_operational_metric import wilson
    budget=read(OWN/'budget.json');result=dict(status='COMPLETE_PILOT_NOT_PRECISION_CERTIFIED' if success else 'STOP_INCOMPLETE',error=error,wall_seconds=time.time()-budget['started_unix'],budget_seconds=10800,outputs_confidential_root=str(OWN/'pilot_2000'),closed_labels_used=False,closed_pools_opened=False,vm_or_8288_controlled=False,script=record(SCRIPT),protocol_commit='b47b1390')
    for filename,key in [('policy.json','bank_pooled_calibration'),('fidelity.json','fidelity'),('population_shift.json','population_shift')]:
        if (OWN/filename).exists():result[key]=read(OWN/filename)
    # Strip private source refs from the public selection summary; no query IDs.
    if (OWN/'pilot_2000/selection.json').exists():result['selection']={k:v for k,v in read(OWN/'pilot_2000/selection.json').items() if k not in ('input','queries')}
    workers=read(OWN/'workers.json') if (OWN/'workers.json').exists() else [];result['peak_rss_plus_driver_bytes']=max([w.get('peak_bytes',0) for w in workers]+[0]);result['workers']=workers
    if success:
        directory=OWN/'pilot_2000';result['pilot']=read(directory/'inference.json');result['pilot'].pop('proposals',None);result['review_60']=read(directory/'review_60/manifest.json');result['costs']=dict(preparation=read(directory/'prepare.json'),cache=read(directory/'cache.json'),encoding=read(directory/'encoding.json'),semantic=read(directory/'semantic.json'))
        for value in result['costs'].values():
            for key in ('keys','features','base_binding','vectors'):value.pop(key,None)
        result['disk_allocated_bytes']=int(subprocess.check_output(['du','-sk',str(OWN)],text=True).split()[0])*1024;result['disk_free_bytes']=shutil.disk_usage(H).free;result['manifest']=model_manifest()
    write(PUBLIC,result);text=DOC.read_text().split('\n## Résultats\n')[0];lines=['## Résultats',f"Statut **{result['status']}**, mur{result['wall_seconds']:.1f}s, pic{result['peak_rss_plus_driver_bytes']/2**30:.3f}Gio."]
    if success:
        lines.extend(['','| Statut | Pilote sans SIRET n/% | Banc référence, calibration OOF poolée % | Politique adaptative historique % |','|---|---:|---:|---:|'])
        for status in ('SUR','PROBABLE','A_VERIFIER'):
            p=result['pilot']['statuses'][status];b=result['bank_pooled_calibration']['statuses'][status];old=result['bank_pooled_calibration']['original_adaptive_reference_percent'][status];lines.append(f"| {status} | {p['count']}/{p['percent']:.2f} | {b['percent']:.2f} | {old:.2f} |")
        lines.extend(['',f"Fidélité full-fit200 : R3 max{result['fidelity']['rank_max_abs_error']}, risque max{result['fidelity']['risk_max_abs_error']}, top1{result['fidelity']['top1_equal']}/200. L’écart aux anciens poids OOF est distinct : ranking{result['fidelity']['fullfit_vs_old_oof_rank_max_abs_difference']}, risque{result['fidelity']['fullfit_vs_old_oof_risk_max_abs_difference']}.",
            f"Sélection :{result['selection']}. Aucun label fermé utilisé ; exclusion par entreprise non revendiquée. Les noms/empreintes d’exclusion ne servent pas aux features.",
            f"Coûts :{result['costs']}. Inférence :{result['pilot']['stages_seconds']}. Disque namespace{result['disk_allocated_bytes']/2**30:.3f}Gio, libre{result['disk_free_bytes']/2**30:.3f}Gio.",
            f"Revue préparée NON lancée : `{directory/'review_60/jugement.html'}`, {result['review_60']['selected']} fiches, {result['review_60']['per_status']}. Statut/score cachés, aucune réponse préremplie. Aucun navigateur ni jugement lancé.",
            '','| Variable | PSI | KS | Médiane banc / pilote |','|---|---:|---:|---|'])
        for key,value in result['population_shift']['metrics'].items():lines.append(f"| {key} | {value['psi']:.4f} | {value['ks_statistic']} | {value['bank_quantiles'][2] if value['bank_quantiles'] else None} / {value['pilot_quantiles'][2] if value['pilot_quantiles'] else None} |")
        lines+=['','Parts de statuts ≠ précision. OOF poolé calibré sur lui-même, transport vers full-fit et décalage de population : aucune garantie98,23% sur les sans-SIRET. Recommandation : faire juger les60 fiches avant automatisation ; ne pas écraser les SIRET existants.']
    else:lines.append('Blocage : '+str(error)+'. Aucun résultat partiel promu, données privées et anciens artefacts conservés.')
    DOC.write_text(text+'\n'+'\n'.join(lines)+'\n');manifest_path=ROOT/'docs/serve_v1_manifest.json'
    if success:write(manifest_path,result['manifest'])
    paths=['docs/serve_v1_pilot.md','docs/serve_v1_pilot.json']+(['docs/serve_v1_manifest.json'] if success else [])
    # Append against exact bytes under lock, commit only this study's paths.
    with (ROOT/'.git/serve_v1_publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if subprocess.check_output(['git','diff','HEAD','--','handover.md'],cwd=ROOT).strip():raise ValueError('Concurrent dirty handover: publication not committed')
        handover=ROOT/'handover.md';before=handover.read_text();line=f"\n- 2026-10-20 — ServeV1 : {result['status']}, {result['wall_seconds']:.1f}s, pic{result['peak_rss_plus_driver_bytes']/2**30:.3f}Gio ; protocole `b47b1390`, script/rapports `docs/serve_v1_pilot.{{md,json}}`, modèles privés `H/serve_v1`, manifesteSHA sans CRM. {('Pilote2000 sans SIRET, fidélité200, revue60 préparée non lancée ; aucune précision cible revendiquée.' if success else 'STOP : '+str(error))} Aucun label/pool/VM/PID8288/étranger/pending/push modifié. Commit isolé auto-référencé par `git log -1 --format=%H -- docs/serve_v1_pilot.json`.\n"
        if handover.read_text()!=before:raise ValueError('Concurrent journal edit')
        handover.write_text(before+line);paths.append('handover.md');subprocess.run(['git','add','--']+paths,cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','feat: publish frozen serve v1 and unlabeled CRM pilot','--']+paths,cwd=ROOT,check=True);write(OWN/'publication.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),success=success))
    print('PUBLICATION',result['status'],flush=True)


def execute(phase,directory=None,metal=False):
    resources(metal=metal);command=[str(H/'runtime/bin/python') if metal else str(ROOT/'.venv/bin/python'),'-u',str(SCRIPT),'_'+phase]
    if directory is not None:command+=['--directory',str(directory)]
    log=OWN/(phase+('_'+Path(directory).name if directory else '')+'.log');start=time.monotonic()
    with log.open('w') as stream:completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    worker=dict(phase=phase,seconds=time.monotonic()-start,exit_code=completed.returncode,log=str(log));receipt=OWN/('worker_'+phase+('_'+Path(directory).name if directory else '')+'.json')
    if receipt.exists():worker.update(read(receipt))
    workers=read(OWN/'workers.json') if (OWN/'workers.json').exists() else [];workers.append(worker);write(OWN/'workers.json',workers)
    if completed.returncode:raise RuntimeError('Worker '+phase+' failed: '+str(completed.returncode)+'; '+str(log))


def run():
    if OWN.exists():raise FileExistsError('Namespace already exists; no implicit overwrite/retraining')
    for path in (SCRIPT,DOC):
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit implementation/protocol before fits')
    OWN.mkdir(mode=0o700);os.umask(0o077);write(OWN/'budget.json',dict(started_unix=time.time(),seconds=10800));sources=[record(SOURCE/'plan.json'),record(SOURCE/'training_queries.json'),record(FT/'data.json'),record(SRC/'data.json'),record(SCRIPT)]
    sources += read(SRC/'plan.json')['implementations']+[record(ROOT/'scripts'/name) for name in ('step2_option_d.py','step2_ranker_v3_group.py','step2_crm_columns_tiers.py')]
    sources += [record(CRM),record(POP)]
    if sources[-2]['sha256']!=read(ROOT/'docs/production_cost_study.json')['inventory']['source']['sha256']:raise ValueError('Raw CRM changed since cost study')
    plan=read(SOURCE/'plan.json')
    for ref in list(plan['official'].values())+list(plan['layout'].values())+[plan['index_manifest'],plan['index_meta']]:verify(ref);sources.append(ref)
    for ref in read(FT/'plan.json')['base_files']:verify(ref);sources.append(ref)
    (OWN/'protocol.md').write_text(DOC.read_text())
    write(OWN/'plan.json',dict(protocol=record(OWN/'protocol.md'),implementation=record(SCRIPT),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources))
    try:
        for phase in ('select','calibrate','fit_rank','fit_risk'):execute(phase)
        directory=OWN/'fidelity_200';directory.mkdir(mode=0o700)
        for phase in ('prepare_fidelity','cache','encode','semantic','infer'):execute(phase,directory,metal=phase=='encode')
        execute('fidelity')
        directory=OWN/'pilot_2000'
        for phase in ('prepare','cache','encode','semantic','infer'):execute(phase,directory,metal=phase=='encode')
        for phase in ('distribution','review'):execute(phase,directory)
        for ref in sources:verify(ref)
        publish(True)
    except Exception as e:
        publish(False,str(e));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase');parser.add_argument('--directory',type=Path);parser.add_argument('--input',type=Path);parser.add_argument('--output-dir',type=Path);args=parser.parse_args();os.umask(0o077)
    if args.phase=='run':run();return
    if args.phase=='serve':
        if not args.input or not args.output_dir or not args.output_dir.resolve().is_relative_to(OWN/'pilot_2000'):raise ValueError('Serve requires CSV input and confidential output-dir under H/serve_v1/pilot_2000')
        args.output_dir.mkdir(mode=0o700);write(args.output_dir/'budget.json',dict(started_unix=time.time(),seconds=10800));os.environ['SIRETO_SERVE_BUDGET_PATH']=str(args.output_dir/'budget.json')
        manifest=read(OWN/'manifest.json')
        for ref in manifest['models']:verify(ref)
        if manifest['policy']!=read(OWN/'policy.json')['policy']:raise ValueError('Frozen policy mismatch')
        con=connection();rows=con.execute("SELECT stc_service_id,client_final,stc_street_number,stc_street_name,stc_zipcode,stc_city,stc_insee_commune FROM read_csv(?,delim=';',all_varchar=true)",[str(args.input)]).fetch_arrow_table().to_pylist();con.close();write(args.output_dir/'input.json',rows);write(args.output_dir/'queries.json',queries_from_raw(rows))
        for phase in ('prepare','cache','encode','semantic','infer'):execute(phase,args.output_dir,metal=phase=='encode')
        return
    resources(metal=args.phase=='_encode');plan=read(OWN/'plan.json');verify(plan['implementation']);verify(plan['protocol']);guard=Guard(metal=False);start=time.monotonic();phase=args.phase.removeprefix('_');directory=args.directory
    if directory is not None and not directory.resolve().is_relative_to(OWN):raise ValueError('Owned output namespace only')
    if phase=='select':select_pilot()
    elif phase=='calibrate':calibrate()
    elif phase=='fit_rank':fit_rank()
    elif phase=='fit_risk':fit_risk()
    elif phase=='prepare_fidelity':prepare(directory,fidelity=True)
    elif phase=='prepare':prepare(directory)
    elif phase=='cache':cache_vectors(directory)
    elif phase=='encode':encode(directory)
    elif phase=='semantic':semantic(directory)
    elif phase=='infer':infer(directory)
    elif phase=='fidelity':fidelity()
    elif phase=='distribution':distribution(directory)
    elif phase=='review':review(directory)
    else:raise ValueError('Unknown phase')
    guard.done.set();peak=max(guard.peak,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if phase=='encode' and (directory/'encoding.json').exists():peak=max(peak,read(directory/'encoding.json')['peak_bytes'])
    write(OWN/('worker_'+phase+('_'+directory.name if directory else '')+'.json'),dict(seconds=time.monotonic()-start,peak_bytes=peak));print('WORKER_DONE',phase,peak,flush=True)


if __name__=='__main__':main()
