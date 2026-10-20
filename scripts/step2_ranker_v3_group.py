"""Frozen G0 address-relative LTR, training-only noise weights, nested risk49."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import step2_risk_model_v1 as old
read,write,record,verify=old.read,old.write,old.record,old.verify
HOME=old.HOME;OWN=HOME/'step2_ranker_v3_group_20261020';SRC=old.OWN/'full';SCRIPT=Path(__file__).resolve();DOC=ROOT/'docs/step2_ranker_v3_group.md'
VARIANTS=('base','group','noise');LEVELS=(96,97,97.5,98)
STRUCT=['group_size','group_distinct_siren','group_active_count','group_closed_count','group_siege_count','group_active_fraction','group_only_active','group_active_name_match_count','group_only_active_name_match','group_only_matching_active','group_secondary_at_crm_hq_elsewhere','group_hq_has_secondary_at_crm_elsewhere','group_candidate_secondary']
SIGNALS=('name_ul','name_etab','name_enseigne','name_max','naf_div','naf_group','gemma_base')
NEW=STRUCT+['group_'+name+'_'+suffix for name in SIGNALS for suffix in ('rank','delta_best')]
EXTRA=['group_name_max_delta_best','group_naf_group_delta_best','group_gemma_base_delta_best'];RFEATURES=old.FEATURES+[n+'_top1' for n in EXTRA]
RISK=dict(old.RISK,monotone_constraints=tuple(list(old.RISK['monotone_constraints'])+[0,0,0]))


def modules():return old.modules()


def check():
    p=read(OWN/'plan.json')
    for ref in p['implementations']+[p['protocol']]:verify(ref)
    old.check(SRC);return p


def resources(directory):
    start=time.monotonic()
    while True:
        if shutil.disk_usage(HOME).free<12*1024**3:raise OSError('CATNAT_FREE_LT12GIB')
        vm,busy,free=old.vm_state()
        if not busy and not(vm and free<8*1024**3):return time.monotonic()-start
        write(directory/'waiting.json',dict(vm=vm,big_workers=busy,free_inactive=free,unix=time.time()));time.sleep(60)


class Guard:
    def __init__(self,directory):
        import psutil
        self.directory=directory;self.proc=psutil.Process();self.peak=self.proc.memory_info().rss;self.stop=threading.Event();threading.Thread(target=self.monitor,daemon=True).start()
    def monitor(self):
        while not self.stop.wait(.2):
            vm,busy,free=old.vm_state();rss=self.proc.memory_info().rss;self.peak=max(self.peak,rss);limit=(5.75 if vm else 8)*1024**3
            reason='CATNAT_FREE_LT12GIB' if shutil.disk_usage(HOME).free<12*1024**3 else 'RSS_LIMIT' if rss>=limit else 'RESOURCE_YIELD' if busy or (vm and free<8*1024**3) else None
            if reason:write(self.directory/'resource_stop.json',dict(reason=reason,peak_bytes=self.peak,limit_bytes=int(limit)));os._exit(75)


def seal():
    s=modules();paths=[SCRIPT,Path(old.__file__),Path(s.__file__),Path(s.bench.__file__),Path(s.old.__file__),Path(s.b.__file__),Path(s.d.__file__),Path(s.old.v2.__file__),Path(old.io.__file__)]
    for path in paths+[DOC]:
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit before fit: '+str(path))
    (OWN/'protocol.md').write_text(DOC.read_text());write(OWN/'plan.json',dict(implementations=[record(p) for p in paths],protocol=record(OWN/'protocol.md'),features=NEW,risk_features=RFEATURES,risk_params=RISK,rank_params=s.old.PARAMS,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),budget_seconds=14400,publication_doc=record(DOC),sources=[record(SRC/'data.json'),record(SRC/'plan.json'),record(ROOT/'docs/label_noise_screen_v1.json')]))


def relative(value,group):
    np=modules().np;finite=np.isfinite(value);best=np.max(np.where(group&finite[None,:],value[None,:],-np.inf),axis=1);rank=1+((value[None,:]>value[:,None])&group&finite[None,:]).sum(1)
    return np.where(finite,rank,np.nan),np.where(finite,value-best,np.nan)


def structural(x,address,xy,keys,semantic):
    s=modules();np=s.np;f={v:k for k,v in enumerate(s.old.CONTENT)}
    group=(address[:,None]==address[None,:])&(address[:,None]>=0);group|=np.square(xy[:,None]-xy[None,:]).sum(2)<=900;np.fill_diagonal(group,True)
    active=x[:,f['v2_is_active']]==1;closed=x[:,f['v2_is_closed']]==1;hq=x[:,f['v2_is_siege']]==1;secondary=(~hq)&(active|closed);crm=x[:,f['full_address_match']]==1
    siren=keys.astype('S9');same=siren[:,None]==siren[None,:];size=group.sum(1);ac=group@active.astype(int);cc=group@closed.astype(int);hc=group@hq.astype(int);match=active&(x[:,f['name_jaro_max']]>=.92);mc=group@match.astype(int)
    values=[size,np.array([len(set(siren[g].tolist())) for g in group]),ac,cc,hc,ac/size,active&(ac==1),mc,match&(ac==1),match&(mc==1),secondary&crm&((same&~group&hq[None,:]&~crm[None,:]).any(1)),hq&((same&~group&secondary[None,:]&crm[None,:]).any(1)),secondary]
    out=np.full((100,len(NEW)),np.nan,dtype=np.float32);out[:,:len(STRUCT)]=np.column_stack(values)
    for name,column in [('name_ul','name_ul_jaro'),('name_etab','name_etab_jaro'),('name_enseigne','v2_enseigne_jaro_max'),('name_max','name_jaro_max')]:
        rank,delta=relative(x[:,f[column]],group);out[:,NEW.index('group_'+name+'_rank')]=rank;out[:,NEW.index('group_'+name+'_delta_best')]=delta
    rank,delta=relative(semantic[:,0],group);out[:,NEW.index('group_gemma_base_rank')]=rank;out[:,NEW.index('group_gemma_base_delta_best')]=delta
    return out,np.packbits(group,axis=1)


def prepare(directory):
    s=modules();np=s.np;check();s.pinned('c8b9e405','docs/label_noise_screen_v1.json');s.pinned('2cf01765','docs/step2_risk_model_v1.json');s.pinned('c37a1a7b','docs/step2_error_families_g0.json')
    data=read(SRC/'data.json');parent=read(verify(data['source_data']));pilot=read(verify(data['source_pilot']));n=data['queries']
    if n!=17427 or data['companies']!=11483 or set(data['folds'])!={2,3,4} or set(data['groups'])&set(pilot['groups']):raise ValueError('Closed population/group drift')
    refs=data['sources']+data['static_features']+[data['source_data'],data['source_pilot']]+check()['sources']+[record(ROOT/'docs/step2_error_families_g0.json')]
    for ref in refs:verify(ref)
    ds=old.dataset(SRC);keys=np.load(old.FT/'keys.npy');ref=read(ROOT/'docs/step2_xgb_ranker_v1.json')['source_refs']['outputs'];verify(ref);refs.append(ref);outputs=read(ref['path'])['learned_native_ooc'];pk=np.array([[c['siret'].encode() for c in outputs[q]] for q in pilot['ids']],dtype='S14');keys=np.concatenate([keys,pk]);del outputs
    unique,inverse=np.unique(keys,return_inverse=True);con=s.duckdb.connect(':memory:',config={'threads':'1','memory_limit':'512MB','max_temp_directory_size':'0B'})
    official=read(s.bench.OWN/'full/data.json')['official_refs'];con.register('requested',s.pa.table({'siret':unique.astype(str),'i':np.arange(len(unique),dtype=np.int32)}));points=np.full((len(unique),2),np.nan)
    batches=con.execute('SELECT r.i,g.x,g.y,g.qualite_xy,g.epsg,g.distance_precision,s.statutDiffusionEtablissement FROM requested r LEFT JOIN read_parquet(?) g USING(siret) LEFT JOIN read_parquet(?) s USING(siret)',[official['study']['path'],official['stock']['path']]).fetch_record_batch(16384)
    usable=0
    for batch in batches:
        for row in batch.to_pylist():
            p=s.b.candidate_point(row)
            if row['statutDiffusionEtablissement']=='O' and p['precision'] in ('housenumber_equivalent','street_imprecise'):points[row['i']]=p['point_l93'];usable+=1
    con.close();points=points[inverse].reshape(*keys.shape,2);del unique,inverse;gc.collect()
    static=np.lib.format.open_memmap(OWN/'static.npy',mode='w+',dtype=np.float32,shape=(len(keys),100,len(NEW)));adj=np.lib.format.open_memmap(OWN/'adjacency.npy',mode='w+',dtype=np.uint8,shape=(len(keys),100,13))
    for i in range(len(keys)):static[i],adj[i]=structural(ds.X[i],ds.addresses[i],points[i],keys[i],ds.semantic[i])
    static.flush();adj.flush();del static,adj,points,ds;gc.collect()
    feature_seal=[record(OWN/name) for name in ('static.npy','adjacency.npy')];write(OWN/'feature_seal.json',dict(features=NEW,files=feature_seal,labels_used=False,typed_public_unique_points=usable,group_transitive=False))
    # Label-side training weights only, after the candidate-only feature seal.
    noise_doc=read(ROOT/'docs/label_noise_screen_v1.json');private=noise_doc['result']['private_records'];verify(private);refs.append(private);fort=np.zeros(n,dtype=bool);seen=set()
    with Path(private['path']).open() as stream:
        for line in stream:
            row=json.loads(line);i=row['g0_index']
            if row['query_id']!=data['ids'][i] or i in seen or row['fold']!=data['folds'][i]:raise ValueError('Noise mask join integrity')
            seen.add(i);fort[i]=row['level']=='FORT'
    if len(seen)!=n or fort.sum()!=95:raise ValueError('Expected95 FORT/all17427')
    np.save(OWN/'fort.npy',fort);families={r['query_id']:r['family'] for r in read(ROOT/'docs/step2_error_families_g0.json')['assignments']};base=np.load(verify(read(ROOT/'docs/step2_g0_bench.json')['variants']['base']['predictions']))['hits'];eligible=np.load(old.FT/'views.npy')[:,:,1].any(1)
    family=[families.get(q,'pool_union_absent_base' if not eligible[i] else 'hors_erreurs_base') for i,q in enumerate(data['ids'])]
    if family.count('colocation')!=1046 or family.count('same_siren_siege')!=267:raise ValueError('Frozen families drift')
    for cv in range(5):
        for kind in ('outer','cal','inner0','inner1','inner2'):
            rr=read(SRC/f'rank{cv}_{kind}/receipt.json');refs += [record(SRC/f'rank{cv}_{kind}/receipt.json'),rr['models'],rr['predictions']]
    payload=dict(source_data=data,queries=n,companies=data['companies'],ids=data['ids'],groups=data['groups'],folds=data['folds'],partition=data['partition'],pilot=pilot,families=family,source_refs=refs,static_refs=feature_seal+[record(OWN/'fort.npy')],fort_count=int(fort.sum()))
    write(OWN/'data.json',payload);return dict(data=record(OWN/'data.json'),feature_seal=record(OWN/'feature_seal.json'),fort_count=95,new_gemma_cache=False,new_signal_fits=0)


def context(rows,nb):
    np=modules().np;out=np.load(OWN/'static.npy',mmap_mode='r')[rows];adj=np.load(OWN/'adjacency.npy',mmap_mode='r')
    for j,i in enumerate(rows):
        group=np.unpackbits(adj[i],axis=1)[:,:100].astype(bool)
        for k,name in enumerate(('naf_div','naf_group')):
            rank,delta=relative(nb[j,:,k],group);out[j,:,NEW.index('group_'+name+'_rank')]=rank;out[j,:,NEW.index('group_'+name+'_delta_best')]=delta
    return out


def nb_inputs(ds,model,tr,te):
    s=modules();np=s.np;train=np.empty((len(tr),100,7),dtype=np.float32);lookup={x:k for k,x in enumerate(tr)};covered=np.zeros(len(tr),dtype=np.uint8)
    for ref in model['nb_training']:
        rows=np.array(ref['scored_indices']);loc=[lookup[x] for x in rows];train[loc]=old.nb_replay(s,ds,ref,rows);covered[loc]+=1
        if set(ds.groups[ref['train_indices']])&set(ds.groups[rows]):raise ValueError('NB OOF company leakage')
    if not(covered==1).all():raise ValueError('NB training coverage')
    score=model['uses'][0]['nb_scoring']
    if score['scored_indices']!=te.tolist():raise ValueError('NB scoring order')
    return train,old.nb_replay(s,ds,score,te)


def risk_features(ds,te,pred,nb,extra):
    np=modules().np;r,_,hits,top=old.scene_features(modules(),ds,te,pred,nb,SRC)
    r=np.column_stack([r,extra[np.arange(len(te)),top][:,[NEW.index(n) for n in EXTRA]]]).astype(np.float32)
    if r.shape!=(len(te),49):raise ValueError('Risk49 schema')
    return r,hits,top


def rank(directory,cv,kind,smoke=False):
    s=modules();np=s.np;ds=old.dataset(SRC);source=read(SRC/f'rank{cv}_{kind}/receipt.json');model=read(verify(source['models']))[0];tr=np.array(source['train_indices']);te=np.array(source['test_indices']);nbtr,nbte=nb_inputs(ds,model,tr,te)
    if set(ds.groups[tr])&set(ds.groups[te]):raise ValueError('Ranking group leakage')
    target=ds.views[tr,:,0].astype(np.float32)
    if s.bench.sha(target)!=model['target_sha256']:raise ValueError('Exact ranking target drift')
    digest=hashlib.sha256()
    for at in range(0,len(tr),128):
        rows=tr[at:at+128];digest.update(np.concatenate([ds.X[rows],nbtr[at:at+128],ds.semantic[rows]],axis=2).tobytes())
    if digest.hexdigest()!=model['features_sha256']:raise ValueError('Original train94 input replay')
    baseline=np.load(verify(source['predictions']));scoreX=np.concatenate([ds.X[te],nbte,ds.semantic[te]],axis=2);replay=s.xgb.XGBRanker(n_jobs=1);replay.load_model(verify(model['model']));bp=replay.predict(scoreX.reshape(-1,94)).reshape(len(te),100)
    if not np.array_equal(bp,baseline['predictions']) or s.bench.sha(scoreX)!=model['uses'][0]['feature_sha256']:raise ValueError('Baseline94 replay')
    del scoreX,replay;gc.collect()
    if smoke:
        chosen=sorted(set(ds.groups[tr]),key=lambda g:hashlib.sha256(('20261020|group-smoke|'+g).encode()).digest())[:100];use=np.isin(ds.groups[tr],chosen);tr=tr[use];nbtr=nbtr[use];target=target[use]
    etr=context(tr,nbtr);ete=context(te,nbte);fort=np.load(OWN/'fort.npy');X=np.concatenate([ds.X[tr],nbtr,ds.semantic[tr],etr],axis=2);del etr,nbtr;gc.collect();receipts={}
    for name in ('group','noise'):
        weights=np.where(fort[tr],0.,1.).astype(np.float32) if name=='noise' else np.ones(len(tr),dtype=np.float32);learner=s.xgb.XGBRanker(**s.old.PARAMS);started=time.monotonic();learner.fit(X.reshape(-1,121),target.ravel(),group=np.full(len(tr),100),sample_weight=weights);path=directory/(name+'.json');learner.save_model(path)
        receipts[name]=dict(model=record(path),params=s.old.PARAMS,features=s.old.CONTENT+s.S1+s.S2+NEW,train_indices=tr.tolist(),train_companies=len(set(ds.groups[tr])),zero_weight_count=int((weights==0).sum()),weight_sha256=s.bench.sha(weights),target_sha256=s.bench.sha(target),feature_sha256=s.bench.sha(X),fit_seconds=time.monotonic()-started)
        del learner;gc.collect()
    del X,target;gc.collect();SX=np.concatenate([ds.X[te],nbte,ds.semantic[te],ete],axis=2)
    for name in VARIANTS:
        if name=='base':pred=bp
        else:
            a=s.xgb.XGBRanker(n_jobs=1);a.load_model(verify(receipts[name]['model']));pred=a.predict(SX.reshape(-1,121)).reshape(len(te),100);b=s.xgb.XGBRanker(n_jobs=1);b.load_model(verify(receipts[name]['model']))
            if not np.array_equal(pred,b.predict(SX.reshape(-1,121)).reshape(len(te),100)):raise ValueError('Ranker serialized replay')
            del a,b
        risk,hits,top=risk_features(ds,te,pred,nbte,ete)
        if name=='base' and not np.array_equal(risk[:,:46],baseline['risk'],equal_nan=True):raise ValueError('Base risk46 drift')
        path=directory/(name+'.npz');np.savez_compressed(path,indices=te,predictions=pred,risk=risk,hits=hits,top=top);receipts.setdefault(name,{})['predictions']=record(path)
    return dict(cv=cv,kind=kind,smoke=smoke,variants=receipts,test_indices=te.tolist(),ranking_fits=2,baseline_replayed=True,nb_replayed=True,new_signal_fits=0,source=record(SRC/f'rank{cv}_{kind}/receipt.json'))


def pack(cv,kind,name):
    np=modules().np;return np.load(verify(read(OWN/f'rank{cv}_{kind}/receipt.json')['variants'][name]['predictions']))


def risk(directory,cv):
    s=modules();np=s.np;_,test,proper,cal,_=old.roles(SRC,cv);lookup={x:i for i,x in enumerate(proper)};fort=np.load(OWN/'fort.npy');receipts={}
    for name in VARIANTS:
        X=np.empty((len(proper),49),dtype=np.float32);y=np.empty(len(proper),dtype=bool);covered=np.zeros(len(proper),dtype=np.uint8)
        for k in range(3):
            p=pack(cv,f'inner{k}',name);loc=[lookup[x] for x in p['indices']];X[loc]=p['risk'];y[loc]=p['hits'][:,1];covered[loc]+=1
        if not(covered==1).all():raise ValueError('Risk proper coverage')
        cp=pack(cv,'cal',name);ep=pack(cv,'outer',name)
        if not np.array_equal(cp['indices'],cal) or not np.array_equal(ep['indices'][:len(test)],test):raise ValueError('Risk row roles')
        weights=np.where(fort[proper],0.,1.) if name=='noise' else np.ones(len(proper));model=s.xgb.train(RISK,s.xgb.DMatrix(X,label=y,weight=weights,feature_names=RFEATURES,nthread=1),num_boost_round=200);path=directory/(name+'.json');model.save_model(path);replay=s.xgb.Booster();replay.load_model(path)
        cx=s.xgb.DMatrix(cp['risk'],feature_names=RFEATURES,nthread=1);ex=s.xgb.DMatrix(ep['risk'],feature_names=RFEATURES,nthread=1);cs=model.predict(cx);es=model.predict(ex)
        if not np.array_equal(cs,replay.predict(cx)) or not np.array_equal(es,replay.predict(ex)):raise ValueError('Risk49 replay')
        choices={};masks={}
        for v in LEVELS:
            t,stats=old.threshold(cs,cp['hits'][:,1],v/100,True);choices[str(v)]=dict(threshold=t,calibration=stats);masks['sur_'+str(v)]=es>=t if t is not None else np.zeros(len(es),dtype=bool)
        np.savez_compressed(directory/(name+'.npz'),indices=ep['indices'],hits=ep['hits'],top=ep['top'],score=es,cal_score=cs,**masks)
        receipts[name]=dict(model=record(path),decisions=record(directory/(name+'.npz')),choices=choices,params=RISK,features=RFEATURES,proper=proper.tolist(),calibration=cal.tolist(),test=test.tolist(),zero_weight_count=int((weights==0).sum()),weights_sha256=s.bench.sha(weights),serialized_replay_equal=True)
    return dict(cv=cv,variants=receipts,risk_fits=3,new_signal_fits=0)


def absolute_boot(hits,groups):
    s=modules();np=s.np;_,codes=np.unique(groups,return_inverse=True);count=np.bincount(codes);sums=np.array([np.bincount(codes,weights=hits[name][:,k],minlength=len(count)) for name in VARIANTS for k in (0,1)]);rng=np.random.default_rng(old.SEED);samples=[]
    for _ in range(100):
        w=rng.multinomial(len(count),np.full(len(count),1/len(count)),size=100);samples.append(100*(w@sums.T)/(w@count)[:,None])
    a=np.concatenate(samples);return {name:{view:np.quantile(a[:,2*i+k],[.025,.975]).tolist() for k,view in enumerate(('exact','union'))} for i,name in enumerate(VARIANTS)}


def bundle(hits,mask,data):
    s=modules();np=s.np;part=np.array(data['partition']);fold=np.array(data['folds'])
    return dict(pooled=s.bench.metrics(hits,mask),by_cv={str(c):s.bench.metrics(hits[part==c],mask[part==c]) for c in range(5)},by_population_fold={str(c):s.bench.metrics(hits[fold==c],mask[fold==c]) for c in (2,3,4)})


def collect(directory):
    s=modules();np=s.np;data=read(OWN/'data.json');n=data['queries'];groups=np.array(data['groups']);part=np.array(data['partition']);hits={name:np.empty((n,2),dtype=bool) for name in VARIANTS};curves={};confirmation={};allrefs=[]
    for name in VARIANTS:
        scores=np.empty((n,100),dtype=np.float32);top=np.empty(n,dtype=np.int16);masks={str(v):np.zeros(n,dtype=bool) for v in LEVELS};pp=[];pt=[];pm={str(v):[] for v in LEVELS}
        for cv in range(5):
            p=pack(cv,'outer',name);rr=read(OWN/f'risk{cv}/receipt.json')['variants'][name];dec=np.load(verify(rr['decisions']));g=p['indices']<n;test=p['indices'][g]
            if not np.array_equal(test,np.flatnonzero(part==cv)):raise ValueError('Outer partition drift')
            scores[test]=p['predictions'][g];top[test]=p['top'][g];hits[name][test]=p['hits'][g];pp.append(p['predictions'][~g]);pt.append(p['top'][~g]);allrefs += [rr['model'],rr['decisions']]
            for v in LEVELS:masks[str(v)][test]=dec['sur_'+str(v)][g];pm[str(v)].append(dec['sur_'+str(v)][~g])
        curves[name]={k:bundle(hits[name],a,data) for k,a in masks.items()};avg=np.mean(pp,axis=0,dtype=np.float32);at=s.b.select_top(avg)[0];pv=np.load(old.FT/'p_pilot_views.npy')[:,:,[s.b.VIEWS.index('exact'),s.b.VIEWS.index('union')]];ph=pv[np.arange(526),at];agree=np.all(np.array(pt)==at[None,:],axis=0)
        confirmation[name]=dict(status='DESCRIPTIVE_NO_FIT',top1=s.bench.metrics(ph),sur={str(v):s.bench.metrics(ph,agree&np.all(pm[str(v)],axis=0)) for v in LEVELS})
        np.savez_compressed(OWN/(name+'.npz'),predictions=scores,top=top,hits=hits[name],**{'sur_'+k:a for k,a in masks.items()})
    previous=np.load(verify(read(SRC/'result.json')['predictions']))
    if not np.array_equal(hits['base'],previous['hits']) or (int(hits['base'][:,0].sum()),int(hits['base'][:,1].sum()))!=(12275,13382):raise ValueError('Baseline top1 invariant')
    family=np.array(data['families']);families={}
    for name in sorted(set(family)):
        mask=family==name;families[name]=dict(count=int(mask.sum()),top1={v:s.bench.metrics(h[mask]) for v,h in hits.items()},deltas=s.boot(hits,groups,mask))
    delta=s.boot(hits,groups);increment=s.boot({'base':hits['group'],'noise':hits['noise']},groups);winner='base'
    if delta['group']['union']['paired_company_bootstrap95_pp'][0]>0:winner='group'
    if delta['noise']['union']['paired_company_bootstrap95_pp'][0]>0 and increment['noise']['union']['paired_company_bootstrap95_pp'][0]>0:winner='noise'
    result=dict(status='COMPLETE_DEVELOPMENT_NOT_PRODUCTION',queries=n,companies=data['companies'],variants={v:dict(top1=s.bench.metrics(hits[v]),by_cv={str(c):s.bench.metrics(hits[v][part==c]) for c in range(5)},predictions=record(OWN/(v+'.npz'))) for v in VARIANTS},deltas=delta,noise_vs_group=increment['noise'],top1_cluster_bootstrap95=absolute_boot(hits,groups),families=families,sur=curves,confirmation526=confirmation,winner=winner,
        features=NEW,risk_features=RFEATURES,rank_params=s.old.PARAMS,risk_params=RISK,fort_count=95,fort_eval_count=95,bootstrap_replicates=10000,bootstrap_seed=old.SEED,source_refs=data['source_refs'],risk_model_refs=allrefs,
        labels_changed=False,pools_changed=False,closed_sets_opened=False,new_signal_fits=0,new_gemma_cache=False,production_allowed=False)
    write(OWN/'result.json',result);return dict(result=record(OWN/'result.json'),winner=winner)


def verify_all(directory):
    s=modules();np=s.np;from scripts.retrieval_site_operational_metric import wilson
    data=read(OWN/'data.json');groups=np.array(data['groups']);fort=np.load(OWN/'fort.npy');models=0;rank_vectors=0;cutoffs=0;refs=[]
    for path in sorted(OWN.glob('rank*/receipt.json')):
        r=read(path);cv=r['cv'];tr0,te0,proper,cal,_=old.roles(SRC,cv)
        for name in ('group','noise'):
            rr=r['variants'][name];tr=np.array(rr['train_indices']);te=np.array(r['test_indices']);assert not set(groups[tr])&set(np.r_[groups,np.array(data['pilot']['groups'])][te]);assert set(tr)<=set(tr0)
            if r['kind']!='outer':assert set(tr)<=set(proper) and not set(groups[tr])&set(groups[cal]) if r['kind'].startswith('inner') else set(tr)==set(proper)
            assert rr['zero_weight_count']==(int(fort[tr].sum()) if name=='noise' else 0)
            for key in ('model','predictions'):verify(rr[key]);refs.append(rr[key])
            models+=1;rank_vectors+=len(te)
    for cv in range(5):
        _,test,proper,cal,_=old.roles(SRC,cv);assert not set(groups[proper])&set(groups[cal]) and not set(groups[np.r_[proper,cal]])&set(groups[test])
        for name in VARIANTS:
            rr=read(OWN/f'risk{cv}/receipt.json')['variants'][name];cp=pack(cv,'cal',name);ep=pack(cv,'outer',name);pred=np.load(verify(rr['decisions']));model=s.xgb.Booster();model.load_model(verify(rr['model']))
            assert model.num_boosted_rounds()==200 and model.feature_names==RFEATURES and rr['zero_weight_count']==(int(fort[proper].sum()) if name=='noise' else 0)
            for matrix,expected in [(cp['risk'],pred['cal_score']),(ep['risk'],pred['score'])]:assert np.array_equal(model.predict(s.xgb.DMatrix(matrix,feature_names=RFEATURES,nthread=1)),expected)
            for level,choice in rr['choices'].items():
                best=None
                for threshold in np.unique(pred['cal_score']):
                    selected=pred['cal_score']>=threshold
                    if wilson(int(cp['hits'][selected,1].sum()),int(selected.sum()))[0]>=float(level)/100:best=float(threshold);break
                assert best==choice['threshold'];assert np.array_equal(pred['sur_'+level],pred['score']>=best if best is not None else np.zeros(len(ep['indices']),dtype=bool));cutoffs+=1
            refs += [rr['model'],rr['decisions']]
    assert models==50 and cutoffs==60 and sum(read(OWN/'result.json')['families'][k]['count'] for k in read(OWN/'result.json')['families'])==17427
    for ref in data['source_refs']+data['static_refs']:verify(ref)
    check();proof=dict(passed=True,new_rankers=models,new_risks=15,smoke_rankers=2,replayed_new_ranking_vectors=rank_vectors,independent_wilson_cutoffs=cutoffs,new_signal_fits=0,company_overlap=0,source_refs_verified=len(data['source_refs']),model_refs=refs,fort_evaluation_rows_retained=95)
    write(OWN/'verification.json',proof);return proof


def report(r):
    def delta(d):return f'{d["delta_pp"]:+.2f} '+str([round(x,2) for x in d['paired_company_bootstrap95_pp']])
    def sur(m):
        p=m['pooled']['precision']['union'];return ('indéf.' if p is None else f'{p["percent"]:.2f}')+f' / {m["pooled"]["coverage"]["percent"]:.2f}'
    lines=['**Fait** —17427 dossiers/11483 entreprises, mêmes5CV/OOF ; docs/script `step2_ranker_v3_group`, `handover.md`.',f'**Preuve** —50LTR+15risques+2smokes,60seuils indépendants/rejeux/SHA/groupes ; durée {r["elapsed_seconds"]/60:.2f}min, pic {r["peak_bytes"]/1024**3:.2f}Gio.', '', '| Top1 | Exact %, IC95 entreprise | UNION %, IC95 entreprise | Δ exact [IC95] | Δ UNION [IC95] |','|---|---|---|---|---|']
    for name in VARIANTS:
        m=r['variants'][name]['top1'];ci=r['top1_cluster_bootstrap95'][name];d=r['deltas'].get(name);cells=[f'{m[v]["percent"]:.2f} '+str([round(x,2) for x in ci[v]]) for v in ('exact','union')];lines.append(f'| {name} | '+ ' | '.join(cells)+(' | — | — |' if d is None else f' | {delta(d["exact"])} | {delta(d["union"])} |'))
    lines+=['', 'Familles historiques, ΔUNION group/noise vsbase : '+ '; '.join(f'{f} n={r["families"][f]["count"]} : '+ ' / '.join(delta(r['families'][f]['deltas'][v]['union']) for v in ('group','noise')) for f in ('colocation','same_siren_siege'))+'.', '', '| Cible Wilson calibration | base P UNION / couverture % | +groupe P/C % | +groupe+bruit P/C % |','|---|---|---|---|']
    for v in LEVELS:lines.append('| '+str(v)+'% | '+' | '.join(sur(r['sur'][name][str(v)]) for name in VARIANTS)+' |')
    lines+=['', '526 descriptifs, top1 exact/UNION : '+ '; '.join(f'{v} {r["confirmation526"][v]["top1"]["exact"]["hits"]}/{r["confirmation526"][v]["top1"]["union"]["hits"]}' for v in VARIANTS)+'.', '**Écarts / choix** —3features risque choisies a priori (nom/S1/S2),49 pour les3recettes ; bruit poids0 des95FORT en train LTR/risque seulement, NB figés.', '**Reste / risques** —Tout le banc évalué, labels/pools/jeux fermés/VM/tiers inchangés ; IC exploratoires, pas garantie98% ni promotion.', 'Recommandation unique : '+('conserver la référence S1+S2-BASE, aucun gain UNION confirmé.' if r['winner']=='base' else 'retenir '+r['winner']+' comme piste de développement, sans promotion et statut A_VERIFIER.')]
    assert len(lines)<=25;return '\n'.join(lines)+'\n'


def publish(stop=None):
    paths=['docs/step2_ranker_v3_group.json','docs/step2_ranker_v3_group.md','handover.md'];plan=read(OWN/'plan.json');verify(plan['publication_doc'])
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign/concurrent dirty publication path: '+path)
    if (ROOT/paths[0]).exists():raise ValueError('Foreign existing publication JSON')
    budget=read(OWN/'budget.json')
    if stop:r=dict(status='STOP_INCOMPLETE',reason=stop,elapsed_seconds=time.time()-budget['started_unix'],production_allowed=False);text='**Fait** —STOP_INCOMPLETE : '+stop+'\n**Preuve** —Reçus conservés, aucun résultat complet validé.\n**Écarts / choix** —Pas de partiel promu.\n**Reste / risques** —Recommandation unique : conserver S1+S2-BASE/A_VERIFIER.\n'
    else:
        r=read(OWN/'result.json');assert read(OWN/'verification.json')['passed'];r.update(elapsed_seconds=time.time()-budget['started_unix']-budget['wait_seconds'],wait_seconds=budget['wait_seconds'],peak_bytes=max(read(p)['peak_bytes'] for p in OWN.glob('*/receipt.json')),verification=record(OWN/'verification.json'),plan=record(OWN/'plan.json'));text=report(r)
    (OWN/'final_report.md').write_text(text);write(OWN/'result.json',r);write(ROOT/paths[0],r);DOC.write_text((OWN/'protocol.md').read_text()+'\n## Résultats\n\n'+text+'\nDétails par CV/fold, familles complètes, n/N, Wilson, gains/pertes et confirmation526 dans le JSON ; reçus/modèles/rejeux dans `'+str(OWN)+'`.\n')
    before=(ROOT/'handover.md').read_text();line=f'\n- 2026-10-20 — Étape2-12 reclasseur v3 groupe : {r["status"]}, durée{r["elapsed_seconds"]:.1f}s, docs `step2_ranker_v3_group`, script et reçus `{OWN}`. Mêmes17427/11483/5CV/OOF, groupe label-free27features, trois variantes,95FORT train seulement,49features risque/4ciblesWilson, labels/pools/jeuxclos/VM/étrangers/pending/push intouchés. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_ranker_v3_group.json`.\n'
    with (ROOT/'handover.md').open('a') as stream:stream.write(line)
    if (ROOT/'handover.md').read_text()!=before+line:raise ValueError('Concurrent journal edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','feat: publish grouped-address ranker v3 and noise-weight comparison','--',*paths],cwd=ROOT,check=True)
    write(OWN/'publication.json',dict(status='VERIFIED_COMMITTED',commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip()))


def execute(stage):
    directory=OWN/stage;directory.mkdir();wait=resources(directory);budget=read(OWN/'budget.json');budget['wait_seconds']+=wait;write(OWN/'budget.json',budget);check()
    with (directory/'worker.log').open('w') as log:
        child=subprocess.Popen([str(old.io.CPU),'-u',str(SCRIPT),'worker','--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if time.time()-budget['started_unix']-budget['wait_seconds']>=14280:child.terminate();child.wait();raise TimeoutError('Four-hour deadline; publication reserve120s')
            time.sleep(.5)
    if child.returncode:raise RuntimeError(f'{stage}: exit{child.returncode}, inspect {directory}/worker.log')
    print('COMPLETE '+stage,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('--stage');a=p.parse_args()
    if a.action=='run':
        OWN.mkdir(parents=True,exist_ok=True)
        with (OWN/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (OWN/'budget.json').exists():raise ValueError('Fresh namespace required')
            write(OWN/'budget.json',dict(started_unix=time.time(),wait_seconds=0.));seal()
            try:
                execute('prepare');execute('smoke');stages=[f'rank{c}_{k}' for c in range(5) for k in ('outer','cal','inner0','inner1','inner2')]
                for i,stage in enumerate(stages):
                    execute(stage)
                    if i==0:
                        elapsed=time.time()-read(OWN/'budget.json')['started_unix'];fit=read(OWN/stage/'receipt.json')['seconds'];projected=elapsed+1.5*24*fit+1200;write(OWN/'projection.json',dict(seconds=projected,first_full_stage=fit,includes_reserve1200=True))
                        if projected>14280:raise TimeoutError('Projection exceeds4h, no reduced benchmark')
                for stage in [f'risk{c}' for c in range(5)]+['collect','verify']:execute(stage)
                publish();write(OWN/'state.json',dict(status='COMPLETE',publication=read(OWN/'publication.json'),wall_seconds=time.time()-read(OWN/'budget.json')['started_unix']));print((OWN/'final_report.md').read_text(),flush=True)
            except BaseException as exc:
                write(OWN/'state.json',dict(status='STOP',reason=str(exc)))
                try:publish(str(exc))
                except BaseException as pub:write(OWN/'publication_error.json',dict(reason=str(pub)))
                raise
        return
    directory=OWN/a.stage
    if directory.parent!=OWN or not directory.exists():raise ValueError('Owned worker namespace required')
    guard=Guard(directory);started=time.monotonic();check()
    if a.stage=='prepare':value=prepare(directory)
    elif a.stage=='smoke':value=rank(directory,0,'cal',True)
    elif a.stage.startswith('rank'):
        cv,kind=a.stage[4:].split('_');value=rank(directory,int(cv),kind)
    elif a.stage.startswith('risk'):value=risk(directory,int(a.stage[4:]))
    elif a.stage=='collect':value=collect(directory)
    elif a.stage=='verify':value=verify_all(directory)
    else:raise ValueError('Unknown stage')
    value.update(seconds=time.monotonic()-started,peak_bytes=guard.peak);write(directory/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
