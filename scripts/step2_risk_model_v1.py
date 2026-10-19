"""Sealed nested-company G0 ranking and Wilson-calibrated binary risk.

BASE cache only, isolated CPU workers, descriptive external526, no promotion.
"""
import argparse
from collections import Counter
import fcntl
import gc
import gzip
import hashlib
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time

for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):os.environ[k]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import step2_gemma_finetune as io
read,write,record,verify=io.read,io.write,io.record,io.verify
HOME=io.HOME;OWN=HOME/'step2_risk_model_v1_20261020';FT=HOME/'step2_gemma_finetune_10h_20261019/full';PREV=io.PREVIOUS
SCRIPT=Path(__file__).resolve();DOC=ROOT/'docs/step2_risk_model_v1.md';SEED=20261017
SIGNALS=['naf_div_logp','naf_group_logp','coloc_div_rank','coloc_group_rank','coloc_div_delta_best','coloc_group_delta_best','naf_missing','gemma_base_cosine','coloc_gemma_delta_best']
FEATURES=['score_top1','margin12','margin13','entropy_top10','native_rank_top1','same_siren_count_top1','same_address_count_top1','near50_count_top1','same_address_or_near50_count_top1','crm_top1_log_distance','crm_top1_distance_missing']+[n+'_'+suffix for n in SIGNALS for suffix in ('top1','top2','delta12')]+['name_jaro_top1','number_exact_top1','street_jaro_top1','postcode_match_top1','active_top1','closed_top1','siege_top1','commune_log_public_count']
METHODS=('risk','margin','scene_legacy','scene_wilson')
RISK=dict(objective='binary:logistic',max_depth=3,eta=.05,min_child_weight=10,reg_lambda=10,reg_alpha=1,subsample=1.,colsample_bytree=1.,tree_method='hist',max_bin=128,seed=SEED,nthread=1,verbosity=0,base_score=.5,eval_metric='logloss',monotone_constraints=tuple(1 if n in ('margin12','margin13') else 0 for n in FEATURES))


def vm_state():
    import psutil
    found=[];busy=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            name=(p.info['name'] or '').lower();args=p.info['cmdline'] or []
            if any(x in name for x in ('qemu','vfkit','virtualization','prl_vm','vmware-vmx')):found.append(p.pid)
            if any(Path(a).name=='train3000_frozen_v142_supervisor.py' for a in args):
                busy.extend(c.pid for c in [p]+p.children(recursive=True) if c.memory_info().rss>4*1024**3)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    v=psutil.virtual_memory();return found,busy,v.free+v.inactive


def admit(directory):
    start=time.monotonic()
    while True:
        if shutil.disk_usage(HOME).free<10*1024**3:raise OSError('CATNAT_FREE_LT10GIB')
        vm,busy,free=vm_state()
        if not busy and not(vm and free<8*1024**3):return time.monotonic()-start
        write(directory/'waiting.json',dict(vm_processes=vm,big_workers=busy,free_inactive=free,unix=time.time()));time.sleep(60)


class Guard:
    def __init__(self,directory):
        import psutil
        self.directory=directory;self.proc=psutil.Process();self.peak=self.proc.memory_info().rss;self.stop=threading.Event()
        threading.Thread(target=self.monitor,daemon=True).start()
    def monitor(self):
        while not self.stop.wait(.2):
            vm,busy,free=vm_state();rss=self.proc.memory_info().rss;self.peak=max(self.peak,rss)
            limit=(5.5 if vm else 7.75)*1024**3
            reason='CATNAT_FREE_LT10GIB' if shutil.disk_usage(HOME).free<10*1024**3 else 'RSS_LIMIT' if rss>=limit else 'RESOURCE_YIELD' if busy or (vm and free<8*1024**3) else None
            if reason:
                write(self.directory/'resource_stop.json',dict(reason=reason,peak_bytes=self.peak,limit_bytes=int(limit),vm_processes=vm));os._exit(75)


def check(out):
    p=read(out/'plan.json')
    for ref in p['implementations']+[p['protocol']]:verify(ref)
    return p


def modules():
    from scripts import step2_colocated_signals as s
    s.wait_resources=lambda:None  # Our supervisor/guard owns admission; never control a VM.
    return s


def seal(out,smoke):
    s=modules();out.mkdir();(out/'protocol.md').write_text(DOC.read_text().split('\n## Résultats\n')[0])
    paths=[SCRIPT,Path(s.__file__),Path(s.bench.__file__),Path(s.old.__file__),Path(s.b.__file__),Path(s.d.__file__),Path(s.old.v2.__file__),Path(io.__file__)]
    for path in paths+[DOC]:
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit before fit: '+str(path))
    write(out/'plan.json',dict(smoke=smoke,started_unix=time.time(),features=FEATURES,risk_params=RISK,rank_params=s.old.PARAMS,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implementations=[record(p) for p in paths],protocol=record(out/'protocol.md')))


def prepare(out,directory):
    s=modules();np=s.np;p=check(out);s.pinned('9a8ecdd6','docs/step2_g0_bench.json');s.pinned('21a464d2','docs/step2_gemma_finetune_10h.json')
    parent=read(s.bench.OWN/'full/data.json');data=read(FT/'data.json');pilot=read(FT/'pilot.json');n=data['queries']
    if n!=17427 or data['budget_reduced'] or data['ids']!=parent['ids'] or data['groups']!=parent['groups'] or data['folds']!=parent['population_folds']:raise ValueError('Population/row/group custody')
    if set(data['folds'])!={2,3,4} or set(data['groups'])&set(pilot['groups']):raise ValueError('Closed fold or pilot company overlap')
    prep=read(FT/'prepare/receipt.json');refs=[record(FT/'prepare/receipt.json'),prep['data'],record(FT/'pilot.json'),record(FT/'plan.json'),record(s.bench.OWN/'full/data.json'),record(ROOT/'docs/step2_g0_bench.json'),record(ROOT/'docs/step2_error_families_g0.json')]
    bypath={r['path']:r for r in prep['artifacts']}
    needed=['keys.npy','origin.npy','views.npy','family.npy','pilot_qtext.npy','pilot_ctext.npy','p_pilot_base.npy','p_pilot_scene.npy','p_pilot_views.npy','p_naf.npy','p_addresses.npy']
    for name in needed:refs.append(bypath[str(FT/name)])
    meta=Path(data['metadata']);fs=read(meta/'feature_seal.json')
    refs+=list(fs['files'].values())+[bypath[str(meta/'naf_targets.npy')],data['source_matrix'],data['scenes']]+list(parent['official_refs'].values())
    for ref in refs:verify(ref)
    if not np.array_equal(np.load(FT/'origin.npy'),np.arange(n)):raise ValueError('Full origin order changed')
    matrix=read(s.bench.SOURCE/'matrix.json');verify(matrix['keys'])
    if not np.array_equal(np.load(FT/'keys.npy'),np.load(matrix['keys']['path'])):raise ValueError('Candidate keys changed')
    if not np.array_equal(np.load(FT/'views.npy'),np.load(verify(parent['views']))):raise ValueError('Labels changed')
    previous=read(PREV/'data.json');oldmeta=Path(previous['metadata'])
    for name in ('naf.npy','naf_targets.npy'):
        if not np.array_equal(np.load(meta/name),np.load(oldmeta/name)):raise ValueError('S1 metadata changed: '+name)
    aa=np.load(meta/'addresses.npy');zz=np.load(oldmeta/'addresses.npy')
    for a,z in zip(aa,zz):
        if not np.array_equal((a[:,None]==a[None,:])&(a[:,None]>=0),(z[:,None]==z[None,:])&(z[:,None]>=0)):raise ValueError('Colocation groups changed')
    if data['classes']!=previous['classes'] or data['names']!=previous['names']:raise ValueError('S1 names/classes changed')
    base=read(FT/'base_encode/receipt.json');binding=read(verify(base['binding']));texts=read(verify(binding['texts']));verify(binding['plan']);refs+=[record(FT/'base_encode/receipt.json'),base['binding'],binding['texts']]
    if binding['adapter'] is not None or base['status']!='BASE_COMPLETED' or base['items']!=len(texts):raise ValueError('Adapted/incomplete cache forbidden')
    oldplan=read(FT/'plan.json')
    for ref in oldplan['base_files']:verify(ref)
    refs+=oldplan['base_files']
    for name in ('encode','confirmation_encode'):
        old=read(PREV/name/'result.json');verify(old['texts']);verify(old['vectors']);refs += [record(PREV/name/'result.json'),old['texts'],old['vectors']]
    qi=np.r_[np.load(meta/'qtext.npy'),np.load(FT/'pilot_qtext.npy')];ci=np.concatenate([np.load(meta/'ctext.npy'),np.load(FT/'pilot_ctext.npy')]);addresses=np.concatenate([np.load(meta/'addresses.npy'),np.load(FT/'p_addresses.npy')])
    # Metadata text order is a prefix of the globally deduplicated BASE order.
    mt=read(verify(fs['files']['texts.json']))
    if texts[:len(mt)]!=mt or len(set(texts))!=len(texts):raise ValueError('BASE text/key binding')
    query=read(s.bench.SOURCE/'training_queries.json');pq=read(s.old.OWN/'full/pilot_queries.json')
    if [q['query_id'] for q in pq]!=pilot['ids']:raise ValueError('Pilot order')
    for i,q in enumerate(query+pq):
        expected=s.QUERY_PROMPT+s.b.text(q['crm_name']).strip() if s.b.text(q['crm_name']).strip() else None
        if (texts[qi[i]] if qi[i]>=0 else None)!=expected:raise ValueError('CRM text is not name-only')
    if ci.max()>=len(texts) or qi.max()>=len(texts):raise ValueError('Missing BASE key')
    for ref in base['shards']:verify(ref)
    refs+=base['shards'];size=base['shard_size'];cache={}
    def vectors(indices):
        result=np.empty((len(indices),768),dtype=np.float32)
        for j in np.unique(indices//size):
            if int(j) not in cache:
                if len(cache)>32:cache.clear()
                cache[int(j)]=np.load(base['shards'][int(j)]['path'],mmap_mode='r')
            mask=indices//size==j;result[mask]=cache[int(j)][indices[mask]%size]
        if not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result,axis=1),1,atol=1e-4):raise ValueError('BASE vector validity')
        return result
    cosine=np.full(ci.shape,np.nan,dtype=np.float32);began=time.monotonic()
    for i in range(len(qi)):
        v=np.flatnonzero(ci[i]>=0)
        if qi[i]>=0 and len(v):cosine[i,v]=vectors(ci[i,v])@vectors(np.array([qi[i]]))[0]
    semantic=np.stack([cosine,s.relative(cosine,addresses)],axis=2)
    base_sem=next(r for r in read(FT/'semantic/receipt.json')['artifacts'] if Path(r['path']).name=='base.npy');verify(base_sem);refs.append(base_sem)
    if not np.array_equal(semantic,np.load(base_sem['path']),equal_nan=True):raise ValueError('BASE cosine replay mismatch')
    np.save(out/'semantic.npy',semantic)
    used=np.unique(np.r_[qi,ci.ravel()]);used=used[used>=0]
    write(out/'cache.json',dict(status='ALL_BASE_KEYS_COVERED',texts=len(texts),needed_keys=len(used),missing=0,new_encoded=0,cosine_replay_equal=True,shards_verified=len(base['shards']),seconds=time.monotonic()-began,base_binding=base['binding'],vectors_adapted_read=False))
    del texts,mt,cache,semantic;gc.collect()
    # Candidate-only official coordinates and commune counts, before any risk targets.
    con=s.connection(directory);keys=np.load(FT/'keys.npy')
    output_ref=s.b.read(ROOT/'docs/step2_xgb_ranker_v1.json')['source_refs']['outputs'];verify(output_ref);refs.append(output_ref)
    outputs=read(output_ref['path'])['learned_native_ooc']
    pk=np.array([[c['siret'].encode() for c in outputs[q]] for q in pilot['ids']],dtype='S14');allkeys=np.concatenate([keys,pk]);unique,inverse=np.unique(allkeys,return_inverse=True)
    con.register('requested',s.pa.table({'siret':unique.astype(str),'i':np.arange(len(unique),dtype=np.int32)}))
    rows=con.execute('SELECT r.i,g.x,g.y,g.qualite_xy,g.epsg,g.distance_precision,s.statutDiffusionEtablissement FROM requested r LEFT JOIN read_parquet(?) g USING(siret) LEFT JOIN read_parquet(?) s USING(siret)',[parent['official_refs']['study']['path'],parent['official_refs']['stock']['path']]).fetch_record_batch(16384)
    points=np.full((len(unique),2),np.nan)
    for batch in rows:
        for row in batch.to_pylist():
            point=s.b.candidate_point(row)
            if row['statutDiffusionEtablissement']=='O' and point['precision'] in ('housenumber_equivalent','street_imprecise'):points[row['i']]=point['point_l93']
    points=points[inverse].reshape(*allkeys.shape,2);neighbours=np.zeros((*allkeys.shape,3),dtype=np.int16)
    for i in range(len(allkeys)):
        a=addresses[i];same=(a[:,None]==a[None,:])&(a[:,None]>=0);xy=points[i];delta=xy[:,None]-xy[None,:];near=np.square(delta).sum(2)<=2500
        neighbours[i]=np.column_stack([same.sum(1),near.sum(1),(same|near).sum(1)])
    cities=[q['crm_insee'] for q in query+pq];wanted=sorted({c for c in cities if c});con.register('communes',s.pa.table({'codeCommuneEtablissement':wanted}))
    counts=dict(con.execute("SELECT s.codeCommuneEtablissement,count(*) FROM read_parquet(?) s SEMI JOIN communes r USING(codeCommuneEtablissement) WHERE s.statutDiffusionEtablissement='O' GROUP BY s.codeCommuneEtablissement",[parent['official_refs']['stock']['path']]).fetchall());con.close()
    np.save(out/'neighbours.npy',neighbours);np.save(out/'density.npy',np.array([math.log1p(counts.get(c,0)) if c else np.nan for c in cities],dtype=np.float32))
    origin=np.arange(n);po=np.arange(526)
    if p['smoke']:
        chosen=sorted(set(data['groups']),key=lambda g:hashlib.sha256(('20261020|risk-smoke|'+g).encode()).digest())[:100]
        origin=np.flatnonzero(np.isin(data['groups'],chosen));po=np.array([i for i,f in enumerate(pilot['folds']) if i in [j for j,x in enumerate(pilot['folds']) if x==f][:12]])
    np.save(out/'origin.npy',origin);np.save(out/'pilot_origin.npy',po)
    payload=dict(source_data=record(FT/'data.json'),source_pilot=record(FT/'pilot.json'),sources=refs,ids=[data['ids'][i] for i in origin],groups=[data['groups'][i] for i in origin],partition=np.array(parent['partition'])[origin].tolist(),folds=np.array(data['folds'])[origin].tolist(),queries=len(origin),companies=len({data['groups'][i] for i in origin}),pilot_queries=len(po),static_features=[record(out/n) for n in ('semantic.npy','neighbours.npy','density.npy','origin.npy','pilot_origin.npy')])
    write(out/'data.json',payload);return dict(data=record(out/'data.json'),cache=record(out/'cache.json'),candidate_only_static_seal=payload['static_features'])


def dataset(out):
    s=modules();np=s.np;data=read(out/'data.json');source=read(verify(data['source_data']));pilot=read(verify(data['source_pilot']));origin=np.load(out/'origin.npy');po=np.load(out/'pilot_origin.npy');meta=Path(source['metadata']);n=len(origin)
    from types import SimpleNamespace
    pv=np.load(FT/'p_pilot_views.npy')[po][:,:,[s.b.VIEWS.index('exact'),s.b.VIEWS.index('union')]]
    ds=SimpleNamespace(X=np.concatenate([np.load(source['source_matrix']['path'],mmap_mode='r')[origin],np.load(FT/'p_pilot_base.npy')[po]]),
        views=np.concatenate([np.load(FT/'views.npy')[origin],pv]),groups=np.array(data['groups']+[pilot['groups'][i] for i in po]),ids=np.array(data['ids']+[pilot['ids'][i] for i in po]),
        names=np.array([source['names'][i] for i in origin]+[pilot['names'][i] for i in po]),classes=source['classes'],
        codes=np.concatenate([np.load(meta/'naf.npy')[origin],np.load(FT/'p_naf.npy')[po]]),targets=np.concatenate([np.load(meta/'naf_targets.npy')[origin],np.full((len(po),2),-1,dtype=np.int16)]),
        addresses=np.concatenate([np.load(meta/'addresses.npy')[origin],np.load(FT/'p_addresses.npy')[po]]),semantic=np.load(out/'semantic.npy',mmap_mode='r')[np.r_[origin,17427+po]],
        neighbours=np.load(out/'neighbours.npy',mmap_mode='r')[np.r_[origin,17427+po]],density=np.load(out/'density.npy')[np.r_[origin,17427+po]],partition=np.array(data['partition']),n=n)
    return ds


def roles(out,cv):
    s=modules();np=s.np;d=read(out/'data.json');groups=np.array(d['groups']);part=np.array(d['partition']);train=np.flatnonzero(part!=cv);test=np.flatnonzero(part==cv)
    companies=np.random.default_rng(SEED+1000+cv).permutation(np.array(sorted(set(groups[train]))));count=math.ceil(.25*len(companies));cal=train[np.isin(groups[train],companies[:count])];proper=train[~np.isin(groups[train],companies[:count])]
    inner=[(proper[a],proper[z]) for a,z in s.GroupKFold(3,shuffle=True,random_state=SEED).split(proper,groups=groups[proper])]
    if any(set(groups[a])&set(groups[z]) for a,z in [(proper,cal),(train,test)]+inner):raise ValueError('Nested role company overlap')
    return train,test,proper,cal,inner


def nb_replay(s,ds,ref,rows):
    np=s.np
    with gzip.open(verify(ref['model']),'rb') as stream:vectorizer,models=pickle.load(stream)
    values=[]
    for k,model in enumerate(models):
        p=model.predict_log_proba(vectorizer.transform(ds.names[rows]));codes=ds.codes[rows,:,k];v=np.full(codes.shape,np.nan,dtype=np.float32);a,z=np.where(codes>=0);v[a,z]=p[a,codes[a,z]];values.append(v)
    rd,dd=s.relative(values[0],ds.addresses[rows],ranks=True);rg,dg=s.relative(values[1],ds.addresses[rows],ranks=True)
    x=np.stack([values[0],values[1],rd,rg,dd,dg,(ds.codes[rows,:,0]<0).astype(np.float32)],axis=2)
    if s.bench.sha(x)!=ref['feature_sha256']:raise ValueError('Serialized S1 scoring replay')
    return x


def scene_features(s,ds,rows,pred,nb,out):
    np=s.np;i=np.arange(len(rows));order=np.argsort(-pred,axis=1,kind='stable');a,z,t=order[:,:3].T;f={n:j for j,n in enumerate(s.old.CONTENT)}
    top10=np.take_along_axis(pred,order[:,:10],axis=1);p=s.d.softmax(top10);x=ds.X[rows];top=x[i,a];other=x[i,z];signals=np.concatenate([nb,ds.semantic[rows]],axis=2)
    vals=[pred[i,a],pred[i,a]-pred[i,z],pred[i,a]-pred[i,t],-(p*np.log(np.maximum(p,1e-30))).sum(1),a+1,top[:,f['candidate_same_siren_count']]]
    vals.extend(ds.neighbours[rows,a,k] for k in range(3));vals.extend([top[:,f['geo_log_distance']],top[:,f['geo_missing']]])
    for k in range(len(SIGNALS)):vals.extend([signals[i,a,k],signals[i,z,k],signals[i,a,k]-signals[i,z,k]])
    vals.extend(top[:,f[n]] for n in ('name_jaro_max','number_exact','street_jaro','postcode_match','v2_is_active','v2_is_closed','v2_is_siege'));vals.append(ds.density[rows])
    risk=np.column_stack(vals).astype(np.float32)
    if risk.shape!=(len(rows),len(FEATURES)):raise ValueError('Risk feature schema drift')
    data=read(out/'data.json');source=read(data['source_data']);origin=np.load(out/'origin.npy');po=np.load(out/'pilot_origin.npy')
    base=np.empty((len(rows),100,46),dtype=np.float32);g=rows<ds.n
    base[g]=np.load(source['scenes']['path'],mmap_mode='r')[origin[rows[g]]]
    base[~g]=np.load(FT/'p_pilot_scene.npy',mmap_mode='r')[po[rows[~g]-ds.n]]
    return risk,s.d.scene(pred,base).astype(np.float32),ds.views[rows,a],a


def rank(out,directory,cv,kind):
    s=modules();np=s.np;ds=dataset(out);train,test,proper,cal,inner=roles(out,cv)
    if kind=='outer':tr=train;te=np.r_[test,np.arange(ds.n,len(ds.ids))]
    elif kind=='cal':tr=proper;te=cal
    else:tr,te=inner[int(kind[-1])]
    class Runner(s.Runner):
        def __init__(self):
            self.ds,self.name,self.directory=ds,'s1_s2',directory/'models';self.directory.mkdir();self.models={};self.receipts=[]
    runner=Runner();pred=runner.score(tr,te,f'{kind}_cv{cv}');model=runner.receipts[0];nb=nb_replay(s,ds,model['uses'][0]['nb_scoring'],te)
    risk,scene,hits,top=scene_features(s,ds,te,pred,nb,out)
    np.savez_compressed(directory/'predictions.npz',indices=te,predictions=pred,risk=risk,scene=scene,hits=hits,top=top)
    write(directory/'models.json',runner.receipts)
    return dict(predictions=record(directory/'predictions.npz'),models=record(directory/'models.json'),train_indices=tr.tolist(),test_indices=te.tolist(),cv=cv,kind=kind,rank_vectors_replayed=len(te),nb_replay_equal=True,fit_seconds=model['seconds'])


def threshold(values,y,target=.98,wilson=True):
    s=modules();np=s.np;order=np.argsort(-values,kind='stable');v=values[order];correct=np.cumsum(y[order].astype(int));counts=np.arange(1,len(v)+1);ends=np.r_[v[:-1]!=v[1:],True]
    if wilson:
        z=1.959963984540054;p=correct/counts;den=1+z*z/counts;lower=(p+z*z/(2*counts)-z*np.sqrt(p*(1-p)/counts+z*z/(4*counts*counts)))/den
    else:lower=correct/counts
    valid=np.flatnonzero(ends&(lower>=target))
    if not len(valid):return None,dict(accepted=0,precision=None,wilson95=None,target=target,rule='wilson_lower' if wilson else 'empirical')
    at=valid[-1];n=int(counts[at]);k=int(correct[at]);return float(v[at]),dict(accepted=n,precision=k/n,wilson95=s.b.wilson(k,n),target=target,rule='wilson_lower' if wilson else 'empirical')


def risk(out,directory,cv):
    s=modules();np=s.np;from sklearn.isotonic import IsotonicRegression
    train,test,proper,cal,inner=roles(out,cv);groups=np.array(read(out/'data.json')['groups']);n=len(groups)
    properX=np.empty((len(proper),len(FEATURES)),dtype=np.float32);properS=np.empty((len(proper),20),dtype=np.float32);properY=np.empty(len(proper),dtype=bool);lookup={v:i for i,v in enumerate(proper)}
    for k in range(3):
        a=np.load(verify(read(out/f'rank{cv}_inner{k}/receipt.json')['predictions']));loc=[lookup[v] for v in a['indices']];properX[loc]=a['risk'];properS[loc]=a['scene'];properY[loc]=a['hits'][:,1]
    cp=np.load(verify(read(out/f'rank{cv}_cal/receipt.json')['predictions']));ep=np.load(verify(read(out/f'rank{cv}_outer/receipt.json')['predictions']))
    if not np.array_equal(cp['indices'],cal) or not np.array_equal(ep['indices'][:len(test)],test):raise ValueError('Risk role row binding')
    model=s.xgb.train(RISK,s.xgb.DMatrix(properX,label=properY,feature_names=FEATURES),num_boost_round=200);path=directory/'risk.json';model.save_model(path)
    rc=s.xgb.Booster();rc.load_model(path);cx=s.xgb.DMatrix(cp['risk'],feature_names=FEATURES);ex=s.xgb.DMatrix(ep['risk'],feature_names=FEATURES);cprob=model.predict(cx);eprob=model.predict(ex)
    if not np.array_equal(eprob,rc.predict(ex)) or not np.array_equal(cprob,rc.predict(cx)):raise ValueError('Risk serialized replay')
    companies=np.random.default_rng(SEED+2000+cv).permutation(np.array(sorted(set(groups[proper]))));isocompanies=companies[:math.ceil(.2*len(companies))];iso_mask=np.isin(groups[proper],isocompanies);fit=~iso_mask
    classifier=s.d.logistic(properY[fit]);classifier.fit(properS[fit],properY[fit]);iso=IsotonicRegression(out_of_bounds='clip',y_min=0.,y_max=1.);iso.fit(s.d.probability(classifier,properS[iso_mask]),properY[iso_mask])
    path2=directory/'scene.pkl'
    with path2.open('wb') as stream:pickle.dump((classifier,iso),stream,protocol=5)
    with path2.open('rb') as stream:rr,ri=pickle.load(stream)
    cscene=iso.predict(s.d.probability(classifier,cp['scene']));escene=iso.predict(s.d.probability(classifier,ep['scene']))
    if not np.array_equal(escene,ri.predict(s.d.probability(rr,ep['scene']))):raise ValueError('Scene serialized replay')
    cs={'risk':cprob,'margin':cp['risk'][:,1],'scene_legacy':cscene,'scene_wilson':cscene};es={'risk':eprob,'margin':ep['risk'][:,1],'scene_legacy':escene,'scene_wilson':escene};thresholds={};selected={}
    for method in METHODS:
        thresholds[method]={}
        for target in (.97,.98,.99):
            t,stats=threshold(cs[method],cp['hits'][:,1],target,method!='scene_legacy');key=str(round(target*100));thresholds[method][key]=dict(threshold=t,calibration=stats)
            selected[method+'_'+key]=es[method]>=t if t is not None else np.zeros(len(eprob),dtype=bool)
    np.savez_compressed(directory/'decisions.npz',indices=ep['indices'],hits=ep['hits'],top=ep['top'],**{m+'_score':v for m,v in es.items()},**selected)
    return dict(model=record(path),scene_model=record(path2),decisions=record(directory/'decisions.npz'),cv=cv,thresholds=thresholds,
        roles=dict(proper=proper.tolist(),calibration=cal.tolist(),test=test.tolist(),logistic=proper[fit].tolist(),isotonic=proper[iso_mask].tolist()),
        company_overlap=0,serialized_replay_equal=True,params=RISK,features=FEATURES,rank_test_refit_uses_all_outer_training=True)


def paired_sur(hits,masks,groups):
    s=modules();np=s.np;_,codes=np.unique(groups,return_inverse=True);den=np.bincount(codes);rng=np.random.default_rng(SEED);names=[n for n in METHODS if n!='risk'];columns=[]
    for name in names:
        columns.extend([np.bincount(codes,weights=masks['risk'].astype(int)-masks[name].astype(int)),np.bincount(codes,weights=(masks['risk']&~hits[:,1]).astype(int)-(masks[name]&~hits[:,1]).astype(int))])
    dif=np.array(columns);samples=[]
    for _ in range(100):
        w=rng.multinomial(len(den),np.full(len(den),1/len(den)),size=100);samples.append(100*(w@dif.T)/(w@den)[:,None])
    samples=np.concatenate(samples);return {name:{metric:dict(delta_pp=float(100*dif[2*k+j].sum()/den.sum()),paired_company_bootstrap95_pp=np.quantile(samples[:,2*k+j],[.025,.975]).tolist()) for j,metric in enumerate(('coverage','false_sur_per_query'))} for k,name in enumerate(names)}


def collect(out,directory):
    s=modules();np=s.np;data=read(out/'data.json');n=data['queries'];m=data['pilot_queries'];part=np.array(data['partition']);hits=np.empty((n,2),dtype=bool);pred=np.empty((n,100),dtype=np.float32);tops=np.empty(n,dtype=int);masks={method:np.zeros(n,dtype=bool) for method in METHODS};curves={method:{key:np.zeros(n,dtype=bool) for key in ('97','98','99')} for method in METHODS};receipts=[];pilotpred=[];pilottop=[];pilotmasks={method:[] for method in METHODS}
    for cv in range(5):
        r=read(out/f'risk{cv}/receipt.json');a=np.load(verify(r['decisions']));outer=np.load(verify(read(out/f'rank{cv}_outer/receipt.json')['predictions']));test=np.flatnonzero(part==cv);count=len(test)
        hits[test]=a['hits'][:count];pred[test]=outer['predictions'][:count];tops[test]=a['top'][:count]
        for method in METHODS:
            masks[method][test]=a[method+'_98'][:count]
            for key in curves[method]:curves[method][key][test]=a[method+'_'+key][:count]
            pilotmasks[method].append(a[method+'_98'][count:])
        pilotpred.append(outer['predictions'][count:]);pilottop.append(a['top'][count:]);receipts.append(r)
    origin=np.load(out/'origin.npy');base=np.load(verify(read(ROOT/'docs/step2_g0_bench.json')['variants']['base']['predictions']))['hits'][origin];s1=np.load(verify(read(PREV/'full_s1/result.json')['predictions']))['hits'][origin]
    metrics={method:dict(pooled=s.bench.metrics(hits,mask),by_cv={str(cv):s.bench.metrics(hits[part==cv],mask[part==cv]) for cv in range(5)},by_population_fold={str(f):s.bench.metrics(hits[np.array(data['folds'])==f],mask[np.array(data['folds'])==f]) for f in (2,3,4)}) for method,mask in masks.items()}
    eligible=[m for m in METHODS if metrics[m]['pooled']['precision']['union'] is not None and metrics[m]['pooled']['precision']['union']['percent']>=98]
    best=max(eligible,key=lambda m:metrics[m]['pooled']['count']) if eligible else max(METHODS,key=lambda m:(metrics[m]['pooled']['precision']['union']['wilson95_percent'][0] if metrics[m]['pooled']['precision']['union'] else -1,metrics[m]['pooled']['count']))
    avg=np.mean(pilotpred,axis=0,dtype=np.float32);ptop=s.b.select_top(avg)[0];po=np.load(out/'pilot_origin.npy');pv=np.load(FT/'p_pilot_views.npy')[po][:,:,[s.b.VIEWS.index('exact'),s.b.VIEWS.index('union')]];ph=pv[np.arange(m),ptop];unanimous=np.all(np.array(pilottop)==ptop[None,:],axis=0)
    pm={method:unanimous&np.all(pilotmasks[method],axis=0) for method in METHODS};confirmation=dict(status='DESCRIPTIVE_CONSUMED_NO_FIT',queries=m,top1=s.bench.metrics(ph),unanimous_top1=int(unanimous.sum()),policies={method:s.bench.metrics(ph,mask) for method,mask in pm.items()},by_model={str(cv):{method:s.bench.metrics(pv[np.arange(m),pilottop[cv]],pilotmasks[method][cv]) for method in METHODS} for cv in range(5)})
    np.savez_compressed(out/'predictions.npz',predictions=pred,hits=hits,top=tops,**masks)
    value=dict(status='MEASURED_AUDIT_PENDING',queries=n,companies=data['companies'],top1={name:s.bench.metrics(h) for name,h in [('base',base),('s1',s1),('s1_s2_base',hits)]},
        deltas_vs_base=s.boot({'base':base,'s1':s1,'s1_s2_base':hits},np.array(data['groups'])),deltas_vs_s1=s.boot({'base':s1,'s1_s2_base':hits},np.array(data['groups'])),
        sur=metrics,curves={method:{key:s.bench.metrics(hits,mask) for key,mask in entries.items()} for method,entries in curves.items()},paired_sur=paired_sur(hits,masks,np.array(data['groups'])),
        confirmation=confirmation,best_reporting_method=best,risk_receipts=receipts,predictions=record(out/'predictions.npz'),cache=read(out/'cache.json'),features=FEATURES,risk_params=RISK,rank_params=s.old.PARAMS,
        labels_changed=False,pools_changed=False,closed_sets_opened=False,production_allowed=False,bootstrap_replicates=10000,bootstrap_seed=SEED,
        nested_ranking_excludes_risk_test_and_calibration=True,calibration_transport='R3 proper -> R3 refit all outer training; held-out measurement only')
    write(out/'result.json',value);return dict(snapshot=record(out/'result.json'))


def audit(out,directory):
    s=modules();np=s.np;r=read(out/'result.json');data=read(out/'data.json');pack=np.load(verify(r['predictions']));method=r['best_reporting_method'];bad=np.flatnonzero(pack[method]&~pack['hits'][:,1]);origin=np.load(out/'origin.npy');allq=s.b.index(read(s.bench.SOURCE/'training_queries.json'));labels=read(s.bench.SOURCE/'label_side.json');geo=read(s.bench.SOURCE/'geometry.json');keys=np.load(FT/'keys.npy',mmap_mode='r')[origin];views=np.load(FT/'views.npy',mmap_mode='r')[origin]
    historical={x['query_id']:x['family'] for x in read(ROOT/'docs/step2_error_families_g0.json')['assignments']};wanted=sorted({keys[i,pack['top'][i]].decode() for i in bad}|{labels[data['ids'][i]]['human_ground_truth_siret'] for i in bad});refs=read(s.bench.OWN/'full/data.json')['official_refs'];records=[]
    sites,units,points={},{},{}
    if wanted:
        con=s.connection(directory);con.register('requested',s.pa.table({'siret':wanted}));con.register('parents',s.pa.table({'siren':sorted({x[:9] for x in wanted})}))
        def fetch(key,cols,table,ident):return s.b.index(con.execute('SELECT '+','.join('s.'+c for c in cols)+f' FROM read_parquet(?) s SEMI JOIN {table} r USING({ident})',[refs[key]['path']]).fetch_arrow_table().to_pylist(),ident)
        sites=fetch('stock',s.b.STOCK_COLUMNS,'requested','siret');units=fetch('legal_units',s.b.UL_COLUMNS,'parents','siren');points=fetch('study',['siret','x','y','qualite_xy','epsg','distance_precision'],'requested','siret');con.close()
    generic=set('MAIRIE COMMUNE ECOLE COLLEGE LYCEE AGENCE CENTRE SERVICE SYNDICAT ASSOCIATION HOTEL'.split());words=generic|set('DE DU DES LA LE LES L D ET EN A AU AUX SAINT SAINTE MUNICIPAL MUNICIPALE PUBLIC PUBLIQUE PRIMAIRE MATERNELLE NATIONALE SAS SARL SA'.split());families=['same_siren_siege','same_siren_state','same_siren_near','same_siren_other','colocation','homonym_elsewhere','label_far','no_number','generic_name','other']
    def entity(key):
        site=sites.get(key,{});unit=units.get(key[:9],{});site=site if site.get('statutDiffusionEtablissement')=='O' else {};unit=unit if unit.get('statutDiffusionUniteLegale')=='O' else {}
        mapping=dict(enseigne1=site.get('enseigne1Etablissement'),enseigne2=site.get('enseigne2Etablissement'),enseigne3=site.get('enseigne3Etablissement'),denomination=site.get('denominationUsuelleEtablissement'),sigle_ul=unit.get('sigleUniteLegale'),denomination_ul=unit.get('denominationUniteLegale'),denomination_usuelle_ul=unit.get('denominationUsuelle1UniteLegale'),nom_ul=unit.get('nomUniteLegale'),prenom_usuel_ul=unit.get('prenomUsuelUniteLegale'),cj_ul=unit.get('categorieJuridiqueUniteLegale'))
        siege=str(site.get('etablissementSiege')).lower();return site,[n.text for n in s.b.build_candidate_names({k:s.b.text(v) for k,v in mapping.items()})],True if siege in ('true','1') else False if siege in ('false','0') else None,s.b.candidate_point(points.get(key))
    for i in bad:
        qid=data['ids'][i];q=allq[qid];ck=keys[i,pack['top'][i]].decode();lk=labels[qid]['human_ground_truth_siret'];a,an,asiege,ap=entity(ck);z,zn,zsiege,zp=entity(lk);same=ck[:9]==lk[:9];cp=s.b.crm_point(geo[qid]['geometry']);cl=s.b.distance(ap,zp);ql=s.b.distance(cp,zp);typed_cl=s.b.threshold(ap,zp) is not None;typed_ql=s.b.threshold(cp,zp) is not None
        fake=dict(crm_number=a.get('numeroVoieEtablissement'),crm_suffix=a.get('indiceRepetitionEtablissement'),crm_street=' '.join(str(a.get(k) or '') for k in ('typeVoieEtablissement','libelleVoieEtablissement')),crm_insee=a.get('codeCommuneEtablissement'))
        coloc=bool(a and z and s.b.address_match(fake,z) and not a.get('dernierNumeroVoieEtablissement') and not a.get('indiceRepetitionDernierNumeroVoieEtablissement'));name=s.b.normalize_name(s.b.text(q['crm_name']));nc=max([s.b.jaro(name,n) for n in an],default=0.);nl=max([s.b.jaro(name,n) for n in zn],default=0.);other_place=bool((a.get('codeCommuneEtablissement') and z.get('codeCommuneEtablissement') and a['codeCommuneEtablissement']!=z['codeCommuneEtablissement']) or (typed_cl and cl is not None and cl>500));tokens=set(re.findall(r'[A-Z]+',s.b.normalize(s.b.text(q['crm_name']))))
        flags=[same and asiege is True and zsiege is False,same and {a.get('etatAdministratifEtablissement'),z.get('etatAdministratifEtablissement')}=={'A','F'},same and typed_cl and cl is not None and cl<=500,same,not same and coloc,not same and nc>=.92 and nl>=.92 and other_place,typed_ql and ql is not None and ql>500,s.b.number(q['crm_number'],q['crm_suffix']) is None,bool(tokens&generic) and tokens<=words,True]
        family=families[flags.index(True)] if views[i,:,1].any() else 'pool_union_absent'
        records.append(dict(query_id=qid,cv=data['partition'][i],population_fold=data['folds'][i],chosen_siret=ck,human_siret=lk,family=family,historical_family=historical.get(qid,'hors_erreurs_base'),flags=dict(zip(families,map(bool,flags))),crm=q,chosen_site={k:str(v) if v is not None else None for k,v in a.items()},human_site={k:str(v) if v is not None else None for k,v in z.items()},chosen_names=an,human_names=zn,chosen_label_distance_m=cl,label_crm_distance_m=ql,labels_corrected=False))
    result=dict(method=method,false_sur=len(records),families=dict(Counter(x['family'] for x in records)),historical_families=dict(Counter(x['historical_family'] for x in records)),records=records,labels_corrected=0)
    write(directory/'errors.json',result);r['false_sur']=result;r['status']='COMPLETE_DEVELOPMENT_NOT_PRODUCTION';write(out/'result.json',r)
    return dict(errors=record(directory/'errors.json'),false_sur=len(records))


def verify_all(out,directory):
    s=modules();np=s.np;data=read(out/'data.json');groups=np.array(data['groups']);models=nb=rank_vectors=0;refs=[]
    for p in sorted(out.glob('rank*/receipt.json')):
        r=read(p);verify(r['predictions']);refs.extend([r['predictions'],r['models']]);records=read(verify(r['models']))
        assert len(records)==1
        for model in records:
            verify(model['model']);refs.append(model['model']);models+=1
            for use in model['uses']:
                assert use['replay_equal'];rank_vectors+=len(use['scored_indices'])
            for rec in model['nb_training']+[u['nb_scoring'] for u in model['uses']]:
                verify(rec['model']);refs.append(rec['model']);nb+=1
                # Pilot rows have groups distinct from every admissible training company.
                source=read(data['source_pilot']);po=np.load(out/'pilot_origin.npy');gg=np.r_[groups,np.array(source['groups'])[po]]
                assert not set(gg[rec['train_indices']])&set(gg[rec['scored_indices']]) and rec['serialized_replay_equal']
    if models!=25:raise ValueError('Expected25 isolated ranking fits')
    for cv in range(5):
        tr,te,proper,cal,inner=roles(out,cv);r=read(out/f'risk{cv}/receipt.json')
        assert r['serialized_replay_equal']
        for key in ('model','scene_model','decisions'):verify(r[key]);refs.append(r[key])
        for kind in ('cal','inner0','inner1','inner2'):
            rr=read(out/f'rank{cv}_{kind}/receipt.json');assert not set(groups[rr['train_indices']])&set(groups[te]);assert set(rr['train_indices'])<=set(proper)
    for ref in data['sources']+data['static_features']:verify(ref)
    check(out);proof=dict(passed=True,ranking_models=models,nb_pipelines=nb,ranking_vectors_replayed=rank_vectors,risk_models=5,scene_models=5,nested_company_overlap=0,model_refs=refs,sources_verified=len(data['sources']),features=FEATURES,labels_changed=False,closed_sets_opened=False,cache_new_encoded=0)
    write(out/'verification.json',proof);return dict(verification=record(out/'verification.json'))


def report(r):
    def metric(x):
        if x is None:return 'indéfinie'
        a,z=x['wilson95_percent'];return f'{x["hits"]}/{x["queries"]}={x["percent"]:.2f}% [{a:.2f};{z:.2f}]'
    lines=['## Fait',f'Statut {r["status"]}, aucune promotion ; `scripts/step2_risk_model_v1.py`, docs `step2_risk_model_v1`, `handover.md`.',f'Durée réelle calcul {r["compute_seconds"]/3600:.3f}h ; mur {r["wall_seconds"]/3600:.3f}h ; pic {r["peak_bytes"]/1024**3:.2f}Gio.','## Preuve']
    if 'top1' not in r:return '\n'.join(lines+[f'Arrêt : {r.get("reason")}. Aucun résultat complet top1/SUR/526 validé.','## Écarts / choix','Aucun résultat partiel présenté comme complet.','## Reste / risques','Recommandation unique : conserver les modèles antérieurs sans promotion.'])+'\n'
    lines += [f'{r["queries"]} dossiers/{r["companies"]} entreprises ;5CV entreprise ; IC95 bootstrap10000 entreprises.', '| Top1 | Exact % | UNION % | Δ UNION vs base [IC95] | Δ UNION vs S1 [IC95] |','|---|---:|---:|---|---|']
    def delta(x):return f'{x["delta_pp"]:+.2f} '+str([round(v,2) for v in x['paired_company_bootstrap95_pp']])
    for name,v in r['top1'].items():
        d=r['deltas_vs_base'].get(name);z=r['deltas_vs_s1'].get(name);lines.append(f'| {name} | {v["exact"]["percent"]:.2f} | {v["union"]["percent"]:.2f} | {delta(d["union"]) if d else "—"} | {delta(z["union"]) if z else "—"} |')
    lines += ['| SUR | Précision UNION, Wilson95 | Précision exacte % | Couverture % |','|---|---|---:|---:|']
    for name in METHODS:
        x=r['sur'][name]['pooled'];ex=x['precision']['exact'];lines.append(f'| {name} | {metric(x["precision"]["union"])} | {ex["percent"] if ex else "indéf."} | {x["coverage"]["percent"]:.2f} |')
    dd=r['deltas_vs_base']['s1_s2_base']['exact'];ss=r['deltas_vs_s1']['s1_s2_base']['exact'];lines += [f'Δ exact S1S2 vs base {delta(dd)} ; vs S1 {delta(ss)}.', 'Couverture@98% (seuil calibration Wilson, pas choix test) : '+', '.join(f'{m}={r["curves"][m]["98"]["coverage"]["percent"]:.2f}%' for m in ('risk','margin','scene_wilson'))+'.', f'Faux SUR meilleur descriptif {r["best_reporting_method"]} : {r["false_sur"]["false_sur"]}, familles {r["false_sur"]["families"]}.']
    c=r['confirmation'];lines += [f'526 descriptifs : exact {metric(c["top1"]["exact"])} ; UNION {metric(c["top1"]["union"])}.', '526 SUR risque : '+metric(c['policies']['risk']['precision']['union'])+f', couverture {c["policies"]["risk"]["coverage"]["percent"]:.2f}% ; consensus5modèles, aucun fit sur526.', f'Cache BASE : {r["cache"]["needed_keys"]} clés couvertes, {r["cache"]["new_encoded"]} nouveau texte ; modèles sérialisés/rejoués, rôles et SHA vérifiés.', '## Écarts / choix','Empilement réellement imbriqué hors entreprises test/calibration ; rang natif G0 (V142 sur526), densité proxy compte public, BASE sans adapter.', '## Reste / risques','Calibration Wilson adaptative et corrélations entreprises ne certifient pas98% ; transport proper→refit explicite,526 consommés, aucune promotion.', 'Recommandation unique : '+r['recommendation']]
    if len(lines)>35:raise ValueError('Report over35 lines')
    return '\n'.join(lines)+'\n'


def publish(out):
    p=check(out);r=read(out/'result.json');budget=read(OWN/'budget.json');r.update(compute_seconds=budget['compute_seconds'],wall_seconds=time.time()-budget['started_unix'],wait_seconds=budget['wait_seconds'],peak_bytes=max([read(x).get('peak_bytes',0) for x in OWN.glob('*/*/receipt.json')]+[0]),plan=record(out/'plan.json'))
    if 'top1' in r:
        r['verification']=record(out/'verification.json');x=r['sur']['risk']['pooled']['precision']['union'];r['recommendation']='Poursuivre un audit indépendant autorisé du risque, sans déploiement.' if x and x['wilson95_percent'][0]>=98 else 'Ne pas promouvoir ce risque sans borne de précision test suffisante.'
    text=report(r);write(out/'result.json',r);(out/'final_report.md').write_text(text)
    paths=['docs/step2_risk_model_v1.json','docs/step2_risk_model_v1.md','handover.md']
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign dirty publication path: '+path)
    if 'top1' in r:
        for ref in read(out/'data.json')['sources']:verify(ref)
    before=(ROOT/'handover.md').read_text();write(ROOT/paths[0],r);DOC.write_text((out/'protocol.md').read_text()+'\n## Résultats\n\n'+text+'\nDétails par CV/fold, courbes97/98/99, IC simultanés, modèles/masques, calibration et faux SUR exhaustifs : JSON et namespace externe.\n')
    line=f'\n- 2026-10-20 — Risque étape2-10 : {r["status"]}, calcul{r["compute_seconds"]:.1f}s, rapport `{out}/final_report.md`, SHA/modèles/rôles imbriqués et résultats `docs/step2_risk_model_v1.json`. S2-BASE seulement, aucun label/pool/jeu fermé/VM/web/push/étranger modifié. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_risk_model_v1.json`.\n'
    with (ROOT/'handover.md').open('a') as stream:stream.write(line)
    if (ROOT/'handover.md').read_text()!=before+line:raise ValueError('Concurrent journal edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','feat: publish nested G0 BASE risk model and Wilson policies','--',*paths],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip();write(out/'publication.json',dict(status='VERIFIED_COMMITTED',commit=commit))


def execute(out,stage):
    directory=out/stage;directory.mkdir()
    for attempt in range(13):
        work=directory/f'attempt{attempt}';work.mkdir();waiting=admit(work);b=read(OWN/'budget.json');b['wait_seconds']+=waiting;b['compute_seconds']=time.time()-b['started_unix']-b['wait_seconds'];write(OWN/'budget.json',b);check(out)
        with (directory/'worker.log').open('a') as log:
            child=subprocess.Popen([str(io.CPU),'-u',str(SCRIPT),'worker','--output-dir',str(out),'--stage',stage,'--directory',str(work)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while child.poll() is None:
                if time.time()-b['started_unix']-b['wait_seconds']>=35900:
                    child.terminate();child.wait();raise TimeoutError('Total10h deadline')
                time.sleep(.5)
            b['compute_seconds']=time.time()-b['started_unix']-b['wait_seconds'];b['history'].append(dict(output=str(out),stage=stage,attempt=attempt,seconds=time.monotonic()-start,exit_code=child.returncode));write(OWN/'budget.json',b)
        if not child.returncode:
            print('COMPLETE '+out.name+'/'+stage,flush=True);return
        stop=read(work/'resource_stop.json') if (work/'resource_stop.json').exists() else {}
        if child.returncode!=75 or stop.get('reason')!='RESOURCE_YIELD':break
        # Preserve every failed owned attempt; only resource yields are retried.
    raise RuntimeError(f'{stage} exit{child.returncode}; {directory}/worker.log')


def projection(out):
    s=modules();data=read(out/'data.json');n=data['queries'];ranks=list((OWN/'mini').glob('rank*/receipt.json'));slope=max(read(x)['fit_seconds']/len(read(x)['train_indices']) for x in ranks)
    for x in PREV.glob('full_s1_cv*/result.json'):
        for model in read(x)['models']:slope=max(slope,model['seconds']/len(model['train_indices']))
    total=0
    for cv in range(5):
        tr,te,proper,cal,inner=roles(out,cv);total+=len(tr)+len(proper)+sum(len(a) for a,z in inner)
    cost=1.5*slope*total+max(600.,read(OWN/'budget.json')['compute_seconds']);b=read(OWN/'budget.json');value=dict(status='ADMITTED' if cost+b['compute_seconds']<35900 else 'STOP_PROJECTION_GT10H',queries=n,cpu_seconds_per_train_row=slope,train_rows_across25fits=total,remaining_projected_seconds=cost,total_projected_seconds=cost+b['compute_seconds'],new_encoding_seconds=0,cache=read(out/'cache.json'),budget_reduced=False,projection_is_not_duration=True)
    write(out/'cost_plan.json',value)
    if value['status']!='ADMITTED':raise TimeoutError('Projection exceeds10h; cached scope full but no admissible projected run')


def pipeline():
    if (OWN/'budget.json').exists():raise ValueError('Fresh namespace required; do not duplicate a run')
    OWN.mkdir(parents=True,exist_ok=True);write(OWN/'budget.json',dict(started_unix=time.time(),compute_seconds=0.,wait_seconds=0.,history=[]));current=OWN/'mini'
    try:
        for smoke in (True,False):
            current=OWN/('mini' if smoke else 'full');seal(current,smoke);execute(current,'prepare')
            if not smoke:projection(current)
            for cv in range(5):
                for kind in ('outer','cal','inner0','inner1','inner2'):execute(current,f'rank{cv}_{kind}')
                execute(current,f'risk{cv}')
            for stage in ('collect','audit','verify'):execute(current,stage)
            if smoke:
                write(current/'smoke_passed.json',dict(passed=True,verification=record(current/'verification.json'),models_replayed=True));print('REAL_SMOKE_PASSED',flush=True)
        budget=read(OWN/'budget.json');budget['compute_seconds']=time.time()-budget['started_unix']-budget['wait_seconds'];write(OWN/'budget.json',budget);publish(current)
        budget['compute_seconds']=time.time()-budget['started_unix']-budget['wait_seconds'];write(OWN/'budget.json',budget)
        if budget['compute_seconds']>36000:raise TimeoutError('Publication exceeded10h total budget')
        write(OWN/'state.json',dict(status='COMPLETE',publication=read(current/'publication.json'),budget=budget));print('COMPLETE '+str(current/'final_report.md'),flush=True)
    except BaseException as exc:
        write(OWN/'state.json',dict(status='STOP',phase=current.name,reason=str(exc)))
        if current.name=='mini':
            target=OWN/'full'
            if not target.exists():seal(target,False)
            current=target
        write(current/'result.json',dict(status='STOP_INCOMPLETE',reason=str(exc),labels_changed=False,pools_changed=False,production_allowed=False))
        try:publish(current)
        except BaseException as pub:write(OWN/'publication_error.json',dict(reason=str(pub)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('--output-dir',type=Path);p.add_argument('--stage');p.add_argument('--directory',type=Path);a=p.parse_args()
    if a.action=='run':
        OWN.mkdir(parents=True,exist_ok=True)
        with (OWN/'supervisor.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pipeline()
        return
    out=a.output_dir
    if out not in (OWN/'mini',OWN/'full'):raise ValueError('Owned worker namespace required')
    directory=a.directory
    if directory.parent!=out/a.stage or not directory.is_relative_to(OWN):raise ValueError('Owned attempt namespace required')
    guard=Guard(directory);began=time.monotonic();check(out)
    if a.stage=='prepare':value=prepare(out,directory)
    elif a.stage.startswith('rank'):
        cv,kind=a.stage[4:].split('_',1);value=rank(out,directory,int(cv),kind)
    elif a.stage.startswith('risk'):value=risk(out,directory,int(a.stage[4:]))
    elif a.stage=='collect':value=collect(out,directory)
    elif a.stage=='audit':value=audit(out,directory)
    elif a.stage=='verify':value=verify_all(out,directory)
    else:raise ValueError('Unknown stage')
    value.update(seconds=time.monotonic()-began,peak_bytes=guard.peak);write(out/a.stage/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
