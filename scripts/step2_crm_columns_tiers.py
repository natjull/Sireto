"""Nested-company experiment: query-only CRM columns and three confidence tiers."""
import argparse
from collections import Counter,defaultdict
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',RAYON_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import step2_ranker_v3_group as g
from scripts import crm_structure_audit as a
r=g.old;read,write,record,verify=g.read,g.write,g.record,g.verify
HOME=g.HOME;OWN=HOME/'step2_crm_columns_tiers_20261020';SRC=g.SRC;SCRIPT=Path(__file__).resolve();DOC=ROOT/'docs/step2_crm_columns_tiers.md'
VARIANTS=('base','rank_crm','risk_crm');LEVELS=g.LEVELS;TIERS=('A_VERIFIER','PROBABLE','SUR');COLS=('stc_categorie_produit','stc_state');MISSING='__MISSING__';OTHER='__OTHER__'


def modules():return g.modules()


def check():
    p=read(OWN/'plan.json')
    for ref in p['implementations']+[p['protocol']]:verify(ref)
    g.check();return p


def elapsed():return time.time()-read(OWN/'budget.json')['started_unix']


def admit(directory):
    start=time.monotonic()
    while True:
        if elapsed()>=10680:raise TimeoutError('Three-hour wall deadline including admission')
        if g.shutil.disk_usage(HOME).free<12*1024**3:raise OSError('CATNAT_FREE_LT12GIB')
        vm,busy,free=r.vm_state()
        if not busy and not(vm and free<8*1024**3):return time.monotonic()-start
        write(directory/'waiting.json',dict(vm=vm,big_workers=busy,free_inactive=free,unix=time.time()));time.sleep(60)


def seal():
    s=modules();paths=[SCRIPT,Path(g.__file__),Path(r.__file__),Path(a.__file__),Path(s.__file__),Path(s.bench.__file__),Path(s.old.__file__),Path(s.b.__file__),Path(s.d.__file__),Path(s.old.v2.__file__),Path(r.io.__file__)]
    for path in paths+[DOC]:
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit before fit: '+str(path))
    (OWN/'protocol.md').write_text(DOC.read_text());write(OWN/'plan.json',dict(implementations=[record(p) for p in paths],protocol=record(OWN/'protocol.md'),publication_doc=record(DOC),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),budget_seconds=10800,rank_params=s.old.PARAMS,risk_params=g.RISK,reference_risk_features=g.RFEATURES,variants=VARIANTS))


def association(values,labels):
    np=modules().np;uv=sorted(set(values));ul=sorted(set(labels));table=np.array([[sum(v==x and y==z for v,y in zip(values,labels)) for z in ul] for x in uv],dtype=float)
    n=table.sum();expected=table.sum(1)[:,None]*table.sum(0)[None,:]/n if n else table;chi=float(np.sum((table-expected)**2/np.where(expected>0,expected,1)))
    scale=min(len(uv)-1,len(ul)-1);v=float(np.sqrt(chi/n/scale)) if scale and n else 0.
    return dict(cramer_v=v,modal_fold_purity=float(table.max(1).sum()/n) if n else None,table={x:{str(z):int(table[i,j]) for j,z in enumerate(ul)} for i,x in enumerate(uv)})


def inspect_columns():
    s=modules();np=s.np;data=read(SRC/'data.json');pilot=read(r.FT/'pilot.json');ids=data['ids']+pilot['ids'];cohort=read(a.OPTION/'cohort.json');refs=[record(a.CRM),record(a.POP/'queries.parquet'),record(a.POP/'manifest.json'),record(a.OPTION/'cohort.json'),record(ROOT/'docs/crm_structure_audit.json')]
    if refs[0]['sha256']!=read(a.R/'datasets/crm_new_data_20260817/location_triage_v1/manifest.json')['sources']['input']['sha256']:raise ValueError('CRM source changed')
    if refs[1]['sha256']!=read(a.POP/'manifest.json')['outputs']['queries.parquet']:raise ValueError('Query metadata source changed')
    queries=a.pq.read_table(a.POP/'queries.parquet',columns=['query_id','crm_record_id','data_origin','oof_fold'],filters=[('oof_fold','in',[2,3,4]),('query_id','in',ids)],use_threads=False).to_pylist();meta={q['query_id']:q for q in queries}
    if len(meta)!=len(ids) or set(meta)!=set(ids) or len(ids)!=17953 or set(data['folds'])!={2,3,4} or set(data['groups'])&set(pilot['groups']):raise ValueError('Cohort custody/closed fold')
    con=a.connection();rows=con.execute("SELECT stc_service_id,trim(coalesce(stc_categorie_produit,'')),trim(coalesce(stc_state,'')) FROM read_csv(?,delim=';',all_varchar=true)",[str(a.CRM)]).fetchall();mapping=defaultdict(set)
    for service,cat,state in rows:
        if cat in (MISSING,OTHER) or state in (MISSING,OTHER):raise ValueError('Reserved encoding token in source')
        mapping[service].add((cat,state))
    if any(len(v)!=1 for v in mapping.values()):raise ValueError('Ambiguous service attributes; no label-based resolution')
    crm=[];origins=[]
    for i,qid in enumerate(ids):
        q=meta[qid];origins.append(q['data_origin']);expected=data['folds'][i] if i<17427 else pilot['folds'][i-17427]
        if q['oof_fold']!=expected:raise ValueError('Population fold metadata drift')
        if q['data_origin']=='REAL_CRM_20260817':
            if q['crm_record_id'] not in mapping:raise ValueError('Missing recent service join')
            crm.append(list(next(iter(mapping[q['crm_record_id']]))))
        elif q['data_origin']=='REAL_CRM_HISTORICAL':crm.append(['',''])
        else:raise ValueError('Unexpected query origin')
    raw={};g0={};recent=np.array(origins[:17427])=='REAL_CRM_20260817';views=np.load(r.FT/'views.npy');eligible=views.any(1)
    for k,col in enumerate(COLS):
        result=con.execute(f"SELECT trim(coalesce({col},'')),count(*),sum(CASE WHEN trim(coalesce(order_siret,''))<>'' THEN 1 ELSE 0 END) FROM read_csv(?,delim=';',all_varchar=true) GROUP BY 1 ORDER BY 2 DESC",[str(a.CRM)]).fetchall()
        # Only aggregated presence, never a raw SIRET value or row-level label flag.
        counts=np.array([[n-present,present] for _,n,present in result],dtype=float);exp=counts.sum(1)[:,None]*counts.sum(0)[None,:]/counts.sum();chi=float(np.sum((counts-exp)**2/np.where(exp>0,exp,1)))
        raw[col]=dict(modalities={v or MISSING:dict(rows=n,label_present=present,percent=100*present/n) for v,n,present in result},presence_cramer_v=float(np.sqrt(chi/counts.sum())))
        vals=np.array([v[k] or MISSING for v in crm[:17427]]);g0[col]=dict(cv=association(vals.tolist(),data['partition']),population_fold=association(vals.tolist(),data['folds']),recent_cv=association(vals[recent].tolist(),np.array(data['partition'])[recent].tolist()),recent_population_fold=association(vals[recent].tolist(),np.array(data['folds'])[recent].tolist()),modalities={v:dict(queries=int((vals==v).sum()),human_label_present_percent=100.,pool_exact=s.b.metric(int(eligible[vals==v,0].sum()),int((vals==v).sum())),pool_union=s.b.metric(int(eligible[vals==v,1].sum()),int((vals==v).sum()))) for v in sorted(set(vals))})
        for name in ('cv','population_fold','recent_cv','recent_population_fold'):
            test=g0[col][name]
            if test['cramer_v']>=.5 or any(sum(c.values())>=.01*(int(recent.sum()) if name.startswith('recent') else 17427) and max(c.values())==sum(c.values()) for c in test['table'].values()):raise ValueError('Potential deterministic/high fold proxy: '+col+'/'+name)
    con.close();assert Counter(origins[:17427])=={'REAL_CRM_20260817':9897,'REAL_CRM_HISTORICAL':7530}
    audit=dict(passed=True,raw=raw,g0=g0,origins=dict(Counter(origins[:17427])),all_g0_labels_present_by_selection=True,raw_label_values_read=False,temporal_availability_verified=False,temporal_limitation='No timestamps: input availability assumed per brief, not historical no-leak certification',join_ambiguous_services=0,source_refs=refs)
    return crm,audit,data,pilot


def prepare(directory):
    s=modules();check();s.pinned('390fd5e1','docs/crm_structure_audit.json');s.pinned('b90c4bc9','docs/step2_ranker_v3_group.json');crm,audit,data,pilot=inspect_columns();refs=data['sources']+data['static_features']+[data['source_data'],data['source_pilot']]+audit['source_refs']+[record(g.OWN/'plan.json'),record(g.OWN/'feature_seal.json'),record(g.OWN/'static.npy'),record(g.OWN/'adjacency.npy')]
    for cv in range(5):
        for kind in ('outer','cal','inner0','inner1','inner2'):
            source=SRC/f'rank{cv}_{kind}/receipt.json';rr=read(source);br=g.OWN/f'rank{cv}_{kind}/receipt.json';refs += [record(source),rr['models'],rr['predictions'],record(br),read(br)['variants']['base']['predictions']]
        rr=g.OWN/f'risk{cv}/receipt.json';refs += [record(rr),read(rr)['variants']['base']['model'],read(rr)['variants']['base']['decisions']]
    for ref in refs:verify(ref)
    payload=dict(queries=17427,companies=11483,ids=data['ids'],groups=data['groups'],folds=data['folds'],partition=data['partition'],pilot=pilot,crm=crm,source_refs=refs)
    write(OWN/'data.json',payload);write(OWN/'anti_leakage.json',audit);return dict(data=record(OWN/'data.json'),anti_leakage=record(OWN/'anti_leakage.json'),sources_verified=len(refs))


def schema(rows):
    values=read(OWN/'data.json')['crm'];counts=Counter(values[i][0] for i in rows if values[i][0]);total=sum(counts.values());cat=sorted(v for v,n in counts.items() if n>=.01*total);states=sorted({values[i][1] for i in rows if values[i][1]});categories=cat+[OTHER,MISSING];states+= [OTHER,MISSING]
    names=['crm_category_'+v for v in categories]+['crm_state_'+v for v in states]+['crm_state_closed_'+v for v in states]+['crm_category_siege_'+v for v in categories]
    return dict(categories=categories,states=states,features=names,training_rows=list(map(int,rows)),observed_product_denominator=total,product_counts=dict(counts),minimum_fraction=.01)


def columns(ds,rows,enc,top=None):
    np=modules().np;raw=read(OWN/'data.json')['crm'];cats=enc['categories'];states=enc['states'];values=[[MISSING if not raw[i][k] else raw[i][k] if raw[i][k] in vocab else OTHER for k,vocab in enumerate((cats,states))] for i in rows];q=np.array([[float(v[0]==x) for x in cats]+[float(v[1]==x) for x in states] for v in values],dtype=np.float32);f={v:k for k,v in enumerate(modules().old.CONTENT)}
    if top is None:
        closed=ds.X[rows,:,f['v2_is_closed']];hq=ds.X[rows,:,f['v2_is_siege']];return np.concatenate([np.broadcast_to(q[:,None,:],(len(rows),100,q.shape[1])),q[:,None,len(cats):]*closed[:,:,None],q[:,None,:len(cats)]*hq[:,:,None]],axis=2)
    return np.column_stack([q,q[:,len(cats):]*ds.X[rows,top,f['v2_is_closed']][:,None],q[:,:len(cats)]*ds.X[rows,top,f['v2_is_siege']][:,None]]).astype(np.float32)


def rank(directory,cv,kind,smoke=False):
    s=modules();np=s.np;ds=r.dataset(SRC);source=read(SRC/f'rank{cv}_{kind}/receipt.json');model=read(verify(source['models']))[0];tr=np.array(source['train_indices']);te=np.array(source['test_indices']);nbtr,nbte=g.nb_inputs(ds,model,tr,te);y=ds.views[tr,:,0].astype(np.float32)
    if set(ds.groups[tr])&set(ds.groups[te]) or s.bench.sha(y)!=model['target_sha256']:raise ValueError('Group/target custody')
    digest=hashlib.sha256()
    for at in range(0,len(tr),128):digest.update(np.concatenate([ds.X[tr[at:at+128]],nbtr[at:at+128],ds.semantic[tr[at:at+128]]],axis=2).tobytes())
    if digest.hexdigest()!=model['features_sha256']:raise ValueError('Baseline train94 replay')
    bx=np.concatenate([ds.X[te],nbte,ds.semantic[te]],axis=2);base=s.xgb.XGBRanker(n_jobs=1);base.load_model(verify(model['model']));bp=base.predict(bx.reshape(-1,94)).reshape(len(te),100)
    if not np.array_equal(bp,np.load(verify(source['predictions']))['predictions']) or s.bench.sha(bx)!=model['uses'][0]['feature_sha256']:raise ValueError('Baseline score94 replay')
    del bx,base,bp;gc.collect()
    if smoke:
        chosen=sorted(set(ds.groups[tr]),key=lambda x:hashlib.sha256(('20261020|crm-columns-smoke|'+x).encode()).digest())[:100];keep=np.isin(ds.groups[tr],chosen);tr=tr[keep];nbtr=nbtr[keep];y=y[keep]
    enc=schema(tr);names=s.old.CONTENT+s.S1+s.S2+enc['features'];X=np.concatenate([ds.X[tr],nbtr,ds.semantic[tr],columns(ds,tr,enc)],axis=2);learner=s.xgb.XGBRanker(**s.old.PARAMS);start=time.monotonic();learner.fit(X.reshape(-1,len(names)),y.ravel(),group=np.full(len(tr),100));seconds=time.monotonic()-start;path=directory/'ranker.json';learner.save_model(path);train_sha=s.bench.sha(X);target_sha=s.bench.sha(y);del X,nbtr,y;gc.collect()
    SX=np.concatenate([ds.X[te],nbte,ds.semantic[te],columns(ds,te,enc)],axis=2);live=learner.predict(SX.reshape(-1,len(names))).reshape(len(te),100);replay=s.xgb.XGBRanker(n_jobs=1);replay.load_model(path);pred=replay.predict(SX.reshape(-1,len(names))).reshape(len(te),100)
    if not np.array_equal(live,pred) or not np.isfinite(pred).all():raise ValueError('Serialized ranking replay')
    extra=g.context(te,nbte);risk,hits,top=g.risk_features(ds,te,pred,nbte,extra);np.savez_compressed(directory/'predictions.npz',indices=te,predictions=pred,risk=risk,hits=hits,top=top)
    return dict(cv=cv,kind=kind,smoke=smoke,model=record(path),predictions=record(directory/'predictions.npz'),train_indices=tr.tolist(),test_indices=te.tolist(),schema=enc,features=names,params=s.old.PARAMS,features_sha256=train_sha,target_sha256=target_sha,fit_seconds=seconds,baseline_replayed=True,serialized_replay_equal=True)


def pack(cv,kind,name):
    if name=='rank_crm':return modules().np.load(verify(read(OWN/f'rank{cv}_{kind}/receipt.json')['predictions']))
    return g.pack(cv,kind,'base')


def choose(values,y,target=.975,probable=False):
    s=modules();np=s.np
    if not len(values):return None
    order=np.argsort(-values,kind='stable');v=values[order];k=np.cumsum(y[order].astype(int));n=np.arange(1,len(v)+1);p=k/n;z=1.959963984540054;lo=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n);ends=np.r_[v[:-1]!=v[1:],True];good=ends&(lo>=(.88 if probable else target))
    if probable:good&=p>=.90
    at=np.flatnonzero(good);return float(v[at[-1]]) if len(at) else None


def tier_policy(score,hits):
    np=modules().np;sur=choose(score,hits[:,1]);remaining=score<sur if sur is not None else np.ones(len(score),dtype=bool);prob=choose(score[remaining],hits[remaining,1],probable=True);return dict(sur=sur,probable=prob)


def tiers(score,policy):
    np=modules().np;out=np.zeros(len(score),dtype=np.int8)
    if policy['probable'] is not None:out[score>=policy['probable']]=1
    if policy['sur'] is not None:out[score>=policy['sur']]=2
    return out


def risk(directory,cv,smoke=False):
    s=modules();np=s.np;ds=r.dataset(SRC);_,test,proper,cal,_=r.roles(SRC,cv);lookup={x:i for i,x in enumerate(proper)};refs={};names=('risk_crm',) if smoke else VARIANTS
    for name in names:
        X=np.empty((len(proper),49),dtype=np.float32);y=np.empty(len(proper),dtype=bool);top=np.empty(len(proper),dtype=int);covered=np.zeros(len(proper),dtype=np.uint8)
        for k in range(3):
            p=pack(cv,f'inner{k}',name);loc=[lookup[x] for x in p['indices']];X[loc]=p['risk'];y[loc]=p['hits'][:,1];top[loc]=p['top'];covered[loc]+=1
        if not(covered==1).all():raise ValueError('Proper OOF coverage')
        cp=pack(cv,'cal',name);ep=pack(cv,'outer',name);tr=proper.copy()
        if smoke:
            selected=sorted(set(ds.groups[tr]),key=lambda v:hashlib.sha256(('20261020|crm-columns-smoke|'+v).encode()).digest())[:100];use=np.isin(ds.groups[tr],selected);tr=tr[use];X=X[use];y=y[use];top=top[use]
        enc=schema(tr) if name=='risk_crm' else None;features=g.RFEATURES+(enc['features'] if enc else []);cx=cp['risk'];ex=ep['risk']
        if enc:X=np.column_stack([X,columns(ds,tr,enc,top)]);cx=np.column_stack([cx,columns(ds,cp['indices'],enc,cp['top'])]);ex=np.column_stack([ex,columns(ds,ep['indices'],enc,ep['top'])])
        if name=='base':
            original=read(g.OWN/f'risk{cv}/receipt.json')['variants']['base'];model=s.xgb.Booster();model.load_model(verify(original['model']));path=Path(original['model']['path']);params=g.RISK;oldpred=np.load(verify(original['decisions']))
        else:
            params=dict(g.RISK,monotone_constraints=tuple(g.RISK['monotone_constraints'])+tuple(0 for _ in features[49:]));model=s.xgb.train(params,s.xgb.DMatrix(X,label=y,feature_names=features,nthread=1),num_boost_round=200);path=directory/(name+'.json');model.save_model(path)
        cs=model.predict(s.xgb.DMatrix(cx,feature_names=features,nthread=1));es=model.predict(s.xgb.DMatrix(ex,feature_names=features,nthread=1));replay=s.xgb.Booster();replay.load_model(path)
        if not np.array_equal(cs,replay.predict(s.xgb.DMatrix(cx,feature_names=features,nthread=1))) or not np.array_equal(es,replay.predict(s.xgb.DMatrix(ex,feature_names=features,nthread=1))):raise ValueError('Risk replay')
        if name=='base' and (not np.array_equal(cs,oldpred['cal_score']) or not np.array_equal(es,oldpred['score'])):raise ValueError('Reference risk49 drift')
        choices={str(v):choose(cs,cp['hits'][:,1],v/100) for v in LEVELS};policy=tier_policy(cs,cp['hits']);ct=tiers(cs,policy);et=tiers(es,policy);deciles=[float(np.quantile(cs,1-d/10,method='higher')) for d in range(1,11)];arrays={'sur_'+str(v):es>=choices[str(v)] if choices[str(v)] is not None else np.zeros(len(es),dtype=bool) for v in LEVELS}
        np.savez_compressed(directory/(name+'.npz'),indices=ep['indices'],score=es,cal_score=cs,hits=ep['hits'],top=ep['top'],tier=et,cal_tier=ct,**arrays)
        refs[name]=dict(model=record(path),decisions=record(directory/(name+'.npz')),features=features,params=params,schema=enc,proper=tr.tolist(),calibration=cal.tolist(),test=test.tolist(),choices=choices,policy=policy,decile_thresholds=deciles,cal_top1_union=int(cp['hits'][:,1].sum()),cal_statuses={TIERS[j]:s.bench.metrics(cp['hits'],ct==j) for j in range(3)},train_target_sha256=s.bench.sha(y),new_model=name!='base',serialized_replay_equal=True)
    chosen=None
    if not smoke:
        chosen='base';score=lambda v:(refs[v]['cal_statuses']['SUR']['count'],refs[v]['cal_statuses']['SUR']['count']+refs[v]['cal_statuses']['PROBABLE']['count'])
        for name in ('risk_crm','rank_crm'):
            if refs[name]['cal_top1_union']>=refs['base']['cal_top1_union'] and score(name)>score(chosen):chosen=name
    return dict(cv=cv,smoke=smoke,variants=refs,chosen=chosen,new_risks=1 if smoke else 2)


def bundle(hits,mask,data):return g.bundle(hits,mask,data)


def absolute_boot(hits,groups):
    s=modules();np=s.np;_,code=np.unique(groups,return_inverse=True);count=np.bincount(code);sums=np.array([np.bincount(code,weights=hits[v][:,k],minlength=len(count)) for v in VARIANTS for k in (0,1)]);rng=np.random.default_rng(r.SEED);samples=[]
    for _ in range(100):
        w=rng.multinomial(len(count),np.full(len(count),1/len(count)),size=100);samples.append(100*(w@sums.T)/(w@count)[:,None])
    b=np.concatenate(samples);return {v:{k:np.quantile(b[:,2*i+j],[.025,.975]).tolist() for j,k in enumerate(('exact','union'))} for i,v in enumerate(VARIANTS)}


def describe(hits,status,data):
    np=modules().np;cats=np.array([q[0] or MISSING for q in data['crm'][:17427]])
    return dict(statuses={t:bundle(hits,status==j,data) for j,t in enumerate(TIERS)},by_product={cat:{t:modules().bench.metrics(hits[cats==cat],status[cats==cat]==j) for j,t in enumerate(TIERS)} for cat in sorted(set(cats))})


def proposals(path,ids,keys,top4,score,status):
    import json
    with path.open('x') as stream:
        for i,qid in enumerate(ids):stream.write(json.dumps(dict(query_id=qid,risk_score=float(score[i]),status=TIERS[int(status[i])],proposals=[str(keys[i,j].decode()) for j in top4[i]]),ensure_ascii=False)+'\n')
    return record(path)


def collect(directory):
    s=modules();np=s.np;data=read(OWN/'data.json');n=17427;part=np.array(data['partition']);keys=np.load(r.FT/'keys.npy');source=read(ROOT/'docs/step2_xgb_ranker_v1.json')['source_refs']['outputs'];outputs=read(verify(source))['learned_native_ooc'];pkeys=np.array([[c['siret'].encode() for c in outputs[q]] for q in data['pilot']['ids']],dtype='S14');del outputs
    buffers={};hits={};sur={};fixed={};confirmation={};choices={str(cv):read(OWN/f'risk{cv}/receipt.json')['chosen'] for cv in range(5)}
    for name in VARIANTS+('selected',):
        score=np.empty(n,dtype=np.float32);status=np.empty(n,dtype=np.int8);ranking=np.empty((n,100),dtype=np.float32);hh=np.empty((n,2),dtype=bool);masks={str(v):np.zeros(n,dtype=bool) for v in LEVELS};dc=np.zeros((10,n),dtype=bool);pp=[];pt=[];ps=[];psc=[]
        for cv in range(5):
            variant=choices[str(cv)] if name=='selected' else name;rr=read(OWN/f'risk{cv}/receipt.json')['variants'][variant];p=pack(cv,'outer',variant);d=np.load(verify(rr['decisions']));test=p['indices'][p['indices']<n];ng=len(test)
            if not np.array_equal(test,np.flatnonzero(part==cv)):raise ValueError('Outer test coverage')
            score[test]=d['score'][:ng];status[test]=d['tier'][:ng];ranking[test]=p['predictions'][:ng];hh[test]=p['hits'][:ng];pp.append(p['predictions'][ng:]);pt.append(p['top'][ng:]);ps.append(d['tier'][ng:]);psc.append(d['score'][ng:])
            for v in LEVELS:masks[str(v)][test]=d['sur_'+str(v)][:ng]
            for j,threshold in enumerate(rr['decile_thresholds']):dc[j,test]=d['score'][:ng]>=threshold
        top4=np.argsort(-ranking,axis=1,kind='stable')[:,:4];np.savez_compressed(OWN/(name+'.npz'),hits=hh,score=score,tier=status,top4=top4,deciles=dc,**{'sur_'+v:a for v,a in masks.items()});hits[name]=hh;buffers[name]=dict(hits=hh,score=score,tier=status,top4=top4,deciles=dc);sur[name]={v:bundle(hh,mask,data) for v,mask in masks.items()};fixed[name]=describe(hh,status,data)
        avg=np.mean(pp,axis=0,dtype=np.float32);p4=np.argsort(-avg,axis=1,kind='stable')[:,:4];at=p4[:,0];pv=np.load(r.FT/'p_pilot_views.npy')[:,:,[s.b.VIEWS.index('exact'),s.b.VIEWS.index('union')]];ph=pv[np.arange(526),at];agree=np.all(np.array(pt)==at[None,:],axis=0);pst=np.min(ps,axis=0);pst[~agree]=0
        confirmation[name]=dict(top1=s.bench.metrics(ph),statuses={t:s.bench.metrics(ph,pst==j) for j,t in enumerate(TIERS)},policy='DESCRIPTIVE_ONLY_FIVE_MODEL_CONSENSUS',proposal_file=proposals(OWN/(name+'_526.jsonl'),data['pilot']['ids'],pkeys,p4,np.mean(psc,axis=0),pst))
        if name=='selected':proposal_ref=proposals(OWN/'selected_g0.jsonl',data['ids'],keys,top4,score,status)
    if (int(hits['base'][:,0].sum()),int(hits['base'][:,1].sum()))!=(12275,13382) or not np.array_equal(hits['base'],hits['risk_crm']):raise ValueError('Unchanged ranker invariant')
    delta=s.boot({v:hits[v] for v in VARIANTS},np.array(data['groups']));dc=buffers['selected']['deciles'];curve=[];previous=np.zeros(n,dtype=bool)
    for j in range(10):
        curve.append(dict(calibration_top_percent=(j+1)*10,cumulative=bundle(hits['selected'],dc[j],data),band=bundle(hits['selected'],dc[j]&~previous,data)));previous=dc[j]
    # Scores below the minimum calibration score belong to the last open tail.
    curve.append(dict(calibration_top_percent='below_calibration_min',cumulative=None,band=bundle(hits['selected'],~previous,data)))
    result=dict(status='COMPLETE_DEVELOPMENT_NOT_PRODUCTION',queries=n,companies=11483,anti_leakage=read(OWN/'anti_leakage.json'),variants={v:dict(top1=s.bench.metrics(hits[v]),deltas=delta.get(v),predictions=record(OWN/(v+'.npz'))) for v in VARIANTS},top1_cluster_bootstrap95=absolute_boot(hits,np.array(data['groups'])),sur=sur,fixed_policy=fixed,selected_cv=choices,selected_policy=fixed['selected'],selected_decile_curve=curve,confirmation526=confirmation,selected_proposals=proposal_ref,source_refs=data['source_refs'],plan=record(OWN/'plan.json'),new_signal_fits=0,new_gemma_cache=False,labels_changed=False,pools_changed=False,closed_sets_opened=False,production_allowed=False,selection='Per-CV calibration only; no global winner chosen from outer tests',exploratory_top1_gain=delta['rank_crm']['union']['paired_company_bootstrap95_pp'][0]>0)
    write(OWN/'result.json',result);return dict(result=record(OWN/'result.json'))


def independent(score,hits,level,probable=False):
    from scripts.retrieval_site_operational_metric import wilson
    np=modules().np
    for threshold in np.unique(score):
        mask=score>=threshold;n=int(mask.sum());k=int(hits[mask,1].sum())
        if k/n>=(.9 if probable else 0) and wilson(k,n)[0]>=(.88 if probable else level):return float(threshold)
    return None


def verify_all(directory):
    s=modules();np=s.np;data=read(OWN/'data.json');ds=r.dataset(SRC);models=0;vectors=0;thresholds=0;proofrefs=[]
    for cv in range(5):
        train,test,proper,cal,inner=r.roles(SRC,cv)
        for kind in ('outer','cal','inner0','inner1','inner2'):
            rr=read(OWN/f'rank{cv}_{kind}/receipt.json');tr=np.array(rr['train_indices']);te=np.array(rr['test_indices']);expected=train if kind=='outer' else proper if kind=='cal' else inner[int(kind[-1])][0]
            assert np.array_equal(tr,expected) and not set(ds.groups[tr])&set(ds.groups[te]);assert schema(tr)==rr['schema'];assert s.bench.sha(ds.views[tr,:,0].astype(np.float32))==rr['target_sha256']
            original=read(SRC/f'rank{cv}_{kind}/receipt.json');m=read(verify(original['models']))[0];nb=r.nb_replay(s,ds,m['uses'][0]['nb_scoring'],te);SX=np.concatenate([ds.X[te],nb,ds.semantic[te],columns(ds,te,rr['schema'])],axis=2);model=s.xgb.XGBRanker(n_jobs=1);model.load_model(verify(rr['model']));pred=model.predict(SX.reshape(-1,len(rr['features']))).reshape(len(te),100);expected=np.load(verify(rr['predictions']));assert np.array_equal(pred,expected['predictions']);rf,hh,tt=g.risk_features(ds,te,pred,nb,g.context(te,nb));assert np.array_equal(rf,expected['risk'],equal_nan=True) and np.array_equal(hh,expected['hits']) and np.array_equal(tt,expected['top']);models+=1;vectors+=len(te);proofrefs += [rr['model'],rr['predictions']];del SX,nb,model,expected;gc.collect()
        rr=read(OWN/f'risk{cv}/receipt.json');assert not set(ds.groups[proper])&set(ds.groups[cal]) and not set(ds.groups[np.r_[proper,cal]])&set(ds.groups[test])
        for name in VARIANTS:
            rec=rr['variants'][name];p=pack(cv,'cal',name);e=pack(cv,'outer',name);d=np.load(verify(rec['decisions']));model=s.xgb.Booster();model.load_model(verify(rec['model']));assert model.feature_names==rec['features'] and model.num_boosted_rounds()==200
            if rec['schema']:assert schema(proper)==rec['schema']
            for source,expected in [(p,d['cal_score']),(e,d['score'])]:
                X=source['risk'];X=np.column_stack([X,columns(ds,source['indices'],rec['schema'],source['top'])]) if rec['schema'] else X;assert np.array_equal(model.predict(s.xgb.DMatrix(X,feature_names=rec['features'],nthread=1)),expected)
            for level in LEVELS:
                target=independent(d['cal_score'],p['hits'],level/100);assert target==rec['choices'][str(level)];assert np.array_equal(d['sur_'+str(level)],d['score']>=target if target is not None else np.zeros(len(e['indices']),dtype=bool));thresholds+=1
            sur=independent(d['cal_score'],p['hits'],.975);remaining=d['cal_score']<sur if sur is not None else np.ones(len(p['indices']),dtype=bool);prob=independent(d['cal_score'][remaining],p['hits'][remaining],.88,True);assert rec['policy']==dict(sur=sur,probable=prob);assert np.array_equal(tiers(d['score'],rec['policy']),d['tier']);thresholds+=2;proofrefs += [rec['model'],rec['decisions']]
        eligible=[v for v in ('base','risk_crm','rank_crm') if rr['variants'][v]['cal_top1_union']>=rr['variants']['base']['cal_top1_union']];winner=max(eligible,key=lambda v:(rr['variants'][v]['cal_statuses']['SUR']['count'],rr['variants'][v]['cal_statuses']['SUR']['count']+rr['variants'][v]['cal_statuses']['PROBABLE']['count']));assert winner==rr['chosen']
    result=read(OWN/'result.json');views=np.load(r.FT/'views.npy');part=np.array(data['partition']);seen=0
    for name in VARIANTS+('selected',):
        p=np.load(OWN/(name+'.npz'));assert np.array_equal(p['hits'],views[np.arange(17427),p['top4'][:,0]]) and np.all(np.isin(p['tier'],[0,1,2]));assert sum(result['fixed_policy'][name]['statuses'][t]['pooled']['count'] for t in TIERS)==17427
        for cv in range(5):
            variant=result['selected_cv'][str(cv)] if name=='selected' else name;rec=read(OWN/f'risk{cv}/receipt.json')['variants'][variant];d=np.load(verify(rec['decisions']));mask=part==cv;assert np.array_equal(p['tier'][mask],d['tier'][:int(mask.sum())])
    import json
    with (OWN/'selected_g0.jsonl').open() as stream:
        p=np.load(OWN/'selected.npz');keys=np.load(r.FT/'keys.npy')
        for i,line in enumerate(stream):
            q=json.loads(line);assert q['query_id']==data['ids'][i] and len(q['proposals'])==4 and q['proposals']==[v.decode() for v in keys[i,p['top4'][i]]];seen+=1
    assert models==25 and thresholds==90 and seen==17427
    for ref in data['source_refs']:verify(ref)
    check();proof=dict(passed=True,new_rankers=25,new_risks=10,smoke_ranker=1,smoke_risk=1,ranking_vectors_replayed=vectors,independent_thresholds=90,company_overlap=0,source_refs_verified=len(data['source_refs']),all17427_statuses_retained=True,top4_proposals_checked=seen,model_refs=proofrefs,elapsed_seconds=elapsed());write(OWN/'verification.json',proof);return proof


def pct(metric):return 'indéf.' if metric is None else f'{metric["percent"]:.2f}'


def report(result):
    audit=result['anti_leakage'];cv=max(audit['g0'][col][kind]['cramer_v'] for col in COLS for kind in ('cv','population_fold','recent_cv','recent_population_fold'));lines=['**Fait** —17427/11483, mêmes5CV/OOF ; `scripts/step2_crm_columns_tiers.py`, rapports `docs/step2_crm_columns_tiers.{json,md}`, handover.',f'**Anti-fuite** —Jointure non ambiguë ; V maximal folds/CV={cv:.3f}. Présence label raw selon état : '+', '.join(f'{v} {m["percent"]:.2f}%' for v,m in audit['raw']['stc_state']['modalities'].items())+' ; temporalité non vérifiée.','', '| A | Exact % [IC95 entreprise] | UNION % [IC95 entreprise] | ΔUNION points [IC95 apparié] |','|---|---|---|---|']
    for name in VARIANTS:
        m=result['variants'][name];ci=result['top1_cluster_bootstrap95'][name];cells=[pct(m['top1'][v])+' '+str([round(x,2) for x in ci[v]]) for v in ('exact','union')];d=m['deltas'];delta='—' if d is None else f'{d["union"]["delta_pp"]:+.2f} '+str([round(x,2) for x in d['union']['paired_company_bootstrap95_pp']]);lines.append('| '+name+' | '+' | '.join(cells)+' | '+delta+' |')
    lines+=['', '| B statut | n / part % | Précision UNION % [Wilson95] | Précision exacte % [Wilson95] |','|---|---|---|---|']
    for t in reversed(TIERS):
        m=result['selected_policy']['statuses'][t]['pooled'];p=m['precision'];cells=[pct(p[v])+(' '+str([round(x,2) for x in p[v]['wilson95_percent']]) if p[v] else '') for v in ('union','exact')];lines.append(f'| {t} | {m["count"]} / {m["coverage"]["percent"]:.2f} | '+' | '.join(cells)+' |')
    lines+=['', '**Écarts / choix** —B choisie par CV sur calibration seulement, jamais gagnant global OOF ; PROBABLE calibré en bande exclusive. Historiques7530=MISSING.', '**Preuve** —`run` puis vérification réelle25LTR/10risques+2smokes/90seuils, top4 des17427 ; '+f'{result["elapsed_seconds"]/60:.2f}min, pic{result["peak_bytes"]/1024**3:.2f}Gio.', '526 descriptifs seuls ; courbes A96/97/97,5/98, déciles et catégories détaillés dans les rapports ; labels/pools/jeuxfermés/VM/tiers intacts.', '**Reste / risques** —Jeux consommés, sélection/calibration adaptatives, temporalité et validation indépendante manquent ; aucun déploiement.', 'Recommandation unique : conserver la référence en production non certifiée/A_VERIFIER ; traiter les trois statuts comme politique de développement à valider indépendamment.']
    assert len(lines)<=25;return '\n'.join(lines)+'\n'


def details(result):
    lines=['\n## Courbes A — précision UNION / couverture %\n','| Cible Wilson calibration | base | rank_crm | risk_crm |','|---|---|---|---|']
    for v in LEVELS:lines.append('| '+str(v)+' | '+' | '.join(pct(result['sur'][name][str(v)]['pooled']['precision']['union'])+' / '+pct(result['sur'][name][str(v)]['pooled']['coverage']) for name in VARIANTS)+' |')
    lines+=['\n## Courbe B — seuils déciles de calibration, cumuls et bandes disjointes\n','| Préfixe calibration % | Couverture test % | UNION cumul % | Exact cumul % | n bande | UNION bande % |','|---|---|---|---|---|---|']
    for row in result['selected_decile_curve']:
        cum=row['cumulative']['pooled'] if row['cumulative'] else None;band=row['band']['pooled'];lines.append('| '+str(row['calibration_top_percent'])+' | '+('— | — | —' if cum is None else pct(cum['coverage'])+' | '+pct(cum['precision']['union'])+' | '+pct(cum['precision']['exact']))+f' | {band["count"]} | '+pct(band['precision']['union'])+' |')
    lines+=['\n## B par catégorie produit brute — toutes cellules, historiques séparés\n','| Catégorie | Statut | n | Part interne % | UNION % [Wilson95] | Exact % [Wilson95] |','|---|---|---|---|---|---|']
    for cat,rows in result['selected_policy']['by_product'].items():
        for t in reversed(TIERS):
            m=rows[t];cells=[pct(m['precision'][v])+(' '+str([round(x,2) for x in m['precision'][v]['wilson95_percent']]) if m['precision'][v] else '') for v in ('union','exact')];lines.append(f'| {cat} | {t} | {m["count"]} | {pct(m["coverage"])} | '+' | '.join(cells)+' |')
    lines+=['\nLes déciles ne coupent pas les ex aequo et ne garantissent pas10% de dossiers test par bande ; la queue sous le minimum calibration est publiée séparément. Les IC de couverture sont descriptifs. Les JSON incluent tous n/N, Wilson, associations anti-fuite, résultats parCV/fold et politiques fixes. Propositions privées et modèles/reçus : `'+str(OWN)+'`. Choix calibration parCV : `'+str(result['selected_cv'])+'`.']
    return '\n'.join(lines)+'\n'


def publish(stop=None):
    paths=['docs/step2_crm_columns_tiers.json','docs/step2_crm_columns_tiers.md','handover.md'];verify(read(OWN/'plan.json')['publication_doc'])
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign/concurrent dirty publication path: '+path)
    if (ROOT/paths[0]).exists():raise ValueError('Foreign existing publication JSON')
    if stop:result=dict(status='STOP_INCOMPLETE',reason=stop,elapsed_seconds=elapsed(),production_allowed=False);text='**Fait** —STOP_INCOMPLETE : '+stop+'\n**Preuve** —Reçus conservés ; aucun partiel promu.\n**Écarts / choix** —Pas de réduction du banc.\n**Reste / risques** —Recommandation unique : garder référence/A_VERIFIER.\n';extra=''
    else:
        result=read(OWN/'result.json');assert read(OWN/'verification.json')['passed'] and elapsed()<10800;result.update(elapsed_seconds=elapsed(),peak_bytes=max(read(p)['peak_bytes'] for p in OWN.glob('*/receipt.json')),verification=record(OWN/'verification.json'));text=report(result);extra=details(result)
    write(OWN/'result.json',result);write(ROOT/paths[0],result);(OWN/'final_report.md').write_text(text);DOC.write_text((OWN/'protocol.md').read_text()+'\n## Résultats\n\n'+text+extra)
    before=(ROOT/'handover.md').read_text();line=f'\n- 2026-10-20 — Étape2-13 CRM/3statuts : {result["status"]}, durée{result["elapsed_seconds"]:.1f}s, docs `step2_crm_columns_tiers`, script et reçus `{OWN}`. Anti-fuiteavantfit,17427/11483/mêmes5CV/OOF/100candidats,25LTR+10risquesnouveaux/S1S2BASEfigés,2usagesCRM,3statuts sélection/calibration apprentissage parCV,95FORTnonretirés,526descriptifs,top4toujours. Labels/pools/jeuxclos/VM/étrangers/pending/web/push intouchés. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_crm_columns_tiers.json`.\n'
    with (ROOT/'handover.md').open('a') as stream:stream.write(line)
    if (ROOT/'handover.md').read_text()!=before+line:raise ValueError('Concurrent journal edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','feat: publish CRM columns and nested three-tier policy','--',*paths],cwd=ROOT,check=True);write(OWN/'publication.json',dict(commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip(),status='VERIFIED_COMMITTED'))


def execute(stage):
    directory=OWN/stage;directory.mkdir();wait=admit(directory);check()
    with (directory/'worker.log').open('w') as log:
        child=subprocess.Popen([str(r.io.CPU),'-u',str(SCRIPT),'worker','--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if elapsed()>=10680:child.terminate();child.wait();raise TimeoutError('Three-hour wall deadline, publication reserve120s')
            time.sleep(.5)
    if child.returncode:raise RuntimeError(f'{stage}: exit{child.returncode}, inspect {directory}/worker.log')
    print('COMPLETE '+stage,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker','inspect']);p.add_argument('--stage');args=p.parse_args()
    if args.action=='inspect':
        _,audit,_,_=inspect_columns();print(audit);return
    if args.action=='run':
        OWN.mkdir(parents=True,exist_ok=True)
        with (OWN/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (OWN/'budget.json').exists():raise ValueError('Fresh namespace required')
            write(OWN/'budget.json',dict(started_unix=time.time()));seal()
            try:
                for stage in ('prepare','smoke_rank','smoke_risk'):execute(stage)
                for i,stage in enumerate(f'rank{cv}_{kind}' for cv in range(5) for kind in ('outer','cal','inner0','inner1','inner2')):
                    execute(stage)
                    if i==0:
                        fit=read(OWN/stage/'receipt.json')['seconds'];projection=elapsed()+1.5*24*fit+1500;write(OWN/'projection.json',dict(seconds=projection,first_full_stage_seconds=fit))
                        if projection>10680:raise TimeoutError('Projection exceeds3h; no reduced benchmark')
                for stage in [f'risk{cv}' for cv in range(5)]+['collect','verify']:execute(stage)
                publish();write(OWN/'state.json',dict(status='COMPLETE',publication=read(OWN/'publication.json'),wall_seconds=elapsed()));print((OWN/'final_report.md').read_text(),flush=True)
            except BaseException as exc:
                write(OWN/'state.json',dict(status='STOP',reason=str(exc)))
                try:publish(str(exc))
                except BaseException as error:write(OWN/'publication_error.json',dict(reason=str(error)))
                raise
        return
    directory=OWN/args.stage
    if directory.parent!=OWN or not directory.exists():raise ValueError('Owned worker namespace required')
    guard=g.Guard(directory);start=time.monotonic();check()
    if args.stage=='prepare':value=prepare(directory)
    elif args.stage=='smoke_rank':value=rank(directory,0,'cal',True)
    elif args.stage=='smoke_risk':value=risk(directory,0,True)
    elif args.stage.startswith('rank'):
        cv,kind=args.stage[4:].split('_');value=rank(directory,int(cv),kind)
    elif args.stage.startswith('risk'):value=risk(directory,int(args.stage[4:]))
    elif args.stage=='collect':value=collect(directory)
    elif args.stage=='verify':value=verify_all(directory)
    else:raise ValueError('Unknown stage')
    value.update(seconds=time.monotonic()-start,peak_bytes=guard.peak);write(directory/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
