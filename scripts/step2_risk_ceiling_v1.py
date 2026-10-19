"""Frozen-OOF SIREN risk, UNION tradeoff and label-side ceiling diagnostics."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import step2_risk_model_v1 as old
read,write,record,verify=old.read,old.write,old.record,old.verify
HOME=old.HOME;OWN=HOME/'step2_risk_ceiling_v1_20261020';SRC=old.OWN/'full';SCRIPT=Path(__file__).resolve();DOC=ROOT/'docs/step2_risk_ceiling_v1.md'
LEVELS=(96,97,97.5,98);VIEWS=('siren','exact','union')


def modules():return old.modules()


def check():
    plan=read(OWN/'plan.json')
    for ref in plan['implementations']+[plan['protocol']]:verify(ref)
    old.check(SRC);return plan


def seal():
    s=modules();paths=[SCRIPT,Path(old.__file__),Path(s.__file__),Path(s.b.__file__),Path(s.bench.__file__),Path(old.io.__file__)]
    for path in paths+[DOC]:
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit before fit: '+str(path))
    (OWN/'protocol.md').write_text(DOC.read_text())
    write(OWN/'plan.json',dict(implementations=[record(p) for p in paths],protocol=record(OWN/'protocol.md'),features=old.FEATURES,params=old.RISK,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),budget_seconds=7200,source_publication='2cf01765',source_attestation='9f032ebc',publication_baseline=[record(DOC),record(ROOT/'handover.md')]))


def pack(cv,kind):
    ref=read(SRC/f'rank{cv}_{kind}/receipt.json')['predictions'];return modules().np.load(verify(ref))


def prepare(directory):
    s=modules();np=s.np;check();refs=[]
    for commit,path in [('2cf01765','docs/step2_risk_model_v1.json'),('9f032ebc','docs/step2_risk_model_v1.md')]:
        content=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        if (ROOT/path).read_bytes()!=content:raise ValueError('Source publication drift: '+path)
        refs.append(record(ROOT/path))
    source=read(SRC/'data.json');previous=read(SRC/'result.json')
    if previous!=read(ROOT/'docs/step2_risk_model_v1.json'):raise ValueError('External/Git source drift')
    for path in [SRC/'data.json',SRC/'result.json',SRC/'verification.json',SRC/'plan.json',SRC/'postverification.json']:refs.append(record(path))
    refs += [source['source_data'],source['source_pilot']]
    consumed=[old.FT/'keys.npy',old.FT/'family.npy',old.FT/'views.npy',old.FT/'p_pilot_views.npy',s.bench.SOURCE/'label_side.json']
    known={r['path']:r for r in source['sources']}
    for path in consumed:
        if str(path) in known:verify(known[str(path)])
        refs.append(record(path))
    s.pinned('7b0005a8','docs/step2_option_d.json');option_ref=read(ROOT/'docs/step2_option_d.json')['cohort']['artifacts']['label_side.json'];verify(option_ref);refs.append(option_ref)
    label=read(s.bench.SOURCE/'label_side.json');ids=source['ids'];n=len(ids);gt=[label[q]['human_ground_truth_siret'] for q in ids]
    if n!=17427 or source['companies']!=11483 or set(source['folds'])!={2,3,4}:raise ValueError('Full population required')
    if [x[:9] for x in gt]!=source['groups'] or [label[q]['fold'] for q in ids]!=source['folds']:raise ValueError('Human target/group/fold mismatch')
    keys=np.load(old.FT/'keys.npy');views=np.load(old.FT/'views.npy');sir=keys.astype('S9')==np.array([x[:9] for x in gt],dtype='S9')[:,None]
    if not np.array_equal(keys==np.array(gt,dtype='S14')[:,None],views[:,:,0]) or (views[:,:,1]&~sir).any():raise ValueError('Ground-truth view integrity')
    if not np.array_equal(np.load(SRC/'origin.npy'),np.arange(n)):raise ValueError('Origin order')
    pilot=read(verify(source['source_pilot']));outref=read(ROOT/'docs/step2_xgb_ranker_v1.json')['source_refs']['outputs'];verify(outref);refs.append(outref)
    outputs=read(outref['path'])['learned_native_ooc'];pkeys=np.array([[c['siret'].encode() for c in outputs[q]] for q in pilot['ids']],dtype='S14')
    psir=pkeys.astype('S9')==np.array(pilot['groups'],dtype='S9')[:,None]
    if set(pilot['groups'])&set(source['groups']) or not np.array_equal(psir,np.load(old.FT/'p_pilot_views.npy')[:,:,s.b.VIEWS.index('siren')]):raise ValueError('Pilot custody')
    family=np.load(old.FT/'family.npy');fr=ROOT/'docs/step2_error_families_g0.json';s.pinned('c37a1a7b','docs/step2_error_families_g0.json');refs.append(record(fr))
    historical={x['query_id'] for x in read(fr)['assignments'] if x['family']=='colocation'}
    if len(historical)!=1046 or set(np.array(ids)[family])!=historical:raise ValueError('Historical family drift')
    for cv in range(5):
        receipt=read(SRC/f'risk{cv}/receipt.json');refs+=[record(SRC/f'risk{cv}/receipt.json'),receipt['model'],receipt['decisions']]
        train,test,proper,cal,inner=old.roles(SRC,cv)
        for k,a in [('proper',proper),('calibration',cal),('test',test)]:
            if receipt['roles'][k]!=a.tolist():raise ValueError('Role drift')
        for kind in ('outer','cal','inner0','inner1','inner2'):
            rr=read(SRC/f'rank{cv}_{kind}/receipt.json');refs += [record(SRC/f'rank{cv}_{kind}/receipt.json'),rr['predictions'],rr['models']]
    for ref in refs:verify(ref)
    np.save(OWN/'siren.npy',np.concatenate([sir,psir]));np.save(OWN/'candidate_siren.npy',np.concatenate([keys.astype('S9'),pkeys.astype('S9')]));np.save(OWN/'eligible.npy',views[:,:,1].any(1));np.save(OWN/'family.npy',family)
    payload=dict(ids=ids,groups=source['groups'],partition=source['partition'],folds=source['folds'],queries=n,companies=source['companies'],sources=refs,
        pilot=pilot,static=[record(OWN/name) for name in ('siren.npy','candidate_siren.npy','eligible.npy','family.npy')])
    write(OWN/'data.json',payload)
    return dict(data=record(OWN/'data.json'),eligible=int(views[:,:,1].any(1).sum()),family=int(family.sum()),signal_fits=0)


def outcomes(p):
    np=modules().np;sir=np.load(OWN/'siren.npy',mmap_mode='r')[p['indices'],p['top']]
    return np.column_stack([sir,p['hits']])


def proper_data(cv):
    s=modules();np=s.np;roles=read(SRC/f'risk{cv}/receipt.json')['roles'];proper=np.array(roles['proper']);lookup={v:i for i,v in enumerate(proper)};X=np.empty((len(proper),46),dtype=np.float32);y=np.empty((len(proper),3),dtype=bool);covered=np.zeros(len(proper),dtype=int)
    for k in range(3):
        p=pack(cv,f'inner{k}');loc=[lookup[v] for v in p['indices']];X[loc]=p['risk'];y[loc]=outcomes(p);covered[loc]+=1
    if not (covered==1).all():raise ValueError('Proper OOF coverage')
    return proper,X,y


def select(score,y,level):
    t,stats=old.threshold(score,y,level/100,True);return dict(threshold=t,calibration=stats)


def accepted(score,choice):
    np=modules().np;t=choice['threshold'];return score>=t if t is not None else np.zeros(len(score),dtype=bool)


def fit(directory,cv,kind,smoke=False):
    s=modules();np=s.np;data=read(OWN/'data.json');proper,X,y=proper_data(cv);use=np.ones(len(proper),dtype=bool)
    if smoke:
        chosen=sorted(set(np.array(data['groups'])[proper]),key=lambda g:hashlib.sha256(('20261020|ceiling-smoke|'+g).encode()).digest())[:100]
        use &= np.isin(np.array(data['groups'])[proper],chosen)
    eligible=np.load(OWN/'eligible.npy');col=0 if kind=='siren' else 2
    if kind=='oracle':use &= eligible[proper]
    cp=pack(cv,'cal');ep=pack(cv,'outer');cy=outcomes(cp);ey=outcomes(ep);cm=np.ones(len(cp['indices']),dtype=bool) if kind=='siren' else eligible[cp['indices']]
    test=ep['indices']<data['queries'];rows=np.arange(len(ep['indices'])) if kind=='siren' else np.flatnonzero(test)
    model=s.xgb.train(old.RISK,s.xgb.DMatrix(X[use],label=y[use,col],feature_names=old.FEATURES,nthread=1),num_boost_round=200);path=directory/'model.json';model.save_model(path);replay=s.xgb.Booster();replay.load_model(path)
    cx=s.xgb.DMatrix(cp['risk'],feature_names=old.FEATURES,nthread=1);ex=s.xgb.DMatrix(ep['risk'][rows],feature_names=old.FEATURES,nthread=1);cs=model.predict(cx);es=model.predict(ex)
    if not np.array_equal(cs,replay.predict(cx)) or not np.array_equal(es,replay.predict(ex)):raise ValueError('Risk model replay')
    levels=(98,99) if kind=='siren' else (98,);choices={str(v):select(cs[cm],cy[cm,col],v) for v in levels}
    np.savez_compressed(directory/'predictions.npz',indices=ep['indices'][rows],cal_indices=cp['indices'],cal_score=cs,score=es,hits=ey[rows],top=ep['top'][rows])
    return dict(model=record(path),predictions=record(directory/'predictions.npz'),cv=cv,kind=kind,smoke=smoke,training_indices=proper[use].tolist(),calibration_indices=cp['indices'][cm].tolist(),choices=choices,
        features=old.FEATURES,params=old.RISK,replay_equal=True,fit_rows=int(use.sum()),test_vectors_replayed=len(es),signal_fits=0)


def metric(y,mask):
    s=modules();m=int(mask.sum());n=len(mask)
    return dict(count=m,queries=n,coverage=s.b.metric(m,n),precision={v:s.b.metric(int(y[mask,k].sum()),m) if m else None for k,v in enumerate(VIEWS)},false_union=int((mask&~y[:,2]).sum()))


def bundle(y,mask,partition,folds):
    np=modules().np
    return dict(pooled=metric(y,mask),by_cv={str(c):metric(y[partition==c],mask[partition==c]) for c in range(5)},by_population_fold={str(c):metric(y[folds==c],mask[folds==c]) for c in (2,3,4)})


def independent(score,y,level):
    from scripts.retrieval_site_operational_metric import wilson
    np=modules().np;best=None;maximum=0.
    # Independent inclusive-cutoff scan; no threshold from a held-out label.
    for t in np.unique(score):
        a=score>=t;lower=wilson(int(y[a].sum()),int(a.sum()))[0];maximum=max(maximum,lower)
        if lower>=level/100 and best is None:best=float(t)
    return best,maximum


def summarize(directory):
    s=modules();np=s.np;data=read(OWN/'data.json');n=data['queries'];part=np.array(data['partition']);fold=np.array(data['folds']);eligible=np.load(OWN/'eligible.npy');family=np.load(OWN/'family.npy');y=np.empty((n,3),dtype=bool)
    sm={str(v):np.zeros(n,dtype=bool) for v in (98,99)};um={str(v):np.zeros(n,dtype=bool) for v in LEVELS};of=np.zeros(n,dtype=bool);orr=of.copy();thresholds={};p_siren=[];p_mask={str(v):[] for v in (98,99)};p_rank=[];replays=0
    for cv in range(5):
        sr=read(OWN/f'siren{cv}/receipt.json');sp=np.load(verify(sr['predictions']));orrf=read(OWN/f'oracle{cv}/receipt.json');op=np.load(verify(orrf['predictions']));cp=pack(cv,'cal');ep=pack(cv,'outer');cy=outcomes(cp);ey=outcomes(ep);g=ep['indices']<n;test=ep['indices'][g];y[test]=ey[g]
        original=read(SRC/f'risk{cv}/receipt.json');booster=s.xgb.Booster();booster.load_model(verify(original['model']));cs=booster.predict(s.xgb.DMatrix(cp['risk'],feature_names=old.FEATURES,nthread=1));es=booster.predict(s.xgb.DMatrix(ep['risk'][g],feature_names=old.FEATURES,nthread=1));saved=np.load(verify(original['decisions']))
        if not np.array_equal(es,saved['risk_score'][g]):raise ValueError('Frozen UNION risk replay')
        thresholds[str(cv)]={'siren':sr['choices'],'union':{},'oracle_refit':orrf['choices']}
        for level in (98,99):
            choice=sr['choices'][str(level)];t,maxlower=independent(sp['cal_score'],cy[:,0],level)
            if t!=choice['threshold']:raise ValueError('SIREN Wilson cutoff')
            choice['maximum_calibration_lower95']=maxlower;a=accepted(sp['score'],choice);sm[str(level)][test]=a[g];p_mask[str(level)].append(a[~g]);replays+=1
        for level in LEVELS:
            choice=select(cs,cy[:,2],level);t,maximum=independent(cs,cy[:,2],level)
            if t!=choice['threshold']:raise ValueError('UNION Wilson cutoff')
            choice['maximum_calibration_lower95']=maximum;thresholds[str(cv)]['union'][str(level)]=choice;um[str(level)][test]=accepted(es,choice);replays+=1
        if not np.array_equal(um['97'][test],saved['risk_97'][g]):raise ValueError('Previous97 policy drift')
        cm=eligible[cp['indices']];choice=select(cs[cm],cy[cm,2],98);t,maximum=independent(cs[cm],cy[cm,2],98)
        if t!=choice['threshold']:raise ValueError('Oracle frozen cutoff')
        choice['maximum_calibration_lower95']=maximum;thresholds[str(cv)]['oracle_frozen']=choice;of[test]=eligible[test]&accepted(es,choice);replays+=1
        choice=orrf['choices']['98'];t,maximum=independent(op['cal_score'][cm],cy[cm,2],98)
        if t!=choice['threshold']:raise ValueError('Oracle refit cutoff')
        choice['maximum_calibration_lower95']=maximum;orr[test]=eligible[test]&accepted(op['score'],choice);replays+=1
        p_rank.append(ep['predictions'][~g]);p_siren.append(np.load(OWN/'candidate_siren.npy',mmap_mode='r')[ep['indices'][~g],ep['top'][~g]])
    previous=np.load(verify(read(SRC/'result.json')['predictions']))
    if not np.array_equal(y[:,1:],previous['hits']):raise ValueError('Global outcomes drift')
    families={name:metric(y[m],um['97'][m]) for name,m in [('dans',family),('hors',~family),('total',np.ones(n,dtype=bool))]}
    if sum(families[k]['count'] for k in ('dans','hors'))!=families['total']['count'] or sum(families[k]['false_union'] for k in ('dans','hors'))!=families['total']['false_union']:raise ValueError('Subgroup accounting')
    for name in ('dans','hors'):families[name]['global_coverage_percent']=100*families[name]['count']/n
    families['false_share_inside_percent']=100*families['dans']['false_union']/families['total']['false_union'] if families['total']['false_union'] else None
    avg=np.mean(p_rank,axis=0,dtype=np.float32);ptop=np.argsort(-avg,axis=1,kind='stable')[:,0];candidates=np.load(OWN/'candidate_siren.npy',mmap_mode='r');ensemble=candidates[n+np.arange(526),ptop];agree=np.all(np.array(p_siren)==ensemble[None,:],axis=0)
    truth=np.array(data['pilot']['groups'],dtype='S9');py=(ensemble==truth)[:,None];confirmation={}
    for level in (98,99):
        mask=agree&np.all(p_mask[str(level)],axis=0);count=int(mask.sum());confirmation[str(level)]=dict(count=count,coverage=s.b.metric(count,526),precision_siren=s.b.metric(int(py[mask,0].sum()),count) if count else None,site_status='A_VERIFIER',by_model={str(c):dict(count=int(p_mask[str(level)][c].sum()),precision_siren=s.b.metric(int(((p_siren[c]==truth)&p_mask[str(level)][c]).sum()),int(p_mask[str(level)][c].sum())) if p_mask[str(level)][c].any() else None) for c in range(5)})
    frozen=bundle(y[eligible],of[eligible],part[eligible],fold[eligible]);refit=bundle(y[eligible],orr[eligible],part[eligible],fold[eligible])
    for x in (frozen,refit):x['diagnostic_only']=True;x['full_population_coverage_percent']=100*x['pooled']['count']/n
    np.savez_compressed(OWN/'decisions.npz',hits=y,eligible=eligible,family=family,oracle_frozen=of,oracle_refit=orr,**{'siren_'+k:v for k,v in sm.items()},**{'union_'+k:v for k,v in um.items()})
    result=dict(status='COMPLETE_DIAGNOSTIC_NOT_PRODUCTION',queries=n,companies=data['companies'],features=old.FEATURES,params=old.RISK,source_refs=data['sources'],thresholds=thresholds,
        siren={k:bundle(y,v,part,fold) for k,v in sm.items()},union={k:bundle(y,v,part,fold) for k,v in um.items()},generator=dict(eligible=int(eligible.sum()),absent=int((~eligible).sum()),frozen=frozen,refit=refit,deployable=False),colocation=families,confirmation_siren526=confirmation,
        independent_cutoffs_reproduced=replays,signal_fits=0,closed_sets_opened=False,labels_changed=False,pools_changed=False,production_allowed=False,decisions=record(OWN/'decisions.npz'))
    write(OWN/'result.json',result);return dict(result=record(OWN/'result.json'),independent_cutoffs_reproduced=replays)


def verify_all(directory):
    s=modules();np=s.np;data=read(OWN/'data.json');groups=np.array(data['groups']);refs=[];vectors=0
    for cv in range(5):
        roles=read(SRC/f'risk{cv}/receipt.json')['roles'];proper=set(roles['proper']);cal=set(roles['calibration']);test=set(roles['test'])
        if set(groups[list(proper)])&set(groups[list(cal)]) or set(groups[list(proper|cal)])&set(groups[list(test)]):raise ValueError('Company leakage')
        for kind in ('siren','oracle'):
            receipt=read(OWN/f'{kind}{cv}/receipt.json');train=set(receipt['training_indices']);c=set(receipt['calibration_indices']);assert train<=proper and c<=cal and not(train&test) and receipt['replay_equal']
            assert receipt['features']==old.FEATURES and receipt['params']['monotone_constraints']==list(old.RISK['monotone_constraints'])
            for key in ('model','predictions'):verify(receipt[key]);refs.append(receipt[key])
            vectors+=receipt['test_vectors_replayed']
    for ref in data['sources']+data['static']:verify(ref)
    check();r=read(OWN/'result.json');assert r['independent_cutoffs_reproduced']==40 and r['signal_fits']==0
    proof=dict(passed=True,full_risk_fits=10,smoke_risk_fits=2,new_signal_fits=0,predictions_replayed=vectors,cutoffs_independently_reproduced=40,company_overlap=0,source_files_verified=len(data['sources']),model_refs=refs)
    write(OWN/'verification.json',proof);return proof


def text_report(r):
    def p(x):
        if x is None:return 'indéf.'
        a,z=x['wilson95_percent'];return f'{x["percent"]:.2f} [{a:.2f};{z:.2f}]'
    def row(label,m,view):return f'| {label} | {m["count"]} | {p(m["precision"][view])} | {m["coverage"]["percent"]:.2f} |'
    lines=['## Fait',f'G0 {r["queries"]} dossiers/{r["companies"]} entreprises, mêmes5CV/OOF/46features ; aucun signal réentraîné.',f'Fichiers `scripts/step2_risk_ceiling_v1.py`, `docs/step2_risk_ceiling_v1.{{json,md}}`, `handover.md`.','## Preuve',f'Durée {r["elapsed_seconds"]:.1f}s, pic {r["peak_bytes"]/1024**3:.2f}Gio ;12fits risque avec smokes,40seuils Wilson indépendants/rejeux/SHA/groupes vérifiés.', '| Cible SIREN, Wilson calibration | SUR | Précision test %, Wilson95 | Couverture % |','|---|---:|---|---:|']
    for level in ('98','99'):lines.append(row(level+'%',r['siren'][level]['pooled'],'siren'))
    lines.append('526 descriptifs entreprise98/99 : '+', '.join(f'{k}%={v["count"]}SUR, P={p(v["precision_siren"])}, C={v["coverage"]["percent"]:.2f}%' for k,v in r['confirmation_siren526'].items())+' ; site A_VERIFIER.')
    lines += ['| UNION, cible Wilson calibration | SUR | Précision test %, Wilson95 | Couverture % |','|---|---:|---|---:|']
    for level in ('96','97','97.5','98'):lines.append(row('Complet '+level+'%',r['union'][level]['pooled'],'union'))
    for name in ('frozen','refit'):lines.append(row('Oracle présence98% '+name,r['generator'][name]['pooled'],'union'))
    lines += ['| Co-adresse historique, risque UNION97% | SUR/N groupe | Précision %, Wilson95 | Couverture groupe % |','|---|---:|---|---:|']
    for name in ('dans','hors','total'):
        m=r['colocation'][name];lines.append(row(name,m,'union').replace('| '+str(m['count'])+' |','| '+str(m['count'])+'/'+str(m['queries'])+' |',1))
    lines += [f'Co-adresse : {r["colocation"]["dans"]["false_union"]}/{r["colocation"]["total"]["false_union"]} fauxSUR ; masque historique1046, pas une mesure causale.', '## Écarts / choix',f'Présence UNION :{r["generator"]["eligible"]}/{r["queries"]}. Oracle filtre proper/cal/test ; contraste poids figés. Couvertures oracle conditionnelles, jamais end-to-end.', '## Reste / risques','Wilson descriptif/adaptatif, entreprise corrélée,526 consommés ; aucune promotion. Détails par CV/fold publiés.', 'Recommandation unique : '+r['recommendation']]
    if len(lines)>30:raise ValueError('Report exceeds30 lines')
    return '\n'.join(lines)+'\n'


def publish(stop=None):
    budget=read(OWN/'budget.json');paths=['docs/step2_risk_ceiling_v1.json','docs/step2_risk_ceiling_v1.md','handover.md']
    for ref in read(OWN/'plan.json')['publication_baseline']:verify(ref)
    if (ROOT/paths[0]).exists():raise ValueError('Foreign existing publication JSON; preserve and stop')
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign/concurrent publication path: '+path)
    if stop:
        result=dict(status='STOP_INCOMPLETE',reason=stop,elapsed_seconds=time.time()-budget['started_unix'],production_allowed=False);report='## Fait\nSTOP_INCOMPLETE : '+stop+'\n## Preuve\nAucun résultat complet validé.\n## Écarts / choix\nPartiels conservés, non promus.\n## Reste / risques\nRecommandation unique : conserver A_VERIFIER.\n'
    else:
        check();result=read(OWN/'result.json');proof=read(OWN/'verification.json');assert proof['passed']
        result.update(elapsed_seconds=time.time()-budget['started_unix'],wait_seconds=budget['wait_seconds'],peak_bytes=max(read(p)['peak_bytes'] for p in OWN.glob('*/receipt.json')),verification=record(OWN/'verification.json'),plan=record(OWN/'plan.json'))
        sir=result['siren']['98']['pooled']['precision']['siren'];result['recommendation']='Auditer indépendamment le statut entreprise SUR/site A_VERIFIER, sans déploiement.' if sir and sir['wilson95_percent'][0]>=98 else 'Conserver A_VERIFIER ; ne pas promouvoir de statut SUR sur ces jeux consommés.'
        report=text_report(result)
    (OWN/'final_report.md').write_text(report);write(OWN/'result.json',result);before=(ROOT/'handover.md').read_text();write(ROOT/paths[0],result)
    annex=''
    if not stop:
        for level in ('98','99'):
            annex+=f'\n### SIREN{level}% par CV entreprise\n\n| CV | SUR | Précision SIREN %, Wilson95 | Couverture % |\n|---|---:|---|---:|\n'
            for cv,m in result['siren'][level]['by_cv'].items():
                p=m['precision']['siren'];pr=f'{p["percent"]:.4f} {p["wilson95_percent"]}' if p else 'indéfinie';annex+=f'| {cv} | {m["count"]} | {pr} | {m["coverage"]["percent"]:.4f} |\n'
    DOC.write_text((OWN/'protocol.md').read_text()+'\n## Résultats\n\n'+report+annex)
    line=f'\n- 2026-10-20 — Étape2-11 plafond SUR : {result["status"]}, durée{result["elapsed_seconds"]:.1f}s, résultats `docs/step2_risk_ceiling_v1.json`, rapport `{OWN}/final_report.md`. Mêmes OOF/plis/signaux gelés, SIREN98/99, UNION96/97/97,5/98, oracle présence exclusivement diagnostique, co-adresse historique1046 ; aucun label/pool/jeu fermé/VM/web/push/étranger/pending touché. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_risk_ceiling_v1.json`.\n'
    with (ROOT/'handover.md').open('a') as stream:stream.write(line)
    if (ROOT/'handover.md').read_text()!=before+line:raise ValueError('Concurrent journal edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','feat: publish frozen-OOF SIREN and UNION risk ceilings','--',*paths],cwd=ROOT,check=True)
    write(OWN/'publication.json',dict(status='VERIFIED_COMMITTED',commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip()))


def execute(stage):
    directory=OWN/stage;directory.mkdir();wait=old.admit(directory);budget=read(OWN/'budget.json');budget['wait_seconds']+=wait;write(OWN/'budget.json',budget);check()
    with (directory/'worker.log').open('w') as log:
        child=subprocess.Popen([str(old.io.CPU),'-u',str(SCRIPT),'worker','--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if time.time()-budget['started_unix']-budget['wait_seconds']>=7140:
                child.terminate();child.wait();raise TimeoutError('Two-hour deadline, publication reserve60s')
            time.sleep(.5)
    if child.returncode:raise RuntimeError(f'{stage}: exit{child.returncode}, {directory}/worker.log')
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
                for stage in ['prepare','smoke_siren','smoke_oracle']+[k+str(c) for c in range(5) for k in ('siren','oracle')]+['summarize','verify']:execute(stage)
                publish();write(OWN/'state.json',dict(status='COMPLETE',publication=read(OWN/'publication.json'),wall_seconds=time.time()-read(OWN/'budget.json')['started_unix']))
                print((OWN/'final_report.md').read_text(),flush=True)
            except BaseException as exc:
                write(OWN/'state.json',dict(status='STOP',reason=str(exc)))
                try:publish(str(exc))
                except BaseException as pub:write(OWN/'publication_error.json',dict(reason=str(pub)))
                raise
        return
    directory=OWN/a.stage
    if directory.parent!=OWN or not directory.exists():raise ValueError('Owned worker namespace required')
    guard=old.Guard(directory);start=time.monotonic();check()
    if a.stage=='prepare':value=prepare(directory)
    elif a.stage.startswith('smoke_'):value=fit(directory,0,a.stage[6:],True)
    elif a.stage.startswith('siren'):value=fit(directory,int(a.stage[5:]),'siren')
    elif a.stage.startswith('oracle'):value=fit(directory,int(a.stage[6:]),'oracle')
    elif a.stage=='summarize':value=summarize(directory)
    elif a.stage=='verify':value=verify_all(directory)
    else:raise ValueError('Unknown stage')
    value.update(seconds=time.monotonic()-start,peak_bytes=guard.peak);write(directory/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
