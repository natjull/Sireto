"""Ten-hour continuation: measured admission, one epoch, fifteen own-query negatives.

Reuses sealed admissible text/data and the existing scientific evaluation only.
"""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import step2_gemma_finetune as r

HOME, CPU, METAL = r.HOME, r.CPU, r.METAL
SOURCE = r.OWN
OWN = HOME/'step2_gemma_finetune_10h_20261019'
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT/'config/step2_gemma_finetune_10h.json'
DOC = ROOT/'docs/step2_gemma_finetune_10h.md'
r.OWN, r.SCRIPT, r.CONFIG, r.DOC = OWN, SCRIPT, CONFIG, DOC
read, write, record, verify = r.read, r.write, r.record, r.verify


def seal(out, mini):
    from scripts import step2_colocated_signals as s
    out.mkdir(exist_ok=False);(out/'protocol.md').write_text(DOC.read_text().split('\n## Résultats\n')[0])
    files=[SCRIPT,Path(r.__file__),ROOT/'scripts/step2_gemma_finetune_verify.py',Path(s.__file__),Path(s.bench.__file__),
        Path(s.old.__file__),Path(s.b.__file__),Path(s.d.__file__),Path(s.old.v2.__file__),ROOT/'scripts/embeddinggemma2_v2_fit.py']
    for path in files+[CONFIG,DOC]:
        if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit code/config/avenant before launch: '+str(path))
        subprocess.run(['git','ls-files','--error-unmatch',str(path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    write(out/'plan.json',dict(smoke=mini,started_unix=time.time(),cfg=read(CONFIG),config=record(CONFIG),protocol=record(out/'protocol.md'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),implementations=[record(p) for p in files],
        base_files=[record(p) for p in sorted((HOME/'base').glob('*')) if p.is_file()]))


def prepare(out,directory,seconds):
    import numpy as np
    from scripts import step2_colocated_signals as s
    plan=r.check(out);cfg=plan['cfg'];source=SOURCE/('smoke' if plan['smoke'] else 'full')
    source_plan=read(source/'plan.json');verify(source_plan['protocol'])
    for ref in source_plan['base_files']:verify(ref)
    for ref in source_plan['implementations']+[source_plan['config']]:
        content=subprocess.check_output(['git','show',source_plan['git_head']+':'+str(Path(ref['path']).relative_to(ROOT))],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=ref['sha256']:raise ValueError('Historical preparation Git custody')
    receipt=read(source/'prepare/receipt.json')
    for ref in receipt['artifacts']:verify(ref)
    data=read(verify(receipt['data']));old_meta=Path(data['metadata']);texts=read(source/'texts.json');original=read(source/'training.json')
    rows=[dict(row,negatives=row['negatives'][:cfg['max_explicit_negatives']]) for row in original]
    groups=np.array(data['groups']);roles={k:np.array(v,dtype=int) for k,v in data['roles'].items()}
    qi=np.load(old_meta/'qtext.npy');ci=np.load(old_meta/'ctext.npy');pqi=np.load(source/'pilot_qtext.npy');pci=np.load(source/'pilot_ctext.npy')
    trials=[];selected_eval=roles['evaluation'];selected_proper=roles['proper'];cal=roles['calibration'];projection=None
    def sample(indices,fraction,purpose):
        companies=sorted(set(groups[indices]),key=lambda g:hashlib.sha256((str(cfg['seed'])+'|10h-'+purpose+'|'+g).encode()).digest())
        chosen=companies[:max(1,math.ceil(fraction*len(companies)))];return indices[np.isin(groups[indices],chosen)]
    if not plan['smoke']:
        mini=OWN/'mini';base=read(mini/'base_encode/receipt.json');adapted=read(mini/'adapted_encode/receipt.json');fit=read(mini/'train/receipt.json')
        semantic=read(mini/'semantic/receipt.json');mini_data=read(mini/'data.json');cached=set()
        for name in ('encode','confirmation_encode'):cached.update(read(verify(read(r.PREVIOUS/name/'result.json')['texts'])))
        slopes=[m['seconds']/len(m['train_indices']) for name in ('base','adapted') for m in read(mini/name/'result.json')['models']]
        options=[(f,1.) for f in cfg['evaluation_fractions']]+[(cfg['evaluation_fractions'][-1],f) for f in cfg['proper_training_fractions'][1:]]
        for ef,tf in options:
            ev=sample(roles['evaluation'],ef,'evaluation');pr=sample(roles['proper'],tf,'training');train=np.sort(np.r_[pr,cal]);keep=np.sort(np.r_[train,ev])
            needed=np.unique(np.r_[qi[keep],ci[keep].reshape(-1),pqi,pci.reshape(-1)]);needed=needed[needed>=0]
            missing=sum(texts[i] not in cached for i in needed);proper_set=set(pr.tolist());nrows=sum(row['row'] in proper_set for row in rows)
            components=dict(base_encode=base['seconds']*missing/max(1,base['new_this_attempt']),adapted_encode=adapted['seconds']*len(needed)/adapted['items'],
                training=fit['training_seconds']/fit['steps']*math.ceil(nrows/cfg['batch_size']),
                cpu_ranking=max(600.,max(slopes)*2*(len(train)+len(pr))),
                cpu_semantic=semantic['seconds']*(len(keep)+526)/(mini_data['queries']+36))
            projected=cfg['projection_multiplier']*sum(components.values())+cfg['projection_fixed_reserve_seconds']
            item=dict(evaluation_fraction=ef,proper_fraction=tf,evaluation_queries=len(ev),evaluation_companies=len(set(groups[ev])),proper_queries=len(pr),
                proper_companies=len(set(groups[pr])),training_pairs=nrows,texts=len(needed),base_missing=missing,components_seconds=components,projected_seconds=projected,available_seconds=seconds)
            trials.append(item)
            if projected<=seconds-60:selected_eval,selected_proper,projection=ev,pr,item;break
        cost=dict(status='ADMITTED' if projection else 'STOP_NO_SCOPE_WITHIN10H',mini_plan=record(mini/'plan.json'),mini_verification=record(mini/'verification.json'),
            trials=trials,selected=projection,selection_uses_scores=False,projection_is_not_measured_full_duration=True)
        write(out/'cost_plan.json',cost)
        if projection is None:raise TimeoutError('No company-random scope fits measured ten-hour projection')
        del cached;gc.collect()
    train=np.sort(np.r_[selected_proper,cal]);keep=np.sort(np.r_[train,selected_eval]);old_to_new=np.full(data['queries'],-1,dtype=int);old_to_new[keep]=np.arange(len(keep))
    proper_set=set(selected_proper.tolist());rows=[row for row in rows if row['row'] in proper_set]
    needed=np.unique(np.r_[qi[keep],ci[keep].reshape(-1),pqi,pci.reshape(-1)]);needed=needed[needed>=0]
    mapping=np.full(len(texts),-1,dtype=np.int32);mapping[needed]=np.arange(len(needed))
    def remap(value):return np.where(value>=0,mapping[np.maximum(value,0)],-1)
    text_values=[texts[i] for i in needed];write(out/'texts.json',text_values);write(directory/'texts.json',text_values)
    for path in source.glob('*.npy'):
        value=np.load(path,mmap_mode='r')
        if path.name.startswith('p_'):np.save(out/path.name,value)
        elif path.name=='pilot_qtext.npy':np.save(out/path.name,remap(value))
        elif path.name=='pilot_ctext.npy':np.save(out/path.name,remap(value))
        else:np.save(out/path.name,value[keep])
    for name in ('naf.npy','naf_targets.npy','addresses.npy'):np.save(directory/name,np.load(old_meta/name,mmap_mode='r')[keep])
    np.save(directory/'qtext.npy',remap(qi[keep]));np.save(directory/'ctext.npy',remap(ci[keep]));shutil.copyfile(source/'pilot.json',out/'pilot.json')
    for row in rows:
        row['row']=int(old_to_new[row['row']]);row['query']=int(mapping[row['query']]);row['positive']=int(mapping[row['positive']]);row['negatives']=[int(mapping[k]) for k in row['negatives']]
        if min([row['query'],row['positive']]+row['negatives'])<0 or len(row['negatives'])>15:raise ValueError('Negative/text mapping')
    write(out/'training.json',rows)
    old_counts=data['role_counts'];data.update(original_role_counts=old_counts,original_queries=data['queries'],original_companies=data['companies'])
    for key in ('ids','groups','names','folds'):data[key]=[data[key][i] for i in keep]
    data['roles']=dict(evaluation=old_to_new[selected_eval].tolist(),proper=old_to_new[selected_proper].tolist(),calibration=old_to_new[cal].tolist(),training=old_to_new[train].tolist())
    data['role_counts']={k:dict(queries=len(v),companies=len(set(np.array(data['groups'])[v]))) for k,v in data['roles'].items()}
    views=np.load(out/'views.npy');proper=np.array(data['roles']['proper']);local_q=np.load(directory/'qtext.npy');local_c=np.load(directory/'ctext.npy');skips=Counter()
    for i in proper:
        pos=np.flatnonzero(views[i,:,0])
        if not len(pos):skips['exact_absent_from_g0']+=1
        elif local_q[i]<0 or local_c[i,pos[0]]<0:skips['missing_text']+=1
    skips['no_safe_negative']=len(proper)-len(rows)-sum(skips.values())
    data.update(metadata=str(directory),queries=len(keep),companies=len(set(data['groups'])),training_pairs=len(rows),skipped_training=dict(skips),
        negative_counts=dict(total=sum(len(row['negatives']) for row in rows),maximum=max(len(row['negatives']) for row in rows)),
        exact_present=int(views[:,:,0].any(1).sum()),union_present=int(views[:,:,1].any(1).sum()),budget_reduced=len(keep)!=data['original_queries'])
    data['sources']+=list(receipt['artifacts'])+[record(source/'plan.json'),record(source/'prepare/receipt.json')]
    write(out/'data.json',data)
    metadata_seal=read(old_meta/'feature_seal.json');metadata_seal['files']={name:record(directory/name) for name in ('naf.npy','addresses.npy','qtext.npy','ctext.npy','texts.json')}
    metadata_seal['candidate_keys_sha256']=s.bench.sha(np.load(out/'keys.npy'));write(directory/'feature_seal.json',metadata_seal)
    artifacts=[record(path) for path in out.glob('*.npy')]+[record(out/name) for name in ('data.json','training.json','texts.json','pilot.json')]+list(metadata_seal['files'].values())+[record(directory/'naf_targets.npy'),record(directory/'feature_seal.json')]
    return dict(data=record(out/'data.json'),artifacts=artifacts,queries=data['queries'],companies=data['companies'],texts=len(needed),roles=data['role_counts'],
        training_pairs=len(rows),max_negatives=data['negative_counts']['maximum'],budget_reduced=data['budget_reduced'],source=record(source/'prepare/receipt.json'))


def train(out,directory,seconds):
    import numpy as np
    import torch
    from scripts.embeddinggemma2_v2_fit import cached_objective
    plan=r.check(out);cfg=plan['cfg'];rows=read(out/'training.json');texts=read(out/'texts.json');model,params,encode,cropped=r.load_model(cfg,True)
    optimizer=torch.optim.AdamW(params.values(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    step=0;cursor=0;elapsed=0.;exposed=set();binding=record(out/'plan.json')['sha256'];began=time.monotonic();last_save=began;own=directory/'checkpoint.pt';yield_requested=False
    def request_yield(signum,frame):
        nonlocal yield_requested
        yield_requested=True
    signal.signal(signal.SIGTERM,request_yield)
    if own.exists():
        state=torch.load(own,map_location='cpu',weights_only=True)
        if state['binding']!=binding:raise ValueError('Checkpoint binding')
        for name,p in params.items():p.data.copy_(state['adapter'][name].to('mps'))
        optimizer.load_state_dict(state['optimizer']);step=state['step'];cursor=state['cursor'];elapsed=state['seconds'];exposed=set(state['exposed'])
    initial={name:p.detach().cpu().clone() for name,p in params.items()} if not step else None
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        return value
    def save():
        nonlocal last_save
        temp=own.with_suffix('.tmp');torch.save(dict(adapter=cpu(params),optimizer=cpu(optimizer.state_dict()),step=step,cursor=cursor,seconds=elapsed+time.monotonic()-began,
            exposed=sorted(exposed),binding=binding),temp);os.replace(temp,own);last_save=time.monotonic()
        write(directory/'checkpoint_receipt.json',dict(step=step,cursor=cursor,unix=time.time(),binding=binding))
    order=np.random.default_rng(cfg['seed']).permutation(len(rows));total=math.ceil(len(rows)/cfg['batch_size']);warmup=max(1,round(total*cfg['warmup_fraction']));losses=[];norms=[];step_seconds=[]
    model.train()
    while cursor<len(order):
        if yield_requested or r.resource_reason():save();return 75
        if time.monotonic()-began>seconds-45:save();raise TimeoutError('Total ten-hour budget reached before one epoch')
        at=time.monotonic();chosen=order[cursor:cursor+cfg['batch_size']];batch=[rows[i] for i in chosen];q=[texts[row['query']] for row in batch];docs=[texts[row['positive']] for row in batch];columns=[]
        for row in batch:
            neg=row['negatives'];columns.append(list(range(len(docs),len(docs)+len(neg))));docs.extend(texts[i] for i in neg)
        mask=np.ones((len(q),len(docs)),dtype=bool)
        for i,col in enumerate(columns):mask[i,[i]+col]=False
        multiplier=(step+1)/warmup if step<warmup else cfg['final_lr_fraction']+(1-cfg['final_lr_fraction'])*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
        for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*multiplier
        optimizer.zero_grad(set_to_none=True);loss=cached_objective(encode,q,docs,mask,np.ones(len(q)),cfg['temperature'],cfg['microbatch_size'])
        norm=torch.nn.utils.clip_grad_norm_(params.values(),cfg['gradient_clip'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        optimizer.step();step+=1;cursor+=len(chosen);exposed.update(int(i) for i in chosen);losses.append(loss);norms.append(float(norm));torch.mps.empty_cache();step_seconds.append(time.monotonic()-at)
        if step<=2 or time.monotonic()-last_save>=cfg['checkpoint_seconds']-60:save()
        write(directory/'progress.json',dict(step=step,cursor=cursor,loss=loss,gradient_norm=float(norm),seconds=elapsed+time.monotonic()-began,exposed=len(exposed),pairs=len(rows),negatives_per_query_max=15))
        if plan['smoke'] and step>=2:break
    save()
    if not step:raise ValueError('No real optimizer step')
    changed=any(not torch.equal(initial[name],p.detach().cpu()) for name,p in params.items()) if initial is not None else True
    if not changed:raise ValueError('Adapter unchanged')
    return dict(status='MINI_TWO_REAL_STEPS' if plan['smoke'] else 'ONE_EPOCH_COMPLETE',checkpoint=record(own),steps=step,epochs=int(cursor>=len(order)),cursor=cursor,
        training_pairs=len(rows),exposed_pairs=len(exposed),exposure_fraction=len(exposed)/len(rows),training_seconds=elapsed+time.monotonic()-began,
        last_losses=losses[-3:],last_gradient_norms=norms[-3:],mean_step_seconds=sum(step_seconds)/len(step_seconds) if step_seconds else None,
        adapter_changed=changed,fresh_initialization=True,historical_adapter_loaded=False,max_negatives=15,tokenization=dict(cropped))


def audit(out):
    from scripts import step2_gemma_finetune_verify as original_audit
    original_audit.verify(out)
    import numpy as np
    data=read(out/'data.json');source=SOURCE/('smoke' if read(out/'plan.json')['smoke'] else 'full');old=read(source/'data.json');old_texts=read(source/'texts.json');texts=read(out/'texts.json')
    originals={old['ids'][row['row']]:row for row in read(source/'training.json')}
    for row in read(out/'training.json'):
        expected=originals[data['ids'][row['row']]]
        if [texts[i] for i in row['negatives']]!=[old_texts[i] for i in expected['negatives'][:15]]:raise ValueError('Not the first fifteen safe hard negatives')
    for role in ('proper','calibration','evaluation'):
        selected=set(np.array(data['groups'])[data['roles'][role]])
        expected={q for q,g in zip(old['ids'],old['groups']) if g in selected}
        actual={data['ids'][i] for i in data['roles'][role]}
        if expected!=actual:raise ValueError('Company sampling omitted a company row')
    if not read(out/'plan.json')['smoke']:
        fit=read(out/'train/receipt.json')
        if fit['epochs']!=1 or fit['exposure_fraction']!=1.:raise ValueError('Incomplete epoch cannot be published as complete')
    proof=read(out/'verification.json');proof.update(negative_prefix15_verified=True,whole_company_sampling_verified=True,
        budget_reduced=data['budget_reduced'],retained_queries=data['queries'],original_queries=data['original_queries'])
    write(out/'verification.json',proof);return dict(artifacts=[record(out/'verification.json')])


def report(value):
    def pct(metric):return 'indéfinie' if metric is None else f'{metric["hits"]}/{metric["queries"]}={metric["percent"]:.2f}%'
    lines=['## Fait',f'Statut {value["status"]} ; aucune promotion, labels et candidats inchangés.',
        f'Durée réelle calcul total : {value["compute_seconds"]/3600:.3f} h (mini inclus), Metal {value["metal_seconds"]/3600:.3f} h ; mur {value["wall_seconds"]/3600:.3f} h.',
        f'Pic worker RSS+driver {value["peak_bytes"]/1024**3:.2f} Gio (<8) ; attente ressources {value["wait_seconds"]/60:.1f} min.', '## Preuve']
    if 'variants' not in value or value['status'].startswith('STOP'):
        lines += [f'Arrêt : {value.get("reason")}. Étapes : {value.get("completed",[])}.','Aucun résultat complet validé pour évaluation/SUR/526.','## Écarts / choix','Avenant10h,15 négatifs propres,1époque ; réduction seulement par entreprises/hash si projection insuffisante.','## Reste / risques','Recommandation unique : conserver S2-BASE sans résultat complet.']
        return '\n'.join(lines)+'\n'
    lines += [f'Rôles retenus : {value["roles"]}.',f'Gradient : {value["training"]["steps"]} pas, 1époque, {value["training"]["exposed_pairs"]}/{value["training"]["training_pairs"]} paires exposées, jusqu’à15 négatifs sûrs.',
        f'Projection préalable : {value["cost_plan"]["selected"]["projected_seconds"]/3600:.2f} h ; réduction de budget : {value["budget_reduced"]}.']
    for name,variant in value['variants'].items():
        lines += [f'{name} exact {pct(variant["top1"]["exact"])} ; UNION {pct(variant["top1"]["union"])}.',
            f'{name} SUR : précision UNION {pct(variant["sur"]["precision"]["union"])} ; couverture {pct(variant["sur"]["coverage"])} (cible interne98%, non garantie).',
            f'{name} écart train−évaluation UNION {variant["anti_memorization"]["ranking_union_gap_pp"]:+.2f} points (train in-sample).',
            f'{name}526 descriptifs : content exact/UNION {variant["pilot"]["content"]["exact"]["hits"]}/{variant["pilot"]["content"]["union"]["hits"]} ; stack {variant["pilot"]["stack"]["exact"]["hits"]}/{variant["pilot"]["stack"]["union"]["hits"]}.']
    delta=value['deltas']['adapted'];family=value['family'];fd=family['deltas']['adapted']['union'] if family['deltas'] else None
    lines += [f'Δ UNION adapté−BASE {delta["union"]["delta_pp"]:+.3f} points, IC95 entreprises {delta["union"]["paired_company_bootstrap95_pp"]}.',
        f'Δ exact {delta["exact"]["delta_pp"]:+.3f} points, IC95 entreprises {delta["exact"]["paired_company_bootstrap95_pp"]}.',
        f'Même adresse : {family["queries"]} cas ; Δ UNION/IC95 {fd}.','## Écarts / choix',
        'Split entreprise initial80/20 conservé ; calibration16% totale hors gradient, évaluation et pilote hors gradient.',
        'Négatifs15 seulement pour InfoNCE ; le ranking requiert toujours ses100 candidats. Dernier checkpoint, aucun réglage par score.',
        '## Reste / risques','Amont déjà consommé/non full-OOF ; train in-sample optimiste ;526 descriptifs, aucune validation production.',
        'Recommandation unique : '+value['recommendation']]
    if len(lines)>30:raise ValueError('Report over30 lines')
    return '\n'.join(lines)+'\n'


def publication(out):
    plan=r.check(out);budget=read(OWN/'budget.json');value=read(out/'result.json');data=read(out/'data.json') if (out/'data.json').exists() else None
    value.update(compute_seconds=budget['compute_seconds'],metal_seconds=budget['metal_seconds'],wait_seconds=budget['wait_seconds'],wall_seconds=time.time()-budget['started_unix'],
        peak_bytes=max([read(p).get('peak_bytes',0) for p in OWN.glob('*/*/receipt.json')]+[read(p).get('peak_bytes',0) for p in OWN.glob('*/*/resource_stop.json')]+[0]),plan=record(out/'plan.json'))
    if (out/'cost_plan.json').exists():value['cost_plan']=read(out/'cost_plan.json')
    if data:value.update(budget_reduced=data['budget_reduced'],original_roles=data['original_role_counts'],skipped_training=data['skipped_training'])
    if (out/'verification.json').exists():value['verification']=record(out/'verification.json')
    write(out/'result.json',value);text=report(value);(out/'final_report.md').write_text(text)
    if plan['smoke']:return
    if data:
        for ref in data['sources']:verify(ref)
    paths=['docs/step2_gemma_finetune_10h.json','docs/step2_gemma_finetune_10h.md','handover.md']
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign dirty publication path: '+path)
    journal=ROOT/'handover.md';before=journal.read_text();write(ROOT/paths[0],value);DOC.write_text((out/'protocol.md').read_text()+'\n## Résultats\n\n'+text)
    line=f'\n- 2026-10-19 — Gemma avenant10h : {value["status"]}, calcul total{value["compute_seconds"]:.1f}s, Metal{value["metal_seconds"]:.1f}s, réduit{value.get("budget_reduced")}, rapport `{out}/final_report.md`, audit/SHA/mesures `docs/step2_gemma_finetune_10h.json`.15négatifs/1époque, aucun poids historique adapté/label/pool/jeu fermé/web/VM/push. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_gemma_finetune_10h.json`.\n'
    with journal.open('a') as stream:stream.write(line)
    if journal.read_text()!=before+line:raise ValueError('Concurrent journal edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','feat: publish ten-hour Gemma adaptation and paired holdout evaluation','--',*paths],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip();write(out/'publication.json',dict(commit=commit))


def worker(out,stage,seconds):
    r.check(out);directory=out/stage;metal=stage in ('train','base_encode','adapted_encode');guard=r.Guard(directory,metal);began=time.monotonic()
    if stage!='prepare':
        for ref in read(out/'prepare/receipt.json')['artifacts']:verify(ref)
    if stage=='prepare':value=prepare(out,directory,seconds)
    elif stage=='train':value=train(out,directory,seconds)
    elif stage in ('base_encode','adapted_encode'):value=r.encode(out,directory,stage=='adapted_encode',seconds)
    elif stage=='semantic':value=r.semantic(out,directory)
    elif stage in ('base','adapted'):value=r.evaluate(out,directory,stage)
    elif stage=='summarize':value=r.summarize(out,directory)
    elif stage=='verify':value=audit(out)
    else:raise ValueError('Unknown stage')
    if value==75:sys.exit(75)
    value.update(seconds=time.monotonic()-began,peak_bytes=guard.peak);write(directory/'receipt.json',value);guard.stop.set()


def supervise(resume=False):
    import psutil
    cfg=read(CONFIG)
    if not resume:
        if (OWN/'budget.json').exists():raise ValueError('Fresh namespace or explicit resume required')
        write(OWN/'budget.json',dict(started_unix=time.time(),last_unix=time.time(),compute_seconds=0.,metal_seconds=0.,wait_seconds=0.,history=[]))
    budget=read(OWN/'budget.json')
    def charge(wait=0.):
        now=time.time();budget['compute_seconds']+=max(0,now-budget['last_unix']-wait);budget['wait_seconds']+=wait;budget['last_unix']=now;write(OWN/'budget.json',budget)
    lease_path=OWN/'active.json'
    if resume and lease_path.exists():
        lease=read(lease_path)
        if not any(h.get('launch_id')==lease['launch_id'] for h in budget['history']):
            try:
                process=psutil.Process(lease['pid'])
                if abs(process.create_time()-lease['created_unix'])<2:raise ValueError('Owned child still active; no duplicate')
            except psutil.NoSuchProcess:pass
            if lease['metal']:budget['metal_seconds']+=max(0,time.time()-lease['started_unix'])
            budget['history'].append(dict(**lease,exit_code=None,reason='CONSERVATIVE_RECOVERY'))
    charge();completed=[];current=OWN/'mini'
    try:
        for mini in (True,False):
            current=OWN/('mini' if mini else 'full')
            if not current.exists():seal(current,mini)
            r.check(current)
            stages=['prepare','train','base_encode','adapted_encode','semantic','base','adapted','summarize','verify']
            for stage in stages:
                directory=current/stage;directory.mkdir(exist_ok=True)
                if (directory/'receipt.json').exists():
                    receipt=read(directory/'receipt.json')
                    for ref in receipt.get('artifacts',[])+receipt.get('shards',[]):verify(ref)
                    for key in ('result','checkpoint','binding'):
                        if key in receipt:verify(receipt[key])
                    completed.append(current.name+'/'+stage);continue
                attempts=0
                while True:
                    waiting=time.time();r.wait_resources(OWN);charge(wait=time.time()-waiting);r.check(current)
                    remaining=cfg['max_total_seconds']-budget['compute_seconds']-120
                    if remaining<60:raise TimeoutError('Ten-hour total compute budget exhausted')
                    write(OWN/'state.json',dict(status='RUNNING',phase=current.name,stage=stage,completed=completed,budget=budget,pid=os.getpid()))
                    metal=stage in ('train','base_encode','adapted_encode');start=time.time()
                    with (directory/'worker.log').open('a') as log:
                        child=subprocess.Popen([str(METAL if metal else CPU),'-u',str(SCRIPT),'worker','--output-dir',str(current),'--stage',stage,'--seconds',str(remaining)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        launch_id=str(time.time_ns());write(lease_path,dict(pid=child.pid,created_unix=psutil.Process(child.pid).create_time(),started_unix=time.time(),stage=stage,phase=current.name,metal=metal,launch_id=launch_id));reason=None
                        while child.poll() is None:
                            if budget['compute_seconds']+time.time()-budget['last_unix']>=cfg['max_total_seconds']-100:
                                reason='TOTAL_COMPUTE_DEADLINE';child.terminate();break
                            if r.resource_reason():reason='RESOURCE_YIELD_OWN_CHILD';child.terminate();break
                            if time.time()-budget['last_unix']>=30:charge()
                            time.sleep(1)
                        try:code=child.wait(timeout=45)
                        except subprocess.TimeoutExpired:child.kill();code=child.wait()
                    elapsed=time.time()-start
                    if metal:budget['metal_seconds']+=elapsed
                    budget['history'].append(dict(phase=current.name,stage=stage,seconds=elapsed,metal=metal,exit_code=code,reason=reason,launch_id=launch_id));charge()
                    if reason=='RESOURCE_YIELD_OWN_CHILD':code=75
                    if code==0:completed.append(current.name+'/'+stage);break
                    if code==75 and attempts<cfg['max_resource_restarts'] and not (directory/'resource_stop.json').exists():
                        attempts+=1;waiting=time.time();time.sleep(30);charge(wait=time.time()-waiting);continue
                    raise RuntimeError(f'{current.name}/{stage} failed exit={code} reason={reason}; {directory}/worker.log')
            charge()
            if mini:
                write(current/'state.json',dict(status='MINI_MEASURED_VERIFIED',budget=record(OWN/'budget.json')))
                print('MINI_MEASURED_VERIFIED',flush=True)
        charge();publication(current);charge()
        if budget['compute_seconds']>cfg['max_total_seconds']:raise TimeoutError('Publication exceeded total compute budget')
        write(OWN/'state.json',dict(status='COMPLETE',completed=completed,budget=budget,publication=read(current/'publication.json')))
        print('COMPLETE '+str(current/'final_report.md'),flush=True)
    except BaseException as exc:
        charge();write(OWN/'state.json',dict(status='STOP',phase=current.name,reason=str(exc),completed=completed,budget=budget))
        if (current/'result.json').exists() and 'variants' in read(current/'result.json'):
            value=read(current/'result.json');value.update(status='STOP_AFTER_MEASUREMENT',reason=str(exc));write(current/'result.json',value)
        else:write(current/'result.json',dict(status='STOP_INCOMPLETE',reason=str(exc),completed=completed,labels_changed=False,candidates_changed=False,production_allowed=False))
        try:
            if current.name=='mini':
                # Publish a failure even before a full plan could be admitted.
                target=OWN/'full'
                if not target.exists():seal(target,False)
                shutil.copyfile(current/'result.json',target/'result.json');current=target
            publication(current)
        except BaseException as publication_error:write(OWN/'publication_error.json',dict(reason=str(publication_error)))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','resume','worker']);p.add_argument('--output-dir',type=Path);p.add_argument('--stage');p.add_argument('--seconds',type=float,default=36000);a=p.parse_args()
    if a.action=='worker':
        if a.output_dir not in (OWN/'mini',OWN/'full'):raise ValueError('Owned namespace required')
        worker(a.output_dir,a.stage,a.seconds);return
    OWN.mkdir(parents=True,exist_ok=True)
    with (OWN/'supervisor.lock').open('a') as lock,(SOURCE/'supervisor.lock').open('a') as previous_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        supervise(a.action=='resume')


if __name__=='__main__':main()
