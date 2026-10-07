"""Dual-proof contrastive continuation, static TRAIN mining, no test/serving fit.

No downstream model changed. Multi-view TRAIN coverage is explicit and weighted
so a many-name company does not dominate through duplicated query examples.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v2_data as data
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v4'

INITIAL=io.HOME/'cached_20261007_a/best_adapter.pt'


def cached_objective(encode,qtexts,dtexts,mask,weights,temperature=.05,micro=2,scale=1.):
    import torch
    with torch.no_grad():
        q=torch.cat([encode(qtexts[i:i+micro]) for i in range(0,len(qtexts),micro)])
        d=torch.cat([encode(dtexts[i:i+micro]) for i in range(0,len(dtexts),micro)])
    q.requires_grad_(True); d.requires_grad_(True)
    value=objective(q,d,mask,weights,temperature)*scale
    if not torch.isfinite(value): raise RuntimeError('Non-finite dual loss')
    dq,dd=torch.autograd.grad(value,(q,d))
    for texts,grad in ((qtexts,dq),(dtexts,dd)):
        for i in range(0,len(texts),micro): encode(texts[i:i+micro]).backward(grad[i:i+micro])
    return float(value.detach())


def objective(q,d,mask,weights,temperature=.05):
    import torch
    scores=(q@d.T/temperature).masked_fill(torch.as_tensor(mask,device=q.device,dtype=torch.bool),-1e4)
    losses=torch.nn.functional.cross_entropy(scores,torch.arange(len(q),device=q.device),reduction='none')
    return (losses*torch.as_tensor(weights,device=q.device,dtype=losses.dtype)).mean()


def plan_rows(rows,documents,parents):
    return [(i,v,max(len(documents[r['positive']]['site_views']),len(parents[r['siren']])))
        for i,r in enumerate(rows) for v in range(max(len(documents[r['positive']]['site_views']),len(parents[r['siren']])))]


def training_texts(batch,docs,parents,task,epoch):
    q=[]; d=[]; doc_keys=[]; weights=[]
    for row,view,count in batch:
        key=row['positive']; doc_keys.append(key)
        views=docs[key]['site_views'] if task=='site' else parents[row['siren']]
        q.append(io.QUERY_PROMPT+row[task+'_query']); d.append(io.DOC_PROMPT+views[(view+epoch)%len(views)])
        weights.append(1/count)
    for row,view,count in batch:
        negatives=row['negatives'][:4] if task=='site' else [n for n in row['negatives'] if docs[n['key']]['siren']!=row['siren']][:1]
        for n in negatives:
            key=n['key']; doc_keys.append(key)
            views=docs[key]['site_views'] if task=='site' else parents[docs[key]['siren']]
            d.append(io.DOC_PROMPT+views[(view+epoch)%len(views)])
    mask=[]
    for i,(r,_,_) in enumerate(batch):
        positive=docs[r['positive']]; row=[]
        for j,key in enumerate(doc_keys):
            other=docs[key]
            blocked=(i!=j and key==positive['key']) or other['siren'] in positive['related_parents']
            if i!=j and other['siren']==r['siren']:
                blocked=blocked or task=='identity' or not data.eligible_other_site(r['crm'],positive,other)
            row.append(blocked if i!=j else False)
        mask.append(row)
    return q,d,mask,weights


def load_model(cfg,initial):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(cfg['seed'])
    torch.mps.set_per_process_memory_fraction(cfg['mps_memory_fraction'])
    config=AutoConfig.from_pretrained(io.HOME/'base',local_files_only=True); config.audio_config=None; config.vision_config=None
    if config.text_config.attention_dropout!=0: raise ValueError('Gradient cache needs deterministic encoder')
    model=AutoModel.from_pretrained(io.HOME/'base',config=config,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('mps')
    tokenizer=AutoTokenizer.from_pretrained(io.HOME/'base',local_files_only=True)
    for p in model.parameters(): p.requires_grad_(False)
    class LoRA(torch.nn.Module):
        def __init__(self,linear):
            super().__init__(); self.base=linear
            self.a=torch.nn.Parameter(torch.empty(8,linear.in_features,device='mps',dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(linear.out_features,8,device='mps',dtype=torch.float32))
        def forward(self,x): return self.base(x)+2*F.linear(F.linear(x.float(),self.a),self.b).to(x.dtype)
    if cfg['lora_rank']!=8 or cfg['lora_alpha']!=16: raise ValueError('Exact V1 adapter initialization')
    for name,module in list(model.named_modules()):
        if name.rsplit('.',1)[-1] in cfg['lora_modules']:
            parent,key=name.rsplit('.',1); setattr(model.get_submodule(parent),key,LoRA(module))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    original=torch.load(initial,map_location='cpu',weights_only=True)
    if set(original['adapter'])!=set(params): raise ValueError('Initial adapter mismatch')
    for name,p in params.items(): p.data.copy_(original['adapter'][name].to('mps'))
    def encode(texts):
        # Fail instead of silent tokenizer cropping; data pack guaranteed <=512.
        tokens=tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
        if tokens['input_ids'].shape[1]>cfg['max_tokens']: raise RuntimeError('Oversized unchecked view/query')
        tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')}
        hidden=model(**tokens).last_hidden_state.float(); mask=tokens['attention_mask'].unsqueeze(-1)
        return F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
    def inference(texts):
        model.eval(); chunks=[]
        with torch.inference_mode():
            for i in range(0,len(texts),8): chunks.append(encode(texts[i:i+8]).cpu().numpy())
        torch.mps.empty_cache(); return np.concatenate(chunks)
    return model,params,encode,inference,tokenizer


def inputs(root):
    manifest=io.read(root/'manifest.json'); cfg=io.read(verify(manifest['config']))
    if manifest['status']!='SEALED_V2_MULTIVIEW_OFFICIAL_TRAIN_VALIDATION_NO_TEST': raise ValueError('Complete data seal')
    artifacts=manifest['artifacts']
    return manifest,cfg,io.read(verify(artifacts['documents.json'])),io.read(verify(artifacts['parent_views.json'])),artifacts


def choose_negatives(row,dense,docs):
    lexical=[n for n in row['negatives'] if n['kind']=='LEXICAL_DIFFERENT_PARENT']
    sibling=[n for n in row['negatives'] if n['kind']=='SAME_PARENT_PROVEN_OTHER_SITE']
    chosen=lexical[:2]+([dense] if dense else [])+sibling[:1]
    for n in lexical[2:]+sibling[1:]:
        if len(chosen)<4: chosen.append(n)
    unique={n['key']:n for n in chosen}
    return list(unique.values())[:4]


class ResourceYield(RuntimeError):
    pass


def resource_guard(deadline, rss_limit=8*1024**3):
    import psutil
    free,level=io.pressure()
    if time.time()>=deadline:
        raise TimeoutError('NIGHT_DEADLINE')
    if free<20 or level not in (0,1,2) or psutil.Process().memory_info().rss>rss_limit:
        raise ResourceYield(f'RESOURCE_YIELD free={free} pressure={level} rss={psutil.Process().memory_info().rss}')


def mine(root,out,deadline):
    """Restartable exact static TRAIN miner. Every flushed shard is hash-bound."""
    manifest,cfg,docs,parents,artifacts=inputs(root); del parents
    out.mkdir(parents=True,exist_ok=True)
    rows=io.read(verify(artifacts['train.json'])); full_bank=io.read(verify(artifacts['train_bank.json']))
    zones={r['crm']['crm_insee'] for r in rows if r['crm']['crm_insee']}
    bank=[k for k in full_bank if docs[k]['insee'] in zones]
    if {docs[k]['siren'] for k in bank}&{r['siren'] for r in io.read(verify(artifacts['validation.json']))}:
        raise ValueError('Mining leaked validation')
    initial=record(INITIAL); code=record(Path(__file__))
    texts=list(dict.fromkeys(io.DOC_PROMPT+docs[k]['site_views'][0] for k in bank))
    binding=dict(data=record(root/'manifest.json'),initial=initial,code=code,
        text_order_sha256=hashlib.sha256('\n'.join(texts).encode()).hexdigest(),total=len(texts))
    binding_path=out/'binding.json'
    if binding_path.exists():
        if io.read(binding_path)!=binding: raise ValueError('Mining resume binding mismatch')
    else: io.write(binding_path,binding)
    model,params,encode,inference,tokenizer=load_model(cfg,verify(initial)); start=time.monotonic()
    shard_refs=[]
    for i in range(0,len(texts),128):
        path=out/f'vectors_{i:07d}.npy'; refpath=path.with_suffix('.json')
        if refpath.exists():
            ref=io.read(refpath); verify(ref)
            values=np.load(path,allow_pickle=False)
            if values.shape!=(min(128,len(texts)-i),768) or not np.isfinite(values).all(): raise ValueError('Invalid mining shard')
        else:
            resource_guard(deadline,cfg['max_rss_bytes'])
            values=inference(texts[i:i+128])
            if not np.isfinite(values).all(): raise ValueError('Non-finite mining vectors')
            temp=path.with_suffix('.npy.tmp')
            with temp.open('wb') as f: np.save(f,values,allow_pickle=False)
            os.replace(temp,path); io.write(refpath,record(path))
        shard_refs.append(io.read(refpath))
        io.write(out/'state.json',dict(status='STATIC_DENSE_TRAIN_MINING',encoded=min(i+128,len(texts)),total=len(texts),
            deadline_unix=deadline,resumable=True))
    # Same exact search support as V2; vectors in disk-backed array, not duplicated in RAM.
    matrix_path=out/'matrix.npy'
    matrix=np.lib.format.open_memmap(matrix_path,mode='w+',dtype=np.float32,shape=(len(texts),768))
    for i,ref in enumerate(shard_refs):
        values=np.load(verify(ref),allow_pickle=False); matrix[i*128:i*128+len(values)]=values
    matrix.flush()
    text_index={t:i for i,t in enumerate(texts)}; by_zone=defaultdict(list)
    for k in bank:
        if docs[k]['insee']: by_zone[docs[k]['insee']].append(k)
    positions={z:np.asarray([text_index[io.DOC_PROMPT+docs[k]['site_views'][0]] for k in keys],dtype=np.int64)
        for z,keys in by_zone.items()}
    results=[]; mined=0
    for i in range(0,len(rows),128):
        resource_guard(deadline,cfg['max_rss_bytes'])
        batch=rows[i:i+128]; qe=inference([io.QUERY_PROMPT+r['site_query'] for r in batch])
        for q,r in zip(qe,batch):
            pos=docs[r['positive']]; keys=by_zone.get(r['crm']['crm_insee'],[]); n=None
            if keys:
                scores=matrix[positions[r['crm']['crm_insee']]]@q; order=np.argsort(-scores,kind='stable')
                existing={x['key'] for x in r['negatives']}
                for loc in order:
                    key=keys[loc]; other=docs[key]
                    if key==r['positive'] or key in existing or other['siren'] in pos['related_parents']: continue
                    if other['siren']==r['siren'] and not data.eligible_other_site(r['crm'],pos,other): continue
                    n=dict(key=key,kind='DENSE_TRAIN_OTHER_SITE' if other['siren']==r['siren'] else 'DENSE_TRAIN_DIFFERENT_PARENT',
                        proven_other_site=other['siren']==r['siren']); break
            mined+=n is not None; results.append({**r,'negatives':choose_negatives(r,n,docs)})
        io.write(out/'state.json',dict(status='DENSE_NEGATIVES_SELECTED',queries=len(results),total=len(rows),resumable=True))
    verify(initial); verify(code); io.write(out/'train.json',results)
    io.write(out/'manifest.json',dict(status='SEALED_STATIC_DENSE_TRAIN_NEGATIVES_NO_RETRIEVAL_RECALL',data=record(root/'manifest.json'),
        initial_adapter=initial,implementation=code,rows=len(rows),train=record(out/'train.json'),
        rows_with_added_dense_negative=mined,native_training_bank_sites=len(bank),mining_first_packed_view_only=True,
        vector_shards=shard_refs,binding=record(binding_path),mining_is_not_full_corpus_retrieval_evaluation=True,
        test_opened=False,downstream_models_changed=False,seconds_this_attempt=time.monotonic()-start))

def prepare(root,mined,out,deadline=None):
    manifest,cfg,docs,parents,artifacts=inputs(root); m=io.read(mined/'manifest.json')
    if m['data']!=record(root/'manifest.json') or m['status']!='SEALED_STATIC_DENSE_TRAIN_NEGATIVES_NO_RETRIEVAL_RECALL': raise ValueError('Mining/data binding')
    rows=io.read(verify(m['train'])); plan=plan_rows(rows,docs,parents)
    if len(plan)>200000: raise RuntimeError('STOP_MULTIVIEW_EXPOSURE_BUDGET_NOT_SILENT_SUBSAMPLE')
    out.mkdir(parents=True,exist_ok=False)
    io.write(out/'launch.json',dict(data=record(root/'manifest.json'),mining=record(mined/'manifest.json'),initial=record(INITIAL),
        implementation=record(Path(__file__)),data_implementation=record(Path(data.__file__)),
        plan_examples_per_epoch=len(plan),original_query_rows=len(rows),weighted_multi_view_repetition=True,
        created_unix=time.time(),deadline_unix=min(time.time()+cfg['max_hours']*3600,deadline or float('inf')),test_opened=False,downstream_models_changed=False))


def worker(out,preflight=False):
    import signal
    requested_stop=False
    def request_stop(signum,frame):
        nonlocal requested_stop
        requested_stop=True
    signal.signal(signal.SIGTERM,request_stop)
    import psutil
    import torch
    launch=io.read(out/'launch.json'); verify(launch['implementation']); verify(launch['data_implementation'])
    manifest,cfg,docs,parents,artifacts=inputs(Path(verify(launch['data'])).parent)
    mined=io.read(verify(launch['mining'])); rows=io.read(verify(mined['train'])); validation=io.read(verify(artifacts['validation.json']))
    if {r['siren'] for r in rows}&{r['siren'] for r in validation}: raise ValueError('Entity overlap')
    plan=plan_rows(rows,docs,parents); expected=len(plan)/len(rows)
    controls=sorted(validation,key=lambda r:hashlib.sha256(('gemma2-v2-control|'+r['query_id']).encode()).digest())[:cfg['validation_queries']]
    model,params,encode,inference,tokenizer=load_model(cfg,verify(launch['initial']))
    optimizer=torch.optim.AdamW(params.values(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    step=0; epoch=0; cursor=0; best=float('inf'); stale=0; exposed=defaultdict(set); own=out/'checkpoint.pt'
    if own.exists() and not preflight:
        state=torch.load(own,map_location='cpu',weights_only=True)
        if state['launch_sha256']!=io.digest(out/'launch.json'): raise ValueError('Resume binding')
        for n,p in params.items(): p.data.copy_(state['adapter'][n].to('mps'))
        optimizer.load_state_dict(state['optimizer']); step=state['step']; epoch=state['epoch']; cursor=state['next_offset']
        best=state['best']; stale=state['stale']; exposed.update({k:set(v) for k,v in state['exposed'].items()})
    def payload():
        # Optimizer GPU tensors copied to CPU before atomic checkpoint.
        def cpu(v):
            if isinstance(v,torch.Tensor): return v.detach().cpu().clone()
            if isinstance(v,dict): return {k:cpu(x) for k,x in v.items()}
            if isinstance(v,list): return [cpu(x) for x in v]
            return v
        return dict(adapter=cpu(params),optimizer=cpu(optimizer.state_dict()),step=step,epoch=epoch,next_offset=cursor,
            best=best,stale=stale,exposed={k:sorted(v) for k,v in exposed.items()},launch_sha256=io.digest(out/'launch.json'))
    def save(name='checkpoint.pt'):
        temp=out/(name+'.tmp'); torch.save(payload(),temp); os.replace(temp,out/name)
    def measure():
        model.eval(); losses=Counter(); batches=0
        with torch.no_grad():
            for i in range(0,len(controls),cfg['batch_size']):
                batch=[(r,0,1) for r in controls[i:i+cfg['batch_size']]]
                for task in ('site','identity'):
                    q,d,mask,weights=training_texts(batch,docs,parents,task,0)
                    qe=torch.cat([encode(q[j:j+2]) for j in range(0,len(q),2)]); de=torch.cat([encode(d[j:j+2]) for j in range(0,len(d),2)])
                    losses[task]+=float(objective(qe,de,mask,weights,cfg['temperature']))
                batches+=1
        model.train(); torch.mps.empty_cache()
        value=sum(cfg[t+'_loss_weight']*losses[t]/batches for t in ('site','identity'))
        if not math.isfinite(value): raise ValueError('Invalid validation')
        return value,{t:losses[t]/batches for t in ('site','identity')}
    if preflight:
        # A genuine worst-long-view backward, not just an inference smoke.
        selected=sorted(plan,key=lambda x:max(len(tokenizer(io.DOC_PROMPT+docs[rows[x[0]]['positive']]['site_views'][x[1]%len(docs[rows[x[0]]['positive']]['site_views'])])['input_ids']),
            len(tokenizer(io.DOC_PROMPT+parents[rows[x[0]]['siren']][x[1]%len(parents[rows[x[0]]['siren']])])['input_ids'])),reverse=True)[:16]
        batches=[[(rows[i],v,n) for i,v,n in selected],[(rows[i],v,n) for i,v,n in plan[:16]]]
        started=time.monotonic(); values=[]
        for batch in batches:
            optimizer.zero_grad(set_to_none=True); loss=0.
            for task in ('site','identity'):
                q,d,mask,weights=training_texts(batch,docs,parents,task,0)
                loss+=cached_objective(encode,q,d,mask,weights,cfg['temperature'],2,cfg[task+'_loss_weight']*expected)
            norm=torch.nn.utils.clip_grad_norm_(params.values(),cfg['gradient_clip'])
            if not torch.isfinite(norm): raise ValueError('Non-finite preflight gradient')
            optimizer.step(); values.append(loss); torch.mps.empty_cache()
        free,level=io.pressure()
        if free<cfg['yield_free_percent'] or level not in (0,1,2) or psutil.Process().memory_info().rss>cfg['max_rss_bytes']: raise RuntimeError('Preflight resource gate')
        io.write(out/'preflight.json',dict(status='PASS_REAL_WORST_LENGTH_DUAL_LOSS_BACKWARD_NO_CHECKPOINT_REUSE',steps=2,losses=values,
            seconds=time.monotonic()-started,mps_driver_bytes=torch.mps.driver_allocated_memory(),rss_bytes=psutil.Process().memory_info().rss,
            free_percent=free,actual_batch16_micro2=True,training_changed_by_preflight=False,test_opened=False))
        return 0
    preflight_result=io.read(out/'preflight.json'); exit_status=io.read(out/'preflight_exit.json')
    if preflight_result['status']!='PASS_REAL_WORST_LENGTH_DUAL_LOSS_BACKWARD_NO_CHECKPOINT_REUSE' or exit_status['exit_code']!=0:
        raise ValueError('Actual long-view preflight completion required')
    if step==0:
        best,components=measure(); save('best_adapter.pt'); save()
        io.write(out/'baseline_v1_on_v2_validation.json',dict(loss=best,components=components,controls=len(controls),retrieval_recall_measured=False))
    total_steps=math.ceil(len(plan)/cfg['batch_size'])*cfg['epochs']; warmup=max(1,round(total_steps*cfg['warmup_fraction'])); process=psutil.Process()
    # Do not terminate before one full pass exposing all prepared positive views.
    def early_stop(): return stale>=cfg['early_stop_checks'] and epoch>=1
    while epoch<cfg['epochs'] and time.time()<launch['deadline_unix'] and not early_stop():
        order=list(range(len(plan))); random.Random(cfg['seed']+epoch).shuffle(order)
        while cursor<len(order) and time.time()<launch['deadline_unix'] and not early_stop():
            if requested_stop:
                save(); io.write(out/'state.json',dict(status='YIELD_TERMINATION_CHECKPOINTED',step=step,epoch=epoch)); return 75
            free,level=io.pressure()
            if free<cfg['yield_free_percent'] or level not in (0,1,2) or process.memory_info().rss>cfg['max_rss_bytes']:
                save(); io.write(out/'state.json',dict(status='YIELD_RESOURCE',step=step,free_percent=free)); return 75
            chosen=[plan[order[j]] for j in range(cursor,min(cursor+cfg['batch_size'],len(order)))]; batch=[(rows[i],v,n) for i,v,n in chosen]
            if step<warmup: multiplier=(step+1)/warmup
            else:
                phase=min(1.,(step-warmup)/max(1,total_steps-warmup)); multiplier=cfg['final_lr_fraction']+(1-cfg['final_lr_fraction'])*.5*(1+math.cos(math.pi*phase))
            for group in optimizer.param_groups: group['lr']=cfg['learning_rate']*multiplier
            optimizer.zero_grad(set_to_none=True); loss=0.
            for task in ('site','identity'):
                q,d,mask,weights=training_texts(batch,docs,parents,task,epoch)
                loss+=cached_objective(encode,q,d,mask,weights,cfg['temperature'],cfg['microbatch_size'],cfg[task+'_loss_weight']*expected)
            norm=torch.nn.utils.clip_grad_norm_(params.values(),cfg['gradient_clip'])
            if not torch.isfinite(norm): raise ValueError('Non-finite gradient')
            optimizer.step(); step+=1; cursor+=len(chosen); torch.mps.empty_cache()
            for row,v,n in batch:
                exposed['site:'+row['positive']].add((v+epoch)%len(docs[row['positive']]['site_views']))
                exposed['identity:'+row['siren']].add((v+epoch)%len(parents[row['siren']]))
            if step%cfg['validation_every_steps']==0:
                measured,components=measure(); improved=measured<best-1e-5; stale=0 if improved else stale+1
                if improved: best=measured; save('best_adapter.pt')
                io.write(out/'last_validation.json',dict(step=step,loss=measured,components=components,best=best,stale=stale,controls=len(controls)))
            if step%10==0 or step<=3: save()
            if step%10==0 or step<=3:
                io.write(out/'state.json',dict(status='TRAINING',step=step,epoch=epoch,next_offset=cursor,loss=loss,best_validation_loss=best,
                    plan_examples=len(plan),learning_rate=optimizer.param_groups[0]['lr'],rss_bytes=process.memory_info().rss,
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),free_percent=free,test_opened=False,downstream_models_changed=False))
        if cursor>=len(order): epoch+=1; cursor=0; save()
    measured,components=measure()
    if measured<best: best=measured; save('best_adapter.pt')
    wanted={**{'site:'+r['positive']:len(docs[r['positive']]['site_views']) for r in rows},**{'identity:'+r['siren']:len(parents[r['siren']]) for r in rows}}
    viewed=sum(len(exposed[k]) for k in wanted); expected_views=sum(wanted.values()); coverage=viewed/expected_views
    status='FIT_COMPLETE' if epoch==cfg['epochs'] else 'EARLY_STOP' if early_stop() else 'TIME_BUDGET_CHECKPOINTED'
    save(); io.write(out/'state.json',dict(status=status,step=step,epoch=epoch,best_validation_loss=best,final_validation_loss=measured,
        view_exposure_fraction=coverage,all_training_positive_views_exposed=coverage>=cfg['minimum_view_exposure_fraction'],
        validation_controls=len(controls),recall_measured=False,test_opened=False,downstream_models_changed=False))
    return 0


def preflight(out):
    with (out/'preflight.log').open('x') as log:
        child=subprocess.run([sys.executable,'-u','-m','scripts.embeddinggemma2_v4_fit','preflight-worker','--output',str(out)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    io.write(out/'preflight_exit.json',dict(own_child_exit_confirmed=True,exit_code=child.returncode)); child.check_returncode()


def supervise(out):
    import psutil
    launch=io.read(out/'launch.json'); manifest=io.read(verify(launch['data'])); cfg=io.read(verify(manifest['config'])); history=[]; restarts=0
    while time.time()<launch['deadline_unix']:
        free,level=io.pressure()
        if free<cfg['admit_free_percent'] or level not in (0,1,2):
            io.write(out/'supervisor.json',dict(status='WAITING_RESOURCES',free_percent=free,history=history)); time.sleep(30); continue
        with (out/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_v4_fit','worker','--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
            io.write(out/'supervisor.json',dict(status='CHILD_RUNNING',pid=os.getpid(),child_pid=child.pid,history=history,restarts=restarts)); reason=None
            while child.poll() is None:
                free,level=io.pressure()
                try: rss=psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess: break
                if free<cfg['supervisor_free_percent'] or level not in (0,1,2) or rss>cfg['max_rss_bytes'] or time.time()>launch['deadline_unix']:
                    reason=dict(free_percent=free,pressure_level=level,rss_bytes=rss,deadline_reached=time.time()>launch['deadline_unix']); child.terminate()
                    try: child.wait(timeout=45)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    break
                time.sleep(1)
            code=child.wait(); history.append(dict(pid=child.pid,exit_code=code,reason=reason,ended_unix=time.time()))
        if code==0:
            io.write(out/'supervisor.json',dict(status='COMPLETE',exit_code=0,history=history)); return
        if (reason is None and code!=75) or restarts>=cfg['max_resource_restarts']:
            io.write(out/'supervisor.json',dict(status='STOP',exit_code=code,history=history)); return
        restarts+=1; time.sleep(30)
    io.write(out/'supervisor.json',dict(status='BUDGET_EXHAUSTED',history=history))


def continuation(source,out):
    """Resource-only migration: preserve optimizer, cursor, weights, split, deadline."""
    import torch
    out.mkdir(parents=True,exist_ok=False)
    original=source/'fit'; old_launch=io.read(original/'launch.json')
    old_data=io.read(verify(old_launch['data'])); old_cfg=io.read(verify(old_data['config']))
    cfg={**old_cfg,'admit_free_percent':30,'yield_free_percent':20,'supervisor_free_percent':15}
    io.write(out/'config.json',cfg)
    dataset=out/'data'; dataset.mkdir()
    new_data={**old_data,'config':record(out/'config.json'),'resource_only_source_manifest':old_launch['data']}
    io.write(dataset/'manifest.json',new_data)
    mined=io.read(verify(old_launch['mining']))
    if mined['data']!=old_launch['data']: raise ValueError('Original mining data mismatch')
    verify(mined['train'])
    fit_root=out/'fit'; fit_root.mkdir()
    source_checkpoint=record(original/'checkpoint.pt')
    checkpoint=torch.load(verify(source_checkpoint),map_location='cpu',weights_only=True)
    if checkpoint['launch_sha256']!=io.digest(original/'launch.json'): raise ValueError('Original checkpoint binding')
    launch={**old_launch,'data':record(dataset/'manifest.json'),'implementation':record(Path(__file__)),
        'continuation':dict(source_launch=record(original/'launch.json'),source_checkpoint=source_checkpoint,
            source_best=record(original/'best_adapter.pt'),only_resource_policy_and_checkpoint_frequency_changed=True)}
    io.write(fit_root/'launch.json',launch)
    for name in ('checkpoint.pt','best_adapter.pt'):
        state=torch.load(original/name,map_location='cpu',weights_only=True)
        if state['launch_sha256']!=io.digest(original/'launch.json'): raise ValueError('Source state binding')
        state['launch_sha256']=io.digest(fit_root/'launch.json')
        temporary=fit_root/(name+'.tmp'); torch.save(state,temporary); os.replace(temporary,fit_root/name)
    night=io.read(source/'night.json')
    io.write(out/'night.json',{**night,'resource_only_source_night':record(source/'night.json'),'implementation':record(Path(__file__))})
    io.write(out/'migration.json',dict(source=record(original/'launch.json'),checkpoint=source_checkpoint,
        source_preflight=record(original/'preflight.json'),step=checkpoint['step'],epoch=checkpoint['epoch'],
        next_offset=checkpoint['next_offset'],optimizer_preserved=True,deadline_preserved=launch['deadline_unix'],
        data_artifacts_unchanged=new_data['artifacts']==old_data['artifacts'],recipe_unchanged=True))
    io.write(out/'state.json',dict(status='PREFLIGHT_CONTINUATION',source_step=checkpoint['step'],test_opened=False))
    try:
        preflight(fit_root)
        io.write(out/'state.json',dict(status='TRAINING_AFTER_REAL_PREFLIGHT',test_opened=False))
        supervise(fit_root)
        supervision=io.read(fit_root/'supervisor.json')
        if supervision['status']!='COMPLETE': raise RuntimeError('Continuation supervisor '+supervision['status'])
        io.write(out/'state.json',dict(status='FIT_DONE_EVALUATION_PENDING',training=record(fit_root/'state.json'),
            deadline_unix=night['deadline_unix'],test_opened=False,downstream_models_changed=False))
    except BaseException as exc:
        io.write(out/'state.json',dict(status='NIGHT_INTERRUPTED',reason=str(exc),test_opened=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['continue-night','preflight-worker','worker','run'])
    p.add_argument('--source',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V4 namespace')
    if a.action=='continue-night': continuation(a.source,a.output)
    elif a.action=='preflight-worker': sys.exit(worker(a.output,True))
    elif a.action=='worker': sys.exit(worker(a.output))
    else: supervise(a.output)

