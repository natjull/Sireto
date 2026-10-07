"""Gradient-cache continuation: same contrastive objective, bounded activations.

Original night artifacts are read-only. Final test is never opened here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as old

BATCH = 4
MICRO = 2
RSS_LIMIT = 5*1024**3


def contrastive(q, d, sirens, owners):
    import torch
    scores=q@d.T/.05
    mask=torch.tensor([[s==o and i!=j for j,o in enumerate(owners)] for i,s in enumerate(sirens)],
        device=q.device,dtype=torch.bool)
    return torch.nn.functional.cross_entropy(scores.masked_fill(mask,-1e4),torch.arange(len(q),device=q.device))


def cached_backward(encode, qtexts, dtexts, sirens, owners, micro=MICRO):
    """Exact chain rule for a deterministic encoder, no accumulated small-batch loss."""
    import torch
    with torch.no_grad():
        q=torch.cat([encode(qtexts[i:i+micro]) for i in range(0,len(qtexts),micro)])
        d=torch.cat([encode(dtexts[i:i+micro]) for i in range(0,len(dtexts),micro)])
    q.requires_grad_(True); d.requires_grad_(True)
    loss=contrastive(q,d,sirens,owners)
    if not torch.isfinite(loss): raise RuntimeError('Non-finite contrastive loss')
    dq,dd=torch.autograd.grad(loss,(q,d))
    for texts, grads in ((qtexts,dq),(dtexts,dd)):
        for i in range(0,len(texts),micro):
            encoded=encode(texts[i:i+micro]); encoded.backward(grads[i:i+micro])
    return float(loss.detach())


def training_texts(batch):
    q=[old.QUERY_PROMPT+r['query'] for r in batch]
    d=[old.DOC_PROMPT+r['positive'] for r in batch]; owners=[r['siren'] for r in batch]
    for r in batch:
        for n in r['negatives'][:2]: d.append(old.DOC_PROMPT+n['text']); owners.append(n['siren'])
    return q,d,[r['siren'] for r in batch],owners


def worker(out, max_steps=None):
    import psutil
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    launch=old.read(out/'launch.json'); source=Path(launch['source'])
    for file,sha in launch['inputs'].items():
        if old.digest(source/file)!=sha: raise RuntimeError('Source dataset/checkpoint drift')
    if old.digest(__file__)!=launch['implementation_sha256']: raise RuntimeError('Implementation drift')
    manifest=old.read(source/'manifest.json')
    rows=old.read(source/'train.json'); validation=old.read(source/'validation.json')
    assert not {r['siren'] for r in rows}&{r['siren'] for r in validation}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261006)
    torch.mps.set_per_process_memory_fraction(.25)  # ~4.4 GiB hard cap, never unlimited.
    config=AutoConfig.from_pretrained(old.HOME/'base',local_files_only=True)
    config.audio_config=None; config.vision_config=None
    if config.text_config.attention_dropout!=0: raise ValueError('Caching requires deterministic forward')
    model=AutoModel.from_pretrained(old.HOME/'base',config=config,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('mps')
    tokenizer=AutoTokenizer.from_pretrained(old.HOME/'base',local_files_only=True)
    for p in model.parameters(): p.requires_grad_(False)
    class LoRA(torch.nn.Module):
        def __init__(self,linear):
            super().__init__(); self.base=linear
            self.a=torch.nn.Parameter(torch.empty(8,linear.in_features,device='mps',dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(linear.out_features,8,device='mps',dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):
            y=self.base(x)
            return y+2*F.linear(F.linear(x.float(),self.a),self.b).to(y.dtype)
    for name,module in list(model.named_modules()):
        if name.rsplit('.',1)[-1] in ('q_proj','v_proj','embedding_projection'):
            parent,key=name.rsplit('.',1); setattr(model.get_submodule(parent),key,LoRA(module))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(params.values(),lr=5e-5,weight_decay=.01)
    own=out/'checkpoint.pt'; state=torch.load(own if own.exists() else source/'checkpoint.pt',
        map_location='cpu',weights_only=True)
    if set(state['adapter'])!=set(params): raise RuntimeError('Adapter namespace mismatch')
    for n,p in params.items(): p.data.copy_(state['adapter'][n].to('mps'))
    optimizer.load_state_dict(state['optimizer'])
    step=state['step']; epoch=state['epoch']; best=state['best']
    cursor=state.get('next_offset',(step % math.ceil(len(rows)/BATCH))*BATCH)
    def encode(texts):
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=160,return_tensors='pt')
        tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')}
        hidden=model(**tokens).last_hidden_state.float(); mask=tokens['attention_mask'].unsqueeze(-1)
        return F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
    controls=sorted((r for r in validation if r['negatives']),
        key=lambda r:hashlib.sha256(r['query_id'].encode()).digest())[:64]
    if len(controls)!=64: raise RuntimeError('Original fixed validation controls required')
    def evaluate():
        model.eval(); values=[]
        with torch.no_grad():
            for row in controls:
                q,d,s,o=training_texts([row]); qe=encode(q)
                de=torch.cat([encode(d[i:i+MICRO]) for i in range(0,len(d),MICRO)])
                values.append(float(contrastive(qe,de,s,o)))
        model.train(); torch.mps.empty_cache()
        if not all(math.isfinite(v) for v in values): raise RuntimeError('Non-finite validation')
        return sum(values)/len(values)
    def cpu(value):
        if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list): return [cpu(v) for v in value]
        if isinstance(value,tuple): return tuple(cpu(v) for v in value)
        return value
    def save(file='checkpoint.pt'):
        payload=dict(adapter=cpu(params),optimizer=cpu(optimizer.state_dict()),step=step,epoch=epoch,
            next_offset=cursor,best=best,random=random.getstate(),launch_sha256=old.digest(out/'launch.json'))
        temp=out/(file+'.tmp'); torch.save(payload,temp); os.replace(temp,out/file)
    start=time.monotonic(); deadline=launch['deadline_unix']; initial_step=step
    recent=None; validation_loss=None; process=psutil.Process()
    save()
    while epoch<2 and time.time()<deadline:
        order=list(range(len(rows))); random.Random(20261006+epoch).shuffle(order)
        while cursor<len(order) and time.time()<deadline:
            free,level=old.pressure()
            if free<50 or level!=1 or process.memory_info().rss>RSS_LIMIT:
                save(); old.write(out/'state.json',dict(status='YIELD_RESOURCE',step=step,free_percent=free)); return 75
            batch=[rows[i] for i in order[cursor:cursor+BATCH]]
            optimizer.zero_grad(set_to_none=True)
            recent=cached_backward(encode,*training_texts(batch))
            norm=torch.nn.utils.clip_grad_norm_(params.values(),1.)
            if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient')
            optimizer.step(); step+=1; cursor+=len(batch); torch.mps.empty_cache()
            if step%100==0:
                validation_loss=evaluate()
                if validation_loss<best: best=validation_loss; save('best_adapter.pt')
            if step%25==0 or step-initial_step<=5: save()
            if step%10==0 or step-initial_step<=5:
                old.write(out/'state.json',dict(status='TRAINING',pid=os.getpid(),step=step,epoch=epoch,
                    next_offset=cursor,loss=recent,validation_loss=validation_loss,best_validation_loss=best,
                    elapsed_seconds=time.monotonic()-start,steps_this_process=step-initial_step,
                    trainable_parameters=sum(p.numel() for p in params.values()),rss_bytes=process.memory_info().rss,
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),free_percent=free,
                    batch_size=BATCH,microbatch_size=MICRO,recall_measured=False,test_opened=False))
            if max_steps is not None and step-initial_step>=max_steps:
                save(); return 0
        if cursor>=len(order): epoch+=1; cursor=0; save()
    validation_loss=evaluate()
    if validation_loss<best: best=validation_loss; save('best_adapter.pt')
    save(); old.write(out/'state.json',dict(status='FIT_COMPLETE' if epoch==2 else 'TIME_BUDGET_CHECKPOINTED',
        step=step,epoch=epoch,best_validation_loss=best,final_validation_loss=validation_loss,
        recall_measured=False,test_opened=False,downstream_models_changed=False))
    return 0


def supervisor(out):
    import psutil
    launch=old.read(out/'launch.json'); restarts=0; history=[]
    while time.time()<launch['deadline_unix']:
        free,level=old.pressure()
        if free<65 or level!=1:
            old.write(out/'supervisor.json',dict(status='WAITING_RESOURCES',pid=os.getpid(),free_percent=free))
            time.sleep(30); continue
        with (out/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_cached','worker',
                '--output',str(out)],cwd=old.ROOT,stdout=log,stderr=subprocess.STDOUT)
            old.write(out/'supervisor.json',dict(status='CHILD_RUNNING',pid=os.getpid(),child_pid=child.pid,
                restarts=restarts,history=history))
            reason=None
            while child.poll() is None:
                free,level=old.pressure()
                try: rss=psutil.Process(child.pid).memory_info().rss
                except psutil.NoSuchProcess: break
                if free<45 or level!=1 or rss>RSS_LIMIT or time.time()>launch['deadline_unix']:
                    reason='RESOURCE_OR_DEADLINE'; child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    break
                time.sleep(1)
            code=child.wait(); history.append(dict(pid=child.pid,exit_code=code,reason=reason,ended_unix=time.time()))
        if code==0:
            old.write(out/'supervisor.json',dict(status='COMPLETE',exit_code=0,history=history)); return
        if reason is None and code!=75 or restarts>=12:
            old.write(out/'supervisor.json',dict(status='STOP',exit_code=code,history=history)); return
        restarts+=1; time.sleep(30)
    old.write(out/'supervisor.json',dict(status='BUDGET_EXHAUSTED',history=history))


def prepare(out, source, hours):
    out.mkdir(parents=True,exist_ok=False)
    source=source.resolve()
    if source!=old.HOME/'night_20261006_a': raise ValueError('Only sealed original TRAIN continuation')
    names=['manifest.json','train.json','validation.json','baseline_diagnostic.json','checkpoint.pt']
    old.write(out/'launch.json',dict(source=str(source),inputs={n:old.digest(source/n) for n in names},
        implementation_sha256=old.digest(__file__),created_unix=time.time(),deadline_unix=time.time()+hours*3600,
        final_test_opened=False,downstream_models_changed=False,revision=old.REVISION))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','worker','run'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--hours',type=float,default=8)
    p.add_argument('--max-steps',type=int); args=p.parse_args()
    if not args.output.resolve().is_relative_to(old.HOME): raise ValueError('Dedicated sidequest only')
    if args.action=='prepare': prepare(args.output,old.HOME/'night_20261006_a',args.hours)
    elif args.action=='worker': sys.exit(worker(args.output,args.max_steps))
    else: supervisor(args.output)
