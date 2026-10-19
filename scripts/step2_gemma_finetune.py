"""One preregistered company holdout, fresh LoRA, frozen G0/V142 candidates.

Sequential resumable workers; no historical adapted weights or reserved labels.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
OWN = HOME/'step2_gemma_finetune_20261019'
PREVIOUS = HOME/'step2_colocated_signals_20261019/full'
DOC = ROOT/'docs/step2_gemma_finetune.md'
CONFIG = ROOT/'config/step2_gemma_finetune.json'
CPU = ROOT/'.venv/bin/python'
METAL = HOME/'runtime/bin/python'
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); os.replace(temp, path)


def record(path):
    path = Path(path); before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('Source changed while hashing: '+str(path))
    return dict(path=str(path), size_bytes=after.st_size, sha256=digest.hexdigest())


def verify(ref):
    value = record(ref['path'])
    if any(value[k] != ref[k] for k in ('sha256', 'size_bytes')): raise ValueError('Seal mismatch: '+ref['path'])
    return Path(ref['path'])


def check(out):
    plan = read(out/'plan.json')
    for ref in plan['implementations']+[plan['config'], plan['protocol']]+plan['base_files']: verify(ref)
    return plan


def resource_reason():
    import psutil
    vm = psutil.virtual_memory(); found = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = (p.info['name'] or '').lower(); args = p.info['cmdline'] or []
            if any(Path(a).name == 'train3000_frozen_v142_supervisor.py' for a in args):
                found += [(c.pid, 'other_big_v142') for c in [p]+p.children(recursive=True) if c.memory_info().rss > 4*1024**3]
            vm_process = any(x in name for x in ('qemu', 'vfkit', 'virtualization', 'prl_vm', 'vmware-vmx'))
            hydraulic = any('hydraulic' in a.lower() for a in args) and name not in ('bash', 'zsh', 'python', 'python3', 'rg', 'grep', 'pi', 'claude')
            if (vm_process or hydraulic) and vm.free+vm.inactive < 10*1024**3: found.append((p.pid, 'vm_free_inactive_lt10gib'))
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    return found


def wait_resources(out):
    while reasons := resource_reason():
        write(out/'waiting.json', dict(status='WAIT_NO_VM_CONTROL', reasons=reasons, unix=time.time()))
        time.sleep(60)


class Guard:
    def __init__(self, directory, metal=False):
        import psutil
        self.directory, self.metal, self.peak = directory, metal, 0
        self.process = psutil.Process(); self.stop = threading.Event()
        threading.Thread(target=self.monitor, daemon=True).start()

    def monitor(self):
        while not self.stop.wait(.1):
            rss = self.process.memory_info().rss
            if self.metal:
                import torch
                driver = torch.mps.driver_allocated_memory()
            else: driver = 0
            self.peak = max(self.peak, rss+driver)
            if self.peak >= 7.5*1024**3:
                write(self.directory/'resource_stop.json', dict(status='STOP_RSS_PLUS_DRIVER_7_5GIB', peak_bytes=self.peak))
                os._exit(75)


def prepare(out, directory):
    import numpy as np
    from scripts import step2_colocated_signals as s
    plan = check(out); result, matrix, queries, labels = s.bench.source()
    parent = read(verify(read(ROOT/'docs/step2_g0_bench.json')['data']))
    cohort = read(s.bench.SOURCE/'cohort.json')
    if cohort['external_pilot_company_overlap'] or cohort['original_pilot_companies_excluded'] != 405: raise ValueError('Pilot custody')
    for ref in cohort['artifacts'].values(): verify(ref)
    groups = np.array(parent['groups']); origin = np.arange(len(groups))
    if plan['smoke']:
        chosen = sorted(set(groups), key=lambda g: hashlib.sha256(('step2-ft-smoke|'+g).encode()).digest())[:60]
        origin = np.flatnonzero(np.isin(groups, chosen)); queries = [queries[i] for i in origin]; groups = groups[origin]
    keys = np.load(matrix['keys']['path'], mmap_mode='r')[origin]
    truth = [labels[q['query_id']]['human_ground_truth_siret'] for q in queries]
    seal = s.metadata(directory, queries, keys, parent['official_refs'], np.arange(len(origin)), truth)
    np.save(out/'origin.npy', origin); np.save(out/'keys.npy', keys)
    views = np.load(verify(parent['views']), mmap_mode='r')[origin]; np.save(out/'views.npy', views)
    if not np.array_equal(keys == np.array(truth, dtype='S14')[:, None], views[:, :, 0]): raise ValueError('Exact label/key drift')
    companies = np.random.default_rng(plan['cfg']['seed']).permutation(np.array(sorted(set(groups))))
    count = math.ceil(.2*len(companies)); eval_groups, train_groups = companies[:count], companies[count:]
    ncal = math.ceil(.2*len(train_groups)); cal_groups, proper_groups = train_groups[:ncal], train_groups[ncal:]
    roles = {'evaluation':np.flatnonzero(np.isin(groups, eval_groups)).tolist(), 'training':np.flatnonzero(np.isin(groups, train_groups)).tolist(),
        'calibration':np.flatnonzero(np.isin(groups, cal_groups)).tolist(), 'proper':np.flatnonzero(np.isin(groups, proper_groups)).tolist()}
    for a,z in [('evaluation','training'), ('calibration','proper')]:
        if set(groups[roles[a]]) & set(groups[roles[z]]): raise ValueError('Split company leakage')
    # Reuse the already sealed descriptive pilot, never its rows in a gradient.
    pilot_dir = PREVIOUS/'confirmation_prepare'; pilot = read(pilot_dir/'pilot.json')
    pilot_seal = read(verify(pilot['seal']))
    for ref in pilot_seal['files'].values(): verify(ref)
    pilot_origin = np.arange(len(pilot['ids']))
    if plan['smoke']:
        pilot_origin = np.array([i for i,f in enumerate(pilot['folds']) if i in [j for j,x in enumerate(pilot['folds']) if x==f][:12]])
    pilot_groups = set(pilot['groups'])
    if set(groups) & pilot_groups: raise ValueError('Pilot company in training bank')
    pqueries = s.b.index(read(s.old.OWN/'full/pilot_queries.json'))
    families = {a['query_id'] for a in read(ROOT/'docs/step2_error_families_g0.json')['assignments'] if a['family']=='colocation'}
    np.save(out/'family.npy', [q['query_id'] in families for q in queries])
    # One global text order; indices remapped, sources remain immutable.
    texts = read(directory/'texts.json'); lookup = {t:i for i,t in enumerate(texts)}
    ptexts = read(verify(pilot_seal['files']['texts.json'])); remap = []
    for text in ptexts:
        if text not in lookup: lookup[text] = len(texts); texts.append(text)
        remap.append(lookup[text])
    remap = np.array(remap, dtype=np.int32)
    def mapped(array): return np.where(array>=0, remap[np.maximum(array,0)], -1)
    np.save(out/'pilot_qtext.npy', mapped(np.load(pilot_dir/'qtext.npy')[pilot_origin]))
    np.save(out/'pilot_ctext.npy', mapped(np.load(pilot_dir/'ctext.npy')[pilot_origin]))
    write(out/'texts.json', texts)
    for name in ('pilot_base.npy', 'pilot_scene.npy', 'pilot_views.npy', 'naf.npy', 'addresses.npy'):
        np.save(out/('p_'+name), np.load(pilot_dir/name, mmap_mode='r')[pilot_origin])
    p = dict(ids=[pilot['ids'][i] for i in pilot_origin], groups=[pilot['groups'][i] for i in pilot_origin], folds=[pilot['folds'][i] for i in pilot_origin],
        names=[s.b.normalize_name(s.b.text(pqueries[pilot['ids'][i]]['crm_name'])) for i in pilot_origin])
    if plan['smoke']:
        from sklearn.model_selection import GroupKFold
        p['outer'] = [(i,a.tolist(),z.tolist()) for i,(a,z) in enumerate(GroupKFold(3,shuffle=True,random_state=plan['cfg']['seed']).split(p['ids'],groups=p['groups']))]
    else: p['outer'] = pilot['outer']
    write(out/'pilot.json', p)
    forbidden = set(eval_groups) | set(cal_groups) | pilot_groups
    qi = np.load(directory/'qtext.npy'); ci = np.load(directory/'ctext.npy'); addresses = np.load(directory/'addresses.npy')
    rows = []; skips = Counter(); negative_counts = Counter()
    for i in roles['proper']:
        positive = np.flatnonzero(views[i,:,0])
        if not len(positive): skips['exact_absent_from_g0'] += 1; continue
        j = int(positive[0]); pos = int(ci[i,j])
        if qi[i]<0 or pos<0: skips['missing_text'] += 1; continue
        allowed = [k for k in range(100) if not views[i,k,1] and ci[i,k]>=0 and ci[i,k]!=pos and keys[i,k].decode()[:9] not in forbidden]
        allowed.sort(key=lambda k:(not(addresses[i,j]>=0 and addresses[i,k]==addresses[i,j]), keys[i,k][:9]!=keys[i,j][:9], k))
        if not allowed: skips['no_safe_negative'] += 1; continue
        for k in allowed[:4]:
            negative_counts['same_address'] += int(addresses[i,j]>=0 and addresses[i,k]==addresses[i,j])
            negative_counts['same_siren'] += int(keys[i,k][:9]==keys[i,j][:9])
        rows.append(dict(row=i, query=int(qi[i]), positive=pos, negatives=[int(ci[i,k]) for k in allowed]))
    if not rows: raise ValueError('No eligible real training pairs')
    write(out/'training.json', rows)
    sources = [record(ROOT/'docs/step2_g0_bench.json'),record(ROOT/'docs/step2_error_families_g0.json'),record(s.bench.SOURCE/'cohort.json'),matrix['features'],matrix['keys'],parent['views'],record(pilot_dir/'pilot.json')]
    sources += list(cohort['artifacts'].values())+list(parent['official_refs'].values())+list(pilot_seal['files'].values())+seal['naf_refs']
    for name in ('pilot_base.npy','pilot_scene.npy','pilot_views.npy'): sources.append(record(pilot_dir/name))
    payload = dict(ids=[q['query_id'] for q in queries], groups=groups.tolist(), folds=[q['fold'] for q in queries],
        names=[s.b.normalize_name(s.b.text(q['crm_name'])) for q in queries], roles=roles, metadata=str(directory), classes=seal['code_classes'],
        source_matrix=matrix['features'], scenes=record(s.bench.OWN/'full/scene.npy'), sources=sources,
        queries=len(groups), companies=len(set(groups)), training_pairs=len(rows), skipped_training=dict(skips), negative_counts=dict(negative_counts),
        role_counts={k:dict(queries=len(v),companies=len(set(groups[v]))) for k,v in roles.items()}, exact_present=int(views[:,:,0].any(1).sum()), union_present=int(views[:,:,1].any(1).sum()))
    write(out/'data.json', payload)
    artifacts = [record(p) for p in out.glob('*.npy')]+[record(out/n) for n in ('data.json','texts.json','training.json','pilot.json')]+list(seal['files'].values())+[record(directory/'naf_targets.npy')]
    return dict(data=record(out/'data.json'), artifacts=artifacts, queries=len(groups), companies=len(set(groups)), texts=len(texts), roles=payload['role_counts'], training_pairs=len(rows), skipped_training=dict(skips))


def load_model(cfg, adapted):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(cfg['seed'])
    if not torch.backends.mps.is_available(): raise ValueError('Metal unavailable')
    torch.mps.set_per_process_memory_fraction(cfg['mps_memory_fraction'])
    config = AutoConfig.from_pretrained(HOME/'base',local_files_only=True); config.audio_config=None; config.vision_config=None
    if config.text_config.attention_dropout != 0: raise ValueError('Gradient cache requires deterministic forwards')
    model = AutoModel.from_pretrained(HOME/'base',config=config,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('mps')
    tokenizer = AutoTokenizer.from_pretrained(HOME/'base',local_files_only=True)
    for p in model.parameters(): p.requires_grad_(False)
    if adapted:
        class LoRA(torch.nn.Module):
            def __init__(self, linear):
                super().__init__(); self.base=linear
                self.a=torch.nn.Parameter(torch.empty(8,linear.in_features,device='mps',dtype=torch.float32))
                self.b=torch.nn.Parameter(torch.zeros(linear.out_features,8,device='mps',dtype=torch.float32))
                torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
            def forward(self,x): return self.base(x)+2*F.linear(F.linear(x.float(),self.a),self.b).to(x.dtype)
        for name,module in list(model.named_modules()):
            if name.rsplit('.',1)[-1] in cfg['lora_modules']:
                parent,key=name.rsplit('.',1); setattr(model.get_submodule(parent),key,LoRA(module))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}; cropped=Counter()
    def encode(texts):
        raw=tokenizer(texts,padding=False,truncation=False)['input_ids']; cropped['texts']+=len(texts); cropped['over512']+=sum(len(v)>512 for v in raw)
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt')
        tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')}
        hidden=model(**tokens).last_hidden_state.float(); mask=tokens['attention_mask'].unsqueeze(-1)
        return F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
    return model,params,encode,cropped


def train(out, directory, seconds):
    import numpy as np
    import torch
    from scripts.embeddinggemma2_v2_fit import cached_objective
    plan=check(out); cfg=plan['cfg']; rows=read(out/'training.json'); texts=read(out/'texts.json')
    model,params,encode,cropped=load_model(cfg,True)
    optimizer=torch.optim.AdamW(params.values(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    step=0; epoch=0; cursor=0; elapsed=0.; exposed=set(); binding=record(out/'plan.json')['sha256']; began=time.monotonic()
    own=directory/'checkpoint.pt'
    if own.exists():
        state=torch.load(own,map_location='cpu',weights_only=True)
        if state['binding']!=binding: raise ValueError('Checkpoint binding')
        for n,p in params.items(): p.data.copy_(state['adapter'][n].to('mps'))
        optimizer.load_state_dict(state['optimizer']); step=state['step'];epoch=state['epoch'];cursor=state['cursor'];elapsed=state['seconds'];exposed=set(state['exposed'])
    initial={n:p.detach().cpu().clone() for n,p in params.items()} if step==0 else None
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu(v) for v in value]
        return value
    def save():
        temp=own.with_suffix('.tmp');torch.save(dict(adapter=cpu(params),optimizer=cpu(optimizer.state_dict()),step=step,epoch=epoch,cursor=cursor,
            seconds=elapsed+time.monotonic()-began,exposed=sorted(exposed),binding=binding),temp);os.replace(temp,own)
    total=math.ceil(len(rows)/cfg['batch_size'])*cfg['epochs'];warmup=max(1,round(total*cfg['warmup_fraction']));losses=[];norms=[]
    def budget():return time.monotonic()-began < max(0,seconds-5) and elapsed+time.monotonic()-began < cfg['max_training_seconds']-5
    model.train()
    while epoch<cfg['epochs'] and budget():
        order=np.random.default_rng(cfg['seed']+epoch).permutation(len(rows))
        while cursor<len(order) and budget():
            if resource_reason():save();return 75
            chosen=order[cursor:cursor+cfg['batch_size']];batch=[rows[i] for i in chosen];q=[texts[r['query']] for r in batch];docs=[texts[r['positive']] for r in batch]
            own_columns=[]
            for r in batch:
                neg=r['negatives'];start=(epoch*4)%len(neg);selected=(neg[start:]+neg[:start])[:4]
                own_columns.append(list(range(len(docs),len(docs)+len(selected))));docs.extend(texts[k] for k in selected)
            mask=np.ones((len(q),len(docs)),dtype=bool)
            for i,cols in enumerate(own_columns):mask[i,[i]+cols]=False
            if step<warmup:multiplier=(step+1)/warmup
            else:multiplier=cfg['final_lr_fraction']+(1-cfg['final_lr_fraction'])*.5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total-warmup))))
            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*multiplier
            optimizer.zero_grad(set_to_none=True)
            loss=cached_objective(encode,q,docs,mask,np.ones(len(q)),cfg['temperature'],cfg['microbatch_size'])
            norm=torch.nn.utils.clip_grad_norm_(params.values(),cfg['gradient_clip'])
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
            optimizer.step();step+=1;cursor+=len(chosen);exposed.update(int(i) for i in chosen);losses.append(loss);norms.append(float(norm));torch.mps.empty_cache()
            if step<=3 or step%cfg['checkpoint_every_steps']==0:save()
            write(directory/'progress.json',dict(step=step,epoch=epoch,cursor=cursor,loss=loss,gradient_norm=float(norm),seconds=elapsed+time.monotonic()-began,exposed=len(exposed),pairs=len(rows)))
            if plan['smoke'] and step>=2:break
        if cursor>=len(order):epoch+=1;cursor=0
        if plan['smoke'] and step>=2:break
    save()
    if step==0:raise ValueError('No optimizer step within budget')
    changed=any(not torch.equal(initial[n],p.detach().cpu()) for n,p in params.items()) if initial is not None else True
    if not changed:raise ValueError('Adapter did not change')
    return dict(status='SMOKE_TWO_REAL_STEPS' if plan['smoke'] else 'TWO_EPOCHS' if epoch==cfg['epochs'] else 'TRAIN_TIME_BUDGET_CHECKPOINTED',
        checkpoint=record(own),steps=step,epochs=epoch,cursor=cursor,training_pairs=len(rows),exposed_pairs=len(exposed),exposure_fraction=len(exposed)/len(rows),
        training_seconds=elapsed+time.monotonic()-began,last_losses=losses[-3:],last_gradient_norms=norms[-3:],adapter_changed=changed,
        fresh_initialization=True,historical_adapter_loaded=False,tokenization=dict(cropped))


def encode(out,directory,adapted,seconds):
    import numpy as np
    import torch
    plan=check(out);cfg=plan['cfg'];texts=read(out/'texts.json');size=cfg['encoding_shard_size'];began=time.monotonic()
    model,params,forward,cropped=load_model(cfg,adapted)
    if adapted:
        ref=read(out/'train/receipt.json')['checkpoint'];state=torch.load(verify(ref),map_location='cpu',weights_only=True)
        if state['binding']!=record(out/'plan.json')['sha256']:raise ValueError('Adapter source binding')
        for n,p in params.items():p.data.copy_(state['adapter'][n].to('mps'))
    else:ref=None
    model.eval();cache={};banks=[]
    if not adapted:
        for name in ('encode','confirmation_encode'):
            old=read(PREVIOUS/name/'result.json');oldtexts=read(verify(old['texts']));values=np.load(verify(old['vectors']),mmap_mode='r');banks.append(values)
            cache.update({t:(len(banks)-1,i) for i,t in enumerate(oldtexts)})
    binding=dict(plan=record(out/'plan.json'),adapter=ref,texts=record(out/'texts.json'))
    if (directory/'binding.json').exists():
        if read(directory/'binding.json')!=binding:raise ValueError('Encoding binding changed')
    else:write(directory/'binding.json',binding)
    shards=[];reused=0;new=0
    with torch.inference_mode():
        for at in range(0,len(texts),size):
            path=directory/f'shard_{at:07d}.npy';receipt=path.with_suffix('.json')
            if receipt.exists():
                verify(read(receipt));shards.append(read(receipt));continue
            if time.monotonic()-began>seconds-5:raise TimeoutError('Four-hour total Metal budget exhausted before complete encoding')
            if resource_reason():return 75
            batch=texts[at:at+size];values=np.empty((len(batch),768),dtype=np.float32);missing=[]
            for i,text in enumerate(batch):
                if text in cache:
                    k,j=cache[text];values[i]=banks[k][j];reused+=1
                else:missing.append(i)
            for start in range(0,len(missing),cfg['encoding_batch_size']):
                ix=missing[start:start+cfg['encoding_batch_size']];values[ix]=forward([batch[i] for i in ix]).cpu().numpy();new+=len(ix)
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4):raise ValueError('Invalid encoded vectors')
            with path.with_suffix('.tmp').open('wb') as stream:np.save(stream,values)
            os.replace(path.with_suffix('.tmp'),path);write(receipt,record(path));shards.append(read(receipt))
            write(directory/'progress.json',dict(encoded=at+len(batch),total=len(texts),seconds=time.monotonic()-began,new=new,reused=reused))
    return dict(status='ADAPTED_ENCODED' if adapted else 'BASE_COMPLETED',items=len(texts),shards=shards,shard_size=size,binding=record(directory/'binding.json'),new_this_attempt=new,reused_this_attempt=reused,tokenization=dict(cropped))


def semantic(out,directory):
    import numpy as np
    from scripts import step2_colocated_signals as s
    data=read(out/'data.json');meta=Path(data['metadata']);p=read(out/'pilot.json');n=data['queries'];m=len(p['ids'])
    qi=np.r_[np.load(meta/'qtext.npy'),np.load(out/'pilot_qtext.npy')];ci=np.concatenate([np.load(meta/'ctext.npy'),np.load(out/'pilot_ctext.npy')]);addresses=np.concatenate([np.load(meta/'addresses.npy'),np.load(out/'p_addresses.npy')])
    artifacts=[]
    for variant in ('base','adapted'):
        receipt=read(out/('adapted_encode' if variant=='adapted' else 'base_encode')/'receipt.json');size=receipt['shard_size'];cache={}
        def vectors(indices):
            result=np.empty((len(indices),768),dtype=np.float32)
            for shard in np.unique(indices//size):
                if int(shard) not in cache:
                    if len(cache)>32:cache.clear()
                    cache[int(shard)]=np.load(verify(receipt['shards'][int(shard)]),mmap_mode='r')
                mask=indices//size==shard;result[mask]=cache[int(shard)][indices[mask]%size]
            return result
        cosine=np.full((n+m,100),np.nan,dtype=np.float32)
        for i in range(n+m):
            valid=np.flatnonzero(ci[i]>=0)
            if qi[i]>=0 and len(valid):cosine[i,valid]=vectors(ci[i,valid])@vectors(np.array([qi[i]]))[0]
        features=np.stack([cosine,s.relative(cosine,addresses)],axis=2);np.save(directory/(variant+'.npy'),features);artifacts.append(record(directory/(variant+'.npy')))
    return dict(artifacts=artifacts,queries=n,pilot_queries=m)


def evaluate(out,directory,variant):
    import numpy as np
    from types import SimpleNamespace
    from scripts import step2_colocated_signals as s
    from sklearn.model_selection import GroupKFold
    check(out);stage_directory=directory;directory=directory/('attempt_'+str(time.time_ns()));directory.mkdir()
    data=read(out/'data.json');p=read(out/'pilot.json');meta=Path(data['metadata']);n=data['queries'];m=len(p['ids']);origin=np.load(out/'origin.npy')
    X=np.concatenate([np.load(data['source_matrix']['path'],mmap_mode='r')[origin],np.load(out/'p_pilot_base.npy')])
    scene=np.concatenate([np.load(data['scenes']['path'],mmap_mode='r')[origin],np.load(out/'p_pilot_scene.npy')])
    pv=np.load(out/'p_pilot_views.npy');views=np.concatenate([np.load(out/'views.npy'),pv[:,:,[s.b.VIEWS.index('exact'),s.b.VIEWS.index('union')]]])
    groups=np.array(data['groups']+p['groups']);ids=np.array(data['ids']+p['ids']);folds=np.array(data['folds']+p['folds'])
    ds=SimpleNamespace(X=X,scene=scene,views=views,groups=groups,ids=ids,folds=folds,names=np.array(data['names']+p['names']),classes=data['classes'],
        codes=np.concatenate([np.load(meta/'naf.npy'),np.load(out/'p_naf.npy')]),targets=np.concatenate([np.load(meta/'naf_targets.npy'),np.full((m,2),-1,dtype=np.int16)]),
        addresses=np.concatenate([np.load(meta/'addresses.npy'),np.load(out/'p_addresses.npy')]),semantic=np.load(out/'semantic'/(variant+'.npy')))
    class Runner(s.Runner):
        def __init__(self):
            self.ds,self.name,self.directory=ds,'s1_s2',directory/'models';self.directory.mkdir();self.models={};self.receipts=[]
    s.wait_resources=lambda: None  # Parent supervisor owns admission and only its own child.
    runner=Runner();roles={k:np.array(v,dtype=int) for k,v in data['roles'].items()};tr=roles['training'];te=roles['evaluation'];proper=roles['proper'];cal=roles['calibration']
    pred=runner.score(tr,te,'external_company_holdout');top=s.b.select_top(pred)[0];hits=views[te,top]
    common=runner.score(proper,np.r_[cal,te],'confidence_calibration_and_holdout');cp,ep=np.split(common,[len(cal)]);ct,cm=s.b.select_top(cp);et,em=s.b.select_top(ep)
    threshold,stats=s.b.threshold_choice(cm,views[cal,ct,1]);selected=em>=threshold if threshold is not None else np.zeros(len(te),dtype=bool)
    # Replay the actual ranking training representation; in-sample diagnostic only.
    train_s1=np.full((len(tr),100,len(s.S1)),np.nan,dtype=np.float32)
    for k,(a,z) in enumerate(GroupKFold(3,shuffle=True,random_state=s.SEED).split(tr,groups=groups[tr])):
        train_s1[z],_=s.nb_features(ds,tr[a],tr[z],directory,f'diagnostic_train_oof{k}')
    trainX=np.concatenate([X[tr],train_s1,ds.semantic[tr]],axis=2);model,_=runner.models[s.d.digest(np.sort(tr).tolist())]
    trainpred=model.predict(trainX.reshape(-1,trainX.shape[2])).reshape(len(tr),100);trainhits=views[tr,s.b.select_top(trainpred)[0]];del trainX,train_s1;gc.collect()
    def cosine_hits(rows):
        values=ds.semantic[rows,:,0];missing=~np.isfinite(values).any(1);chosen=np.argmax(np.where(np.isfinite(values),values,-np.inf),axis=1);value=views[rows,chosen].copy();value[missing]=False
        return dict(metrics=s.bench.metrics(value),all_cosines_missing=int(missing.sum()))
    pilot_indices=np.arange(n,n+m);pilotpred=runner.score(tr,pilot_indices,'descriptive_external_content_pilot')
    SX=s.old.stack_cube(scene[pilot_indices],pilotpred);y=pv[:,:,s.b.VIEWS.index('exact')].astype(np.float32)
    class Stack(s.old.StackRunner):
        def score(self,train,test,columns,purpose):
            values=super().score(train,test,columns,purpose)
            if self.fixed is None:
                _,receipt=self.models[s.d.digest(np.sort(train).tolist())];replay=s.xgb.XGBRanker(n_jobs=1);replay.load_model(verify(receipt['model']))
                actual=replay.predict(self.X[test].reshape(-1,len(s.old.STACK))).reshape(len(test),100)
                if not np.array_equal(values,actual):raise ValueError('Stack replay mismatch')
            return values
    outer=[(int(f),np.array(a),np.array(z)) for f,a,z in p['outer']];pg=np.array(p['groups']);pi=np.array(p['ids']);pf=np.array(p['folds'])
    stack=Stack(SX,y,pg,pi,directory/'stack_models');stacked=np.full_like(pilotpred,np.nan)
    for f,a,z in outer:stacked[z]=stack.score(a,z,list(range(len(s.old.STACK))),f'outer{f}')
    fixed=Stack(SX,y,pg,pi,directory/'fixed_content',fixed=pilotpred);confidence={}
    for name,scorer in [('content',fixed),('stack',stack)]:
        confidence[name],_=s.old.v2.confidence(scorer,outer,scene[pilot_indices],pv,pg,pf,pi,list(range(len(s.old.STACK))),directory,name)
    pilot_result=dict(status='DESCRIPTIVE_ALREADY_CONSUMED526' if m==526 else 'SMOKE_DESCRIPTIVE',queries=m,
        content={v:s.b.metric(int(pv[np.arange(m),s.b.select_top(pilotpred)[0],j].sum()),m) for j,v in enumerate(s.b.VIEWS)},
        stack={v:s.b.metric(int(pv[np.arange(m),s.b.select_top(stacked)[0],j].sum()),m) for j,v in enumerate(s.b.VIEWS)},confidence=confidence,stack_models=stack.receipts)
    np.savez_compressed(directory/'predictions.npz',hits=hits,predictions=pred,common=ep,sur=selected,train_hits=trainhits,pilot_content=pilotpred,pilot_stack=stacked)
    value=dict(variant=variant,top1=s.bench.metrics(hits),by_fold={str(f):s.bench.metrics(hits[folds[te]==f]) for f in (2,3,4)},
        availability=s.bench.metrics(views[te].any(1)),sur=s.bench.metrics(views[te,et],selected),sur_by_fold={str(f):s.bench.metrics(views[te,et][folds[te]==f],selected[folds[te]==f]) for f in (2,3,4)},
        threshold=threshold,calibration=stats,confidence_common_top1=s.bench.metrics(views[te,et]),
        anti_memorization=dict(ranking_training_in_sample=s.bench.metrics(trainhits),ranking_evaluation=s.bench.metrics(hits),
            ranking_union_gap_pp=float(100*(trainhits[:,1].mean()-hits[:,1].mean())),cosine_proper=cosine_hits(proper),cosine_evaluation=cosine_hits(te)),
        pilot=pilot_result,models=runner.receipts,predictions=record(directory/'predictions.npz'),artifacts=[record(directory/'predictions.npz')])
    write(stage_directory/'result.json',value);return dict(result=record(stage_directory/'result.json'),artifacts=value['artifacts'])


def summarize(out,directory):
    import numpy as np
    from scripts import step2_colocated_signals as s
    data=read(out/'data.json');te=np.array(data['roles']['evaluation']);groups=np.array(data['groups'])[te];family=np.load(out/'family.npy')[te]
    variants={k:read(out/k/'result.json') for k in ('base','adapted')};hits={k:np.load(verify(v['predictions']))['hits'] for k,v in variants.items()}
    delta=s.boot(hits,groups);fd=s.boot(hits,groups,family) if family.any() else None
    r=dict(status='COMPLETE_DEVELOPMENT_NOT_PRODUCTION',smoke=check(out)['smoke'],data=record(out/'data.json'),roles=data['role_counts'],training=read(out/'train/receipt.json'),
        variants=variants,deltas=delta,family=dict(queries=int(family.sum()),deltas=fd,top1={k:s.bench.metrics(h[family]) for k,h in hits.items()} if family.any() else None),
        bootstrap_replicates=10000,bootstrap_seed=s.SEED,labels_changed=False,candidates_changed=False,closed_sets_opened=False,production_allowed=False,
        recommendation='Poursuivre un audit indépendant hors production.' if delta['adapted']['union']['paired_company_bootstrap95_pp'][0]>0 else 'Conserver S2-BASE, sans promotion de cet adaptateur.')
    write(out/'result.json',r);return dict(result=record(out/'result.json'))


def report(r):
    lines=['## Fait',f'Statut : {r["status"]}. V142/G0 et labels inchangés ; aucune promotion.']
    lines += [f'Durée murale réelle : {r["wall_seconds"]/60:.1f} min ; Metal cumulé : {r["metal_seconds"]/60:.1f} min ; pic RSS+driver : {r["peak_bytes"]/1024**3:.2f} Gio.']
    if 'variants' not in r:
        return '\n'.join(lines+['## Preuve',f'Arrêt : {r.get("reason")}. Étapes achevées : {r.get("completed", [])}.','Résultats complets/SUR/526 non mesurés.','## Écarts / choix','Calibration interne disjointe du gradient ; jamais de métrique partielle présentée comme complète.','## Reste / risques','Recommandation unique : conserver S2-BASE tant que cette expérience ne fournit pas de résultat complet.'])+'\n'
    def pct(v):return 'indéfinie' if v is None else f'{v["hits"]}/{v["queries"]}={v["percent"]:.2f}%'
    lines+=['## Preuve',f'Entreprises/rôles : {r["roles"]}.',f'Gradient : {r["training"]["steps"]} pas, {r["training"]["exposed_pairs"]}/{r["training"]["training_pairs"]} paires exposées, {r["training"]["training_seconds"]/60:.1f} min.']
    for k,v in r['variants'].items():
        lines += [f'{k} évaluation : exact {pct(v["top1"]["exact"])} ; union {pct(v["top1"]["union"])}.',
            f'{k} SUR : union {pct(v["sur"]["precision"]["union"])} ; couverture {pct(v["sur"]["coverage"])} ; seuil interne98%, pas garantie.',
            f'{k} écart ranking train−évaluation UNION : {v["anti_memorization"]["ranking_union_gap_pp"]:+.2f} pts (train in-sample).',
            f'{k} pilote {v["pilot"]["queries"]} descriptif : content exact/union {v["pilot"]["content"]["exact"]["hits"]}/{v["pilot"]["content"]["union"]["hits"]} ; stack {v["pilot"]["stack"]["exact"]["hits"]}/{v["pilot"]["stack"]["union"]["hits"]}.']
    dd=r['deltas']['adapted'];lines += [f'Δ UNION adapté−BASE : {dd["union"]["delta_pp"]:+.3f} pts ; IC95 entreprise {dd["union"]["paired_company_bootstrap95_pp"]}.',
        f'Δ exact : {dd["exact"]["delta_pp"]:+.3f} pts ; IC95 entreprise {dd["exact"]["paired_company_bootstrap95_pp"]}.',
        f'Famille même adresse : {r["family"]["queries"]} cas ; deltas/IC : {r["family"]["deltas"]}.','## Écarts / choix',
        'Cache BASE initial=3000 dossiers ; complété sur tout le banc. Gradient proper64%, calibration16%, évaluation20% par entreprise.',
        'Dernier checkpoint, hyperparamètres fixes ; faux négatifs union/textes identiques et entreprises tenues à l’écart masqués.','## Reste / risques',
        'Développement sur amont consommé/non full-OOF ; train in-sample optimiste, pilote526 descriptif ; Wilson/rôles/SHA dans JSON.',
        'Recommandation unique : '+r['recommendation']]
    if len(lines)>30:raise ValueError('Report exceeds30 lines')
    return '\n'.join(lines)+'\n'


def publish(out,directory):
    plan=check(out);r=read(out/'result.json');budget=read(out/'budget.json');r.update(wall_seconds=time.time()-plan['started_unix'],metal_seconds=budget['metal_seconds'],
        peak_bytes=max([read(p).get('peak_bytes',0) for p in out.glob('*/receipt.json')]+[read(p).get('peak_bytes',0) for p in out.glob('*/resource_stop.json')]+[0]),plan=record(out/'plan.json'))
    write(out/'result.json',r);text=report(r);(out/'final_report.md').write_text(text)
    if plan['smoke']:return dict(smoke=True,report=record(out/'final_report.md'))
    for ref in read(out/'data.json')['sources'] if (out/'data.json').exists() else []:verify(ref)
    paths=['docs/step2_gemma_finetune.json','docs/step2_gemma_finetune.md','handover.md']
    for path in paths:
        if subprocess.check_output(['git','diff','HEAD','--',path],cwd=ROOT).strip():raise ValueError('Foreign dirty publication path: '+path)
    before=(ROOT/'handover.md').read_text();write(ROOT/paths[0],r);DOC.write_text((out/'protocol.md').read_text()+'\n## Résultats\n\n'+text)
    line=f'\n- 2026-10-19 — Gemma étape2 : {r["status"]}, Metal{r["metal_seconds"]:.1f}s, rapport `{out}/final_report.md`, protocole/recette/rôles/SHA et résultat `docs/step2_gemma_finetune.json`. Aucun poids historique adapté/label/pool/jeu fermé/web/VM/push. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_gemma_finetune.json`.\n'
    with (ROOT/'handover.md').open('a') as stream:stream.write(line)
    if (ROOT/'handover.md').read_text()!=before+line:raise ValueError('Concurrent handover edit')
    subprocess.run(['git','add','--',paths[0]],cwd=ROOT,check=True);subprocess.run(['git','diff','--check','--',*paths],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m','feat: publish company-held-out Gemma adaptation experiment','--',*paths],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',paths[0]],cwd=ROOT,text=True).strip();write(out/'publication.json',dict(commit=commit));return dict(commit=commit)


def run(out, smoke=False, resume=False):
    if not resume:
        out.mkdir(parents=True,exist_ok=False);(out/'protocol.md').write_text(DOC.read_text().split('\n## Résultats\n')[0]);cfg=read(CONFIG)
        from scripts import step2_colocated_signals as s
        implementations=[SCRIPT,Path(s.__file__),Path(s.bench.__file__),Path(s.old.__file__),Path(s.b.__file__),Path(s.d.__file__),Path(s.old.v2.__file__),ROOT/'scripts/embeddinggemma2_v2_fit.py']
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for path in [SCRIPT,CONFIG,DOC]:
            if subprocess.check_output(['git','diff','HEAD','--',str(path)],cwd=ROOT).strip():raise ValueError('Commit protocol/code/config before any training')
            subprocess.run(['git','ls-files','--error-unmatch',str(path)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        write(out/'plan.json',dict(smoke=smoke,started_unix=time.time(),cfg=cfg,config=record(CONFIG),protocol=record(out/'protocol.md'),git_head=head,
            implementations=[record(p) for p in implementations],base_files=[record(p) for p in sorted((HOME/'base').glob('*')) if p.is_file()]))
        # Full work must follow a successful real smoke, whose Metal also counts.
        if not smoke:
            smoke_state=read(OWN/'smoke/state.json')
            if smoke_state['status']!='COMPLETE':raise ValueError('Successful real smoke required before full run')
            smoke_budget=read(OWN/'smoke/budget.json')
            smoke_result=read(OWN/'smoke/result.json')
            if smoke_result['status']!='COMPLETE_DEVELOPMENT_NOT_PRODUCTION':raise ValueError('Incomplete smoke')
            write(out/'smoke_proof.json',dict(state=record(OWN/'smoke/state.json'),result=record(OWN/'smoke/result.json'),budget=record(OWN/'smoke/budget.json')))
        else:smoke_budget={'metal_seconds':0.}
        write(out/'budget.json',dict(metal_seconds=smoke_budget['metal_seconds'],training_seconds=0.,history=[]))
    import psutil
    plan=check(out);budget=read(out/'budget.json');completed=[]
    lease_path=out/'active.json'
    if lease_path.exists():
        lease=read(lease_path)
        if not any(h.get('launch_id')==lease['launch_id'] for h in budget['history']):
            try:
                process=psutil.Process(lease['pid'])
                if abs(process.create_time()-lease['created_unix'])<2:raise ValueError('Owned child still running; do not launch a duplicate')
            except psutil.NoSuchProcess:pass
            elapsed=max(0,time.time()-lease['started_unix'])
            if lease['metal']:budget['metal_seconds']+=elapsed
            if lease['stage']=='train':budget['training_seconds']+=elapsed
            budget['history'].append(dict(**lease,seconds=elapsed,exit_code=None,reason='CONSERVATIVE_RECOVERY_ACCOUNTING'))
            write(out/'budget.json',budget)
    stages=[('prepare',False),('train',True),('base_encode',True),('adapted_encode',True),('semantic',False),('base',False),('adapted',False),('summarize',False)]
    try:
        for stage,metal in stages:
            directory=out/stage;directory.mkdir(exist_ok=True)
            if (directory/'receipt.json').exists():
                receipt=read(directory/'receipt.json')
                for ref in receipt.get('artifacts',[]):verify(ref)
                if 'result' in receipt:verify(receipt['result'])
                if 'checkpoint' in receipt:verify(receipt['checkpoint'])
                if 'shards' in receipt:
                    for ref in receipt['shards']:verify(ref)
                completed.append(stage);continue
            attempts=0
            while True:
                wait_resources(out);check(out)
                remaining=plan['cfg']['max_metal_seconds']-budget['metal_seconds']
                seconds=min(remaining,plan['cfg']['max_training_seconds']-budget['training_seconds']) if stage=='train' else remaining
                if metal and seconds<10:raise TimeoutError('Metal budget exhausted')
                write(out/'state.json',dict(status='RUNNING',stage=stage,completed=completed,budget=budget,pid=os.getpid()))
                start=time.monotonic()
                with (directory/'worker.log').open('a') as log:
                    child=subprocess.Popen([str(METAL if metal else CPU),'-u',str(SCRIPT),'worker','--stage',stage,'--output-dir',str(out),'--seconds',str(seconds)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    launch_id=str(time.time_ns())
                    write(lease_path,dict(pid=child.pid,created_unix=psutil.Process(child.pid).create_time(),stage=stage,started_unix=time.time(),metal=metal,launch_id=launch_id))
                    # Only this owned child can be terminated, never a VM or foreign job.
                    reason=None
                    while child.poll() is None:
                        if metal and time.monotonic()-start>seconds:
                            reason='METAL_DEADLINE';child.terminate();break
                        if resource_reason():
                            reason='RESOURCE_YIELD_OWN_CHILD_ONLY';child.terminate();break
                        time.sleep(.5)
                    try:code=child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();code=child.wait()
                elapsed=time.monotonic()-start
                if metal:budget['metal_seconds']+=elapsed
                if stage=='train':budget['training_seconds']+=elapsed
                budget['history'].append(dict(stage=stage,seconds=elapsed,metal=metal,exit_code=code,reason=reason,launch_id=launch_id));write(out/'budget.json',budget)
                if reason=='RESOURCE_YIELD_OWN_CHILD_ONLY':code=75
                if code==0:completed.append(stage);break
                if code==75 and attempts<plan['cfg']['max_resource_restarts']:
                    attempts+=1;time.sleep(30);continue
                raise RuntimeError(f'{stage} failed exit={code} reason={reason}; see {directory}/worker.log')
        publish_dir=out/'publication';publish_dir.mkdir(exist_ok=True);publication=publish(out,publish_dir);write(publish_dir/'receipt.json',publication)
        write(out/'state.json',dict(status='COMPLETE',completed=completed,publication=read(out/'publication.json') if (out/'publication.json').exists() else None))
    except BaseException as exc:
        write(out/'state.json',dict(status='STOP',reason=str(exc),completed=completed))
        if (out/'result.json').exists() and 'variants' in read(out/'result.json'):
            write(out/'publication_error.json',dict(reason=str(exc),measured_result_preserved=True));raise
        write(out/'result.json',dict(status='STOP_INCOMPLETE',reason=str(exc),completed=completed,labels_changed=False,candidates_changed=False,production_allowed=False))
        directory=out/'publication';directory.mkdir(exist_ok=True)
        try:publish(out,directory)
        except BaseException as publication_error:write(out/'publication_error.json',dict(reason=str(publication_error)))
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','resume','worker']);parser.add_argument('--smoke',action='store_true');parser.add_argument('--output-dir',type=Path);parser.add_argument('--stage');parser.add_argument('--seconds',type=float,default=14400)
    args=parser.parse_args();out=args.output_dir or OWN/('smoke' if args.smoke else 'full')
    if not out.resolve().is_relative_to(OWN) or out.resolve()==OWN:raise ValueError('Owned namespace required')
    if args.action!='worker':
        import fcntl
        OWN.mkdir(parents=True,exist_ok=True)
        with (OWN/'supervisor.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run(out,args.smoke,args.action=='resume')
        return
    plan=check(out);directory=out/args.stage;metal=args.stage in ('train','base_encode','adapted_encode');guard=Guard(directory,metal);began=time.monotonic()
    for ref in read(out/'prepare/receipt.json').get('artifacts',[]) if args.stage!='prepare' else []:verify(ref)
    if args.stage=='prepare':value=prepare(out,directory)
    elif args.stage=='train':
        value=train(out,directory,args.seconds)
        if value==75:sys.exit(75)
    elif args.stage in ('base_encode','adapted_encode'):
        value=encode(out,directory,args.stage=='adapted_encode',args.seconds)
        if value==75:sys.exit(75)
    elif args.stage=='semantic':value=semantic(out,directory)
    elif args.stage in ('base','adapted'):value=evaluate(out,directory,args.stage)
    elif args.stage=='summarize':value=summarize(out,directory)
    else:raise ValueError('Unknown stage')
    value.update(seconds=time.monotonic()-began,peak_bytes=guard.peak);write(directory/'receipt.json',value);guard.stop.set()


if __name__=='__main__':main()
