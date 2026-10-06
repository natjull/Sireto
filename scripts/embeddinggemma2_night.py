"""Isolated local contrastive sidequest. Never reads the final test in training.

No serving change, no retrieval recall claim, no downstream model update.
The official base model and current retrieval artifacts remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path('/Users/nathanjullia/Documents/Projets/SIRETO')
DATA = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
INDEX = DATA/'indexes/hierarchical_retrieval_v3/4b332899d8698354'
LEDGER = DATA/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
GEN = DATA/'audits/retrieval_v34_fast_resume/full_job/generation/manifest.json'
PANEL = DATA/'audits/retrieval_v328_serving_panel/package_20261005_a/source/panel.json'
CURRENT = DATA/'audits/retrieval_v359_persistent_launcher/job_20261006_a/state.json'
REVISION = '914f7f89142e33e77833254d9c9b90c3cef7303b'
SEED = 'embeddinggemma2-20261006-siren-70-10-20-v1'
QUERY_PROMPT = 'task: search result | query: '
DOC_PROMPT = 'title: none | text: '


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def grouped_split(labels):
    counts = Counter(r['human_ground_truth_siren'] for r in labels)
    groups = {}
    for siren, count in counts.items():
        # Coarse public/multi-site strata, only split metadata, NEVER a feature.
        bucket = (siren.startswith(('1', '2')), count > 1)
        groups.setdefault(bucket, []).append(siren)
    membership = {}
    for entities in groups.values():
        entities.sort(key=lambda s: hashlib.sha256((SEED+'|'+s).encode()).digest())
        a, b = round(.7*len(entities)), round(.8*len(entities))
        for i, siren in enumerate(entities):
            membership[siren] = 'train' if i < a else 'validation' if i < b else 'test'
    return membership


def clean_text(names, addresses, postcode=''):
    def safe(s):
        # IDs in raw CRM strings/official strings cannot become model features.
        return re.sub(r'(?<!\d)(?:\d{14}|\d{9})(?!\d)', ' ', str(s))
    return 'noms: '+ ' | '.join(dict.fromkeys(safe(n).strip() for n in names if str(n).strip())) + \
        ' adresse: '+ ' | '.join(dict.fromkeys(safe(a).strip() for a in addresses if str(a).strip()))+' '+postcode


def pressure():
    s = subprocess.check_output(['/usr/bin/memory_pressure', '-Q'], text=True, timeout=5)
    match = re.search(r'free percentage:\s*(\d+)%', s)
    level = int(subprocess.check_output(['/usr/sbin/sysctl', '-n',
        'kern.memorystatus_vm_pressure_level'], text=True, timeout=5).strip())
    if match is None: raise RuntimeError('Missing real memory counters')
    return int(match[1]), level


def prepare(out):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend, RetrievalQuery
    out.mkdir(parents=True, exist_ok=False)
    excluded = {q['query_id'] for q in read(PANEL)['queries']}
    # Filter BEFORE materializing labels: neither closed folds nor current-panel GT.
    predicate = ds.field('fold').isin([2, 3, 4]) & ~ds.field('query_id').isin(excluded)
    labels = ds.dataset(LEDGER).to_table(filter=predicate).to_pylist()
    assert len(labels) == 21571 and not excluded & {r['query_id'] for r in labels}
    assert all(r['fold'] in (2, 3, 4) and re.fullmatch(r'\d{14}', r['human_ground_truth_siret'])
        and r['human_ground_truth_siret'].startswith(r['human_ground_truth_siren']) for r in labels)
    groups = grouped_split(labels)
    by_id = {r['query_id']: r for r in labels}
    queries = {}
    for ref in read(GEN)['batches']:
        for shard in read(ref['path'])['shards']:
            source = read(shard['path'])['files']['queries.parquet']['path']
            for q in pq.read_table(source, filters=[('fold', 'in', [2, 3, 4])], use_threads=False).to_pylist():
                if q['query_id'] in by_id: queries[q['query_id']] = q
    assert set(queries) == set(by_id)
    split_rows = {k: [] for k in ('train', 'validation', 'test')}
    for qid, label in by_id.items():
        split_rows[groups[label['human_ground_truth_siren']]].append({**queries[qid],
            'siret': label['human_ground_truth_siret'], 'siren': label['human_ground_truth_siren']})
    for split, rows in split_rows.items():
        write(out/(split+'_membership.json'), rows)
    # Test stays as a sealed membership file; no positive-document hydration or mining.
    backend = TantivyBackend(INDEX); tv = backend._tantivy
    if backend.searcher.num_docs != 82772691: raise RuntimeError('Official corpus drift')
    def doc_text(doc):
        return clean_text([v for f in ('legal_names','trade_names','site_names','historical_names','supporting_names')
            for v in doc.get(f, [])], [v for f in ('addresses','historical_addresses','supporting_addresses')
            for v in doc.get(f, [])], backend._first(doc, 'postcode'))
    for split in ('train', 'validation'):
        rows = split_rows[split]; positives = {}; ids = sorted({r['siret'] for r in rows})
        for start in range(0, len(ids), 128):
            part = ids[start:start+128]
            parsed = tv.Query.term_set_query(backend.index.schema, 'siret', part)
            for _, loc in backend.searcher.search(parsed, limit=len(part)+1, count=False).hits:
                d = backend.searcher.doc(loc).to_dict()
                positives[backend._first(d,'siret')] = doc_text(d)
        dataset = []; absent = []; no_negative = 0
        for i, row in enumerate(rows):
            if row['siret'] not in positives:
                absent.append(row['query_id']); continue
            rq = RetrievalQuery(row['crm_name'], row['crm_street'], row['crm_number'], row['crm_postcode'], row['crm_insee'])
            parsed = backend._combined_query(rq, backend._fielded_name_query(rq.name), document_type='siret')
            negatives = []
            for _, loc in backend.searcher.search(parsed, limit=12, count=False).hits:
                d = backend.searcher.doc(loc).to_dict(); siren = backend._first(d, 'siren')
                # Never mine negatives from another split's labelled SIREN.
                if siren == row['siren'] or (siren in groups and groups[siren] != split): continue
                negatives.append({'text': doc_text(d), 'siren': siren})
                if len(negatives) == 3: break
            if not negatives: no_negative += 1
            dataset.append(dict(query_id=row['query_id'], siren=row['siren'],
                query=clean_text([row['crm_name']], [' '.join(row[k] for k in
                    ('crm_number','crm_suffix','crm_street'))], row['crm_postcode']),
                positive=positives[row['siret']], negatives=negatives))
            if i % 250 == 0:
                write(out/'state.json', dict(status='PREPARING', split=split, rows=i, recall_measured=False))
        write(out/(split+'.json'), dataset)
        write(out/(split+'_coverage.json'), dict(total=len(rows), usable_training_rows=len(dataset),
            missing_official_positive_query_ids=absent, no_lexical_negative=no_negative,
            excluded_from_retrieval_denominator=False))
    write(out/'manifest.json', dict(model='google/embeddinggemma-2', revision=REVISION, seed=SEED,
        rows={k:len(v) for k,v in split_rows.items()}, groups=Counter(groups.values()),
        split_hashes={k:digest(out/(k+'_membership.json')) for k in split_rows},
        training_hashes={k:digest(out/(k+'.json')) for k in ('train','validation')},
        ledger_sha256=digest(LEDGER), current_panel_excluded=2000, safe_folds=[2,3,4],
        test_opened_by_training=False, fresh_pipeline_holdout=False, recall_measured=False,
        historical_development_population=True, downstream_models_changed=False))
    write(out/'state.json', dict(status='PREPARED', recall_measured=False))


def train(out, hours):
    import torch
    import torch.nn.functional as F
    import psutil
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261006)
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    torch.mps.set_per_process_memory_fraction(.16)
    manifest = read(out/'manifest.json')
    rows, validation = read(out/'train.json'), read(out/'validation.json')
    for k in ('train', 'validation'):
        if digest(out/(k+'.json')) != manifest['training_hashes'][k]: raise RuntimeError('Training data drift')
    assert not {r['siren'] for r in rows} & {r['siren'] for r in validation}
    base = HOME/'base'; cfg = AutoConfig.from_pretrained(base, local_files_only=True)
    cfg.vision_config = None; cfg.audio_config = None
    model = AutoModel.from_pretrained(base, config=cfg, dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True).to('mps')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    for p in model.parameters(): p.requires_grad_(False)
    class LoRA(torch.nn.Module):
        def __init__(self, linear):
            super().__init__(); self.base=linear
            self.a=torch.nn.Parameter(torch.empty(8,linear.in_features,device='mps',dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(linear.out_features,8,device='mps',dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):
            y=self.base(x)
            return y+F.linear(F.linear(x.float(),self.a),self.b).to(y.dtype)*2.
    for name, module in list(model.named_modules()):
        if name.rsplit('.',1)[-1] in ('q_proj','v_proj','embedding_projection'):
            parent, field=name.rsplit('.',1); setattr(model.get_submodule(parent),field,LoRA(module))
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters.values(), lr=5e-5, weight_decay=.01)
    def encode(texts):
        tokens=tokenizer(texts,padding=True,truncation=True,max_length=160,return_tensors='pt')
        tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')}
        hidden=model(**tokens).last_hidden_state.float()
        mask=tokens['attention_mask'].unsqueeze(-1)
        return F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
    def loss(batch):
        queries=encode([QUERY_PROMPT+r['query'] for r in batch])
        texts=[DOC_PROMPT+r['positive'] for r in batch]
        owners=[r['siren'] for r in batch]
        for r in batch:
            for n in r['negatives'][:2]: texts.append(DOC_PROMPT+n['text']); owners.append(n['siren'])
        docs=encode(texts); logits=queries@docs.T/.05
        for i,r in enumerate(batch):
            for j,owner in enumerate(owners):
                if owner==r['siren'] and i!=j: logits[i,j]=-1e4
        return F.cross_entropy(logits,torch.arange(len(batch),device='mps'))
    # Fixed diagnostic validation sample selected by query ID, not model outcomes.
    controls=sorted((r for r in validation if r['negatives']),
        key=lambda r:hashlib.sha256(r['query_id'].encode()).digest())[:64]
    def evaluate():
        if not controls: raise RuntimeError('No contrastive validation controls')
        model.eval()
        with torch.no_grad(): value=sum(float(loss([r]).item()) for r in controls)/len(controls)
        model.train(); torch.mps.empty_cache(); return value
    start=time.monotonic(); deadline=start+hours*3600; step=0; best=float('inf'); epoch=0
    checkpoint=out/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for n,p in parameters.items(): p.data.copy_(state['adapter'][n].to('mps'))
        optimizer.load_state_dict(state['optimizer']); step=state['step']; epoch=state['epoch']
        best=state['best']; random.setstate(state['random'])
    else:
        random.seed(20261006); baseline=evaluate(); best=baseline
        write(out/'baseline_diagnostic.json',dict(contrastive_loss=baseline,validation_controls=len(controls),
            retrieval_recall_measured=False, not_retrieval_evaluation=True))
    def save(name):
        state=dict(adapter={n:p.detach().cpu() for n,p in parameters.items()},
            optimizer=optimizer.state_dict(),step=step,epoch=epoch,best=best,random=random.getstate())
        temporary=out/(name+'.tmp'); torch.save(state,temporary); os.replace(temporary,out/name)
    process=psutil.Process(); recent=None
    save('checkpoint.pt')
    while time.monotonic()<deadline and epoch<2:
        indices=list(range(len(rows))); random.Random(20261006+epoch).shuffle(indices)
        cursor=(step % math.ceil(len(rows)/4))*4 if checkpoint.exists() else 0
        for offset in range(cursor,len(indices),4):
            if time.monotonic()>=deadline: break
            # Trainer yields well before retrieval's 50% free-memory guard.
            free, level=pressure()
            if free<62 or level!=1 or process.memory_info().rss>3*1024**3:
                save('checkpoint.pt'); write(out/'state.json',dict(status='YIELD_RESOURCE',step=step,
                    free_percent=free,pressure_level=level,recall_measured=False)); return 75
            batch=[rows[i] for i in indices[offset:offset+4]]
            optimizer.zero_grad(set_to_none=True); value=loss(batch)
            if not torch.isfinite(value): raise RuntimeError('Non-finite loss')
            value.backward(); torch.nn.utils.clip_grad_norm_(parameters.values(),1.)
            optimizer.step(); step+=1; recent=float(value.item()); torch.mps.empty_cache()
            if step%100==0:
                measured=evaluate()
                if measured<best: best=measured; save('best_adapter.pt')
                save('checkpoint.pt')
            if step%10==0:
                write(out/'state.json',dict(status='TRAINING',pid=os.getpid(),step=step,epoch=epoch,
                    loss=recent,best_validation_loss=best,trainable_parameters=sum(p.numel() for p in parameters.values()),
                    mps_driver_bytes=torch.mps.driver_allocated_memory(),rss_bytes=process.memory_info().rss,
                    elapsed_seconds=time.monotonic()-start,recall_measured=False))
        epoch+=1
    measured=evaluate()
    if measured<best: best=measured; save('best_adapter.pt')
    save('checkpoint.pt')
    write(out/'state.json',dict(status='TRAINING_COMPLETE_DIAGNOSTIC_ONLY',step=step,epoch=epoch,
        best_validation_loss=best,final_validation_loss=measured,recall_measured=False,
        final_test_opened=False,downstream_models_changed=False))
    return 0


def run(out, hours):
    import psutil
    out.mkdir(parents=True,exist_ok=True)
    write(out/'supervisor.json',dict(status='STARTING',pid=os.getpid(),revision=REVISION))
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
        'HF_HUB_OFFLINE':'1','TMPDIR':str(HOME)}
    deadline=time.monotonic()+hours*3600; restarts=0
    while time.monotonic()<deadline:
        free,level=pressure()
        if free<72 or level!=1:
            write(out/'supervisor.json',dict(status='WAITING_RESOURCES',pid=os.getpid(),free_percent=free))
            time.sleep(45); continue
        with (out/'training.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_night','train',
                '--output',str(out),'--hours',str(max(.01,(deadline-time.monotonic())/3600))],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(out/'supervisor.json',dict(status='CHILD_RUNNING',pid=os.getpid(),child_pid=child.pid,restarts=restarts))
            resource_stop=False
            while child.poll() is None:
                free,level=pressure()
                rss=psutil.Process(child.pid).memory_info().rss if child.poll() is None else 0
                if free<58 or level!=1 or rss>3*1024**3 or time.monotonic()>deadline:
                    child.terminate()
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    resource_stop=True; break
                time.sleep(1)
            code=75 if resource_stop else child.wait()
        if code==0:
            write(out/'supervisor.json',dict(status='COMPLETE',exit_code=0,pid=os.getpid())); return
        if code!=75 or restarts>=12:
            write(out/'supervisor.json',dict(status='STOP',exit_code=code,pid=os.getpid())); return
        restarts+=1; time.sleep(45)
    write(out/'supervisor.json',dict(status='NIGHT_BUDGET_EXHAUSTED',restarts=restarts,pid=os.getpid()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','train','run'])
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--hours',type=float,default=8)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(HOME): raise ValueError('Dedicated CATNAT sidequest only')
    if args.action=='prepare': prepare(args.output)
    elif args.action=='train': sys.exit(train(args.output,args.hours))
    else: run(args.output,args.hours)
