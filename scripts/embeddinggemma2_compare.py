"""Raw/best-adapter diagnostic and exact dense search over COMPLETE CRM communes.

Validation only. Not a national benchmark, not a replacement for initial100.
Generation process only sees sanitized CRM + naturally enumerated official sites.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'night_20261006_a'
ADAPTER = io.HOME/'cached_20261007_a/best_adapter.pt'
OUTPUT = io.HOME/'comparison_v1'
SEED = 'embeddinggemma2-zone-probe-v1'
CRM = ('query_id','fold','crm_name','crm_number','crm_suffix','crm_street','crm_postcode','crm_insee')


def zone_query(backend, zone):
    tv=backend._tantivy
    return tv.Query.boolean_query([(tv.Occur.Must,tv.Query.term_query(backend.index.schema,f,v,'basic'))
        for f,v in (('document_type','siret'),('insee',zone))])


def document_text(backend, doc):
    return io.clean_text([v for f in ('legal_names','trade_names','site_names','historical_names','supporting_names')
        for v in doc.get(f,[])], [v for f in ('addresses','historical_addresses','supporting_addresses')
        for v in doc.get(f,[])], backend._first(doc,'postcode'))


def query_text(row):
    return io.clean_text([row['crm_name']],[' '.join(row[k] for k in ('crm_number','crm_suffix','crm_street'))],row['crm_postcode'])


def inventory():
    return [(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted((io.INDEX/'tantivy').iterdir()) if p.is_file()]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=io.read(SOURCE/'manifest.json')
    if io.digest(SOURCE/'validation_membership.json')!=manifest['split_hashes']['validation']:
        raise ValueError('Validation membership drift')
    # Membership has existing labels: selection uses ONLY CRM fields, never outcomes/GT.
    queries=[{k:r[k] for k in CRM} for r in io.read(SOURCE/'validation_membership.json')]
    if len(queries)!=2212 or any(q['fold'] not in (2,3,4) for q in queries): raise ValueError('Safe validation only')
    counts=Counter(q['crm_insee'] for q in queries if q['crm_insee'])
    backend=TantivyBackend(io.INDEX)
    if backend.searcher.num_docs!=82772691: raise ValueError('Full official index drift')
    before=inventory(); zones=[]; inspected=[]
    for zone in sorted((z for z in counts if counts[z]>=4),key=lambda z:hashlib.sha256((SEED+'|'+z).encode()).digest()):
        n=backend.searcher.search(zone_query(backend,zone),limit=1,count=True).count
        inspected.append(dict(zone=zone,queries=counts[zone],official_sites=n))
        if 1000<=n<=12000: zones.append(dict(zone=zone,queries=counts[zone],official_sites=n))
        if len(zones)==6: break
    if len(zones)!=6: raise ValueError('Six predeclared bounded communes required')
    selected={z['zone'] for z in zones}
    io.write(out/'panel.json',[q for q in queries if q['crm_insee'] in selected])
    corpora=[]
    for zone in zones:
        result=backend.searcher.search(zone_query(backend,zone['zone']),limit=zone['official_sites']+1,count=True)
        if result.count!=zone['official_sites'] or len(result.hits)!=result.count: raise ValueError('Incomplete commune')
        docs=[]
        for _,loc in result.hits:
            d=backend.searcher.doc(loc).to_dict()
            docs.append(dict(siret=backend._first(d,'siret'),siren=backend._first(d,'siren'),text=document_text(backend,d)))
        docs.sort(key=lambda d:d['siret'])
        if len({d['siret'] for d in docs})!=len(docs): raise ValueError('Duplicate official site')
        path=out/(zone['zone']+'.json'); io.write(path,docs); corpora.append({**zone,'file':record(path)})
    if inventory()!=before: raise ValueError('Index changed during extraction')
    io.write(out/'launch.json',dict(panel=record(out/'panel.json'),corpora=corpora,selection_inspected=inspected,
        seed=SEED,policy='FIRST6_HASH_CRM_COMMUNES_MIN4_VALIDATION_QUERIES_1000_TO12000_ALL_INDEXED_SITES',
        adapter=record(ADAPTER),training_manifest=record(SOURCE/'manifest.json'),validation=record(SOURCE/'validation.json'),
        code=record(Path(__file__)),training_text_builder=record(Path(io.__file__)),
        index_manifest=record(io.INDEX/'manifest.json'),index_meta=record(io.INDEX/'tantivy/meta.json'),
        index_inventory=before,index_segment_full_hashes_verified=False,base_revision=io.REVISION,
        scope='DEVELOPMENT_VALIDATION_COMPLETE_COMMUNES_NOT_NATIONAL_NOT_PRISTINE_PIPELINE_HOLDOUT',
        geometry_used=False,all_sites_of_siren_used=False,test_opened=False,fold0_opened=False,fold1_opened=False,
        serving_changed=False,downstream_models_changed=False))


def top100(previous, scores, sites):
    # Streaming EXACT search across all official corpus rows, no lexical pool or reranker.
    combined=previous+[(float(s),d['siret'],d['siren']) for s,d in zip(scores,sites)]
    return sorted(combined,key=lambda x:(-x[0],x[1]))[:100]


def load_encoder(adapter):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.manual_seed(20261006)
    torch.mps.set_per_process_memory_fraction(.25)
    config=AutoConfig.from_pretrained(io.HOME/'base',local_files_only=True)
    config.audio_config=None; config.vision_config=None
    model=AutoModel.from_pretrained(io.HOME/'base',config=config,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('mps')
    tokenizer=AutoTokenizer.from_pretrained(io.HOME/'base',local_files_only=True)
    class LoRA(torch.nn.Module):
        def __init__(self,linear):
            super().__init__(); self.base=linear
            self.a=torch.nn.Parameter(torch.empty(8,linear.in_features,device='mps',dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(linear.out_features,8,device='mps',dtype=torch.float32))
        def forward(self,x):
            return self.base(x)+2*F.linear(F.linear(x.float(),self.a),self.b).to(x.dtype)
    for name,module in list(model.named_modules()):
        if name.rsplit('.',1)[-1] in ('q_proj','v_proj','embedding_projection'):
            parent,key=name.rsplit('.',1); setattr(model.get_submodule(parent),key,LoRA(module))
    state=torch.load(adapter,map_location='cpu',weights_only=True)
    params=dict(model.named_parameters())
    if set(state['adapter'])!={n for n in params if n.endswith(('.a','.b'))}: raise ValueError('Adapter mismatch')
    def mode(tuned):
        for name,value in state['adapter'].items():
            params[name].data.copy_(value.to('mps') if tuned or name.endswith('.a') else torch.zeros_like(params[name]))
        model.eval()
    def encode(texts):
        batches=[]
        with torch.inference_mode():
            for i in range(0,len(texts),16):
                tokens=tokenizer(texts[i:i+16],padding=True,truncation=True,max_length=160,return_tensors='pt')
                tokens={k:v.to('mps') for k,v in tokens.items() if k in ('input_ids','attention_mask')}
                hidden=model(**tokens).last_hidden_state.float(); mask=tokens['attention_mask'].unsqueeze(-1)
                e=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1)
                batches.append(e.cpu().numpy())
        torch.mps.empty_cache()
        return np.concatenate(batches)
    return mode,encode,tokenizer,state['step']


def guard(start):
    import psutil
    free,level=io.pressure()
    if free<50 or level!=1 or psutil.Process().memory_info().rss>5*1024**3 or time.monotonic()-start>1500:
        raise RuntimeError('STOP_RESOURCE_OR_25MIN_BUDGET')


def diagnostic(out, mode, encode, launch):
    # Supervised validation contrastive diagnostic, NEVER claimed as retrieval recall.
    rows=io.read(verify(launch['validation'])); texts=[]
    for r in rows: texts.extend([io.QUERY_PROMPT+r['query'],io.DOC_PROMPT+r['positive'],*[io.DOC_PROMPT+n['text'] for n in r['negatives'][:2]]])
    texts=list(dict.fromkeys(texts)); start=time.monotonic(); result={}
    for tuned in (False,True):
        mode(tuned); vectors={}
        for i in range(0,len(texts),128):
            vectors.update(zip(texts[i:i+128],encode(texts[i:i+128]))); guard(start)
            io.write(out/'state.json',dict(status='CONTRASTIVE_DIAGNOSTIC_NOT_RECALL',tuned=tuned,encoded=min(i+128,len(texts)),total=len(texts)))
        scores=[]; losses=[]; with_neg=0
        for r in rows:
            negatives=r['negatives'][:2]
            if not negatives: continue
            q=vectors[io.QUERY_PROMPT+r['query']]; d=np.stack([vectors[io.DOC_PROMPT+r['positive']],*[vectors[io.DOC_PROMPT+n['text']] for n in negatives]])
            s=d@q; scores.append(float(s[0])>max(float(x) for x in s[1:])); logits=s/.05
            losses.append(float(np.log(np.exp(logits-logits.max()).sum())+logits.max()-logits[0])); with_neg+=1
        result['tuned' if tuned else 'raw']=dict(rows=len(rows),rows_with_negatives=with_neg,rows_without_negatives=len(rows)-with_neg,
            positive_beats_all_local_bm25_negatives=sum(scores),pairwise_success_rate=sum(scores)/len(scores),mean_contrastive_loss=float(np.mean(losses)))
    io.write(out/'contrastive_report.json',dict(status='ALL_USABLE_VALIDATION_DIAGNOSTIC_NOT_RETRIEVAL',variants=result,
        never_full_corpus_recall=True,test_opened=False,downstream_models_changed=False))


def generate(out):
    launch=io.read(out/'launch.json')
    for ref in (launch['code'],launch['training_text_builder'],launch['index_manifest'],launch['index_meta'],launch['training_manifest']): verify(ref)
    if inventory()!=[tuple(r) for r in launch['index_inventory']]: raise ValueError('Index inventory drift')
    # This process does not open validation_membership/test/labels or supervised pairs.
    panel=io.read(verify(launch['panel'])); mode,encode,tokenizer,step=load_encoder(verify(launch['adapter']))
    start=time.monotonic(); outputs=[]; truncated=Counter(); unique=Counter()
    for tuned in (False,True):
        mode(tuned); variant='tuned' if tuned else 'raw'; by_id={}
        for zone in launch['corpora']:
            queries=[q for q in panel if q['crm_insee']==zone['zone']]; docs=io.read(verify(zone['file']))
            texts=list(dict.fromkeys(io.DOC_PROMPT+d['text'] for d in docs)); unique[variant]+=len(texts)
            qe=encode([io.QUERY_PROMPT+query_text(q) for q in queries]); vectors={}
            for i in range(0,len(texts),128):
                batch=texts[i:i+128]; vectors.update(zip(batch,encode(batch))); guard(start)
                truncated[variant]+=sum(len(x)>160 for x in tokenizer(batch,truncation=False)['input_ids'])
                io.write(out/'state.json',dict(status='COMPLETE_COMMUNE_DENSE_GENERATION',variant=variant,zone=zone['zone'],
                    encoded=min(i+128,len(texts)),unique_zone_texts=len(texts),seconds=time.monotonic()-start))
            best=[[] for q in queries]
            for i in range(0,len(docs),256):
                chunk=docs[i:i+256]; de=np.stack([vectors[io.DOC_PROMPT+d['text']] for d in chunk]); similarities=qe@de.T
                for j in range(len(queries)): best[j]=top100(best[j],similarities[j],chunk)
            for q,top in zip(queries,best):
                if len(top)!=min(100,len(docs)) or len({x[1] for x in top})!=len(top): raise ValueError('Cap/duplicate')
                by_id[q['query_id']]=dict(query_id=q['query_id'],fold=q['fold'],zone=zone['zone'],variant=variant,
                    candidates=[dict(score=s,siret=t,siren=n) for s,t,n in top])
        outputs.extend(by_id[q['query_id']] for q in panel)
    if inventory()!=[tuple(r) for r in launch['index_inventory']]: raise ValueError('Index changed')
    io.write(out/'outputs.json',outputs)
    io.write(out/'generation.json',dict(status='SEALED_ALL_SELECTED_VALIDATION_BOTH_VARIANTS_COMPLETE_COMMUNES',
        queries=len(panel),outputs=record(out/'outputs.json'),seconds=time.monotonic()-start,adapter_step=step,
        unique_encoded_document_texts=unique,truncated_document_texts=truncated,truth_opened=False,test_opened=False))


def evaluate(out):
    import pyarrow.parquet as pq
    launch=io.read(out/'launch.json'); gen=io.read(out/'generation.json'); outputs=io.read(verify(gen['outputs']))
    supervision=io.read(out/'generation_exit.json')
    if supervision.get('exit_code')!=0 or not supervision.get('own_child_exit_confirmed'): raise ValueError('Actual generator completion required')
    panel=io.read(verify(launch['panel'])); scope={q['query_id'] for q in panel}
    manifest=io.read(verify(launch['training_manifest']))
    if io.digest(SOURCE/'validation_membership.json')!=manifest['split_hashes']['validation']: raise ValueError('Membership drift')
    labels={r['query_id']:r for r in io.read(SOURCE/'validation_membership.json') if r['query_id'] in scope}
    evconfig=io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    ev=io.read(verify(evconfig['evaluation'])); side=io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True or side['ledger']!=evconfig['ledger']: raise ValueError('Sealed label-side required')
    for ref in (side['policy'],*side['files'].values()): verify(ref)
    accepts=pq.read_table(side['files']['acceptable_operational.parquet']['path'],filters=[('fold','in',[2,3,4]),('query_id','in',sorted(scope))],use_threads=False).to_pylist()
    acceptable={qid:{labels[qid]['siret']} for qid in scope}
    for r in accepts: acceptable[r['query_id']].add(r['siret'])
    metrics={}; outcomes={}
    for variant in ('raw','tuned'):
        rows=[r for r in outputs if r['variant']==variant]
        if {r['query_id'] for r in rows}!=scope or len(rows)!=len(scope): raise ValueError('Whole predeclared panel')
        values=[]
        for r in rows:
            gt=labels[r['query_id']]; cs=r['candidates']; ids={d['siret'] for d in cs}
            values.append(dict(query_id=r['query_id'],zone=r['zone'],exact_hit=gt['siret'] in ids,
                siren_hit=gt['siren'] in {d['siren'] for d in cs},operational_hit=bool(ids&acceptable[r['query_id']]),
                exact_rank=next((i+1 for i,d in enumerate(cs) if d['siret']==gt['siret']),None)))
        outcomes[variant]=values
        metrics[variant]=dict(queries=len(values),**{k:sum(v[k] for v in values) for k in ('exact_hit','siren_hit','operational_hit')})
    raw={r['query_id']:r for r in outcomes['raw']}; tuned={r['query_id']:r for r in outcomes['tuned']}
    corpus_ids={z['zone']:{d['siret'] for d in io.read(verify(z['file']))} for z in launch['corpora']}
    outside=[q['query_id'] for q in panel if labels[q['query_id']]['siret'] not in corpus_ids[q['crm_insee']]]
    io.write(out/'outcomes.json',outcomes)
    io.write(out/'report.json',dict(status='MEASURED_SMALL_COMPLETE_COMMUNE_DENSE_VALIDATION_NOT_NATIONAL_GATE',
        variants=metrics,gained=sum(tuned[q]['exact_hit'] and not raw[q]['exact_hit'] for q in scope),
        lost=sum(raw[q]['exact_hit'] and not tuned[q]['exact_hit'] for q in scope),gt_outside_commune_partition=outside,
        all_queries_in_denominator=True,national_recall_measured=False,production_latency_measured=False,
        combined_initial100_recall_measured=False,operational_sidecar=ev['label_side'],generation=record(out/'generation.json'),
        test_opened=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False))


def diagnostic_worker(out):
    launch=io.read(out/'launch.json'); mode,encode,_,_=load_encoder(verify(launch['adapter']))
    diagnostic(out,mode,encode,launch)


def run(out):
    prepare(out)
    for action,timeout in (('diagnostic',1560),('generate',1560)):
        with (out/(action+'.log')).open('x') as log:
            child=subprocess.run([sys.executable,'-u','-m','scripts.embeddinggemma2_compare',action,'--output',str(out)],
                stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        io.write(out/(('generation' if action=='generate' else action)+'_exit.json'),dict(own_child_exit_confirmed=True,exit_code=child.returncode))
        child.check_returncode()
    evaluate(out)
    io.write(out/'state.json',dict(status='COMPLETE_VALIDATION_COMPARISON',test_opened=False,downstream_models_changed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','diagnostic','generate','evaluate']); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated comparison namespace')
    {'run':run,'diagnostic':diagnostic_worker,'generate':generate,'evaluate':evaluate}[a.action](a.output)
