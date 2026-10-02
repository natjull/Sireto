"""Training-only contrastive representation probe; NOT a retrieval evaluation.

GT documents are supervised training targets, NEVER inserted into an index
or retrieval evaluation. Natural historical candidates are hard negatives
for TRAINING ONLY, not a serving reservoir. No national build is launched.
The recipe is a bounded query adapter baseline, not SC-Block reproduction.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts.refit_retrieval_v50_serving_features import SOURCE
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_learned_v41 import official_text,unit_vectors,fit_query_transform,TrainingMembership,transform_queries
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v51_dense_training.json'


def query_text(raw):
    return official_text([raw['crm_name']],f"{raw['crm_number']} {raw['crm_suffix']} {raw['crm_street']} {raw['crm_postcode']}".strip())


def safe_negatives(raw,held,registry):
    held_entities=set(registry[str(held)])
    if raw['fold']==held or raw['positive_siren'] in held_entities: return []
    return [n for n in raw['natural_negatives'] if n['siren'] not in held_entities and n['siret'] not in raw['acceptable']]


def prepare(output,cfg):
    source=json.loads(verify(SOURCE).read_text()); v38.verify_launch(source)
    if source['status']!='SEALED_OOF_JOINT_INITIAL_METRICS': raise ValueError('Unsealed training authority')
    launch=json.loads(verify(source['launch']).read_text()); v38.verify_launch(launch)
    rows={r['query_id']:r for group in launch['rows_by_held'].values() for r in group}
    if len(rows)!=128 or any(r['fold'] not in (2,3,4) for r in rows.values()): raise ValueError('Training panel drift')
    ids={r['positive_siret'] for r in rows.values()}|{n['siret'] for r in rows.values() for n in r['natural_negatives']}
    with Budget(cfg['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); tv=backend._tantivy
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official index count drift')
        parsed=tv.Query.term_set_query(backend.index.schema,'siret',sorted(ids)); docs={}; parents={}
        for _,loc in backend.searcher.search(parsed,limit=len(ids)+1,count=False).hits:
            doc=backend.searcher.doc(loc).to_dict(); s=backend._first(doc,'siret'); p=backend._first(doc,'siren')
            if s in docs or s not in ids or not s.startswith(p): raise ValueError('Official text identity drift')
            names=[str(value) for field in v38.NAMES for value in doc.get(field,[])]
            addresses=[str(value) for field in ('addresses','historical_addresses','supporting_addresses') for value in doc.get(field,[])]
            docs[s]=official_text(names,' | '.join(dict.fromkeys(addresses))+' '+backend._first(doc,'postcode')); parents[s]=p
        if set(docs)!=ids: raise ValueError('Missing training source documents; do not filter denominator')
        for r in rows.values():
            if parents[r['positive_siret']]!=r['positive_siren']: raise ValueError('Positive membership drift')
            for n in r['natural_negatives']:
                if parents[n['siret']]!=n['siren']: raise ValueError('Negative membership drift')
        texts={qid:query_text(r) for qid,r in rows.items()}
    v38.verify_launch(launch)
    write_json(output/'inputs.json',dict(rows=list(rows.values()),query_texts=texts,document_texts=docs,
        registry=launch['registry'],resources=budget.report(),truth_scope='TRAINER_ONLY',all_positive_documents_preserved=True,
        site_view_policy='OFFICIAL_NAMES_AND_ADDRESSES_ONE_JOINED_VIEW_NOT_TEMPORAL_GRAPH_MODEL'))
    return source


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']
    for ref in [launch['config_ref'],launch['inputs'],*launch['model_refs'].values(),*launch['implementation'].values()]: verify(ref)
    inputs=json.loads(verify(launch['inputs']).read_text()); rows=inputs['rows']; registry=inputs['registry']
    # Isolate Torch/OpenMP from national index/DuckDB runtimes.
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer,assert_tokenizer_healthy
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started=time.monotonic()
    with Budget(cfg['resources']) as budget:
        model=SentenceTransformer(cfg['model_path'],device='cpu',local_files_only=True)
        _repair_exported_tokenizer(model,cfg['model_path']); assert_tokenizer_healthy(model.tokenizer)
        model.max_seq_length=cfg['max_length']
        docids=sorted(inputs['document_texts']); qids=[r['query_id'] for r in rows]
        texts=[inputs['query_texts'][q] for q in qids]+[inputs['document_texts'][s] for s in docids]
        token_counts=[len(model.tokenizer(t)['input_ids']) for t in texts]
        began=time.monotonic()
        encoded=unit_vectors(model.encode(texts,batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False))
        encode_seconds=time.monotonic()-began
        queries=dict(zip(qids,encoded[:len(qids)])); docs=dict(zip(docids,encoded[len(qids):])); reports={}; refs={}
        for held in (2,3,4):
            train=[]; q=[]; p=[]; n=[]
            for r in rows:
                negatives=safe_negatives(r,held,registry)
                if not negatives: continue
                # Mining by the unchanged generic encoder on TRAINING rows,
                # after entity exclusion, not by heldout quality or labels.
                hard=max(negatives,key=lambda d:float(queries[r['query_id']]@docs[d['siret']]))
                train.append(TrainingMembership(r['query_id'],r['fold'],r['positive_siren'],hard['siren']))
                q.append(queries[r['query_id']]); p.append(docs[r['positive_siret']]); n.append(docs[hard['siret']])
            weight,fit=fit_query_transform(np.array(q),np.array(p),np.array(n),train,held,set(registry[str(held)]),
                ridge=cfg['ridge'],negative_weight=cfg['negative_weight'])
            path=output/f'query_transform_fold{held}.npy'; np.save(path,weight,allow_pickle=False); refs[str(held)]=record(path)
            controls=[]
            for r in rows:
                if r['fold']!=held: continue
                natural={d['siret']:d for d in r['natural_negatives'] if d['siret'] not in r['acceptable']}
                if not natural: raise ValueError('No natural contrastive controls; do not drop heldout query')
                base=queries[r['query_id']]; transformed=transform_queries(base[None],weight)[0]
                positive=docs[r['positive_siret']]; negative=np.array([docs[s] for s in sorted(natural)])
                pb=float(base@positive); pa=float(transformed@positive)
                nb=negative@base; na=negative@transformed
                controls.append(dict(query_id=r['query_id'],fold=held,negative_pairs=len(natural),
                    mean_margin_before=float(np.mean(pb-nb)),mean_margin_after=float(np.mean(pa-na)),
                    hardest_margin_before=float(pb-nb.max()),hardest_margin_after=float(pa-na.max()),
                    negative_outrank_fraction_before=float(np.mean(nb>=pb)),negative_outrank_fraction_after=float(np.mean(na>=pa))))
            reports[str(held)]=dict(fit=fit,controls=controls,control_queries=len(controls),
                hardest_margin_before=float(np.mean([r['hardest_margin_before'] for r in controls])),
                hardest_margin_after=float(np.mean([r['hardest_margin_after'] for r in controls])),
                negative_outrank_fraction_before=float(np.mean([r['negative_outrank_fraction_before'] for r in controls])),
                negative_outrank_fraction_after=float(np.mean([r['negative_outrank_fraction_after'] for r in controls])))
    write_json(output/'worker.json',dict(reports=reports,transforms=refs,resources=budget.report(),
        encode_seconds=encode_seconds,seconds=time.monotonic()-started,encoded_queries=len(qids),encoded_documents=len(docids),
        truncated_texts=sum(n>cfg['max_length'] for n in token_counts),max_length=cfg['max_length'],
        recall_measured=False,index_built=False,training_panel_only=True,output_candidates=None,
        no_claim_of_national_recall_or_slo=True,identifiers_as_features=False))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        source=prepare(output,cfg)
        model_root=Path(cfg['model_path']); model_refs={str(p.relative_to(model_root)):record(p) for p in sorted(model_root.rglob('*')) if p.is_file()}
        if not model_refs: raise ValueError('Local generic model unavailable')
        paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_learned_v41.py',ROOT/'src/xgb_matcher/semantic.py',
            ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'src/xgb_matcher/hierarchical_retrieval.py',Path(v38.__file__),
            ROOT/'scripts/refit_retrieval_v50_serving_features.py']
        launch=dict(config=cfg,config_ref=record(CONFIG),inputs=record(output/'inputs.json'),model_refs=model_refs,
            implementation={str(p):record(p) for p in paths},training_authority=SOURCE)
        write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_v51_dense_training','worker','--output',str(output)],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(cfg['resources']['max_seconds'],max(1,180-(time.monotonic()-began))),
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        for ref in [launch['config_ref'],launch['inputs'],*launch['model_refs'].values(),*launch['implementation'].values()]: verify(ref)
        v38.verify_launch(source)
        report=json.loads((output/'worker.json').read_text())
        if not report['resources']['pass_resource_policy']: raise ValueError('Cost policy failed')
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_DENSE_MARGIN_PROBE',launch=record(output/'launch.json'),
            worker=record(output/'worker.json'),seconds=time.monotonic()-began,recall_measured=False,index_built=False,
            national_build_launched=False,fold0_opened=False,fold1_opened=False,ready_for_fold1=False,
            downstream_models_unchanged=True,full_query_population=False))
        print(json.dumps(dict(reports={f:{k:v for k,v in r.items() if k not in ('controls','fit')} for f,r in report['reports'].items()},
            encoded_documents=report['encoded_documents'],encode_seconds=report['encode_seconds'],truncated_texts=report['truncated_texts'],recall_measured=False)))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker']); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); run(a.output) if a.mode=='run' else worker(a.output)
