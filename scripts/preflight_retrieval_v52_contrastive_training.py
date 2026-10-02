"""Larger training-only structured contrastive probe; no recall is computed."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_contrastive_v52 import surface,combine,fit
from src.xgb_matcher.retrieval_learned_v41 import transform_queries
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v52_contrastive_training.json'


def prepare(output,cfg):
    generation=v38.load(v38.GEN,v38.GEN_SHA); ev=v38.load(v38.EVAL,v38.EVAL_SHA)
    labels=pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),filters=[('fold','in',[2,3,4])]).to_pylist()
    labelmap={r['query_id']:r for r in labels}
    registry={str(f):sorted({r['human_ground_truth_siren'] for r in labels if r['fold']==f}) for f in (2,3,4)}
    side=json.loads(verify(ev['label_side']).read_text())
    for ref in side['files'].values(): verify(ref)
    acceptable=defaultdict(set)
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        acceptable[r['query_id']].add(r['siret'])
    queries=[]; candidates=defaultdict(dict); refs=[]
    for b in range(cfg['first_batch'],cfg['first_batch']+cfg['batches']):
        ref=generation['batches'][b]; batch=json.loads(verify(ref).read_text()); refs.append(ref)
        if batch['start']!=b*32 or batch['count']!=32 or len(batch['shards'])!=1: raise ValueError('Source batch drift')
        sr=batch['shards'][0]; shard=json.loads(verify(sr).read_text()); refs.append(sr)
        if shard['status']!='SEALED_PAIRED_SHARD' or shard['truth_columns_opened'] or shard['quality_measured']:
            raise ValueError('Historical mining source must be sealed label blind')
        qref=shard['files']['queries.parquet']; dref=shard['files']['direct.parquet']; jref=shard['files']['joint.parquet']
        refs.extend([qref,dref,jref]); part=pq.read_table(verify(qref)).to_pylist(); queries.extend(part)
        batchids={r['query_id'] for r in part}
        batchfold={r['query_id']:r['fold'] for r in part}
        if len(part)!=32 or any(set(r)!=v38.QUERY_FIELDS or r['fold'] not in (2,3,4) for r in part): raise ValueError('Forbidden query input')
        direct=pq.read_table(verify(dref),filters=[('channel_rank','<=',cfg['direct_rank_per_channel'])]).to_pylist()
        joint=pq.read_table(verify(jref),filters=[('candidate_rank','<=',cfg['joint_rank'])]).to_pylist()
        for r in direct:
            if r['relation_type']!='SIRET_NATIVE' or r['relation_direction'] or r['query_id'] not in batchids or r['fold']!=batchfold[r['query_id']]: raise ValueError('Non-native negative source')
            candidates[r['query_id']][r['source_siret']]=r['siren']
        for r in joint:
            if r['query_id'] not in batchids or r['fold']!=batchfold[r['query_id']]: raise ValueError('Joint query leakage')
            candidates[r['query_id']][r['siret']]=r['siren']
    if len(queries)!=cfg['training_queries'] or len({q['query_id'] for q in queries})!=len(queries): raise ValueError('Training denominator drift')
    rows=[]
    for raw in queries:
        qid=raw['query_id']; label=labelmap[qid]
        if raw['fold']!=label['fold'] or not acceptable[qid]: raise ValueError('Official training label-side drift')
        negatives=[dict(siret=s,siren=p) for s,p in sorted(candidates[qid].items()) if s not in acceptable[qid]]
        if not negatives: raise ValueError('No safe natural negative; do not filter query')
        rows.append(dict(raw,positive_siret=label['human_ground_truth_siret'],positive_siren=label['human_ground_truth_siren'],negatives=negatives))
    ids=sorted({r['positive_siret'] for r in rows}|{n['siret'] for r in rows for n in r['negatives']})
    fields={}; masked=0
    with Budget(cfg['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); tv=backend._tantivy
        expected_index=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
        if backend.searcher.num_docs!=expected_index['num_documents']: raise ValueError('Official index cardinality drift')
        for start in range(0,len(ids),4096):
            subset=ids[start:start+4096]
            for _,loc in backend.searcher.search(tv.Query.term_set_query(backend.index.schema,'siret',subset),limit=len(subset)+1,count=False).hits:
                doc=backend.searcher.doc(loc).to_dict(); s=backend._first(doc,'siret'); p=backend._first(doc,'siren')
                if s in fields or s not in subset or not s.startswith(p): raise ValueError('Source identity drift')
                names=[str(v) for f in v38.NAMES for v in doc.get(f,[])]
                addresses=[str(v) for f in v38.ADDRESSES for v in doc.get(f,[])]
                addresses.append(backend._first(doc,'postcode'))
                name,nm=surface(names,'nom'); address,am=surface(addresses,'adresse'); masked+=nm+am
                fields[s]=dict(name=name,address=address,siren=p)
        if set(fields)!=set(ids): raise ValueError('Source document absent; do not shrink denominator')
        qfields={}
        for r in rows:
            if fields[r['positive_siret']]['siren']!=r['positive_siren'] or any(fields[n['siret']]['siren']!=n['siren'] for n in r['negatives']): raise ValueError('Membership drift')
            name,nm=surface([r['crm_name']],'nom')
            address,am=surface([' '.join(r[k] for k in ('crm_number','crm_suffix','crm_street','crm_postcode'))],'adresse'); masked+=nm+am
            qfields[r['query_id']]=dict(name=name,address=address)
    im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
    authority=dict(index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
        index_documents=im['num_documents'],inventory=v38.inventory(),source_refs=refs,ledger=record(v38.LEDGER),label_side=ev['label_side'])
    write_json(output/'inputs.json',dict(rows=rows,registry=registry,query_fields=qfields,document_fields=fields,masked_identifier_spans=masked,
        resources=budget.report(),training_queries=len(rows),truth_scope='TRAINER_ONLY',all_positive_documents_preserved=True,
        site_view_policy='SEPARATE_NAME_ADDRESS_FIXED_EMBEDDINGS_NO_TEMPORAL_GRAPH_IMPLEMENTED'))
    return authority


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; v38.verify_launch(launch)
    for ref in [launch['config_ref'],launch['inputs'],*launch['model_refs'].values()]: verify(ref)
        inputs=json.loads(verify(launch['inputs']).read_text()); rows=inputs['rows']; registry=inputs['registry']
    import torch
    from sentence_transformers import SentenceTransformer
    from src.xgb_matcher.semantic import _repair_exported_tokenizer,assert_tokenizer_healthy
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started=time.monotonic()
    with Budget(cfg['resources']) as budget:
        model=SentenceTransformer(cfg['model_path'],device='cpu',local_files_only=True)
        _repair_exported_tokenizer(model,cfg['model_path']); assert_tokenizer_healthy(model.tokenizer); model.max_seq_length=cfg['max_length']
        qids=[r['query_id'] for r in rows]; dids=sorted(inputs['document_fields'])
        surfaces=[inputs['query_fields'][q] for q in qids]+[inputs['document_fields'][s] for s in dids]
        texts=[r['name'] for r in surfaces]+[r['address'] for r in surfaces]
        token_counts=[len(model.tokenizer(t)['input_ids']) for t in texts]
        model.encode(texts[:16],batch_size=16,show_progress_bar=False)
        before=time.monotonic(); head=model.encode(texts[:256],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        estimated=(time.monotonic()-before)*len(texts)/256
        if estimated>cfg['max_estimated_encoding_seconds']: raise ValueError('STOP_ENCODING_COST_PREFLIGHT')
        remainder=model.encode(texts[256:],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        encoded=np.concatenate([head,remainder]); encode_seconds=time.monotonic()-before
        length=len(surfaces); vectors=combine(encoded[:length],encoded[length:])
        query=dict(zip(qids,vectors[:len(qids)])); docs=dict(zip(dids,vectors[len(qids):]))
        q=np.array([query[r['query_id']] for r in rows]); p=np.array([docs[r['positive_siret']] for r in rows])
        negatives=[np.array([docs[n['siret']] for n in r['negatives']]) for r in rows]
        memberships=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren'],negative_sirens=[n['siren'] for n in r['negatives']]) for r in rows]
        reports={}; transforms={}
        for held in (2,3,4):
            began=time.monotonic(); w,fitted=fit(q,p,negatives,memberships,held,registry,ridge=cfg['ridge'],temperature=cfg['temperature'],
                hard_count=cfg['hard_count'],maxiter=cfg['maxiter'])
            path=output/f'query_transform_fold{held}.npy'; np.save(path,w,allow_pickle=False); transforms[str(held)]=record(path)
            controls=[]
            for i,r in enumerate(rows):
                if r['fold']!=held: continue
                adapted=transform_queries(q[i:i+1],w)[0]; base=q[i]; positive=p[i]; negative=negatives[i]
                pb=float(base@positive); pa=float(adapted@positive); nb=negative@base; na=negative@adapted
                controls.append(dict(query_id=r['query_id'],fold=held,negative_pairs=len(negative),
                    hardest_margin_before=float(pb-nb.max()),hardest_margin_after=float(pa-na.max()),
                    negative_outrank_fraction_before=float(np.mean(nb>=pb)),negative_outrank_fraction_after=float(np.mean(na>=pa))))
            reports[str(held)]=dict(fit=fitted,controls=controls,control_queries=len(controls),fit_seconds=time.monotonic()-began,
                hardest_margin_before=float(np.mean([r['hardest_margin_before'] for r in controls])),
                hardest_margin_after=float(np.mean([r['hardest_margin_after'] for r in controls])),
                negative_outrank_fraction_before=float(np.mean([r['negative_outrank_fraction_before'] for r in controls])),
                negative_outrank_fraction_after=float(np.mean([r['negative_outrank_fraction_after'] for r in controls])))
    # Only training representations saved; no index created or recall measured.
    np.save(output/'training_query_vectors.npy',q,allow_pickle=False)
    np.save(output/'training_document_vectors.npy',vectors[len(qids):],allow_pickle=False)
    write_json(output/'worker.json',dict(reports=reports,transforms=transforms,resources=budget.report(),
        query_vectors=record(output/'training_query_vectors.npy'),document_vectors=record(output/'training_document_vectors.npy'),
        document_ids=dids,query_ids=qids,encoding_estimate_seconds=estimated,encode_seconds=encode_seconds,seconds=time.monotonic()-started,
        encoded_queries=len(qids),encoded_documents=len(dids),dimension=vectors.shape[1],truncated_texts=sum(n>cfg['max_length'] for n in token_counts),
        masked_identifier_spans=inputs['masked_identifier_spans'],recall_measured=False,index_built=False,training_panel_only=True,
        output_candidates=None,no_claim_of_national_recall_or_slo=True,identifiers_as_features=False))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        authority=prepare(output,cfg); modelroot=Path(cfg['model_path'])
        paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_contrastive_v52.py',ROOT/'src/xgb_matcher/retrieval_learned_v41.py',
            ROOT/'src/xgb_matcher/semantic.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',Path(v38.__file__),
            ROOT/'src/xgb_matcher/hierarchical_retrieval.py',ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py']
        launch=dict(**authority,config=cfg,config_ref=record(CONFIG),inputs=record(output/'inputs.json'),
            implementation={str(p):record(p) for p in paths},model_refs={str(p.relative_to(modelroot)):record(p) for p in sorted(modelroot.rglob('*')) if p.is_file()})
        write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_v52_contrastive_training','worker','--output',str(output)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=min(cfg['resources']['max_seconds'],max(1,cfg['controller_seconds']-(time.monotonic()-began))),
                env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
                    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2'})
        v38.verify_launch(launch)
        for ref in [*launch['source_refs'],launch['inputs'],launch['config_ref'],*launch['model_refs'].values()]: verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_STRUCTURED_CONTRASTIVE_PROBE',launch=record(output/'launch.json'),
            worker=record(output/'worker.json'),seconds=time.monotonic()-began,recall_measured=False,index_built=False,national_build_launched=False,
            fold0_opened=False,fold1_opened=False,ready_for_fold1=False,downstream_models_unchanged=True,full_query_population=False))
        report=json.loads((output/'worker.json').read_text())
        print(json.dumps(dict(reports={f:{k:v for k,v in r.items() if k not in ('controls','fit')} for f,r in report['reports'].items()},
            encoded_documents=report['encoded_documents'],encode_seconds=report['encode_seconds'],seconds=report['seconds'],recall_measured=False)))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker']); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); run(a.output) if a.mode=='run' else worker(a.output)
