"""Training-only OOF semantic sparse query decoder on existing official index."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_sparse_query_v59 import vocabulary,targets,unit,select,expand,compiled_component,semantic_score,field_bags
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json
CONFIG=ROOT/'config/retrieval_v59_sparse_query.json'

def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; v38.verify_launch(launch)
    source=json.loads(verify(cfg['vector_source']).read_text()); sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    data=json.loads(verify(sl['inputs']).read_text()); cached=json.loads(verify(source['worker']).read_text())
    proof=json.loads(verify(cfg['proof_source']).read_text()); pl=json.loads(verify(proof['launch']).read_text()); v38.verify_launch(proof)
    rows=data['rows']; scenes=[s for ref in proof['features'] for s in json.loads(verify(ref).read_text())['scenes']]
    if source['recall_measured'] or proof['recall_measured'] or len(rows)!=512 or len(scenes)!=512: raise ValueError('Training-only source drift')
    if [r['query_id'] for r in rows]!=cached['query_ids'] or any(
        any(r[k]!=s[k] for k in ('query_id','fold','positive_siren')) or len(r['negatives'])!=len(s['negatives']) for r,s in zip(rows,scenes)):
        raise ValueError('Training authority alignment drift')
    q=unit(np.load(verify(cached['query_vectors']),allow_pickle=False)[:,:384])
    with Budget(cfg['resources']) as budget:
        backend=TantivyBackend(v38.INDEX); documents={}; tv=backend._tantivy; ids=cached['document_ids']
        if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official catalog count drift')
        for start in range(0,len(ids),2048):
            subset=ids[start:start+2048]
            for _,loc in backend.searcher.search(tv.Query.term_set_query(backend.index.schema,'siret',subset),limit=len(subset)+1,count=False).hits:
                doc=backend.searcher.doc(loc).to_dict(); s=backend._first(doc,'siret'); p=backend._first(doc,'siren')
                if s not in subset or s in documents or not s.startswith(p): raise ValueError('Document identity drift')
                documents[s]={f:doc.get(f,[]) for f in v38.NAMES}
        if set(documents)!=set(ids): raise ValueError('Missing source document; do not shrink training')
        vocab=vocabulary(documents,cfg['vocabulary_cap'],cfg['minimum_document_frequency'],cfg['maximum_document_fraction'])
        if len(vocab)<64: raise ValueError('Insufficient official word vocabulary')
        y=targets(rows,documents,vocab); documents={s:field_bags(fields) for s,fields in documents.items()}; reports={}; refs={}
        for held in (2,3,4):
            projection,reports[str(held)]=select(q,y,vocab,rows,scenes,documents,data['registry'],held,cfg['ridges'],cfg['alphas'])
            path=output/f'query_word_projection_fold{held}.npy'; np.save(path,projection,allow_pickle=False); refs[str(held)]=record(path)
            # First4 predetermined TRAINING rows: exact same field membership,
            # place gate and weighted terms at training and native execution.
            for i in range(4):
                from scripts import retrieval_v42_indexed_metric as v42
                terms=expand(q[i],projection,vocab,rows[i]['crm_name']); part=compiled_component(backend,v42.as_query(rows[i]),terms)
                if part is None: continue
                ids_i=[rows[i]['positive_siret'],*[n['siret'] for n in rows[i]['negatives']]]
                gate=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',ids_i),0.)
                scores={backend._first(backend.searcher.doc(loc).to_dict(),'siret'):float(score)
                    for score,loc in backend.searcher.search(tv.Query.boolean_query([(tv.Occur.Must,gate),(tv.Occur.Must,part)]),limit=len(ids_i),count=False).hits}
                featuremap={rows[i]['positive_siret']:scenes[i]['positive_features'],
                    **{n['siret']:p['features'] for n,p in zip(rows[i]['negatives'],scenes[i]['negatives'])}}
                if any(abs(scores.get(s,0.)-semantic_score(documents[s],terms,featuremap[s][4]))>1e-5 for s in ids_i):
                    raise ValueError('Sparse native/training feature parity failure')
        write_json(output/'vocabulary.json',vocab)
    v38.verify_launch(launch); verify(cfg['vector_source']); verify(cfg['proof_source'])
    write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_SEMANTIC_SPARSE_QUERY_PROBE',reports=reports,projections=refs,
        vocabulary=record(output/'vocabulary.json'),vocabulary_words=len(vocab),official_training_documents=len(documents),
        source_scope='OFFICIAL_INDEX_TRAINING_CORPUS_ONLY_NOT_FULL_NATIONAL_VOCABULARY',query_representation='V52_CACHED_NORMALIZED_NAME_EMBEDDING_384',
        source_refs=dict(vectors=cfg['vector_source'],proofs=cfg['proof_source']),launch=record(output/'launch.json'),
        resources=budget.report(),seconds=time.monotonic()-launch['monotonic_start'],recall_measured=False,
        full_national_search_performed=False,output_candidates=False,index_built=False,fold0_opened=False,fold1_opened=False,
        ready_for_fold1=False,downstream_models_unchanged=True,training_serving_semantic_parity_audited=True,
        no_crm_alias_dictionary=True,no_identifier_features=True,
        limitations=['NAME_TRANSLATION_ONLY_OTHER_NATIVE_SITE_PROOFS_RETAINED','BAN_GRAPH_TIME_LEGAL_CATEGORY_NOT_ADDED_TO_THIS_ABLATION',
            'INNER_HEAD_CRITERION_NOT_RECALL100','CACHED_NAME_ENCODER_PREFIX_AND_TRUNCATION_MUST_MATCH_SERVING',
            'NATIVE_PLACE_GATE_FOR_ADDED_TERMS_ONLY_NOT_A_GLOBAL_CORPUS_FILTER']))
    print(json.dumps(dict(vocabulary=len(vocab),reports={f:dict(selected=r['selected'],before=r['outer_before'],after=r['outer_after']) for f,r in reports.items()},recall_measured=False)),flush=True)

def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text()); began=time.monotonic()
    try:
        launch=dict(**index_authority(),config=cfg,monotonic_start=began,truth_scope='TRAINER_ONLY')
        for p in (Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',ROOT/'src/xgb_matcher/retrieval_native_v57.py',ROOT/'src/xgb_matcher/retrieval_nested_v53.py'):
            launch['implementation'][str(p)]=record(p)
        write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_v59_sparse_query','worker','--output',str(output)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker']); p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    if a.mode=='run': run(a.output)
    else: worker(a.output)
