"""Nested official-initialism and learned-sparse native head probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from scripts import retrieval_v38_initial_top100 as v38
from scripts import retrieval_v42_indexed_metric as v42
from scripts.retrieval_v56_exact_fields import index_authority
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
from src.xgb_matcher.retrieval_initialisms_v67 import EXACT_FIELDS,patterns,membership,acronym_components,select
from src.xgb_matcher.retrieval_sparse_query_v59 import unit
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json
CONFIG=ROOT/'config/retrieval_v67_stored_initialisms.json'


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; v38.verify_launch(launch)
    source=json.loads(verify(cfg['vector_source']).read_text())
    sl=json.loads(verify(source['launch']).read_text()); v38.verify_launch(sl)
    data=json.loads(verify(sl['inputs']).read_text()); cache=json.loads(verify(source['worker']).read_text())
    proof=json.loads(verify(cfg['proof_source']).read_text()); v38.verify_launch(proof)
    rows=data['rows']; scenes=[s for ref in proof['features'] for s in json.loads(verify(ref).read_text())['scenes']]
    if source['recall_measured'] or proof['recall_measured'] or len(rows)!=512 or len(scenes)!=512:
        raise ValueError('Training-only source drift')
    if cache['query_ids']!=[r['query_id'] for r in rows] or any(
        any(r[k]!=s[k] for k in ('query_id','fold','positive_siren')) or len(r['negatives'])!=len(s['negatives']) for r,s in zip(rows,scenes)):
        raise ValueError('Teacher alignment drift')
    q=unit(np.load(verify(cache['query_vectors']),allow_pickle=False)[:,:384])
    budget=Budget(cfg['resources']); reports={}; models={}; refs={}; vocabs={}; documents={}; aliases={}
    try:
        with budget:
            backend=TantivyBackend(v38.INDEX); tv=backend._tantivy; ids=cache['document_ids']
            if backend.searcher.num_docs!=launch['index_documents']: raise ValueError('Official national count drift')
            for start in range(0,len(ids),2048):
                subset=ids[start:start+2048]
                for _,loc in backend.searcher.search(tv.Query.term_set_query(backend.index.schema,'siret',subset),limit=len(subset)+1,count=False).hits:
                    d=backend.searcher.doc(loc).to_dict(); s=backend._first(d,'siret'); p=backend._first(d,'siren')
                    if s not in subset or s in documents or not s.startswith(p): raise ValueError('Official identity drift')
                    documents[s]={f:d.get(f,[]) for f in v38.NAMES}
                    aliases[s]=[v for f in ('legal_names','trade_names','site_names') for v in d.get(f,[])]
            if set(documents)!=set(ids): raise ValueError('Missing source doc; do not drop training row')
            from src.xgb_matcher.hierarchical_retrieval import normalize_text
            from src.xgb_matcher.retrieval_proof_v56 import signal
            current_exact_audited=0
            for r,scene in zip(rows,scenes):
                name=signal(r['crm_name'])
                for sid,native in [(r['positive_siret'],scene['positive_features']),
                    *[(n['siret'],p['features']) for n,p in zip(r['negatives'],scene['negatives'])]]:
                    exact=float(bool(name and name in {normalize_text(a) for a in aliases[sid]}))
                    if exact!=native[2]: raise ValueError('Stored current aliases vs cached native exact proof mismatch')
                    current_exact_audited+=1
            positive_acronyms=sum(bool(membership(r['crm_name'],aliases[r['positive_siret']])) for r in rows)
            negative_acronyms=sum(bool(membership(r['crm_name'],aliases[n['siret']])) for r in rows for n in r['negatives'])
            print(json.dumps(dict(stored_alias_values=sum(len(a) for a in aliases.values()),
                current_exact_vectors_audited=current_exact_audited,positive_acronyms=positive_acronyms,
                negative_acronyms=negative_acronyms)),flush=True)
            parity=[i for i,r in enumerate(rows) if patterns(r['crm_name'])][:cfg['parity_queries']]
            if len(parity)!=cfg['parity_queries']: raise ValueError('Insufficient declared parity sample')
            parity_vectors=0
            for i in parity:
                r=rows[i]; s=scenes[i]; qraw=v42.as_query(r)
                dids=[r['positive_siret'],*[n['siret'] for n in r['negatives']]]
                nat=[s['positive_features'],*[n['features'] for n in s['negatives']]]
                expected={sid:[a,a*n[4],a*n[5],a*n[7]] for sid,n in zip(dids,nat)
                    for a in [membership(r['crm_name'],aliases[sid]) ]}
                gate=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',dids),0.)
                for j,part in enumerate(acronym_components(backend,qraw)):
                    scored={} if part is None else {backend._first(backend.searcher.doc(loc).to_dict(),'siret'):float(score)
                        for score,loc in backend.searcher.search(tv.Query.boolean_query([(tv.Occur.Must,gate),(tv.Occur.Must,part)]),limit=len(dids),count=False).hits}
                    if any(abs(scored.get(sid,0.)-e[j])>1e-5 for sid,e in expected.items()):
                        raise ValueError('Official initialism offline/native parity drift')
                parity_vectors+=len(dids)
            for held in (2,3,4):
                print(json.dumps(dict(starting_fold=held)),flush=True); before=time.monotonic()
                projection,vocab,models[str(held)],reports[str(held)]=select(q,rows,scenes,documents,aliases,data['registry'],held)
                path=output/f'query_projection_fold{held}.npy'; np.save(path,projection,allow_pickle=False); refs[str(held)]=record(path)
                path=output/f'vocabulary_fold{held}.json'; write_json(path,vocab); vocabs[str(held)]=record(path)
                reports[str(held)]['fit_seconds']=time.monotonic()-before
                print(json.dumps(dict(fold=held,selected=reports[str(held)]['selected'],before=reports[str(held)]['outer_before'],after=reports[str(held)]['outer_after'])),flush=True)
        v38.verify_launch(launch)
        for ref in (cfg['vector_source'],cfg['proof_source']): verify(ref)
        write_json(output/'models.json',models)
        write_json(output/'manifest.json',dict(status='SEALED_TRAINING_ONLY_OFFICIAL_INITIALISM_NATIVE_PROBE',
            models=record(output/'models.json'),projections=refs,vocabularies=vocabs,reports=reports,
            launch=record(output/'launch.json'),resources=budget.report(),seconds=time.monotonic()-launch['monotonic_start'],
            alias_value_source='STORED_CURRENT_ROLE_FIELDS_NOT_UNSTORED_EXACT_FIELDS',
            stored_alias_values=sum(len(a) for a in aliases.values()),
            current_exact_vectors_audited=current_exact_audited,
            positive_initialisms=positive_acronyms,negative_initialisms=negative_acronyms,
            parity_queries=len(parity),parity_vectors=parity_vectors,native_current_alias_parity=True,
            training_queries=512,official_training_documents=len(documents),recall_measured=False,index_built=False,
            no_crm_alias_dictionary=True,no_identifiers_as_features=True,fold0_opened=False,fold1_opened=False,
            output_candidates=False,full_national_search_performed=False,ready_for_fold1=False,downstream_models_unchanged=True,
            limitations=['CURRENT_EXACT_ALIASES_ONLY','BAN_GEO_GRAPH_TIME_NOT_ADDED','SOURCE_CONTROLS_NOT_NATIONAL_RECALL100',
                'FOLD_LOCAL_SAFE_TRAINING_VOCABULARY_NOT_NATIONAL_ALL_WORDS']))
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),reports=reports,
            resources=budget.report() if hasattr(budget,'elapsed') else None,recall_measured=False,models_deployable=False))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic(); cfg=json.loads(CONFIG.read_text())
    try:
        launch=dict(**index_authority(),config=cfg,monotonic_start=began,truth_scope='TRAINER_ONLY')
        paths=[Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_initialisms_v67.py',
            ROOT/'src/xgb_matcher/retrieval_joint_contrastive_v63.py',ROOT/'src/xgb_matcher/retrieval_sparse_query_v59.py',
            ROOT/'src/xgb_matcher/retrieval_native_v57.py',ROOT/'src/xgb_matcher/retrieval_nested_v53.py',
            ROOT/'docs/retrieval_v67_stored_initialisms_probe.md']
        launch['implementation'].update({str(p):record(p) for p in paths}); write_json(output/'launch.json',launch)
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.preflight_retrieval_v67_stored_initialisms','worker','--output',str(output)],
                check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','RAYON_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-8000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,recall_measured=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker']); p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(); (run if a.mode=='run' else worker)(a.output)

