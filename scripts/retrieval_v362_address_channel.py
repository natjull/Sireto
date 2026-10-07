"""One independent address-only initial100 experiment, not a fused pool/reranker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts import evaluate_retrieval_v358_postseal as ev
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,RetrievalQuery,normalize_text
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v362_address_channel')
EVALUATION=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v358_postseal_evaluation/run_20261006_a')


def generate(out):
    launch=io.read(out/'launch.json'); panel=io.read(verify(launch['panel']))
    backend=TantivyBackend(io.INDEX); tv=backend._tantivy; before=(io.INDEX/'tantivy/meta.json').stat()
    rows=[]; start=time.monotonic()
    for i,q in enumerate(panel['queries']):
        text=normalize_text(' '.join(q[k] for k in ('crm_number','crm_suffix','crm_street')))
        rq=RetrievalQuery(q['crm_name'],text,q['crm_number'],q['crm_postcode'],q['crm_insee'])
        if text:
            words,_=backend.index.parse_query_lenient(text,
                default_field_names=['addresses','historical_addresses','supporting_addresses'],conjunction_by_default=False)
            chars=backend._character_query('address_ngrams',text)
            parsed=backend._combined_query(rq,tv.Query.disjunction_max_query([words,chars],tie_breaker=0.),document_type='siret')
        else: parsed=tv.Query.empty_query()
        found=[]
        for score,loc in backend.searcher.search(parsed,limit=100,count=False).hits:
            doc=backend.searcher.doc(loc).to_dict()
            found.append(dict(siret=backend._first(doc,'siret'),siren=backend._first(doc,'siren'),score=float(score)))
        if len(found)>100 or len({r['siret'] for r in found})!=len(found): raise ValueError('Direct100 cap violation')
        rows.append(dict(query_id=q['query_id'],fold=q['fold'],candidates=found))
        if i%250==0:
            free,level=io.pressure()
            if free<50 or level!=1 or time.monotonic()-start>240: raise RuntimeError('STOP_EXPERIMENT_RESOURCE')
    after=(io.INDEX/'tantivy/meta.json').stat()
    if before.st_mtime_ns!=after.st_mtime_ns or len(rows)!=2000 or io.digest(__file__)!=launch['code']['sha256']:
        raise ValueError('Source/code drift or incomplete population')
    io.write(out/'outputs.json',rows)
    io.write(out/'generation.json',dict(status='SEALED_ADDRESS_ONLY_DIRECT100_ALL2000',queries=2000,
        outputs=record(out/'outputs.json'),launch=record(out/'launch.json'),seconds=time.monotonic()-start,
        truth_opened=False,fold0_opened=False,fold1_opened=False,score_after_pool=False))


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    io.write(out/'launch.json',dict(panel=record(io.PANEL),code=record(Path(__file__)),
        policy='BM25_ADDRESS_WORD_OR_CHAR_MAX_INSEE_ELSE_POSTCODE_DIRECT100',
        no_mixed_pool=True,no_reranker=True,truth_opened=False,test_scope='ALREADY_CONSUMED_2000_DEV_NOT_NEW_HOLDOUT'))
    with (out/'generation.log').open('x') as log:
        child=subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v362_address_channel','generate','--output',str(out)],
            stdout=log,stderr=subprocess.STDOUT,timeout=270)
    io.write(out/'supervision.json',dict(own_child_exit_confirmed=True,exit_code=child.returncode))
    child.check_returncode()
    generated=io.read(out/'generation.json'); rows=io.read(verify(generated['outputs']))
    if generated['queries']!=2000 or len(rows)!=2000: raise ValueError('All2000 before labels')
    permission,_=ev.permitted(EVALUATION)
    labels,_,_,_,_,_=ev.targets(ev.config(),permission)
    labels={r['query_id']:r for r in labels}
    old=io.read(verify(io.read(EVALUATION/'manifest.json')['report']))
    prior={r['query_id']:r['exact_hit'] for r in old['outcomes']}
    hits=[]; changed=[]
    for row in rows:
        gt=labels[row['query_id']]['human_ground_truth_siret']
        hit=gt in {r['siret'] for r in row['candidates']}; hits.append(hit)
        changed.append(dict(query_id=row['query_id'],fold=row['fold'],exact_hit=hit,previous_exact_hit=prior[row['query_id']]))
    result=dict(status='MEASURED_ADDRESS_ONLY_INITIAL100_DEV_NOT_FUSED_RETRIEVAL',queries=2000,
        exact_hits=sum(hits),exact_recall=sum(hits)/2000,
        address_finds_prior_misses=sum(r['exact_hit'] and not r['previous_exact_hit'] for r in changed),
        prior_finds_address_misses=sum(not r['exact_hit'] and r['previous_exact_hit'] for r in changed),
        combined_top100_recall_measured=False,source_report=record(EVALUATION/'metrics/report.json'),
        generation=record(out/'generation.json'),supervision=record(out/'supervision.json'),
        geographic_guard_is_hard=True,not_generalization_validation=True,
        serving_changed=False,fold0_opened=False,fold1_opened=False,downstream_models_changed=False)
    io.write(out/'outcomes.json',changed); io.write(out/'report.json',result); print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['generate','run']); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated experiment namespace')
    if a.action=='generate': generate(a.output)
    else: run(a.output)
