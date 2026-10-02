"""Exact federated initial top100, national lexical corpus plus BODACC paths.

The first-stage score of a SIRET is the maximum native search score of its
views. Sorted frontiers are consumed until 100 unique sites are emitted.
There is no fixed candidate pool, new pairwise feature score or reranker.
"""
import argparse
from collections import Counter
import heapq
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v40_initial_top100.json'
RESOURCE=dict(max_seconds=80,max_rss_bytes=6*1024**3,max_swapout_pages_delta=0)


def relational_query(index,query):
    clauses=[]
    for field,text in (('anchor_names',query.name),('event_address',query.address)):
        words=v38.tokens(text)
        if not words: return tantivy.Query.empty_query()
        parts=[(tantivy.Occur.Should,tantivy.Query.term_query(index.schema,field,w,'basic')) for w in words]
        clauses.append((tantivy.Occur.Must,tantivy.Query.boost_query(tantivy.Query.boolean_query(parts),1/len(words))))
    geo=[tantivy.Query.const_score_query(tantivy.Query.term_query(index.schema,f,v,'basic'),1.)
         for f,v in (('postcode',query.postcode),('insee',query.insee)) if v]
    if geo: clauses.append((tantivy.Occur.Should,tantivy.Query.disjunction_max_query(geo,tie_breaker=0.)))
    return tantivy.Query.boolean_query(clauses)


class NativeStream:
    """Only native scores/locations are prefetched; identities decoded on pop."""
    def __init__(self,searcher,query,decode,probe_first=False):
        self.searcher=searcher; self.query=query; self.decode=decode
        self.page=[]; self.offset=0; self.exhausted=False; self.calls=0; self.decoded=0
        self.previous=float('inf'); self.ids=set(); self.locations=set()
        self.probe_first=probe_first

    def peek(self,remaining):
        if not self.page and not self.exhausted:
            # The first hit bounds the source; later pages follow remaining
            # output capacity. Neither source has a candidate-depth ceiling.
            n=1 if not self.offset and self.probe_first else remaining
            self.page=list(self.searcher.search(self.query,limit=n,offset=self.offset,count=False).hits)
            self.calls+=1; self.offset+=len(self.page)
            self.exhausted=len(self.page)<n
        return self.page[0][0] if self.page else None

    def pop(self):
        score,loc=self.page.pop(0)
        if not math.isfinite(score) or score>self.previous:
            raise ValueError('Native source score order drift')
        self.previous=score; row=self.decode(self.searcher.doc(loc).to_dict())
        if row['siret'] in self.ids: raise ValueError('Duplicate native site document or unstable pagination')
        self.ids.add(row['siret']); self.decoded+=1
        return dict(row,score=float(score))


def collect(streams,cap=100):
    if not 0<cap<=100: raise ValueError('Initial output cap')
    frontier=[]; outputs=[]; seen={}; consumed=Counter()
    for key,stream in streams.items():
        score=stream.peek(cap)
        if score is not None: heapq.heappush(frontier,(-score,key))
    while frontier and len(outputs)<cap:
        negative,key=heapq.heappop(frontier); row=streams[key].pop(); consumed[key]+=1
        if row['score']!=-negative: raise ValueError('Frontier score drift')
        if row['siret'] not in seen:
            seen[row['siret']]=row['siren']; outputs.append(dict(row,winning_source=key))
        elif seen[row['siret']]!=row['siren']:
            raise ValueError('Conflicting official membership')
        if len(outputs)<cap:
            score=streams[key].peek(cap-len(outputs))
            if score is not None: heapq.heappush(frontier,(-score,key))
    if len({r['siret'] for r in outputs})!=len(outputs): raise ValueError('Duplicate output')
    # At the cap, the just-consumed source need not be advanced. Its last
    # score still bounds every unread hit; include it in the stopping proof.
    bound=max([-x[0] for x in frontier]+([outputs[-1]['score']] if len(outputs)==cap else []),default=None)
    return outputs,dict(consumed=dict(consumed),next_score_upper_bound=bound)


def search(base,graph,query):
    def lexical(doc):
        return {k:base._first(doc,k) for k in ('siren','siret')}
    def relational(doc):
        return json.loads(bytes(doc['identity'][0]))
    streams=dict(national=NativeStream(base.searcher,v38.query_for(base,query,'primary'),lexical),
                 bodacc=NativeStream(graph.searcher(),relational_query(graph,query),relational,probe_first=True))
    rows,trace=collect(streams)
    for r in rows:
        if len(r['siret'])!=14 or len(r['siren'])!=9 or not r['siret'].isdigit() or not r['siret'].startswith(r['siren']):
            raise ValueError('Invalid official identity')
    trace['sources']={k:dict(native_calls=s.calls,native_hits_fetched=s.offset,identities_decoded=s.decoded) for k,s in streams.items()}
    return rows,trace


def verify_launch(launch):
    v38.verify_launch(launch)
    verify(launch['graph_manifest'])
    for ref in launch['graph_files'].values(): verify(ref)


def worker(output,start):
    launch=json.loads((output/'launch.json').read_text()); verify_launch(launch)
    rows=[]; budget=Budget(RESOURCE)
    with budget:
        base=TantivyBackend(v38.INDEX)
        graph=tantivy.Index.open(str(Path(launch['graph_manifest']['path']).parent/'tantivy'))
        if base.searcher.num_docs!=launch['index_documents'] or graph.searcher().num_docs!=launch['graph_sites']:
            raise ValueError('Corpus document count drift')
        for raw in launch['queries'][start:start+8]:
            query=RetrievalQuery(name=raw['crm_name'],address=' '.join(raw[k] for k in ('crm_number','crm_suffix','crm_street') if raw[k]),
                number=raw['crm_number'],insee=raw['crm_insee'],postcode=raw['crm_postcode'])
            t=time.monotonic(); candidates,trace=search(base,graph,query)
            rows.append(dict(query_id=raw['query_id'],fold=raw['fold'],candidates=candidates,trace=trace,seconds=time.monotonic()-t))
    verify_launch(launch)
    write_json(output/f'batch{start:03d}.json',dict(rows=rows,resources=budget.report()))


def run(output):
    output.mkdir(parents=True,exist_ok=False); began=time.monotonic()
    try:
        cfg=json.loads(CONFIG.read_text()); generation=v38.load(v38.GEN,v38.GEN_SHA)
        queries,refs=v38.input_rows(generation)
        im=v38.load(v38.INDEX/'manifest.json',v38.INDEX_SHA)
        if im['build_scope']!='full' or im['contains_crm_labels']: raise ValueError('National corpus drift')
        gm=json.loads(verify(cfg['graph_manifest']).read_text())
        if gm['status']!='SEALED_BODACC_INITIAL_SEARCH_COMPONENT' or gm['truth_opened'] or gm['crm_opened']:
            raise ValueError('Relational corpus drift')
        for ref in gm['files'].values(): verify(ref)
        paths=[Path(__file__),CONFIG,Path(v38.__file__),ROOT/'src/xgb_matcher/hierarchical_retrieval.py',
            ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
            ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',ROOT/'docs/retrieval_v38_initial_top100_contract.md']
        launch=dict(queries=queries,query_sources=refs,generation=record(v38.GEN),
            index_manifest=record(v38.INDEX/'manifest.json'),index_meta=record(v38.INDEX/'tantivy/meta.json'),
            inventory=v38.inventory(),index_documents=im['num_documents'],index_sites=im['num_establishment_documents'],
            graph_manifest=cfg['graph_manifest'],graph_files=gm['files'],graph_sites=gm['counts']['sites'],
            implementation={str(p):record(p) for p in paths},truth_opened=False,full_national_site_corpus=True,
            collector_limit=100,score='MAX_NATIVE_VIEW_SCORE_WITHOUT_DOWNSTREAM_SCORING',
            fixed_candidate_depth=None,national_segments_fully_rehashed=False,resources=RESOURCE)
        write_json(output/'launch.json',launch); batches=[]; allrows=[]
        for start in range(0,128,8):
            with (output/f'batch{start:03d}.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-m','scripts.retrieval_v40_initial_top100','worker','--output',str(output),'--start',str(start)],
                    check=True,timeout=min(90,max(1,900-(time.monotonic()-began))),stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'RAYON_NUM_THREADS':'2'})
            p=output/f'batch{start:03d}.json'; batch=json.loads(p.read_text())
            expected={(r['query_id'],r['fold']) for r in queries[start:start+8]}
            if len(batch['rows'])!=8 or {(r['query_id'],r['fold']) for r in batch['rows']}!=expected or not batch['resources']['pass_resource_policy']:
                raise ValueError('Worker population/resources drift')
            allrows.extend(batch['rows']); batches.append(record(p))
            print(json.dumps(dict(completed=len(allrows),target=128,resources=batch['resources'])),flush=True)
            if len(allrows)==16:
                mean=sum(r['seconds'] for r in allrows)/16
                write_json(output/'cost_gate.json',dict(mean_seconds=mean,passed=mean<=cfg['mean_seconds_cost_gate'],truth_opened=False))
                if mean>cfg['mean_seconds_cost_gate']: raise ValueError('STOP_COST_PREFLIGHT')
        verify_launch(launch)
        write_json(output/'manifest.json',dict(status='SEALED_V40_INITIAL_TOP100',queries=128,launch=record(output/'launch.json'),
            batches=batches,seconds=time.monotonic()-began,truth_opened=False,full_corpus=True,full_query_population=False,ready_for_fold1=False))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),seconds=time.monotonic()-began,truth_opened=False,ready_for_fold1=False))
        raise


def evaluate(output):
    if (output/'evaluation.json').exists(): raise ValueError('Evaluation is immutable')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status']!='SEALED_V40_INITIAL_TOP100' or manifest['truth_opened']: raise ValueError('Complete sealed generation required')
    launch=json.loads(verify(manifest['launch']).read_text()); verify_launch(launch)
    generated=[r for ref in manifest['batches'] for r in json.loads(verify(ref).read_text())['rows']]
    ids={r['query_id'] for r in launch['queries']}
    if len(generated)!=128 or {r['query_id'] for r in generated}!=ids: raise ValueError('Denominator drift')
    cfg=json.loads(CONFIG.read_text()); bm=json.loads(verify(cfg['baseline_manifest']).read_text())
    if bm['status']!='SEALED_INITIAL_TOP100' or bm['truth_opened']: raise ValueError('Unsealed historical comparator')
    bl=json.loads(verify(bm['launch']).read_text())
    if bl['queries']!=launch['queries']: raise ValueError('Comparator population differs')
    baseline={r['query_id']:r for ref in bm['batches'] for r in json.loads(verify(ref).read_text())['rows']}
    ev=v38.load(v38.EVAL,v38.EVAL_SHA); ls=json.loads(verify(ev['label_side']).read_text())
    for ref in ls['files'].values(): verify(ref)
    labels={r['query_id']:r for r in pq.read_table(verify(dict(path=v38.LEDGER,sha256=v38.LEDGER_SHA)),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    catalog={r['query_id']:r for r in pq.read_table(verify(ev['files']['official_catalog_sidecar.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist() if r['query_id'] in ids}
    acceptable={}
    for r in pq.read_table(verify(ev['files']['acceptable_operational.parquet']),filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids: acceptable.setdefault(r['query_id'],set()).add(r['siret'])
    if any(set(x)!=ids for x in (labels,catalog,acceptable,baseline)): raise ValueError('Incomplete evaluation authority')
    outcomes=[]
    for r in generated:
        label=labels[r['query_id']]
        if r['fold']!=label['fold'] or r['fold'] not in (2,3,4): raise ValueError('Fold drift')
        row=dict(query_id=r['query_id'],**{c:label[c] for c in v38.SEGMENTS})
        for mode,candidates in (('primary',r['candidates']),('comparator',baseline[r['query_id']]['results']['primary'])):
            found={c['siret'] for c in candidates}
            if len(found)!=len(candidates) or len(found)>100: raise ValueError('Output cap/uniqueness')
            hit=label['human_ground_truth_siret'] in found
            row[mode+'_exact']=hit; row[mode+'_operational']=bool(found & acceptable[r['query_id']])
            row[mode+'_loss']='HIT' if hit else 'NOT_RETRIEVED_AT100' if catalog[r['query_id']]['official_catalog_present'] else 'OFFICIAL_CATALOG_ABSENT'
        outcomes.append(row)
    segments={}
    for column in v38.SEGMENTS:
        groups={}
        for r in outcomes:
            value=r[column]; key='UNKNOWN' if value is None or value=='' else str(value)
            groups.setdefault(key,[]).append(r)
        segments[column]={k:{m:v38.summarize(g,m) for m in v38.MODES} for k,g in groups.items()}
    result=dict(status='MEASURED_V40_INITIAL_TOP100_PILOT',decision='PIVOT_GENERATION',primary=v38.summarize(outcomes,'primary'),
        comparator=v38.summarize(outcomes,'comparator'),comparator_scope='SEALED_V38_PRIMARY_ON_IDENTICAL_QUERIES',
        gains=sum(r['primary_exact'] and not r['comparator_exact'] for r in outcomes),
        losses=sum(r['comparator_exact'] and not r['primary_exact'] for r in outcomes),segments=segments,outcomes=outcomes,
        loss_attribution={m:dict(Counter(r[m+'_loss'] for r in outcomes)) for m in v38.MODES},
        latency=dict(zip(('p50','p95','p99'),np.quantile([r['seconds'] for r in generated],[.5,.95,.99]).tolist())),
        latency_scope='Initial federated search and reference extraction, uncontrolled cache, not production SLO',
        source_winners=dict(Counter(c['winning_source'] for r in generated for c in r['candidates'])),
        generation=record(output/'manifest.json'),baseline=cfg['baseline_manifest'],ledger=record(v38.LEDGER),label_side=ev['label_side'],
        excluded_missing_human_labels=ev['excluded_missing_human_labels'],fine_loss_attribution_measured=False,
        fold0_opened=False,fold1_opened=False,ready_for_fold1=False,model_used=False,full_query_population=False,
        sirene_succession_paths_included=False,score_calibration_learned=False,
        implementation=record(Path(__file__)))
    if result['primary']['exact_recall']>=.992:
        result['decision']='STOP'; result['decision_reason']='PILOT_ONLY_FULL_OOF_CONFIDENCE_SLO_AND_FINE_ATTRIBUTION_UNMEASURED'
    write_json(output/'evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('primary','comparator','gains','losses','latency','source_winners','decision')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker','evaluate'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--start',type=int); args=p.parse_args()
    if args.mode=='run': run(args.output)
    elif args.mode=='evaluate': evaluate(args.output)
    else:
        if args.start not in range(0,128,8): p.error('Worker start')
        worker(args.output,args.start)
