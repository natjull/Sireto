"""Independent typed corporate/context native branch, ONE native100.

All whole-source corporate incidences, strict mother clauses and dated group
members used; no CRM/GT aliases or target selection. Relation != identity.
V81 proofs/coefficients/encoder frozen. Additional query components precede
the only collector; no capped pool, no candidate reranking, no new training.
"""
import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v81_indirect_lexical_query as source
from scripts import embeddinggemma2_v72_corporate_incidence as graph
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v88/typed_context_native_20261013_a'
CACHE=source.CACHE
LEXICAL_FIELDS=source.LEXICAL_FIELDS
PROOF_FIELDS=source.PROOF_FIELDS
FEATURE_FIELDS=source.FEATURE_FIELDS
RECIPE=dict(source.RECIPE,hypothesis='TYPED_WHOLE_SOURCE_CONTEXT_NATIVE_BRANCH_FROZEN_ACTOR_WEIGHT',
    source_match='ALL_MEANINGFUL_PUBLISHED_NAME_TOKENS_PRESENT_IN_QUERY_MIN_TWO_OR_EXACT_SINGLE',
    source_match_not_entity_resolution=True,related_owner_names_not_site_aliases=True,
    corporate_scope='ONE_ROLE_EDGE_OR_DISTINCT_PEER_SHARING_ONE_ACTOR_NOT_OWNERSHIP',
    role_confidence='HALF_DIV_SQRT_ACTOR_SUBJECT_DEGREE_NO_PRUNING',
    dated_group_confidence=.5,reported_mother_confidence=.5,
    duplicate_context='MAX_NOT_ADDITIVE_VOTES',
    extra_admin_support_required=True,original_national_branch_unrestricted=True,
    context_weight='EXACT_V71_CORPORATE_ACTOR_WEIGHT_TIMES_SOURCE_CONFIDENCE',
    no_coefficient_refitting=True,no_production_promotion=True)
ACTIVE=OUTPUT
CONTEXT=None
guard=source.guard
feature_path=source.feature_path
lexical_parts=source.lexical_parts


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;source.bind_runtime(out)


def initialize(out):
    source.initialize(out)
    launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    for stage in ('prepare','generate'):
        ex=io.read(source.OUTPUT/(stage+'_exit.json'))
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Complete owned V81')
    gx=io.read(graph.OUTPUT/'owned_exit.json')
    if gx['exit_code'] or not gx['own_child_exit_confirmed']:raise ValueError('Complete whole graph')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_v81=record(source.OUTPUT/'features_generation.json'),
        v81_prepare_exit=record(source.OUTPUT/'prepare_exit.json'),graph_generation=record(graph.OUTPUT/'generation.json'),
        graph_exit=record(graph.OUTPUT/'owned_exit.json'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(graph.__file__))])
    io.write(out/'launch.json',launch)


def authority(out):
    launch=source.authority(out)
    for n in ('prepared_v81','v81_prepare_exit','graph_generation','graph_exit'):verify(launch[n])
    return launch


def prepare(out):
    import duckdb
    import pyarrow.parquet as pq
    launch=authority(out);fg=io.read(verify(launch['prepared_v81']));gg=io.read(verify(launch['graph_generation']))
    artifacts=gg['artifacts'];names=io.read(verify(artifacts['names.json']))
    panel=io.read(verify(launch['panel']));stop=source.pipeline.corporate.audit_utils.STOP
    normalize=source.original.structural.normalize
    def meaningful(name):return frozenset(normalize(name or '').split())-stop
    qtokens={q['query_id']:meaningful(q['crm_name']) for q in panel};inverted=defaultdict(set)
    for qid,tokens in qtokens.items():
        for token in tokens:inverted[token].add(qid)
    def matches(name):
        tokens=meaningful(name)
        if not tokens:return set()
        possible=set.intersection(*(inverted.get(t,set()) for t in tokens))
        return {q for q in possible if len(tokens)>=2 or tokens==qtokens[q]}
    matched=defaultdict(set)
    for parent,views in names.items():
        for name in views:
            for qid in matches(name):matched[qid].add(int(parent))
    with np.load(verify(artifacts['incidence.npz']),allow_pickle=False) as f:
        ids=f['ids'];subjects=f['subject'];actors=f['actor']
    degree=np.bincount(actors,minlength=len(ids))
    # Full graph joins; neither a topk owner nor a capped leaf pool.
    context={q['query_id']:{} for q in panel}
    def add(qid,parent,confidence,kind):
        parent=str(int(parent)).zfill(9)
        if parent not in names or confidence<=0:return
        item=context[qid].setdefault(parent,dict(siren=parent,name_views=names[parent],confidence=0.,kinds=[]))
        item['confidence']=max(item['confidence'],float(confidence))
        if kind not in item['kinds']:item['kinds'].append(kind)
    for qid,parents in matched.items():
        guard(launch['deadline_unix']);at=np.searchsorted(ids,np.array(sorted(parents),dtype=ids.dtype))
        outgoing=np.isin(subjects,at);actor_nodes=np.unique(actors[outgoing])
        for a in actor_nodes:
            d=max(1,int(degree[a]));confidence=.5/np.sqrt(d)
            add(qid,ids[a],confidence,'PUBLISHED_CORPORATE_ROLE_ACTOR')
            for p in subjects[actors==a]:
                if int(ids[p]) not in parents:add(qid,ids[p],confidence,'DISTINCT_COMMON_ACTOR_PEER')
        incoming=np.isin(actors,at)
        for p,a in zip(subjects[incoming],actors[incoming]):
            add(qid,ids[p],.5/np.sqrt(max(1,int(degree[a]))),'PUBLISHED_SUBJECT_WITH_NAMED_CORPORATE_ACTOR')
    # Dated global consolidation context, not current ownership.
    group=io.read(verify(launch['group_generation']));members=io.read(verify(group['members']))
    group_rows=members['members']+[dict(member_siren=members['root']['siren'],member_name=members['root']['name'])]
    triggers=set().union(*(matches(r['member_name']) for r in group_rows))
    for qid in triggers:
        for r in group_rows:add(qid,r['member_siren'],.5,'PUBLISHED_2021_GLOBAL_CONSOLIDATION_CONTEXT')
    # Published name of a reported mother is textual context attached only to
    # the structured announced subject. Never resolve the mother by homonym.
    mother=io.read(verify(launch['mother_generation']));facts=pq.read_table(verify(mother['candidate_facts']),use_threads=False).to_pylist()
    absent={str(r['subject_siren']).zfill(9) for r in facts}-set(names)
    if absent:
        c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
        columns=['denominationUniteLegale','denominationUsuelle1UniteLegale','denominationUsuelle2UniteLegale','denominationUsuelle3UniteLegale','sigleUniteLegale']
        rows=c.execute('SELECT siren,'+','.join(columns)+" FROM read_parquet(?) WHERE siren IN (SELECT unnest(?)) AND statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000'",[str(source.pipeline.corporate.audit_utils.STOCK),sorted(absent)]).fetchall();c.close()
        for row in rows:names[str(row[0])]=sorted({normalize(v) for v in row[1:] if v and normalize(v)})
    for r in facts:
        for qid in matches(r['context_parent_name']):add(qid,r['subject_siren'],.5,'REPORTED_COMMON_MOTHER_PROJECT_CONTEXT')
    # Self-owner expansions carry no novel relational evidence.
    for qid,parents in matched.items():
        for parent in parents:context[qid].pop(str(parent).zfill(9),None)
    result={q:sorted(rows.values(),key=lambda r:r['siren']) for q,rows in context.items()}
    io.write(out/'context_expansion.json',result);engine.seal(out/'context_expansion.json')
    io.write(out/'features_generation.json',dict(fg,immutable_v81=launch['prepared_v81'],
        context_expansion=record(out/'context_expansion.json'),labels_used=False))
    io.write(out/'source_receipt.json',dict(queries=526,queries_with_expansion=sum(bool(v) for v in result.values()),
        expansion_owners=sum(map(len,result.values())),largest_uncapped_owner_expansion=max(map(len,result.values())),
        graph_nodes=len(ids),graph_incidences=len(subjects),group_members=len(group_rows),mother_facts=len(facts),
        no_ids_as_encoder_or_score_features=True,no_own_alias_created=True,labels_read=False))
    print(io.read(out/'source_receipt.json'),flush=True)
    del context,names,ids,subjects,actors;gc.collect()


class BeforeOnlyCollector:
    def __init__(self,backend,q,w,entries):
        self.searcher=backend.searcher;self.backend=backend;self.q=q;self.w=w;self.entries=entries
    def doc(self,loc):return self.searcher.doc(loc)
    def search(self,query,**kwargs):
        if kwargs.get('limit')!=100:raise ValueError('Exactly one native100 collector')
        tv=self.backend._tantivy;admin=source.original.lexical_parts(self.backend,self.q)[4];parts=[query]
        if admin is not None:
            admin=tv.Query.const_score_query(admin,0.)
            if self.entries and self.w[6]>0:
                maximum=(1<<RECIPE['quantization_bits'])-1
                confidence=np.array([e['confidence'] for e in self.entries],dtype=float)
                if not np.isfinite(confidence).all() or (confidence<0).any() or (confidence>1).any():raise ValueError('Bounded source context')
                integers=np.floor(confidence*maximum).astype(np.uint32);planes=[]
                for bit in range(RECIPE['quantization_bits']):
                    selected=np.flatnonzero(integers & (1<<bit))
                    if len(selected):
                        owner=tv.Query.term_set_query(self.backend.index.schema,'siren',[self.entries[i]['siren'] for i in selected])
                        planes.append(tv.Query.const_score_query(owner,float(self.w[6]*(1<<bit)/maximum)))
                if planes:
                    parts.append(tv.Query.boolean_query([(tv.Occur.Must,admin),
                        (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in planes]))]))
        combined=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts])
        # Added branches still require leaf documents, because parent docs also
        # carry siren/name. No materialization before this single search call.
        leaf=tv.Query.const_score_query(tv.Query.term_query(self.backend.index.schema,'document_type','siret','basic'),0.)
        combined=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,combined)])
        return self.searcher.search(combined,**kwargs)


class BackendProxy:
    def __init__(self,backend,searcher):self.backend=backend;self.searcher=searcher
    def __getattr__(self,name):return getattr(self.backend,name)


def native_search(backend,q,keys,features,coefficients):
    global CONTEXT
    if CONTEXT is None:CONTEXT=io.read(verify(io.read(ACTIVE/'features_generation.json')['context_expansion']))
    entries=CONTEXT[q['query_id']];proxy=BackendProxy(backend,BeforeOnlyCollector(backend,q,coefficients,entries))
    rows,cost=source.native_search(proxy,q,keys,features,coefficients)
    cost.update(typed_context_owner_expansions=len(entries),one_collector_before_materialization=True,
        added_relations_not_identity=True,source_scores_max_not_duplicate_votes=True,
        context_quantization_abs_error_bound=float(coefficients[6])/((1<<RECIPE['quantization_bits'])-1))
    return rows,cost


def stage(action,out):
    bind_runtime(out);source.pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    elif action=='bind_models':source.bind_models(out)
    else:getattr(source.pipeline.learned,action)(out)
    if action=='evaluate':
        r=io.read(out/'report.json');r.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            typed_context_not_entity_identity=True,frozen_v81_coefficients=True,no_extra_fit=True)
        io.write(out/'report.json',r)


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=source.pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('bind_models','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v88_typed_context_native',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=deadline-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=deadline-30:raise TimeoutError('EIGHT_HOUR_END')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} exit {rc}; no partial recall')
            break
    stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','bind_models','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise


