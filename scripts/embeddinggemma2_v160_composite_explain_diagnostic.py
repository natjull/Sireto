"""Two explicitly post-evaluation selected public cases; not recall/validation.

Exact bound V142 score code, execution-only subset/recompute/capture changes.
Replay and owned0 seal precede labels and documentary EXPLAIN. No new scorer.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for _env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_env]='1'
import numpy as np
import psutil
from scripts import embeddinggemma2_v142_bano_native_ablation as original
from scripts import embeddinggemma2_night as io

HOME=Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
OUT=HOME/'v160/composite_explain_diagnostic_20261014_night'
QIDS=('NEWCRM:1c6805a5f39956c1909201e3e901b453','NEWCRM:d6171348a29c3e0781b2dd602f54d06b')
EXPECTED='568fadfc66f35ff3856290ee8a3a77466993d9ebba0b601d08975958f9cba9d7'
ATOL=1e-5
PREPARATION_STARTED=1791948960  # Root authorization03:36UTC;20min includes preparation.
RECIPE=dict(scope='TWO_PUBLIC_V157_GEO_PLAUSIBLE_CASES_EXPLICITLY_CHOSEN_POSTEVAL_DIAGNOSTIC_NOT_RECALL',
    qids=QIDS,original='EXACT_V142_ALL_NEW_SOURCES_MAX_BANO',coefficient='SAME58_FROZEN_OOF_DIAGNOSTIC_NOT_SERVING',
    execution_only='TWO_QUERY_SUBSET_FORCE_RECOMPUTE_NO_REUSE_CAPTURE_FINAL_QUERY',
    replay='IDENTICAL_ORDERED_SIRET_OUTPUTS_AND_SCORES_ABSOLUTE_TOL1e-5_RTO0_ELSE_STOP',
    query_fidelity='FINAL_QUERY_DEBUG_REPR_SHA_AND_ALL_ORIGINAL_SCORE_BINDINGS_EQUAL_STAGEA_STAGEB',
    label_boundary='AFTER_REPLAY_WHOLE_TWO_OWNED0_MANIFEST',
    target_lookup='PUBLIC_TARGET_IN_ORIGINAL_NUMERIC_UNIVERSE_ZERO_SCORE_ID_LOOKUP_SEPARATE_FROM_FULL_QUERY',
    explain='EXISTING_Query.explain_REAL_SEARCHER_TARGET_FIRST_CUT100_SCORE_FIDELITY_SAME_ATOL',
    contribution='ORIGINAL51_PREQUANT_GROUPS_AND_QUANTIZED_BLOCK_ARITHMETIC_NOT_NEW_SCORE_ENGINE',
    no_retrieval_recipe_fit_label_coordinate_policy_change=True,no_full_rank_claim=True,
    no_natural_person_names_contacts_or_explanation_term_strings_exported=True,
    threads=1,rss_bytes=6*1024**3,free_percent=25,pressure_levels=[0,1,2],disk_reserve_gib=40,
    max_execution_seconds=1200,preparation_started_unix=PREPARATION_STARTED,deadline_unix=1791954368,closed_sets_opened=False)

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def record(p):return dict(path=str(p),sha256=sha(p),size_bytes=Path(p).stat().st_size)
def verify(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('Hash drift '+str(p))
    return p
def guard():
    rss=psutil.Process().memory_info().rss;free,pressure=io.pressure()
    start=float(os.environ.get('V160_DIAGNOSTIC_STARTED',PREPARATION_STARTED))
    if time.time()>=1791954368 or time.time()-start>1200 or rss>RECIPE['rss_bytes'] or free<25 or pressure not in (0,1,2) or psutil.disk_usage(HOME).free<40*1024**3:
        raise RuntimeError(f'STOP preserve diagnostic rss={rss} free={free} pressure={pressure}')
    return rss

class Captured(Exception):
    def __init__(self,query,searcher,query_sha,schema):self.query=query;self.searcher=searcher;self.query_sha=query_sha;self.schema=schema

def frozen_function(mode,selected,captures):
    if sha(Path(original.__file__))!=EXPECTED:raise ValueError('Exact original V142 code')
    text=inspect.getsource(original.generate)
    changes={'    for q in panel:\n':'    for q in panel:\n        if q["query_id"] not in DIAGNOSTIC_SELECTED: continue\n',
        '            if equivalent_upstream:\n':'            if False:  # Execution-only: replay same inputs instead of reuse\n',
        '            elif not np.any(score) and not entries:\n':'            elif False:  # Execution-only: recompute same zero-source query\n'}
    for before,after in changes.items():
        if text.count(before)!=1:raise ValueError('Execution-only rewrite bound count')
        text=text.replace(before,after)
    anchor='            if False:  # Execution-only: replay same inputs instead of reuse\n'
    text=text.replace(anchor,'            DIAGNOSTIC_TRACE[qid]=dict(keys=keys,score=score,entries=entries,model=model)\n'+anchor)
    end='    if len(rows) != 526 or any(len({p[\'siret\'] for p in cs}) != len(cs) for cs in rows.values()):'
    if text.count(end)!=1:raise ValueError('Exact whole-run finalization boundary')
    text=text.split(end)[0]+'    return rows, costs, DIAGNOSTIC_TRACE\n'
    text=text.replace("status='ALL526_NATIVE_BANO_NO_LABELS'","status='TWO_QUERY_REPLAY_NOT_EVALUATION'").replace('total=526,','total=2,')
    real_backend=original.TantivyBackend
    def backend_factory(path):
        backend=real_backend(path);real=backend.searcher
        class CaptureSearcher:
            def doc(self,address):return real.doc(address)
            def search(self,query,**kwargs):
                guard()
                if kwargs.get('limit')!=100 or len(captures)>=len(selected):raise ValueError('Only same original collector100')
                representation=repr(query);query_sha=hashlib.sha256(representation.encode()).hexdigest()
                captures.append(dict(query_sha256=query_sha,repr_bytes=len(representation.encode())))
                del representation;gc.collect();guard()
                if mode=='capture':raise Captured(query,real,query_sha,backend.index.schema)
                return real.search(query,**kwargs)
        backend.searcher=CaptureSearcher();return backend
    ns=dict(vars(original));ns.update(TantivyBackend=backend_factory,guard=guard,
        DIAGNOSTIC_SELECTED=set(selected),DIAGNOSTIC_TRACE={})
    exec(compile(text,str(Path(__file__))+':execution_only_original_generate','exec'),ns)
    return ns['generate'],ns

def inputs(directory):
    directory.mkdir(parents=True)
    (directory/'calibration').mkdir();(directory/'all_new_sources_max_bano/native').mkdir(parents=True)
    for name in ('launch.json','features_generation.json','prepared.json','calibration/generation.json'):
        write(directory/name,read(original.ROOT/name))

def replay():
    guard()
    if OUT.exists():raise ValueError('Fresh namespace')
    OUT.mkdir(parents=True);write(OUT/'recipe.json',RECIPE)
    refs=dict(implementation=record(Path(__file__)),original_implementation=record(Path(original.__file__)),
        original_launch=record(original.ROOT/'launch.json'),original_generation=record(original.ROOT/'all_new_sources_max_bano/generation.json'))
    write(OUT/'launch.json',dict(inputs=refs,recipe=record(OUT/'recipe.json'),selected_posteval=True,labels_read=False))
    directory=OUT/'replay_inputs';inputs(directory);captures=[]
    fn,ns=frozen_function('replay',QIDS,captures);rows,costs,trace=fn(directory,'all_new_sources_max_bano')
    if set(rows)!=set(QIDS) or len(captures)!=2:raise ValueError('Exactly two real collectors')
    panel=read(read(original.ROOT/'launch.json')['panel']['path']);order=[q['query_id'] for q in panel if q['query_id'] in QIDS]
    result=[]
    for qid,capture in zip(order,captures):
        stem=hashlib.sha256(qid.encode()).hexdigest();expected=original.ROOT/'all_new_sources_max_bano/native'/(stem+'.json')
        old=read(expected)['candidates'];fresh=rows[qid]
        if [r['siret'] for r in old]!=[r['siret'] for r in fresh]:raise ValueError('STOP replay output IDs/order differs')
        error=max(abs(a['score']-b['score']) for a,b in zip(old,fresh))
        if error>ATOL:raise ValueError('STOP replay score mismatch, no tolerance relaxation')
        result.append(dict(query_id=qid,expected_native=record(expected),actual_native=record(directory/'all_new_sources_max_bano/native'/(stem+'.json')),
            final_query=capture,model=record(Path(read(expected)['model']['path'])),max_score_abs_error=error,
            raw_leaf_scores_sha256=hashlib.sha256(trace[qid]['score'].tobytes()).hexdigest(),ordered_keys_sha256=hashlib.sha256(trace[qid]['keys'].tobytes()).hexdigest()))
    write(OUT/'replay_report.json',dict(status='COMPLETE_EXACT_TWO_QUERY_REPLAY_BEFORE_LABELS',queries=result,
        labels_read=False,collector_scope=2,recipe=record(OUT/'recipe.json'),launch=record(OUT/'launch.json'),rss_bytes=guard()))
    print('REPLAY_EXACT',json.dumps(result),flush=True)

def redacted_tree(node):
    desc=str(node.get('description','')).lower()
    kind='BM25' if 'bm25' in desc else 'Boost' if 'boost' in desc else 'Constant' if 'const' in desc else 'Sum' if 'sum' in desc else 'Other'
    return dict(kind=kind,value=node.get('value'),details=[redacted_tree(c) for c in node.get('details',[])])

def arithmetic(features,weights,leaf_scores,at,confidence,admin_matches):
    maximum=(1<<20)-1
    def quant(value,bound):return bound*int(np.floor(np.clip(value/bound,0,1)*maximum))/maximum if bound>0 else 0.
    x=features[at].astype(float);w=np.asarray(weights[7:],dtype=float)
    groups={'own_name_address_and_site_semantic':[0,2,3,4,9], 'own_geo_and_interactions':[5,6,7,42,43,44,45,46],
            'parent_context_and_own_address_interaction':[1,8,16,17,18,19]}
    used={i for ids in groups.values() for i in ids};groups['other_typed_context_and_residual']=[i for i in range(51) if i not in used]
    raw={k:float(x[ids]@w[ids]) for k,ids in groups.items()}
    numeric=quant(float(x[:22]@w[:22]),float(w[:22].sum()))+quant(float(x[22:]@w[22:]),float(w[22:].sum()))
    leaf=quant(float(leaf_scores[at]),float(leaf_scores.max(initial=0)))
    owner=float(weights[6])*int(np.floor(confidence*maximum))/maximum if admin_matches else 0.
    return dict(prequant_numeric_groups=raw,prequant_numeric_total=float(x@w),quantized_numeric_block_arithmetic=numeric,
        added_leaf_raw=float(leaf_scores[at]),added_leaf_quantized_arithmetic=leaf,added_owner_quantized_arithmetic=owner,
        caveat='Group sums are exact-real prequant arithmetic; native explanation is authoritative. Bitplane/native floating-point rounding can differ slightly.')

def explain():
    guard();manifest=read(OUT/'replay_manifest.json');ex=read(verify(manifest['owned_exit']))
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Replay actual owned0 before labels')
    report=read(verify(manifest['report']));verify(report['recipe']);verify(report['launch'])
    for r in report['queries']:verify(r['expected_native']);verify(r['actual_native']);verify(r['model'])
    nr=read(HOME/'v101/postseal_name_roles_20261013_a/report.json');sr=read(HOME/'v103/raw_alias_coverage_20261013_a/report.json')
    labels=read(verify(nr['cases']));statuses={r['qid']:r for r in read(verify(sr['statuses']))}
    launch=read(original.ROOT/'launch.json');fg=read(original.ROOT/'features_generation.json')
    keys_all=np.load(verify(launch['keys']),mmap_mode='r');public=np.load(verify(launch['public_mask']),mmap_mode='r')
    cases=[];directory=OUT/'explanations';directory.mkdir()
    for original_row in report['queries']:
        qid=original_row['query_id'];s=statuses[qid]
        if s['site_diffusion']!='O' or s['owner_diffusion']!='O':raise ValueError('Protected diagnostic target STOP')
        target=next(r['label_siret'] for r in labels if r['query_id']==qid)
        inp=OUT/('capture_'+hashlib.sha256(qid.encode()).hexdigest());inputs(inp);captures=[]
        fn,ns=frozen_function('capture',(qid,),captures)
        try:fn(inp,'all_new_sources_max_bano')
        except Captured as c:captured=c
        else:raise ValueError('Final Query not captured without collector')
        if captured.query_sha!=original_row['final_query']['query_sha256']:raise ValueError('STOP final query SHA differs')
        trace=ns['DIAGNOSTIC_TRACE'][qid];keys=trace['keys'];score=trace['score'];model=trace['model']
        if hashlib.sha256(score.tobytes()).hexdigest()!=original_row['raw_leaf_scores_sha256'] or hashlib.sha256(keys.tobytes()).hexdigest()!=original_row['ordered_keys_sha256']:
            raise ValueError('STOP scoring bindings differ')
        with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:features=f['features']
        expected=read(original_row['expected_native']['path']);query_input=expected['query'];old=expected['candidates'];cut=old[-1]['score']
        case=dict(query_id=qid,query_sha256=captured.query_sha,cut100_score=cut,documents=[],target_full_rank='NOT_MEASURED')
        for role,key,expected_score in [('first',old[0]['siret'],old[0]['score']),('cut100',old[-1]['siret'],cut),('target',target,None)]:
            k=np.uint64(int(key));canonical=int(np.searchsorted(keys_all,k))
            if canonical>=len(keys_all) or keys_all[canonical]!=k or not public[canonical]:raise ValueError('Protected/absent document STOP before DocAddress lookup')
            at=int(np.searchsorted(keys,k))
            if role=='target' and (at>=len(keys) or keys[at]!=k):raise ValueError('Target absent numeric universe STOP no injection')
            import tantivy as tv
            # A separate zero-score technical locator; never part of captured scoring Query.
            lookup=tv.Query.boolean_query([(tv.Occur.Must,tv.Query.const_score_query(tv.Query.term_query(captured.schema,'siret',key,'basic'),0.)),
                (tv.Occur.Must,tv.Query.const_score_query(tv.Query.term_query(captured.schema,'document_type','siret','basic'),0.))])
            hits=captured.searcher.search(lookup,limit=2,count=False).hits
            if len(hits)!=1:raise ValueError('Unique existing document required')
            address=hits[0][1];tree=json.loads(captured.query.explain(captured.searcher,address).to_json());value=float(tree['value'])
            if expected_score is not None and abs(value-expected_score)>ATOL:raise ValueError('STOP explain head/cut score fidelity')
            doc=captured.searcher.doc(address).to_dict()
            def field(name):
                v=doc.get(name,'');return v[0] if isinstance(v,list) and v else v
            admin=bool((query_input['crm_insee'] and field('insee')==query_input['crm_insee']) or (query_input['crm_postcode'] and field('postcode')==query_input['crm_postcode']))
            confidence=max((r['confidence'] for r in trace['entries'] if r['siren_ref']==str(int(k)//100000).zfill(9)),default=0.)
            component=arithmetic(features,model['coefficients'],score,at,confidence,admin) if at<len(keys) and keys[at]==k else dict(numeric_absent=True)
            if 'quantized_numeric_block_arithmetic' in component:component['dynamic_lexical_and_inherited_context_residual_approx']=value-component['quantized_numeric_block_arithmetic']-component['added_leaf_quantized_arithmetic']-component['added_owner_quantized_arithmetic']
            case['documents'].append(dict(role=role,public_both=True,score=value,expected_returned_score=expected_score,
                margin_over_original_cut100=value-cut,component_arithmetic=component,explanation_sha256=hashlib.sha256(json.dumps(tree,sort_keys=True).encode()).hexdigest(),redacted_tree=redacted_tree(tree)))
            del doc,tree;guard()
        cases.append(case);del features,trace,ns,captured;gc.collect()
    write(OUT/'explain_report.json',dict(status='COMPLETE_TWO_SELECTED_POSTEVAL_PUBLIC_DOCUMENT_EXPLANATIONS_NOT_RECALL',
        cases=cases,replay_manifest=record(OUT/'replay_manifest.json'),no_natural_name_or_contact_export=True,
        no_source_feature_label_or_recipe_change=True,no_full_rank_generalization_gain_SLO_or_label_error_claim=True,rss_bytes=guard()))
    print('EXPLAIN_COMPLETE',json.dumps([{k:v for k,v in c.items() if k!='documents'} for c in cases]),flush=True)

def run(stage):
    if stage=='replay':os.environ['V160_DIAGNOSTIC_STARTED']=str(PREPARATION_STARTED)
    else:os.environ['V160_DIAGNOSTIC_STARTED']=str(read(OUT/'replay_owned_child_exit.json')['started_unix'])
    started=float(os.environ['V160_DIAGNOSTIC_STARTED']);guard()
    child=subprocess.Popen([sys.executable,'-u','-m','scripts.embeddinggemma2_v160_composite_explain_diagnostic',stage])
    try:
        while child.poll() is None:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guard()
                if psutil.Process(child.pid).memory_info().rss>RECIPE['rss_bytes']:raise RuntimeError('Child RSS cap')
    except BaseException:
        child.terminate();child.wait(timeout=10);raise
    if not OUT.exists():raise RuntimeError('No diagnostic namespace')
    exitpath=OUT/(stage+'_owned_child_exit.json');write(exitpath,dict(exit_code=child.returncode,own_child_exit_confirmed=True,pid=child.pid,started_unix=started,finished_unix=time.time()))
    if child.returncode:raise RuntimeError('STOP diagnostic owned nonzero, preserve artifacts')
    write(OUT/(stage+'_manifest.json'),dict(status='COMPLETE_SEALED_TWO_CASE_'+stage.upper(),report=record(OUT/(stage+'_report.json')),owned_exit=record(exitpath)))
    print(json.dumps(record(OUT/(stage+'_manifest.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('run_replay','run_explain','replay','explain'));a=p.parse_args()
    if a.action.startswith('run_'):run(a.action.removeprefix('run_'))
    else:globals()[a.action]()
