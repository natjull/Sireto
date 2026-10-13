"""Typed published supplier/client-brand indirect lexical retrieval, ONE100.

Weak brand hints, not resolved corporate edges/identity. Frozen V71 weights;
same V75 numerical proofs. The extra national lexical branch has query admin
support, while the original national branch remains unrestricted. Every
public corporate owner whose published current names match each source brand
is eligible; no target lookup, topk company/leaf pool or candidate reranking.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v75_published_supplier_context as supplier
from scripts import embeddinggemma2_v71_frozen_residual as pipeline
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v81/indirect_lexical_query_20261013_a'
PREVIOUS=supplier.OUTPUT
CACHE=pipeline.CACHE
LEXICAL_FIELDS=pipeline.LEXICAL_FIELDS
PROOF_FIELDS=pipeline.PROOF_FIELDS
FEATURE_FIELDS=pipeline.FEATURE_FIELDS
RECIPE=dict(supplier.RECIPE,hypothesis='NATIONAL_TYPED_INDIRECT_LEXICAL_QUERY_PLUS_FROZEN_SOURCE_SCORE',
    indirect_name_weight='EXACT_FROZEN_ORIGINAL_WORD_CHAR_NAME_WEIGHTS_TIMES_SOURCE_CONFIDENCE_05',
    indirect_scope='ALL_NATIONAL_CURRENT_PUBLIC_CORPORATE_OWNER_NAME_MATCHES_NOT_PROOF_UNIVERSE_ONLY',
    original_national_branch_unrestricted=True,extra_branch_admin_support_required=True,
    source_brand_hints_not_identity_claims=True,no_extra_parameter_fit=True,
    indirect_names_not_inserted_into_candidate_own_aliases=True,no_source_names_as_crm_gt_aliases=True)
ACTIVE=OUTPUT
INDIRECT=None
guard=pipeline.guard
feature_path=pipeline.feature_path
lexical_parts=pipeline.lexical_parts


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;pipeline.bind_runtime(out)


def initialize(out):
    supplier.initialize(out);launch=io.read(out/'launch.json')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        supplier_features=record(PREVIOUS/'features_generation.json'),supplier_exit=record(PREVIOUS/'prepare_exit.json'))
    launch['helpers'].append(record(Path(supplier.__file__)));io.write(out/'launch.json',launch)


def authority(out):
    launch=supplier.authority(out)
    verify(launch['supplier_features']);verify(launch['supplier_exit'])
    return launch


def prepare(out):
    import duckdb
    launch=authority(out);ex=io.read(verify(launch['supplier_exit']))
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('All526 source features complete')
    source=io.read(verify(launch['supplier_features']));review=io.read(verify(launch['visual_review']))
    stop=pipeline.corporate.audit_utils.STOP
    brands=[sorted(set(original.structural.normalize(r['text']).split())-stop) for r in review['brands']]
    provider=original.structural.normalize(review['provider_published_brand'])
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.execute("CREATE MACRO namekey(v) AS trim(regexp_replace(upper(strip_accents(coalesce(v,''))), '[^A-Z0-9]+',' ','g'))")
    cols=['denominationUniteLegale','denominationUsuelle1UniteLegale','denominationUsuelle2UniteLegale','denominationUsuelle3UniteLegale','sigleUniteLegale']
    expression="concat_ws(' ',"+','.join(cols)+')';clauses=[];parameters=[]
    for words in brands:
        if not words:raise ValueError('Informative published brand')
        clauses.append('('+' AND '.join('regexp_matches(namekey('+expression+'),?)' for _ in words)+')')
        parameters.extend('(^| )'+w+'( |$)' for w in words)
    sql='SELECT siren,'+','.join('('+clause+') b'+str(i) for i,clause in enumerate(clauses))+" FROM read_parquet(?) WHERE statutDiffusionUniteLegale='O' AND categorieJuridiqueUniteLegale<>'1000' AND ("+' OR '.join(clauses)+')'
    records=c.execute(sql,[*parameters,str(pipeline.corporate.audit_utils.STOCK),*parameters]).fetchall()
    c.close();owners=[[] for _ in brands]
    for row in records:
        for i,match in enumerate(row[1:]):
            if match:owners[i].append(str(row[0]))
    data=dict(provider_published_brand=provider,client_brands=[dict(tokens=words,owner_refs=sorted(set(refs))) for words,refs in zip(brands,owners)],
        source=launch['published_generation'],review=launch['visual_review'],stock=record(pipeline.corporate.audit_utils.STOCK),
        all_national_public_corporate_names_scanned=True,physical_people_not_read=True,legal_entity_edges_not_resolved=True,
        own_site_aliases_not_changed=True,refs_are_lookup_only_not_numerical_features=True,labels_read=False)
    io.write(out/'indirect_names.json',data);engine.seal(out/'indirect_names.json')
    io.write(out/'features_generation.json',dict(source,immutable_supplier_source=launch['supplier_features'],
        indirect_names=record(out/'indirect_names.json'),labels_used=False))


def native_search(backend,q,keys,features,coefficients):
    global INDIRECT
    w=np.asarray(coefficients,dtype=float);tv=backend._tantivy;start=time.monotonic()
    old_query,base_error=pipeline.base.native_query(backend,q,keys,features[:,:22],w[pipeline.BASE_AT]);parts=[old_query]
    actor=pipeline.corporate.corporate_component(backend,q)
    if actor is not None and w[6]>0:parts.append(tv.Query.boost_query(actor,float(w[6])))
    residual=w[29:];bound=float(residual.sum());maximum=(1<<RECIPE['quantization_bits'])-1
    if bound and len(keys):
        scores=features[:,22:].astype(float)@residual;integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32)
        refs=[f'{int(k):014d}' for k in keys]
        for bit in range(RECIPE['quantization_bits']):
            chosen=np.flatnonzero(integers & (1<<bit))
            if len(chosen):parts.append(tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in chosen]),float(bound*(1<<bit)/maximum)))
    if INDIRECT is None:INDIRECT=io.read(verify(io.read(ACTIVE/'features_generation.json')['indirect_names']))
    provider=set(INDIRECT['provider_published_brand'].split());query_tokens=set(original.structural.normalize(q['crm_name'] or '').split())
    expanded=False
    if provider and provider<=query_tokens:
        admin=original.lexical_parts(backend,q)[4]
        if admin is not None:
            admin=tv.Query.const_score_query(admin,0.)
            for brand in INDIRECT['client_brands']:
                if not brand['owner_refs']:continue
                eq=dict(q,crm_name=' '.join(brand['tokens']));name_parts=original.lexical_parts(backend,eq)[:2]
                weighted=[tv.Query.boost_query(p,float(w[i]*RECIPE['weak_context_confidence'])) for i,p in enumerate(name_parts) if p is not None and w[i]>0]
                if not weighted:continue
                owner=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siren',brand['owner_refs']),0.)
                parts.append(tv.Query.boolean_query([(tv.Occur.Must,owner),(tv.Occur.Must,admin),
                    (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in weighted]))]))
                expanded=True
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    query=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))])
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        doc=backend.searcher.doc(loc).to_dict();siret=backend._first(doc,'siret');siren=backend._first(doc,'siren')
        if len(siret)!=14 or not siret.isdigit() or not siret.startswith(siren) or not np.isfinite(score):raise ValueError('Valid native official leaf score')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('One native unique100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(total_seconds=time.monotonic()-start,query_build_seconds=built-start,search_seconds=time.monotonic()-built,
        full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,no_prepool=True,one_collector=True,
        indirect_lexical_expansion_used=expanded,original_national_branch_unrestricted=True,
        original_name_coefficients_exact=True,source_confidence_fixed=.5,quantization_abs_error_bound=base_error+bound/maximum)


def bind_models(out):supplier.bind_models(out)


def stage(action,out):
    bind_runtime(out);pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    elif action=='bind_models':bind_models(out)
    else:getattr(pipeline.learned,action)(out)
    if action=='evaluate':
        report=io.read(out/'report.json');report.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            original_weights_exact=True,no_extra_fitting=True,weak_indirect_lexical_not_identity=True)
        io.write(out/'report.json',report)


def run(out):
    import fcntl,subprocess
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);deadline=pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('bind_models','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v81_indirect_lexical_query',action,'--output',str(out)]
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
            if rc:raise RuntimeError(f'{action} exit {rc}, no partial recall')
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
