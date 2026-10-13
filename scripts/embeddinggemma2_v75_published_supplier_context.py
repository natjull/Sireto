"""Frozen query-model transfer to weak published supplier/client-brand context.

Brands are not resolved legal entity edges. Published supplier brand matches
the query and a published client brand matches a public candidate owner name:
this is weak relational lexical evidence, never the site's own alias.
Whole client section, whole public corporate owner scope, every526 query.
"""
import argparse
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v71_frozen_residual as pipeline
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v75/published_supplier_context_20261013_a'
PREVIOUS=pipeline.OUTPUT
PUBLISHED=io.HOME/'v74/archived_client_visuals_20261013_a'
DEADLINE=pipeline.corporate.DEADLINE
CACHE=pipeline.CACHE
LEXICAL_FIELDS=pipeline.LEXICAL_FIELDS
PROOF_FIELDS=pipeline.PROOF_FIELDS
FEATURE_FIELDS=pipeline.FEATURE_FIELDS
RECIPE=dict(pipeline.RECIPE,hypothesis='FROZEN_V71_QUERY_MODELS_WEAK_PUBLISHED_SUPPLIER_CLIENT_BRAND_CONTEXT',
    supplier_relation_not_identity_or_ownership=True,client_brand_not_resolved_legal_entity=True,
    weak_context_confidence=.5,confidence_frozen_before_outcomes=True,
    provider_match='ALL_PUBLISHED_BRAND_TOKENS_IN_QUERY_NOT_FUZZY_ALIAS',
    client_match='ALL_INFORMATIVE_PUBLISHED_CLIENT_BRAND_TOKENS_IN_ANY_CURRENT_PUBLIC_CORPORATE_OWNER_NAME',
    role_parameter_transfer='MAX_WEAK_SUPPLIER_CONTEXT_IN_FOUR_GENERIC_RELATED_NAME_X_OWN_SITE_FEATURES',
    no_refit_on_supplier_examples=True,no_target_identifier_coded=True,
    all_declared_source_clients_used=True,current_public_legal_names_only=True,
    provider_homonym_ambiguity_not_claimed_resolved=True,no_provider_brand_hint_unresolved_legal_entity=False)
guard=pipeline.guard
feature_path=pipeline.feature_path
bind_runtime=pipeline.bind_runtime
lexical_parts=pipeline.lexical_parts
native_search=pipeline.native_search


def initialize(out):
    pipeline.initialize(out)
    previous_exit=io.read(PREVIOUS/'prepare_exit.json')
    if previous_exit['exit_code'] or not previous_exit['own_child_exit_confirmed']:raise ValueError('Complete query-only source features')
    source=io.read(PUBLISHED/'generation.json');review=io.read(PUBLISHED/'visual_review.json')
    if {r['source_sha256'] for r in review['brands']}!={r['source']['sha256'] for r in source['images']}:raise ValueError('Every actually reviewed source image')
    for r in source['images']:verify(r['source']);verify(r['preview'])
    launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        previous_features=record(PREVIOUS/'features_generation.json'),
        published_generation=record(PUBLISHED/'generation.json'),visual_review=record(PUBLISHED/'visual_review.json'),
        frozen_model_producer=str(PREVIOUS))
    launch['helpers'].append(record(Path(pipeline.__file__)));io.write(out/'launch.json',launch)


def authority(out):
    launch=pipeline.authority(out)
    for n in ('previous_features','published_generation','visual_review'):verify(launch[n])
    return launch


def prepare(out):
    import duckdb
    import pyarrow as pa
    launch=authority(out);gen=io.read(verify(launch['previous_features']));review=io.read(verify(launch['visual_review']))
    stop=pipeline.corporate.audit_utils.STOP
    brands=[set(pipeline.original.structural.normalize(r['text']).split())-stop for r in review['brands']]
    if any(not b for b in brands):raise ValueError('Every published client has informative source tokens')
    provider=set(pipeline.original.structural.normalize(review['provider_published_brand']).split())
    parents=set();keys_all=set()
    for ref in gen['features'].values():
        with np.load(verify(ref),allow_pickle=False) as f:
            keys_all.update(map(int,f['keys']));parents.update(map(int,f['keys']//100000))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.register('parent_universe',pa.table({'ref':np.array(sorted(parents),dtype=np.uint64)}))
    c.register('site_universe',pa.table({'ref':np.array(sorted(keys_all),dtype=np.uint64)}));del parents,keys_all
    cols=['denominationUniteLegale','denominationUsuelle1UniteLegale','denominationUsuelle2UniteLegale','denominationUsuelle3UniteLegale','sigleUniteLegale']
    rows=c.execute('SELECT u.siren,'+','.join('u.'+k for k in cols)+" FROM read_parquet(?) u JOIN parent_universe p ON CAST(u.siren AS UBIGINT)=p.ref WHERE u.statutDiffusionUniteLegale='O' AND u.categorieJuridiqueUniteLegale<>'1000'",[str(pipeline.corporate.audit_utils.STOCK)]).fetchall()
    client_name_hints={};counts=np.zeros(len(brands),dtype=np.int64)
    for row in rows:
        hits=[any(b<=set(pipeline.original.structural.normalize(v).split()) for v in row[1:] if v) for b in brands]
        counts+=hits
        if any(hits):client_name_hints[int(row[0])]=1.
    addresses={}
    batches=c.execute("""SELECT e.siret,e.codeCommuneEtablissement,e.numeroVoieEtablissement,
        e.indiceRepetitionEtablissement,e.typeVoieEtablissement,e.libelleVoieEtablissement
        FROM read_parquet(?) e JOIN site_universe p ON CAST(e.siret AS UBIGINT)=p.ref
        WHERE e.statutDiffusionEtablissement='O'""",[str(verify(launch['stock_source']))]).fetch_record_batch(rows_per_batch=8192)
    for batch in batches:
        guard(DEADLINE)
        for r in batch.to_pylist():
            addresses[int(r['siret'])]=(r['codeCommuneEtablissement'],
                pipeline.original.structural.normalize(r['numeroVoieEtablissement'] or ''),
                pipeline.original.structural.normalize(r['indiceRepetitionEtablissement'] or ''),
                pipeline.original.structural.normalize(' '.join(str(r[k] or '') for k in ('typeVoieEtablissement','libelleVoieEtablissement')),street=True))
    c.close();refs={};modified=0;start=time.monotonic()
    for q in io.read(verify(launch['panel'])):
        guard(DEADLINE);qid=q['query_id'];path=feature_path(out,qid)
        if not engine.sealed(path):
            query_tokens=set(pipeline.original.structural.normalize(q['crm_name'] or '').split())
            if not provider<=query_tokens:
                os.link(verify(gen['features'][qid]),path);engine.seal(path)
            else:
                with np.load(verify(gen['features'][qid]),allow_pickle=False) as f:keys=f['keys'];x=f['features'].copy()
                if x.shape!=(len(keys),51):raise ValueError('Frozen exact source proof schema')
                strength=np.array([client_name_hints.get(int(k)//100000,0.) for k in keys],dtype=np.float32)*RECIPE['weak_context_confidence']
                street=pipeline.original.structural.normalize(q['crm_street'] or '',street=True)
                number=pipeline.original.structural.normalize(q['crm_number'] or '');suffix=pipeline.original.structural.normalize(q['crm_suffix'] or '')
                a=[addresses.get(int(k)) for k in keys]
                cm=np.array([bool(s and q['crm_insee'] and s[0]==q['crm_insee']) for s in a],dtype=np.float32)
                st=cm*np.array([bool(s and street and street==s[3]) for s in a],dtype=np.float32)
                house=st*np.array([bool(s and number and number==s[1] and suffix==s[2]) for s in a],dtype=np.float32)
                spatial=np.maximum(x[:,5],x[:,42])*np.array([s is not None for s in a],dtype=np.float32)
                x[:,47:51]=np.maximum(x[:,47:51],strength[:,None]*np.column_stack((spatial,cm,st,house)))
                if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded weak published brand/site evidence')
                tmp=path.with_suffix('.pending.npz');np.savez(tmp,keys=keys,features=x);os.replace(tmp,path);engine.seal(path)
                modified+=1;del keys,x,a;gc.collect()
        refs[qid]=record(path);engine.progress(out,'ALL_QUERY_WEAK_SUPPLIER_CONTEXT_PROOFS',len(refs),total=526,elapsed_s=time.monotonic()-start)
    io.write(out/'source_hint_receipt.json',dict(published_source=launch['published_generation'],review=launch['visual_review'],
        informative_client_tokens=[sorted(b) for b in brands],candidate_owner_name_matches_by_brand=counts.tolist(),
        corporate_owners_with_lexical_brand_hint=len(client_name_hints),modified_query_shards_this_execution=modified,
        legal_entity_ids_not_resolved=True,no_own_alias_created=True,no_labels_read=True))
    io.write(out/'features_generation.json',dict(gen,features=refs,supplier_source=launch['published_generation'],
        source_hint_receipt=record(out/'source_hint_receipt.json'),weak_unresolved_brand_lexical_context_only=True,labels_used=False))


def bind_models(out):
    while True:
        guard(DEADLINE)
        ex=io.read(PREVIOUS/'calibrate_exit.json') if (PREVIOUS/'calibrate_exit.json').exists() else None
        if (PREVIOUS/'calibration/generation.json').exists() and ex and ex['exit_code']==0 and ex['own_child_exit_confirmed']:break
        if (PREVIOUS/'calibrate_error.json').exists() or (PREVIOUS/'run_error.json').exists():raise ValueError('Frozen producer failed, no model fallback')
        time.sleep(10)
    ref=record(PREVIOUS/'calibration/generation.json');cal=io.read(verify(ref))
    if cal['feature_fields']!=FEATURE_FIELDS:raise ValueError('Identical frozen feature schema')
    for r in cal['models'].values():verify(r)
    root=out/'calibration';root.mkdir(exist_ok=True)
    io.write(root/'generation.json',cal)
    io.write(out/'model_binding.json',dict(producer=ref,owned_exit=record(PREVIOUS/'calibrate_exit.json'),
        coefficients_unchanged=True,no_new_fitting=True,source_role_parameter_transfer=True,
        new_supplier_features_not_seen_in_training=True,recipe=record(out/'recipe.json')))


def stage(action,out):
    bind_runtime(out);pipeline.learned.proof=sys.modules[__name__]
    if action=='prepare':prepare(out)
    elif action=='bind_models':bind_models(out)
    else:getattr(pipeline.learned,action)(out)
    if action=='evaluate':
        report=io.read(out/'report.json');report.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            frozen_v71_coefficients=True,no_supplier_fitting=True,weak_brand_hints_not_resolved_legal_edges=True)
        io.write(out/'report.json',report)


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action,artifact in [('prepare','features_generation.json'),('bind_models','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(DEADLINE);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v75_published_supplier_context',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=DEADLINE-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=DEADLINE-30:raise TimeoutError('EIGHT_HOUR_END')
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
