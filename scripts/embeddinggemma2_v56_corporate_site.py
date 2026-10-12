"""Eight-hour source-independent corporate/site initial100 experiment.

Original Gemma and downstream models are frozen. Corporate names are typed
context, never the site's own alias or an inferred ownership relationship.
One native national collector100; all official corporate parties are eligible.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np

from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_v32_native_features as base
from scripts import embeddinggemma2_v32_learned_query as learned
from scripts import embeddinggemma2_v30_native_features as roles
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as audit_utils
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record, verify

START = 1791841524.0
DEADLINE = 1791870324.0
OUTPUT = io.HOME/'v56/corporate_site_20261013_a'
PREPARED = base.OUTPUT
CACHE = original.CACHE
ROLE_SOURCE = roles.ROLE_SOURCE
STOCK = audit_utils.STOCK.with_name('StockEtablissement.parquet')
LEXICAL_FIELDS = list(base.LEXICAL_FIELDS)+['national_corporate_actor_token_context']
EXTRA = [f'corporate_actor_{kind}_code_{code}' for code in roles.ROLE_CODES
         for kind in ('x_own_geo','x_own_commune','x_own_street','x_own_number_street')]
PROOF_FIELDS = list(original.PROOF_FIELDS)+EXTRA
FEATURE_FIELDS = LEXICAL_FIELDS+PROOF_FIELDS
RECIPE = dict(original.RECIPE, feature_fields=FEATURE_FIELDS,
    full_source_corporate_names_before_native100=True,
    company_role_bins=roles.ROLE_CODES, source_cutoff='2026-08-01',
    global_actor_score='INFORMATIVE_QUERY_TOKEN_IDF_COVERAGE_MAX_PER_SUBJECT',
    query_token_normalization='OFFICIAL_UNICODE_NORMALIZATION_NO_LEARNED_CRM_ALIASES',
    director_context_is_not_ownership_or_own_site_identity=True,
    company_subject_and_related_ids_required=True, physical_people_not_read=True,
    interactions='ACTOR_NAME_WITH_OWN_SITE_GEOGRAPHY_CURRENT_PUBLIC_ADDRESS_ONLY',
    all526_consumed_development_not_pristine_control=True,
    mandate_start_unix=START, mandate_deadline_unix=DEADLINE,
    no_unit_tests=True, no_external_spend=True)
ACTIVE_OUTPUT = OUTPUT
CORPORATE = None
LAST_CORPORATE = None


def guard(deadline):
    if time.time() >= deadline:
        raise TimeoutError('EXPLICIT_EIGHT_HOUR_DEADLINE_ARTIFACTS_PRESERVED')
    engine.guard()


def initialize(out):
    if out.exists():
        raise ValueError('Fresh experiment namespace')
    guard(DEADLINE)
    historical = io.read(PREPARED/'launch.json')
    for ref in (historical['implementation'], historical['recipe'], historical['panel'],
                *historical['helpers']):
        verify(ref)
    generation = io.read(PREPARED/'features_generation.json')
    owned = io.read(PREPARED/'prepare_exit.json')
    if generation['queries'] != 526 or owned['exit_code'] or not owned['own_child_exit_confirmed']:
        raise ValueError('Complete historical source, not restarting expired work')
    out.mkdir(parents=True); (out/'features').mkdir()
    io.write(out/'recipe.json', RECIPE)
    launch = dict(historical, created_unix=time.time(), deadline_unix=DEADLINE,
        implementation=record(Path(__file__)), recipe=record(out/'recipe.json'),
        prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'),
        role_source=record(ROLE_SOURCE), stock_source=record(STOCK),
        helpers=[*historical['helpers'], record(Path(base.__file__)),
                 record(Path(learned.__file__)), record(Path(roles.__file__)),
                 record(Path(audit_utils.__file__))],
        mandate_start_unix=START, explicit_new_human_window=True)
    io.write(out/'launch.json', launch)
    io.write(out/'mandate.json', dict(start_unix=START, deadline_unix=DEADLINE,
        deadline_paris='2026-10-13 07:45:24 Europe/Paris',
        exact_initial100_target=.992, old_nights_not_resumed=True,
        frozen_encoder=True, frozen_downstream=True, no_external_spend=True))


def authority(out):
    launch = io.read(out/'launch.json')
    for field in ('implementation','recipe','panel','prepared_generation','prepared_exit'):
        verify(launch[field])
    for ref in launch['helpers']:
        verify(ref)
    guard(launch['deadline_unix'])
    return launch


def feature_path(out,qid):
    return original.feature_path(out,qid)


def bind_runtime(out):
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT=out
    base.bind_runtime(out)


def compile_context(out,launch):
    import pyarrow.parquet as pq
    saved=engine.sealed(out/'corporate_names.json')
    if saved:
        return io.read(saved)
    context=defaultdict(lambda: defaultdict(set))
    skipped=0;future=0;undated=0
    columns=['subject_siren','related_siren','related_denomination','role_code','effective_date','observed_at']
    for batch in pq.ParquetFile(verify(launch['role_source'])).iter_batches(batch_size=8192,columns=columns):
        guard(launch['deadline_unix'])
        for row in batch.to_pylist():
            subject=row['subject_siren'];actor=row['related_siren']
            if not re.fullmatch(r'[0-9]{9}',subject or '') or not re.fullmatch(r'[0-9]{9}',actor or '') or subject==actor:
                skipped+=1;continue
            if str(row['effective_date'] or '')[:10]>'2026-08-01' or str(row['observed_at'] or '')[:10]>'2026-08-01':
                future+=1;continue
            if not row['effective_date']:undated+=1
            name=original.structural.normalize(original.data.safe(row['related_denomination']))
            name=re.split(r'\b(?:REPRESENTE|REPRESENTEE|REPRESENTES|REPRESENTEES|REPR)\s+(?:PAR|P[.]?)\b',name,maxsplit=1)[0].strip()
            code=str(row['role_code']);code=code if code in roles.ROLE_CODES[:-1] else 'OTHER'
            if name:context[subject][code].add(name)
    result={p:{c:sorted(v) for c,v in groups.items()} for p,groups in context.items()}
    io.write(out/'corporate_names.json',result);engine.seal(out/'corporate_names.json')
    io.write(out/'context_receipt.json',dict(source=launch['role_source'],subjects=len(result),
        invalid_missing_ids_excluded=skipped,future_excluded=future,unknown_dates_retained=undated,
        all_source_not_gt_seeded=True,relation_context_not_ownership=True))
    return result


def prepare(out):
    import duckdb
    import pyarrow as pa
    launch=authority(out);generation=io.read(verify(launch['prepared_generation']))
    context=compile_context(out,launch);panel=io.read(verify(launch['panel']))
    # Select the immutable full evidence universe, not labels or previous100.
    all_keys=set()
    for ref in generation['features'].values():
        with np.load(verify(ref),allow_pickle=False) as f:all_keys.update(map(int,f['keys']))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    (out/'spill').mkdir(exist_ok=True);c.execute('SET temp_directory=?',[str(out/'spill')])
    c.register('evidence_universe',pa.table({'key':np.array(sorted(all_keys),dtype=np.uint64)}));del all_keys
    cols=['siret','statutDiffusionEtablissement','codeCommuneEtablissement','numeroVoieEtablissement',
          'indiceRepetitionEtablissement','typeVoieEtablissement','libelleVoieEtablissement']
    rows=c.execute('SELECT '+','.join('e."'+x+'"' for x in cols)+
        ' FROM read_parquet(?) e JOIN evidence_universe p ON CAST(e.siret AS UBIGINT)=p.key',
        [str(verify(launch['stock_source']))]).fetch_record_batch(rows_per_batch=8192)
    site={}
    for batch in rows:
        guard(launch['deadline_unix'])
        for row in batch.to_pylist():
            if row['statutDiffusionEtablissement']!='O':continue
            key=int(row['siret'])
            site[key]=(row['codeCommuneEtablissement'],
                original.structural.normalize(row['numeroVoieEtablissement']),
                original.structural.normalize(row['indiceRepetitionEtablissement']),
                original.structural.normalize(' '.join(str(row[x] or '') for x in ('typeVoieEtablissement','libelleVoieEtablissement')),street=True))
    c.close();refs={};total=0;started=time.monotonic()
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        if not engine.sealed(path):
            with np.load(verify(generation['features'][qid]),allow_pickle=False) as f:
                keys=f['keys'];old=f['features']
            x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32);x[:,:len(original.PROOF_FIELDS)]=old
            parents,group=np.unique(keys//100000,return_inverse=True)
            name=original.structural.normalize(original.data.safe(q['crm_name']))
            street=original.structural.normalize(q['crm_street'],street=True)
            number=original.structural.normalize(q['crm_number']);suffix=original.structural.normalize(q['crm_suffix'])
            commune=np.zeros(len(keys),dtype=np.float32);street_match=np.zeros(len(keys),dtype=np.float32);house=np.zeros(len(keys),dtype=np.float32)
            for i,k in enumerate(keys):
                s=site.get(int(k))
                if s is None:continue
                commune[i]=float(bool(q['crm_insee']) and s[0]==q['crm_insee'])
                street_match[i]=commune[i]*float(bool(street) and street==s[3])
                house[i]=street_match[i]*float(bool(number) and number==s[1] and suffix==s[2])
            for j,code in enumerate(roles.ROLE_CODES):
                values=np.array([original.structural.similarity(name,tuple(context.get(f'{int(p):09d}',{}).get(code,())))
                    for p in parents],dtype=np.float32)[group]
                at=len(original.PROOF_FIELDS)+4*j
                x[:,at]=values*x[:,5];x[:,at+1]=values*commune
                x[:,at+2]=values*street_match;x[:,at+3]=values*house
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded typed corporate/site proof')
            tmp=path.with_suffix('.pending.npz');np.savez(tmp,keys=keys,features=x);os.replace(tmp,path);engine.seal(path)
            total+=len(keys);del keys,old,x;gc.collect()
        refs[qid]=record(path)
        engine.progress(out,'ALL_CORPORATE_OWN_SITE_PROOFS',len(refs),total=526,elapsed_s=time.monotonic()-started)
    io.write(out/'features_generation.json',dict(generation,features=refs,
        corporate_names=record(out/'corporate_names.json'),context_receipt=record(out/'context_receipt.json'),
        full_feature_rows=total,implementation=launch['implementation'],labels_used=False,
        published_parent_names=generation['published_parent_names']))


def corporate_component(backend,q):
    global CORPORATE,LAST_CORPORATE
    name=original.structural.normalize(original.data.safe(q['crm_name']))
    if LAST_CORPORATE and LAST_CORPORATE[0] is backend and LAST_CORPORATE[1]==name:return LAST_CORPORATE[2]
    if CORPORATE is None:
        gen=io.read(ACTIVE_OUTPUT/'features_generation.json');rows=io.read(verify(gen['corporate_names']))
        aliases=sorted({text for groups in rows.values() for names in groups.values() for text in names})
        position={text:i for i,text in enumerate(aliases)}
        inverse=defaultdict(set);owners=defaultdict(set)
        for i,text in enumerate(aliases):
            for token in audit_utils.tokens(text):inverse[token].add(i)
        for parent,groups in rows.items():
            for names in groups.values():
                for text in names:owners[position[text]].add(parent)
        CORPORATE=(len(aliases),inverse,owners)
    n,inverse,owner_at=CORPORATE;qt=audit_utils.tokens(name);alias_scores=defaultdict(float)
    norm=0.
    for token in qt:
        df=len(inverse.get(token,()));weight=float(np.log1p((n+.5)/(df+.5)));norm+=weight
        for alias in inverse.get(token,()):alias_scores[alias]+=weight
    scores=defaultdict(float)
    for alias,value in alias_scores.items():
        for parent in owner_at[alias]:scores[parent]=max(scores[parent],value)
    tv=backend._tantivy;parts=[];maximum=(1<<RECIPE['quantization_bits'])-1
    if norm and scores:
        owners=sorted(scores);ints=np.array([int(min(1.,scores[p]/norm)*maximum) for p in owners],dtype=np.uint32)
        for bit in range(RECIPE['quantization_bits']):
            chosen=np.flatnonzero(ints & (1<<bit))
            if len(chosen):parts.append(tv.Query.const_score_query(
                tv.Query.term_set_query(backend.index.schema,'siren',[owners[i] for i in chosen]),float((1<<bit)/maximum)))
    result=tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]) if parts else None
    LAST_CORPORATE=(backend,name,result)
    return result


def lexical_parts(backend,q):
    return base.lexical_parts(backend,q)+[corporate_component(backend,q)]


def native_search(backend,q,keys,features,coefficients):
    # Reuse verified quantization/one-collector implementation in this process.
    old=(base.LEXICAL_FIELDS,base.PROOF_FIELDS,base.FEATURE_FIELDS,base.lexical_parts)
    try:
        base.LEXICAL_FIELDS=LEXICAL_FIELDS;base.PROOF_FIELDS=PROOF_FIELDS;base.FEATURE_FIELDS=FEATURE_FIELDS
        base.lexical_parts=lambda backend,q: original.lexical_parts(backend,q)+[base.parent_component(backend,q),corporate_component(backend,q)]
        return base.native_search(backend,q,keys,features,coefficients)
    finally:base.LEXICAL_FIELDS,base.PROOF_FIELDS,base.FEATURE_FIELDS,base.lexical_parts=old


def stage(action,out):
    learned.proof=sys.modules[__name__]
    learned.RECIPE=dict(learned.RECIPE,model='NATIONAL_CORPORATE_OWN_SITE_NONNEGATIVE_PAIRWISE_L2',
        global_actor_independent_discovery=True,feature_fields=FEATURE_FIELDS)
    if action=='prepare':prepare(out)
    else:getattr(learned,action)(out)


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned completion required')
            continue
        while True:
            guard(DEADLINE)
            command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v56_corporate_site',action,'--output',str(out)]
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
            if engine.INTERRUPTED or time.time()>=DEADLINE-30:raise TimeoutError('EIGHT_HOUR_END_ARTIFACTS_PRESERVED')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} owned exit {rc}; no partial recall')
            break
    stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','calibrate','generate','evaluate'])
    p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
