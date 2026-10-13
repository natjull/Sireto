"""Explicit source headquarters/parent-context proofs, native national initial100.

Current own public addresses, never CRM/GT aliases, HQ/sibling coordinates or
synthetic points. Original historical evidence and V32 serving weights stay.
No partial-ratio street matching, name-independent address proof. New learner
residual uses nested SIREN-held-out baseline offsets, not in-sample margins.
"""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from scripts import embeddinggemma2_v81_indirect_lexical_query as source
from scripts import embeddinggemma2_v78_nested_residual_offset as nested
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v95/parent_headquarters_context_20261013_a'
CACHE=source.CACHE
LEXICAL_FIELDS=source.LEXICAL_FIELDS
PROOF_FIELDS=source.PROOF_FIELDS+['public_current_headquarters_flag',
    'headquarters_x_parent_identity','headquarters_x_parent_identity_parent_geo',
    'headquarters_x_parent_portfolio_name','headquarters_x_parent_portfolio_name_parent_geo']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
BASE_AT=source.pipeline.BASE_AT
EXTRA_AT=np.array([6]+list(range(29,len(FEATURE_FIELDS))))
RECIPE=dict(source.RECIPE,hypothesis='EXPLICIT_HEADQUARTERS_AS_PARENT_CONTEXT_NOT_OWN_SITE_EVIDENCE',
    headquarters_source='PUBLIC_CURRENT_STOCK_ESTABLISHMENT_SIEGE_BOOLEAN_NOT_SIRET_ORDINAL',
    parent_evidence_may_support_headquarters_alternative=True,
    own_names_addresses_and_coordinates_not_inherited=True,
    not_a_hard_headquarters_preference=True,nonheadquarters_and_closed_retained=True,
    original_frozen_v32_serving_weights_unchanged=True,
    residual_training_offsets='NESTED_OUTER_AND_INNER_SIREN_DISJOINT',
    baseline_training_margin_in_sample_not_nested_oof=False,
    no_new_encoder_fit=True,full_encoder_oof=False,source_recipe_diagnostic_only=True,
    production_promotion_allowed=False)
ACTIVE=OUTPUT
guard=source.guard
feature_path=source.feature_path
lexical_parts=source.lexical_parts
native_search=source.native_search
TYPES=set('RUE AVENUE BOULEVARD ROUTE CHEMIN IMPASSE PLACE ALLEE SQUARE QUAI PASSAGE SENTIER RUELLE'.split())
ARTICLES=set('DE DU DES LA LE LES D L'.split())


def content(street):
    words=source.original.structural.normalize(street or '',street=True).split()
    if words and words[0] in TYPES:words.pop(0)
    return ' '.join(sorted(w for w in words if w not in ARTICLES))


def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;source.bind_runtime(out)


def initialize(out):
    source.initialize(out);launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    ex=io.read(source.OUTPUT/'prepare_exit.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Complete official source')
    nx=io.read(nested.OUTPUT/'calibrate_exit.json')
    if nx['exit_code'] or not nx['own_child_exit_confirmed']:raise ValueError('Complete nested source')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_v81=record(source.OUTPUT/'features_generation.json'),
        original_baseline_calibration=record(source.pipeline.base.OUTPUT/'calibration/generation.json'),
        original_nested_calibration=record(nested.OUTPUT/'calibration/generation.json'),nested_exit=record(nested.OUTPUT/'calibrate_exit.json'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(nested.__file__))])
    io.write(out/'launch.json',launch)


def authority(out):
    launch=source.authority(out)
    for n in ('prepared_v81','original_baseline_calibration','original_nested_calibration','nested_exit'):verify(launch[n])
    return launch


def prepare(out):
    import duckdb
    launch=authority(out);gen=io.read(verify(launch['prepared_v81']))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    rows=c.execute("SELECT siret FROM read_parquet(?) WHERE statutDiffusionEtablissement='O' AND etablissementSiege",[str(verify(launch['stock_source']))]).fetchall()
    headquarters=np.array(sorted(int(r[0]) for r in rows),dtype=np.uint64);c.close();del rows
    refs={};start=time.monotonic()
    for q in io.read(verify(launch['panel'])):
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        if not engine.sealed(path):
            with np.load(verify(gen['features'][qid]),allow_pickle=False) as f:keys=f['keys'];old=f['features']
            if old.shape!=(len(keys),51):raise ValueError('Exact V81 source values')
            loc=np.searchsorted(headquarters,keys);flag=loc<len(headquarters)
            flag[flag]&=headquarters[loc[flag]]==keys[flag];hq=flag.astype(np.float32)
            x=np.column_stack((old,hq,hq*old[:,1],hq*old[:,17],hq*old[:,18],hq*old[:,19])).astype(np.float32)
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded source-only parent/HQ context')
            tmp=path.with_suffix('.pending.npz');np.savez_compressed(tmp,keys=keys,features=x);os.replace(tmp,path);engine.seal(path)
            del keys,old,loc,flag,hq,x;gc.collect()
        refs[qid]=record(path);engine.progress(out,'ALL_PUBLIC_SOURCE_HEADQUARTERS_CONTEXT',len(refs),total=526,elapsed_s=time.monotonic()-start)
    io.write(out/'features_generation.json',dict(gen,features=refs,
        public_headquarters_source=launch['stock_source'],labels_used=False,
        original_nonlexical_values_unchanged=True,headquarters_not_own_site_alias_or_point=True,
        archive_codec='LOSSLESS_NPZ_DEFLATE_NO_VALUE_OR_SCOPE_CHANGE'))


def fit_residual(difference,weights,l2):
    # Called only after all TRAIN shards exist; bind their exact order, reuse
    # already completed20 doubly-purged baselines, never another fold's label.
    launch=authority(ACTIVE);root=ACTIVE/'calibration';original=io.read(verify(launch['original_baseline_calibration']))
    excluded=set(io.read(verify(original['encoder_control_queries'])))
    ordered=[q['query_id'] for q in io.read(verify(launch['panel'])) if q['query_id'] not in excluded]
    shards={q:record(engine.sealed(root/(hashlib.sha256(q.encode()).hexdigest()+'.npz'))) for q in ordered}
    io.write(root/'pair_binding.json',dict(ordered_training_queries=ordered,training_shards=shards,
        newly_mined_train_only=True,old_baseline_pair_values_not_modified=True))
    inner=root/'inner_baselines';inner.mkdir(exist_ok=True)
    for p in (nested.OUTPUT/'calibration/inner_baselines').glob('outer_*_inner_*.json'):
        if p.name.endswith('.seal.json'):continue
        dest=inner/p.name
        if not dest.exists():os.link(engine.sealed(p),dest);engine.seal(dest)
        d=io.read(dest)
        if d['source_calibration']['sha256']!=launch['original_baseline_calibration']['sha256']:raise ValueError('Same doubly-purged baseline authority')
    nested.ACTIVE=ACTIVE;nested.authority=authority;nested.BASE_AT=BASE_AT;nested.EXTRA_AT=EXTRA_AT;nested.FEATURE_FIELDS=FEATURE_FIELDS
    return nested.fit_residual(difference,weights,l2)


def stage(action,out):
    bind_runtime(out);source.pipeline.learned.proof=sys.modules[__name__]
    source.pipeline.learned.fit_pairwise=fit_residual
    source.pipeline.learned.RECIPE=dict(source.pipeline.learned.RECIPE,model='FROZEN_V32_NESTED_HEADQUARTERS_CONTEXT_RESIDUAL',
        feature_fields=FEATURE_FIELDS,nested_baseline_offsets=True,no_hyperparameter_grid=True)
    if action=='prepare':prepare(out)
    else:getattr(source.pipeline.learned,action)(out)
    if action=='evaluate':
        r=io.read(out/'report.json');r.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,
            old_baseline_exact=True,headquarters_is_parent_context_not_site_alias=True,nested_offsets=True)
        io.write(out/'report.json',r)


def run(out):
    import fcntl,subprocess
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=source.pipeline.corporate.DEADLINE
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(deadline);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v95_parent_headquarters_context',action,'--output',str(out)]
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
            if engine.INTERRUPTED or time.time()>=deadline-30:raise TimeoutError('USER_STOP_OR_EIGHT_HOUR_END')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} exit {rc}; no partial recall')
            break
    stage('evaluate',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','run','prepare','calibrate','generate','evaluate']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        elif a.action=='run':run(a.output)
        else:stage(a.action,a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise

