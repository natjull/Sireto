"""Source-rich additive native100 diagnostic with V32 baseline frozen.

No replacement of proven query coefficients. Related enterprise names remain
context, never own aliases. All sources are independent of CRM/GT extraction.
Group membership is historical consolidated context, not current ownership.
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

from scripts import embeddinggemma2_v56_corporate_site as corporate
from scripts import embeddinggemma2_v68_filled_site_geo as filled
from scripts import embeddinggemma2_v72_corporate_incidence as graph
from scripts import embeddinggemma2_v32_native_features as base
from scripts import embeddinggemma2_v32_learned_query as learned
from scripts import embeddinggemma2_v23_native_features as original
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v73/frozen_corporate_peer_20261013_a'
PREVIOUS=io.HOME/'v71/frozen_residual_20261013_a'
CORP=io.HOME/'v57/corporate_site_nullsafe_20261013_a'
GEO=filled.OUTPUT
MOTHER=io.HOME/'v65/bodacc_strict_common_parent_20261013_a'
GROUP=io.HOME/'v70/published_group_members_20261013_a'
CACHE=original.CACHE
LEXICAL_FIELDS=list(corporate.LEXICAL_FIELDS)
PROOF_FIELDS=list(corporate.PROOF_FIELDS)+filled.PROOF_FIELDS[22:]+[
    'related_name_x_own_geo_shared','related_name_x_own_commune_shared',
    'related_name_x_own_street_shared','related_name_x_own_house_shared',
    'corporate_peer_name_x_own_geo','corporate_peer_name_x_own_commune',
    'corporate_peer_name_x_own_street','corporate_peer_name_x_own_house',
    'unresolved_corporate_actor_name_x_own_geo','unresolved_corporate_actor_name_x_own_commune',
    'unresolved_corporate_actor_name_x_own_street','unresolved_corporate_actor_name_x_own_house']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
BASE_AT=np.array(list(range(6))+list(range(7,29)))
EXTRA_AT=np.array([6]+list(range(29,len(FEATURE_FIELDS))))
RECIPE=dict(original.RECIPE,feature_fields=FEATURE_FIELDS,
    hypothesis='V32_FROZEN_ADDITIVE_TYPED_COMMON_CORPORATE_ACTOR_PEER_CONTEXT',
    graph_context='TWO_SOURCE_ROLE_INCIDENCES_NOT_OWNERSHIP_OR_IDENTITY',
    graph_peer_name='MAX_ALL_DISTINCT_OTHER_PUBLIC_SUBJECTS_CURRENT_UL_PUBLISHED_NAMES',
    graph_hub_attenuation='DIV_SQRT_DISTINCT_SUBJECT_DEGREE_MINUS_ONE_NO_TOPK_PRUNING',
    no_self_peer=True,role_codes_retained_source_not_mislabelled_as_ownership=True,
    unresolved_foreign_actor_name='DIRECT_PUBLIC_CORPORATE_ROLE_CONTEXT_ONLY_NEVER_ENTITY_JOINED',
    old_coefficients='EXACT_SAME_HELDOUT_SIREN_FOLD_V32_UNCHANGED',
    old_score_quantization='SEPARATE_BASE_QUERY_PRESERVES_OLD_SCORE_WHEN_RESIDUAL_ZERO',
    new_score='NONNEGATIVE_PAIRWISE_LOGISTIC_WITH_FIXED_BASE_MARGIN_OFFSET_L2_001',
    shared_context='RNE_CORPORATE_ROLES_PUBLISHED_COMMON_PARENT_CLAUSES_HISTORICAL_CONSOLIDATED_PEER_NAMES',
    new_names_never_own_site_aliases=True,group_snapshot='2021-12-31_NOT_CURRENT_MEMBERSHIP',
    group_scope='53_FRENCH_GLOBALLY_CONSOLIDATED_ENTITIES_INCLUDING_ROOT_NOT_ALL_GROUP',
    stock_public_address_only=True,no_provider_brand_hint_unresolved_legal_entity=True,
    numerical_scope='ORIGINAL_FULL_UNPRUNED_PROOF_UNIVERSE_NO_FIXED_POOL',
    eligibility='DIAGNOSTIC_ONLY_NEW_PRIMARY_SOURCES_AND_BAN_SCORE_INELIGIBLE',
    production_promotion_allowed=False,deadline_unix=corporate.DEADLINE)
ACTIVE=OUTPUT


def guard(deadline):corporate.guard(deadline)
def feature_path(out,qid):return original.feature_path(out,qid)
def bind_runtime(out):
    global ACTIVE
    ACTIVE=out;corporate.bind_runtime(out)
def lexical_parts(backend,q):return corporate.lexical_parts(backend,q)


def initialize(out):
    corporate.initialize(out)
    launch=io.read(out/'launch.json');io.write(out/'recipe.json',RECIPE)
    refs={}
    for name,path in [('corp_features',CORP),('geo_features',GEO)]:
        ex=io.read(path/'prepare_exit.json')
        if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete source features')
        refs[name]=record(path/'features_generation.json')
    prior_exit=io.read(PREVIOUS/'prepare_exit.json');gx=io.read(graph.OUTPUT/'owned_exit.json')
    if prior_exit['exit_code'] or not prior_exit['own_child_exit_confirmed'] or gx['exit_code'] or not gx['own_child_exit_confirmed']:raise ValueError('Complete owned whole-source inputs')
    launch.update(previous_features=record(PREVIOUS/'features_generation.json'),
        graph_generation=record(graph.OUTPUT/'generation.json'),graph_exit=record(graph.OUTPUT/'owned_exit.json'))
    oldcal=io.read(base.OUTPUT/'calibration/generation.json')
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),**refs,
        frozen_calibration=record(base.OUTPUT/'calibration/generation.json'),
        frozen_models=oldcal['models'],mother_generation=record(MOTHER/'generation.json'),
        mother_exit=record(MOTHER/'compile_exit.json'),group_generation=record(GROUP/'generation.json'),
        helpers=[*launch['helpers'],record(Path(corporate.__file__)),record(Path(filled.__file__)),record(Path(graph.__file__))])
    io.write(out/'launch.json',launch)


def authority(out):
    launch=corporate.authority(out)
    for name in ('corp_features','geo_features','frozen_calibration','mother_generation','mother_exit','group_generation','previous_features','graph_generation','graph_exit'):verify(launch[name])
    for ref in launch['frozen_models'].values():verify(ref)
    return launch



class PublicIncidence(graph.Incidence):
    # Source references retain their exact nine-digit namespace (2414 nodes
    # begin with zero). Do not modify immutable V72's source compiler.
    def __init__(self):
        self.generation=io.read(graph.OUTPUT/'generation.json');refs=self.generation['artifacts']
        source=io.read(verify(self.generation['launch']))
        for k in ('implementation','roles','stock'):verify(source[k])
        with np.load(verify(refs['incidence.npz']),allow_pickle=False) as f:
            self.ids=f['ids'];self.subject=f['subject'];self.actor=f['actor']
        names=io.read(verify(refs['names.json']));self.unknown=io.read(verify(refs['unresolved_actor_names.json']))
        self.aliases=sorted({v for values in names.values() for v in values});positions={v:i for i,v in enumerate(self.aliases)}
        owner=[];alias=[]
        for i,p in enumerate(self.ids):
            for text in names[f'{int(p):09d}']:owner.append(i);alias.append(positions[text])
        self.owner_at=np.asarray(owner,dtype=np.int32);self.alias_at=np.asarray(alias,dtype=np.int32)
        self.degree=np.bincount(self.actor,minlength=len(self.ids))
        self.unknown_aliases=sorted({n for vals in self.unknown.values() for n in vals})
        positions={v:i for i,v in enumerate(self.unknown_aliases)};owner=[];alias=[]
        for p,vals in self.unknown.items():
            at=int(np.searchsorted(self.ids,int(p)))
            if at>=len(self.ids) or self.ids[at]!=int(p):raise ValueError('Published public corporate subject binding')
            for text in vals:owner.append(at);alias.append(positions[text])
        self.unknown_owner=np.asarray(owner,dtype=np.int32);self.unknown_at=np.asarray(alias,dtype=np.int32)
    def direct(self,name):
        from rapidfuzz import process,fuzz
        result=np.zeros(len(self.ids),dtype=np.float32)
        if name and self.unknown_aliases:
            values=process.cdist([name],self.unknown_aliases,scorer=fuzz.WRatio,processor=None,workers=1,dtype=np.float32)[0]/100
            np.maximum.at(result,self.unknown_owner,values[self.unknown_at])
        return result


def prepare(out):
    import duckdb
    import pyarrow as pa
    launch=authority(out);gen=io.read(verify(launch['previous_features']));incidence=PublicIncidence()
    all_keys=set()
    for ref in gen['features'].values():
        with np.load(verify(ref),allow_pickle=False) as f:all_keys.update(map(int,f['keys']))
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.register('universe',pa.table({'ref':np.array(sorted(all_keys),dtype=np.uint64)}));del all_keys
    rows=c.execute("""SELECT e.siret,e.codeCommuneEtablissement,e.numeroVoieEtablissement,
        e.indiceRepetitionEtablissement,e.typeVoieEtablissement,e.libelleVoieEtablissement
        FROM read_parquet(?) e JOIN universe p ON CAST(e.siret AS UBIGINT)=p.ref
        WHERE e.statutDiffusionEtablissement='O'""",[str(verify(launch['stock_source']))]).fetch_record_batch(rows_per_batch=8192)
    sites={}
    for batch in rows:
        guard(launch['deadline_unix'])
        for r in batch.to_pylist():
            sites[int(r['siret'])]=(r['codeCommuneEtablissement'],
                original.structural.normalize(r['numeroVoieEtablissement'] or ''),
                original.structural.normalize(r['indiceRepetitionEtablissement'] or ''),
                original.structural.normalize(' '.join(str(r[k] or '') for k in ('typeVoieEtablissement','libelleVoieEtablissement')),street=True))
    c.close();refs={};start=time.monotonic();panel=io.read(verify(launch['panel']))
    for q in panel:
        guard(launch['deadline_unix']);qid=q['query_id'];path=feature_path(out,qid)
        if not engine.sealed(path):
            with np.load(verify(gen['features'][qid]),allow_pickle=False) as f:keys=f['keys'];old=f['features']
            if old.shape!=(len(keys),51):raise ValueError('Exact complete prior proof schema')
            parents,group=np.unique(keys//100000,return_inverse=True);loc=np.searchsorted(incidence.ids,parents)
            valid=loc<len(incidence.ids);valid[valid]&=incidence.ids[loc[valid]]==parents[valid]
            name=original.structural.normalize(q['crm_name'] or '')
            graph_values=incidence.query(name);unknown_values=incidence.direct(name)
            peer=np.zeros(len(parents),dtype=np.float32);unknown=np.zeros(len(parents),dtype=np.float32)
            peer[valid]=graph_values[loc[valid]];unknown[valid]=unknown_values[loc[valid]]
            address=[sites.get(int(k)) for k in keys]
            commune=np.array([bool(s and q['crm_insee'] and s[0]==q['crm_insee']) for s in address],dtype=np.float32)
            street=original.structural.normalize(q['crm_street'] or '',street=True)
            number=original.structural.normalize(q['crm_number'] or '');suffix=original.structural.normalize(q['crm_suffix'] or '')
            st=commune*np.array([bool(s and street and street==s[3]) for s in address],dtype=np.float32)
            house=st*np.array([bool(s and number and number==s[1] and suffix==s[2]) for s in address],dtype=np.float32)
            own_geo=np.maximum(old[:,5],old[:,42])*np.array([s is not None for s in address],dtype=np.float32)
            proof=np.column_stack((own_geo,commune,st,house))
            x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32);x[:,:51]=old
            x[:,51:55]=peer[group,None]*proof;x[:,55:59]=unknown[group,None]*proof
            if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Bounded typed public peer/actor context')
            tmp=path.with_suffix('.pending.npz');np.savez(tmp,keys=keys,features=x);os.replace(tmp,path);engine.seal(path)
            del old,keys,x,address,proof;gc.collect()
        refs[qid]=record(path);engine.progress(out,'FULL_TYPED_CORPORATE_PEER_PROOFS',len(refs),total=526,elapsed_s=time.monotonic()-start)
    io.write(out/'features_generation.json',dict(gen,features=refs,previous_generation=launch['previous_features'],
        corporate_graph=launch['graph_generation'],no_self_peer=True,no_own_name_or_address_inheritance=True,
        peer_names_current_stock_only_not_full_historical_names=True,labels_used=False))

def fit_residual(difference,weights,l2):
    from scipy.optimize import minimize
    from scipy.special import expit
    # The original fitter calls once for the smallest as-yet-unsealed fold.
    missing=[k for k in range(5) if not engine.sealed(ACTIVE/'calibration'/f'model_{k}.json')]
    if not missing:raise ValueError('Explicit ordered calibration fold')
    fold=min(missing);launch=authority(ACTIVE);model=io.read(verify(launch['frozen_models'][str(fold)]))
    if model['feature_fields']!=base.FEATURE_FIELDS:raise ValueError('Exact V32 feature schema')
    old=np.asarray(model['coefficients']);margin=difference[:,BASE_AT]@old
    a=difference[:,EXTRA_AT].astype(float);weights=weights/weights.sum()
    scale=np.maximum(np.sqrt(np.sum(a*a*weights[:,None],axis=0)),1e-4);a/=scale
    def objective(theta):
        z=margin+a@theta
        return float(weights@np.logaddexp(0.,-z)+.5*l2*np.dot(theta,theta)), -(a.T@(weights*expit(-z)))+l2*theta
    result=minimize(objective,np.zeros(len(EXTRA_AT)),jac=True,bounds=[(0,None)]*len(EXTRA_AT),method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10,'gtol':1e-7})
    if not result.success:raise ValueError(result.message)
    coefficient=np.zeros(len(FEATURE_FIELDS));coefficient[BASE_AT]=old;coefficient[EXTRA_AT]=result.x/scale
    return coefficient,dict(training_pairwise_loss=float(result.fun),iterations=int(result.nit),
        fixed_baseline=launch['frozen_models'][str(fold)],baseline_weights_unchanged=True,
        baseline_training_margin_in_sample_not_nested_oof=True,train_pairs=len(difference),residual_scale=scale.tolist())


def native_search(backend,q,keys,features,coefficients):
    w=np.asarray(coefficients,dtype=float);tv=backend._tantivy;start=time.monotonic()
    old_query,base_error=base.native_query(backend,q,keys,features[:,:22],w[BASE_AT])
    parts=[old_query]
    actor=corporate.corporate_component(backend,q)
    if actor is not None and w[6]>0:parts.append(tv.Query.boost_query(actor,float(w[6])))
    residual=w[29:];bound=float(residual.sum());maximum=(1<<RECIPE['quantization_bits'])-1
    if bound and len(keys):
        scores=features[:,22:].astype(float)@residual
        integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32);refs=[f'{int(k):014d}' for k in keys]
        for bit in range(RECIPE['quantization_bits']):
            chosen=np.flatnonzero(integers & (1<<bit))
            if len(chosen):parts.append(tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in chosen]),float(bound*(1<<bit)/maximum)))
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    query=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))])
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        doc=backend.searcher.doc(loc).to_dict();siret=backend._first(doc,'siret');siren=backend._first(doc,'siren')
        if len(siret)!=14 or not siret.isdigit() or not siret.startswith(siren):raise ValueError('Valid official leaf')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('Absolute native unique100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(total_seconds=time.monotonic()-start,query_build_seconds=built-start,search_seconds=time.monotonic()-built,
        full_nonlexical_leaves=len(keys),no_prepool=True,baseline_query_not_collected_separately=True,
        baseline_quantization_unchanged=True,quantization_abs_error_bound=base_error+bound/maximum)


def stage(action,out):
    bind_runtime(out);learned.proof=sys.modules[__name__];learned.fit_pairwise=fit_residual
    learned.RECIPE=dict(learned.RECIPE,model='FROZEN_V32_ADDITIVE_RELATED_SOURCE_RESIDUAL',feature_fields=FEATURE_FIELDS,
        frozen_baseline=True,eligibility=RECIPE['eligibility'])
    if action=='prepare':prepare(out)
    else:getattr(learned,action)(out)
    if action=='evaluate':
        report=io.read(out/'report.json');report.update(eligibility=RECIPE['eligibility'],production_promotion_allowed=False,frozen_baseline_weights=True)
        io.write(out/'report.json',report)


def run(out):
    import fcntl
    lock=(io.HOME/(out.name+'.owned.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for action,artifact in [('prepare','features_generation.json'),('calibrate','calibration/generation.json'),('generate','generation.json')]:
        if (out/artifact).exists():
            ex=io.read(out/(action+'_exit.json'))
            if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete stage')
            continue
        while True:
            guard(corporate.DEADLINE);command=[str(engine.CPU),'-u','-m','scripts.embeddinggemma2_v73_frozen_corporate_peer',action,'--output',str(out)]
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RAYON_NUM_THREADS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out))
            with (out/(action+'.log')).open('ab',buffering=0) as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                io.write(out/(action+'_process.json'),dict(pid=child.pid,command=command,owned=True,started_unix=time.time()))
                while child.poll() is None:
                    if engine.INTERRUPTED or time.time()>=corporate.DEADLINE-30:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
                        break
                    time.sleep(5)
                rc=child.wait()
            io.write(out/(action+'_exit.json'),dict(exit_code=rc,pid=child.pid,own_child_exit_confirmed=True,finished_unix=time.time()))
            if engine.INTERRUPTED or time.time()>=corporate.DEADLINE-30:raise TimeoutError('EIGHT_HOUR_END')
            if rc==75:time.sleep(30);continue
            if rc:raise RuntimeError(f'{action} exit {rc}, no partial recall')
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
