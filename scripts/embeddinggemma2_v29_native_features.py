"""V29: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v29/source_robust_native_20261012_a'
RECIPE=dict(source.RECIPE,feature_reuse='EXACT_SEALED_V23_ALL526',feature_values_changed=False,
    scorer='LEARNED_TANH_HIDDEN8_BOUNDED_SIGMOID_PLUS_LINEAR_NATIVE_LEXICAL',
    signed_hidden_weights_allow_learned_contradictions=True,
    source_robustness_training='EXPECTED_POINT25_ALL_NUMERIC_CHANNEL_DROPOUT',
    outside_proof_coverage='IDENTICAL_MODEL_ON_ZERO_FEATURES_ALL_NATIONAL_LEAVES',
    no_fixed_prepool=True)

def initialize(out,deadline):
    source.initialize(out,deadline)
    previous=source.authority(PREPARED)
    generation=io.read(PREPARED/'features_generation.json')
    ex=io.read(PREPARED/'prepare_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or generation['queries']!=526:
        raise ValueError('Owned complete original features only')
    launch=io.read(out/'launch.json')
    if previous['panel']!=launch['panel']:raise ValueError('Same complete query panel')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        prepared_exit=record(PREPARED/'prepare_exit.json'))
    launch['helpers'].append(record(Path(source.__file__)))
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit'):verify(launch[name])
    return launch

def prepare(out):
    launch=authority(out)
    generation=io.read(verify(launch['prepared_generation']))
    if generation['status']!='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS':
        raise ValueError('Complete query-only corpus required')
    io.write(out/'features_generation.json',dict(generation,
        immutable_shared_source=launch['prepared_generation'],labels_used=False))
    io.write(out/'worker_state.json',dict(status='REUSED_SEALED_IDENTICAL_V23_PROOFS',full_feature_rows=generation['full_feature_rows']))


def neural_scores(features,model):
    from scipy.special import expit
    scale=np.asarray(model['numeric_scale']);w=np.asarray(model['hidden_weights']);b=np.asarray(model['hidden_bias'])
    v=np.asarray(model['output_weights']);c=float(model['output_bias']);amplitude=float(model['amplitude'])
    if scale.shape!=(len(PROOF_FIELDS),) or w.shape!=(len(PROOF_FIELDS),8) or b.shape!=(8,) or v.shape!=(8,) or not 0<=amplitude<=30:
        raise ValueError('Typed fixed eight-hidden bounded model')
    if not all(np.isfinite(z).all() for z in (scale,w,b,v,np.asarray([c,amplitude]))) or (scale<=0).any():
        raise ValueError('Finite learned model')
    result=np.empty(len(features),dtype=np.float64)
    for start in range(0,len(features),131072):
        x=features[start:start+131072].astype(float)/scale
        result[start:start+len(x)]=amplitude*expit(np.tanh(x@w+b)@v+c)
    baseline=float(amplitude*expit(np.tanh(b)@v+c))
    if not np.isfinite(result).all() or (result<0).any() or (result>amplitude+1e-8).any():raise ValueError('Bounded proof score')
    return result,baseline

def native_search(backend,q,keys,features,model):
    started=time.monotonic();tv=backend._tantivy;parts=[]
    weights=np.asarray(model['lexical_coefficients'],dtype=float)
    if weights.shape!=(len(LEXICAL_FIELDS),) or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Native lexical weights')
    for component,weight in zip(lexical_parts(backend,q),weights):
        if component is not None and weight>0:parts.append(tv.Query.boost_query(component,float(weight)))
    scores,baseline=neural_scores(features,model)
    bound=float(model['amplitude']);maximum=(1<<RECIPE['quantization_bits'])-1
    refs=[f'{int(k):014d}' for k in keys]
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    if bound>0:
        integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32)
        for bit in range(RECIPE['quantization_bits']):
            positions=np.flatnonzero(integers & (1<<bit))
            if len(positions):
                mask=tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in positions])
                parts.append(tv.Query.const_score_query(mask,float(bound*(1<<bit)/maximum)))
        base=float(bound*np.floor(baseline/bound*maximum)/maximum)
        if base>0:
            # ALL national leaves outside prepared proof coverage receive f(0).
            # Missing cache coverage is not a free candidate bonus or exclusion.
            uncovered=tv.Query.boolean_query([(tv.Occur.Must,leaf),
                (tv.Occur.MustNot,tv.Query.term_set_query(backend.index.schema,'siret',refs))]) if refs else leaf
            parts.append(tv.Query.const_score_query(uncovered,base))
    del scores,refs
    query=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]) if parts else tv.Query.empty_query()
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        raw=backend.searcher.doc(loc).to_dict();siret=backend._first(raw,'siret');siren=backend._first(raw,'siren')
        if not siret.isdigit() or len(siret)!=14 or len(siren)!=9 or not siret.startswith(siren) or not np.isfinite(score):
            raise ValueError('Valid native national leaf')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('Absolute unique100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(query_build_seconds=built-started,search_seconds=time.monotonic()-built,total_seconds=time.monotonic()-started,
        quantization_abs_error_bound=bound/maximum,full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,
        zero_feature_score_outside_coverage=baseline,full_corpus_nonlinear_before100=True,no_prepool=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)



