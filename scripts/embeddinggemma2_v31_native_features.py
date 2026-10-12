"""V31: unchanged sealed V23 full proofs; only training objective/mining changes."""
from pathlib import Path
import argparse
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from src.xgb_matcher.retrieval_v20_workbench import record, verify

PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
OUTPUT=io.HOME/'v31/local_lifecycle_views_20261012_a'
MODEL_SOURCE=io.HOME/'v25/coherent_query_experts_20261012_a'
# Compiler receives one already learned full-stream query-view score.
FEATURE_FIELDS=list(source.LEXICAL_FIELDS)+['same_learned_local_lifecycle_score']
RECIPE=dict(source.RECIPE,feature_reuse='EXACT_SEALED_V23_ALL526',feature_values_changed=False,
    scorer='V25_SAME_MODEL_MAX_OWN_SCORE_ONE_DATED_ECONOMIC_SUCCESSION_LOCAL_SOURCE_VIEW',
    no_mixture_of_heterogeneous_scores=True,source_local_columns=[0,2,3,4,5,6,7,9,10],
    source_view_no_parent_borrowing_no_relation_fields=True,
    original_source_family_scope_before_one_link_only=True,
    candidate_context_is_not_own_name_address_or_identity=True,relationship_hops=1,no_new_fit=True)

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
    prior=io.read(MODEL_SOURCE/'calibration/generation.json')
    ex=io.read(MODEL_SOURCE/'generate_exit.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed']:raise ValueError('Complete owned V25 source')
    launch.update(model_generation=record(MODEL_SOURCE/'calibration/generation.json'),
        reference_outputs=record(MODEL_SOURCE/'outputs.json'),family_links=record(PREPARED/'family_links.parquet'))
    launch['helpers'].append(record(Path(source.__file__)))
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for name in ('prepared_launch','prepared_generation','prepared_exit','model_generation','reference_outputs','family_links'):verify(launch[name])
    return launch

def prepare(out):
    launch=authority(out)
    generation=io.read(verify(launch['prepared_generation']))
    if generation['status']!='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS':
        raise ValueError('Complete query-only corpus required')
    io.write(out/'features_generation.json',dict(generation,
        immutable_shared_source=launch['prepared_generation'],labels_used=False))
    io.write(out/'worker_state.json',dict(status='REUSED_SEALED_IDENTICAL_V23_PROOFS',full_feature_rows=generation['full_feature_rows']))

def native_query(backend,q,keys,features,coefficients):
    """ONE native national query, learned lexical + ALL numeric proof views.

    Bit planes attach a *computed proof score* to its official reference. IDs
    have no learned coefficient and are never numerical model inputs. No top-k
    prepool, no sampled subset, no capped channel union is materialized here.
    """
    coefficients=np.asarray(coefficients,dtype=float)
    if coefficients.shape!=(len(FEATURE_FIELDS),) or not np.isfinite(coefficients).all() or (coefficients<0).any():
        raise ValueError('Finite learned nonnegative typed coefficients')
    tv=backend._tantivy;parts=[]
    for component,weight in zip(lexical_parts(backend,q),coefficients[:5]):
        if component is not None and weight>0:parts.append(tv.Query.boost_query(component,float(weight)))
    weights=coefficients[5:];bound=float(weights.sum());bits=RECIPE['quantization_bits'];maximum=(1<<bits)-1
    if bound>0 and len(keys):
        scores=features.astype(np.float64)@weights
        if not np.isfinite(scores).all() or (scores<0).any() or (scores>bound+1e-5).any():raise ValueError('Full numeric proof score bound')
        integers=np.floor(np.clip(scores/bound,0,1)*maximum).astype(np.uint32)
        refs=[f'{int(k):014d}' for k in keys]
        for bit in range(bits):
            selected=np.flatnonzero(integers & (1<<bit))
            if len(selected):
                mask=tv.Query.term_set_query(backend.index.schema,'siret',[refs[i] for i in selected])
                parts.append(tv.Query.const_score_query(mask,float(bound*(1<<bit)/maximum)))
        del scores,integers,refs
    if not parts:return tv.Query.empty_query(),bound/maximum
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]),bound/maximum


def compiled_search(backend,q,keys,features,coefficients):
    started=time.monotonic();query,error=native_query(backend,q,keys,features,coefficients)
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(query,limit=100,count=False).hits:
        raw=backend.searcher.doc(loc).to_dict();siret=backend._first(raw,'siret');siren=backend._first(raw,'siren')
        if not siret.isdigit() or len(siret)!=14 or len(siren)!=9 or not siret.startswith(siren) or not np.isfinite(score):
            raise ValueError('Valid official native leaf and score')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({r['siret'] for r in rows})!=len(rows):raise ValueError('Native unique absolute100')
    rows.sort(key=lambda r:(-r['score'],r['siret']))
    return rows,dict(query_build_seconds=built-started,search_seconds=time.monotonic()-built,
        total_seconds=time.monotonic()-started,quantization_abs_error_bound=error,
        full_nonlexical_leaves=len(keys),national_lexical_sites=43896818,no_prepool=True)


_CONTEXT=None
def context(out):
    global _CONTEXT
    if _CONTEXT is None:
        launch=authority(out)
        _CONTEXT=(source.EdgeViews(launch,launch['family_links']),io.read(verify(launch['geo_sources'])),{})
    return _CONTEXT

def native_search(backend,q,keys,features,coefficients):
    edges,geo,bank=context(OUTPUT)
    weights=np.asarray(coefficients,dtype=float);d=len(source.FEATURE_FIELDS)
    if weights.shape!=(2*d,) or not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Exact V25 head weights')
    coherence=float(features[:,7].max()) if len(features) else 0.
    effective=coherence*weights[:d]+(1-coherence)*weights[d:]
    numeric=effective[5:];scores=features.astype(float)@numeric
    # SOURCE view is strictly site-local and contains no relation or parent
    # borrowing: projecting it once cannot make a two-hop path.
    local_columns=[0,2,3,4,5,6,7,9,10]
    local=features[:,local_columns].astype(float)@numeric[local_columns]
    zone=q['crm_insee']
    if zone not in bank:
        path=source.CACHE/'zones'/(zone or 'UNKNOWN')/'features_binding.json'
        if path.exists():
            binding=io.read(engine.sealed(path))
            if not binding['full_unpruned_scores']:raise ValueError('Full original source zone')
            bank[zone]=np.asarray(binding['sirets'],dtype=np.uint64)//100000
        else:bank[zone]=np.asarray([],dtype=np.uint64)
    parents=np.union1d(bank[zone],np.asarray([int(k)//100000 for k,s in geo[q['query_id']]],dtype=np.uint64))
    sources=source.expand_families(edges.keys,parents)
    selected=edges.select(sources)
    frm=np.searchsorted(keys,edges.sources[selected]);to=np.searchsorted(keys,edges.targets[selected])
    if (frm>=len(keys)).any() or (to>=len(keys)).any() or not np.array_equal(keys[frm],edges.sources[selected]) or not np.array_equal(keys[to],edges.targets[selected]):
        raise ValueError('Exact original full source-family / single-link / target-family binding')
    before=scores.copy()
    np.maximum.at(scores,to,local[frm])
    projected=int((scores>before).sum())
    # Sum of numeric coefficients is still a strict bound for MAX of scores
    # computed with the SAME learned weights, not heterogeneous raw scales.
    reduced=np.concatenate((effective[:5],[numeric.sum()]))
    bound=float(numeric.sum())
    normalized=(scores/bound if bound>0 else scores)[:,None]
    rows,cost=compiled_search(backend,q,keys,normalized,reduced)
    cost.update(typed_direct_links=len(selected),promoted_context_views=projected,
        source_view_excludes_parent_and_relations=True,relationship_hops=1,query_only_coherence=coherence)
    return rows,cost


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)


