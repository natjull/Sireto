
"""V38 changes identity scope, reuses V25 heads; no labels or new fit in generation."""
from pathlib import Path
import argparse,gc,os,time
import numpy as np
from scripts import embeddinggemma2_v23_native_features as source
from scripts.embeddinggemma2_v23_native_features import *
from scripts import embeddinggemma2_v36_native_features as bank_source
from src.xgb_matcher.retrieval_v20_workbench import record,verify
# Restore source after wildcard import for explicit provenance.
from scripts import embeddinggemma2_v23_native_features as source
PREPARED=io.HOME/'v23/hierarchical_native_20261012_a'
MODEL_SOURCE=io.HOME/'v25/coherent_query_experts_20261012_a'
BANK_SOURCE=io.HOME/'v36/global_cached_parent_dense_forder_20261012_a'
OUTPUT=io.HOME/'v38/global_identity_exact_bounds_20261012_a'
ACTIVE_OUTPUT=OUTPUT
BANK=None
FEATURE_FIELDS=list(source.LEXICAL_FIELDS)+['numeric_remainder_with_global_identity_interactions']
RECIPE=dict(source.RECIPE,
    score='SAME_V25_HEADS_GLOBAL_CACHED_PARENT_IDENTITY_REPLACES_LOCAL_IDENTITY',
    identity_scope='ALL2126387_CACHED_PARENTS_NOT_ALL_FRENCH_LEGAL_ENTITIES',
    parent_identity_broadcast_all_national_members=True,
    original_site_geo_names_addresses_and_relation_fields_unchanged=True,
    parent_x_own_address_and_parent_identity_x_parent_geo_recomputed=True,
    no_new_fit=True,no_parent_beam=True,no_quantization_tolerance_relaxed=True,
    pruning='ADMISSIBLE_NUMERIC100TH_LOWER_BOUND_VS_PARENT_ONLY_UPPER_BOUND',
    no_fixed_parent_pool=True,eligibility_zero_score=True)

def bind_runtime(out):
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT=out

def initialize(out,deadline):
    source.initialize(out,deadline)
    launch=io.read(out/'launch.json');prior=source.authority(PREPARED)
    if prior['panel']!=launch['panel']:raise ValueError('Same526')
    io.write(out/'recipe.json',RECIPE)
    launch.update(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        prepared_launch=record(PREPARED/'launch.json'),prepared_generation=record(PREPARED/'features_generation.json'),
        model_generation=record(MODEL_SOURCE/'calibration/generation.json'),
        reference_outputs=record(MODEL_SOURCE/'outputs.json'),
        pending_bank_launch=record(BANK_SOURCE/'launch.json'))
    launch['helpers'].extend([record(Path(source.__file__)),record(Path(bank_source.__file__))])
    io.write(out/'launch.json',launch)

def authority(out):
    launch=source.authority(out)
    for key in ('prepared_launch','prepared_generation','model_generation','reference_outputs','pending_bank_launch'):verify(launch[key])
    return launch

def prepare(out):
    launch=authority(out);original=io.HOME/'v37/global_identity_existing_heads_20261012_a'
    ex=io.read(original/'prepare_exit.json');gen=io.read(original/'features_generation.json')
    if ex['exit_code']!=0 or not ex['own_child_exit_confirmed'] or not gen['logical_scores_copied_exactly']:raise ValueError('Owned complete rowmajor bank')
    for ref in (gen['global_parent_scores'],gen['parent_keys']):verify(ref)
    io.write(out/'features_generation.json',dict(gen,immutable_rowmajor_reuse=record(original/'features_generation.json')))

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




def native_search(backend,q,keys,features,coefficients):
    global BANK
    started=time.monotonic();launch=authority(ACTIVE_OUTPUT)
    if BANK is None:
        gen=io.read(ACTIVE_OUTPUT/'features_generation.json');panel=io.read(verify(launch['panel']))
        scores=np.load(verify(gen['global_parent_scores']),mmap_mode='r',allow_pickle=False)
        parents=np.load(verify(gen['parent_keys']),mmap_mode='r',allow_pickle=False)
        BANK=(scores,parents,[f'{int(k):09d}' for k in parents],{r['query_id']:i for i,r in enumerate(panel)})
    matrix,parents,refs,positions=BANK
    identity=matrix[positions[q['query_id']]]
    w=np.asarray(coefficients,dtype=float);d=len(source.FEATURE_FIELDS)
    if w.shape!=(2*d,) or not np.isfinite(w).all() or (w<0).any():raise ValueError('Original V25 typed heads')
    r=float(features[:,7].max()) if len(features) else 0.
    effective=r*w[:d]+(1-r)*w[d:];numeric=effective[5:].copy()
    at=np.searchsorted(parents,keys//100000);inside=at<len(parents);inside[inside]&=parents[at[inside]]==(keys[inside]//100000)
    x=features.copy()
    x[inside,1]=identity[at[inside]]
    x[:,8]=x[:,1]*x[:,3];x[:,17]=x[:,1]*x[:,16]
    direct=float(numeric[1]);numeric[1]=0
    remainder=x.astype(float)@numeric;bound=float(numeric.sum())
    reduced=np.concatenate((effective[:5],[bound]))
    normalized=(remainder/bound if bound>0 else remainder)[:,None]
    query,error=native_query(backend,q,keys,normalized,reduced)
    tv=backend._tantivy;parts=[query];maximum=(1<<RECIPE['quantization_bits'])-1
    if direct>0:
        integers=np.floor(np.clip(identity,0,1)*maximum).astype(np.uint32)
        for bit in range(RECIPE['quantization_bits']):
            chosen=np.flatnonzero(integers & (1<<bit))
            if len(chosen):
                lookup=tv.Query.term_set_query(backend.index.schema,'siren',[refs[i] for i in chosen])
                parts.append(tv.Query.const_score_query(lookup,float(direct*(1<<bit)/maximum)))
    leaf=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    eligible=None;lower_bound=None;upper_parent=None;parent_only_count=len(parents)
    if len(remainder)>=100 and direct>0:
        # Complete numeric corpus, no collected/reranked candidate pool.
        # All referenced endpoints are unique members of the pinned national
        # endpoint catalog/index. Float32 sums use conservative 64-epsilon bounds.
        floor_at=np.argpartition(remainder,len(remainder)-100)[-100:]
        tentative=float(remainder[floor_at].min()-bound/maximum-64*np.finfo(np.float32).eps*bound)
        if tentative>0:
            # Actual index membership check of the 100 witness references;
            # only count is used. No returned seed list becomes production.
            witness=tv.Query.term_set_query(backend.index.schema,'siret',[f'{int(keys[i]):014d}' for i in floor_at])
            check=tv.Query.boolean_query([(tv.Occur.Must,leaf),(tv.Occur.Must,witness)])
            receipt=backend.searcher.search(check,limit=1,count=True)
            if receipt.count==100:
                lower_bound=tentative
                upper_parent=direct*np.asarray(identity,dtype=float)*(1+64*np.finfo(np.float32).eps)
                competitive=np.flatnonzero(upper_parent>=lower_bound)
                parent_only_count=len(competitive)
                choices=[(tv.Occur.Should,query)]
                if len(competitive):
                    choices.append((tv.Occur.Should,tv.Query.term_set_query(backend.index.schema,'siren',[refs[i] for i in competitive])))
                eligible=tv.Query.const_score_query(tv.Query.boolean_query(choices),0.)
    conditions=[(tv.Occur.Must,leaf),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))]
    if eligible is not None:conditions.append((tv.Occur.Must,eligible))
    combined=tv.Query.boolean_query(conditions)
    built=time.monotonic();rows=[]
    for score,loc in backend.searcher.search(combined,limit=100,count=False).hits:
        raw=backend.searcher.doc(loc).to_dict();siret=backend._first(raw,'siret');siren=backend._first(raw,'siren')
        if not siret.isdigit() or len(siret)!=14 or len(siren)!=9 or not siret.startswith(siren) or not np.isfinite(score):raise ValueError('Native national leaf')
        rows.append(dict(siret=siret,siren=siren,score=float(score)))
    if len(rows)>100 or len({z['siret'] for z in rows})!=len(rows):raise ValueError('Absolute unique100')
    rows.sort(key=lambda z:(-z['score'],z['siret']))
    return rows,dict(total_seconds=time.monotonic()-started,query_build_seconds=built-started,search_seconds=time.monotonic()-built,
        quantization_abs_error_bound=(bound+direct)/maximum,full_nonlexical_leaves=len(keys),
        global_identity_parents=len(parents),national_lexical_sites=43896818,no_prepool=True,
        reused_v25_heads_no_new_fit=True,parent_identity_scope_changed=True,
        admissible_numeric_lower_bound=lower_bound,parent_only_competitive=parent_only_count,
        exact_zero_score_eligibility_pruning=eligible is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);a=p.parse_args()
    if a.action=='initialize':initialize(a.output,a.deadline_unix)
    else:prepare(a.output)


