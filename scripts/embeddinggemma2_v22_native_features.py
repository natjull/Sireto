"""Full-stream proof features and native national learned-query primitives.

No labels, sampled candidate pool, reranker, or downstream mutation here.
All unpruned Gemma leaves, all geographic footprint leaves, and their direct
official successor views contribute before Tantivy's ONE collector100.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v21_relation_views as relations
from scripts import embeddinggemma2_v2_data as data
from scripts import retrieval_v38_initial_top100 as lexical
from src.xgb_matcher.hierarchical_retrieval import normalize_text
from src.xgb_matcher.retrieval_query_ban_v327 import surface
from src.xgb_matcher.retrieval_v20_workbench import record,verify

CACHE=relations.CACHE
STRUCTURE=io.HOME/'v21/relation_views_20261011_c'
OUTPUT=io.HOME/'v22/learned_native_20261012_a'
LEXICAL_FIELDS=['word_names','char_names','word_addresses','char_addresses','administrative_geo']
PROOF_FIELDS=['site_cosine','parent_cosine','own_name_similarity','own_address_proof','own_street_similarity',
    'own_geo_quality','site_x_geo','name_x_geo','parent_x_address','name_x_address',
    'own_geo_name_distance','succession_geo_context','succession_geo_close_context',
    'succession_geo_recent_context','succession_dense_context','succession_dense_close_context']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(scope='ALL526_CONSUMED_DEVELOPMENT_FULL_PROOF_CORPORA_PLUS_NATIONAL_LEXICAL',
    feature_fields=FEATURE_FIELDS,score='LEARNED_NONNEGATIVE_LINEAR_PROOF_QUERY_NATIVE_TANTIVY_SUM',
    native_collector=100,quantization_bits=20,
    lexical_word_normalization='EXACT_SHARED_INDEX_NFKD_ACCENT_FOLD_ASCII_ALNUM',
    quantization='FLOOR_FULL_NONLEXICAL_SUM_FIXED_COEFFICIENT_BOUND_BINARY_TERMSET_PLANES',
    quantization_is_declared_scientific_approximation=True,
    corpus='ALL_ORIGINAL_COMPLETE_COMMUNE_LEAVES_ALL_NATIONAL_BAN_FOOTPRINT_LEAVES_AND_DIRECT_SUCCESSIONS',
    national_lexical='UNRESTRICTED_ORIGINAL_FULL82772691_DOCUMENT_INDEX_SIRET_ONLY',
    zero_lexical_allowed_for_nonlexical_proof=True,wrong_or_missing_insee_not_a_global_filter=True,
    source_roles='OWN_AND_PARENT_AND_DIRECT_SUCCESSION_FEATURES_SEPARATE',
    relationship_policy=relations.RECIPE['eligibility'],relationship_hops=1,
    relationship_is_not_own_alias=True,identifiers='REFERENCE_LOOKUP_AND_OUTPUT_ONLY_NEVER_SCORE_FEATURE',
    full_stream_no_fixed_prepool=True,no_reranking_of_capped_channels=True,
    no_new_encoder_fit=True,no_downstream_change=True,no_closed_fold_opening=True)


def guard(deadline):
    if time.time()>=deadline:raise TimeoutError('EXPLICIT_SEVEN_HOUR_DEADLINE')
    engine.guard()


def feature_path(out,qid):
    return out/'features'/(hashlib.sha256(qid.encode()).hexdigest()+'.npz')


def initialize(out,deadline):
    if out.exists():raise ValueError('Fresh namespace only')
    if deadline<=time.time() or deadline>time.time()+7*3600:raise ValueError('User absolute deadline, no new night')
    source=engine.authority(structural.SOURCE)
    panel=io.read(verify(source['panel']));surface(panel)
    if len(panel)!=526:raise ValueError('All526 typed safe-fold query surface')
    old=io.read(CACHE/'launch.json');verify(old['implementation'])
    sx=io.read(STRUCTURE/'generate_exit.json');sg=io.read(STRUCTURE/'generation.json')
    if sx['exit_code']!=0 or not sx['own_child_exit_confirmed'] or sg['status']!='COMPLETE_SEALED_ALL526':
        raise ValueError('Complete query-only structural source owned exit0')
    out.mkdir(parents=True);(out/'features').mkdir();io.write(out/'recipe.json',RECIPE)
    launch=dict(created_unix=time.time(),deadline_unix=deadline,recipe=record(out/'recipe.json'),
        implementation=record(Path(__file__)),helpers=[record(Path(m.__file__)) for m in (io,engine,structural,relations,lexical,data)],
        source_launch=record(structural.SOURCE/'launch.json'),feature_launch=record(CACHE/'launch.json'),
        structural_launch=record(STRUCTURE/'launch.json'),structural_generation=record(STRUCTURE/'generation.json'),
        panel=source['panel'],geo_sources=record(engine.sealed(STRUCTURE/'geo_sources.json')),
        links=sg['links'],geometry=old['geo'],xy=source['geographic']['xy'],keys=source['geographic']['keys'],
        index_manifest=source['index_manifest'],index_meta=source['index_meta'],index_inventory=source['index_inventory'],
        adapter=old['adapter'],all526_development_only=True,unit_tests_requested=False,
        labels_opened_in_feature_generation=False,closed_test_opened=False,downstream_changed=False)
    launch['helpers'].append(record(io.ROOT/'src/xgb_matcher/hierarchical_retrieval.py'))
    io.write(out/'launch.json',launch)


def authority(out):
    launch=io.read(out/'launch.json')
    for ref in (launch['implementation'],launch['recipe'],launch['source_launch'],launch['feature_launch'],
                launch['structural_launch'],launch['structural_generation'],*launch['helpers']):verify(ref)
    guard(launch['deadline_unix'])
    return launch


class EdgeViews:
    def __init__(self,launch):
        import pyarrow.parquet as pq
        table=pq.read_table(verify(launch['links']),use_threads=False)
        p=np.array(table['siretEtablissementPredecesseur'].to_pylist(),dtype=np.uint64)
        s=np.array(table['siretEtablissementSuccesseur'].to_pylist(),dtype=np.uint64)
        dates=table['dateLienSuccession'].to_numpy().astype('datetime64[D]')
        continuity=table['continuiteEconomique'].to_pylist()
        if not all(v is True for v in continuity) or np.isnat(dates).any() or (dates>np.datetime64('2026-08-01')).any():
            raise ValueError('Exactly the original typed dated economic-continuity observations')
        self.keys=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
        self.xy=np.load(verify(launch['xy']),mmap_mode='r',allow_pickle=False)
        self.sources=np.concatenate((p,s));self.targets=np.concatenate((s,p))
        age=np.maximum(0,(np.datetime64('2026-08-01')-dates).astype(float)/365.25)
        self.recency=np.tile(1/(1+age),2).astype(np.float32)
        a=self.coordinates(self.sources);b=self.coordinates(self.targets)
        close=np.zeros(len(a),dtype=np.float32);valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
        close[valid]=1/(1+np.hypot(*(a[valid]-b[valid]).T)/300)
        self.close=close
        loc=np.searchsorted(self.keys,self.targets);valid=loc<len(self.keys)
        valid[valid]&=self.keys[loc[valid]]==self.targets[valid]
        order=np.flatnonzero(valid);order=order[np.argsort(self.sources[order],kind='stable')]
        self.sources=self.sources[order];self.targets=self.targets[order]
        self.recency=self.recency[order];self.close=self.close[order]
        del table,p,s,dates,age,a,b,continuity

    def coordinates(self,keys):
        loc=np.searchsorted(self.keys,keys);found=loc<len(self.keys)
        found[found]&=self.keys[loc[found]]==keys[found]
        coords=np.full((len(keys),2),np.nan);coords[found]=self.xy[loc[found]]
        valid=np.isfinite(coords).all(axis=1)&(coords[:,0]>=0)&(coords[:,0]<=1300000)&(coords[:,1]>=6000000)&(coords[:,1]<=7300000)
        coords[~valid]=np.nan;return coords

    def select(self,keys):
        lo=np.searchsorted(self.sources,keys,'left');hi=np.searchsorted(self.sources,keys,'right');counts=hi-lo
        n=int(counts.sum())
        return np.arange(n)+np.repeat(lo-np.cumsum(counts)+counts,counts)

    def close_files(self):
        self.keys._mmap.close();self.xy._mmap.close()


def make_features(own_keys,own_features,geo_rows,geometry,edges):
    """No candidate ceiling anywhere before the final native search."""
    own_keys=np.asarray(own_keys,dtype=np.uint64)
    if len(np.unique(own_keys))!=len(own_keys) or own_features.shape!=(len(own_keys),5):raise ValueError('Full unique aligned own scores')
    geo_keys=np.array([r[0] for r in geo_rows],dtype=np.uint64)
    sources=np.union1d(own_keys,geo_keys);edge=edges.select(sources)
    keys=np.union1d(sources,edges.targets[edge]);x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32)
    own=np.searchsorted(keys,own_keys);x[own,:5]=np.clip(own_features,0,1)
    coords=edges.coordinates(keys);valid=np.isfinite(coords).all(axis=1)
    if geometry['spatial_crs_compatible']:
        reliability=.5 if geometry['ban_declared_weak'] else 1.
        distance=np.hypot(coords[:,0]-geometry['ban_x_l93'],coords[:,1]-geometry['ban_y_l93'])
        x[valid,5]=reliability/(1+distance[valid]/(300+max(0,geometry['ban_uncertainty_m'])))
    x[:,6]=x[:,0]*x[:,5];x[:,7]=x[:,2]*x[:,5]
    x[:,8]=x[:,1]*x[:,3];x[:,9]=x[:,2]*x[:,3]
    if len(geo_keys):x[np.searchsorted(keys,geo_keys),10]=[r[1] for r in geo_rows]
    if len(edge):
        fr=np.searchsorted(keys,edges.sources[edge]);to=np.searchsorted(keys,edges.targets[edge])
        geo=x[fr,10];dense=.75*x[fr,0]+.25*x[fr,1]
        for col,values in ((11,geo),(12,geo*edges.close[edge]),(13,geo*edges.recency[edge]),
                           (14,dense),(15,dense*edges.close[edge])):
            np.maximum.at(x[:,col],to,values)
    if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():raise ValueError('Finite bounded typed proof scores')
    return keys,x


def prepare(out):
    launch=authority(out);cache=io.read(verify(launch['feature_launch']));panel=io.read(verify(launch['panel']));surface(panel)
    geo=io.read(verify(launch['geo_sources']));geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    edges=EdgeViews(launch);refs={};commune_sites=0;total_rows=0;started=time.monotonic()
    try:
        for zone,corpus in sorted(cache['corpora'].items()):
            guard(launch['deadline_unix']);dest=CACHE/'zones'/(zone or 'UNKNOWN');qs=[q for q in panel if q['crm_insee']==zone]
            arrays={};own_keys=np.array([],dtype=np.uint64)
            if (dest/'features_binding.json').exists():
                binding=io.read(engine.sealed(dest/'features_binding.json'))
                if binding['corpus']!=corpus or binding['queries']!=qs or binding['adapter']!=launch['adapter'] or binding['recipe']!=cache['recipe'] or not binding['full_unpruned_scores']:
                    raise ValueError('Exact full feature query/corpus/model binding')
                own_keys=np.array(binding['sirets'],dtype=np.uint64)
                arrays={f:np.load(engine.sealed(dest/(f+'.npy')),mmap_mode='r',allow_pickle=False) for f in ('site','identity','name','address','street')}
                if any(a.shape!=(len(qs),len(own_keys)) for a in arrays.values()):raise ValueError('Exact full source matrix dimensions')
            elif io.read(verify(corpus)):raise ValueError('Missing nonempty original corpus features')
            try:
                for i,q in enumerate(qs):
                    guard(launch['deadline_unix']);path=feature_path(out,q['query_id']);saved=engine.sealed(path)
                    receipt=path.with_suffix('.binding.json')
                    expected=dict(query=q,recipe=launch['recipe'],feature_launch=launch['feature_launch'],
                        geo_sources=launch['geo_sources'],links=launch['links'],adapter=launch['adapter'])
                    if saved:
                        if io.read(engine.sealed(receipt))!=expected:raise ValueError('Same scientific binding for resumed features')
                        with np.load(saved,allow_pickle=False) as values:n=len(values['keys'])
                    else:
                        own=np.column_stack([arrays[f][i] for f in arrays]) if len(own_keys) else np.zeros((0,5),dtype=np.float32)
                        keys,x=make_features(own_keys,own,geo[q['query_id']],geometries[q['query_id']],edges);n=len(keys)
                        temp=path.with_suffix('.pending.npz');np.savez(temp,keys=keys,features=x)
                        os.replace(temp,path);engine.seal(path);io.write(receipt,expected);engine.seal(receipt)
                        del own,keys,x
                    refs[q['query_id']]=record(path);total_rows+=n
                    engine.progress(out,'QUERY_ONLY_FULL_PROOF_FEATURES',len(refs),total=len(panel),
                        zone=zone,full_feature_rows=total_rows,elapsed_s=time.monotonic()-started)
            finally:
                for a in arrays.values():a._mmap.close()
            commune_sites+=len(own_keys);gc.collect()
        if commune_sites!=2554848 or set(refs)!={q['query_id'] for q in panel}:raise ValueError('Every complete commune/all526 before calibration')
        authority(out);io.write(out/'features_generation.json',dict(status='COMPLETE_SEALED_QUERY_ONLY_ALL526_FULL_PROOFS',
            features=refs,queries=526,commune_sites=commune_sites,full_feature_rows=total_rows,
            labels_used=False,no_candidate_budget=True,no_gt_directed_edges=True))
    finally:edges.close_files()


def lexical_parts(backend,q):
    """Five independent actual global-index score features, no geo filter."""
    tv=backend._tantivy
    name=normalize_text(data.safe(q['crm_name']))
    address=normalize_text(' '.join(data.safe(q[k]) for k in ('crm_number','crm_suffix','crm_street') if data.safe(q[k])))
    def maximum(fields,text):
        parts=[lexical.mean_query(backend,f,lexical.tokens(text)) for f in fields]
        parts=[p for p in parts if p is not None]
        return tv.Query.disjunction_max_query(parts,tie_breaker=0.) if parts else None
    geos=[tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,f,v,'basic'),1.)
          for f,v in (('insee',q['crm_insee']),('postcode',q['crm_postcode'])) if v]
    return [maximum(lexical.NAMES,name),
        lexical.mean_query(backend,'current_name_ngrams',lexical.tokens(name,True)),
        maximum(lexical.ADDRESSES,address),
        lexical.mean_query(backend,'address_ngrams',lexical.tokens(address,True)),
        tv.Query.disjunction_max_query(geos,tie_breaker=0.) if geos else None]


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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare']);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--deadline-unix',type=float);args=p.parse_args()
    if args.action=='initialize':initialize(args.output,args.deadline_unix)
    else:prepare(args.output)
