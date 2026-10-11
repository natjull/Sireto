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
OUTPUT=io.HOME/'v23/hierarchical_native_20261012_a'
LEXICAL_FIELDS=['word_names','char_names','word_addresses','char_addresses','administrative_geo']
PROOF_FIELDS=['site_cosine','parent_cosine','own_name_similarity','own_address_proof','own_street_similarity',
    'own_geo_quality','site_x_geo','name_x_geo','parent_x_address','name_x_address',
    'own_geo_name_distance','succession_geo_context','succession_geo_close_context',
    'succession_geo_recent_context','succession_dense_context','succession_dense_close_context',
    'parent_geo_quality','parent_identity_x_parent_geo','parent_portfolio_name_similarity',
    'parent_portfolio_name_x_geo','succession_geo_quality','succession_site_context']
FEATURE_FIELDS=LEXICAL_FIELDS+PROOF_FIELDS
RECIPE=dict(scope='ALL526_CONSUMED_DEVELOPMENT_FULL_PROOF_CORPORA_PLUS_NATIONAL_LEXICAL',
    feature_fields=FEATURE_FIELDS,all_siren_sites_nationwide_before_cutoff=True,
    portfolio_names_are_parent_context_not_site_aliases=True,
    global_parent_geometry_is_context_not_candidate_address=True,
    source_family_then_one_succession_hop_then_target_family_no_second_link=True,score='LEARNED_NONNEGATIVE_LINEAR_PROOF_QUERY_NATIVE_TANTIVY_SUM',
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
        names=old['names'],raw_successions=io.read(relations.SUCCESSION_CONFIG)['source'],
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
    def __init__(self,launch,projected_links=None):
        import pyarrow.parquet as pq
        table=pq.read_table(verify(projected_links or launch['links']),use_threads=False)
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


def expand_families(registry,parents):
    parents=np.asarray(sorted(set(map(int,parents))),dtype=np.uint64)
    lo=np.searchsorted(registry,parents*100000)
    hi=np.searchsorted(registry,(parents+1)*100000);counts=hi-lo
    locations=np.arange(int(counts.sum()))+np.repeat(lo-np.cumsum(counts)+counts,counts)
    return np.asarray(registry[locations])


def compile_family_links(out,launch,cache,geo):
    """Full original succession scan with QUERY-ONLY full ownership scope."""
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    saved=engine.sealed(out/'family_links.parquet')
    if saved:return record(saved)
    parents={int(k)//100000 for rows in geo.values() for k,s in rows}
    for zone,ref in cache['corpora'].items():
        path=CACHE/'zones'/(zone or 'UNKNOWN')/'features_binding.json'
        if path.exists():
            b=io.read(engine.sealed(path));parents.update(int(k)//100000 for k in b['sirets'])
    registry=np.load(verify(launch['keys']),mmap_mode='r',allow_pickle=False)
    try:keys=expand_families(registry,parents)
    finally:registry._mmap.close()
    source=verify(launch['raw_successions'])
    if pq.ParquetFile(source).metadata.num_rows!=9730972:raise ValueError('Complete official original successions')
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='1GB'")
    tmp=out/'duck_tmp';tmp.mkdir(exist_ok=True);con.execute('SET temp_directory=?',[str(tmp)])
    con.register('selected',pa.table({'key':pa.array(keys,type=pa.uint64())}))
    sql="""SELECT DISTINCT s.* FROM read_parquet(?) s WHERE
        continuiteEconomique IS TRUE AND dateLienSuccession <= DATE '2026-08-01' AND
        regexp_full_match(siretEtablissementPredecesseur,'[0-9]{14}') AND
        regexp_full_match(siretEtablissementSuccesseur,'[0-9]{14}') AND
        siretEtablissementPredecesseur <> siretEtablissementSuccesseur AND
        (CAST(siretEtablissementPredecesseur AS UBIGINT) IN (SELECT key FROM selected) OR
         CAST(siretEtablissementSuccesseur AS UBIGINT) IN (SELECT key FROM selected))
        ORDER BY siretEtablissementPredecesseur,siretEtablissementSuccesseur,dateLienSuccession,
                 transfertSiege,continuiteEconomique,dateDernierTraitementLienSuccession"""
    try:relations.write_selection(con,sql,[str(source)],out/'family_links.pending.parquet')
    finally:con.close()
    os.replace(out/'family_links.pending.parquet',out/'family_links.parquet');engine.seal(out/'family_links.parquet')
    io.write(out/'family_sources.json',dict(query_only=True,parent_context_owners=len(parents),
        all_national_source_family_sites=len(keys),original_successions=launch['raw_successions'],
        original_observations=9730972,compiled_links=record(out/'family_links.parquet'),
        no_gt_used=True,no_fixed_candidate_budget=True,no_second_succession_hop=True))
    return record(out/'family_links.parquet')


def load_name_roles(con,parents):
    """All historical/current official sibling aliases remain PARENT context."""
    con.execute('DROP TABLE IF EXISTS temp.wanted_parent')
    con.execute('CREATE TEMP TABLE wanted_parent(parent INTEGER PRIMARY KEY)')
    con.executemany('INSERT INTO wanted_parent VALUES(?)',[(int(p),) for p in parents])
    sql="""SELECT w.parent,a.kind,a.owner_key,a.name
        FROM temp.wanted_parent w CROSS JOIN main.aliases a
        WHERE a.kind=1 AND a.owner_key=w.parent
        UNION ALL SELECT w.parent,a.kind,a.owner_key,a.name
        FROM temp.wanted_parent w CROSS JOIN main.aliases a
        WHERE a.kind=2 AND a.owner_key>=w.parent*100000 AND a.owner_key<(w.parent+1)*100000"""
    from collections import defaultdict
    portfolio=defaultdict(set);legal=defaultdict(set);sites=defaultdict(set)
    cursor=con.execute(sql)
    while True:
        batch=cursor.fetchmany(8192)
        if not batch:break
        engine.guard()
        for parent,kind,key,name in batch:
            normalized=structural.normalize(data.safe(name))
            if not normalized:continue
            portfolio[parent].add(normalized)
            (legal if kind==1 else sites)[key].add(normalized)
    return {p:tuple(sorted(v)) for p,v in portfolio.items()},{p:tuple(v) for p,v in legal.items()},{s:tuple(v) for s,v in sites.items()}


def make_features(own_keys,own_features,geo_rows,geometry,edges,query_name):
    """Every site of every evidence parent, one direct link hop, no leaf budget."""
    own_keys=np.asarray(own_keys,dtype=np.uint64)
    if len(np.unique(own_keys))!=len(own_keys) or own_features.shape!=(len(own_keys),5):
        raise ValueError('Full unique aligned own scores')
    geo_keys=np.array([r[0] for r in geo_rows],dtype=np.uint64)
    parents=np.union1d(own_keys//100000,geo_keys//100000)
    sources=expand_families(edges.keys,parents)
    edge=edges.select(sources)
    final_parents=np.union1d(parents,edges.targets[edge]//100000)
    keys=expand_families(edges.keys,final_parents)
    x=np.zeros((len(keys),len(PROOF_FIELDS)),dtype=np.float32)
    own=np.searchsorted(keys,own_keys);x[own,:5]=np.clip(own_features,0,1)
    kp=keys//100000;groups,group=np.unique(kp,return_inverse=True)
    parent_identity=np.zeros(len(groups),dtype=np.float32)
    if len(own):np.maximum.at(parent_identity,group[own],x[own,1])
    # Company identity is legitimate OWNERSHIP CONTEXT on every sibling, never
    # the SITE cosine or address of another branch.
    x[:,1]=parent_identity[group]
    coords=edges.coordinates(keys);valid=np.isfinite(coords).all(axis=1)
    if geometry['spatial_crs_compatible']:
        reliability=.5 if geometry['ban_declared_weak'] else 1.
        distance=np.hypot(coords[:,0]-geometry['ban_x_l93'],coords[:,1]-geometry['ban_y_l93'])
        x[valid,5]=reliability/(1+distance[valid]/(300+max(0,geometry['ban_uncertainty_m'])))
    name=structural.normalize(data.safe(query_name))
    parent_name=np.array([structural.similarity(name,edges.portfolios.get(int(p),())) for p in groups],dtype=np.float32)
    # Fill only official own endpoint aliases plus LEGAL owner aliases; sibling
    # aliases are EXCLUSIVELY in the separate parent_portfolio columns.
    for i,k in enumerate(keys):
        names=edges.own_names.get(int(k),())+edges.legal_names.get(int(k)//100000,())
        x[i,2]=max(x[i,2],structural.similarity(name,names))
    x[:,6]=x[:,0]*x[:,5];x[:,7]=x[:,2]*x[:,5]
    x[:,8]=x[:,1]*x[:,3];x[:,9]=x[:,2]*x[:,3]
    if len(geo_keys):x[np.searchsorted(keys,geo_keys),10]=[r[1] for r in geo_rows]
    parent_geo=np.zeros(len(groups),dtype=np.float32);np.maximum.at(parent_geo,group,x[:,5])
    x[:,16]=parent_geo[group];x[:,17]=x[:,1]*x[:,16]
    x[:,18]=parent_name[group];x[:,19]=x[:,18]*x[:,16]
    if len(edge):
        fr=np.searchsorted(keys,edges.sources[edge]);to=np.searchsorted(keys,edges.targets[edge])
        geo=x[fr,10];dense=.75*x[fr,0]+.25*x[fr,1]
        for col,values in ((11,geo),(12,geo*edges.close[edge]),(13,geo*edges.recency[edge]),
                           (14,dense),(15,dense*edges.close[edge]),(20,x[fr,5]),(21,x[fr,0])):
            np.maximum.at(x[:,col],to,values)
    if not np.isfinite(x).all() or (x<0).any() or (x>1.000001).any():
        raise ValueError('Finite bounded typed own/parent/succession proofs')
    return keys,x


def prepare(out):
    launch=authority(out);cache=io.read(verify(launch['feature_launch']));panel=io.read(verify(launch['panel']));surface(panel)
    geo=io.read(verify(launch['geo_sources']));geometries={r['query_id']:r['geometry'] for r in io.read(verify(launch['geometry']))}
    links=compile_family_links(out,launch,cache,geo)
    edges=EdgeViews(launch,links);refs={};commune_sites=0;total_rows=0;started=time.monotonic()
    import sqlite3
    names_con=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
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
            possible={int(k)//100000 for k in own_keys}
            possible.update(int(k)//100000 for q in qs for k,s in geo[q['query_id']])
            source_family=expand_families(edges.keys,possible);possible_edges=edges.select(source_family)
            possible.update(map(int,edges.targets[possible_edges]//100000))
            edges.portfolios,edges.legal_names,edges.own_names=load_name_roles(names_con,sorted(possible))
            del source_family,possible_edges,possible
            try:
                for i,q in enumerate(qs):
                    guard(launch['deadline_unix']);path=feature_path(out,q['query_id']);saved=engine.sealed(path)
                    receipt=path.with_suffix('.binding.json')
                    expected=dict(query=q,recipe=launch['recipe'],feature_launch=launch['feature_launch'],
                        geo_sources=launch['geo_sources'],links=links,raw_successions=launch['raw_successions'],adapter=launch['adapter'])
                    if saved:
                        if io.read(engine.sealed(receipt))!=expected:raise ValueError('Same scientific binding for resumed features')
                        with np.load(saved,allow_pickle=False) as values:n=len(values['keys'])
                    else:
                        own=np.column_stack([arrays[f][i] for f in arrays]) if len(own_keys) else np.zeros((0,5),dtype=np.float32)
                        keys,x=make_features(own_keys,own,geo[q['query_id']],geometries[q['query_id']],edges,q['crm_name']);n=len(keys)
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
            labels_used=False,no_candidate_budget=True,no_gt_directed_edges=True,
            family_sources=record(out/'family_sources.json'),raw_successions=launch['raw_successions']))
    finally:edges.close_files();names_con.close()


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

