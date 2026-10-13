"""Query-only typed supplier/client-site context, frozen before outcomes.

Prepared published brand references are NOT legal identity edges. They can
support a site only through that site's own spatial evidence or a separate
dated direct economic-continuity observation. This compiler does not search
the index, read labels, fit anything, or emit a fixed candidate pool.
"""
import argparse
import gc
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
from scripts.embeddinggemma2_v131_mechanism_audit import HOME, BASE, read, write, record, verify

OUTPUT=HOME/'v133/supplier_site_context_20261013_night'
RECIPE=dict(hypothesis='PUBLISHED_SUPPLIER_CLIENT_BRAND_CONTEXT_WITH_OWN_SITE_SUPPORT',
    source_confidence=.5,source_confidence_reused_v75_without_tuning=True,
    provider_match='ALL_PUBLISHED_PROVIDER_BRAND_TOKENS_IN_QUERY',
    client_match='EXISTING_V81_ALL_NATIONAL_CURRENT_PUBLIC_CORPORATE_OWNER_NAMES',
    brand_match_is_weak_context_not_ownership_or_alias=True,
    own_context='HALF_TIMES_MAX_ORIGINAL_OWN_GEO_AND_FILLED_OWN_SITE_GEO',
    temporal_context='HALF_TIMES_SOURCE_OWN_GEO_TIMES_DIRECT_LINK_DISTANCE_KERNEL',
    temporal_link='EXPLICIT_ECONOMIC_CONTINUITY_TRUE_ON_OR_BEFORE_2026_08_01_ONE_HOP',
    temporal_context_not_own_address_or_identity=True,
    distance_kernel='ONE_DIV_ONE_PLUS_DISTANCE_OVER300_METRES_PREEXISTING_SCALE',
    duplicate_same_source='MAX_NOT_SUM_BETWEEN_BRANDS_OR_LINKS',
    score_coefficient='UNCHANGED_HELDOUT_NATIONAL_CORPORATE_ACTOR_COEFFICIENT_W6',
    integration='OWN_AND_TEMPORAL_SEPARATE_ABLATIONS_BEFORE_ONE_NATIVE100_COLLECTOR',
    full_numeric_universe_unchanged=True,no_fixed_prepool=True,no_new_training=True,
    no_source_alias_inserted=True,no_positive_injected=True,
    all526_query_only=True,production_promotion_allowed=False,
    eligibility='DIAGNOSTIC_ONLY_INHERITED_SOURCE_POLICY_INELIGIBLE')


def normalize(value):
    value=re.sub(r'(?<!\d)(?:\d{14}|\d{9})(?!\d)',' ',str(value or ''))
    text=''.join(c for c in unicodedata.normalize('NFKD',value.upper()) if not unicodedata.combining(c))
    words=re.findall(r'[A-Z0-9]+',text.replace('&',' ET '))
    return {w for w in words if w not in {'SA','SAS','SASU','SARL','EURL','SCI','STE','ETS','ND'}}


def compile_source(out,deadline):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    if out.exists():raise ValueError('Fresh source namespace, no active config changes')
    if not time.time()<deadline<=time.time()+8*3600:raise ValueError('Absolute authorized night deadline')
    gen,ex=read(BASE/'generation.json'),read(BASE/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Original complete owned baseline')
    launch=read(BASE/'launch.json');fg=read(BASE/'features_generation.json')
    panel=read(verify(launch['panel']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Whole safe development query panel')
    indirect=read(verify(fg['indirect_names']))
    published=read(verify(launch['published_generation']))
    review=read(verify(launch['visual_review']))
    if {r['source_sha256'] for r in review['brands']}!={r['source']['sha256'] for r in published['images']}:
        raise ValueError('Exact complete reviewed client section')
    for item in published['images']:verify(item['source']);verify(item['preview'])
    provider=normalize(indirect['provider_published_brand'])
    if not provider or provider!=normalize(review['provider_published_brand']):raise ValueError('Published provider surface only')
    matching=[q for q in panel if provider<=normalize(q['crm_name'])]
    owners={int(s) for b in indirect['client_brands'] for s in b['owner_refs']}
    if not owners:raise ValueError('Existing whole-source current-name matches required')
    out.mkdir(parents=True);(out/'spill').mkdir();(out/'shards').mkdir()
    write(out/'recipe.json',RECIPE)
    write(out/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),
        panel=launch['panel'],source_brand_matches=fg['indirect_names'],published_source=launch['published_generation'],
        published_review=launch['visual_review'],original_features=record(BASE/'features_generation.json'),
        original_models=record(BASE/'calibration/generation.json'),deadline_unix=deadline,
        source_query_selection_only=True,labels_read=False,created_unix=time.time()))
    # Query-only family context compiler already scanned whole official links.
    family=read(verify(fg['family_sources']))
    source_links=verify(family['compiled_links'])
    table=pq.read_table(source_links,columns=['siretEtablissementPredecesseur','siretEtablissementSuccesseur',
        'dateLienSuccession','continuiteEconomique'],use_threads=False)
    predecessor=np.asarray(table['siretEtablissementPredecesseur'].to_pylist(),dtype=np.uint64)
    successor=np.asarray(table['siretEtablissementSuccesseur'].to_pylist(),dtype=np.uint64)
    dates=np.asarray(table['dateLienSuccession'].to_numpy(),dtype='datetime64[D]')
    continuity=np.asarray(table['continuiteEconomique'].to_pylist(),dtype=bool)
    if not continuity.all() or np.isnat(dates).any() or (dates>np.datetime64('2026-08-01')).any():
        raise ValueError('Only original dated economic continuity source facts')
    left=np.concatenate((predecessor,successor));right=np.concatenate((successor,predecessor))
    selected=np.isin(left//100000,np.fromiter(owners,dtype=np.uint64))
    left,right=left[selected],right[selected]
    del table,predecessor,successor,dates,continuity,selected;gc.collect()
    # Publicness is checked before geometry is assigned; no protected address
    # reconstruction. IDs are selectors/output references, never score inputs.
    wanted_owners=owners|set(map(int,right//100000))
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(out/'spill')])
    con.register('owner_scope',pa.table({'siren':[str(s).zfill(9) for s in sorted(wanted_owners)]}))
    unit=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet')
    public=np.asarray(con.execute("""SELECT CAST(e.siret AS UBIGINT) site_key
       FROM read_parquet(?) e JOIN owner_scope w USING(siren)
       JOIN read_parquet(?) u USING(siren)
       WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'
       ORDER BY site_key""",[launch['stock_source']['path'],str(unit)]).fetchnumpy()['site_key'],dtype=np.uint64)
    con.close()
    valid=np.isin(left,public)&np.isin(right,public);left,right=left[valid],right[valid]
    # Coordinates from fixed national atlas, not OSM or hand-moved points.
    registry=np.load(launch['keys']['path'],mmap_mode='r',allow_pickle=False)
    xy=np.load(launch['xy']['path'],mmap_mode='r',allow_pickle=False)
    def coords(refs):
        at=np.searchsorted(registry,refs);valid=at<len(registry)
        valid[valid]&=registry[at[valid]]==refs[valid]
        values=np.full((len(refs),2),np.nan);values[valid]=xy[at[valid]]
        okay=np.isfinite(values).all(axis=1)&(values[:,0]>=0)&(values[:,0]<=1300000)&(values[:,1]>=6000000)&(values[:,1]<=7300000)
        values[~okay]=np.nan
        return values
    a,b=coords(left),coords(right)
    okay=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    close=np.zeros(len(left));close[okay]=1/(1+np.hypot(*(a[okay]-b[okay]).T)/300.)
    registry._mmap.close();xy._mmap.close();del a,b,okay;gc.collect()
    extras={};stats=[]
    import hashlib
    for q in panel:
        if time.time()>=deadline:raise TimeoutError('Night ended; preserve source checkpoints')
        qid=q['query_id']
        if q not in matching:
            extras[qid]=dict(own=None,temporal=None,unchanged_query_no_published_provider_match=True)
            continue
        with np.load(verify(fg['features'][qid]),allow_pickle=False) as f:
            keys=f['keys'];x=f['features']
            if x.shape!=(len(keys),51):raise ValueError('Unchanged original full proof universe')
            permitted=np.isin(keys,public)
            own_brand=np.isin(keys//100000,np.fromiter(owners,dtype=np.uint64))&permitted
            own_geo=np.maximum(x[:,5],x[:,42]).astype(np.float64)
            own=.5*own_geo*own_brand
            lo=np.searchsorted(keys,left);hi=np.searchsorted(keys,right)
            linked=(lo<len(keys))&(hi<len(keys))
            rows=np.flatnonzero(linked)
            rows=rows[(keys[lo[rows]]==left[rows])&(keys[hi[rows]]==right[rows])]
            temporal=np.zeros(len(keys),dtype=np.float64)
            np.maximum.at(temporal,hi[rows],.5*own_geo[lo[rows]]*close[rows])
            temporal*=permitted
            if not np.isfinite(own).all() or not np.isfinite(temporal).all() or max(own.max(initial=0),temporal.max(initial=0))>.500001:
                raise ValueError('Bounded frozen weak source context')
            refs={}
            for channel,values in [('own',own),('temporal',temporal)]:
                chosen=np.flatnonzero(values>0)
                dest=out/'shards'/(hashlib.sha256((qid+channel).encode()).hexdigest()+'.npz')
                np.savez_compressed(dest,keys=keys[chosen],confidence=values[chosen])
                refs[channel]=record(dest)
            extras[qid]=dict(refs,published_relation_not_alias=True,public_sites_only=True)
            stats.append(dict(query_id=qid,query=q,full_numeric_references=len(keys),own_supported_leaves=int(np.count_nonzero(own)),
                temporal_supported_leaves=int(np.count_nonzero(temporal)),temporal_public_direct_observations=len(rows)))
        write(out/'state.json',dict(status='QUERY_ONLY_PUBLISHED_CONTEXT_SITE_COMPILER',queries=len(extras),total=526,
            modified_query_inputs=len(stats),updated_unix=time.time()))
        gc.collect()
    if len(extras)!=526:raise ValueError('All526 represented')
    write(out/'query_extras.json',extras);write(out/'source_coverage.json',stats)
    write(out/'generation.json',dict(status='COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT',
        implementation=record(Path(__file__)),recipe=record(out/'recipe.json'),query_panel=launch['panel'],
        extras=record(out/'query_extras.json'),coverage=record(out/'source_coverage.json'),
        source_brands=fg['indirect_names'],published_review=launch['visual_review'],source_links=family['compiled_links'],
        source_xy=launch['xy'],source_keys=launch['keys'],source_stock=launch['stock_source'],
        source_legal_units=record(unit),queries=526,provider_matched_queries=len(matching),
        all_other_queries_unchanged=True,public_reference_scope=len(public),
        no_index_search=True,no_targets_read=True,no_candidates_or_recall=True,
        no_fit=True,production_promotion_allowed=False,observed_unix=time.time()))
    print('COMPLETE_QUERY_ONLY_ALL526_SUPPLIER_SITE_CONTEXT',len(matching),len(public),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--deadline-unix',required=True,type=float);args=p.parse_args()
    compile_source(args.output,args.deadline_unix)
