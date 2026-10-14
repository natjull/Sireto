"""One query-only OSM preflight: whole published commune suffix, never GT.

Same frozen POIs/aliases/STOP/radii. Full old matcher OR suffix-decomposed
matcher, all526, no score/candidate/SIRET/label read or native retrieval.
"""
import argparse
from collections import Counter,defaultdict
import csv
import math
import os
from pathlib import Path
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','ARROW_NUM_THREADS','RAYON_NUM_THREADS'):os.environ[k]='1'
import psutil
from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v128_named_geo_feasibility import STOP
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUT=io.HOME/'v161/osm_commune_suffix_preflight_20261014_night'
SOURCE=io.HOME/'v129/osm_local_20261013_night_e'
COG=io.HOME/'v98/official_initial_alignment_20261013_b/v_commune_2026.csv'
DELEGATES=Path('/Volumes/CATNAT_DATA/oubusiness/raw/depp_primary_school_enrolment/2026-07-31_primary-v2/geo_api_communes_deleguees_2026-07-31.json')
DEADLINE=1791954368
RECIPE=dict(hypothesis='WHOLE_EXACT_PUBLISHED_ROUTED_COMMUNE_SUFFIX_IS_ADDRESS_QUALIFIER_NOT_NEW_BUSINESS_ALIAS',
    routing='CRM_INSEE_CURRENT_COM_ARM_ELSE_EXISTING_PUBLISHED_DELEGATED_PARENT_EXACT_NO_CP_OR_FUZZY',
    decomposition='ENTIRE_NORMALIZED_OFFICIAL_LIBELLE_TOKEN_SEQUENCE_AT_END_OF_NORMALIZED_CRM_NAME',
    no_partial_common_token_removal=True,no_city_prefix_or_neighbourhood_rule=True,no_manual_city_list=True,
    no_postcode_removal=True,no_synthetic_initials_or_crm_alias=True,raw_query_unchanged=True,
    stop='EXACT_FROZEN_V128_ARTICLE_STOP_AFTER_SUFFIX_DECOMPOSITION',
    source_name_variants='EXACT_V132_SEPARATE_PUBLISHED_NAMES_SPLIT_SEMICOLON_OR_PIPE_NOT_TOKEN_UNION',
    matcher='OLD_FULL_MATCH_OR_REDUCED_MATCH_NONEMPTY_ALL_QUERY_TOKENS_ONE_VARIANT',
    no_operator_string_matching=True,operator_presence_only=True,no_raw_poi_or_person_or_address_export=True,
    all526=True,no_labels_or_ranks_or_scores=True,no_native_source_pbf_scan_network_fit_or_index_change=True,
    ambiguity_diagnostics='DISTINCT_MATCHED_OBJECTS_SINGLE_INFORMATIVE_REMAINDER_AND_OBJECT_CATEGORY_COUNTS_ONLY',
    raw_query_preservation='ORIGINAL_SEALED_PANEL_REFERENCE_AND_QUERY_ID_NOT_DUPLICATE_PERSONAL_NAME_FIELDS',
    max_seconds=600,max_rss_bytes=1024**3,threads=1,disk_reserve_bytes=40*1024**3,free25=True,
    pressure_levels=[0,1,2],deadline_unix=DEADLINE)


def guard(start=None):
    free,level=io.pressure();rss=psutil.Process().memory_info().rss;disk=psutil.disk_usage(str(io.HOME)).free
    if time.time()>=DEADLINE or (start is not None and time.monotonic()-start>600) or rss>1024**3 or free<25 or level not in (0,1,2) or disk<40*1024**3:
        raise RuntimeError(f'Preflight yields rss={rss} free={free} level={level} disk={disk}')


def initialize():
    guard()
    if OUT.exists():raise ValueError('Fresh immutable preflight namespace')
    g=io.read(SOURCE/'generation.json');p=io.read(SOURCE/'projection_generation.json')
    for name in ('scan_owned_tool_exit.json','projection_owned_tool_exit.json'):
        ex=io.read(SOURCE/name)
        if ex['exit_code'] or not ex['own_tool_exit_confirmed']:raise ValueError('Owned frozen whole source exits0')
    if g['status']!='COMPLETE_ALL526_LOCAL_OSM_SOURCE' or p['status']!='COMPLETE_LOSSLESS_RETRIEVAL_PROJECTION_ALL526' or g['all_queries']!=526:
        raise ValueError('All526 source only')
    launch=io.read(verify(g['launch']));panel=io.read(verify(launch['recipe']['query_only']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Same safe query inputs')
    OUT.mkdir(parents=True);io.write(OUT/'recipe.json',RECIPE)
    io.write(OUT/'launch.json',dict(implementation=record(Path(__file__)),recipe=record(OUT/'recipe.json'),
        source_generation=record(SOURCE/'generation.json'),projection_generation=record(SOURCE/'projection_generation.json'),
        source_exit=record(SOURCE/'scan_owned_tool_exit.json'),projection_exit=record(SOURCE/'projection_owned_tool_exit.json'),
        context=p['context'],query_panel=launch['recipe']['query_only'],cog=record(COG),delegates=record(DELEGATES),
        normalizer=record(io.ROOT/'scripts/embeddinggemma2_v17_structural_retrieval.py'),
        frozen_stop=record(io.ROOT/'scripts/embeddinggemma2_v128_named_geo_feasibility.py'),labels_read=False))


def preflight():
    import ijson
    import re
    start=time.monotonic();guard(start);l=io.read(OUT/'launch.json')
    for k in ('implementation','recipe','source_generation','projection_generation','source_exit','projection_exit','cog','delegates','normalizer','frozen_stop'):verify(l[k])
    context=verify(l['context']);panel=io.read(verify(l['query_panel']));queries={q['query_id']:q for q in panel}
    cities=defaultdict(set)
    with verify(l['cog']).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not {'COM','TYPECOM','LIBELLE'}<=set(reader.fieldnames):raise ValueError('Official COG schema')
        for r in reader:
            if r['TYPECOM'] in ('COM','ARM'):cities[r['COM']].add(normalize(r['LIBELLE']))
    delegate_rows=io.read(verify(l['delegates']));delegates={r['code']:r['codeParent'] for r in delegate_rows if r.get('codeParent')}
    rows=[];seen=set();counts=Counter();categories=Counter();matches_hist=Counter();changed_hist=Counter()
    with context.open('rb') as stream:
        for qid,c in ijson.kvitems(stream,'',use_float=True):
            guard(start)
            if qid not in queries or qid in seen:raise ValueError('Unique complete query source')
            seen.add(qid);q=queries[qid];code=q['crm_insee'];route='CURRENT_EXACT'
            if code not in cities and delegates.get(code) in cities:code=delegates[code];route='EXISTING_PUBLISHED_DELEGATED_PARENT'
            city=next(iter(cities[code])) if len(cities.get(code,()))==1 else ''
            nt=normalize(q['crm_name']).split();ct=city.split();old=set(nt)-STOP
            suffix=bool(ct and len(nt)>=len(ct) and nt[-len(ct):]==ct)
            remaining=nt[:-len(ct)] if suffix else nt;reduced=set(remaining)-STOP
            changed=bool(suffix and reduced and reduced!=old)
            old_ids=set();reduced_ids=set();objects={};operator_ids=set();valid_point_ids=set();match_category={}
            for obj in c['pois']:
                key=(obj['osm_type'],int(obj['osm_id']));objects[key]=obj
                variants=[part for n in obj['names'] for part in re.split(r'[;|]',n) if part.strip()]
                tokens=[set(normalize(n).split())-STOP for n in variants]
                if old and any(old<=t for t in tokens):old_ids.add(key)
                if changed and any(reduced<=t for t in tokens):reduced_ids.add(key)
                if obj.get('tags',{}).get('operator'):operator_ids.add(key)
                if any(len(p)==2 and all(math.isfinite(float(x)) for x in p) for p in obj.get('points_l93',[])):valid_point_ids.add(key)
                match_category[key]=[k+':'+str(v) for k,v in obj.get('tags',{}).items() if k in ('shop','amenity','office','tourism','healthcare','leisure','club','education')]
            union=old_ids|reduced_ids;new=union-old_ids
            if not old_ids<=union:raise ValueError('Old full matcher must be retained exactly')
            for key in new:categories.update(match_category[key])
            counts.update(dict(all_queries=1,official_city_route_available=bool(city),whole_city_suffix_present=suffix,
                decomposition_changed=changed,suffix_but_no_remaining_informative_name=suffix and not reduced,
                changed_single_informative_token=changed and len(reduced)==1,source_geometry_available=bool(c['spatial_crs_compatible']),
                changed_without_source_geometry=changed and not c['spatial_crs_compatible'],changed_with_empty_poi_source=changed and not c['pois'],
                queries_old_matches=bool(old_ids),queries_union_matches=bool(union),queries_new_objects=bool(new),queries_gained_first_match=bool(new and not old_ids),
                changed_queries_multiple_union_objects=changed and len(union)>1,queries_new_operator_present=bool(new&operator_ids),
                queries_new_valid_points=bool(new&valid_point_ids),old_object_appearances=len(old_ids),union_object_appearances=len(union),new_object_appearances=len(new),
                new_operator_object_appearances=len(new&operator_ids),new_valid_point_object_appearances=len(new&valid_point_ids)))
            matches_hist[len(union)]+=1
            if changed:changed_hist[len(union)]+=1
            rows.append(dict(query_id=qid,fold=q['fold'],raw_query_reference=l['query_panel'],crm_insee_reference=q['crm_insee'],
                official_routed_insee=code if city else None,route=route if city else 'NO_UNAMBIGUOUS_PUBLISHED_CITY',
                published_city_name=city or None,removed_commune_tokens=ct if suffix else [],raw_informative_token_count=len(old),remaining_informative_token_count=len(reduced),
                full_suffix_present=suffix,reduced_match_enabled=changed,remaining_empty_disables_new_match=bool(suffix and not reduced),
                single_token_remainder=bool(changed and len(reduced)==1),old_matches=len(old_ids),reduced_matches=len(reduced_ids),union_matches=len(union),new_matches=len(new),
                new_operator_present_objects=len(new&operator_ids),new_valid_geometry_objects=len(new&valid_point_ids),spatial_crs_compatible=bool(c['spatial_crs_compatible']),
                published_poi_objects=len(objects),old_matches_retained=True,source_only_not_recall_or_identity=True))
    if seen!=set(queries):raise ValueError('No denominator or empty source query removal')
    guard(start);verify(l['context']);io.write(OUT/'all526_preflight.json',rows)
    io.write(OUT/'report.json',dict(status='COMPLETE_ALL526_QUERY_ONLY_OSM_WHOLE_COMMUNE_SUFFIX_PREFLIGHT_NO_GT',
        implementation=record(Path(__file__)),launch=record(OUT/'launch.json'),recipe=record(OUT/'recipe.json'),
        all526=record(OUT/'all526_preflight.json'),counts=dict(counts),union_match_object_histogram={str(k):v for k,v in sorted(matches_hist.items())},
        changed_union_match_object_histogram={str(k):v for k,v in sorted(changed_hist.items())},new_matched_object_categories=dict(categories),
        same_source_same_geographic_scope_same_name_variants_same_stop=True,old_match_object_coverage_monotonic_not_recall_guarantee=True,
        no_gt_rank_siret_model_or_native_queries=True,no_network_pbf_scan_or_person_operator_address_export=True,
        source_current_community_not_asof_cutoff_certified=True,new_operator_flag_not_official_identity=True,
        publication_city_not_house_or_location_estimate=True,elapsed_seconds=time.monotonic()-start))
    print(dict(counts=counts,changed_union_multiplicity=dict(changed_hist),elapsed_seconds=time.monotonic()-start),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','preflight']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as e:
        if OUT.exists():io.write(OUT/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
