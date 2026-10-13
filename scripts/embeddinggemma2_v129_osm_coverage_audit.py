"""Query-only coverage audit of complete OSM, no GT or retrieval outcomes.

Exact token/number rules are those predeclared by V132. This report cannot
alter the source radius, names, feature recipe or a human denominator.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v128_named_geo_feasibility import STOP
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=io.HOME/'v129/osm_local_20261013_night_e'


def number(value):
    match=re.fullmatch(r'\s*(\d+)\s*(BIS|TER|QUATER|[A-Z])?\s*',normalize(value))
    return (int(match[1]),match[2] or '') if match and int(match[1])>0 else None


def main():
    if (ROOT/'coverage_audit.json').exists():raise ValueError('Immutable audit already exists')
    generation=io.read(ROOT/'generation.json');exit_record=io.read(ROOT/'scan_owned_tool_exit.json')
    if generation['status']!='COMPLETE_ALL526_LOCAL_OSM_SOURCE' or exit_record['exit_code']!=0:
        raise ValueError('Complete source and observed exit0 required')
    projection=io.read(ROOT/'projection_generation.json')
    if io.read(ROOT/'projection_owned_tool_exit.json')['exit_code']!=0:raise ValueError('Projection exit0 first')
    context=verify(projection['context']);launch=io.read(verify(generation['launch']))
    panel=io.read(verify(launch['recipe']['query_only']));queries={q['query_id']:q for q in panel}
    shapes=io.read(verify(generation['files']['shapes']))
    sys.path.insert(0,str(ROOT.parent/'osm_local_20261013_night/deps'));import ijson
    counters=Counter();house_classes=Counter();name_token_counts=Counter();rows=[]
    unique_pois=set();unique_addresses=set();seen=set();max_pois=0;max_addresses=0
    with context.open('rb') as stream:
        for qid,c in ijson.kvitems(stream,'',use_float=True):
            if qid not in queries or qid in seen:raise ValueError('Exact unique source scope')
            seen.add(qid);q=queries[qid];tokens=set(normalize(q['crm_name']).split())-STOP
            name_token_counts[len(tokens)]+=1
            nmatch=0;amatch=0;qm=number(q.get('crm_number','')+' '+q.get('crm_suffix',''))
            street=normalize(q['crm_street'],street=True)
            for p in c['pois']:
                unique_pois.add((p['osm_type'],p['osm_id']))
                variants=[part for name in p['names'] for part in re.split(r'[;|]',name) if part.strip()]
                counters['published_name_variants_context_appearances']+=len(variants)
                counters['informative_name_variants_context_appearances']+=sum(bool(set(normalize(n).split())-STOP) for n in variants)
                if tokens and any(tokens <= set(normalize(n).split())-STOP for n in variants):nmatch+=1
            for a in c['addresses']:
                unique_addresses.add((a['osm_type'],a['osm_id']));tags=a['tags'];raw=tags.get('addr:housenumber','')
                parsed=number(raw)
                if parsed:cls='SINGLE_NUMBER_SUFFIX'
                elif not str(raw).strip():cls='EMPTY'
                elif re.search(r'[;,|/–—-]|\bET\b',str(raw),flags=re.I):cls='LIST_RANGE_OR_COMPOSITE'
                else:cls='OTHER_NONEXACT_OR_ZERO'
                house_classes[cls]+=1
                counters['addresses_without_city_context_appearances']+=not bool(tags.get('addr:city'))
                counters['addresses_without_postcode_context_appearances']+=not bool(tags.get('addr:postcode'))
                if qm and qm==parsed and street==normalize(tags.get('addr:street',''),street=True):amatch+=1
            np=len(c['pois']);na=len(c['addresses']);max_pois=max(max_pois,np);max_addresses=max(max_addresses,na)
            counters['contexts']+=1;counters['located_contexts']+=c['spatial_crs_compatible']
            counters['contexts_without_any_poi_or_address']+=not(np or na)
            counters['contexts_with_exact_query_name_variant_match']+=nmatch>0
            counters['contexts_with_exact_single_house_street_match']+=amatch>0
            counters['contexts_with_multiple_exact_house_objects']+=amatch>1
            counters['contexts_with_street_lines']+=bool(shapes[qid]['lines_l93'])
            counters['poi_context_appearances']+=np;counters['address_context_appearances']+=na
            rows.append(dict(query_id=qid,located=c['spatial_crs_compatible'],pois=np,addresses=na,
                informative_query_tokens=len(tokens),exact_name_variant_matched_objects=nmatch,
                exact_single_house_street_objects=amatch,matched_road_lines=len(shapes[qid]['lines_l93'])))
    if seen!=set(queries) or len(seen)!=526:raise ValueError('Every526 query preserved')
    if len(unique_pois)!=generation['counts']['pois'] or len(unique_addresses)!=generation['counts']['addresses']:
        raise ValueError('Lossless projection unique-object count binding')
    io.write(ROOT/'coverage_query_only.json',rows)
    io.write(ROOT/'coverage_audit.json',dict(status='COMPLETE_ALL526_QUERY_ONLY_COVERAGE_AUDIT',
        source_generation=record(ROOT/'generation.json'),source_exit=record(ROOT/'scan_owned_tool_exit.json'),
        implementation=record(Path(__file__)),projection=record(ROOT/'projection_generation.json'),
        query_rows=record(ROOT/'coverage_query_only.json'),counters=dict(counters),
        house_number_classes_context_appearances=dict(house_classes),informative_query_token_count_distribution=dict(name_token_counts),
        unique_named_pois=len(unique_pois),unique_addresses=len(unique_addresses),
        max_neighbourhood_named_pois=max_pois,max_neighbourhood_addresses=max_addresses,
        labels_read=False,no_recall_or_identity_claim=True,no_new_scientific_filter=True,completed_unix=time.time()))
    print('COMPLETE_QUERY_ONLY_COVERAGE',json.dumps(dict(counters)),dict(house_classes),flush=True)


if __name__=='__main__':main()
