"""Read-only, query-only GIS audit of already sealed OSM points.

This is not another geocoder or retrieval run. It measures whether global
objects selected by OTHER queries' BAN buffers can inform the sixty queries
without a compatible BAN point. Administrative routing uses published commune
polygons, never a centroid as house/site evidence. No labels are opened.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import resource
import time

import duckdb

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from scripts.embeddinggemma2_v128_named_geo_feasibility import STOP
from scripts.embeddinggemma2_v129_osm_coverage_audit import number
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME / 'v129/osm_local_20261013_night_e'
ROOT = io.HOME / 'v129/independent_anchor_audit_20261014_a'
CONTOURS = Path('/Volumes/CATNAT_DATA/oubusiness/silver/commune_contours_100m_2026-03-09.parquet')
CENTRES = Path('/Volumes/CATNAT_DATA/oubusiness/silver/communes_2026-07-23.parquet')
DELEGATES = Path('/Volumes/CATNAT_DATA/oubusiness/raw/depp_primary_school_enrolment/2026-07-31_primary-v2/geo_api_communes_deleguees_2026-07-31.json')


def main():
    started = time.time()
    if (ROOT / 'report.json').exists():
        raise ValueError('Immutable audit report already exists')
    ROOT.mkdir(parents=True, exist_ok=True)
    generation = io.read(SOURCE / 'generation.json')
    source_exit = io.read(SOURCE / 'scan_owned_tool_exit.json')
    if generation['status'] != 'COMPLETE_ALL526_LOCAL_OSM_SOURCE' or source_exit['exit_code'] != 0 or not source_exit['own_tool_exit_confirmed']:
        raise ValueError('Complete and observed source exit0 required')
    launch = io.read(verify(generation['launch']))
    panel = io.read(verify(launch['recipe']['query_only']))
    located = {q['query_id']: q['spatial_crs_compatible'] for q in io.read(SOURCE / 'queries.json')}
    if len(panel) != 526 or set(located) != {q['query_id'] for q in panel}:
        raise ValueError('Full 526 query-only binding required')
    delegates = {row['code']: row['chefLieu'] for row in io.read(DELEGATES) if row.get('chefLieu')}
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.execute("SET memory_limit='512MB'")
    con.execute("SET temp_directory='" + str(ROOT / 'duck_tmp') + "'")
    con.execute('LOAD spatial')  # Already installed locally; no extension download.
    commune_codes = {row[0] for row in con.execute('SELECT code_commune FROM read_parquet(?)', [str(CONTOURS)]).fetchall()}
    def route(code):
        if code in commune_codes:
            return code, 'PUBLISHED_CURRENT_INSEE'
        if delegates.get(code) in commune_codes:
            return delegates[code], 'PUBLISHED_DELEGATED_TO_CURRENT_INSEE'
        return '', 'NO_PUBLISHED_CURRENT_OR_DELEGATED_ROUTE'
    covered_routes = {route(q['crm_insee'])[0] for q in panel if located[q['query_id']]}
    targets = [q for q in panel if not located[q['query_id']]]
    if len(targets) != 60:
        raise ValueError('Original sixty unlocated queries required')
    rows = {}
    by_commune = defaultdict(list)
    for q in targets:
        code, routing = route(q['crm_insee'])
        rows[q['query_id']] = dict(query_id=q['query_id'], crm_insee=q['crm_insee'],
            routed_insee=code, administrative_routing=routing,
            commune_has_other_located_query=bool(code and code in covered_routes),
            global_published_points=dict(pois=0, addresses=0, roads=0),
            matches=dict(name_objects=0, exact_house_street_objects=0, exact_street_road_objects=0),
            matching_published_positions=dict(name=0, house=0, street=0))
        if code:
            by_commune[code].append(dict(qid=q['query_id'], tokens=set(normalize(q['crm_name']).split())-STOP,
                street=normalize(q['crm_street'], street=True),
                house=number(q.get('crm_number', '') + ' ' + q.get('crm_suffix', ''))))
    codes = sorted(by_commune)
    con.execute('CREATE TEMP TABLE communes AS SELECT code_commune, crs, ST_GeomFromWKB(geometry_wkb) geom FROM read_parquet(?) WHERE code_commune IN (SELECT unnest(?))', [str(CONTOURS), codes])
    if con.execute("SELECT count(*) FROM communes WHERE crs <> 'EPSG:2154'").fetchone()[0]:
        raise ValueError('Source coordinates and commune polygons must share EPSG:2154')
    con.execute('CREATE TEMP TABLE bounds AS SELECT *, ST_XMin(geom) xmin, ST_XMax(geom) xmax, ST_YMin(geom) ymin, ST_YMax(geom) ymax FROM communes')
    positions = defaultdict(lambda: defaultdict(set))
    category_counts = defaultdict(Counter)
    counters = Counter()
    for kind in ('pois', 'addresses', 'roads'):
        columns = "{osm_type:'VARCHAR',osm_id:'BIGINT',x:'DOUBLE',y:'DOUBLE',names:'VARCHAR[]',tags:'JSON',name_key:'VARCHAR'}"
        query = """SELECT b.code_commune, o.osm_type, o.osm_id, o.x, o.y, o.names, o.tags, o.name_key
          FROM read_json(?, columns=""" + columns + """, maximum_object_size=16777216) o
          JOIN bounds b ON o.x BETWEEN b.xmin AND b.xmax AND o.y BETWEEN b.ymin AND b.ymax
          WHERE ST_Covers(b.geom, ST_Point(o.x,o.y))"""
        cursor = con.execute(query, [generation['files'][kind]['path']])
        while batch := cursor.fetchmany(1024):
            for code, osm_type, osm_id, x, y, names, tags_json, name_key in batch:
                category_counts[code][kind] += 1
                tags = json.loads(tags_json) if tags_json else {}
                variants = [p for n in (names or []) for p in re.split(r'[;|]', n) if p.strip()]
                tokens = [set(normalize(n).split())-STOP for n in variants] if kind == 'pois' else []
                street = normalize(tags.get('addr:street', ''), street=True) if kind == 'addresses' else ''
                house = number(tags.get('addr:housenumber', '')) if kind == 'addresses' else None
                for q in by_commune[code]:
                    key = q['qid']; r = rows[key]
                    if kind == 'pois' and q['tokens'] and any(q['tokens'] <= t for t in tokens):
                        r['matches']['name_objects'] += 1
                        positions[key]['name'].add((x,y))
                    if kind == 'addresses' and q['house'] and q['house'] == house and q['street'] and q['street'] == street:
                        r['matches']['exact_house_street_objects'] += 1
                        positions[key]['house'].add((x,y))
                    if kind == 'roads' and q['street'] and q['street'] == name_key:
                        r['matches']['exact_street_road_objects'] += 1
                        positions[key]['street'].add((x,y))
            # This audit must remain bounded, not become another extraction.
            if time.time()-started > 300:
                raise TimeoutError('Five minute query-only GIS audit limit')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1024**3:
                raise MemoryError('One GiB audit RSS limit')
        print('SCANNED_EXISTING_GLOBAL_ASSET', kind, flush=True)
    for qid, r in rows.items():
        code = r['routed_insee']
        r['global_published_points'] = {kind: category_counts[code][kind] for kind in ('pois','addresses','roads')}
        r['matching_published_positions'] = {kind:len(positions[qid][kind]) for kind in ('name','house','street')}
        counters['queries'] += 1
        counters['current_insee_route'] += r['administrative_routing'] == 'PUBLISHED_CURRENT_INSEE'
        counters['delegated_insee_route'] += r['administrative_routing'] == 'PUBLISHED_DELEGATED_TO_CURRENT_INSEE'
        counters['unrouted'] += not bool(code)
        counters['commune_has_other_located_query'] += r['commune_has_other_located_query']
        counters['commune_has_any_existing_global_point'] += any(r['global_published_points'].values())
        counters['has_name_match'] += r['matches']['name_objects'] > 0
        counters['has_house_street_match'] += r['matches']['exact_house_street_objects'] > 0
        counters['has_street_road_match'] += r['matches']['exact_street_road_objects'] > 0
        counters['has_any_published_match'] += any(r['matches'].values())
        counters['multiple_name_positions'] += r['matching_published_positions']['name'] > 1
        counters['multiple_house_positions'] += r['matching_published_positions']['house'] > 1
        counters['multiple_street_positions'] += r['matching_published_positions']['street'] > 1
    io.write(ROOT/'queries.json', list(rows.values()))
    report = dict(status='COMPLETE_QUERY_ONLY_EXISTING_GLOBAL_OSM_ADMIN_JOIN',
        source_generation=record(SOURCE/'generation.json'), source_exit=record(SOURCE/'scan_owned_tool_exit.json'),
        implementation=record(Path(__file__)), panel=launch['recipe']['query_only'],
        source_assets={k:generation['files'][k] for k in ('pois','addresses','roads')},
        admin_inputs=dict(contours=record(CONTOURS),centres_available_not_used_as_locations=record(CENTRES),delegates=record(DELEGATES)),
        counters=dict(counters), distinct_routed_target_communes=len(by_commune),
        commune_objects={k:dict(v) for k,v in category_counts.items()}, query_rows=record(ROOT/'queries.json'),
        labels_read=False, recall_measured=False, no_current_recipe_mutation=True,
        point_only_commune_join=True, source_is_ban_seeded_partial_national_index=True,
        limitations=['100m generalised administrative borders, not parcel boundaries',
            'Road membership uses published representative vertex, not complete line intersection',
            'Way POI/address points are means of referenced nodes, not certified entrances',
            'Multiple OSM objects/positions may refer to one physical place',
            'Absence is not proof of missing national OSM evidence; original extraction was BAN-seeded',
            'OSM provenance can include imported BAN; distinct publisher is not statistical independence'],
        elapsed_s=time.time()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        completed_unix=time.time())
    io.write(ROOT/'report.json', report)
    print('COMPLETE',json.dumps(dict(counters)),flush=True)


if __name__ == '__main__':
    main()
