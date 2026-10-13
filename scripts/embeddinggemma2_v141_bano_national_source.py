"""Provenance-preserving national CSV.gz continuation of the BANO source.

The producer's national archive is distinct from the regional lot union.
Every CSV row and the gzip CRC are verified; only explicit OSM source rows
enter the address index. Query context does not read any BAN geometry.
"""
from collections import Counter, defaultdict
import csv
import gzip
from pathlib import Path
import re
import resource
import sqlite3
import time

from scripts import embeddinggemma2_v141_bano_osm_source as base
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93

ROOT=base.io.HOME/'v141/bano_osm_20261013_night_b'
ARCHIVE=base.RAW/'full.csv.gz'
HEADERS=base.RAW/'full.csv.gz.headers'
LISTING=base.io.HOME/'v141/bano_osm_20261013_night_a/listing.html'


def main():
    started=time.time()
    if ROOT.exists():raise ValueError('Fresh immutable source namespace required')
    ROOT.mkdir(parents=True)
    # The original preparation failed before launch/adresses; keep it intact.
    base.ROOT=ROOT
    base.guard()
    headers={k.lower():v.strip() for line in HEADERS.read_text().splitlines() if ':' in line for k,v in [line.split(':',1)]}
    if ARCHIVE.stat().st_size!=int(headers['content-length']):raise ValueError('Observed complete download size required')
    if ARCHIVE.stat().st_size>base.NETWORK_LIMIT:raise ValueError('Network scope invalid')
    listing=LISTING.read_text();lots=[]
    for line in listing.splitlines():
        match=re.search(r'href="(bano-[0-9AB]+\.csv)".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})',line)
        if match:lots.append(dict(name=match[1],last_modified_minute=match[2]))
    if len(lots)!=110 or len({x['name'] for x in lots})!=110:raise ValueError('Frozen publication listing mismatch')
    source_launch=base.io.read(base.SOURCE/'launch.json')
    panel=base.io.read(base.verify(source_launch['recipe']['query_only']))
    if len(panel)!=526 or len({q['query_id'] for q in panel})!=526:raise ValueError('All526 query-only binding')
    launch=dict(status='FROZEN_NATIONAL_CSV_GZIP_OSM_ONLY',implementation=base.record(Path(__file__)),
        shared_preparation_module=base.record(Path(base.__file__)),query_only=source_launch['recipe']['query_only'],
        archive=base.record(ARCHIVE),headers=base.record(HEADERS),http_headers=headers,
        archive_url=base.URL+'full.csv.gz',listing=base.record(LISTING),regional_lots_observed=lots,
        regional_union_identity_not_verified=True,producer_documentation=base.DOC,
        nomenclature_version='BANO v2, producer wiki revision 2876454',retained_source_codes=['OSM'],
        excluded_source_codes=['CAD','C+O','OD','O+O'],no_JSON_unknown_provenance_used=True,
        columns=['id','numero','voie','code_post','ville','source','lat','lon'],
        commune_reference=base.record(base.COMMUNES),delegated_reference=base.record(base.DELEGATES),
        routing='current INSEE; original+current delegated INSEE; else ALL published CP communes',
        match='V132 normalized exact street + V132 single-number/suffix parser',
        street_evidence='ALL source OSM published address points of exact street in routed communes',
        house_evidence='EXACT single number/suffix subset, never centre or street centroid',
        no_BAN_geometry_input=True,osm_historical_BAN_import_possible=True,
        current_community_diagnostic_only=True,closed_sets_or_labels_read=False,
        resource_guards=dict(network_bytes=base.NETWORK_LIMIT,storage_bytes=base.STORAGE_LIMIT,
            rss_bytes=int(1.5*1024**3),free_percent=25,CATNAT_reserve_bytes=40*1024**3,threads=1),
        preparation_deadline_unix=base.DEADLINE,started_unix=started)
    base.io.write(ROOT/'launch.json',launch)
    database=ROOT/'addresses.sqlite';con=sqlite3.connect(database)
    con.execute('PRAGMA cache_size=-65536');con.execute('PRAGMA temp_store=FILE');con.execute('PRAGMA journal_mode=DELETE')
    con.execute('CREATE TABLE addresses(source_row INTEGER PRIMARY KEY,bano_id TEXT NOT NULL,insee TEXT NOT NULL,street TEXT NOT NULL,street_key TEXT NOT NULL,number_raw TEXT NOT NULL,number_integer INTEGER,suffix TEXT,postcode TEXT,city TEXT,lat REAL,lon REAL)')
    counters=Counter();source_counts=Counter();prefix_counts=defaultdict(Counter);batch=[]
    with gzip.open(ARCHIVE,'rt',encoding='utf8',newline='') as stream:
        for row_index,row in enumerate(csv.reader(stream),start=1):
            if len(row)!=8:raise ValueError(f'National CSV8 drift at row{row_index}')
            bid,num,street,postcode,city,source,lat_raw,lon_raw=row
            if source not in ('OSM','OD','O+O','C+O','CAD'):raise ValueError(f'Unknown source {source!r}')
            if not re.match(r'^(?:\d{5}|2[AB]\d{3})',bid):raise ValueError(f'Invalid public commune prefix {bid!r}')
            prefix=bid[:3] if bid.startswith(('97','98')) else bid[:2]
            prefix_counts[prefix][source]+=1;source_counts[source]+=1;counters['all_csv_rows']+=1
            if source=='OSM':
                lat=float(lat_raw);lon=float(lon_raw)
                if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError('Invalid WGS84 point')
                parsed=base.number(num)
                batch.append((row_index,bid,bid[:5],street,base.normalize(street,street=True),num,
                    parsed[0] if parsed else None,parsed[1] if parsed else None,postcode,city,lat,lon))
                counters['retained_OSM_rows']+=1;counters['nonexact_house_numbers_retained']+=not bool(parsed)
                if len(batch)>=8192:con.executemany('INSERT INTO addresses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',batch);batch.clear()
            if row_index%262144==0:
                base.guard();con.commit()
                base.io.write(ROOT/'state.json',dict(status='STREAMING_NATIONAL_CSV',counters=dict(counters),
                    source_counts=dict(source_counts),updated_unix=time.time()))
                print('ROWS',row_index,'OSM',counters['retained_OSM_rows'],flush=True)
    if batch:con.executemany('INSERT INTO addresses VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',batch)
    con.commit();base.guard()
    con.execute('CREATE INDEX by_commune_street ON addresses(insee,street_key,number_integer,suffix)');con.commit()
    base.io.write(ROOT/'schema_preflight.json',dict(status='PASS_ALL_NATIONAL_CSV8_WITH_SOURCE_AND_GZIP_CRC',
        rows=counters['all_csv_rows'],source_counts=dict(source_counts),columns=launch['columns']))
    communes=base.io.read(base.COMMUNES);current={c['code'] for c in communes};postal={}
    for c in communes:
        for cp in c['codesPostaux']:postal.setdefault(cp,set()).add(c['code'])
    delegates={c['code']:c['chefLieu'] for c in base.io.read(base.DELEGATES) if c.get('chefLieu')}
    context={};coverage=Counter()
    for q in panel:
        code=q['crm_insee'];cp=q['crm_postcode'];street_key=base.normalize(q['crm_street'],street=True)
        house=base.number(q.get('crm_number','')+' '+q.get('crm_suffix',''))
        if code in current:codes=[code];method='CURRENT_INSEE'
        elif delegates.get(code) in current:codes=sorted({code,delegates[code]});method='PUBLISHED_DELEGATED_INSEE'
        else:codes=sorted(postal.get(cp,set()));method='ALL_PUBLISHED_POSTCODE_COMMUNES'
        points=[];house_refs=[]
        if codes and street_key:
            sql='SELECT source_row,bano_id,insee,street,number_raw,number_integer,suffix,postcode,city,lat,lon FROM addresses WHERE street_key=? AND insee IN ('+','.join('?' for _ in codes)+')'
            for sr,bid,ci,st,nraw,ni,suffix,pc,city,lat,lon in con.execute(sql,[street_key,*codes]):
                l93_ok=-6<=lon<=10 and 41<=lat<=52
                point=dict(source_archive='full.csv.gz',source_row=sr,bano_id=bid,insee=ci,street=st,number=nraw,
                    source='OSM',postcode=pc,city=city,lat=lat,lon=lon,l93_compatible=l93_ok,
                    point_l93=list(lonlat_to_l93(lon,lat)) if l93_ok else None,
                    precision='PUBLISHED_ADDRESS_POINT_NOT_CERTIFIED_ENTRANCE',not_official_operator_identity=True)
                points.append(point)
                if house and (ni,suffix)==house:house_refs.append(len(points)-1)
        positions={(p['lon'],p['lat']) for p in points};hp={(points[i]['lon'],points[i]['lat']) for i in house_refs}
        context[q['query_id']]=dict(commune_routing=method,commune_codes=codes,exact_street_key=street_key,
            query_single_house_number=list(house) if house else None,published_street_address_cloud=points,
            exact_house_point_indices=house_refs,house_distinct_positions=len(hp),street_distinct_positions=len(positions),
            house_ambiguous=len(hp)>1,street_cloud_not_single_house=True,no_centroid_as_house=True,
            no_BAN_anchor_used=True,missing=not bool(points))
        coverage['all_queries']+=1;coverage['no_commune_route']+=not bool(codes)
        coverage['has_OSM_street_cloud']+=bool(points);coverage['has_exact_OSM_house']+=bool(house_refs)
        coverage['multiple_exact_house_positions']+=len(hp)>1;coverage['has_unsupported_L93_positions']+=any(not p['l93_compatible'] for p in points)
    con.close();base.guard()
    base.io.write(ROOT/'query_context.json',context)
    published_prefixes={l['name'][5:-4] for l in lots};observed_prefixes=set(prefix_counts)
    generation=dict(status='COMPLETE_NATIONAL_BANO_OSM_ALL526_QUERY_ONLY_SOURCE',launch=base.record(ROOT/'launch.json'),
        schema_preflight=base.record(ROOT/'schema_preflight.json'),index=base.record(database),
        query_context=base.record(ROOT/'query_context.json'),all_queries=526,counters=dict(counters),
        source_counts=dict(source_counts),prefix_source_counts={k:dict(v) for k,v in prefix_counts.items()},
        regional_published_prefixes_without_national_rows=sorted(published_prefixes-observed_prefixes),
        national_prefixes_without_regional_listing=sorted(observed_prefixes-published_prefixes),
        national_archive_not_verified_equal_to_regional_union=True,coverage=dict(coverage),
        no_BAN_seed_used=True,no_statistical_independence_guarantee=True,current_community_diagnostic_only=True,
        no_GT_or_candidate_identifiers_read=True,no_recall_measurement=True,all_sources_archived_not_only_OSM=True,
        elapsed_s=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,completed_unix=time.time())
    base.io.write(ROOT/'generation.json',generation);base.io.write(ROOT/'state.json',generation)
    print('COMPLETE_NATIONAL_BANO_OSM_SOURCE',dict(coverage),flush=True)


if __name__=='__main__':main()
