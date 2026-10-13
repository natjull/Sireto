"""National public BANO CSV source, source=OSM only, no BAN-seeded search.

No labels, candidate SIRETs or retrieval outcomes are read. All published
department lots are archived, including overseas, with provenance retained.
House evidence is a published address point, never an administrative centre;
street evidence is the entire matching cloud of published address points.
"""
from collections import Counter
import csv
import email.utils
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import sqlite3
import time
import urllib.request

import pyproj

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v129_osm_coverage_audit import number
from scripts.embeddinggemma2_v17_structural_retrieval import normalize
from src.xgb_matcher.retrieval_v20_workbench import record, verify

ROOT = io.HOME/'v141/bano_osm_20261013_night_a'
RAW = Path('/Users/nathanjullia/Documents/Projets/SIRETO/tmp/bano_public_20261013_lZ8b88')
SOURCE = io.HOME/'v129/osm_local_20261013_night_e'
COMMUNES = Path('/Volumes/CATNAT_DATA/oubusiness/raw/rna_association_stock/2026-07-01_rna-v1/communes_geo_api.json')
DELEGATES = Path('/Volumes/CATNAT_DATA/oubusiness/raw/depp_primary_school_enrolment/2026-07-31_primary-v2/geo_api_communes_deleguees_2026-07-31.json')
URL = 'https://bano.openstreetmap.fr/data/'
DOC = 'https://wiki.openstreetmap.org/wiki/FR:France/Base_Adresses_Nationale_Ouverte_(BANO)'
DEADLINE = 1791936000  # 2026-10-14 00:00:00 UTC; preparation only.
NETWORK_LIMIT = int(2.5*1024**3)
STORAGE_LIMIT = 32*1024**3


def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('Explicit midnight source preparation deadline')
    free, pressure = io.pressure()
    if free < 25 or pressure not in (0,1,2):
        raise RuntimeError(f'Real memory pressure: free={free}, pressure={pressure}')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > int(1.5*1024**3):
        raise MemoryError('Explicit 1.5 GiB RSS guard')
    if shutil.disk_usage(io.HOME).free < 40*1024**3:
        raise OSError('Explicit CATNAT 40 GiB reserve')
    if sum(p.stat().st_size for base in (ROOT,RAW) for p in base.rglob('*') if p.is_file()) > STORAGE_LIMIT:
        raise OSError('Explicit 32 GiB task storage guard')


def main():
    started=time.time()
    if ROOT.exists():
        raise ValueError('Fresh immutable namespace required')
    ROOT.mkdir(parents=True)
    guard()
    launch_source=io.read(SOURCE/'launch.json')
    panel=io.read(verify(launch_source['recipe']['query_only']))
    if len(panel)!=526 or len({q['query_id'] for q in panel})!=526:
        raise ValueError('Full unique ALL526 query-only scope required')
    request=urllib.request.Request(URL,headers={'User-Agent':'SIRETO-local-source-audit/1.0'})
    with urllib.request.urlopen(request,timeout=30) as response:
        listing=response.read(1024**2+1)
        if len(listing)>1024**2:raise ValueError('Unexpected listing volume')
        listing_headers=dict(response.headers)
    (ROOT/'listing.html').write_bytes(listing)
    lots=[]
    for line in listing.decode('utf8').splitlines():
        match=re.search(r'href="(bano-[0-9AB]+\.csv)".*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})',line)
        if match:lots.append(dict(name=match[1],listing_last_modified_minute=match[2]))
    lots.sort(key=lambda x:x['name'])
    if len(lots)!=106 or len({l['name'] for l in lots})!=106 or lots[0]['name']!='bano-01.csv':
        raise ValueError('Producer listing changed; no selective scope reduction')
    launch=dict(status='FROZEN_NATIONAL_BANO_SOURCE_OSM_ONLY',implementation=record(Path(__file__)),
        query_only=launch_source['recipe']['query_only'],listing=record(ROOT/'listing.html'),
        listing_headers=listing_headers,lots=lots,raw_archive_root=str(RAW),
        producer_documentation=DOC,source_nomenclature=dict(OSM='OSM names and numbers',
            **{'O+O':'OSM names, BAL numbers','OD':'BAL names and numbers','C+O':'OSM names, BAN numbers','CAD':'BAN names and numbers'}),
        retained_source_codes=['OSM'],no_json_without_provenance=True,
        probe_lot='bano-01.csv',probe_chosen_by_lexicographic_order_not_queries=True,
        columns=['id','numero','voie','code_post','ville','source','lat','lon'],
        commune_reference=record(COMMUNES),delegate_reference=record(DELEGATES),
        routing='current INSEE; original+current delegated INSEE; else all published CP communes',
        address_match='exact V132 street normalization and single number/suffix parser',
        street_cloud='ALL source=OSM published points on exact normalized street in routed communes',
        no_centroid_house=True,no_BAN_geometry_input=True,osm_may_historically_import_BAN=True,
        overseas_kept_WGS84_but_L93_unsupported=True,diagnostic_community_current_source_only=True,
        labels_read=False,recall_measured=False,network_limit_bytes=NETWORK_LIMIT,
        storage_limit_bytes=STORAGE_LIMIT,deadline_unix=DEADLINE,started_unix=started)
    io.write(ROOT/'launch.json',launch)
    database=ROOT/'addresses.sqlite';con=sqlite3.connect(database)
    con.execute('PRAGMA cache_size=-65536');con.execute('PRAGMA temp_store=FILE')
    con.execute('PRAGMA journal_mode=DELETE')
    con.execute('CREATE TABLE addresses(source_file TEXT NOT NULL,source_row INTEGER NOT NULL,bano_id TEXT NOT NULL,insee TEXT NOT NULL,street TEXT NOT NULL,street_key TEXT NOT NULL,number_raw TEXT NOT NULL,number_integer INTEGER,suffix TEXT,postcode TEXT,city TEXT,lat REAL,lon REAL,PRIMARY KEY(source_file,source_row)) WITHOUT ROWID')
    counters=Counter();archives=[];network=len(listing);batch=[]
    for lot_index,lot in enumerate(lots):
        guard();path=RAW/lot['name']
        if path.exists():raise ValueError('Unbound existing CSV archive')
        with urllib.request.urlopen(URL+lot['name'],timeout=60) as response, path.open('xb') as output:
            headers=dict(response.headers);length=int(response.headers.get('Content-Length','0'))
            if not length or network+length>NETWORK_LIMIT:raise ValueError('National CSV network budget invalidated')
            actual_minute=email.utils.parsedate_to_datetime(response.headers['Last-Modified']).strftime('%Y-%m-%d %H:%M')
            if actual_minute!=lot['listing_last_modified_minute']:raise ValueError('Producer lot changed after frozen listing')
            sha=hashlib.sha256();downloaded=0
            while chunk:=response.read(1024**2):
                output.write(chunk);sha.update(chunk);downloaded+=len(chunk)
            if downloaded!=length:raise ValueError('Incomplete CSV download')
        network+=downloaded;source_counts=Counter();raw_rows=0
        with path.open('r',encoding='utf8',newline='') as stream:
            for row_index,row in enumerate(csv.reader(stream),start=1):
                if len(row)!=8:raise ValueError(f'CSV8 schema drift {lot["name"]}:{row_index}')
                bano_id,num,street,postcode,city,source,lat_raw,lon_raw=row
                raw_rows+=1;source_counts[source]+=1;counters['all_csv_rows']+=1
                if source not in ('OSM','O+O','OD','C+O','CAD'):raise ValueError('Unknown producer source code')
                if source!='OSM':continue
                if not re.match(r'^(?:\d{5}|2[AB]\d{3})',bano_id):raise ValueError('Invalid public commune prefix')
                lat=float(lat_raw);lon=float(lon_raw)
                if not (-90<=lat<=90 and -180<=lon<=180):raise ValueError('Invalid published WGS84 point')
                parsed=number(num)
                batch.append((lot['name'],row_index,bano_id,bano_id[:5],street,normalize(street,street=True),num,
                    parsed[0] if parsed else None,parsed[1] if parsed else None,postcode,city,lat,lon))
                counters['retained_OSM_rows']+=1;counters['nonexact_house_numbers_retained']+=not bool(parsed)
                if len(batch)>=8192:
                    con.executemany('INSERT INTO addresses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);batch.clear()
                if row_index%131072==0:guard()
        if batch:con.executemany('INSERT INTO addresses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);batch.clear()
        con.commit()
        archives.append(dict(path=str(path),sha256=sha.hexdigest(),size_bytes=downloaded,
            url=URL+lot['name'],headers=headers,rows=raw_rows,source_counts=dict(source_counts)))
        if lot_index==0:
            io.write(ROOT/'schema_preflight.json',dict(status='PASS_CSV8_EXPLICIT_SOURCE',
                probe=archives[-1],all_rows_schema_checked=True,first_lexicographic_lot=True,
                no_JSON_merged_provenance=True,columns=launch['columns'],source_counts=dict(source_counts)))
        io.write(ROOT/'state.json',dict(status='PREPARING_NATIONAL_OSM_ADDRESS_INDEX',completed_lots=lot_index+1,
            total_lots=len(lots),network_bytes=network,counters=dict(counters),updated_unix=time.time()))
        print('LOT_COMPLETE',lot_index+1,lot['name'],dict(source_counts),flush=True)
    guard()
    con.execute('CREATE INDEX by_commune_street ON addresses(insee,street_key,number_integer,suffix)')
    con.commit()
    communes=io.read(COMMUNES);current={c['code'] for c in communes};postal={}
    for c in communes:
        for cp in c['codesPostaux']:postal.setdefault(cp,set()).add(c['code'])
    delegates={c['code']:c['chefLieu'] for c in io.read(DELEGATES) if c.get('chefLieu')}
    transform=pyproj.Transformer.from_crs('EPSG:4326','EPSG:2154',always_xy=True)
    context={};coverage=Counter()
    for q in panel:
        code=q['crm_insee'];cp=q['crm_postcode'];street_key=normalize(q['crm_street'],street=True)
        house=number(q.get('crm_number','')+' '+q.get('crm_suffix',''))
        if code in current:codes=[code];method='CURRENT_INSEE'
        elif delegates.get(code) in current:codes=sorted({code,delegates[code]});method='PUBLISHED_DELEGATED_INSEE'
        else:codes=sorted(postal.get(cp,set()));method='ALL_PUBLISHED_POSTCODE_COMMUNES'
        points=[];house_refs=[]
        if codes and street_key:
            sql='SELECT source_file,source_row,bano_id,insee,street,number_raw,number_integer,suffix,postcode,city,lat,lon FROM addresses WHERE street_key=? AND insee IN ('+','.join('?' for _ in codes)+')'
            for sf,sr,bid,ci,st,nraw,ni,suffix,pc,city,lat,lon in con.execute(sql,[street_key,*codes]):
                l93_ok=-6<=lon<=10 and 41<=lat<=52
                xy=list(transform.transform(lon,lat)) if l93_ok else None
                point=dict(source_file=sf,source_row=sr,bano_id=bid,insee=ci,street=st,number=nraw,
                    source='OSM',postcode=pc,city=city,lat=lat,lon=lon,l93_compatible=l93_ok,
                    point_l93=xy,precision='PUBLISHED_ADDRESS_POINT_NOT_CERTIFIED_ENTRANCE',
                    not_official_operator_identity=True)
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
    con.close();guard()
    io.write(ROOT/'archives.json',archives);io.write(ROOT/'query_context.json',context)
    generation=dict(status='COMPLETE_NATIONAL_BANO_OSM_ALL526_QUERY_ONLY_SOURCE',launch=record(ROOT/'launch.json'),
        schema_preflight=record(ROOT/'schema_preflight.json'),archives=record(ROOT/'archives.json'),
        index=record(database),query_context=record(ROOT/'query_context.json'),all_queries=526,
        all_lots=len(lots),network_bytes=network,counters=dict(counters),coverage=dict(coverage),
        source_is_national_not_BAN_neighbourhood_seeded=True,only_source_OSM_used=True,
        no_GT_or_candidate_identifiers_read=True,no_recall_measurement=True,
        no_statistical_independence_guarantee=True,current_community_diagnostic_only=True,
        elapsed_s=time.time()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        completed_unix=time.time())
    io.write(ROOT/'generation.json',generation);io.write(ROOT/'state.json',generation)
    print('COMPLETE_NATIONAL_BANO_OSM_SOURCE',json.dumps(dict(coverage)),flush=True)


if __name__=='__main__':
    main()
