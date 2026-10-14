"""Sparse community-estimated candidate OWN-address points, source-only.

Never writes canonical coordinates, fits, retrieves or opens labels/outcomes.
The complete V144 deduplicated universe is audited before sparse export.
"""
from collections import Counter
from functools import lru_cache
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import psutil
import pyarrow.parquet as pq

from scripts import embeddinggemma2_v144_candidate_own_address_feasibility as prior
from src.xgb_matcher.retrieval_geo_v16 import lonlat_to_l93

OUT = prior.ROOT / 'v145/candidate_own_address_source_20261014_night'
PRIOR_HASH = '01c7a6545a79f70f22cb15e1fbadf113145a7c75735e0e6befec4a6a13982f70'
RECIPE = dict(scope='ALL526_V144_COMPLETE_5430064_DEDUPLICATED_ORIGINAL_NUMERIC_REFERENCES',
    source_role='SPARSE_CURRENT_COMMUNITY_ESTIMATED_OWN_ADDRESS_POINTS_DIAGNOSTIC_ONLY',
    candidate_eligibility='NO_EXISTING_SUPPORTED_OWN_XY_AND_EXACT_SAME_SITE_EO_UO',
    own_address='SAME_SITE_OFFICIAL_PRIMARY_NUMBER_SUFFIX_TYPE_STREET_INSEE_OR_UNIQUE_PUBLISHED_OWN_CP',
    number_parser='UNCHANGED_V144_V132_SINGLE_INTEGER_AND_OPTIONAL_BIS_TER_QUATER_OR_LETTER',
    street_parser='UNCHANGED_V17_NORMALIZE_STREET_TRUE', ranges='ANY_EXPLICIT_RANGE_END_REJECTED',
    source_filter='EXACT_OSM_ONLY_SEALED_NATIONAL_BANO_SQLITE',
    route='OWN_INSEE_AS_PUBLISHED_ELSE_ONLY_ONE_CURRENT_PUBLISHED_CP_COMMUNE_IF_OWN_INSEE_EMPTY',
    source_admission='EXACT_ADDRESS_ONE_DISTINCT_WGS84_POSITION_ACROSS_ALL_MATCHING_SOURCE_ROWS',
    projection='LONLAT_TO_L93_GRS80_V16_THEN_FINITE_0_LE_X_LE_1300000_6000000_LE_Y_LE_7300000',
    source_wgs84_compatibility='UNCHANGED_V144_MINUS6_LE_LON_LE_10_41_LE_LAT_LE_52',
    no_multipoint_selection=True, no_centroid=True, no_crm_address=True, no_parent_or_neighbour_inheritance=True,
    no_suffix_alias_added=True, no_canonical_coordinate_write=True, no_retrieval_or_training=True,
    no_labels_or_outcomes=True, no_recall_claim=True, no_operator_identity_or_timestamp_certification=True,
    identifiers='TECHNICAL_SOURCE_AND_SITE_JOIN_ONLY_NOT_FEATURES',
    denominator='ALL5430064_REFS_AND_ALL526_QUERIES_UNCHANGED_INCLUDING4_EMPTY_NUMERIC',
    exported_keys='SORTED_UNIQUE_UINT64_SPARSE_SUBSET_OF_PREEXISTING_REFERENCE_UNIVERSE',
    threads=1, max_rss_bytes=512 * 1024**2, deadline_unix=1791954368)


def guard():
    if time.time() >= RECIPE['deadline_unix']:
        raise TimeoutError('Absolute night deadline')
    rss = psutil.Process().memory_info().rss
    if rss > RECIPE['max_rss_bytes']:
        raise MemoryError(f'Source-only RSS cap: {rss}')
    return rss


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            guard()
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def verify(ref):
    if record(ref['path']) != ref:
        raise ValueError('Hash/size drift: ' + ref['path'])
    return Path(ref['path'])


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, sort_keys=True, indent=2) + '\n')


def snapshot_flags(refs, launch, public):
    supported = np.zeros(len(refs), dtype=np.bool_)
    both_public = np.zeros(len(refs), dtype=np.bool_)
    seen = np.zeros(len(refs), dtype=np.bool_)
    mappings = [np.load(launch[k]['path'], mmap_mode='r') for k in ('keys', 'xy')]
    mappings.append(np.load(public['public_mask']['path'], mmap_mode='r'))
    total = len(mappings[0])
    for first in range(0, total, 262144):
        end = min(first + 262144, total)
        kk = np.array(mappings[0][first:end])
        at = np.searchsorted(refs, kk)
        good = at < len(refs)
        good[good] &= refs[at[good]] == kk[good]
        take = np.flatnonzero(good)
        p = at[take]
        xy = np.asarray(mappings[1][first:end])[take]
        supported[p] = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] <= 1300000) & (xy[:, 1] >= 6000000) & (xy[:, 1] <= 7300000)
        both_public[p] = mappings[2][first:end][take]
        seen[p] = True
        if first % (262144 * 8) == 0:
            for mm in mappings:
                mm._mmap.close()
            mappings = [np.load(launch[k]['path'], mmap_mode='r') for k in ('keys', 'xy')]
            mappings.append(np.load(public['public_mask']['path'], mmap_mode='r'))
        guard()
    for mm in mappings:
        mm._mmap.close()
    if not seen.all():
        raise ValueError('Complete numeric reference universe endpoint join')
    return supported, both_public


def prepare():
    guard()
    if OUT.exists():
        raise ValueError('Fresh immutable source namespace required')
    OUT.mkdir(parents=True)
    write('recipe.json', RECIPE)  # Frozen/exported BEFORE source scanning.
    result = prior.read(prior.OUT / 'feasibility.json')
    if record(prior.OUT / 'feasibility.json')['sha256'] != PRIOR_HASH:
        raise ValueError('Exact completed source-only V144 receipt')
    launch = prior.read(prior.V143 / 'launch.json')
    public = prior.read(prior.V143 / 'public_source.json')
    bg = prior.read(prior.BANO / 'generation.json')
    bl = prior.read(prior.BANO / 'launch.json')
    if public['site_source'] != launch['stock_source'] or public['owner_source'] != launch['public_owners']:
        raise ValueError('Exact same-site and same-owner public source bindings')
    if result['counters']['unique_reference_sites'] != 5430064 or bg['all_queries'] != 526 or bl['retained_source_codes'] != ['OSM']:
        raise ValueError('Unchanged complete reference universe and explicit OSM source')
    refs = dict(implementation=record(Path(__file__)), recipe=record(OUT / 'recipe.json'),
        v144_receipt=record(prior.OUT / 'feasibility.json'), universe=record(prior.OUT / 'deduplicated_refs.npy'),
        v143_launch=record(prior.V143 / 'launch.json'), public_source=record(prior.V143 / 'public_source.json'),
        bano_generation=record(prior.BANO / 'generation.json'), bano_launch=bg['launch'], bano_index=bg['index'],
        own_site=launch['stock_source'], official_owners=launch['public_owners'], endpoint_keys=launch['keys'],
        official_supported_own_xy=launch['xy'], both_public_mask=public['public_mask'], communes=bl['commune_reference'],
        prior_helper=record(Path(prior.__file__)), projection_helper=record(Path(sys.modules[lonlat_to_l93.__module__].__file__)))
    for ref in refs.values():
        verify(ref)
    write('launch.json', dict(status='FROZEN_SPARSE_COMMUNITY_OWN_ADDRESS_SOURCE_DIAGNOSTIC', inputs=refs,
          source_bindings_all_rehashed=True, created_unix=time.time(), labels_read=False, outcomes_read=False,
          canonical_coordinates_mutated=False, retrieval_executed=False, training_executed=False))
    print('FROZEN_SOURCE_RECIPE_AND_ALL_INPUT_HASHES', flush=True)


def raw_reason(row):
    raw = [row['coordonneeLambertAbscisseEtablissement'], row['coordonneeLambertOrdonneeEtablissement']]
    if not any(raw):
        return 'raw_both_absent'
    if not all(raw):
        return 'raw_incomplete'
    try:
        x, y = map(float, raw)
    except ValueError:
        return 'raw_invalid'
    if not math.isfinite(x) or not math.isfinite(y):
        return 'raw_nonfinite'
    ci = row['codeCommuneEtablissement']
    if ci.startswith(('97', '98')):
        return 'raw_finite_overseas_not_supported'
    if not ci:
        return 'raw_finite_no_insee'
    if not (0 <= x <= 1300000 and 6000000 <= y <= 7300000):
        return 'raw_finite_outside_l93_bounds'
    raise ValueError('Missing supported snapshot despite supported raw own XY')


def build():
    started = time.time()
    launch145 = prior.read(OUT / 'launch.json')
    bindings = launch145['inputs']
    if prior.read(verify(bindings['recipe'])) != RECIPE:
        raise ValueError('Frozen pre-scan recipe')
    verify(bindings['implementation'])
    launch = prior.read(bindings['v143_launch']['path'])
    public = prior.read(bindings['public_source']['path'])
    refs = np.load(bindings['universe']['path'], mmap_mode='r')
    if len(refs) != 5430064 or np.any(refs[1:] <= refs[:-1]):
        raise ValueError('Exact sorted deduplicated complete numeric universe')
    supported, both_public = snapshot_flags(refs, launch, public)
    counters = Counter(all_queries=526, empty_numeric_queries=4, all_numeric_reference_sites=len(refs),
          existing_supported_own_xy=int(supported.sum()), existing_own_xy_missing=int((~supported).sum()),
          both_public=int(both_public.sum()), missing_both_public=int((~supported & both_public).sum()))
    write('source_preflight.json', dict(status='PASS_COMPLETE_UNIVERSE_AND_EXACT_PUBLIC_BINDINGS', counters=dict(counters)))
    postal = {}
    for row in prior.read(bindings['communes']['path']):
        for cp in row['codesPostaux']:
            postal.setdefault(cp, set()).add(row['code'])
    bano = sqlite3.connect(f"file:{bindings['bano_index']['path']}?mode=ro", uri=True)
    bano.execute('PRAGMA query_only=ON')
    bano.execute('PRAGMA cache_size=-16384')
    sparse = sqlite3.connect(OUT / 'sparse_source.sqlite')
    sparse.execute('PRAGMA cache_size=-16384')
    sparse.execute('CREATE TABLE points(key INTEGER PRIMARY KEY,x REAL,y REAL,lon REAL,lat REAL,provenance TEXT)')

    @lru_cache(maxsize=8192)
    def exact(ci, street, ni, suffix):
        return tuple(bano.execute('SELECT source_row,bano_id,insee,street_key,number_integer,suffix,lat,lon FROM addresses WHERE insee=? AND street_key=? AND number_integer=? AND suffix=? ORDER BY source_row', (ci, street, ni, suffix)))

    cols = ['siret', 'statutDiffusionEtablissement', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
            'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
            'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement', 'codePostalEtablissement',
            'coordonneeLambertAbscisseEtablissement', 'coordonneeLambertOrdonneeEtablissement']
    seen = np.zeros(len(refs), dtype=np.bool_)
    for batch_index, batch in enumerate(pq.ParquetFile(bindings['own_site']['path']).iter_batches(batch_size=8192, columns=cols, use_threads=False), 1):
        guard()
        nums = np.asarray(batch.column(0).to_pylist(), dtype=np.uint64)
        at = np.searchsorted(refs, nums)
        valid = at < len(refs)
        valid[valid] &= refs[at[valid]] == nums[valid]
        for i in np.flatnonzero(valid):
            p = int(at[i])
            if seen[p]:
                raise ValueError('Duplicate own SIRENE source site')
            seen[p] = True
            counters['own_source_present'] += 1
            if supported[p]:
                continue
            counters['missing_own_source_present'] += 1
            if not both_public[p]:
                counters['excluded_not_both_public'] += 1
                continue
            row = {name: batch.column(j)[int(i)].as_py() or '' for j, name in enumerate(cols)}
            if row['statutDiffusionEtablissement'] != 'O':
                raise ValueError('Own source public mask drift')
            counters['missing_public_' + raw_reason(row)] += 1
            if row['dernierNumeroVoieEtablissement'] or row['indiceRepetitionDernierNumeroVoieEtablissement']:
                counters['explicit_range_rejected'] += 1
                continue
            house = prior.number(row['numeroVoieEtablissement'] + ' ' + row['indiceRepetitionEtablissement'])
            street = prior.normalize(row['typeVoieEtablissement'] + ' ' + row['libelleVoieEtablissement'], street=True)
            if not house or not street:
                counters['missing_single_house_or_own_street'] += 1
                continue
            ci = row['codeCommuneEtablissement']
            route = 'OWN_INSEE'
            if not ci:
                codes = postal.get(row['codePostalEtablissement'], set())
                if len(codes) != 1:
                    counters['no_safe_commune_route'] += 1
                    continue
                ci = next(iter(codes))
                route = 'UNIQUE_PUBLISHED_OWN_CP'
            counters['eligible_exact_own_address'] += 1
            rows = exact(ci, street, *house)
            if not rows:
                counters['no_exact_osm_address_point'] += 1
                continue
            counters['exact_osm_address_point_available'] += 1
            positions = {(r[6], r[7]) for r in rows}
            if len(positions) != 1:
                counters['ambiguous_positions_rejected'] += 1
                continue
            lat, lon = next(iter(positions))
            if not (-6 <= lon <= 10 and 41 <= lat <= 52):
                counters['unsupported_wgs84_region_rejected'] += 1
                continue
            x, y = lonlat_to_l93(lon, lat)
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1300000 and 6000000 <= y <= 7300000):
                counters['projected_actual_l93_bounds_rejected'] += 1
                continue
            provenance = dict(site_key=int(nums[i]), source='OSM', archive='full.csv.gz',
                source_role='COMMUNITY_ESTIMATED_OWN_ADDRESS_NOT_OFFICIAL_OWN_XY', route=route,
                own_published_primary_address={k: row[k] for k in cols[2:10]},
                normalized_own_address=dict(insee=ci, street_key=street, number_integer=house[0], suffix=house[1]),
                bano_rows=[dict(source_row=r[0], bano_id=r[1], insee=r[2], street_key=r[3], number_integer=r[4], suffix=r[5]) for r in rows],
                position_wgs84=dict(lat=lat, lon=lon), point_l93=[x, y],
                public_site_and_owner=True, official_supported_own_xy_missing=True,
                source_current_community=True, official_operator_identity_certified=False, address_timestamp_certified=False,
                identifiers_technical_not_features=True)
            sparse.execute('INSERT INTO points VALUES (?,?,?,?,?,?)', (int(nums[i]), x, y, lon, lat, json.dumps(provenance, sort_keys=True)))
            counters['exported_points'] += 1
            counters['export_route_' + route] += 1
            counters['exported_bano_rows'] += len(rows)
        if batch_index % 100 == 0:
            sparse.commit()
            write('state.json', dict(phase='SOURCE_ONLY_SPARSE_OWN_ADDRESS_SCAN', counters=dict(counters),
                  scanned_rows=batch_index * 8192, rss_bytes=guard(), elapsed_s=time.time() - started))
            print('SCAN', batch_index * 8192, counters['own_source_present'], counters['exported_points'], flush=True)
    bano.close()
    if not seen.all() or counters['own_source_present'] != len(refs):
        raise ValueError('Every original numeric reference audited, no denominator filter')
    sparse.commit()
    n = sparse.execute('SELECT count(*) FROM points').fetchone()[0]
    if n != counters['exported_points'] or n > 357962:
        raise ValueError('Sparse source conservation and prior exact unique regional subset')
    keys = np.lib.format.open_memmap(OUT / 'keys.npy', mode='w+', dtype=np.uint64, shape=(n,))
    xy = np.lib.format.open_memmap(OUT / 'xy_points.npy', mode='w+', dtype=np.float64, shape=(n, 2))
    wgs = np.lib.format.open_memmap(OUT / 'wgs84_lonlat.npy', mode='w+', dtype=np.float64, shape=(n, 2))
    cur = sparse.execute('SELECT key,x,y,lon,lat,provenance FROM points ORDER BY key')
    index = 0
    with (OUT / 'provenance.jsonl').open('x') as stream:
        while rows := cur.fetchmany(8192):
            for key, x, y, lon, lat, provenance in rows:
                keys[index] = key
                xy[index] = x, y
                wgs[index] = lon, lat
                stream.write(provenance + '\n')
                index += 1
            guard()
    sparse.close()
    for mm in (keys, xy, wgs):
        mm.flush()
    if index != n or np.any(keys[1:] <= keys[:-1]) or not np.isfinite(xy).all():
        raise ValueError('Sorted unique finite sparse output')
    if not np.all((xy[:, 0] >= 0) & (xy[:, 0] <= 1300000) & (xy[:, 1] >= 6000000) & (xy[:, 1] <= 7300000)):
        raise ValueError('Every projected point actually compatible L93')
    for ref in bindings.values():
        verify(ref)  # All source snapshots unchanged after source-only scan.
    write('generation.json', dict(status='COMPLETE_HASHED_SPARSE_COMMUNITY_OWN_ADDRESS_SOURCE_NOT_RETRIEVAL',
        launch=record(OUT / 'launch.json'), recipe=record(OUT / 'recipe.json'), counters=dict(counters),
        artifacts={name: record(OUT / name) for name in ('keys.npy', 'xy_points.npy', 'wgs84_lonlat.npy', 'provenance.jsonl')},
        complete_universe_audited=True, source_snapshots_rehashed_before_and_after=True,
        labels_read=False, outcomes_read=False, canonical_coordinates_written=False,
        retrieval_executed=False, training_executed=False, recall_measured=False,
        source_estimated_not_official=True, public_source_checks_passed=True,
        sparse_sorted_unique=True, final_rss_bytes=guard(), elapsed_s=time.time() - started, completed_unix=time.time()))
    print('COMPLETE_SPARSE_SOURCE', dict(counters), flush=True)


def run():
    prepare()
    started = time.time()
    with (OUT / 'source.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v145_candidate_own_address_source', 'build'], stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    guard()
                except BaseException:
                    child.terminate()
                    child.wait(timeout=10)
                    raise
    receipt = dict(exit_code=child.returncode, own_child_exit_confirmed=True, owned_pid=child.pid,
        runtime_s=time.time() - started, completed_unix=time.time(), implementation=record(Path(__file__)),
        labels_read=False, outcomes_read=False, no_retrieval_or_training=True)
    write('owned_tool_exit.json', receipt)
    if child.returncode:
        raise RuntimeError('Owned sparse source process failed; inspect source.log')
    g = prior.read(OUT / 'generation.json')
    for ref in g['artifacts'].values():
        verify(ref)
    write('manifest.json', dict(status='READY_HASHED_SOURCE_ONLY_FOR_LATER_DIAGNOSTIC_PROJECTION_NOT_RECALL',
        generation=record(OUT / 'generation.json'), recipe=record(OUT / 'recipe.json'),
        own_exit=record(OUT / 'owned_tool_exit.json'), rows=g['counters']['exported_points'],
        shards=[dict(rows=g['counters']['exported_points'], **g['artifacts'])],
        full_universe_5430064_and_all526_retained=True, diagnostic_only=True, source_estimated_not_official=True,
        labels_read=False, outcomes_read=False, canonical_coordinates_written=False,
        retrieval_executed=False, training_executed=False, recall_measured=False))
    print(json.dumps(dict(manifest=record(OUT / 'manifest.json'), rows=g['counters']['exported_points'], exit_code=child.returncode)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'build'))
    args = parser.parse_args()
    globals()[args.action]()
