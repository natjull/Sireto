"""ALL526 own-street cloud feasibility for public sites without own XY/house.

No candidate point, centroid, coordinate fill, retrieval, training or outcomes.
Only bounded SQL aggregates of the explicit OSM source are exported.
"""
from collections import Counter
from functools import lru_cache
import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pyarrow.parquet as pq

from scripts import embeddinggemma2_v145_candidate_own_address_source as source

OUT = source.prior.ROOT / 'v147/candidate_own_street_footprint_feasibility_20261014_night'
os.environ['SQLITE_TMPDIR'] = str(OUT / 'sqlite_tmp')
RECIPE = dict(scope='ALL5430064_ORIGINAL_DEDUPLICATED_NUMERIC_REFS_ALL526_INCLUDING4_EMPTY',
    role='OWN_PUBLISHED_STREET_OR_LIEU_ADDRESS_CLOUD_FEASIBILITY_NOT_EXACT_SITE',
    eligibility='MISSING_SUPPORTED_OWN_XY_BOTH_PUBLIC_EO_UO_NO_SIMPLE_HOUSE_NUMBER',
    simple_house_parser='UNCHANGED_V144_V132_SINGLE_NUMBER_SUFFIX',
    street_parser='UNCHANGED_V17_NORMALIZE_STREET_TRUE_NO_NEW_ALIAS',
    explicit_range_exclusion='V145_RANGE_END_FIELDS_OR_RAW_DIGITS_SEPARATOR_MINUS_EN_EM_SEMICOLON_SLASH_COMMA_AMPERSAND_PLUS_DIGITS',
    simple_house_excluded_from_street_scope=True,
    own_address='SAME_SITE_PRIMARY_TYPE_STREET_INSEE_ELSE_UNIQUE_PUBLISHED_OWN_CP_IF_INSEE_EMPTY',
    join='EXACT_NORMALIZED_OWN_STREET_KEY_AND_OWN_ROUTED_INSEE',
    source='EXACT_OSM_ONLY_SEALED_NATIONAL_BANO_SQLITE_UNCHANGED',
    sql='AGGREGATES_ONLY_SOURCE_ROWS_DISTINCT_WGS84_POSITIONS_MIN_MAX_LAT_LON',
    cloud_positions='ALL_MATCHING_PUBLISHED_POINTS_AMBIGUITY_RETAINED_NEVER_ONE_SELECTED',
    bbox='WGS84_MIN_MAX_AND_DESCRIPTIVE_HAVERSINE_DIAGONAL_EARTH_RADIUS6371000M_NOT_FOOTPRINT_RADIUS',
    coordinates='NO_CANDIDATE_POINT_NO_CENTROID_NO_CANONICAL_COORDINATE_WRITE',
    singleton='ONE_PUBLISHED_STREET_POINT_STILL_NOT_EXACT_SITE_OR_HOUSE',
    position_bins=[1, 10, 100, 1000], bbox_percentiles=[50, 90, 99],
    no_crm_address=True, no_parent_or_neighbour_inheritance=True, no_labels_or_outcomes=True,
    no_retrieval_or_training=True, no_recall_gain_or_label_error_claim=True,
    no_operator_identity_or_timestamp_certification=True, future_activation_not_inferred=True,
    cache_max_aggregate_keys=8192, threads=1, max_rss_bytes=512 * 1024**2, deadline_unix=1791954368)


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def explicit_range(row):
    return bool(row['dernierNumeroVoieEtablissement'] or row['indiceRepetitionDernierNumeroVoieEtablissement'] or
                re.search(r'\d+\s*[-–—;/,&+]\s*\d+', row['numeroVoieEtablissement']))


def bbox_diagonal(lat_min, lat_max, lon_min, lon_max):
    dlat = math.radians(lat_max - lat_min)
    dlon = math.radians(lon_max - lon_min)
    aa = math.sin(dlat / 2)**2 + math.cos(math.radians(lat_min)) * math.cos(math.radians(lat_max)) * math.sin(dlon / 2)**2
    return 2 * 6371000 * math.asin(math.sqrt(min(1., max(0., aa))))


def prepare():
    source.guard()
    if OUT.exists():
        raise ValueError('Fresh immutable footprint analytical namespace')
    OUT.mkdir(parents=True)
    (OUT / 'sqlite_tmp').mkdir()
    write('recipe.json', RECIPE)  # Fixed BEFORE any analytical source scan.
    old = source.prior.read(source.OUT / 'launch.json')['inputs']
    bindings = {k: v for k, v in old.items() if k not in ('implementation', 'recipe')}
    bindings.update(implementation=source.record(Path(__file__)), recipe=source.record(OUT / 'recipe.json'),
                    source_helper=source.record(Path(source.__file__)), prior_source_manifest=source.record(source.OUT / 'manifest.json'))
    for ref in bindings.values():
        source.verify(ref)
    bg = source.prior.read(bindings['bano_generation']['path'])
    bl = source.prior.read(bindings['bano_launch']['path'])
    vl = source.prior.read(bindings['v143_launch']['path'])
    public = source.prior.read(bindings['public_source']['path'])
    if bg['all_queries'] != 526 or bl['retained_source_codes'] != ['OSM'] or public['site_source'] != vl['stock_source'] or public['owner_source'] != vl['public_owners']:
        raise ValueError('Exact complete OSM and eO/uO source bindings')
    write('launch.json', dict(status='FROZEN_OWN_STREET_CLOUD_FEASIBILITY_SOURCE_ONLY', inputs=bindings,
          source_bindings_all_rehashed=True, created_unix=time.time(), labels_read=False, outcomes_read=False,
          canonical_coordinates_mutated=False, retrieval_executed=False, training_executed=False))
    print('FROZEN_FOOTPRINT_RECIPE_AND_ALL_INPUT_HASHES', flush=True)


def build():
    started = time.time()
    bindings = source.prior.read(OUT / 'launch.json')['inputs']
    if source.prior.read(source.verify(bindings['recipe'])) != RECIPE:
        raise ValueError('Frozen footprint recipe drift')
    source.verify(bindings['implementation'])
    launch = source.prior.read(bindings['v143_launch']['path'])
    public = source.prior.read(bindings['public_source']['path'])
    refs = np.load(bindings['universe']['path'], mmap_mode='r')
    if len(refs) != 5430064 or np.any(refs[1:] <= refs[:-1]):
        raise ValueError('Complete original sorted reference universe')
    supported, both_public = source.snapshot_flags(refs, launch, public)
    counters = Counter(all_queries=526, empty_numeric_queries=4, all_numeric_reference_sites=len(refs),
          existing_supported_own_xy=int(supported.sum()), existing_own_xy_missing=int((~supported).sum()),
          both_public=int(both_public.sum()), missing_both_public=int((~supported & both_public).sum()))
    write('source_preflight.json', dict(status='PASS_ALL_ORIGINAL_REFS_ENDPOINT_AND_PUBLIC_SOURCE_BINDINGS', counters=dict(counters)))
    postal = {}
    for row in source.prior.read(bindings['communes']['path']):
        for cp in row['codesPostaux']:
            postal.setdefault(cp, set()).add(row['code'])
    bano = sqlite3.connect(f"file:{bindings['bano_index']['path']}?mode=ro", uri=True)
    bano.execute('PRAGMA query_only=ON')
    bano.execute('PRAGMA cache_size=-16384')
    bano.execute('PRAGMA temp_store=FILE')
    clouds = sqlite3.connect(OUT / 'street_cloud_aggregates.sqlite')
    clouds.execute('PRAGMA cache_size=-16384')
    clouds.execute('CREATE TABLE clouds(insee TEXT,street_key TEXT,source_rows INTEGER,positions INTEGER,lat_min REAL,lat_max REAL,lon_min REAL,lon_max REAL,bbox_diagonal_m REAL,candidate_count INTEGER,PRIMARY KEY(insee,street_key)) WITHOUT ROWID')

    @lru_cache(maxsize=8192)
    def cloud(ci, street):
        # Distinct published coordinate groups remain inside SQL, not Python tuples.
        return bano.execute('''SELECT coalesce(sum(n),0),count(*),min(lat),max(lat),min(lon),max(lon)
            FROM (SELECT lat,lon,count(*) n FROM addresses WHERE insee=? AND street_key=? GROUP BY lat,lon)''', (ci, street)).fetchone()

    cols = ['siret', 'statutDiffusionEtablissement', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
            'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
            'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement', 'codePostalEtablissement']
    seen = np.zeros(len(refs), dtype=np.bool_)
    for bi, batch in enumerate(pq.ParquetFile(bindings['own_site']['path']).iter_batches(batch_size=8192, columns=cols, use_threads=False), 1):
        source.guard()
        nums = np.asarray(batch.column(0).to_pylist(), dtype=np.uint64)
        at = np.searchsorted(refs, nums)
        valid = at < len(refs)
        valid[valid] &= refs[at[valid]] == nums[valid]
        for i in np.flatnonzero(valid):
            p = int(at[i])
            if seen[p]:
                raise ValueError('Duplicate own source site')
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
            if explicit_range(row):
                counters['explicit_ranges_or_lists_rejected'] += 1
                counters['inline_ranges_or_lists_rejected'] += bool(re.search(r'\d+\s*[-–—;/,&+]\s*\d+', row['numeroVoieEtablissement']))
                continue
            if source.prior.number(row['numeroVoieEtablissement'] + ' ' + row['indiceRepetitionEtablissement']):
                counters['simple_house_not_street_scope'] += 1
                continue
            counters['without_simple_house_in_street_scope'] += 1
            kind = 'published_number_missing' if not row['numeroVoieEtablissement'] else 'published_number_not_simple'
            counters[kind] += 1
            street = source.prior.normalize(row['typeVoieEtablissement'] + ' ' + row['libelleVoieEtablissement'], street=True)
            if not street:
                counters['no_own_street_key'] += 1
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
            counters['eligible_exact_own_street'] += 1
            nr, np_, la0, la1, lo0, lo1 = cloud(ci, street)
            if not nr:
                counters['no_exact_osm_street_cloud'] += 1
                continue
            diag = bbox_diagonal(la0, la1, lo0, lo1)
            counters['has_exact_osm_street_cloud'] += 1
            counters[kind + '_has_exact_osm_street_cloud'] += 1
            counters['cloud_route_' + route] += 1
            counters['cloud_one_position_not_house'] += np_ == 1
            counters['cloud_multiple_positions_not_exact_site'] += np_ > 1
            bin_ = '1' if np_ == 1 else '2_10' if np_ <= 10 else '11_100' if np_ <= 100 else '101_1000' if np_ <= 1000 else 'over1000'
            counters['candidate_position_bin_' + bin_] += 1
            counters['candidate_cloud_has_outside_l93_region'] += not (-6 <= lo0 <= lo1 <= 10 and 41 <= la0 <= la1 <= 52)
            clouds.execute('''INSERT INTO clouds VALUES (?,?,?,?,?,?,?,?,?,1)
                ON CONFLICT(insee,street_key) DO UPDATE SET candidate_count=candidate_count+1''', (ci, street, nr, np_, la0, la1, lo0, lo1, diag))
        if bi % 100 == 0:
            clouds.commit()
            write('state.json', dict(phase='QUERY_ONLY_OWN_STREET_CLOUD_AGGREGATES', counters=dict(counters),
                  scanned_rows=bi * 8192, rss_bytes=source.guard(), elapsed_s=time.time() - started))
            print('SCAN', bi * 8192, counters['own_source_present'], counters['has_exact_osm_street_cloud'], flush=True)
    bano.close()
    clouds.commit()
    if not seen.all() or counters['own_source_present'] != 5430064:
        raise ValueError('Complete original denominator source join')
    if counters['missing_both_public'] != counters['explicit_ranges_or_lists_rejected'] + counters['simple_house_not_street_scope'] + counters['without_simple_house_in_street_scope']:
        raise ValueError('Disjoint missing/public scope conservation')
    if counters['without_simple_house_in_street_scope'] != counters['no_own_street_key'] + counters['no_safe_commune_route'] + counters['eligible_exact_own_street']:
        raise ValueError('Every own-street scope loss accounted for')
    if counters['eligible_exact_own_street'] != counters['no_exact_osm_street_cloud'] + counters['has_exact_osm_street_cloud']:
        raise ValueError('Exact cloud source join conservation')
    count = clouds.execute('SELECT count(*) FROM clouds').fetchone()[0]
    counters['distinct_matching_own_commune_street_keys'] = count
    diagonals = np.zeros(count, dtype=float)
    weights = np.zeros(count, dtype=np.int64)
    with (OUT / 'street_cloud_aggregates.jsonl').open('x') as stream:
        for i, row in enumerate(clouds.execute('SELECT * FROM clouds ORDER BY insee,street_key')):
            ci, street, nr, np_, la0, la1, lo0, lo1, diag, weight = row
            diagonals[i], weights[i] = diag, weight
            stream.write(json.dumps(dict(own_insee=ci, own_normalized_street_key=street,
                source='OSM', source_archive='full.csv.gz', source_rows=nr, distinct_published_positions=np_,
                bbox_wgs84=dict(lat_min=la0, lat_max=la1, lon_min=lo0, lon_max=lo1),
                bbox_descriptive_haversine_diagonal_m=diag, candidate_count=weight,
                role='STREET_ADDRESS_POINT_CLOUD_NOT_SITE_OR_HOUSE', ambiguity_retained=True,
                no_candidate_point_or_centroid=True, no_identity_or_timestamp_certification=True), sort_keys=True) + '\n')
    clouds.close()
    if int(weights.sum()) != counters['has_exact_osm_street_cloud']:
        raise ValueError('Distinct street/candidate count conservation')
    summaries = {}
    if count:
        order = np.argsort(diagonals)
        cumulative = np.cumsum(weights[order])
        summaries = dict(distinct_street_bbox_diagonal_m=dict(min=float(diagonals.min()), max=float(diagonals.max()),
            mean=float(diagonals.mean()), **{f'p{p}': float(np.percentile(diagonals, p)) for p in RECIPE['bbox_percentiles']}),
            candidate_weighted_bbox_diagonal_m=dict(mean=float(np.sum(diagonals * weights) / weights.sum()),
                **{f'p{p}': float(diagonals[order[min(count - 1, np.searchsorted(cumulative, p / 100 * cumulative[-1]))]]) for p in RECIPE['bbox_percentiles']}))
    for ref in bindings.values():
        source.verify(ref)
    write('feasibility.json', dict(status='COMPLETE_ALL526_OWN_STREET_CLOUD_QUERY_ONLY_FEASIBILITY_NOT_EXACT_SITE',
        launch=source.record(OUT / 'launch.json'), recipe=source.record(OUT / 'recipe.json'), counters=dict(counters),
        bbox_summaries=summaries, street_aggregates=source.record(OUT / 'street_cloud_aggregates.jsonl'),
        full_original_universe_and_all526_retained=True, source_snapshots_rehashed_before_and_after=True,
        labels_read=False, outcomes_read=False, no_candidate_point_or_centroid=True,
        no_canonical_coordinate_write=True, no_retrieval_or_training=True, no_recall_claim=True,
        street_footprint_never_exact_site=True, future_activation_not_inferred=True,
        final_rss_bytes=source.guard(), elapsed_s=time.time() - started, completed_unix=time.time()))
    print('COMPLETE_OWN_STREET_CLOUD_FEASIBILITY', dict(counters), flush=True)


def run():
    prepare()
    started = time.time()
    with (OUT / 'source.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v147_candidate_own_street_footprint_feasibility', 'build'], stdout=log, stderr=subprocess.STDOUT)
        while child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    source.guard()
                except BaseException:
                    child.terminate()
                    child.wait(timeout=10)
                    raise
    write('owned_tool_exit.json', dict(exit_code=child.returncode, own_child_exit_confirmed=True,
        owned_pid=child.pid, runtime_s=time.time() - started, completed_unix=time.time(),
        labels_read=False, outcomes_read=False, no_retrieval_or_training=True))
    if child.returncode:
        raise RuntimeError('Owned footprint analytical process failed, inspect source.log')
    report = source.prior.read(OUT / 'feasibility.json')
    source.verify(report['street_aggregates'])
    write('manifest.json', dict(status='COMPLETE_OWNED_SOURCE_ONLY_FOOTPRINT_FEASIBILITY_NOT_RECALL',
        report=source.record(OUT / 'feasibility.json'), own_exit=source.record(OUT / 'owned_tool_exit.json'),
        recipe=source.record(OUT / 'recipe.json'), diagnostic_only=True,
        labels_read=False, outcomes_read=False, no_coordinate_fill_or_activation=True))
    print(json.dumps(dict(manifest=source.record(OUT / 'manifest.json'), exit_code=child.returncode)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'build'))
    args = parser.parse_args()
    globals()[args.action]()
