"""Post-seal ALL526 label-side own-street-cloud coverage, never retrieval.

Uses exact V147 helpers/bindings. Protected targets are never looked up.
SQLite aggregates are reconciled against every sealed JSONL row BEFORE labels.
"""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import duckdb
import numpy as np

from scripts import embeddinggemma2_v147_candidate_own_street_footprint_feasibility as frozen

source = frozen.source
ROOT = source.prior.ROOT
OUT = ROOT / 'v147/target_coverage_diagnostic_20261014_night'
NAMES = ROOT / 'v101/postseal_name_roles_20261013_a'
STATUS = ROOT / 'v103/raw_alias_coverage_20261013_a'
PREVIOUS = ROOT / 'v145/target_coverage_diagnostic_20261014_night'
os.environ['SQLITE_TMPDIR'] = str(OUT / 'spill')
RECIPE = dict(role='POSTSEAL_LABEL_SIDE_ALL526_OWN_STREET_FOOTPRINT_COVERAGE_NOT_RECALL',
    source='UNCHANGED_COMPLETE_V147_SOURCE_ONLY_AGGREGATES',
    sqlite_authority='FULL_EXACT_RECONCILIATION_AGAINST_PRIOR_HASHED_JSONL_BEFORE_LABELS',
    matching='EXACT_FROZEN_V147_HELPERS_AND_OWN_PRIMARY_ADDRESS_ROUTE_NO_ALTERNATIVE',
    protected='SKIP_BEFORE_TARGET_ID_OR_OWN_ADDRESS_LOOKUP',
    scope='ALL526_FIRST_THEN_EXISTING20_REMAINING_INCLUDING18_PUBLIC_AND2_PROTECTED',
    target_original_numeric_universe_required=True, supported_own_xy_excluded=True,
    no_new_source_or_geometry=True, no_centroid_or_canonical_coordinate_write=True,
    no_labels_or_recipes_changed=True, no_retrieval_or_fit=True, no_gain_or_label_error_claim=True,
    closed_sets_opened=False, threads=1, max_rss_bytes=512 * 1024**2, deadline_unix=1791954368)


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def audit():
    source.guard()
    if OUT.exists():
        raise ValueError('Fresh label-side namespace required')
    OUT.mkdir(parents=True)
    (OUT / 'spill').mkdir()
    write('recipe.json', RECIPE)
    manifest = source.prior.read(frozen.OUT / 'manifest.json')
    prior_exit = source.prior.read(source.verify(manifest['own_exit']))
    prior_report = source.prior.read(source.verify(manifest['report']))
    source.verify(manifest['recipe'])
    if prior_exit['exit_code'] or not prior_exit['own_child_exit_confirmed'] or source.prior.read(manifest['recipe']['path']) != frozen.RECIPE:
        raise ValueError('Exact completed source before any label-side reading')
    launch = source.prior.read(source.verify(prior_report['launch']))
    bindings = launch['inputs']
    for ref in bindings.values():
        source.verify(ref)
    pipeline_launch = source.prior.read(bindings['v143_launch']['path'])
    normalizer_path = str(Path(sys.modules[source.prior.normalize.__module__].__file__).resolve())
    normalizer_refs = [r for r in pipeline_launch['helpers'] if r['path'] == normalizer_path]
    if not normalizer_refs:
        raise ValueError('Original frozen normalizer helper binding required, no replacement matching')
    source.verify(normalizer_refs[0])
    jsonl = source.verify(prior_report['street_aggregates'])
    db_path = frozen.OUT / 'street_cloud_aggregates.sqlite'
    cloud = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    cloud.execute('PRAGMA query_only=ON')
    cloud.execute('PRAGMA cache_size=-8192')
    checked = 0
    with jsonl.open() as stream:
        for line in stream:
            row = json.loads(line)
            bb = row['bbox_wgs84']
            expected = (row['source_rows'], row['distinct_published_positions'], bb['lat_min'], bb['lat_max'],
                        bb['lon_min'], bb['lon_max'], row['bbox_descriptive_haversine_diagonal_m'], row['candidate_count'])
            actual = cloud.execute('SELECT source_rows,positions,lat_min,lat_max,lon_min,lon_max,bbox_diagonal_m,candidate_count FROM clouds WHERE insee=? AND street_key=?',
                                   (row['own_insee'], row['own_normalized_street_key'])).fetchone()
            if row['source'] != 'OSM' or actual != expected:
                raise ValueError('Previously unhashbound SQLite differs from sealed source aggregate')
            checked += 1
    if checked != 23795 or checked != cloud.execute('SELECT count(*) FROM clouds').fetchone()[0]:
        raise ValueError('Every prior source aggregate and no extra SQLite key')
    sqlite_ref = source.record(db_path)
    write('source_preflight.json', dict(status='PASS_ALL_PRIOR_SEALED_SOURCE_AGGREGATES_BEFORE_LABEL_READING',
        source_manifest=source.record(frozen.OUT / 'manifest.json'), source_exit=manifest['own_exit'],
        sealed_jsonl=prior_report['street_aggregates'], exact_matching_sqlite=sqlite_ref,
        aggregate_rows_checked=checked, labels_read=False, sources_mutated=False))

    # Label-side phase starts only after complete source preflight above.
    names_report = source.prior.read(NAMES / 'report.json')
    statuses_report = source.prior.read(STATUS / 'report.json')
    previous_report = source.prior.read(PREVIOUS / 'report.json')
    previous_exit = source.prior.read(PREVIOUS / 'owned_tool_exit.json')
    if previous_exit['exit_code'] or not previous_exit['own_tool_exit_confirmed']:
        raise ValueError('Existing complete label-side coverage receipt')
    if statuses_report['stock_sites'] != bindings['own_site'] or statuses_report['stock_owners'] != bindings['official_owners']:
        raise ValueError('Same official site and owner diffusion authority')
    labels = {r['query_id']: r for r in source.prior.read(source.verify(names_report['cases']))}
    statuses = {r['qid']: r for r in source.prior.read(source.verify(statuses_report['statuses']))}
    previous = source.prior.read(source.verify(previous_report['all526']))
    panel = {q['query_id']: q for q in source.prior.read(source.verify(pipeline_launch['panel']))}
    if len(previous) != 526 or set(panel) != set(labels) or set(panel) != set(statuses) or {r['query_id'] for r in previous} != set(panel):
        raise ValueError('Exact ALL526 label-side/query scope')
    if any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('Safe folds only, no closed data')
    keys = np.load(bindings['endpoint_keys']['path'], mmap_mode='r')
    xy = np.load(bindings['official_supported_own_xy']['path'], mmap_mode='r')
    public = np.load(bindings['both_public_mask']['path'], mmap_mode='r')
    universe = np.load(bindings['universe']['path'], mmap_mode='r')
    rows = []
    targets = {}
    for old in previous:
        qid = old['query_id']
        s = statuses[qid]
        protected = s['site_diffusion'] != 'O' or s['owner_diffusion'] != 'O'
        if protected != old['protected_target_not_looked_up']:
            raise ValueError('Existing protected label-side status drift')
        row = dict(query_id=qid, fold=panel[qid]['fold'], already_exact=old['already_exact'],
                   protected_not_looked_up=protected, street_cloud_available=False)
        rows.append(row)
        if protected:
            row['scope'] = 'PROTECTED_SKIP_NO_TARGET_LOOKUP'
            continue
        target = int(labels[qid]['label_siret'])
        row['label_siret'] = f'{target:014d}'
        p = int(np.searchsorted(keys, target))
        endpoint = p < len(keys) and keys[p] == target
        row['official_endpoint_present'] = bool(endpoint)
        if not endpoint:
            row['scope'] = 'OFFICIAL_ENDPOINT_ABSENT'
            continue
        if not public[p]:
            raise ValueError('Both-public label status disagrees with immutable source mask')
        point = np.asarray(xy[p], dtype=float)
        supported = bool(np.isfinite(point).all() and 0 <= point[0] <= 1300000 and 6000000 <= point[1] <= 7300000)
        row['existing_supported_own_xy'] = supported
        a = int(np.searchsorted(universe, target))
        inside = a < len(universe) and universe[a] == target
        row['in_original_numeric_universe'] = bool(inside)
        if not inside:
            row['scope'] = 'NOT_IN_ORIGINAL_NUMERIC_REFERENCE_UNIVERSE'
        elif supported:
            row['scope'] = 'OWN_SUPPORTED_XY_PRESENT_NOT_FOOTPRINT_SCOPE'
        else:
            targets[target] = None
    postal = {}
    for c in source.prior.read(bindings['communes']['path']):
        for cp in c['codesPostaux']:
            postal.setdefault(cp, set()).add(c['code'])
    if targets:
        con = duckdb.connect()
        con.execute('SET threads=1')
        con.execute("SET memory_limit='128MB'")
        con.execute('SET temp_directory=?', [str(OUT / 'spill')])
        cols = ['siret', 'statutDiffusionEtablissement', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
                'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
                'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement', 'codePostalEtablissement']
        values = [f'{k:014d}' for k in sorted(targets)]
        sql = 'SELECT ' + ','.join(cols) + ' FROM read_parquet(?) WHERE siret IN (' + ','.join('?' for _ in values) + ')'
        for record in con.execute(sql, [bindings['own_site']['path'], *values]).fetchall():
            obj = {k: v or '' for k, v in zip(cols, record)}
            target = int(obj['siret'])
            if targets[target] is not None or obj['statutDiffusionEtablissement'] != 'O':
                raise ValueError('Public own-source lookup unique and unchanged')
            targets[target] = obj
        con.close()
    for row in rows:
        source.guard()
        if 'scope' in row:
            continue
        target = int(row['label_siret'])
        own = targets[target]
        if own is None:
            raise ValueError('Existing official endpoint missing own primary source record')
        if frozen.explicit_range(own):
            row['scope'] = 'EXPLICIT_RANGE_OR_LIST_REJECTED'
            continue
        if source.prior.number(own['numeroVoieEtablissement'] + ' ' + own['indiceRepetitionEtablissement']):
            row['scope'] = 'SIMPLE_HOUSE_NOT_STREET_SCOPE'
            continue
        street = source.prior.normalize(own['typeVoieEtablissement'] + ' ' + own['libelleVoieEtablissement'], street=True)
        if not street:
            row['scope'] = 'NO_OWN_STREET_KEY'
            continue
        ci = own['codeCommuneEtablissement']
        route = 'OWN_INSEE'
        if not ci:
            codes = postal.get(own['codePostalEtablissement'], set())
            if len(codes) != 1:
                row['scope'] = 'NO_SAFE_OWN_COMMUNE_ROUTE'
                continue
            ci = next(iter(codes))
            route = 'UNIQUE_PUBLISHED_OWN_CP'
        row.update(own_normalized_street_key=street, own_routed_insee=ci, route=route)
        hit = cloud.execute('SELECT source_rows,positions,lat_min,lat_max,lon_min,lon_max,bbox_diagonal_m FROM clouds WHERE insee=? AND street_key=?', (ci, street)).fetchone()
        if hit is None:
            row['scope'] = 'ELIGIBLE_NO_PRIOR_EXACT_OWN_STREET_CLOUD'
        else:
            row.update(scope='ELIGIBLE_PRIOR_EXACT_OWN_STREET_CLOUD', street_cloud_available=True,
                cloud_source_rows=hit[0], cloud_distinct_positions=hit[1],
                cloud_bbox_wgs84=dict(lat_min=hit[2], lat_max=hit[3], lon_min=hit[4], lon_max=hit[5]),
                cloud_bbox_descriptive_diagonal_m=hit[6], cloud_not_site_or_house=True,
                ambiguity_retained_no_point_selected=True)
    cloud.close()
    remaining = [r for r in rows if not r['already_exact']]
    public_remaining = [r for r in remaining if not r['protected_not_looked_up']]
    if len(rows) != 526 or len(remaining) != 20 or len(public_remaining) != 18:
        raise ValueError('Exact ALL526 and remaining20/public18 denominator')
    for ref in [sqlite_ref, prior_report['street_aggregates'], bindings['implementation'], bindings['source_helper']]:
        source.verify(ref)
    write('all526_coverage.json', rows)
    write('remaining20_coverage.json', remaining)
    summary = dict(all526=526, public_targets=sum(not r['protected_not_looked_up'] for r in rows),
        protected_skipped=sum(r['protected_not_looked_up'] for r in rows),
        all526_scope_counts=dict(Counter(r['scope'] for r in rows)),
        public_targets_with_prior_cloud=sum(r['street_cloud_available'] for r in rows),
        public_current_hits_with_prior_cloud=sum(r['street_cloud_available'] and r['already_exact'] for r in rows),
        remaining20=20, remaining_public18=18, remaining_scope_counts=dict(Counter(r['scope'] for r in remaining)),
        remaining_public_with_prior_cloud=sum(r['street_cloud_available'] for r in public_remaining),
        remaining_protected_skipped=len(remaining) - len(public_remaining))
    write('report.json', dict(status='COMPLETE_POSTSEAL_ALL526_OWN_STREET_CLOUD_TARGET_COVERAGE_NOT_RECALL',
        implementation=source.record(Path(__file__)), source_preflight=source.record(OUT / 'source_preflight.json'),
        recipe=source.record(OUT / 'recipe.json'), existing_label_side_names=source.record(NAMES / 'report.json'),
        public_diffusion_status=source.record(STATUS / 'report.json'), previous_coverage=source.record(PREVIOUS / 'report.json'),
        all526=source.record(OUT / 'all526_coverage.json'), remaining20=source.record(OUT / 'remaining20_coverage.json'),
        summary=summary, label_side_never_given_to_source_or_generation=True,
        no_retrieval_fit_coordinate_fill_new_geometry_or_label_change=True,
        no_label_error_or_recall_gain_claim=True, closed_sets_opened=False, observed_unix=time.time(), rss_bytes=source.guard()))
    print(json.dumps(summary), flush=True)


def run():
    child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.embeddinggemma2_v147_target_coverage_diagnostic', 'audit'])
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
    if not OUT.exists():
        raise RuntimeError('No label-side diagnostic namespace')
    write('owned_tool_exit.json', dict(exit_code=child.returncode, own_child_exit_confirmed=True,
                                      owned_pid=child.pid, completed_unix=time.time()))
    if child.returncode:
        raise RuntimeError('Owned label-side audit failed')
    write('manifest.json', dict(status='COMPLETE_SEALED_LABEL_SIDE_OWN_STREET_COVERAGE_NOT_RECALL',
        report=source.record(OUT / 'report.json'), owned_exit=source.record(OUT / 'owned_tool_exit.json'),
        source_preflight=source.record(OUT / 'source_preflight.json'),
        no_source_or_generation_policy_mutation=True, no_recall_gain_or_label_error_claim=True))
    print(json.dumps(source.record(OUT / 'manifest.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'audit'))
    args = parser.parse_args()
    globals()[args.action]()
