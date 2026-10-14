"""Read-only ALL526 candidate OWN-address BANO feasibility, never recall.

No labels, outcomes, CRM addresses, retrieval execution, fits or source writes.
Numeric reference keys are deduplicated before any source coverage counts.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import psutil
import pyarrow.parquet as pq

from scripts.embeddinggemma2_v17_structural_retrieval import normalize

ROOT = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
V143 = ROOT / 'v143/local_evidence_empty_recovery_20261013_night'
BANO = ROOT / 'v141/bano_osm_20261013_night_d'
OUT = ROOT / 'v144/candidate_own_address_feasibility_20261014_night'
DEADLINE = 1791954368
MAX_RSS = 512 * 1024**2


def read(path):
    return json.loads(Path(path).read_text())


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def guard():
    if time.time() >= DEADLINE:
        raise TimeoutError('Absolute night deadline')
    rss = psutil.Process().memory_info().rss
    if rss > MAX_RSS:
        raise MemoryError(f'Read-only analytical RSS cap: {rss}')
    return rss


def number(value):
    match = re.fullmatch(r'\s*(\d+)\s*(BIS|TER|QUATER|[A-Z])?\s*', normalize(value))
    return (int(match[1]), match[2] or '') if match and int(match[1]) > 0 else None


def main():
    started = time.time()
    if OUT.exists():
        raise ValueError('Fresh isolated analytical namespace required')
    OUT.mkdir(parents=True)
    launch = read(V143 / 'launch.json')
    fg = read(V143 / 'features_generation.json')
    bg = read(BANO / 'generation.json')
    bl = read(BANO / 'launch.json')
    public = read(V143 / 'public_source.json')
    if len(fg['features']) != 526 or public['site_source'] != launch['stock_source'] or public['owner_source'] != launch['public_owners']:
        raise ValueError('Exact ALL526 and public source bindings')
    if bg['all_queries'] != 526 or bl['retained_source_codes'] != ['OSM']:
        raise ValueError('Complete explicit OSM-only published address source')
    inputs = dict(v143_launch=str(V143 / 'launch.json'), v143_features=str(V143 / 'features_generation.json'),
                  keys=launch['keys'], xy=launch['xy'], own_site=launch['stock_source'],
                  public_owners=launch['public_owners'], both_public_mask=public['public_mask'],
                  bano_index=bg['index'], bano_launch=bg['launch'],
                  commune_reference=bl['commune_reference'], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write('launch.json', dict(status='QUERY_ONLY_OWN_ADDRESS_FEASIBILITY', inputs=inputs,
          labels_read=False, outcomes_read=False, crm_address_used=False, sources_mutated=False,
          threads=1, max_rss_bytes=MAX_RSS, deadline_unix=DEADLINE,
          source_suffix_aliases_added=False, candidate_range_end_rejected=True,
          cp_fallback='ONLY_ONE_PUBLISHED_CURRENT_COMMUNE_WHEN_OWN_INSEE_ABSENT',
          source_address_timestamp_and_owner_identity_certified=False,
          large_input_hashes='INHERITED_BINDINGS_RECORDED_NOT_REHASHED', started_unix=started))
    scratch = sqlite3.connect(OUT / 'deduplicated_refs.sqlite')
    scratch.execute('PRAGMA cache_size=-16384')
    scratch.execute('PRAGMA temp_store=FILE')
    scratch.execute('CREATE TABLE refs(key INTEGER PRIMARY KEY)')
    raw_rows = 0
    empty_queries = 0
    for index, ref in enumerate(fg['features'].values(), 1):
        guard()
        with np.load(ref['path'], allow_pickle=False) as z:
            keys = z['keys']
            raw_rows += len(keys)
            empty_queries += not bool(len(keys))
            for first in range(0, len(keys), 8192):
                scratch.executemany('INSERT OR IGNORE INTO refs VALUES (?)', ((int(k),) for k in keys[first:first + 8192]))
            del keys
        scratch.commit()
        if index % 50 == 0:
            write('state.json', dict(phase='DEDUPLICATING_ALL526_REFERENCES', queries=index,
                  raw_rows=raw_rows, rss_bytes=guard(), elapsed_s=time.time() - started))
            print('DEDUP', index, raw_rows, flush=True)
    count = scratch.execute('SELECT count(*) FROM refs').fetchone()[0]
    refs = np.lib.format.open_memmap(OUT / 'deduplicated_refs.npy', mode='w+', dtype=np.uint64, shape=(count,))
    offset = 0
    cur = scratch.execute('SELECT key FROM refs ORDER BY key')
    while rows := cur.fetchmany(8192):
        refs[offset:offset + len(rows)] = [r[0] for r in rows]
        offset += len(rows)
        guard()
    refs.flush()
    scratch.close()
    # Release source-reference mmap pages periodically; total working set stays bounded.
    own_xy = np.zeros(count, dtype=np.uint8)
    both_public = np.zeros(count, dtype=np.bool_)
    endpoint_seen = np.zeros(count, dtype=np.bool_)
    all_keys = np.load(launch['keys']['path'], mmap_mode='r')
    all_xy = np.load(launch['xy']['path'], mmap_mode='r')
    all_public = np.load(public['public_mask']['path'], mmap_mode='r')
    total = len(all_keys)
    for first in range(0, total, 262144):
        end = min(total, first + 262144)
        kk = np.array(all_keys[first:end])
        positions = np.searchsorted(refs, kk)
        valid = positions < count
        valid[valid] &= refs[positions[valid]] == kk[valid]
        take = np.flatnonzero(valid)
        target = positions[take]
        coords = np.asarray(all_xy[first:end])[take]
        finite = np.isfinite(coords).all(axis=1)
        supported = finite & (coords[:, 0] >= 0) & (coords[:, 0] <= 1300000) & (coords[:, 1] >= 6000000) & (coords[:, 1] <= 7300000)
        own_xy[target] = finite.astype(np.uint8) + supported.astype(np.uint8)
        both_public[target] = all_public[first:end][take]
        endpoint_seen[target] = True
        if first % (262144 * 8) == 0:
            for mm in (all_keys, all_xy, all_public):
                mm._mmap.close()
            all_keys = np.load(launch['keys']['path'], mmap_mode='r')
            all_xy = np.load(launch['xy']['path'], mmap_mode='r')
            all_public = np.load(public['public_mask']['path'], mmap_mode='r')
        guard()
    for mm in (all_keys, all_xy, all_public):
        mm._mmap.close()
    counters = Counter(unique_reference_sites=count, raw_query_reference_rows=raw_rows,
          empty_numeric_queries=empty_queries, all_queries=526,
          absent_endpoint=int((~endpoint_seen).sum()), own_xy_missing=int((own_xy == 0).sum()),
          own_xy_nonmissing=int((own_xy > 0).sum()), own_xy_l93_supported=int((own_xy == 2).sum()),
          both_public=int(both_public.sum()))
    write('source_preflight.json', dict(status='PASS_DEDUPLICATED_UNIVERSE_AND_PUBLIC_SOURCE_JOIN',
          counters=dict(counters), bano_osm_rows=bg['counters']['retained_OSM_rows'],
          bano_quarantined_records=bg['counters']['quarantined_nonempty_records'],
          large_input_hashes='INHERITED_BINDINGS_RECORDED_NOT_REHASHED', labels_read=False))
    print('SOURCE_PREFLIGHT', dict(counters), flush=True)
    postal = {}
    for row in read(bl['commune_reference']['path']):
        for cp in row['codesPostaux']:
            postal.setdefault(cp, set()).add(row['code'])
    bano = sqlite3.connect(f"file:{bg['index']['path']}?mode=ro", uri=True)
    bano.execute('PRAGMA query_only=ON')
    bano.execute('PRAGMA cache_size=-16384')

    @lru_cache(maxsize=8192)
    def exact(ci, street, ni, suffix):
        rows = list(bano.execute('SELECT lat,lon FROM addresses WHERE insee=? AND street_key=? AND number_integer=? AND suffix=?', (ci, street, ni, suffix)))
        positions = set(rows)
        compatible = {(lat, lon) for lat, lon in positions if -6 <= lon <= 10 and 41 <= lat <= 52}
        return len(rows), len(positions), len(compatible)

    cols = ['siret', 'statutDiffusionEtablissement', 'numeroVoieEtablissement', 'indiceRepetitionEtablissement',
            'dernierNumeroVoieEtablissement', 'indiceRepetitionDernierNumeroVoieEtablissement',
            'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement', 'codePostalEtablissement']
    seen = np.zeros(count, dtype=np.bool_)
    for batch_index, batch in enumerate(pq.ParquetFile(launch['stock_source']['path']).iter_batches(batch_size=8192, columns=cols, use_threads=False), 1):
        guard()
        nums = np.asarray(batch.column(0).to_pylist(), dtype=np.uint64)
        at = np.searchsorted(refs, nums)
        valid = at < count
        valid[valid] &= refs[at[valid]] == nums[valid]
        for i in np.flatnonzero(valid):
            p = int(at[i])
            if seen[p]:
                raise ValueError('Duplicate own SIRENE source site')
            seen[p] = True
            counters['own_source_present'] += 1
            row = {name: batch.column(j)[int(i)].as_py() or '' for j, name in enumerate(cols)}
            state = 'xy_missing' if own_xy[p] == 0 else 'xy_nonmissing'
            counters[state + '_own_source_present'] += 1
            if row['statutDiffusionEtablissement'] != 'O' or not both_public[p]:
                counters[state + '_excluded_not_both_public'] += 1
                continue
            counters[state + '_both_public'] += 1
            if row['dernierNumeroVoieEtablissement'] or row['indiceRepetitionDernierNumeroVoieEtablissement']:
                counters[state + '_explicit_range_excluded'] += 1
                continue
            house = number(row['numeroVoieEtablissement'] + ' ' + row['indiceRepetitionEtablissement'])
            if not house:
                counters[state + '_no_single_number'] += 1
                continue
            street = normalize(row['typeVoieEtablissement'] + ' ' + row['libelleVoieEtablissement'], street=True)
            if not street:
                counters[state + '_no_own_street'] += 1
                continue
            ci = row['codeCommuneEtablissement']
            if ci:
                route = 'own_insee'
            else:
                codes = postal.get(row['codePostalEtablissement'], set())
                if len(codes) != 1:
                    counters[state + '_no_safe_commune_route'] += 1
                    continue
                ci = next(iter(codes))
                route = 'unique_published_own_cp'
            counters[state + '_eligible_exact_own_address'] += 1
            nr, np_, nl = exact(ci, street, *house)
            if not nr:
                counters[state + '_no_exact_osm_point'] += 1
                continue
            counters[state + '_has_exact_osm_point'] += 1
            counters[state + '_match_route_' + route] += 1
            counters[state + '_exact_source_rows'] += nr
            counters[state + '_distinct_source_positions'] += np_
            counters[state + '_ambiguous_multiple_positions'] += np_ > 1
            counters[state + '_unique_published_position'] += np_ == 1
            counters[state + '_has_l93_compatible_exact_point'] += nl > 0
            counters[state + '_unique_l93_compatible_position'] += nl == 1 and np_ == 1
            counters[state + '_has_unsupported_l93_point'] += nl < np_
        if batch_index % 100 == 0:
            write('state.json', dict(phase='READ_ONLY_EXACT_OWN_ADDRESS_JOIN', scanned_source_rows=batch_index * 8192,
                  counters=dict(counters), rss_bytes=guard(), elapsed_s=time.time() - started))
            print('OWN_JOIN', batch_index * 8192, counters['own_source_present'], counters['xy_missing_has_exact_osm_point'], flush=True)
    bano.close()
    counters['own_source_absent'] = int((~seen).sum())
    write('feasibility.json', dict(status='COMPLETE_ALL526_DEDUPLICATED_CANDIDATE_OWN_ADDRESS_QUERY_ONLY_FEASIBILITY',
          counters=dict(counters), inputs=inputs, no_labels_or_outcomes_read=True,
          no_candidate_coordinate_written=True, source_osm_current_community_diagnostic_only=True,
          no_recall_gain_or_identity_timestamp_certification_claim=True,
          ambiguous_positions_retained_not_selected=True, rss_bytes=guard(),
          elapsed_s=time.time() - started, completed_unix=time.time()))
    print('COMPLETE', dict(counters), flush=True)


if __name__ == '__main__':
    main()
