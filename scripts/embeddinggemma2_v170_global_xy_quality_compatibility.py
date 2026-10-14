"""Whole-source public XY compatibility aggregates only; no labels or scores."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE = Path('/Volumes/CATNAT_DATA/oubusiness/raw/sirene/geolocalisation_etablissements_2026-07-01.parquet')
RECEIPT = Path('/Volumes/CATNAT_DATA/oubusiness/raw/receipts/sirene_geolocalisation.json')
AUTHORITY = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v142/bano_native_20261013_night/launch.json')
COLS = ['siret', 'x', 'y', 'qualite_xy', 'epsg', 'distance_precision']
DEADLINE = 1791954368
RECIPE = dict(scope='WHOLE_JULY_OFFICIAL_SOURCE_PUBLIC_eO_uO_CANONICAL_BINDING_NO_QUERY_NO_GT',
    source_rows=37575264, canonical_rows=43896818, columns=COLS,
    compatibility='STRICT_EPSG_STRING_2154_AND_EXACT_FLOAT64_EQUALITY_FROM_ORIGINAL_SOURCE_VALUES',
    secondary_diagnostic='STRICT_EPSG2154_CURRENT_FLOAT32_CAST_EQUALS_ORIGINAL_JULY_FLOAT32_NO_EPSILON_NOT_SAME_POINT_PROOF',
    no_tolerance=True, no_coordinate_repair=True, no_quality_transfer=True,
    quality_codes=['11', '12', '21', '22', '33', 'MISSING', 'OTHER_CODE'],
    threads=1, batch_rows=65536, max_rss_bytes=int(2.5*1024**3),
    max_seconds=600, min_free_percent=25, pressure_levels=[0, 1, 2],
    disk_reserve_bytes=40*1024**3, no_spill=True, deadline_unix=DEADLINE,
    labels_read=False, query_inputs_read=False, coordinates_emitted=False,
    score_or_source_modified=False, result='PROVENANCE_COMPATIBILITY_NOT_RETRIEVAL_OR_QC_TRANSFER')


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    pending = p.with_suffix('.pending.json')
    pending.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    os.replace(pending, p)


def main(out):
    started = time.monotonic()
    peak = 0
    process = psutil.Process()
    def guard():
        nonlocal peak
        rss = process.memory_info().rss
        peak = max(peak, rss)
        pressure_text = subprocess.check_output(['/usr/bin/memory_pressure', '-Q'], text=True, timeout=5)
        free = int(pressure_text.split('free percentage:')[1].split('%')[0].strip())
        pressure = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True, timeout=5))
        if (time.time() >= DEADLINE or time.monotonic()-started > 600 or rss > RECIPE['max_rss_bytes']
                or free < 25 or pressure not in (0, 1, 2)
                or shutil.disk_usage('/Volumes/CATNAT_DATA').free < RECIPE['disk_reserve_bytes']):
            raise RuntimeError('STOP source-only resource guard; no partial aggregate promotion')
        return dict(rss_bytes=rss, peak_rss_bytes=peak, free_percent=free, pressure=pressure,
                    elapsed_seconds=time.monotonic()-started)
    def reference(p, expected=None):
        p = Path(p)
        before = p.stat()
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024**2), b''):
                h.update(block)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Input changed while hashing')
        digest = h.hexdigest()
        if expected and digest != expected:
            raise ValueError('Frozen source hash mismatch')
        guard()
        return dict(path=str(p), sha256=digest, size_bytes=after.st_size,
                    mtime_ns=after.st_mtime_ns, inode=after.st_ino)
    if out.exists():
        raise ValueError('Fresh immutable source namespace required')
    out.mkdir(parents=True)
    try:
        guard()
        authority = read(AUTHORITY)
        receipt = read(RECEIPT)
        refs = dict(implementation=reference(Path(__file__)), authority=reference(AUTHORITY), receipt=reference(RECEIPT),
                    source=reference(SOURCE, receipt['sha256']))
        for name in ('keys', 'xy', 'public_mask'):
            refs[name] = reference(authority[name]['path'], authority[name]['sha256'])
        f = pq.ParquetFile(SOURCE)
        if f.metadata.num_rows != RECIPE['source_rows'] or not set(COLS) <= set(f.schema_arrow.names):
            raise ValueError('Frozen whole source schema/row count mismatch')
        if f.schema_arrow.field('x').type != pa.float32() or f.schema_arrow.field('y').type != pa.float32():
            raise ValueError('Q32 diagnostic requires actually stored original float32XY')
        selected_bytes = sum(f.metadata.row_group(i).column(f.schema_arrow.names.index(c)).total_compressed_size
                             for i in range(f.metadata.num_row_groups) for c in COLS)
        write(out/'launch.json', dict(recipe=RECIPE, references=refs, source_semantic_date=receipt['semantic_date'],
             source_publication_date=receipt['publication_date'], selected_compressed_bytes=selected_bytes,
             source_columns={c:str(f.schema_arrow.field(c).type) for c in COLS},
             candidate_quality_not_in_current_xy=True, source_only=True))
        keys = np.load(refs['keys']['path'], mmap_mode='r', allow_pickle=False)
        xy = np.load(refs['xy']['path'], mmap_mode='r', allow_pickle=False)
        public = np.load(refs['public_mask']['path'], mmap_mode='r', allow_pickle=False)
        if len(keys) != RECIPE['canonical_rows'] or keys.dtype != np.uint64 or xy.shape != (len(keys), 2) or xy.dtype != np.float64 or public.shape != keys.shape or public.dtype != bool:
            raise ValueError('Canonical array header mismatch')
        seen = np.zeros(len(keys), dtype=bool)
        counts = Counter()
        quality = defaultdict(Counter)
        quality_q32 = defaultdict(Counter)
        counts_q32 = Counter()
        public_quality_availability = Counter()
        precision = defaultdict(lambda: dict(finite_nonnegative=0, missing_or_negative=0, minimum=None, maximum=None, sum=0.))
        rows = 0
        for batch_no, batch in enumerate(f.iter_batches(batch_size=65536, columns=COLS, use_threads=False)):
            cost = guard()
            ident = batch.column(0)
            if ident.null_count or not bool(pc.all(pc.match_substring_regex(ident, '^[0-9]{14}$')).as_py()):
                raise ValueError('Malformed source key; aggregate audit stops')
            ids = pc.cast(ident, pa.uint64()).to_numpy(zero_copy_only=False)
            pos = np.searchsorted(keys, ids)
            inside = pos < len(keys)
            inside[inside] &= keys[pos[inside]] == ids[inside]
            counts['CANONICAL_KEY_ABSENT'] += int(np.count_nonzero(~inside))
            locations = pos[inside]
            if len(np.unique(locations)) != len(locations) or np.any(seen[locations]):
                raise ValueError('Duplicate source key; no dropped rows')
            seen[locations] = True
            eligible = inside.copy()
            eligible[inside] &= public[pos[inside]]
            counts['CURRENT_NON_PUBLIC_EXCLUDED'] += int(np.count_nonzero(inside & ~eligible))
            selected = np.flatnonzero(eligible)
            if len(selected):
                sub = batch.take(pa.array(selected))
                epsg = pc.equal(sub.column(4), '2154').fill_null(False).to_numpy(zero_copy_only=False)
                counts['SOURCE_OTHER_OR_MISSING_CRS'] += int(np.count_nonzero(~epsg))
                current = np.asarray(xy[pos[selected]], dtype=np.float64)
                x = pc.cast(sub.column(1), pa.float64()).to_numpy(zero_copy_only=False)
                y = pc.cast(sub.column(2), pa.float64()).to_numpy(zero_copy_only=False)
                finite = np.isfinite(x) & np.isfinite(y)
                supported = finite & (x >= 0) & (x <= 1300000) & (y >= 6000000) & (y <= 7300000)
                own_supported = (np.isfinite(current).all(axis=1) & (current[:,0] >= 0) & (current[:,0] <= 1300000)
                                 & (current[:,1] >= 6000000) & (current[:,1] <= 7300000))
                equal = (current[:,0] == x) & (current[:,1] == y)
                category = np.full(len(selected), 'SOURCE_OTHER_OR_MISSING_CRS', dtype='U40')
                category[epsg & ~finite] = 'SOURCE_XY_NONFINITE'
                category[epsg & finite & ~supported] = 'SOURCE_XY_UNSUPPORTED'
                category[epsg & supported & ~own_supported] = 'CURRENT_XY_MISSING_OR_UNSUPPORTED'
                category[epsg & supported & own_supported & ~equal] = 'DIFFERENT_COORDINATES'
                category[epsg & supported & own_supported & equal] = 'EXACT_COORDINATE_MATCH'
                equal_q32 = (current[:,0].astype(np.float32) == x.astype(np.float32)) & (current[:,1].astype(np.float32) == y.astype(np.float32))
                category_q32 = category.copy()
                category_q32[epsg & supported & own_supported & ~equal_q32] = 'Q32_DIFFERENT_COORDINATES'
                category_q32[epsg & supported & own_supported & equal_q32] = 'Q32_STORED_COORDINATE_COMPATIBLE'
                qvals = sub.column(3).to_pylist()
                distances = pc.cast(sub.column(5), pa.float64()).to_numpy(zero_copy_only=False)
                for code in RECIPE['quality_codes']:
                    qm = np.fromiter(((v == code if code not in ('MISSING','OTHER_CODE') else
                        (v is None or v == '' if code == 'MISSING' else v is not None and v != '' and v not in ('11','12','21','22','33')))
                        for v in qvals), dtype=bool, count=len(qvals))
                    mask = qm & epsg
                    public_quality_availability[code] += int(np.count_nonzero(qm))
                    quality[code].update(Counter(category[mask].tolist()))
                    quality_q32[code].update(Counter(category_q32[mask].tolist()))
                    p = precision[code]
                    good = mask & np.isfinite(distances) & (distances >= 0)
                    p['missing_or_negative'] += int(np.count_nonzero(mask & ~good))
                    if np.any(good):
                        values = distances[good]
                        p['finite_nonnegative'] += len(values)
                        low, high = float(values.min()), float(values.max())
                        p['minimum'] = low if p['minimum'] is None else min(p['minimum'], low)
                        p['maximum'] = high if p['maximum'] is None else max(p['maximum'], high)
                        p['sum'] += float(values.sum())
                counts.update(Counter(category[epsg].tolist()))
                counts_q32.update(Counter(category_q32[epsg].tolist()))
            rows += batch.num_rows
            if batch_no % 16 == 0:
                write(out/'state.json', dict(status='SOURCE_ONLY_GLOBAL_COMPATIBILITY_RUNNING', rows=rows, expected_rows=RECIPE['source_rows'], **cost))
        if rows != RECIPE['source_rows'] or sum(counts.values()) != rows:
            raise ValueError('Whole-source exhaustive classification failed')
        canonical_public = int(np.count_nonzero(public))
        canonical_public_seen = int(np.count_nonzero(public & seen))
        for name in ('CANONICAL_KEY_ABSENT', 'CURRENT_NON_PUBLIC_EXCLUDED', 'SOURCE_OTHER_OR_MISSING_CRS'):
            counts_q32[name] = counts[name]
        if sum(counts_q32.values()) != rows:
            raise ValueError('Whole-source exhaustive Q32 diagnostic classification failed')
        for name, ref in refs.items():
            stat = Path(ref['path']).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (ref['size_bytes'], ref['mtime_ns'], ref['inode']):
                raise ValueError('Source binding changed before aggregate seal')
        report = dict(status='COMPLETE_WHOLE_SOURCE_PUBLIC_XY_COMPATIBILITY_AGGREGATES_NO_GT', rows=rows,
            classification=dict(counts), epsg2154_quality_classification={k:dict(v) for k,v in quality.items()},
            q32_diagnostic_classification=dict(counts_q32), q32_epsg2154_quality_classification={k:dict(v) for k,v in quality_q32.items()},
            q32_not_exact_same_point_precision_or_current_site_proof=True,
            public_quality_availability=dict(public_quality_availability),
            distance_precision_stats=dict(precision), canonical_public_rows=canonical_public,
            canonical_public_seen_in_july=canonical_public_seen, canonical_public_absent_from_july=canonical_public-canonical_public_seen,
            all_rows_classified=True, exact_comparison_no_tolerance=True,
            source_original_float32_cast_to_float64=True, no_precision_or_quality_transferred=True,
            no_coordinate_repaired=True, source_only_no_labels_or_query_inputs=True,
            metadata_rechecked_after_full_source=True, **guard())
        write(out/'report.json', report)
        write(out/'manifest.json', dict(status=report['status'], launch=reference(out/'launch.json'), report=reference(out/'report.json'),
            implementation=refs['implementation'], no_score_feature_source_or_label_change=True))
        print(json.dumps({k:report[k] for k in ('status','rows','classification','q32_diagnostic_classification','epsg2154_quality_classification','q32_epsg2154_quality_classification','canonical_public_absent_from_july','peak_rss_bytes','elapsed_seconds')}), flush=True)
    except BaseException as exc:
        write(out/'stop.json', dict(status='STOP_INCOMPLETE_SOURCE_AGGREGATES_NOT_USABLE', reason=str(exc), elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    main(args.output)
