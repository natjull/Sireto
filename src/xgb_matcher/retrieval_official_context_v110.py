"""Lossless official site context aligned to the entire V96 national registry.

No CRM/label access, normalization, feature inference, or corpus selection.
Identifiers are typed join keys only. Date values are observations, not proof
of historical validity, and raw activity codes have no inferred taxonomy.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPECS = {'activity_codes': np.uint16, 'state_codes': np.uint16,
         'headquarters_codes': np.uint8, 'creation_days': np.int32,
         'current_period_start_days': np.int32}
DATE_NULL = np.iinfo(np.int32).min
COLUMNS = ('siret', 'siren', 'main_activity', 'administrative_state',
           'is_headquarters', 'creation_date', 'current_period_start')


def codebook(values):
    """Reserve distinct NULL and empty codes; preserve every other raw string."""
    result = [None, ''] + sorted(v for v in values if v is not None and v != '')
    if len(result) > 2**16:
        raise ValueError('Raw context vocabulary exceeds uint16; no truncation')
    return result


def encode(values, book):
    index = pc.index_in(values, value_set=pa.array(book, type=pa.string()))
    if index.null_count:
        raise ValueError('Raw context codebook drift; no silent UNKNOWN mapping')
    return pc.cast(index, pa.uint16()).to_numpy(zero_copy_only=False)


def coarse_state(values):
    return pc.if_else(pc.or_(pc.is_null(values), pc.fill_null(pc.equal(values, ''), False)), 0,
        pc.if_else(pc.fill_null(pc.equal(values, 'A'), False), 1,
        pc.if_else(pc.fill_null(pc.equal(values, 'F'), False), 2, 3))).to_numpy(zero_copy_only=False)


def join_keys(values, length, dtype):
    if not pc.all(pc.fill_null(pc.match_substring_regex(values, f'^[0-9]{{{length}}}$'), False)).as_py():
        raise ValueError('Invalid ASCII official join key')
    return pc.cast(values, pa.from_numpy_dtype(dtype)).to_numpy(zero_copy_only=False)


def positions(keys, parents, raw_keys, raw_parents, seen):
    """Vector search in uint64 avoids implicit full-array dtype conversion."""
    if raw_keys.dtype != np.uint64 or raw_parents.dtype != np.uint32:
        raise ValueError('Join key dtype drift')
    pos = np.searchsorted(keys, raw_keys)
    if np.any(pos >= len(keys)) or not np.array_equal(keys[pos], raw_keys):
        raise ValueError('Official site absent from complete registry')
    if not np.array_equal(parents[pos], raw_parents) or np.any(raw_keys // 100000 != raw_parents):
        raise ValueError('Official site parent drift')
    if np.any(seen[pos]) or len(np.unique(pos)) != len(pos):
        raise ValueError('Duplicate official site context')
    seen[pos] = True
    return pos


def build(source, registry, output, progress=lambda row: None, batch_size=131072):
    file = pq.ParquetFile(source); output = Path(output); registry = Path(registry)
    n = file.metadata.num_rows
    if not 0 < n < 2**32:
        raise ValueError('National context cardinality drift')
    types = {**{k: pa.string() for k in COLUMNS[:4]}, 'is_headquarters': pa.bool_(),
             'creation_date': pa.date32(), 'current_period_start': pa.date32()}
    if any(file.schema_arrow.field(k).type != t for k, t in types.items()):
        raise ValueError('Official context source schema drift')
    keys = np.load(registry/'endpoint_keys.npy', mmap_mode='r', allow_pickle=False)
    parents = np.load(registry/'endpoint_parents.npy', mmap_mode='r', allow_pickle=False)
    states = np.load(registry/'endpoint_states.npy', mmap_mode='r', allow_pickle=False)
    for a, dtype in ((keys, np.uint64), (parents, np.uint32), (states, np.uint8)):
        if a.dtype != dtype or a.shape != (n,):
            raise ValueError('Full official registry dtype/cardinality drift')
    for lo in range(0, n, batch_size):
        hi = min(n, lo + batch_size)
        if (np.any(keys[lo:hi] >= 10**14) or
            np.any(keys[lo:hi] // 100000 != parents[lo:hi]) or
            np.any(keys[max(1, lo):hi] <= keys[max(1, lo)-1:hi-1])):
            raise ValueError('Registry order or membership drift')
    unique = {'main_activity': set(), 'administrative_state': set()}
    for batch in file.iter_batches(batch_size=batch_size, columns=list(unique), use_threads=False):
        for i, k in enumerate(unique):
            unique[k].update(pc.unique(batch.column(i)).to_pylist())
    books = {k: codebook(v) for k, v in unique.items()}
    progress(dict(phase='complete_raw_codebooks',sites=n,activity_codes=len(books['main_activity']),
                  state_codes=len(books['administrative_state'])))
    arrays = {k: np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(n,))
              for k,d in SPECS.items()}
    seen = np.zeros(n, dtype=np.bool_); count = 0
    counts = {'main_activity': np.zeros(len(books['main_activity']),dtype=np.int64),
              'administrative_state': np.zeros(len(books['administrative_state']),dtype=np.int64),
              'headquarters': np.zeros(3,dtype=np.int64)}
    missing_dates = {'creation_days': 0, 'current_period_start_days': 0}
    for batch in file.iter_batches(batch_size=batch_size,columns=list(COLUMNS),use_threads=False):
        vals = dict(zip(COLUMNS,batch.columns))
        raw_keys = join_keys(vals['siret'],14,np.uint64)
        raw_parents = join_keys(vals['siren'],9,np.uint32)
        pos = positions(keys,parents,raw_keys,raw_parents,seen)
        if not np.array_equal(states[pos],coarse_state(vals['administrative_state'])):
            raise ValueError('Official state differs from sealed national registry')
        for raw,dest in (('main_activity','activity_codes'),('administrative_state','state_codes')):
            coded = encode(vals[raw],books[raw]); arrays[dest][pos] = coded
            restored = pc.take(pa.array(books[raw],type=pa.string()),pa.array(arrays[dest][pos]))
            if not restored.equals(vals[raw]):
                raise ValueError('Persisted raw string roundtrip drift')
            counts[raw] += np.bincount(coded,minlength=len(books[raw]))
        hq = pc.if_else(pc.is_null(vals['is_headquarters']),0,
                       pc.if_else(pc.fill_null(vals['is_headquarters'],False),2,1))
        hq = hq.to_numpy(zero_copy_only=False); arrays['headquarters_codes'][pos] = hq
        restored = pc.take(pa.array([None,False,True],type=pa.bool_()),pa.array(arrays['headquarters_codes'][pos]))
        if not restored.equals(vals['is_headquarters']):
            raise ValueError('Persisted headquarters roundtrip drift')
        counts['headquarters'] += np.bincount(hq,minlength=3)
        for raw,dest in (('creation_date','creation_days'),('current_period_start','current_period_start_days')):
            days = pc.cast(vals[raw],pa.int32())
            if pc.any(pc.fill_null(pc.equal(days,int(DATE_NULL)),False)).as_py():
                raise ValueError('Date sentinel collision')
            arrays[dest][pos] = pc.fill_null(days,int(DATE_NULL)).to_numpy(zero_copy_only=False)
            saved = pa.array(arrays[dest][pos],type=pa.int32())
            restored = pc.if_else(pc.equal(saved,int(DATE_NULL)),pa.scalar(None,type=pa.int32()),saved)
            if not restored.equals(days):
                raise ValueError('Persisted date roundtrip drift')
            missing_dates[dest] += days.null_count
        count += len(batch)
        if count % (batch_size*16) == 0:
            progress(dict(phase='context_aligned_by_official_key',sites=count,total=n))
    if count != n or not np.all(seen):
        raise ValueError('Incomplete official context; no site removal permitted')
    for array in arrays.values(): array.flush()
    return dict(sites=n,codebooks=books,counts={k:v.tolist() for k,v in counts.items()},
                missing_dates=missing_dates,null_date_sentinel=int(DATE_NULL),
                headquarters_codes={'UNKNOWN':0,'FALSE':1,'TRUE':2},closed_sites_removed=False,
                null_and_empty_distinct=True,activity_taxonomy_inferred=False,
                identifiers_as_features=False,dates_are_historical_validity=False,
                context_serving_integrated=False,full_registry_exact_alignment=True,
                persisted_context_roundtrip_rows=count)


class OfficialContext:
    """Typed lookup by sealed national site POSITION, never an identifier feature."""
    def __init__(self,directory,stats):
        self.stats = stats
        self.arrays = {k:np.load(Path(directory)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in SPECS}
        for k,dtype in SPECS.items():
            if self.arrays[k].dtype != dtype or self.arrays[k].shape != (stats['sites'],):
                raise ValueError('Context array dtype or cardinality drift')

    def at(self,position):
        if isinstance(position,(bool,np.bool_)) or not isinstance(position,(int,np.integer)) or not 0<=position<self.stats['sites']:
            raise ValueError('Invalid national site position')
        books = self.stats['codebooks']
        result = {raw:books[raw][int(self.arrays[dest][position])] for raw,dest in
                  (('main_activity','activity_codes'),('administrative_state','state_codes'))}
        hq = int(self.arrays['headquarters_codes'][position])
        if hq not in (0,1,2): raise ValueError('Headquarters code drift')
        result['is_headquarters'] = (None,False,True)[hq]
        for k in ('creation_days','current_period_start_days'):
            value = int(self.arrays[k][position]); result[k] = None if value==DATE_NULL else value
        return result
