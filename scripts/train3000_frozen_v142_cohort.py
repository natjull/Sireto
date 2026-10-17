"""Seal the train3000 cohort; project exclusion metadata, never open reserved files."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ijson
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scripts import train3000_frozen_v142 as preflight
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_compare as old
from scripts.retrieval_site_operational_metric import read, record

OUT = preflight.OUT/'extension_20261017'
SEED = '20261017'


def save(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable cohort checkpoint differs: '+str(path))
        return
    preflight.publish(path, value if isinstance(value, dict) else json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def digest_ids(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()


def projected(path, fields=('query_id',), selected=None, negatives=False, root='item'):
    """Streaming projection; unselected sirens/texts/positives are not collected."""
    rows, row = [], None
    with Path(path).open('rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == root and event == 'start_map':
                row = dict(has_negatives=False) if negatives else {}
            elif prefix == root and event == 'end_map':
                if selected is None or row['query_id'] in selected:
                    rows.append(row)
                row = None
            elif row is not None:
                if prefix == root+'.query_id' and event == 'string':
                    row['query_id'] = value
                elif negatives and prefix == root+'.negatives.item' and event in ('start_map', 'string'):
                    row['has_negatives'] = True
                elif prefix.startswith(root+'.') and '.' not in prefix[len(root)+1:] and event in ('string', 'number', 'boolean', 'null'):
                    field = prefix[len(root)+1:]
                    if field in fields:
                        if selected is not None and 'query_id' not in row:
                            raise ValueError('Selected projection requires query_id before target metadata')
                        if selected is None or row['query_id'] in selected:
                            row[field] = value
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source query')
    return rows


def population():
    refs = []
    def checked(ref):
        if isinstance(ref, Path):
            current = record(ref)
        else:
            current = record(ref['path'], ref['sha256'])
        refs.append(current)
        return Path(current['path'])
    previous = read(preflight.RESULT)
    ledger = checked(previous['ledger'])
    manifest = read(checked(io.HOME/'night_20261006_a/manifest.json'))
    if manifest['ledger_sha256'] != previous['ledger']['sha256'] or manifest['safe_folds'] != [2, 3, 4]:
        raise ValueError('Population authority drift')
    train_path = io.HOME/'night_20261006_a/train_membership.json'
    train_ids = {r['query_id'] for r in projected(checked(dict(path=str(train_path), sha256=manifest['split_hashes']['train'])))}
    census = read(checked(io.HOME/'v10/census_20261008_a/census.json'))
    val_ids = {r['query_id'] for r in projected(checked(census['sources']['panel']))}
    current_path = io.PANEL
    current_ids = {r['query_id'] for r in projected(checked(current_path), root='queries.item')}
    if (len(train_ids), len(val_ids), len(current_ids)) != (15036, 2212, 2000):
        raise ValueError(f'Original non-test split counts drift: train={len(train_ids)}, validation={len(val_ids)}, encoder-excluded={len(current_ids)}')
    if train_ids & val_ids or current_ids & (train_ids | val_ids):
        raise ValueError('Original query partitions overlap')
    table = ds.dataset(ledger)
    folds = ds.field('fold').isin([2, 3, 4])
    all_ids = set(table.to_table(columns=['query_id'], filter=folds).column(0).to_pylist())
    non_test = train_ids | val_ids | current_ids
    test_ids = all_ids - non_test
    if not non_test <= all_ids or len(test_ids) != manifest['rows']['test']:
        raise ValueError('Cannot establish test4323 exclusion by complement')
    frozen = read(preflight.FROZEN)
    inputs = preflight.FrozenInputs(frozen)
    launch = inputs.json(preflight.RUN/'launch.json')
    panel = inputs.json(launch['panel'])
    pilot_ids = {q['query_id'] for q in panel}
    pilot = table.to_table(columns=['query_id', 'human_ground_truth_siren'],
        filter=folds & ds.field('query_id').isin(sorted(pilot_ids))).to_pylist()
    pilot_companies = {r['human_ground_truth_siren'] for r in pilot}
    if len(pilot_ids) != 526 or len(pilot_companies) != 405 or not pilot_ids <= val_ids:
        raise ValueError('Pilot scope drift')
    # Reproduce only the historical checkpoint-control selection, not the
    # reserved962 labels. First pass collects IDs/negative-presence only.
    fit = read(checked(io.HOME/'v6/deadline_recovery_20261008_a/fit/launch.json'))
    data = read(checked(fit['data']))
    config = read(checked(data['config']))
    validation = checked(data['artifacts']['validation.json'])
    meta = projected(validation)
    current_controls = {r['query_id'] for r in sorted(meta,
        key=lambda r: hashlib.sha256(('gemma2-v2-control|'+r['query_id']).encode()).digest())[:config['validation_queries']]}
    ancestor = read(checked(io.HOME/'cached_20261007_a/launch.json'))
    ancestor_path = checked(dict(path=str(Path(ancestor['source'])/'validation.json'), sha256=ancestor['inputs']['validation.json']))
    meta = projected(ancestor_path, negatives=True)
    ancestor_controls = {r['query_id'] for r in sorted((r for r in meta if r['has_negatives']),
        key=lambda r: hashlib.sha256(r['query_id'].encode()).digest())[:64]}
    if len(current_controls) != 256 or len(ancestor_controls) != 64:
        raise ValueError('Historical consumed checkpoint-control counts drift')
    controls = projected(validation, fields=('query_id', 'siren'), selected=current_controls)
    controls += projected(ancestor_path, fields=('query_id', 'siren'), selected=ancestor_controls)
    consumed_companies = pilot_companies | {r['siren'] for r in controls}
    # Push the allowed-company predicate into Arrow. No reserved962 target
    # SIREN/SIRET is returned to this program, and its membership file is not read.
    consumed_val = set(table.to_table(columns=['query_id'], filter=folds &
        ds.field('query_id').isin(sorted(val_ids)) &
        ds.field('human_ground_truth_siren').isin(sorted(consumed_companies))).column(0).to_pylist())
    reserved_control = val_ids - consumed_val
    if len(reserved_control) != 962 or pilot_ids & reserved_control:
        raise ValueError(f'Cannot establish exact control962 exclusion: reconstructed {len(reserved_control)}, expected962')
    permitted = non_test - reserved_control
    rows = table.to_table(columns=['query_id', 'fold', 'insee', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filter=folds & ds.field('query_id').isin(sorted(permitted)) &
        ~ds.field('human_ground_truth_siren').isin(sorted(pilot_companies))).to_pylist()
    if any(len(r['human_ground_truth_siret']) != 14 or not r['human_ground_truth_siret'].isdigit()
           or r['human_ground_truth_siret'][:9] != r['human_ground_truth_siren'] for r in rows):
        raise ValueError('Unusable label: publish missingness before sampling, never silently filter')
    if len(rows) < 3000 or {r['query_id'] for r in rows} & (test_ids | reserved_control | pilot_ids):
        raise ValueError('Insufficient or protected population')
    models = {k: inputs.json(v['v71']) for k, v in previous['models'].items()}
    bases = {k: inputs.json(v['v32']) for k, v in previous['models'].items()}
    seed = frozen['assignment']['group_seed']
    for r in rows:
        siren = r['human_ground_truth_siren']
        k = str(int.from_bytes(hashlib.sha256((seed+'|'+siren).encode()).digest()[:8], 'big') % 5)
        h = hashlib.sha256(siren.encode()).hexdigest()
        if h in models[k]['train_siren_hashes'] or h in bases[k]['train_siren_hashes']:
            raise ValueError('Non-OOF company must be excluded and counted before sampling')
        r['model_id'] = int(k)
    inputs.unchanged()
    audit = dict(status='POPULATION_ALLOWED_BEFORE_STRATIFIED_DRAW', safe_folds=[2, 3, 4],
        total_234=len(all_ids), test4323_excluded=len(test_ids), control962_excluded=len(reserved_control),
        test_id_set_sha256=digest_ids(test_ids), control_id_set_sha256=digest_ids(reserved_control),
        excluded_pilot_companies=len(pilot_companies), excluded_pilot_company_queries=len(permitted)-len(rows),
        eligible_queries=len(rows), eligible_companies=len({r['human_ground_truth_siren'] for r in rows}),
        by_fold=dict(Counter(str(r['fold']) for r in rows)), coefficient_oof_ok=len(rows), coefficient_oof_failed=0,
        by_model=dict(Counter(str(r['model_id']) for r in rows)), full_encoder_oof=False,
        reserved_membership_files_opened=False, reserved_targets_returned=False,
        control_rule='validation minus pilot and ancestor64/current256 checkpoint-control companies',
        test_rule='ledger234 minus original train15036, query-only validation2212 and encoder-excluded2000 IDs',
        exclusion_scope='identities used only for custody, never generation evidence', refs=refs)
    return rows, panel, census, audit


def density(zone, sites):
    return 'ABSENT' if not zone else '1_999' if sites < 1000 else '1000_9999' if sites < 10000 else '10000_PLUS'


def draw(rows, n, seed):
    strata = defaultdict(list)
    for r in rows:
        strata[str(r['fold'])+'|'+r['density']].append(r)
    quotas = {k: n*len(v)//len(rows) for k, v in strata.items()}
    extra = n-sum(quotas.values())
    for k in sorted(strata, key=lambda k: (-(n*len(strata[k]) % len(rows)), k))[:extra]:
        quotas[k] += 1
    chosen = []
    for k, values in sorted(strata.items()):
        chosen += sorted(values, key=lambda r: hashlib.sha256((seed+'|'+k+'|'+r['query_id']).encode()).digest())[:quotas[k]]
    if len(chosen) != n or len({r['query_id'] for r in chosen}) != n:
        raise ValueError('Stratified sample cardinality')
    return chosen, {k: dict(population=len(v), selected=quotas[k]) for k, v in sorted(strata.items())}


def seal():
    started = time.monotonic()
    if (OUT/'sample.json').exists():
        result = read(OUT/'sample.json')
        for ref in [*result['population']['refs'], *result['query_sources']]:
            record(ref['path'], ref['sha256'])
        save(ROOT/'docs/train3000_frozen_v142_sample.json', result)
        return result
    rows, pilot, census, audit = population()
    # Count-only metadata: no document hydration, native scoring or encoder.
    # This bounded lightweight census may coexist with hydraulic; all proof
    # construction/search workers must wait for the VM to stop.
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    backend = TantivyBackend(io.INDEX)
    if backend.searcher.num_docs != 82772691:
        raise ValueError('Frozen corpus changed')
    before = old.inventory()
    counts = {}
    for zone in sorted({r['insee'] or '' for r in rows}):
        counts[zone] = backend.searcher.search(old.zone_query(backend, zone), limit=1, count=True).count if zone else 0
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1024**3:
            raise RuntimeError('STOP_METADATA_RSS_1_GIB: do not compete with hydraulic')
    if old.inventory() != before:
        raise ValueError('Index changed during census')
    del backend
    for r in rows:
        r['insee'] = r['insee'] or ''
        r['indexed_commune_sites'] = counts[r['insee']]
        r['density'] = density(r['insee'], r['indexed_commune_sites'])
    sample, quotas = draw(rows, 3000, SEED)
    smoke, smoke_quotas = draw(sample, 20, SEED+'|smoke')
    pilot_counts = {z['zone']: z['indexed_sites'] for z in census['zones']}
    for q in pilot:
        q['density'] = density(q['crm_insee'], pilot_counts[q['crm_insee']])
    fidelity, fidelity_quotas = draw(pilot, 20, SEED+'|fidelity')
    ids = {r['query_id'] for r in sample}
    # Project exactly the frozen CRM representation after selected-ID/fold
    # predicates, not ledger labels or encoder training examples.
    query_refs, queries = [], {}
    generation = read(io.GEN)
    for batch in generation['batches']:
        for shard in read(batch['path'])['shards']:
            ref = read(shard['path'])['files']['queries.parquet']
            part = pq.read_table(ref['path'], columns=list(old.CRM),
                filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(ids))], use_threads=False).to_pylist()
            if part:
                record(ref['path'], ref['sha256'])
                query_refs.append(ref)
            for q in part:
                if q['query_id'] in queries:
                    raise ValueError('Duplicate selected CRM input')
                queries[q['query_id']] = q
    if set(queries) != ids:
        raise ValueError('Missing original CRM representation for selected query')
    if any(queries[r['query_id']]['crm_insee'] != r['insee'] for r in sample):
        raise ValueError('Ledger commune differs from actual query; do not stratify on wrong input')
    sample_record = dict(status='SEALED_STRATIFIED_3000_NO_CANDIDATES', seed=SEED,
        sampling='PROPORTIONAL_FOLD_X_OFFICIAL_INDEX_COMMUNE_SIZE_LARGEST_REMAINDERS_SHA256_PERMUTATION',
        population=audit, strata=quotas, selected_query_ids=[r['query_id'] for r in sample],
        selected_id_set_sha256=digest_ids(ids), smoke_query_ids=[r['query_id'] for r in smoke],
        fidelity_query_ids=[q['query_id'] for q in fidelity], smoke_strata=smoke_quotas,
        fidelity_strata=fidelity_quotas, elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        query_sources=query_refs, implementation=record(Path(__file__)),
        oof_models=read(preflight.RESULT)['models'],
        no_generation=True, no_training=True, no_label_change=True)
    save(OUT/'population.json', audit)
    save(OUT/'commune_counts.json', counts)
    save(OUT/'label_side_sample.json', dict(rows=sample, not_a_feature_or_constructor_input=True))
    save(OUT/'queries.json', [queries[r['query_id']] for r in sample])
    save(OUT/'assignment.json', {r['query_id']:r['model_id'] for r in sample})
    save(OUT/'pilot_fidelity_queries.json', [{k:q[k] for k in old.CRM} for q in fidelity])
    save(OUT/'sample.json', sample_record)
    doc = ROOT/'docs/train3000_frozen_v142_sample.json'
    save(doc, sample_record)
    return sample_record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.seal:
        result = seal()
    elif (OUT/'sample.json').exists():
        result = read(OUT/'sample.json')
    else:
        result = dict(status='NOT_SEALED', namespace=str(OUT))
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected_query_ids','query_sources','refs')}, indent=2, ensure_ascii=False))
