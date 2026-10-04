"""Compile ALL reverse site addresses from the complete immutable V154 prefix.

Original successful staged/dictionary certificates keep their original launch.
The original stopped reverse stage is retained but never imported. V159 only
changes physical writer fanout; V147 pack and approved V153 limits unchanged.
"""
import argparse
import functools
import json
import os
from pathlib import Path
import sys

import duckdb
import numpy as np
import pyarrow.parquet as pq

import scripts.compile_retrieval_v154_address_store as original
from src.xgb_matcher.retrieval_address_stage_v159 import stage
from src.xgb_matcher.retrieval_address_reverse_v147 import pack
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_memory_policy_v153 import Budget, LIMITS, POLICY, quiet, validate_report
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record, verify, write_json
from src.xgb_matcher.retrieval_supervisor_v157 import supervise

CONFIG = ROOT/'config/retrieval_v160_reverse_resume.json'
GLOBAL = 'SEALED_V160_COMPLETE_ORIGINAL_ADDRESS_DICTIONARY_AND_SITE_REVERSE'
STAGED = 'SEALED_V160_ALL_ORIGINAL_ADDRESS_FACT_SITE_STAGING'
PHASES = ('reverse-stage', 'reverse-pack')


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def config():
    cfg = json.loads(CONFIG.read_text())
    if (cfg['resources'] != LIMITS or duckdb.__version__ != cfg['duckdb_version']
            or np.__version__ != cfg['numpy_version']
            or any(not isinstance(cfg[k], int) or isinstance(cfg[k], bool) or cfg[k] <= 0 for k in (
                'expected_sites', 'expected_source_rows', 'expected_original_observations',
                'expected_lexical_observations', 'expected_aliases', 'expected_facts',
                'dictionary_buckets', 'site_quantum', 'writer_group_parts'))
            or cfg['dictionary_buckets'] > 254 or not 1 <= cfg['writer_group_parts'] <= 16):
        raise ValueError('Pinned whole-source storage policy and unchanged resource limits required')
    return cfg


def prefix(cfg):
    """Independently audit every successful original prefix part, no STOP import."""
    path = verify(cfg['dictionary'])
    if path.name != 'dictionary.json':
        raise ValueError('Exact complete original dictionary authority required')
    root = path.parent
    launch, plan = original.load(root)
    stop = json.loads(verify(cfg['original_stop']).read_text())
    if (Path(cfg['original_stop']['path']).resolve() !=
            (original.directory(root, 'reverse-stage', 0)/'status.json').resolve()
            or stop['status'] != 'STOP'):
        raise ValueError('Actual original reverse-stage STOP must remain pinned')
    d = original.authority(root, 'dictionary', 'SEALED_V154_COMPLETE_ALL_ADDRESS_DICTIONARY')
    s = original.authority(root, 'staged', 'SEALED_V154_ALL_ORIGINAL_ADDRESS_STAGING')
    full = original.projection(plan['source_projection'], launch['config'])
    if (full != {k: plan[k] for k in full} or d['staged'] != record(root/'staged.json')
            or s['source'] != plan['source_projection']
            or full['sites'] != cfg['expected_sites'] or full['source_rows'] != cfg['expected_source_rows']
            or full['observations'] != cfg['expected_original_observations']
            or full['lexical_observations'] != cfg['expected_lexical_observations']
            or d['aliases'] != cfg['expected_aliases'] or d['facts'] != cfg['expected_facts']
            or d['lexical_observations'] != full['lexical_observations']
            or d['original_observations'] != full['observations']
            or launch['config']['dictionary_buckets'] != cfg['dictionary_buckets']
            or launch['config']['site_quantum'] != cfg['site_quantum']):
        raise ValueError('Entire original projection/dictionary/namespace required')
    groups = {str(i): [] for i in range(cfg['dictionary_buckets'])}
    original_obs = lexical = size = 0
    refs = []
    for i, spec in enumerate(plan['parts']):
        m = original.phase_record(root, 'part', i)
        st = m['stats']
        if (m['inputs'] != dict(part=spec) or st['original_observations'] != spec['observations']
                or st['lexical_observations'] != spec['lexical_observations']
                or not st['all_eligible_rows_unique_and_exact']):
            raise ValueError('Every original successful projection part required')
        seen = 0
        for relative, ref in m['artifacts'].items():
            f = verify(ref)
            b = f.parent.name.removeprefix('bucket=')
            if (b not in groups or f.parent.name != 'bucket='+b
                    or not f.resolve().is_relative_to((original.directory(root, 'part', i)/'staging').resolve())):
                raise ValueError('Whole original physical staging paths required')
            n = pq.ParquetFile(f).metadata.num_rows
            groups[b].append(dict(index=i, artifact=ref, rows=n))
            seen += n
        if seen != spec['lexical_observations']:
            raise ValueError('Every original lexical observation retained required')
        original_obs += st['original_observations']
        lexical += seen
        size += m['output_bytes']
        refs.append(record(original.directory(root, 'part', i)/'manifest.json'))
    if (s['buckets'] != groups or s['phase_authorities'] != refs or s['output_bytes'] != size
            or s['observations'] != original_obs or original_obs != full['observations']
            or s['lexical_observations'] != lexical or lexical != full['lexical_observations']):
        raise ValueError('Complete original staged authority, not available successes only')
    aliases = facts = observations = size = 0
    refs = []
    if len(d['buckets']) != cfg['dictionary_buckets']:
        raise ValueError('Every original dictionary bucket required')
    for i, b in enumerate(d['buckets']):
        m = original.phase_record(root, 'dictionary', i)
        st = m['stats']
        if (b['bucket'] != i or b['alias_start'] != aliases or b['artifacts'] != m['artifacts']
                or b['aliases'] != st['addresses'] or b['facts'] != st['facts']
                or b['observations'] != st['observations']
                or st['observations'] != sum(row['rows'] for row in groups[str(i)])
                or m['inputs'] != dict(staged=record(root/'staged.json'), bucket=i)
                or not st['every_observation_linked_once']
                or not st['every_site_address_and_attributes_reconstructed_exactly']):
            raise ValueError('Every original dictionary fact/observation certificate required')
        for name, n in (('addresses.parquet', b['aliases']), ('facts.parquet', b['facts']),
                        ('observation_links.parquet', b['observations'])):
            if pq.ParquetFile(verify(b['artifacts'][name])).metadata.num_rows != n:
                raise ValueError('Every complete dictionary export required')
        aliases += b['aliases']
        facts += b['facts']
        observations += b['observations']
        size += m['output_bytes']
        refs.append(record(original.directory(root, 'dictionary', i)/'manifest.json'))
    if (d['phase_authorities'] != refs or d['aliases'] != aliases or d['facts'] != facts
            or d['lexical_observations'] != observations or d['output_bytes'] != size):
        raise ValueError('Complete original dictionary denominator required')
    return launch, plan, d, s


def source_epochs(cfg, launch, plan, d, s):
    root = Path(cfg['dictionary']['path']).parent
    refs = [cfg['dictionary'], cfg['original_stop'], record(root/'launch.json'), launch['plan'],
            record(root/'staged.json'), plan['source_projection'], plan['catalog']]
    refs.extend(s['phase_authorities'])
    refs.extend(d['phase_authorities'])
    for part in plan['parts']:
        refs.extend([part['source'], part['source_manifest']])
    for b in d['buckets']:
        refs.extend(b['artifacts'].values())
    _, raw = original.projection_backend(plan['source_projection']).load(Path(plan['source_projection']['path']).parent)
    refs.extend(row['reference'] for row in raw['sources'].values())
    unique = {r['path']: r for r in refs}
    return [dict(reference=r, fingerprint=fingerprint(path)) for path, r in sorted(unique.items())]


def init(output):
    cfg = config()
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('CATNAT-only resume root required')
    if output.exists():
        raise FileExistsError('Immutable reverse resume already exists')
    if not Path(cfg['dictionary']['path']).is_file():
        raise ValueError('UNFINISHED_COMPLETE_DICTIONARY_NO_WORKER_LAUNCHED')
    gate = quiet(cfg['initial_quiet_seconds'], cfg['min_system_free_percent'], cfg['max_quiet_wait_seconds'])
    with Budget(LIMITS) as budget:
        old, p, d, s = prefix(cfg)
        epochs = source_epochs(cfg, old, p, d, s)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'preflight.json', gate)
    write_json(output/'plan.json', dict(status='SEALED_V160_COMPLETE_ORIGINAL_PREFIX_RESUME_PLAN',
        dictionary=cfg['dictionary'], original_stop=cfg['original_stop'], original_launch=d['launch'],
        source_projection=p['source_projection'], catalog=p['catalog'], sites=p['sites'], parents=p['parents'],
        aliases=d['aliases'], facts=d['facts'], observations=p['observations'], source_rows=p['source_rows'],
        lexical_observations=p['lexical_observations'], prefix_staged=d['staged'],
        reused_output_bytes=s['output_bytes']+d['output_bytes'], prefix_epochs=epochs,
        prefix_audit_resources=budget.report(), **{k: False for k in PROTECTED}))
    implementation = dict(old['implementation'])
    for path in (Path(__file__), CONFIG, ROOT/'src/xgb_matcher/retrieval_address_stage_v159.py',
                 ROOT/'src/xgb_matcher/retrieval_supervisor_v157.py'):
        implementation[str(path)] = record(path)
    write_json(output/'launch.json', dict(config=cfg, plan=record(output/'plan.json'),
        preflight=record(output/'preflight.json'), implementation=implementation,
        **{k: False for k in PROTECTED}))


def load(output):
    if any(output.glob('seal_*_stop.json')):
        raise ValueError('STOP_SEAL_DO_NOT_RESTART')
    l = json.loads((output/'launch.json').read_text())
    if (l['config'] != config() or any(l.get(k) is not False for k in PROTECTED)
            or not output.resolve().is_relative_to(Path(l['config']['output_root']).resolve())):
        raise ValueError('Protected immutable reverse resume launch required')
    for ref in l['implementation'].values():
        verify(ref)
    p = json.loads(verify(l['plan']).read_text())
    validate_report(p['prefix_audit_resources'])
    if (p['status'] != 'SEALED_V160_COMPLETE_ORIGINAL_PREFIX_RESUME_PLAN'
            or any(p.get(k) is not False for k in PROTECTED)
            or p['dictionary'] != l['config']['dictionary'] or p['original_stop'] != l['config']['original_stop']):
        raise ValueError('Complete original immutable prefix plan required')
    for item in p['prefix_epochs']:
        if fingerprint(item['reference']['path']) != item['fingerprint']:
            raise ValueError('Original full prefix epoch drift')
    return l, p


def directory(output, phase, index):
    return output/phase/f'part{index:05d}'


def maximum(output, phase):
    l, p = load(output)
    if phase == 'reverse-stage':
        return l['config']['dictionary_buckets']
    if phase == 'reverse-pack':
        return (p['sites']+l['config']['site_quantum']-1)//l['config']['site_quantum']
    raise ValueError('Planned full reverse phase required')


def phase_record(output, phase, index):
    dest = directory(output, phase, index)
    if not (dest/'manifest.json').is_file():
        raise ValueError('UNFINISHED_ALL_REVERSE_PHASES')
    if any((dest/name).exists() for name in ('worker_stop.json', 'status.json')):
        raise ValueError('STOP_PHASE_SUPERVISION_OR_WORKER_DO_NOT_RESTART')
    m = json.loads((dest/'manifest.json').read_text())
    validate_report(m['resources'])
    if (m['status'] != 'SEALED_V160_PHYSICAL_'+phase.upper() or m['phase'] != phase or m['index'] != index
            or m['launch'] != record(output/'launch.json') or any(m.get(k) is not False for k in PROTECTED)
            or m['complete_retrieval_index'] is not False):
        raise ValueError('Exact protected complete physical reverse phase required')
    gate = json.loads(verify(m['preflight']).read_text())
    if (verify(m['preflight']).resolve() != (dest/'preflight.json').resolve()
            or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']
            or gate['policy_id'] != POLICY):
        raise ValueError('Explicit unchanged authorized pressure phase gate required')
    for name, ref in m['artifacts'].items():
        if verify(ref).resolve() != (dest/name).resolve():
            raise ValueError('Actual immutable reverse artifact paths required')
    return m


def authority(output, name, status):
    load(output)
    m = json.loads((output/(name+'.json')).read_text())
    validate_report(m['seal_resources'])
    if (m['status'] != status or m['launch'] != record(output/'launch.json')
            or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] is not False):
        raise ValueError('Protected complete original reverse authority required')
    return m


def close_keys(keys):
    if keys is not None:
        for holder in (keys, keys.previous):
            for name in ('site', 'parent', 'extra_site', 'extra_parent', 'new_site', 'new_parent'):
                a = getattr(holder, name, None)
                if hasattr(a, '_mmap'):
                    a._mmap.close()


def worker(output, phase, index):
    l, p = load(output)
    cfg = l['config']
    dest = directory(output, phase, index)
    if (phase not in PHASES or not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < maximum(output, phase)
            or any(f.name not in ('preflight.json', 'worker.log') for f in dest.iterdir())):
        raise ValueError('Exact fresh planned physical phase required, no retry')
    budget = c = keys = None
    try:
        gate = json.loads((dest/'preflight.json').read_text())
        if not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed'] or gate['policy_id'] != POLICY:
            raise ValueError('Explicit unchanged authorized pressure gate required')
        with Budget(LIMITS) as budget:
            c = duckdb.connect()
            c.execute('SET threads=1')
            c.execute('SET memory_limit=?', [cfg['duckdb_memory']])
            c.execute('SET temp_directory=?', [str(dest/'spill')])
            c.execute('SET max_temp_directory_size=?', [cfg['spill_limit']])
            if phase == 'reverse-stage':
                d = json.loads(verify(p['dictionary']).read_text())
                b = d['buckets'][index]
                original.phase_record(Path(p['dictionary']['path']).parent, 'dictionary', index)
                keys = AppendKeys(p['catalog'])
                stats = stage(c, verify(b['artifacts']['facts.parquet']), dest, keys, index, b['alias_start'],
                              b['aliases'], b['facts'], cfg['site_quantum'], cfg['writer_group_parts'])
                inputs = dict(dictionary=p['dictionary'], bucket=b)
                close_keys(keys)
                keys = None
            else:
                s = authority(output, 'reverse_staged', STAGED)
                items = s['parts'][str(index)]
                files = [verify(item['artifact']) for item in items]
                start = index*cfg['site_quantum']
                stop = min(start+cfg['site_quantum'], p['sites'])
                stats = pack(c, files, dest, start, stop, p['sites'], p['aliases'], sum(item['rows'] for item in items))
                inputs = dict(reverse_staged=record(output/'reverse_staged.json'), index=index, start=start, stop=stop)
            files = [Path(f) for f in stats.pop('files', [])]
            files.extend(dest.glob('*.parquet'))
            files.extend(dest.glob('*.npy'))
            artifacts = {str(f.relative_to(dest)): record(f) for f in sorted(set(files))}
            c.close()
            c = None
            load(output)
            size = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
            if size > cfg['max_phase_bytes']:
                raise ValueError('STOP_REVERSE_PHASE_SIZE')
        write_json(dest/'manifest.json', dict(status='SEALED_V160_PHYSICAL_'+phase.upper(), phase=phase, index=index,
            launch=record(output/'launch.json'), preflight=record(dest/'preflight.json'), inputs=inputs, stats=stats,
            artifacts=artifacts, output_bytes=size, resources=budget.report(), complete_retrieval_index=False,
            **{k: False for k in PROTECTED}))
        print(json.dumps(dict(phase=phase, index=index, resources=budget.report())), flush=True)
    except BaseException as exc:
        if c is not None:
            c.close()
        write_json(dest/'worker_stop.json', dict(status='STOP', reason=str(exc),
            resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
            **{k: False for k in PROTECTED}))
        raise
    finally:
        close_keys(keys)


def execute(output, phase, index):
    if (phase not in PHASES or not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < maximum(output, phase)):
        raise ValueError('Planned complete reverse phase required')
    l, _ = load(output)
    dest = directory(output, phase, index)
    dest.mkdir(parents=True, exist_ok=False)
    cfg = l['config']
    try:
        write_json(dest/'preflight.json', quiet(cfg['phase_quiet_seconds'], cfg['min_system_free_percent'], cfg['max_quiet_wait_seconds']))
        with (dest/'worker.log').open('x') as log:
            report = supervise([sys.executable, '-u', '-m', 'scripts.resume_retrieval_v160_reverse_store',
                                'worker', '--output', str(output), '--phase', phase, '--index', str(index)],
                               log, {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        write_json(dest/'supervision.json', report)
        print((dest/'worker.log').read_text()[-2500:], flush=True)
    except BaseException as exc:
        write_json(dest/'status.json', dict(status='STOP', reason=str(exc), **{k: False for k in PROTECTED}))
        raise


def bounded_seal(function):
    @functools.wraps(function)
    def wrapped(output):
        l, _ = load(output)
        cfg = l['config']
        budget = None
        gate = quiet(cfg['phase_quiet_seconds'], cfg['min_system_free_percent'], cfg['max_quiet_wait_seconds'])
        try:
            with Budget(LIMITS) as budget:
                name, data = function(output)
        except RuntimeError as exc:
            write_json(output/('seal_'+function.__name__+'_stop.json'), dict(status='STOP', reason=str(exc),
                resources=budget.report() if budget is not None and hasattr(budget, 'elapsed') else None,
                **{k: False for k in PROTECTED}))
            raise
        path = output/('seal_'+name+'_preflight.json')
        write_json(path, gate)
        write_json(output/(name+'.json'), dict(data, preflight=record(path), seal_resources=budget.report()))
    return wrapped


@bounded_seal
def seal_staging(output):
    l, p = load(output)
    if (output/'reverse_staged.json').exists():
        raise ValueError('Immutable complete reverse staging already sealed')
    d = json.loads(verify(p['dictionary']).read_text())
    groups = {str(i): [] for i in range(maximum(output, 'reverse-pack'))}
    total = size = 0
    refs = []
    for i, b in enumerate(d['buckets']):
        m = phase_record(output, 'reverse-stage', i)
        st = m['stats']
        if (m['inputs'] != dict(dictionary=p['dictionary'], bucket=b) or st['facts'] != b['facts']
                or not st['every_fact_staged_once'] or not st['every_original_site_parent_and_attribute_exact']
                or st['writer_group_parts'] != l['config']['writer_group_parts']
                or st['maximum_physical_writers_per_copy'] > st['writer_group_parts']
                or st['site_or_proof_or_alias_cap'] is not None or st['identifiers_as_features']):
            raise ValueError('Every original dictionary fact staged once unchanged required')
        seen = 0
        for relative, ref in m['artifacts'].items():
            f = verify(ref)
            part = f.parent.name.removeprefix('site_part=')
            if (part not in groups or f.parent.name != 'site_part='+part
                    or not f.resolve().is_relative_to((directory(output, 'reverse-stage', i)/'staging').resolve())):
                raise ValueError('Every known original physical site partition required')
            n = pq.ParquetFile(f).metadata.num_rows
            groups[part].append(dict(index=i, artifact=ref, rows=n))
            seen += n
        if seen != b['facts']:
            raise ValueError('Every original staged fact exactly once required')
        total += seen
        size += m['output_bytes']
        refs.append(record(directory(output, 'reverse-stage', i)/'manifest.json'))
    if total != p['facts'] or len(d['buckets']) != l['config']['dictionary_buckets']:
        raise ValueError('Whole original reverse fact denominator required')
    return 'reverse_staged', dict(status=STAGED, launch=record(output/'launch.json'), source=p['dictionary'],
        parts=groups, aliases=p['aliases'], facts=total, phase_authorities=refs, output_bytes=size,
        complete_retrieval_index=False, **{k: False for k in PROTECTED})


@bounded_seal
def seal(output):
    l, p = load(output)
    if (output/'manifest.json').exists():
        raise ValueError('Immutable complete original reverse store already sealed')
    s = authority(output, 'reverse_staged', STAGED)
    intervals = []
    refs = []
    cursor = facts = 0
    size = p['reused_output_bytes']+s['output_bytes']
    for i in range(maximum(output, 'reverse-pack')):
        m = phase_record(output, 'reverse-pack', i)
        st = m['stats']
        stop = min(cursor+l['config']['site_quantum'], p['sites'])
        if (m['inputs'] != dict(reverse_staged=record(output/'reverse_staged.json'), index=i, start=cursor, stop=stop)
                or st['start'] != cursor or st['stop'] != stop or st['sites'] != p['sites']
                or st['aliases'] != p['aliases'] or st['facts'] != sum(row['rows'] for row in s['parts'][str(i)])
                or not st['every_original_field_and_site_csr_exact'] or not st['all_provenance_keys_preserved']):
            raise ValueError('Every original whole-site interval and fact required')
        intervals.append(st)
        refs.append(record(directory(output, 'reverse-pack', i)/'manifest.json'))
        cursor = stop
        facts += st['facts']
        size += m['output_bytes']
    if cursor != p['sites'] or facts != p['facts'] or size > l['config']['max_total_bytes']:
        raise ValueError('Whole original site/fact/storage denominator required')
    prefix(l['config'])
    load(output)
    return 'manifest', dict(status=GLOBAL, launch=record(output/'launch.json'), source_projection=p['source_projection'],
        catalog=p['catalog'], dictionary=p['dictionary'], prefix_staged=p['prefix_staged'], original_stop=p['original_stop'],
        reverse_staged=record(output/'reverse_staged.json'), intervals=intervals, phase_authorities=refs,
        sites=p['sites'], aliases=p['aliases'], facts=facts, original_observations=p['observations'],
        lexical_observations=p['lexical_observations'], output_bytes=size, every_original_site_and_fact_preserved=True,
        all_original_observations_and_provenance_retained=True, stopped_reverse_phase_imported=False,
        lexical_address_access_built=False, complete_retrieval_index=False, ready_for_fold1=False,
        **{k: False for k in PROTECTED})


def remaining(output):
    for phase, finish, name, status in (('reverse-stage', seal_staging, 'reverse_staged', STAGED),
                                        ('reverse-pack', seal, 'manifest', GLOBAL)):
        for i in range(maximum(output, phase)):
            dest = directory(output, phase, i)
            if dest.exists():
                phase_record(output, phase, i)
            else:
                execute(output, phase, i)
        if not (output/(name+'.json')).exists():
            finish(output)
        else:
            authority(output, name, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['init', 'phase', 'worker', 'remaining', 'seal-staging', 'seal'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--index', type=int)
    a = parser.parse_args()
    if a.mode in ('phase', 'worker'):
        if a.phase is None or a.index is None:
            parser.error('Actual planned reverse phase and index required')
        (execute if a.mode == 'phase' else worker)(a.output, a.phase, a.index)
    else:
        {'init': init, 'remaining': remaining, 'seal-staging': seal_staging, 'seal': seal}[a.mode](a.output)
