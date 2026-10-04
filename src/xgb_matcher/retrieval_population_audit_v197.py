"""Whole128 controls and full shared-source audit; no labels or scoring change."""
import hashlib
import json
from pathlib import Path

from . import retrieval_postscore_audit_v187 as audit
from .retrieval_query_population_v192 import batches

MAX_AUDIT_BYTES = 5 * 1024**3
MAX_AUDIT_FILES = 4096


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def positions(batch):
    if type(batch) is not int or not 0 <= batch < 64:
        raise ValueError('Original full128 physical pair0..63 required')
    return list(batches()[batch])


def controls(rows, expected):
    if (not isinstance(rows, list) or [r.get('position') for r in rows] != expected
            or any(set(r) != {'position', 'query_id', 'fold', 'signature'}
                or type(r['position']) is not int or type(r['fold']) is not int or r['fold'] not in (2, 3, 4)
                or not isinstance(r['query_id'], str) or not r['query_id']
                or not isinstance(r['signature'], str) or len(r['signature']) != 64
                or any(c not in '0123456789abcdef' for c in r['signature']) for r in rows)
            or len({r['query_id'] for r in rows}) != len(rows)):
        raise ValueError('Exact ordered OWN query controls and safe recipes required')
    if expected != list(range(128)) and expected not in [positions(i) for i in range(64)]:
        raise ValueError('Entire128 or an exact frozen physical pair only')
    return [r.copy() for r in rows]


def external(refs, output):
    root = Path(output).resolve()
    return audit.references(r for r in refs if not Path(r['path']).resolve().is_relative_to(root))


def capsule(groups, epochs, output):
    if set(groups) != set(audit.GROUPS):
        raise ValueError('Every original full source group required')
    refs = external([r for g in audit.GROUPS for r in groups[g]], output)
    by_path = {e['reference']['path']: e for e in audit.canonical_epochs(epochs)}
    captured = [by_path[r['path']] for r in refs]
    if any(e['reference'] != r for e, r in zip(captured, refs)):
        raise ValueError('Original captured epochs for every exact shared source required')
    audit.check_epochs(captured)
    return dict(status='PENDING_V197_ALL_SHARED_SOURCE_FULL_HASH', references=refs,
        source_epochs=captured, targets_digest=audit.targets_digest(refs), epochs_digest=digest(captured),
        pending_full_hash=True, initial100_emitted=False, truth_opened=False, recall_measured=False)


def validate_capsule(value):
    refs = audit.references(value['references'])
    epochs = audit.canonical_epochs(value['source_epochs'])
    if (not refs or value['status'] != 'PENDING_V197_ALL_SHARED_SOURCE_FULL_HASH'
            or refs != value['references'] or epochs != value['source_epochs']
            or {r['path']: r for r in refs} != {e['reference']['path']: e['reference'] for e in epochs}
            or value['targets_digest'] != audit.targets_digest(refs) or value['epochs_digest'] != digest(epochs)
            or value['pending_full_hash'] is not True or value['initial100_emitted'] is not False
            or value['truth_opened'] is not False or value['recall_measured'] is not False):
        raise ValueError('Exact immutable complete source capsule, still quarantined, required')
    audit.check_epochs(epochs)
    return value


def require_sources(value, refs, epochs, output, *, exact=False):
    validate_capsule(value)
    own = external(refs, output)
    original = {e['reference']['path']: e for e in audit.canonical_epochs(epochs)}
    baseline = {e['reference']['path']: e for e in value['source_epochs']}
    if exact and own != value['references']:
        raise ValueError('ALL identical external source targets for every physical scorer required')
    for ref in own:
        p = ref['path']
        if p not in baseline or baseline[p]['reference'] != ref or original.get(p) != baseline[p]:
            raise ValueError('Source subset must match exact original hashes AND pre-read epochs')
    audit.check_epochs([original[r['path']] for r in own])
    return dict(shared_source_targets_digest=value['targets_digest'],
        shared_source_epochs_digest=value['epochs_digest'], all_own_source_references_checked=True,
        exact_full_source_set_checked=exact)


def chunks(refs, max_bytes=MAX_AUDIT_BYTES, max_files=MAX_AUDIT_FILES):
    if (type(max_bytes) is not int or not 0 < max_bytes <= MAX_AUDIT_BYTES
            or type(max_files) is not int or not 0 < max_files <= MAX_AUDIT_FILES):
        raise ValueError('Fixed positive physical audit bounds required')
    result, current, size = [], [], 0
    for ref in audit.references(refs):
        n = Path(ref['path']).stat().st_size
        if n > max_bytes:
            raise ValueError('STOP: single full file exceeds declared audit chunk budget')
        if current and (size+n > max_bytes or len(current) == max_files):
            result.append(current); current, size = [], 0
        current.append(ref); size += n
    if current: result.append(current)
    if not result: raise ValueError('No empty or sampled full-source audit')
    return result


def require_receipts(plan, receipts):
    if len(receipts) != len(plan['chunks']):
        raise ValueError('Every FULL byte-audit receipt required before ANY initial100')
    for i, (refs, r) in enumerate(zip(plan['chunks'], receipts)):
        if (r['chunk'] != i or type(r['chunk']) is not int or r['signature'] != plan['signature']
                or r['files'] != len(refs) or type(r['files']) is not int
                or r['exact_target_digest'] != audit.targets_digest(refs)
                or r['full_bytes_hashed'] != sum(Path(p['path']).stat().st_size for p in refs)
                or type(r['full_bytes_hashed']) is not int
                or r['every_expected_source_full_hash_verified'] is not True
                or r['truth_opened'] is not False or r['recall_measured'] is not False
                or r['initial100_emitted'] is not False):
            raise ValueError('Exact complete producer-bound FULL-hash receipt required')


def outputs(rows, population, expected_controls, sites, facts, partitions):
    controls(expected_controls, list(range(128)))
    if not isinstance(rows, list) or [r.get('position') for r in rows] != list(range(128)):
        raise ValueError('ALL128 ordered outputs, never a partial denominator')
    for row, control in zip(rows, expected_controls):
        source, unused = population.select(row['position'])
        t, candidates = row['trace'], row['candidates']
        if (any(row[k] != control[k] for k in ('position', 'query_id', 'fold'))
                or row['query_id'] != source['query_id'] or row['fold'] != source['fold']
                or t['signature'] != control['signature'] or t['sites_completed'] != sites
                or t['address_facts'] != facts or t['physical_partitions'] != partitions
                or t['initial_cap'] != 100 or t['fixed_reservoir'] is not None
                or t['score_after_fixed_pool'] is not False or t['every_site_scored_once'] is not True
                or t['same_original_score_never_rescored'] is not True
                or len(candidates) > 100 or len({v['siret'] for v in candidates}) != len(candidates)):
            raise ValueError('Own original query/score and whole official source with absolute100 required')
    return rows
