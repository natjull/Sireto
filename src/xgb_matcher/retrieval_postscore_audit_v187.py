"""Full post-score file audit, physically separated from the unchanged score.

Packets are produced by source-bound supervised scorers, not user inputs.
The national caller must verify every producer/receipt actual exit and hash.
No partial partition is publishable as initial100 before ALL group receipts.
"""
import hashlib
import json
from pathlib import Path

from .retrieval_name_pages_v151 import fingerprint
from .retrieval_v20_workbench import verify

GROUPS = ('facts', 'postings', 'context')
STATUS = 'PENDING_V187_FULL_POSTSCORE_SOURCE_HASH'


def canonical(ref):
    if (not isinstance(ref, dict) or set(ref)-{'path', 'sha256', 'size_bytes'}
            or not isinstance(ref.get('path'), str) or not Path(ref['path']).is_absolute()
            or not isinstance(ref.get('sha256'), str) or len(ref['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in ref['sha256'])):
        raise ValueError('Exact absolute source reference and SHA256 required')
    if 'size_bytes' in ref and (type(ref['size_bytes']) is not int or ref['size_bytes'] < 0):
        raise ValueError('Original reference size must be an actual nonnegative integer')
    # Old authorities sometimes omit optional size. Same path/hash still means
    # the SAME bytes; never drop a differing checksum under a path-only dedup.
    return dict(path=ref['path'], sha256=ref['sha256'])


def references(refs):
    result = {}
    sizes = {}
    for original in refs:
        ref = canonical(original)
        path = ref['path']
        if path in result and result[path] != ref:
            raise ValueError('Contradictory source hashes across physical phases')
        if 'size_bytes' in original:
            if path in sizes and sizes[path] != original['size_bytes']:
                raise ValueError('Contradictory source sizes across physical phases')
            sizes[path] = original['size_bytes']
        result[path] = ref
    return [result[k] for k in sorted(result)]


def canonical_epochs(entries):
    result = {}
    for entry in entries:
        if set(entry) != {'reference', 'fingerprint'}:
            raise ValueError('Typed unchanged source epoch required')
        ref = canonical(entry['reference'])
        fp = entry['fingerprint']
        if (not isinstance(fp, (list, tuple)) or len(fp) != 4
                or any(type(v) is not int or v < 0 for v in fp)):
            raise ValueError('Original device/inode/size/mtime epoch required')
        value = dict(reference=ref, fingerprint=list(fp))
        if ref['path'] in result and result[ref['path']] != value:
            raise ValueError('Contradictory hash or epoch across physical phases')
        if 'size_bytes' in entry['reference'] and entry['reference']['size_bytes'] != fp[2]:
            raise ValueError('Original recorded size and epoch must agree')
        result[ref['path']] = value
    return [result[k] for k in sorted(result)]


def check_epochs(entries):
    for entry in entries:
        if list(fingerprint(entry['reference']['path'])) != entry['fingerprint']:
            raise ValueError('Post-score original source epoch drift')


def packet(groups, source_epochs, signature, span, sites, facts):
    if (set(groups) != set(GROUPS) or any(not groups[k] for k in GROUPS)
            or not isinstance(signature, str) or len(signature) != 64
            or any(c not in '0123456789abcdef' for c in signature)
            or type(sites) is not int or sites <= 0 or type(facts) is not int or facts < 0
            or set(span) != {'start', 'stop', 'facts'} or any(type(v) is not int for v in span.values())
            or not 0 <= span['start'] < span['stop'] <= sites or not 0 <= span['facts'] <= facts):
        raise ValueError('Complete physical source audit groups and identical recipe required')
    epochs = canonical_epochs(source_epochs)
    by_path = {e['reference']['path']: e for e in epochs}
    clean = {k: references(groups[k]) for k in GROUPS}
    for refs in clean.values():
        for ref in refs:
            if ref['path'] not in by_path or by_path[ref['path']]['reference'] != ref:
                raise ValueError('Every full-hash target requires its exact original captured epoch')
    check_epochs(epochs)
    return dict(status=STATUS, groups=clean, source_epochs=epochs, signature=signature, span=span.copy(),
                sites=sites, facts=facts, pending_full_hash=True, initial100_emitted=False,
                truth_opened=False, recall_measured=False)


def validate_packet(value):
    rebuilt = packet(value['groups'], value['source_epochs'], value['signature'], value['span'], value['sites'], value['facts'])
    if value != rebuilt:
        raise ValueError('Exact complete pending source-bound audit packet required')
    return rebuilt


def plan(packets, sites, facts):
    if not packets:
        raise ValueError('Every physical score packet required before full audit')
    checked = [validate_packet(p) for p in packets]
    signature = checked[0]['signature']
    cursor = count = 0
    epochs = canonical_epochs([e for p in checked for e in p['source_epochs']])
    for p in checked:
        if (p['sites'] != sites or p['facts'] != facts or p['signature'] != signature
                or p['span']['start'] != cursor):
            raise ValueError('Same entire site/fact source and contiguous once-only physical scores required')
        cursor = p['span']['stop']
        count += p['span']['facts']
    if cursor != sites or count != facts:
        raise ValueError('All original national sites and address facts before source audit required')
    groups = {k: references([r for p in checked for r in p['groups'][k]]) for k in GROUPS}
    check_epochs(epochs)
    return dict(groups=groups, source_epochs=epochs, signature=signature, sites=sites, facts=facts)


def targets_digest(refs):
    return hashlib.sha256(json.dumps(references(refs), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit_group(value, group, progress=None):
    if group not in GROUPS or set(value['groups']) != set(GROUPS) or not value['groups'][group]:
        raise ValueError('Explicit entire original source hash group required')
    check_epochs(value['source_epochs'])
    refs = references(value['groups'][group])
    byte_count = 0
    for i, ref in enumerate(refs):
        path = verify(ref)  # FULL bytes, never stat/epoch substituted for hash.
        byte_count += path.stat().st_size
        if progress is not None and ((i+1) % 1024 == 0 or i+1 == len(refs)):
            progress(dict(group=group, full_files_hashed=i+1, expected_files=len(refs), full_bytes_hashed=byte_count))
    check_epochs(value['source_epochs'])
    return dict(group=group, files=len(refs), full_bytes_hashed=byte_count,
                exact_target_digest=targets_digest(refs), signature=value['signature'],
                every_expected_source_full_hash_verified=True, initial100_emitted=False,
                truth_opened=False, recall_measured=False)


def require_receipts(value, receipts):
    if len(receipts) != len(GROUPS) or {r.get('group') for r in receipts} != set(GROUPS):
        raise ValueError('Every original source audit receipt required; no partial admission')
    for receipt in receipts:
        refs = references(value['groups'][receipt['group']])
        if (receipt.get('signature') != value['signature'] or receipt.get('files') != len(refs)
                or type(receipt.get('files')) is not int
                or receipt.get('exact_target_digest') != targets_digest(refs)
                or receipt.get('every_expected_source_full_hash_verified') is not True
                or receipt.get('initial100_emitted') is not False
                or receipt.get('truth_opened') is not False or receipt.get('recall_measured') is not False
                or type(receipt.get('full_bytes_hashed')) is not int
                or receipt['full_bytes_hashed'] != sum(Path(r['path']).stat().st_size for r in refs)):
            raise ValueError('Entire unchanged source hash target set and successful receipt required')
    check_epochs(value['source_epochs'])
