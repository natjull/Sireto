"""Two own-query recipes, shared complete physical sources and full audits.

No scoring change, source subset, reranker or metric. National callers must
verify actual producers/parents, every full receipt and postmerge ledger.
"""
import gc
import hashlib
import json

from . import retrieval_postscore_audit_v187 as audit
from .retrieval_v20_workbench import verify

STATUS = 'PENDING_V195_COMPLETE_PAIR_SOURCE_HASH'


def records(rows):
    if (not isinstance(rows, list) or len(rows) != 2
            or [r.get('position') for r in rows] != [0, 1]
            or any(set(r) != {'position', 'query_id', 'fold', 'signature'}
                   or type(r['position']) is not int or type(r['fold']) is not int
                   or r['fold'] not in (2, 3, 4) or not isinstance(r['query_id'], str)
                   or not r['query_id'] or not isinstance(r['signature'], str)
                   or len(r['signature']) != 64 or any(c not in '0123456789abcdef' for c in r['signature'])
                   for r in rows)
            or len({r['query_id'] for r in rows}) != 2):
        raise ValueError('Two exact own-query controls and independent safe recipes required')
    return [r.copy() for r in rows]


def signature(rows):
    return hashlib.sha256(json.dumps(records(rows), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def groups(refs):
    if set(refs) != set(audit.GROUPS) or any(not refs[k] for k in audit.GROUPS):
        raise ValueError('All original fact/posting/context sources required')
    # This is physical audit grouping, not feature/candidate filtering. Every
    # path is kept once, with exact checksum conflicts rejected globally.
    all_refs = audit.references([r for k in audit.GROUPS for r in refs[k]])
    remaining = {r['path']: r for r in all_refs}
    result = {}
    for group in audit.GROUPS:
        clean = audit.references(refs[group])
        result[group] = [r for r in clean if r['path'] in remaining]
        for r in result[group]: remaining.pop(r['path'])
    if remaining or any(not result[k] for k in audit.GROUPS):
        raise ValueError('Every full target kept once in nonempty audit groups required')
    return result


def packet(refs, epochs, rows, span, sites, facts):
    controls = records(rows)
    if (type(sites) is not int or sites <= 0 or type(facts) is not int or facts < 0
            or set(span) != {'start', 'stop', 'facts'} or any(type(v) is not int for v in span.values())
            or not 0 <= span['start'] < span['stop'] <= sites or not 0 <= span['facts'] <= facts):
        raise ValueError('Complete original physical site/fact span required')
    clean = groups(refs)
    captured = audit.canonical_epochs(epochs)
    targets = {r['path']: r for k in audit.GROUPS for r in clean[k]}
    if targets != {e['reference']['path']: e['reference'] for e in captured}:
        raise ValueError('ALL and only exact full source targets and original captured epochs required')
    audit.check_epochs(captured)
    return dict(status=STATUS, groups=clean, source_epochs=captured, rows=controls,
        signature=signature(controls), span=span.copy(), sites=sites, facts=facts,
        pending_full_hash=True, initial100_emitted=False, truth_opened=False, recall_measured=False)


def validate_packet(value):
    rebuilt = packet(value['groups'], value['source_epochs'], value['rows'], value['span'], value['sites'], value['facts'])
    if rebuilt != value:
        raise ValueError('Exact complete immutable pair source packet required')
    return rebuilt


def part_control(part, row, span):
    stats = part['stats']
    if (part['signature'] != row['signature'] or part['span'] != span
            or part['partition_only_not_initial_output'] is not True
            or part['same_scores_never_rescored'] is not True
            or stats['sites_completed'] != span['stop']-span['start']
            or stats['address_facts'] != span['facts'] or stats['every_site_scored_once'] is not True
            or stats['fixed_reservoir'] is not None or stats['score_after_fixed_pool'] is not False
            or stats['initial_cap'] != 100 or len(part['candidates']) > 100):
        raise ValueError('Each OWN recipe scores every physical site/fact once before exact reduction')


def score_plan(output, cfg, stage):
    cursor = count = 0
    controls = None
    refs = {k: [] for k in audit.GROUPS}
    captured, producer_refs, packet_refs = [], [], []
    for i, span in enumerate(cfg['expected_spans']):
        m = stage(output, f'score{i:03d}')
        p = json.loads(verify(m['artifacts']['packet']).read_text())
        p = validate_packet(p)
        if (p['span'] != span or p['span']['start'] != cursor or p['sites'] != cfg['sites']
                or p['facts'] != cfg['facts'] or m['trace']['rows'] != p['rows']):
            raise ValueError('Whole contiguous national spans and source-bound pair producers required')
        if controls is None: controls = p['rows']
        if controls != p['rows']:
            raise ValueError('Same two own-query recipes across ALL physical spans required')
        for row in controls:
            part = json.loads(verify(m['artifacts']['part'+str(row['position'])]).read_text())
            part_control(part, row, span)
            del part
        cursor = span['stop']; count += span['facts']
        captured = audit.canonical_epochs([*captured, *p['source_epochs']])
        for k in audit.GROUPS: refs[k] = audit.references([*refs[k], *p['groups'][k]])
        producer_refs.append(m['_reference']); packet_refs.append(m['artifacts']['packet'])
        del p, m
        gc.collect()  # Own Python metadata only, no system purge.
    if cursor != cfg['sites'] or count != cfg['facts']:
        raise ValueError('ALL national sites and facts for BOTH queries required')
    return dict(groups=groups(refs), source_epochs=captured, rows=controls,
                signature=signature(controls), sites=cfg['sites'], facts=cfg['facts']), producer_refs, packet_refs


def audited_plan(output, cfg, stage):
    value, producers, packets = score_plan(output, cfg, stage)
    receipts, audit_refs = [], []
    for group in audit.GROUPS:
        m = stage(output, 'audit-'+group)
        r = json.loads(verify(m['artifacts']['receipt']).read_text())
        if (r['score_stages'] != producers or r['audit_packets'] != packets
                or r['group'] != group or m['trace']['group'] != group):
            raise ValueError('Actual full receipts bound to ALL current pair producers required')
        receipts.append(r)
        audit_refs.append(m['_reference'])
    audit.require_receipts(value, receipts)
    value.update(producers=producers, packets=packets, audits=audit_refs)
    return value
