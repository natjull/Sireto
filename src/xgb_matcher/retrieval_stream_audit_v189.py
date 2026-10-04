"""Same complete187 plan and receipts, one physical score packet at a time."""
import gc
import json

from . import retrieval_postscore_audit_v187 as audit
from .retrieval_v20_workbench import verify


def score_plan(output, cfg, stage):
    cursor = facts = 0
    signature = None
    groups = {k: [] for k in audit.GROUPS}
    captured, score_refs, packet_refs = [], [], []
    for i in range(cfg['partitions']):
        score, ref = stage(output, f'score{i:03d}')
        if score['source_hash_audit_pending'] is not True:
            raise ValueError('Same complete original quarantined score required')
        packet_ref = score['artifacts']['audit_packet']
        packet = json.loads(verify(packet_ref).read_text())
        part = json.loads(verify(score['artifacts']['partition']).read_text())
        if (packet['signature'] != score['trace']['signature'] or packet['signature'] != part['signature']
                or packet['source_epochs'] != score['source_epochs'] or packet['span'] != part['span']
                or part['partition_only_not_initial_output'] is not True
                or part['same_scores_never_rescored'] is not True
                or part['stats']['sites_completed'] != packet['span']['stop']-packet['span']['start']
                or part['stats']['address_facts'] != packet['span']['facts']
                or part['stats']['every_site_scored_once'] is not True
                or part['stats']['fixed_reservoir'] is not None
                or part['stats']['score_after_fixed_pool'] is not False or part['stats']['initial_cap'] != 100):
            raise ValueError('Exact original whole physical score, recipe, captured epochs and packet required')
        del score, part
        packet = audit.validate_packet(packet)
        if signature is None:
            signature = packet['signature']
        if (packet['sites'] != cfg['sites'] or packet['facts'] != cfg['facts']
                or packet['signature'] != signature or packet['span']['start'] != cursor):
            raise ValueError('Identical entire world and contiguous once-only physical scores required')
        cursor = packet['span']['stop']
        facts += packet['span']['facts']
        captured = audit.canonical_epochs([*captured, *packet['source_epochs']])
        for group in audit.GROUPS:
            groups[group] = audit.references([*groups[group], *packet['groups'][group]])
        score_refs.append(ref); packet_refs.append(packet_ref)
        del packet
        gc.collect()  # Owned metadata only, no system purge.
    if cursor != cfg['sites'] or facts != cfg['facts']:
        raise ValueError('All original national sites/facts before source audit required')
    audit.check_epochs(captured)
    return dict(groups=groups, source_epochs=captured, signature=signature,
                sites=cfg['sites'], facts=cfg['facts']), score_refs, packet_refs


def audited_plan(output, cfg, stage):
    value, score_refs, packet_refs = score_plan(output, cfg, stage)
    receipts = []
    for group in audit.GROUPS:
        phase, ref = stage(output, 'audit-'+group)
        receipt = json.loads(verify(phase['artifacts']['receipt']).read_text())
        if (receipt['group'] != group or phase['trace']['group'] != group
                or receipt['score_stages'] != score_refs or receipt['audit_packets'] != packet_refs):
            raise ValueError('Every current actual entire source receipt bound to original score stages required')
        receipts.append(receipt)
        del phase
    audit.require_receipts(value, receipts)
    return value
