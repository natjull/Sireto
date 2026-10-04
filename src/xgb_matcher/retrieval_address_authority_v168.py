"""V168 source audit: exact full prefix and reverse in separate guarded children.

Never relabel V160 as V154, never accept the V159 single-bucket cost probe or
successful phases alone. Original prefixes keep their original authorities.
No lexical/scoring kernel, resource policy, corpus or candidate cap changes.
"""
import json
from pathlib import Path
import math

from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_memory_policy_v153 import validate_report, LIMITS, POLICY
from .retrieval_v20_workbench import record, verify

STATUS = 'SEALED_V160_COMPLETE_ORIGINAL_ADDRESS_DICTIONARY_AND_SITE_REVERSE'


def parent_exit(directory):
    """Worker PASS alone is not proof of a supervised successful child exit."""
    path = directory/'supervision.json'
    if not path.is_file():
        raise ValueError('UNCONFIRMED_PARENT_EXIT_NO_COMPLETE_STORE')
    s = json.loads(path.read_text())
    if (s.get('policy_id') != POLICY or s.get('limits') != LIMITS
            or type(s.get('exit_code')) is not int or s['exit_code'] != 0
            or s.get('own_child_exit_confirmed') is not True
            or type(s.get('child_pid')) is not int or s['child_pid'] <= 0
            or type(s.get('peak_rss_bytes')) is not int or not 0 <= s['peak_rss_bytes'] <= LIMITS['max_rss_bytes']
            or type(s.get('minimum_system_free_percent')) is not int
            or not LIMITS['min_system_free_percent'] <= s['minimum_system_free_percent'] <= 100
            or type(s.get('sample_count')) is not int or s['sample_count'] < 1
            or type(s.get('swapout_pages_delta')) is not int or s['swapout_pages_delta'] < 0
            or s.get('global_swap_is_worker_attributed') is not False
            or not isinstance(s.get('wall_seconds'), (float, int)) or isinstance(s['wall_seconds'], bool)
            or not math.isfinite(s['wall_seconds']) or not 0 <= s['wall_seconds'] <= LIMITS['max_seconds']):
        raise ValueError('UNCONFIRMED_PARENT_EXIT_NO_COMPLETE_STORE')
    return record(path)


def store(reference, cfg, prefix_reference):
    """Audit every reverse proof against an independently sealed full prefix."""
    import scripts.resume_retrieval_v160_reverse_store as backend
    certificate_path = verify(prefix_reference)
    if certificate_path.name != 'manifest.json':
        raise ValueError('Exact supervised prefix audit certificate required')
    parent_exit(certificate_path.parent)
    prefix = json.loads(certificate_path.read_text())
    validate_report(prefix['resources'])
    if (prefix['status'] != 'SEALED_V168_COMPLETE_ORIGINAL_PREFIX_AUDIT'
            or prefix['source_store'] != reference
            or any(prefix.get(k) is not False for k in PROTECTED)
            or prefix['complete_retrieval_index']):
        raise ValueError('Same complete protected prefix audit required')
    for epoch in prefix['source_epochs']:
        if backend.fingerprint(epoch['reference']['path']) != epoch['fingerprint']:
            raise ValueError('Full prefix drift between guarded source audits')
    path = verify(reference)
    root = path.parent
    l, p = backend.load(root)
    original_catalog = prefix['catalog']
    d = json.loads(verify(prefix['dictionary']).read_text())
    m = json.loads(path.read_text())
    validate_report(m['seal_resources'])
    if (path.resolve() != (root/'manifest.json').resolve() or m['status'] != STATUS
            or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
            or m['ready_for_fold1'] or m['lexical_address_access_built'] or m['stopped_reverse_phase_imported']
            or m['sites'] != cfg['expected_sites'] or m['original_observations'] != cfg['expected_original_observations']
            or not m['every_original_site_and_fact_preserved']
            or not m['all_original_observations_and_provenance_retained']):
        raise ValueError('Complete protected V160 store required, not prefix/partial/probe')
    if (m['launch'] != record(root/'launch.json') or m['dictionary'] != p['dictionary']
            or m['catalog'] != p['catalog'] or m['source_projection'] != p['source_projection']
            or m['original_stop'] != p['original_stop'] or m['prefix_staged'] != p['prefix_staged']
            or p['source_rows'] != cfg['expected_source_rows']
            or p['sites'] != m['sites'] or p['observations'] != m['original_observations']
            or p['lexical_observations'] != m['lexical_observations']
            or p['aliases'] != m['aliases'] or p['facts'] != m['facts']
            or original_catalog != m['catalog'] or d['aliases'] != m['aliases'] or d['facts'] != m['facts']):
        raise ValueError('Same entire original source/dictionary/namespace required')
    s = backend.authority(root, 'reverse_staged', backend.STAGED)
    if (m['reverse_staged'] != record(root/'reverse_staged.json') or s['source'] != p['dictionary']
            or s['facts'] != d['facts'] or s['aliases'] != d['aliases']
            or len(s['phase_authorities']) != l['config']['dictionary_buckets']):
        raise ValueError('Same complete original reverse staging required')
    # Check the staged facts count for EACH bucket as well as the global sum.
    for i, b in enumerate(d['buckets']):
        phase = backend.phase_record(root, 'reverse-stage', i)
        parent_exit(backend.directory(root, 'reverse-stage', i))
        st = phase['stats']
        if (phase['inputs'] != dict(dictionary=p['dictionary'], bucket=b)
                or s['phase_authorities'][i] != record(backend.directory(root, 'reverse-stage', i)/'manifest.json')
                or st['facts'] != b['facts'] or not st['every_fact_staged_once']
                or not st['every_original_site_parent_and_attribute_exact']
                or st['site_or_proof_or_alias_cap'] is not None or st['identifiers_as_features']):
            raise ValueError('Every original fact/site staging certificate required')
    maximum = backend.maximum(root, 'reverse-pack')
    if len(m['intervals']) != maximum or len(m['phase_authorities']) != maximum:
        raise ValueError('Every whole-site original reverse interval required')
    cursor = facts = 0
    for i in range(maximum):
        phase = backend.phase_record(root, 'reverse-pack', i)
        parent_exit(backend.directory(root, 'reverse-pack', i))
        st = phase['stats']
        stop = min(cursor+l['config']['site_quantum'], p['sites'])
        if (m['phase_authorities'][i] != record(backend.directory(root, 'reverse-pack', i)/'manifest.json')
                or m['intervals'][i] != st or st['start'] != cursor or st['stop'] != stop
                or st['sites'] != m['sites'] or st['aliases'] != d['aliases']
                or phase['inputs'] != dict(reverse_staged=m['reverse_staged'], index=i, start=cursor, stop=stop)
                or st['facts'] != sum(row['rows'] for row in s['parts'][str(i)])
                or not st['every_original_field_and_site_csr_exact'] or not st['all_provenance_keys_preserved']):
            raise ValueError('Every unchanged site/fact reverse proof required')
        for ref in st['sources']:
            verify(ref)
        cursor = stop
        facts += st['facts']
    if cursor != m['sites'] or facts != d['facts']:
        raise ValueError('Whole original site/fact denominator required')
    backend.load(root)
    for epoch in prefix['source_epochs']:
        if backend.fingerprint(epoch['reference']['path']) != epoch['fingerprint']:
            raise ValueError('Full prefix drift during guarded reverse audit')
    return m, d


def epochs(reference, m, d):
    import scripts.resume_retrieval_v160_reverse_store as backend
    root = Path(reference['path']).parent
    _, p = backend.load(root)
    refs = [reference, m['launch'], m['catalog'], m['dictionary'], m['reverse_staged'],
            m['prefix_staged'], m['original_stop'], *m['phase_authorities']]
    refs.extend(item['reference'] for item in p['prefix_epochs'])
    for phase in backend.PHASES:
        for i in range(backend.maximum(root, phase)):
            refs.append(parent_exit(backend.directory(root, phase, i)))
    for b in d['buckets']:
        refs.extend(b['artifacts'].values())
    for st in m['intervals']:
        refs.extend([*st['artifacts'].values(), st['numbers'], st['ordered_source'], *st['sources']])
    unique = {r['path']: r for r in refs}
    return [dict(reference=r, fingerprint=backend.fingerprint(path)) for path, r in sorted(unique.items())]
