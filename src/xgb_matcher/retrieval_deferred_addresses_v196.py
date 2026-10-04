"""Complete V177 reader: original certificates/typed CSR now, full hash after.

A source-bound QUARANTINE optimization, not a candidate/recall certificate.
All 1024 original arrays and authority checks remain. Never use a proof or
candidate before the national caller's actual complete after-hash receipts.
"""
import json
from pathlib import Path

import duckdb
import numpy as np

from .retrieval_address_authority_v168 import parent_exit
from .retrieval_memory_policy_v153 import POLICY, validate_report
from .retrieval_name_pages_v151 import fingerprint
from .retrieval_national_names_v138 import PackedTerms
from .retrieval_paged_addresses_v178 import PagedAddresses
from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_postscore_audit_v187 import canonical
from .retrieval_v20_workbench import record, verify


def phase_certificate(output, channel, bucket, context, seal):
    """EXACT original177 phase controls, deferring only its four data hashes."""
    from scripts import compile_retrieval_v177_packed_addresses as backend
    l, p, e = context
    backend.request(l['config'], channel, bucket)
    dest = backend.directory(output, channel, bucket)
    if any((dest/n).exists() for n in ('worker_stop.json', 'status.json', 'supervisor_stop.json')):
        raise ValueError('STOP_PACK_DO_NOT_RESTART')
    offset = channel*l['config']['term_buckets']+bucket
    phase_ref = seal['phase_authorities'][offset]
    if verify(phase_ref).resolve() != (dest/'manifest.json').resolve():
        raise ValueError('Exact original fully sealed phase certificate required')
    parent_exit(dest)
    m = json.loads((dest/'manifest.json').read_text())
    validate_report(m['resources'])
    s = m['stats']
    expected = dict(channel=channel, bucket=bucket, postings=s['postings'], terms=s['terms'], artifacts=m['artifacts'])
    if (seal['access'][offset] != expected
        or m['status'] != 'SEALED_V177_EXACT_ADDRESS_PACK' or m['launch'] != record(output/'launch.json')
        or (m['channel'],m['bucket']) != (channel,bucket) or (s['channel'],s['bucket']) != (channel,bucket)
        or s['postings'] != sum(x['channel_counts'][str(channel)] for x in p['buckets'][str(bucket)])
        or not all(s[k] for k in ('every_original_posting_and_token_reconstructed_exactly','all_postings_sorted_unique',
            'native_unsigned_batched_csr_audit','source_semantics_unchanged','no_posting_or_owner_cap'))
        or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
        or m['preflight'] != record(dest/'preflight.json')
        or set(m['artifacts']) != {'post_ids.npy','post_offsets.npy','token_data.npy','token_offsets.npy'}):
        raise ValueError('Same protected original all-source pack certificate required')
    gate = json.loads(verify(m['preflight']).read_text())
    if gate['policy_id'] != POLICY or not gate['pass_system_quiet_policy'] or not gate['worker_swap_budget_changed']:
        raise ValueError('Actual original authorized pack preflight required')
    for name, ref in m['artifacts'].items():
        canonical(ref)  # Exact original checksum is retained, NEVER replaced by stat.
        path = Path(ref['path'])
        if path.resolve() != dest/name or not path.is_file():
            raise ValueError('Every exact original own pack artifact required')
        if 'size_bytes' in ref and path.stat().st_size != ref['size_bytes']:
            raise ValueError('Full original certified file size required')
    return m


class DeferredHashAddresses(PagedAddresses):
    """Same full real readonly source maps and query math; pending full hash."""
    def __init__(self, reference):
        from scripts import compile_retrieval_v177_packed_addresses as backend
        self.valid = False
        self.pending_full_source_hash = True
        self.in_coverage = False
        self.access = {}
        self.files = {}
        self.references = []
        self.c = None
        self.backend = backend
        try:
            path = verify(reference)
            root = path.parent
            if path.resolve() != root / 'manifest.json':
                raise ValueError('Exact complete supervised address seal required')
            l, p, e = backend.load(root)
            supervision = parent_exit(root / 'seal')
            m = json.loads(path.read_text())
            validate_report(m['seal_resources'])
            if (m['status'] != backend.STATUS or m['launch'] != record(root / 'launch.json')
                    or any(m[k] != p[k] for k in ('staging', 'source_store', 'dictionary', 'catalog',
                                                  'aliases', 'sites', 'facts', 'postings'))
                    or m['complete_retrieval_index'] or m['ready_for_fold1']
                    or any(m.get(k) is not False for k in PROTECTED)):
                raise ValueError('Same protected entire official address source required')
            if len(m['access']) != 4*l['config']['term_buckets'] or len(m['phase_authorities']) != 4*l['config']['term_buckets']:
                raise ValueError('Every original supervised pack certificate required')
            self.aliases = self.names = p['aliases']
            self.source = p['source_store']
            self.buckets = l['config']['term_buckets']
            expected = []
            phase_refs = []
            authority_refs = [reference, supervision, record(root / 'seal/preflight.json')]
            posts = terms = size = 0
            for ch in range(4):
                for b in range(self.buckets):
                    phase = phase_certificate(root, ch, b, (l, p, e), m)
                    s = phase['stats']
                    dest = backend.directory(root, ch, b)
                    ref = record(dest / 'manifest.json')
                    phase_refs.append(ref)
                    authority_refs.extend([ref, parent_exit(dest), phase['preflight']])
                    expected.append(dict(channel=ch, bucket=b, postings=s['postings'], terms=s['terms'],
                                         artifacts=phase['artifacts']))
                    arrays = {}
                    access = PackedTerms.__new__(PackedTerms)
                    access.a = arrays
                    access.terms = 0
                    self.access[ch, b] = access
                    specs = dict(post_ids=(np.uint32, s['postings']), post_offsets=(np.uint64, s['terms']+1),
                                 token_offsets=(np.uint64, s['terms']+1), token_data=(np.uint8, s['token_bytes']))
                    for name, (dtype, n) in specs.items():
                        ref = phase['artifacts'][name+'.npy']  # FULL after-hash is mandatory before admission, not at this open.
                        epoch = fingerprint(ref['path'])
                        a = np.load(ref['path'], mmap_mode='r', allow_pickle=False)
                        arrays[name] = a
                        if (not isinstance(a, np.memmap) or a.flags.writeable or a.dtype != dtype
                                or a.shape != (n,) or fingerprint(ref['path']) != epoch):
                            raise ValueError('Every unchanged unsigned readonly whole pack required')
                        self.files.setdefault((ch, b), {})[name] = dict(
                            path=Path(ref['path']), fingerprint=epoch, dtype=a.dtype, shape=a.shape)
                        self.references.append(ref)
                    if (arrays['post_offsets'][0] != 0 or arrays['post_offsets'][-1] != s['postings']
                            or arrays['token_offsets'][0] != 0 or arrays['token_offsets'][-1] != s['token_bytes']):
                        raise ValueError('Same whole original token/posting CSR required')
                    self.access[ch, b].terms = s['terms']
                    posts += s['postings']
                    terms += s['terms']
                    size += phase['output_bytes']
            if (expected != m['access'] or phase_refs != m['phase_authorities']
                    or (posts, terms, size) != (m['postings'], m['terms'], m['output_bytes'])):
                raise ValueError('Every original supervised complete pack once required')
            self.launch, self.epochs = l, e
            self.authorities = {r['path']: r for r in authority_refs}
            self.metadata = m
            self.c = duckdb.connect()
            self.c.execute('SET threads=1')
            self.c.execute("SET memory_limit='32MiB'")
            self.valid = True
            self.check_live()
        except BaseException:
            self.close()
            raise

    def posting(self, channel, token):
        self.pending_full_source_hash = True
        return super().posting(channel, token)

    def snapshot(self, text, directory, max_seconds=20., chunk=524288):
        self.pending_full_source_hash = True
        result, stats = super().snapshot(text, directory, max_seconds, chunk)
        return result, dict(stats, source_hashes_checked_at_open_not_per_query=False,
            source_full_hash_audit_pending=True, source_full_hash_admitted_here=False,
            source_original_certificates_and_all_typed_csr_checked=True)

    def verify_after(self):
        # Inherited implementation FULL-hashes every original CSR array and
        # checks the entire original source/STOP ledger on both sides.
        super().verify_after()
        self.pending_full_source_hash = False
