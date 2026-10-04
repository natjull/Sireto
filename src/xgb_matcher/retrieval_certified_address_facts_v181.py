"""Same full V160/V147 facts using V178's already certified source epochs.

No monolithic raw-prefix rehash during query initialization. The original
full V175 audits remain authoritative; all exact original refs/epochs must
be present, and every interval is consumed by unchanged BoundAddressFacts.
Unmeasured nationally, not a retrieval/recall/SLO certificate.
"""
import json
from pathlib import Path

import numpy as np

from .retrieval_address_authority_v168 import STATUS, parent_exit
from .retrieval_address_reverse_v147 import FIELDS
from .retrieval_initial_engine_v158 import CompleteAddressFacts, BoundAddressFacts, fingerprint
from .retrieval_memory_policy_v153 import validate_report
from .retrieval_paged_addresses_v178 import PagedAddresses, PagedAliasSnapshot
from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_v20_workbench import verify


class CertifiedBoundFacts(BoundAddressFacts):
    def finish(self):
        super().finish()
        self.snapshot.verify_after()  # No same-epoch mmap mutation may escape into candidates.


class CertifiedAddressFacts(CompleteAddressFacts):
    """Reuse complete raw/reverse audit, not a smaller store or unaudited epoch.

Inherited _open/bind/Bound.batches remain exactly V158/V147: same own
number/CP/INSEE/current/provenance. Only the initialization authority path
changes. No borrowed big maps survive physical interval boundaries.
"""
    def __init__(self, address_index):
        self.valid = False
        self.active = None
        if (not isinstance(address_index, PagedAddresses) or not address_index.valid
                or address_index.metadata['status'] != address_index.backend.STATUS
                or any(address_index.metadata.get(k) is not False for k in PROTECTED)):
            raise ValueError('Live entire supervised V178/V177 source required')
        self.index = address_index
        address_index.check_live()
        self.reference = address_index.source
        path = verify(self.reference)
        self.root = path.parent
        self.stop_scopes = []
        self._check_stops()
        m = json.loads(path.read_text())
        validate_report(m['seal_resources'])
        if (path.resolve() != path.parent / 'manifest.json' or m['status'] != STATUS
                or any(m.get(k) is not False for k in PROTECTED)
                or m['complete_retrieval_index'] or m['ready_for_fold1']
                or m['lexical_address_access_built'] or m['stopped_reverse_phase_imported']
                or not m['every_original_site_and_fact_preserved']
                or not m['all_original_observations_and_provenance_retained']
                or any(m[k] != address_index.metadata[k] for k in ('aliases', 'sites', 'facts', 'catalog', 'dictionary'))):
            raise ValueError('Same entire protected original V160 source required')
        self.metadata = m
        self.sites, self.aliases, self.facts = m['sites'], m['aliases'], m['facts']
        self.catalog_reference = m['catalog']
        self.dictionary_reference = m['dictionary']
        self.intervals = m['intervals']
        self.starts = []
        self.checked = []
        epochs = {item['reference']['path']: item for item in address_index.epochs['epochs']}
        self.used = {}
        def certified(ref):
            entry = epochs.get(ref['path'])
            if (entry is None or entry['reference'] != ref
                    or tuple(entry['fingerprint']) != fingerprint(ref['path'])):
                raise ValueError('Exact original full-audit reference and unchanged epoch required')
            self.used[ref['path']] = ref
            return tuple(entry['fingerprint'])
        for ref in (self.reference, m['launch'], m['catalog'], m['dictionary'], m['reverse_staged'],
                    m['prefix_staged'], m['original_stop']):
            certified(ref)
        if len(m['phase_authorities']) != len(self.intervals) or not self.intervals:
            raise ValueError('Every original site/fact interval once required')
        cursor = facts = 0
        for ref, st in zip(m['phase_authorities'], self.intervals):
            certified(ref)
            scope = Path(ref['path']).parent
            self.stop_scopes.append(scope)
            supervision = parent_exit(scope)
            certified(supervision)
            phase = json.loads(verify(ref).read_text())
            validate_report(phase['resources'])
            if (phase['stats'] != st or st['start'] != cursor or not cursor < st['stop'] <= self.sites
                    or st['sites'] != self.sites or st['aliases'] != self.aliases
                    or not st['every_original_field_and_site_csr_exact'] or not st['all_provenance_keys_preserved']
                    or st['site_or_proof_or_alias_cap'] is not None or st['identifiers_as_features']
                    or any(phase.get(k) is not False for k in PROTECTED)):
                raise ValueError('Every unchanged original full reverse interval required')
            refs = [*st['artifacts'].values(), st['numbers'], st['ordered_source'], *st['sources']]
            original_epochs = {r['path']: certified(r) for r in refs}
            types = {**FIELDS, 'number_codes': np.uint32, 'site_offsets': np.uint64}
            if set(st['artifacts']) != {k+'.npy' for k in types}:
                raise ValueError('Every original typed coherent fact attribute required')
            specs = {k: (np.dtype(dtype), (st['stop']-st['start']+1 if k == 'site_offsets' else st['facts'],))
                     for k, dtype in types.items()}
            self.checked.append(dict(epochs=original_epochs, specs=specs))
            self.starts.append(cursor)
            cursor = st['stop']
            facts += st['facts']
        if cursor != self.sites or facts != self.facts:
            raise ValueError('Whole original site/fact denominator required')
        self.epochs = [epochs[path] for path in self.used]
        self.valid = True
        self.assert_live()

    def _check_stops(self):
        # The pinned historical original_stop is not a successful phase. Only
        # the resumed successful source and its actual reverse phases are live.
        if any(self.root.glob('seal_*_stop.json')) or any(
                (scope / name).exists() for scope in self.stop_scopes
                for name in ('worker_stop.json', 'status.json', 'supervisor_stop.json')):
            raise ValueError('STOP_ORIGINAL_REVERSE_SOURCE_DO_NOT_USE')

    def assert_live(self):
        if not self.valid:
            raise ValueError('Closed or failed complete certified address facts')
        try:
            self._check_stops()
            self.index.check_live()
            for entry in self.epochs:
                if tuple(entry['fingerprint']) != fingerprint(entry['reference']['path']):
                    raise ValueError('Complete certified address source epoch drift')
        except BaseException:
            self.valid = False
            raise

    def verify_after(self):
        try:
            self.assert_live()
            for ref in self.used.values():
                verify(ref)
            self.assert_live()
        except BaseException:
            self.valid = False
            raise

    def bind(self, snapshot, query, fact_chunk=65536):
        if not isinstance(snapshot, PagedAliasSnapshot):
            raise ValueError('Immutable source-bound file-backed address snapshot required')
        # Keep every original V158 validation, changing only finish's stronger check.
        super().bind(snapshot, query, fact_chunk)
        snapshot.assert_live()
        return CertifiedBoundFacts(self, snapshot, query, fact_chunk)
