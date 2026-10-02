"""Observation-preserving initial search prototype, NOT national validation.

Name/address interaction happens within one official observation and one
address alternative. No free join across sources, dates, siblings or aliases.
The collector holds at most100 DISTINCT SIRET groups throughout search.
Parent-only observations are retained by callers, never invented into SIRET.
"""
from collections import defaultdict
from dataclasses import dataclass
import heapq

from scripts.retrieval_v38_initial_top100 import tokens
from src.xgb_matcher.official_evidence import OfficialEvidence
from src.xgb_matcher.retrieval_proof_v56 import signal


def termset(value):
    return frozenset(tokens(signal(value)))


@dataclass(frozen=True)
class Address:
    terms: frozenset
    number: str
    insee: str
    postcode: str


@dataclass(frozen=True)
class Observation:
    evidence_id: str
    source_record_id: str
    source: str
    subject_kind: str
    siren: str
    siret: str
    valid_from: str | None
    valid_to: str | None
    observed_at: str | None
    is_current: bool
    names: tuple
    addresses: tuple

    @classmethod
    def from_row(cls, row):
        official = OfficialEvidence.from_dict(row)
        return cls(official.evidence_id, official.source_record_id, official.source.value,
            official.subject_kind.value, official.siren, official.siret,
            official.valid_from, official.valid_to, official.observed_at, official.is_current,
            tuple(termset(n.normalized_value) for n in official.names),
            tuple(Address(termset(a.normalized_value), signal(a.number), a.insee, a.postcode)
                  for a in official.addresses))


def query(row):
    # Explicit CRM-only feature whitelist, no IDs, labels or candidate input.
    allowed = {'query_id', 'fold', 'crm_name', 'crm_number', 'crm_suffix',
               'crm_street', 'crm_insee', 'crm_postcode'}
    if set(row)-allowed:
        raise ValueError('Forbidden query features')
    if row.get('fold', 2) not in (2, 3, 4):
        raise ValueError('Closed query fold')
    address = ' '.join(row.get(k, '') for k in ('crm_number', 'crm_suffix', 'crm_street'))
    return dict(name=termset(row.get('crm_name', '')), address=termset(address),
                number=signal(row.get('crm_number', '')), insee=row.get('crm_insee', ''),
                postcode=row.get('crm_postcode', ''))


def coverage(asked, present):
    return len(asked & present)/len(asked) if asked else 0.


def score(observation, q):
    """Fixed COST/CORRECTNESS kernel, not learned or quality-selected weights.

    Unmatched/out-of-date observations are not excluded by geography or state.
    Max over individual name aliases and address alternatives in ONE row.
    """
    name = max((coverage(q['name'], n) for n in observation.names), default=0.)
    if not observation.addresses:
        return name
    values = []
    for address in observation.addresses:
        addr = coverage(q['address'], address.terms)
        place = bool((q['insee'] and q['insee'] == address.insee) or
                     (q['postcode'] and q['postcode'] == address.postcode))
        number = bool(q['number'] and q['number'] == address.number and place)
        # Interaction cannot use a name or address from a different row.
        values.append(name+addr+name*addr+float(place)+float(number))
    return max(values)


class GroupTop:
    """Exact max-over-observations SIRET collapse with bounded live state."""
    def __init__(self, cap=100):
        if not isinstance(cap, int) or not 1 <= cap <= 100:
            raise ValueError('Absolute initial100 cap')
        self.cap = cap; self.live = {}; self.heap = []; self.peak_live = 0

    def _purge(self):
        while self.heap and self.live.get(self.heap[0][2]) != self.heap[0][0]:
            heapq.heappop(self.heap)

    def offer(self, key, value):
        if value <= 0:
            return
        item = (value, -int(key), key)
        if key in self.live:
            if value <= self.live[key]:
                return
        elif len(self.live) >= self.cap:
            self._purge()
            if item <= self.heap[0]:
                return
            _, _, removed = heapq.heappop(self.heap); del self.live[removed]
        self.live[key] = value; heapq.heappush(self.heap, item)
        self.peak_live = max(self.peak_live, len(self.live))
        if len(self.heap) > 2*self.cap:
            self.heap = [(v, -int(k), k) for k, v in self.live.items()]; heapq.heapify(self.heap)

    def result(self):
        return sorted(self.live.items(), key=lambda r:(-r[1], r[0]))


class ObservationIndex:
    def __init__(self, observations):
        observations = tuple(observations)
        self.observations = tuple(o for o in observations if o.subject_kind == 'SIRET')
        self.parent_only_count = sum(o.subject_kind == 'SIREN' for o in observations)
        self.postings = defaultdict(list)
        for i, o in enumerate(self.observations):
            keys = set('N:'+t for n in o.names for t in n)
            keys.update('A:'+t for a in o.addresses for t in a.terms)
            keys.update('C:'+a.insee for a in o.addresses if a.insee)
            keys.update('Z:'+a.postcode for a in o.addresses if a.postcode)
            for key in keys:
                self.postings[key].append(i)

    def search(self, row):
        q = query(row); keys = ['N:'+t for t in q['name']]+['A:'+t for t in q['address']]
        keys += [p+q[f] for p, f in (('C:', 'insee'), ('Z:', 'postcode')) if q[f]]
        # Posting union is corpus traversal, NOT a fixed top-X candidate pool.
        # Every matching observation is scored by the FINAL kernel.
        positions = sorted({i for k in keys for i in self.postings.get(k, ())})
        heap = GroupTop()
        for i in positions:
            o = self.observations[i]; heap.offer(o.siret, score(o, q))
        return heap.result(), dict(scored_observations=len(positions), total_observations=len(self.observations),
            peak_live_siret_groups=heap.peak_live, peak_heap_entries_bound=201,
            parent_only_not_projected=self.parent_only_count, fixed_candidate_reservoir=False)

    def exhaustive(self, row):
        # Independent whole-cohort oracle only; not production retrieval.
        q = query(row); best = {}
        for o in self.observations:
            value = score(o, q)
            if value > 0:
                best[o.siret] = max(best.get(o.siret, 0.), value)
        return sorted(best.items(), key=lambda r:(-r[1], r[0]))[:100]
