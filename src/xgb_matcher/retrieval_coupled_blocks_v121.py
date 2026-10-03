"""Exact initial100 over coupled official name/address/site blocks.

Prototype search kernel, not a national index publication. Blocks only bound
the SAME leaf score; every site remains reachable. No fixed candidate pool,
offset rereads, locality/closed-state filter or downstream model invocation.
Owners and relation hypotheses must be resolved by the source adapter.
"""
from dataclasses import dataclass
import heapq
import time

import numpy as np

from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_activity_joint_v114 import vector

SCALES = np.array([25., 100., 500., 2000.])


def channels(text):
    text = normalized(text)
    compact = text.replace(' ', '_')
    return (frozenset(text.split()),
            frozenset(compact[i:i+3] for i in range(max(0, len(compact)-2))),
            frozenset(compact[i:i+4] for i in range(max(0, len(compact)-3))))


@dataclass(frozen=True)
class Proof:
    text: str
    grade: float = 1.
    current_exact: bool = False
    # Opaque provenance only, never a score/partition feature.
    reference: tuple = ()
    relation_hypothesis: bool = False


@dataclass(frozen=True)
class Site:
    siret: str
    siren: str
    names: tuple[Proof, ...] = ()
    addresses: tuple[Proof, ...] = ()
    postcode: str = ''
    insee: str = ''
    number: str = ''
    headquarters: bool | None = False
    xy: tuple | None = None
    activity_class: int = -1
    state: str | None = None


@dataclass(frozen=True)
class Query:
    name: str = ''
    address: str = ''
    postcode: str = ''
    insee: str = ''
    number: str = ''
    xy: tuple | None = None
    reliable_geo: bool = False


def point(value):
    if value is None:
        return None
    p = np.asarray(value, dtype=np.float64)
    if p.shape != (2,) or not np.isfinite(p).all() or np.any(np.abs(p) > 1e9):
        raise ValueError('Finite same-CRS projected XY required')
    return p


def weights(value):
    w = np.asarray(value, dtype=np.float64)
    if w.shape != (30,) or not np.isfinite(w).all() or np.any(w < 0) or np.any(w > 64) or not np.any(w > 0):
        raise ValueError('Nonnegative initial30 weights required')
    return w


def evidence(terms, aliases, cache):
    """Maximum of whole variants, never tokens pooled across different aliases."""
    coverage = exact = 0.
    for proof in aliases:
        text = normalized(proof.text)
        if text not in cache:
            cache[text] = channels(text)
        value = max((len(q & s) / len(q) if q else 0.) for q, s in zip(terms, cache[text]))
        coverage = max(coverage, value * proof.grade)
        if proof.current_exact and text and text == cache['_query_text']:
            exact = max(exact, proof.grade)
    return coverage, exact


def native(n, a, ne, ae, p, number, siege, geometry, reliable):
    return np.array([n, a, ne, ae, p, number*p, n*p, a*p,
                     (n+a)*p if n > 0 and a > 0 else 0., siege*p, ne*ae*p,
                     *geometry, *(geometry if reliable else np.zeros(4))], dtype=np.float64)


class CoupledBlocks:
    def __init__(self, sites, block_size=128):
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
            raise ValueError('Positive physical block size required')
        self.sites = tuple(sites)
        if not self.sites or len({s.siret for s in self.sites}) != len(self.sites):
            raise ValueError('Nonempty unique official sites required')
        self.xy = np.full((len(self.sites), 2), np.nan, dtype=np.float64)
        for i, site in enumerate(self.sites):
            if (len(site.siret) != 14 or len(site.siren) != 9 or not site.siret.isascii()
                or not site.siren.isascii() or not site.siret.isdecimal() or not site.siren.isdecimal()
                or not site.siret.startswith(site.siren)):
                raise ValueError('Official site membership required')
            if not isinstance(site.activity_class, int) or isinstance(site.activity_class,bool) or site.activity_class < -1:
                raise ValueError('Explicit activity class or UNKNOWN=-1 required')
            if site.headquarters is not None and not isinstance(site.headquarters,bool):
                raise ValueError('Headquarters true/false/UNKNOWN required')
            p = point(site.xy)
            if p is not None:
                self.xy[i] = p
            for proof in (*site.names, *site.addresses):
                if not np.isfinite(proof.grade) or not 0 <= proof.grade <= 1:
                    raise ValueError('Official proof grade outside 0..1')
                if not isinstance(proof.current_exact,bool) or not isinstance(proof.relation_hypothesis,bool):
                    raise ValueError('Explicit proof flags required')
                if proof.relation_hypothesis and proof.current_exact:
                    raise ValueError('Relation hypothesis cannot claim exact current alias')
        known = np.flatnonzero(np.isfinite(self.xy[:,0]))
        unknown = np.flatnonzero(~np.isfinite(self.xy[:,0]))
        self.groups = []
        def split(ids):
            if len(ids) <= block_size:
                if len(ids):
                    self.groups.append(ids)
                return
            span = np.ptp(self.xy[ids], axis=0)
            axis = int(np.argmax(span))
            ids = ids[np.argsort(self.xy[ids, axis], kind='stable')]
            mid = len(ids)//2; split(ids[:mid]); split(ids[mid:])
        split(known)
        # Unknown geometry still has every lexical/address/activity channel.
        for start in range(0,len(unknown),block_size):
            self.groups.append(unknown[start:start+block_size])
        n = len(self.groups)
        self.lo = np.full((n,2),np.nan); self.hi = self.lo.copy()
        self.max_grade = np.zeros((n,2)); self.class_sets = []
        # Union token postings are upper bounds ONLY, not coherent leaf proofs.
        self.postings = [[{} for _ in range(3)] for _ in range(2)]
        self.exact = [{}, {}]
        self.places = [set() for _ in range(n)]
        self.numbers = [set() for _ in range(n)]
        self.sieges = np.zeros(n)
        for b, ids in enumerate(self.groups):
            if np.isfinite(self.xy[ids[0],0]):
                self.lo[b] = np.min(self.xy[ids],axis=0); self.hi[b] = np.max(self.xy[ids],axis=0)
            classes = set()
            union = [[set() for _ in range(3)] for _ in range(2)]
            exact = [{}, {}]
            for i in ids:
                site = self.sites[i]
                classes.add(site.activity_class)
                self.places[b].update(('CP:'+site.postcode,'INSEE:'+site.insee))
                self.numbers[b].add(normalized(site.number))
                self.sieges[b] = max(self.sieges[b],float(site.headquarters is True))
                for side, aliases in enumerate((site.names,site.addresses)):
                    for proof in aliases:
                        text = normalized(proof.text)
                        if not text or proof.grade == 0:
                            continue
                        self.max_grade[b,side] = max(self.max_grade[b,side],proof.grade)
                        for j, terms in enumerate(channels(text)):
                            union[side][j].update(terms)
                        if proof.current_exact:
                            exact[side][text] = max(exact[side].get(text,0.),proof.grade)
            self.class_sets.append(classes)
            for side in range(2):
                for j in range(3):
                    for token in union[side][j]:
                        self.postings[side][j].setdefault(token,[]).append(b)
                for text, grade in exact[side].items():
                    self.exact[side].setdefault(text,[]).append((b,grade))
        for side in range(2):
            for j in range(3):
                self.postings[side][j] = {t:np.array(ids,dtype=np.int32) for t,ids in self.postings[side][j].items()}
        if sorted(i for ids in self.groups for i in ids) != list(range(len(self.sites))):
            raise ValueError('Physical blocks lost/duplicated sites')

    def _prepare(self, query, w, activity):
        point(query.xy)
        if not isinstance(query.reliable_geo,bool):
            raise ValueError('Explicit BAN reliability required')
        if any(not isinstance(k,int) or k < 0 or not np.isfinite(v) or not 0 <= v <= 1
               for k,v in activity.items()):
            raise ValueError('All activity class grades must be explicit 0..1')
        caches = [{'_query_text':normalized(query.name)}, {'_query_text':normalized(query.address)}]
        return (channels(query.name), channels(query.address)), caches

    def features(self, i, query, activity, terms, caches):
        site = self.sites[i]
        n, ne = evidence(terms[0],site.names,caches[0])
        a, ae = evidence(terms[1],site.addresses,caches[1])
        place = float(bool(query.postcode and site.postcode == query.postcode)
                      or bool(query.insee and site.insee == query.insee))
        num = float(bool(normalized(query.number)) and normalized(site.number) == normalized(query.number))
        geometry = np.zeros(4)
        if query.xy is not None and np.isfinite(self.xy[i,0]):
            distance = float(np.sqrt(np.sum((self.xy[i]-point(query.xy))**2)))
            geometry = 1./(1.+distance/SCALES)
        base = native(n,a,ne,ae,place,num,float(site.headquarters is True),geometry,query.reliable_geo)
        return vector(base,activity.get(site.activity_class,0.))

    def bounds(self, query, w, activity, terms):
        count = len(self.groups); covers = np.zeros((count,2)); exact = np.zeros((count,2))
        for side, value in enumerate((query.name,query.address)):
            for j, tokens in enumerate(terms[side]):
                hits = np.zeros(count,dtype=np.int32)
                for token in tokens:
                    if token in self.postings[side][j]:
                        hits[self.postings[side][j][token]] += 1
                if tokens:
                    covers[:,side] = np.maximum(covers[:,side],hits/len(tokens))
            covers[:,side] *= self.max_grade[:,side]
            for b,grade in self.exact[side].get(normalized(value),[]):
                exact[b,side] = grade
        places = {prefix+value for prefix,value in (('CP:',query.postcode),('INSEE:',query.insee)) if value}
        p = np.array([float(bool(places & values)) for values in self.places])
        number = normalized(query.number)
        num = np.array([float(bool(number) and number in values) for values in self.numbers])
        geometry = np.zeros((count,4)); known = np.isfinite(self.lo[:,0])
        if query.xy is not None and np.any(known):
            xy = point(query.xy)
            delta = np.maximum(self.lo[known]-xy,0.) + np.maximum(xy-self.hi[known],0.)
            distance = np.sqrt(np.sum(delta*delta,axis=1))
            # Round distance DOWN conservatively before monotone proximity.
            guard = 64*np.finfo(float).eps*(1.+np.linalg.norm(xy)+
                np.linalg.norm(self.lo[known],axis=1)+np.linalg.norm(self.hi[known],axis=1))
            geometry[known] = 1./(1.+np.maximum(0.,distance-guard)[:,None]/SCALES)
        result = np.zeros(count)
        for b in range(count):
            f = native(*covers[b],*exact[b],p[b],num[b],self.sieges[b],geometry[b],query.reliable_geo)
            grade = max((activity.get(k,0.) for k in self.class_sets[b]),default=0.)
            value = float(np.sum(vector(f,grade)*w))
            # All positive components exactly zero => exact zero bound.
            result[b] = np.nextafter(value+128*np.finfo(float).eps*(1.+value+float(np.sum(w))),np.inf) if value else 0.
        return result

    def search(self, query, value, activity=None, limit=100, max_seconds=20.):
        if not isinstance(limit,int) or isinstance(limit,bool) or not 1 <= limit <= 100:
            raise ValueError('Absolute initial output cap100')
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Positive query cost guard required')
        began = time.monotonic(); w = weights(value); activity = {} if activity is None else activity
        terms,caches = self._prepare(query,w,activity)
        upper = self.bounds(query,w,activity,terms)
        order = np.argsort(-upper,kind='stable'); heap = []; visited = []; cutoff = 0.
        for b in order:
            if time.monotonic()-began > max_seconds:
                raise TimeoutError('STOP_COUPLED_BLOCK_QUERY_COST')
            if upper[b] <= 0. or (len(heap)==limit and upper[b] < cutoff):
                break
            visited.append(int(b))
            for offset,i in enumerate(self.groups[b]):
                if offset % 64 == 0 and time.monotonic()-began > max_seconds:
                    raise TimeoutError('STOP_COUPLED_BLOCK_QUERY_COST')
                score = float(np.sum(self.features(i,query,activity,terms,caches)*w))
                if score > upper[b]:
                    raise ValueError('Unsound coupled block upper bound')
                if score <= 0.:
                    continue
                item = (score,-int(self.sites[i].siret),int(i))
                if len(heap) < limit:
                    heapq.heappush(heap,item)
                elif item > heap[0]:
                    heapq.heapreplace(heap,item)
                cutoff = heap[0][0] if len(heap)==limit else 0.
        rows = [dict(siret=self.sites[i].siret,siren=self.sites[i].siren,score=score)
                for score,_,i in sorted(heap,reverse=True)]
        if time.monotonic()-began > max_seconds:
            raise TimeoutError('STOP_COUPLED_BLOCK_QUERY_COST')
        return rows,dict(certified=True,sites=len(self.sites),blocks=len(self.groups),
            blocks_completed=len(visited),sites_completed=sum(len(self.groups[b]) for b in visited),
            blocks_pruned=len(self.groups)-len(visited),seconds=time.monotonic()-began,
            initial_cap=limit,fixed_reservoir=None,score_after_fixed_pool=False,
            locality_or_closed_state_filter=False,global_boundary_ties=True,
            source_schema='COHERENT_WORD_CHAR3_CHAR4_NATIVE19_ACTIVITY30',
            corpus_scope='CALLER_SUPPLIED_REQUIRES_SEPARATE_NATIONAL_COVERAGE_AUTHORITY')

    def exhaustive(self, query, value, activity=None, limit=100):
        w = weights(value); activity = {} if activity is None else activity
        terms,caches = self._prepare(query,w,activity)
        rows = [dict(siret=site.siret,siren=site.siren,
            score=float(np.sum(self.features(i,query,activity,terms,caches)*w)))
            for i,site in enumerate(self.sites)]
        return sorted((r for r in rows if r['score']>0.),key=lambda r:(-r['score'],r['siret']))[:limit]
