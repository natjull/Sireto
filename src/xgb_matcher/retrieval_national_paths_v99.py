"""Initial100 over full national direct names plus typed one-hop succession."""
import time

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings
from src.xgb_matcher.retrieval_national_membership_v96 import NationalMembership
from src.xgb_matcher.retrieval_scoped_paths_v90 import (OwnerFacts,SuccessionCSR,ScopedPaths,NameStream,literal_scores,
    SpatialStream,geo_features,geo_parameters,native_query,TimedNativeStream)
from src.xgb_matcher.retrieval_scoped_paths_v87 import GRADES
from src.xgb_matcher.retrieval_ranked_path_v91 import collect,UncertifiedPathBudget


class TermAccess:
    """Exact same term tuples on demand, not millions of boxed Python tuples."""
    def __init__(self,ids,buckets,starts,stops):
        self.ids=ids; self.buckets=buckets; self.starts=starts; self.stops=stops
    def __contains__(self,term): return term in self.ids
    def __len__(self): return len(self.ids)
    def __iter__(self): return iter(self.ids)
    def __getitem__(self,term):
        i=self.ids[term]; return int(self.buckets[i]),int(self.starts[i]),int(self.stops[i])


class CompactWordPostings(WordPostings):
    """Only physical term representation changes; inherited score arithmetic."""
    def __init__(self,stats):
        self.facts=stats['facts']; self.postings={}; ids={}; n=sum(b['words'] for b in stats['buckets'])
        buckets=np.empty(n,dtype=np.uint8); starts=np.empty(n,dtype=np.uint32); stops=np.empty(n,dtype=np.uint32); offset=0
        for b in stats['buckets']:
            table=pq.read_table(b['vocabulary'],use_threads=False); bucket=b['bucket']; count=len(table)
            if count!=b['words']: raise ValueError('Complete vocabulary count drift')
            self.postings[bucket]=np.load(b['postings_path'],mmap_mode='r',allow_pickle=False)
            buckets[offset:offset+count]=bucket
            starts[offset:offset+count]=table['posting_start'].to_numpy(zero_copy_only=False)
            stops[offset:offset+count]=table['posting_stop'].to_numpy(zero_copy_only=False)
            for i,token in enumerate(table['token'].to_pylist(),start=offset):
                if token in ids: raise ValueError('Duplicate global word key')
                ids[token]=i
            offset+=count
        if offset!=n: raise ValueError('Whole vocabulary denominator drift')
        self.terms=TermAccess(ids,buckets,starts,stops)


class FullRegistryWithRelations:
    def __init__(self,registry,relations): self.registry=registry; self.relations=relations; self.a=registry.a
    def endpoint_position(self,siret): return self.registry.endpoint_position(siret)
    def parent_endpoint_positions(self,siren): return self.registry.parent_endpoint_positions(siren)
    def validate_relation_membership(self):
        keys=self.a['endpoint_keys']; g=self.relations.a
        positions=np.searchsorted(keys,g['endpoint_keys'])
        if (np.any(positions>=len(keys)) or not np.array_equal(keys[positions],g['endpoint_keys']) or
                not np.array_equal(self.a['endpoint_parents'][positions],g['endpoint_parents'])):
            raise ValueError('Succession/full national membership conflict')
    def neighbors_position(self,position):
        key=f'{int(self.a["endpoint_keys"][position]):014d}'; pos=self.relations.endpoint_position(key)
        if pos is not None: yield from self.relations.neighbors_position(pos)


class NationalScopedPaths(ScopedPaths):
    def targets(self,row):
        fact=row['fact_id']; a=self.owners.a
        if row['score']!=self.scores[fact]: raise ValueError('Sorted fact score drift')
        # An official observation outside this registry is retained, not
        # fabricated into a site. Both sorted/random access agree on scope0.
        if int(a['fact_subjects'][fact])==2 and int(a['fact_scopes'][fact])==3:
            if self.graph.endpoint_position(f'{int(a["fact_owners"][fact]):014d}') is None: return
        yield from super().targets(row)

    def best(self,siret):
        if siret in self.target_cache: return self.target_cache[siret]
        pos=self.graph.endpoint_position(siret)
        if pos is None:
            result=dict(score=0.,fact_id=None,relation_observation=None,outside_national_site_scope=True)
        else:
            score,fact=self.anchor_best(pos)
            result=dict(score=score,fact_id=fact,relation_observation=None,outside_national_site_scope=False)
            for edge in self.graph.neighbors_position(pos):
                anchor=self.graph.endpoint_position(edge['siret'])
                if anchor is None: raise ValueError('Official relation endpoint absent from full registry')
                incoming,other_fact=self.anchor_best(anchor); value=incoming*GRADES[edge['site_binding_code']]
                if value>result['score']:
                    result=dict(score=value,fact_id=other_fact,relation_observation=edge['observation_index'],
                        site_binding_code=edge['site_binding_code'],outside_national_site_scope=False,
                        historical_site_coexistence_proven=False,name_relation_co_validity_proven=False)
        result['outside_succession_endpoint_scope']=self.graph.relations.endpoint_position(siret) is None
        self.target_cache[siret]=result; return result


def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=NationalScopedPaths(owners,graph,scores)
    native=TimedNativeStream(backend,native_query(backend,q,weights,terms,alpha))
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def complete(rows):
        ids=[r['siret'] for r in rows]; base=native.complete(ids)
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(native_score=base[s],path_name_score=paths.best(s)['score'],site_score=float(values[i])) for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    try:
        rows,trace=collect(native,names,spatial,paths.targets,complete,path_weight=path_weight,
            name_score_max=1.,site_score_max=reliability,max_seconds=remaining,page_size=256)
    except UncertifiedPathBudget as exc:
        exc.trace.update(word_index=word_trace,native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,
            native_sorted_seconds=native.sorted_seconds,seconds_including_preparation=time.monotonic()-began); raise
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,unique_documents_visited=trace['unique_targets_completed'],
        native_corpus_unchanged=True,name_site_scope='ALL_NATIONAL_REGISTRY_SITES',succession_hops=1,
        geo_query_available=spatial.point is not None,ban_reliability_factor=reliability,
        native_postings_visits_not_instrumented=True,proof_co_validity_unknown=True,identifier_features=False,
        native_point_calls=native.point_calls,native_point_seconds=native.point_seconds,native_sorted_seconds=native.sorted_seconds,
        physical_completion_buffer=512,physical_fact_page_size=256,compact_term_lookup_same_scores=True)
    return rows,trace
