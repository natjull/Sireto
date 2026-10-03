"""Source-bound national name snapshots with every sealed V149 parent join.

No candidate pool, fitted alias, label lookup or source truncation. Lexical
and one-hop semantics are unchanged V138/V137; only missing UL joins change.
"""
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .retrieval_append_catalog_v141 import AppendKeys
from .retrieval_name_rebind_v149 import patched_positions,ReboundNameSnapshot
from .retrieval_national_names_v138 import scatter_positive
from .retrieval_proof_bridge_v137 import PROTECTED,MISSING,one_hop_names
from .retrieval_v20_workbench import verify,record


def close_maps(arrays):
    for a in arrays.values():
        if isinstance(a,np.memmap):a._mmap.close()


class SourceSnapshot:
    """Name evidence borrows readonly source maps until its reader closes."""
    def __init__(self,reader,snapshot):self.reader=reader;self.snapshot=snapshot;self.sites=snapshot.sites
    def gather(self,ids):
        if self.reader.keys is None:raise ValueError('Closed official name source snapshot')
        return self.snapshot.gather(ids)


class ReboundNames:
    """Verify complete immutable authorities at open and again before sealing.

Original V149 validation is reused, never rewritten to reinterpret its
measured source or resource certificates. Buckets are mapped one at a time.
The thin catalog exposes name-snapshot joins, not full membership/layout.
"""
    def __init__(self,reference):
        from scripts.compile_retrieval_v149_name_rebind import load as original_load,STATUS
        self.reference=reference;self.refs=[reference];self.graph={};self.keys=None;self.new={}
        try:
            m=json.loads(verify(reference).read_text());self.metadata=m
            if (m['status']!=STATUS or any(m.get(k) is not False for k in PROTECTED)
                or m['complete_retrieval_index'] or m['ready_for_fold1'] or m['downstream_models_changed']
                or not m['resources']['pass_resource_policy'] or m['resources']['swapout_pages_delta']!=0):
                raise ValueError('Protected complete successful V149 joins required')
            root=Path(reference['path']).parent
            launch,_,(bridge,cat,owners)=original_load(root);cfg=launch['config']
            if (m['launch']!=record(root/'launch.json') or m['sources']!=dict(bridge=cfg['bridge'],catalog=cfg['catalog'],owners=bridge['sources']['owners'])
                or m['sites']!=cat['stats']['sites'] or m['parents']!=cat['stats']['parents']
                or len(m['buckets'])!=len(bridge['buckets'])):
                raise ValueError('Whole unchanged official name/append source namespace required')
            st=m['stats']
            if (st['facts']!=owners['facts'] or st['site_facts']!=bridge['stats']['site_facts']
                or st['previous_unbound_parent_facts']!=bridge['stats']['unbound_parent_facts']
                or st['recovered_parent_facts']+st['remaining_unbound_parent_facts']!=st['previous_unbound_parent_facts']
                or any(st[k] is not True for k in ('every_original_fact_audited','every_previous_bound_position_unchanged',
                    'every_patch_reconstructed_exactly','all_original_observations_scopes_and_exact_flags_preserved','remaining_unbound_facts_preserved'))
                or st['name_or_owner_or_member_cap'] is not None or st['identifiers_as_features'] or st['site_alias_lifted_to_parent']):
                raise ValueError('Every original fact and exact append-only patch required')
            self.refs.extend((m['launch'],launch['preflight'],*m['sources'].values(),owners['source'],bridge['sources']['graph']))
            self.keys=AppendKeys(cfg['catalog']);self.refs.extend(self.keys.references)
            self.names=owners['names'];self.source=owners['source'];self.items=m['buckets']
            self.sites=m['sites'];self.parents=m['parents'];self.old_sites=bridge['stats']['sites'];self.old_parents=bridge['stats']['parents']
            self.new['new_site_parent_positions']=self.map(cat['artifacts']['new_site_parent_positions.npy'])
            pp=self.new['new_site_parent_positions']
            if (pp.dtype!=np.uint32 or pp.shape!=(self.sites-self.old_sites,)
                or np.any(pp>=self.parents) or not np.array_equal(self.keys.reconstruct(pp,'parent'),
                    (self.keys.new_site//np.uint64(100000)).astype(np.uint32))):
                raise ValueError('Every new site-parent join exact required')
            self.catalog=SimpleNamespace(sites=self.sites,parents=self.parents,keys=self.keys,
                previous=SimpleNamespace(sites=self.old_sites,parents=self.old_parents),new=self.new)
            recovered=[];sums={k:0 for k in ('facts','site_facts','previous_unbound_parent_facts','recovered_parent_facts','remaining_unbound_parent_facts')}
            output_bytes=0
            for i,(item,old) in enumerate(zip(self.items,bridge['buckets'])):
                b=item['source_bucket'];s=item['stats'];a={};previous={};patch={}
                try:
                    if (item['bucket']!=i or b!=old['bucket'] or item['previous_artifacts']!=old['artifacts']
                        or set(item['artifacts'])!={'fact_rows.npy','parent_positions.npy'} or s['facts']!=b['facts']
                        or s['site_facts']!=old['stats']['site_facts'] or s['previous_unbound_parent_facts']!=old['stats']['unbound_parent_facts']
                        or s['recovered_parent_facts']+s['remaining_unbound_parent_facts']!=s['previous_unbound_parent_facts']
                        or any(s[k] is not True for k in ('every_original_fact_audited','every_previous_bound_position_unchanged',
                            'every_patch_reconstructed_exactly','every_unbound_fact_preserved'))
                        or s['name_or_owner_or_member_cap'] is not None or s['identifiers_as_features'] or s['site_alias_lifted_to_parent']):
                        raise ValueError('Every complete original physical owner bucket required')
                    a={k.removesuffix('.npy'):self.map(ref) for k,ref in b['artifacts'].items()}
                    previous={k.removesuffix('.npy'):self.map(ref) for k,ref in old['artifacts'].items()}
                    patch={k.removesuffix('.npy'):self.map(ref) for k,ref in item['artifacts'].items()}
                    for key,dtype in dict(parent_key=np.uint32,owner_key=np.uint64,subject_code=np.uint8,scope_code=np.uint8,current_exact=np.bool_,name_fact_offsets=np.uint64).items():
                        n=b['names']+1 if key=='name_fact_offsets' else b['facts']
                        if a[key].dtype!=dtype or a[key].shape!=(n,):raise ValueError('Whole typed original name fact fields required')
                    offsets=a['name_fact_offsets']
                    if offsets[0]!=0 or offsets[-1]!=b['facts'] or np.any(offsets[1:]<=offsets[:-1]):
                        raise ValueError('Every nonempty original alias-owner range required')
                    if any(v.dtype!=np.uint32 or v.shape!=(b['facts'],) for v in previous.values()):
                        raise ValueError('Whole unchanged previous positions required')
                    r,p=patch['fact_rows'],patch['parent_positions']
                    patched=patched_positions(previous['fact_parent_positions'],r,p,self.old_parents,self.parents)
                    if (len(r)!=s['recovered_parent_facts'] or np.any(a['subject_code'][r]!=1)
                        or np.any(~np.isin(a['scope_code'][r],[1,2])) or np.any(previous['fact_site_positions'][r]!=MISSING)
                        or not np.array_equal(self.keys.reconstruct(p,'parent'),a['parent_key'][r])
                        or not np.array_equal(a['owner_key'][r],a['parent_key'][r].astype(np.uint64))
                        or np.unique(p).tolist()!=s['recovered_parent_positions']):
                        raise ValueError('Every patch must remain an exact official UL proof')
                    del patched
                    for name,ref in item['artifacts'].items():
                        if verify(ref).resolve()!=(root/f'bucket{i:03d}'/name).resolve():raise ValueError('Actual sealed sparse patch path required')
                        output_bytes+=Path(ref['path']).stat().st_size
                    recovered.extend(s['recovered_parent_positions'])
                    for k in sums:sums[k]+=s[k]
                finally:close_maps(a);close_maps(previous);close_maps(patch)
            recovered=np.unique(np.array(recovered,dtype=np.uint32));self.recovered_parents=recovered
            if (any(sums[k]!=st[k] for k in sums) or len(recovered)!=st['recovered_parents'] or output_bytes!=m['output_bytes']
                or int(np.sum(np.isin(pp,recovered)))!=st['appended_sites_with_recovered_parent_proofs']):
                raise ValueError('Complete global join counts and every output byte required')
            graph=json.loads(verify(bridge['sources']['graph']).read_text())
            if (graph['status']!='SEALED_V133_COMPLETE_ALL_RAW_OFFICIAL_SUCCESSION' or any(graph.get(k) is not False for k in PROTECTED)
                or graph['complete_retrieval_index'] or graph['ready_for_fold1']
                or not graph['resources']['pass_resource_policy'] or graph['resources']['swapout_pages_delta']!=0
                or graph['stats']['endpoints']!=bridge['stats']['graph_endpoints']
                or graph['stats']['observations']!=bridge['stats']['graph_observations']):
                raise ValueError('Whole original successful official succession source required')
            gl=json.loads(verify(graph['launch']).read_text());self.refs.append(graph['launch'])
            for ref in gl['implementation'].values():verify(ref)
            sizes=dict(site_parent_positions=self.old_sites,graph_site_positions=bridge['stats']['graph_endpoints'],
                graph_parent_positions=bridge['stats']['graph_endpoints'],predecessor_graph_positions=bridge['stats']['graph_observations'],
                successor_graph_positions=bridge['stats']['graph_observations'])
            for key,n in sizes.items():
                v=self.graph[key]=self.map(bridge['graph'][key+'.npy'])
                if v.dtype!=np.uint32 or v.shape!=(n,):raise ValueError('Whole exact source-bound graph positions required')
            v=self.graph['continuity_codes']=self.map(graph['artifacts']['continuity_codes.npy'])
            if (v.dtype!=np.uint8 or v.shape!=(bridge['stats']['graph_observations'],) or np.any(v>2)
                or np.any(self.graph['site_parent_positions']>=self.old_parents)
                or np.any(self.graph['graph_site_positions']>=self.old_sites)
                or np.any(self.graph['graph_parent_positions']>=self.old_parents)
                or np.any(self.graph['predecessor_graph_positions']>=bridge['stats']['graph_endpoints'])
                or np.any(self.graph['successor_graph_positions']>=bridge['stats']['graph_endpoints'])):
                raise ValueError('Every original graph endpoint and UNKNOWN/FALSE/TRUE flag required')
            # Source fields are inherited from the audited immutable V149,
            # not assumed absent merely because the former namespace was old.
        except BaseException:
            self.close();raise

    def map(self,ref):
        self.refs.append(ref);return np.load(verify(ref),mmap_mode='r',allow_pickle=False)

    def close(self):
        close_maps(self.graph);self.graph={};close_maps(self.new);self.new={}
        if self.keys is not None:
            arrays={}
            for c in (self.keys,self.keys.previous):
                for name in ('site','parent','extra_site','extra_parent','new_site','new_parent'):
                    a=getattr(c,name,None)
                    if isinstance(a,np.memmap):arrays[name+str(id(c))]=a
            close_maps(arrays);self.keys=None

    def verify_after(self):
        from scripts.compile_retrieval_v149_name_rebind import load as original_load
        original_load(Path(self.reference['path']).parent)
        unique={(r['path'],r['sha256']):r for r in self.refs}
        for ref in unique.values():verify(ref)

    def query(self,index,text,max_seconds=20.,chunk=131072):
        if (self.keys is None or index.names!=self.names or index.source!=self.source or not isinstance(text,str)
            or isinstance(max_seconds,bool) or not np.isfinite(max_seconds) or max_seconds<=0
            or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0):
            raise ValueError('Whole identical alias source, text and physical cost guard required')
        began=time.monotonic()
        def guard():
            if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_REBOUND_NAME_QUERY_COST')
        coverage,exact,lexical=index.coverage(text,max_seconds=max_seconds,chunk=chunk);lex_done=time.monotonic()
        pc=np.zeros(self.parents);pe=np.zeros(self.parents,dtype=bool);sc=np.zeros(self.old_sites);se=np.zeros(self.old_sites,dtype=bool)
        processed=0
        for item in self.items:
            b=item['source_bucket'];src={};old={};patch={}
            try:
                # Inputs hashed/validated at open and before output sealing;
                # readonly per-bucket maps avoid holding the corpus in RAM.
                src={k:np.load(b['artifacts'][k+'.npy']['path'],mmap_mode='r',allow_pickle=False) for k in ('name_fact_offsets','subject_code','current_exact')}
                old={k.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in item['previous_artifacts'].items()}
                patch={k.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in item['artifacts'].items()}
                pp=patched_positions(old['fact_parent_positions'],patch['fact_rows'],patch['parent_positions'],self.old_parents,self.parents)
                sl=slice(b['name_start'],b['name_start']+b['names'])
                trace=scatter_positive(coverage[sl],exact[sl],src,dict(fact_parent_positions=pp,fact_site_positions=old['fact_site_positions']),pc,pe,sc,se,chunk)
                processed+=trace['references_processed'];del pp
            finally:close_maps(src);close_maps(old);close_maps(patch)
            guard()
        scatter_done=time.monotonic()
        # The whole-alias intermediates no longer contribute to any output.
        # Free them BEFORE relation allocation, without dropping a single fact.
        del coverage,exact;gc.collect();guard()
        relation=one_hop_names(pc,sc,self.graph['graph_site_positions'],self.graph['graph_parent_positions'],
            self.graph['predecessor_graph_positions'],self.graph['successor_graph_positions'],self.graph['continuity_codes'],chunk)
        relation_done=time.monotonic();guard()
        snapshot=ReboundNameSnapshot(self.catalog,self.graph['site_parent_positions'],pc,pe,sc,se,relation)
        for a in (pc,pe,sc,se,relation):a.flags.writeable=False
        ids=np.arange(self.old_sites,self.sites,dtype=np.uint32);nc,ne=snapshot.gather(ids)
        stats=dict(lexical_seconds=lex_done-began,scatter_seconds=scatter_done-lex_done,relation_seconds=relation_done-scatter_done,
            lexical=lexical,references_processed=processed,original_parent_proofs=int(np.count_nonzero(pc[:self.old_parents])),
            appended_parent_proofs=int(np.count_nonzero(pc[self.old_parents:])),appended_parent_exact=int(np.count_nonzero(pe[self.old_parents:])),
            direct_site_proofs=int(np.count_nonzero(sc)),related_endpoint_proofs=int(np.count_nonzero(relation)),
            appended_sites_with_name_proofs=int(np.count_nonzero(nc)),appended_sites_with_direct_exact=int(np.count_nonzero(ne)),
            all_sealed_parent_patches_applied=True,no_alias_or_owner_or_member_cap=True,site_alias_lifted_to_parent=False,
            aliases_released_before_relations=True,recall_measured=False,initial100_emitted=False)
        guard();stats['seconds']=time.monotonic()-began
        return SourceSnapshot(self,snapshot),stats
