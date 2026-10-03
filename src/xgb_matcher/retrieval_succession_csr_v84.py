"""Full official one-hop observation adjacency; IDs are storage/join keys only."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SITE_CLASSES={'CURRENT_ADDRESS_GEO_CONCORDANT_NOT_HISTORICAL_PROOF':1,
              'CURRENT_LOCATION_DIFFERS_OR_GEO_DISAGREES':2,'CURRENT_SITE_BINDING_INCOMPLETE':3}
ARRAYS=('endpoint_keys','endpoint_parents','endpoint_offsets','endpoint_observations',
        'parent_keys','parent_offsets','parent_endpoints','predecessor_positions','successor_positions',
        'effective_days','updated_micros','site_classes','predecessor_states','successor_states','headquarters_transfer')


def build(bindings,output,progress=lambda value:None):
    file=pq.ParquetFile(bindings); n=file.metadata.num_rows
    if not 0<n<2**31: raise ValueError('Source observation count outside uint32 incidence range')
    if (file.schema_arrow.field('reported_effective_date').type!=pa.date32()
        or file.schema_arrow.field('source_updated_at').type!=pa.timestamp('us')):
        raise ValueError('Official date/timestamp unit drift')
    ids={k:np.empty(n,dtype=np.uint64) for k in ('predecessor','successor')}
    parents={k:np.empty(n,dtype=np.uint32) for k in ids}
    effective=np.empty(n,dtype=np.int32); updated=np.empty(n,dtype=np.int64)
    classes=np.empty(n,dtype=np.uint8); states={k:np.empty(n,dtype=np.uint8) for k in ids}
    transfer=np.empty(n,dtype=np.uint8); offset=0
    columns=[*(k+'_'+f for k in ids for f in ('siret','siren','state')),
             'reported_effective_date','source_updated_at','site_binding','headquarters_transfer',
             'economic_continuity','historical_site_coexistence_proven','relation_source','relation_type']
    for batch in file.iter_batches(batch_size=131072,columns=columns,use_threads=False):
        d={name:batch.column(i) for i,name in enumerate(batch.schema.names)}; stop=offset+batch.num_rows
        for k in ids:
            for field,width,dtype,dest in (('siret',14,pa.uint64(),ids[k]),('siren',9,pa.uint32(),parents[k])):
                values=d[k+'_'+field]
                if values.null_count or not pc.all(pc.equal(pc.utf8_length(values),width)).as_py():
                    raise ValueError('Official identifier length/null drift')
                dest[offset:stop]=pc.cast(values,dtype).to_numpy(zero_copy_only=False)
            status=d[k+'_state']
            # 0 null/unknown,1 active,2 closed,3 other: original string stays
            # in the hash-addressed authoritative row, never invented.
            states[k][offset:stop]=np.array([0 if x in (None,'') else 1 if x=='A' else 2 if x=='F' else 3
                                            for x in status.to_pylist()],dtype=np.uint8)
        if (not pc.all(pc.fill_null(d['economic_continuity'],False)).as_py()
            or pc.any(pc.fill_null(d['historical_site_coexistence_proven'],True)).as_py()
            or not pc.all(pc.fill_null(pc.equal(d['relation_source'],'SIRENE'),False)).as_py()
            or not pc.all(pc.fill_null(pc.equal(d['relation_type'],'ESTABLISHMENT_SUCCESSION'),False)).as_py()):
            raise ValueError('Official succession source semantics drift')
        effective[offset:stop]=pc.fill_null(pc.cast(d['reported_effective_date'],pa.int32()),np.iinfo(np.int32).min).to_numpy(zero_copy_only=False)
        updated[offset:stop]=pc.fill_null(pc.cast(d['source_updated_at'],pa.int64()),np.iinfo(np.int64).min).to_numpy(zero_copy_only=False)
        try: classes[offset:stop]=[SITE_CLASSES[x] for x in d['site_binding'].to_pylist()]
        except KeyError as exc: raise ValueError('Unknown official site binding class') from exc
        transfer[offset:stop]=[0 if x is None else 2 if x is True else 1 for x in d['headquarters_transfer'].to_pylist()]
        offset=stop
    if offset!=n or any(np.any(ids[k]//100000!=parents[k]) for k in ids) or np.any(ids['predecessor']==ids['successor']):
        raise ValueError('Official membership or self-path drift')
    progress(dict(phase='read_complete_official_observations',observations=n))
    anchors=np.concatenate([ids['predecessor'],ids['successor']]); anchor_parents=np.concatenate([parents['predecessor'],parents['successor']])
    order=np.argsort(anchors,kind='stable'); sorted_keys=anchors[order]; sorted_parents=anchor_parents[order]
    if np.any((sorted_keys[1:]==sorted_keys[:-1])&(sorted_parents[1:]!=sorted_parents[:-1])):
        raise ValueError('Conflicting official endpoint parent')
    starts=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1]
    endpoint_keys=sorted_keys[starts]; endpoint_parents=sorted_parents[starts]
    endpoint_offsets=np.r_[starts,2*n].astype(np.uint32)
    endpoint_observations=(order%n).astype(np.uint32)
    del anchors,anchor_parents,order,sorted_keys,sorted_parents,starts,parents
    pred=np.searchsorted(endpoint_keys,ids['predecessor']).astype(np.uint32)
    succ=np.searchsorted(endpoint_keys,ids['successor']).astype(np.uint32); del ids
    progress(dict(phase='complete_bidirectional_incidence',endpoints=len(endpoint_keys),incidences=2*n))
    order=np.argsort(endpoint_parents,kind='stable'); sorted_parents=endpoint_parents[order]
    starts=np.r_[0,np.flatnonzero(sorted_parents[1:]!=sorted_parents[:-1])+1]
    parent_keys=sorted_parents[starts]; parent_offsets=np.r_[starts,len(endpoint_keys)].astype(np.uint32)
    parent_endpoints=order.astype(np.uint32)
    del order,sorted_parents,starts
    arrays=dict(endpoint_keys=endpoint_keys,endpoint_parents=endpoint_parents,endpoint_offsets=endpoint_offsets,
        endpoint_observations=endpoint_observations,parent_keys=parent_keys,parent_offsets=parent_offsets,
        parent_endpoints=parent_endpoints,predecessor_positions=pred,successor_positions=succ,
        effective_days=effective,updated_micros=updated,site_classes=classes,
        predecessor_states=states['predecessor'],successor_states=states['successor'],headquarters_transfer=transfer)
    if set(arrays)!=set(ARRAYS): raise ValueError('CSR artifact inventory drift')
    for name,value in arrays.items(): np.save(Path(output)/(name+'.npy'),value,allow_pickle=False)
    return dict(observations=n,endpoint_sites=len(endpoint_keys),legal_owners=len(parent_keys),directed_incidences=2*n,
        observations_deduplicated=False,semantic_hops=1,max_endpoint_observations=int(np.max(np.diff(endpoint_offsets))),
        max_parent_endpoint_members=int(np.max(np.diff(parent_offsets))),identifiers_as_features=False,
        historical_site_coexistence_proven=False,closed_endpoints_removed=False,
        arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype)) for name,value in arrays.items()})


class SuccessionCSR:
    """Readonly exhaustive adjacency over hash-verified files supplied by caller."""
    def __init__(self,directory):
        self.a={k:np.load(Path(directory)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ARRAYS}

    def endpoint_position(self,siret):
        if not (isinstance(siret,str) and len(siret)==14 and siret.isascii() and siret.isdigit()):
            raise ValueError('Invalid endpoint join key')
        keys=self.a['endpoint_keys']; pos=int(np.searchsorted(keys,int(siret)))
        return pos if pos<len(keys) and keys[pos]==int(siret) else None

    def parent_endpoint_positions(self,siren):
        if not (isinstance(siren,str) and len(siren)==9 and siren.isascii() and siren.isdigit()):
            raise ValueError('Invalid parent join key')
        keys=self.a['parent_keys']; pos=int(np.searchsorted(keys,int(siren)))
        if pos>=len(keys) or keys[pos]!=int(siren): return iter(())
        lo,hi=self.a['parent_offsets'][pos:pos+2]
        return iter(self.a['parent_endpoints'][int(lo):int(hi)])

    def neighbors_position(self,position):
        a=self.a; lo,hi=a['endpoint_offsets'][position:position+2]
        for raw in a['endpoint_observations'][int(lo):int(hi)]:
            r=int(raw); p=int(a['predecessor_positions'][r]); s=int(a['successor_positions'][r])
            if position not in (p,s): raise ValueError('CSR incidence mismatch')
            target=s if position==p else p
            yield dict(siret=f'{int(a["endpoint_keys"][target]):014d}',siren=f'{int(a["endpoint_parents"][target]):09d}',
                anchor_siret=f'{int(a["endpoint_keys"][position]):014d}',anchor_siren=f'{int(a["endpoint_parents"][position]):09d}',
                observation_index=r,direction='FORWARD' if position==p else 'REVERSE',
                reported_effective_day=int(a['effective_days'][r]),source_updated_micros=int(a['updated_micros'][r]),
                site_binding_code=int(a['site_classes'][r]),historical_site_coexistence_proven=False,
                anchor_state_code=int(a['predecessor_states' if position==p else 'successor_states'][r]),
                target_state_code=int(a['successor_states' if position==p else 'predecessor_states'][r]),
                headquarters_transfer_code=int(a['headquarters_transfer'][r]))

    def neighbors_name_owner(self,name):
        """Raw scoped relations, NOT identity/alias or query-site admission."""
        subject=name['subject_kind']; kind=name['name_kind']; parent=name['siren']
        if subject=='SIRET' and kind in ('LEGAL','SIGN','TRADE','USUAL'):
            pos=self.endpoint_position(name['siret'])
            if pos is None: return
            if f'{int(self.a["endpoint_parents"][pos]):09d}'!=parent:
                raise ValueError('Name site/parent scope conflict')
            positions=(pos,)
        elif subject=='SIREN' and kind in ('LEGAL','TRADE','USUAL'):
            positions=self.parent_endpoint_positions(parent)
        else:
            raise ValueError('Context-only or unverified name scope is not lexical path input')
        for pos in positions: yield from self.neighbors_position(int(pos))
