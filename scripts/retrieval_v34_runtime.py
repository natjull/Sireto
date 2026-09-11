"""Physical-only accelerator for the frozen v33 recipe. No truth-side inputs.

Caches are scoped to immutable searcher instances and keyed by document address
or complete query arguments, never by query_id or a known answer.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import threading
import time
import sys
import gc

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts import retrieval_v33_full_replay as v33
from src.xgb_matcher.retrieval_parent_discovery_v18 import SiretChannelSpec, collapse_siret_hits_to_parent_rows, SIGNAL_SCHEMA
from src.xgb_matcher.retrieval_v20_workbench import write_json,sql_path


class DocMapping(dict):
    pass


def footprint(value):
    if isinstance(value,dict):return sys.getsizeof(value)+sum(footprint(k)+footprint(v) for k,v in value.items())
    if isinstance(value,(tuple,list)):return sys.getsizeof(value)+sum(footprint(v) for v in value)
    if hasattr(value,'__dict__'):return sys.getsizeof(value)+footprint(vars(value))
    return sys.getsizeof(value)


class BoundedCache:
    def __init__(self,capacity=1024,max_bytes=16*1024**2):
        self.values=OrderedDict();self.capacity=capacity;self.max_bytes=max_bytes;self.bytes=0
    def get(self,key):
        if key not in self.values:return None
        self.values.move_to_end(key);return self.values[key][0]
    def put(self,key,value):
        size=footprint(value)
        if key in self.values:self.bytes-=self.values.pop(key)[1]
        if size>self.max_bytes:return
        self.values[key]=(value,size);self.bytes+=size
        while len(self.values)>self.capacity or self.bytes>self.max_bytes:
            _,(_,old_size)=self.values.popitem(last=False);self.bytes-=old_size


class StoredDocument:
    def __init__(self,key,value):self.key=key;self.value=value
    def to_dict(self):
        result=DocMapping(self.value);result.document_key=self.key;return result


class CachedSearcher:
    def __init__(self,searcher,capacity=1024):
        self.original=searcher;self.capacity=capacity;self.cache=BoundedCache(capacity);self.lock=threading.RLock()
        self.hits=0;self.misses=0
    def __getattr__(self,key):return getattr(self.original,key)
    def doc(self,address):
        key=(address.segment_ord,address.doc)
        with self.lock:
            found=self.cache.get(key)
            if found is not None:self.hits+=1;return found
        value=StoredDocument(key,self.original.doc(address).to_dict())
        with self.lock:
            self.misses+=1;self.cache.put(key,value)
        return value


def instrument(backend):
    if isinstance(backend.searcher,CachedSearcher):return backend
    backend.searcher=CachedSearcher(backend.searcher)
    original=backend._record;records=BoundedCache();lock=threading.RLock()
    def cached_record(document):
        key=getattr(document,'document_key',None)
        if key is None:return original(document)
        with lock:
            found=records.get(key)
            if found is not None:return found
        result=original(document)
        with lock:
            records.put(key,result)
        return result
    backend._record=cached_record
    for name in ('by_siret','linked_sirens','sites_for_siren'):
        method=getattr(backend,name,None)
        if method is not None:
            if name=='sites_for_siren':
                def memoize_sites(method):
                    cached=lru_cache(maxsize=128)(lambda *args:tuple(method(*args)))
                    return lambda *args:list(cached(*args))
                setattr(backend,name,memoize_sites(method))
            else:setattr(backend,name,lru_cache(maxsize=128)(method))
    return backend


class FastRuntime:
    def __init__(self):
        self.native_base=v33.TantivyBackend;self.native_parent=v33._build_backends
        self.native_address=v33._write_address_signals
        self.side=None;self.official=None;self.parent_backends=None;self.base=None
        self.pool=ThreadPoolExecutor(max_workers=4)
    def setup_legacy(self):
        if self.official is not None:return
        v15=v33.cfg(v33.CONFIGS[2])['authorities'];b=v33.cfg(v33.CONFIGS[3])['authorities']['backends']
        vc=v33.RetrievalV11Config.load(v33.ROOT/'config'/v33.CONFIGS[0])
        oc=replace(v33.OfficialEvidenceRetrievalConfig.load(v33.ROOT/'config'/v33.CONFIGS[1]),
          enable_direct_parent_expansion=False,enable_consensus_parent_expansion=False,relation_seed_limit=vc.relation_seed_limit,search_workers=4)
        self.side=v33.RetrievalSidecarReader(Path(v15['sidecars']['path']),cache_size=128)
        self.side._connection.execute("SET memory_limit='2GB'");self.side._connection.execute('SET threads=2')
        self.official=v33.OfficialEvidenceRetriever.from_index_paths(v15['base_index']['path'],
          [b['overlay_official']['path'],str(v33.DATA/'indices/dossier_temporal_tail_v1/66f38cc146c51da6')],oc,
          temporal_overlay_index_path=b['temporal']['path'],legal_entity_role_index_path=b['roles']['path'],
          sector_official_index_path=[b['sector']['path'],str(v33.DATA/'indices/official_sector_data_es_v1/37ea905013a83660')])
        for _,backend in self.official.backends:instrument(backend)
        self.retriever=v33.AllHumanV11Retriever(self.official,self.side,vc)
        self.original_local=self.side._local_path;self.original_geo=self.side._geo_path
    def project_sidecars(self,queries,output):
        # Lossless geographic predicate pushdown. Every site in each requested
        # commune/CP is retained, including the original full-commune counts.
        c=duckdb.connect();c.execute("SET memory_limit='2GB'");c.execute('SET threads=2')
        spill=output/'_scope_spill';spill.mkdir(exist_ok=True)
        c.execute(f'SET temp_directory={sql_path(spill)}')
        self.side._connection.execute(f'SET temp_directory={sql_path(spill)}')
        c.register('regions',pa.Table.from_pylist([dict(insee=q.insee,postcode=q.postcode) for _,_,q in queries]))
        for key,path in [('local',self.original_local),('geo',self.original_geo)]:
            target=output/(key+'_scope.parquet')
            c.execute(f'''COPY (SELECT s.* FROM read_parquet(?) s WHERE
              s.insee IN (SELECT insee FROM regions WHERE insee<>'') OR
              s.postcode IN (SELECT postcode FROM regions WHERE insee='' AND postcode<>''))
              TO {sql_path(target)} (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(path)])
            setattr(self.side,'_'+key+'_path',target)
        c.close()
    def legacy(self,queries,output):
        self.setup_legacy();self.project_sidecars(queries,output);raw=[];timings=[]
        for qid,fold,q in queries:
            start=time.monotonic();r=self.retriever.retrieve(q)
            raw.extend(dict(query_id=qid,fold=fold,siren=x.siren,siret=x.siret,union_rank=x.union_rank) for x in r.candidates)
            timings.append(dict(query_id=qid,seconds=time.monotonic()-start,candidates=len(r.candidates),stages=dict(r.strategy_counts)))
            print(f'LEGACY {len(timings)}/{len(queries)} {timings[-1]["seconds"]:.3f}s',flush=True)
        pq.write_table(pa.Table.from_pylist(raw,schema=v33.RAW_SCHEMA),output/'raw.parquet',compression='zstd')
        write_json(output/'legacy_timing.json',timings)
        # Release mapped indexes before the other index families are opened.
        # Keeping every stage resident raised RSS without useful cross-stage reuse.
        self.official._search_pool.shutdown(wait=True)
        self.retriever=None;self.official=None;gc.collect()
    def address(self,*args,**kwargs):
        kwargs['sidecar_path']=self.side._local_path
        result=self.native_address(*args,**kwargs)
        # Only these two temporary projections were created by this runtime.
        local,geo=self.side._local_path,self.side._geo_path
        self.side.close();self.side=None
        local.unlink();geo.unlink()
        return result
    def parents(self,config):
        return self.native_parent(config)
    def base_backend(self,path):
        if self.base is None:self.base=instrument(self.native_base(path))
        if Path(path)!=self.base.index_path:raise ValueError('Foreign base index')
        return self.base
    def direct(self,*,config,retrieval_queries,index_root,output_path):
        backend=self.base_backend(index_root);specs=tuple(SiretChannelSpec.from_mapping(r) for r in config['siret_channels'])
        order={s.name:i for i,s in enumerate(specs)}
        with pq.ParquetWriter(output_path,SIGNAL_SCHEMA,compression='zstd') as writer:
            for offset in range(0,len(retrieval_queries),8):
                rows=[]
                futures={self.pool.submit(backend.search,q,spec.backend_channel,spec.limit):(qid,fold,spec)
                  for qid,fold,q in retrieval_queries[offset:offset+8] for spec in specs}
                for future in as_completed(futures):
                    qid,fold,spec=futures[future]
                    rows.extend(collapse_siret_hits_to_parent_rows(query_id=qid,fold=fold,channel=spec.name,hits=future.result()))
                order_rows=sorted(rows,key=lambda r:(r['query_id'],order[r['channel']],r['channel_rank'],r['siren']))
                writer.write_table(pa.Table.from_pylist(order_rows,schema=SIGNAL_SCHEMA));del futures,rows,order_rows
    def install(self):
        v33.legacy=self.legacy;v33._build_backends=self.parents;v33._write_direct_signals=self.direct;v33.TantivyBackend=self.base_backend
        v33._write_address_signals=self.address
    def close(self):
        self.pool.shutdown(wait=True)
        if self.official is not None:self.official._search_pool.shutdown(wait=True)
        if self.side is not None:self.side.close()
