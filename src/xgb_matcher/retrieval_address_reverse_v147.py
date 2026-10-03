"""Lossless address-fact -> site CSR, no candidates or corpus filtering.

Core compiler/reader only. It does not certify a national launch, make an
incomplete projection complete, fit a model or evaluate recall. Provenance
keys point back to V145 dictionary facts and ALL original observations.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_address_dictionary_v145 import parquet_list
from src.xgb_matcher.retrieval_initial_collector_v146 import AddressBatch,bounded_vector
from src.xgb_matcher.retrieval_variant_factor_v119 import literal
from src.xgb_matcher.retrieval_v20_workbench import record,verify

FACT_SCHEMA=pa.schema([(k,pa.string() if k=='number_text' else pa.bool_() if k=='current_exact' else pa.uint32())
    for k in ('fact_id','address_id','site_position','parent_position','postcode_code','insee_code','number_text','current_exact')])
STAGE_SCHEMA=pa.schema([(k,pa.string() if k=='number_text' else pa.bool_() if k=='current_exact'
    else pa.uint8() if k=='source_bucket' else pa.uint32()) for k in ('site_position','parent_position','alias_position',
    'postcode_code','insee_code','number_text','current_exact','source_bucket','source_fact_id')])
FIELDS={k:(np.bool_ if k=='current_exact' else np.uint8 if k=='source_bucket' else np.uint32)
    for k in STAGE_SCHEMA.names if k!='number_text'}


def stage(c,source,output,keys,source_bucket,alias_start,aliases,facts,site_quantum=65536):
    """Every V145 fact, explicit original owner, physically partitioned by site.

    `keys` must be the independently verified full V141 namespace. Every
    original SIRET/parent join is reconstructed here; identities never enter
    similarity features. Physical quantum is NOT a fact/member/candidate cap.
    """
    if (any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (source_bucket,alias_start,aliases,facts))
        or source_bucket>=254 or alias_start+aliases>=2**32 or facts>=2**32
        or not isinstance(site_quantum,int) or isinstance(site_quantum,bool) or site_quantum<=0):
        raise ValueError('Complete typed dictionary/site storage ranges required')
    output=Path(output)
    if (output/'staging').exists():raise ValueError('Immutable reverse storage already started')
    ref=record(source);f=pq.ParquetFile(source)
    if f.schema_arrow!=FACT_SCHEMA or f.metadata.num_rows!=facts:raise ValueError('Whole original V145 facts required')
    c.execute('CREATE TEMP VIEW original147 AS SELECT * FROM read_parquet('+literal(source)+',file_row_number=true)')
    n,bad=c.execute('''SELECT count(*),count(*) FILTER(WHERE fact_id IS NULL OR fact_id<>file_row_number
        OR address_id IS NULL OR address_id>=? OR site_position IS NULL OR site_position>=?
        OR parent_position IS NULL OR parent_position>=? OR current_exact IS NULL
        OR postcode_code IS NULL OR (postcode_code<>4294967295 AND postcode_code>=60466176)
        OR insee_code IS NULL OR (insee_code<>4294967295 AND insee_code>=60466176)
        OR number_text IS NULL OR regexp_full_match(number_text,'([A-Z0-9]+( [A-Z0-9]+)*)?') IS NOT TRUE)
        FROM original147''',[aliases,keys.sites,keys.parents]).fetchone()
    if n!=facts or bad:raise ValueError('Every original address fact/ordinal/attribute required')
    for batch in f.iter_batches(batch_size=131072,columns=['site_position','parent_position'],use_threads=False):
        site=batch.column(0).to_numpy(zero_copy_only=False);parent=batch.column(1).to_numpy(zero_copy_only=False)
        raw=keys.reconstruct(site,'site');rp=keys.reconstruct(parent,'parent')
        if not np.array_equal((raw//np.uint64(100000)).astype(np.uint32),rp):
            raise ValueError('Every original explicit address site/parent must reconstruct exactly')
    c.execute('''CREATE TEMP TABLE staged_rows147 AS SELECT site_position,parent_position,
        CAST(address_id+? AS UINTEGER) alias_position,postcode_code,insee_code,number_text,current_exact,
        CAST(? AS UTINYINT) source_bucket,fact_id source_fact_id,
        CAST(site_position//? AS UINTEGER) site_part FROM original147''',[alias_start,source_bucket,site_quantum])
    if facts:
        c.execute('COPY staged_rows147 TO '+literal(output/'staging')+
            ' (FORMAT PARQUET,PARTITION_BY(site_part),WRITE_PARTITION_COLUMNS true,COMPRESSION ZSTD)')
        files=sorted((output/'staging').glob('site_part=*/*.parquet'))
        c.execute('CREATE TEMP VIEW exported147 AS SELECT * FROM read_parquet('+parquet_list(files)+',hive_partitioning=false)')
        n,unique,badpart=c.execute('SELECT count(*),count(DISTINCT(source_bucket,source_fact_id)),count(*) FILTER(WHERE site_part<>site_position//?) FROM exported147',[site_quantum]).fetchone()
        diff=' OR '.join('s.'+k+' IS DISTINCT FROM e.'+k for k in (*STAGE_SCHEMA.names,'site_part'))
        bad=c.execute('SELECT count(*) FROM staged_rows147 s FULL OUTER JOIN exported147 e USING(source_bucket,source_fact_id) '
            'WHERE s.source_fact_id IS NULL OR e.source_fact_id IS NULL OR '+diff).fetchone()[0]
        if n!=facts or unique!=facts or badpart or bad:raise ValueError('Every staged address fact must reconstruct exactly')
    else:files=[]
    verify(ref)
    return dict(facts=facts,files=[str(p) for p in files],source=ref,every_fact_staged_once=True,
        every_original_site_parent_and_attribute_exact=True,site_quantum=site_quantum,
        site_or_proof_or_alias_cap=None,identifiers_as_features=False)


def pack(c,sources,output,start,stop,sites,aliases,expected,batch_rows=131072):
    """Whole physical site interval, no pruning; all fields checked on reread."""
    if (any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (start,stop,sites,aliases,expected))
        or not 0<=start<stop<=sites<2**32-1 or aliases>=2**32 or expected>=2**32
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0):
        raise ValueError('Whole typed site interval/fact namespace required')
    output=Path(output)
    if (any(output.glob('*.npy')) or any((output/k).exists() for k in
        ('ordered_facts.parquet','numbers.parquet','manifest.json','worker_stop.json'))):
        raise ValueError('Immutable reverse interval already started')
    refs=[record(p) for p in sources]
    if sources:
        c.execute('CREATE TEMP VIEW inputs147 AS SELECT '+','.join(STAGE_SCHEMA.names)+
            ' FROM read_parquet('+parquet_list(sources)+',hive_partitioning=false)')
    else:
        c.register('empty147',pa.Table.from_batches([],schema=STAGE_SCHEMA))
        c.execute('CREATE TEMP VIEW inputs147 AS SELECT * FROM empty147')
    for p in sources:
        s=pq.ParquetFile(p).schema_arrow
        if any(s.field(k).type!=field.type for k,field in zip(STAGE_SCHEMA.names,STAGE_SCHEMA)):
            raise ValueError('Exact typed staged fact fields required')
    n,unique,bad=c.execute('''SELECT count(*),count(DISTINCT(source_bucket,source_fact_id)),count(*) FILTER(WHERE
        site_position IS NULL OR site_position<? OR site_position>=? OR alias_position IS NULL OR alias_position>=?
        OR parent_position IS NULL OR current_exact IS NULL OR source_bucket IS NULL OR source_bucket>=254
        OR source_fact_id IS NULL OR postcode_code IS NULL OR (postcode_code<>4294967295 AND postcode_code>=60466176)
        OR insee_code IS NULL OR (insee_code<>4294967295 AND insee_code>=60466176)
        OR number_text IS NULL OR regexp_full_match(number_text,'([A-Z0-9]+( [A-Z0-9]+)*)?') IS NOT TRUE)
        FROM inputs147''',[start,stop,aliases]).fetchone()
    if n!=expected or n!=unique or bad:raise ValueError('Every whole-interval fact/provenance exactly once required')
    sorted_path=output/'ordered_facts.parquet'
    c.execute('COPY (SELECT * FROM inputs147 ORDER BY site_position,source_bucket,source_fact_id) TO '+literal(sorted_path)+
        ' (FORMAT PARQUET,COMPRESSION ZSTD)')
    f=pq.ParquetFile(sorted_path)
    if f.schema_arrow!=STAGE_SCHEMA or f.metadata.num_rows!=expected:raise ValueError('Exact complete sorted fact projection required')
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(expected,)) for k,d in FIELDS.items()}
    arrays['number_codes']=np.lib.format.open_memmap(output/'number_codes.npy',mode='w+',dtype=np.uint32,shape=(expected,))
    offsets=np.lib.format.open_memmap(output/'site_offsets.npy',mode='w+',dtype=np.uint64,shape=(stop-start+1,))
    counts=np.zeros(stop-start,dtype=np.uint64);book=[''];lookup={'':0};cursor=0;last=-1
    for batch in f.iter_batches(batch_size=batch_rows,use_threads=False):
        values={k:batch.column(batch.schema.get_field_index(k)).to_numpy(zero_copy_only=False) for k in FIELDS}
        size=len(batch);sp=values['site_position']
        if np.any(sp[1:]<sp[:-1]) or (size and int(sp[0])<last):raise ValueError('Whole site ordering required')
        if size:last=int(sp[-1])
        for k,a in arrays.items():
            if k!='number_codes':a[cursor:cursor+size]=values[k]
        numbers=batch.column(batch.schema.get_field_index('number_text')).to_pylist()
        codes=np.empty(size,dtype=np.uint32)
        for i,text in enumerate(numbers):
            if text not in lookup:
                if len(book)>=2**32:raise ValueError('No number vocabulary truncation permitted')
                lookup[text]=len(book);book.append(text)
            codes[i]=lookup[text]
        arrays['number_codes'][cursor:cursor+size]=codes
        np.add.at(counts,sp-start,1);cursor+=size
    if cursor!=expected:raise ValueError('Every source fact must be exported')
    offsets[0]=0;np.cumsum(counts,dtype=np.uint64,out=offsets[1:])
    arrays['site_offsets']=offsets
    for a in arrays.values():a.flush()
    # Preserve once-normalized numbers verbatim, including compatibility
    # Unicode that became 9 ASCII digits. Do NOT normalize the dictionary again.
    pq.write_table(pa.table({'number_text':pa.array(book,type=pa.string())}),output/'numbers.parquet',compression='zstd')
    exported={k:np.load(output/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in arrays}
    words=pq.ParquetFile(output/'numbers.parquet').read()['number_text'].combine_chunks();cursor=0
    # The sorted Parquet projection itself is compared field-by-field with
    # every original input, not only with arrays made from that projection.
    diff=' OR '.join('i.'+k+' IS DISTINCT FROM o.'+k for k in STAGE_SCHEMA.names)
    bad=c.execute('SELECT count(*) FROM inputs147 i FULL OUTER JOIN read_parquet('+literal(sorted_path)+') o '
        'USING(source_bucket,source_fact_id) WHERE i.source_fact_id IS NULL OR o.source_fact_id IS NULL OR '+diff).fetchone()[0]
    if bad:raise ValueError('Every original fact must survive ordering exactly')
    for batch in f.iter_batches(batch_size=max(1,batch_rows//2),use_threads=False):
        size=len(batch)
        for k in FIELDS:
            if not np.array_equal(exported[k][cursor:cursor+size],batch.column(batch.schema.get_field_index(k)).to_numpy(zero_copy_only=False)):
                raise ValueError('Every exported address field must reconstruct exactly')
        if not pc.take(words,pa.array(exported['number_codes'][cursor:cursor+size])).equals(batch.column(batch.schema.get_field_index('number_text'))):
            raise ValueError('Every original normalized number must reconstruct exactly')
        recovered=np.searchsorted(exported['site_offsets'],np.arange(cursor,cursor+size,dtype=np.uint64),side='right')-1+start
        if not np.array_equal(recovered,exported['site_position'][cursor:cursor+size]):
            raise ValueError('Every explicit site must reconstruct through CSR exactly')
        cursor+=size
    for ref in refs:verify(ref)
    if cursor!=expected or int(offsets[-1])!=expected:raise ValueError('Whole address fact/CSR denominator required')
    for a in (*arrays.values(),*exported.values()):a._mmap.close()
    return dict(start=start,stop=stop,sites=sites,aliases=aliases,facts=expected,number_strings=len(book),sources=refs,
        artifacts={k+'.npy':record(output/(k+'.npy')) for k in arrays},numbers=record(output/'numbers.parquet'),
        ordered_source=record(sorted_path),every_original_field_and_site_csr_exact=True,
        all_provenance_keys_preserved=True,site_or_proof_or_alias_cap=None,identifiers_as_features=False,
        national_source_coverage_certified_here=False,recall_measured=False)


class ReversePart:
    """Readonly physical interval, not a national completeness certificate."""
    def __init__(self,stats):
        self.stats=stats
        if not stats['every_original_field_and_site_csr_exact'] or not stats['all_provenance_keys_preserved']:
            raise ValueError('Complete original address fact reconstruction required')
        for ref in (*stats['sources'],stats['ordered_source']):verify(ref)
        self.a={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in stats['artifacts'].items()}
        self.book=pq.ParquetFile(verify(stats['numbers'])).read()['number_text'].to_pylist()
        specs={**FIELDS,'number_codes':np.uint32,'site_offsets':np.uint64}
        if set(self.a)!=set(specs) or any(self.a[k].dtype!=d or self.a[k].shape!=
            ((stats['stop']-stats['start']+1,) if k=='site_offsets' else (stats['facts'],)) for k,d in specs.items()):
            raise ValueError('Whole typed address inverse arrays required')
        offsets=self.a['site_offsets']
        if offsets[0]!=0 or offsets[-1]!=stats['facts'] or np.any(offsets[1:]<offsets[:-1]):
            raise ValueError('Whole original site CSR required')
        if (len(self.book)!=stats['number_strings'] or self.book[:1]!=[''] or len(set(self.book))!=len(self.book)
            or np.any(self.a['number_codes']>=len(self.book))):raise ValueError('Exact original number vocabulary required')

    def close(self):
        for a in self.a.values():a._mmap.close()

    def batches(self,ids,snapshot,chunk=131072):
        if (ids.dtype!=np.uint32 or ids.ndim!=1 or not len(ids) or ids[0]<self.stats['start'] or ids[-1]>=self.stats['stop']
            or not np.array_equal(ids,np.arange(int(ids[0]),int(ids[0])+len(ids),dtype=np.uint32))
            or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0):
            raise ValueError('Exact contained physical site interval required')
        if not isinstance(snapshot,AliasSnapshot) or snapshot.aliases!=self.stats['aliases']:
            raise ValueError('Identical checked whole alias query snapshot required')
        coverage,exact=snapshot.coverage,snapshot.exact
        if coverage.flags.writeable or exact.flags.writeable:raise ValueError('Query snapshot must stay readonly')
        offsets=self.a['site_offsets'];lo=int(offsets[int(ids[0])-self.stats['start']]);hi=int(offsets[int(ids[-1])-self.stats['start']+1])
        for start in range(lo,hi,chunk):
            sl=slice(start,min(start+chunk,hi));alias=self.a['alias_position'][sl]
            if np.any(alias>=len(coverage)):raise ValueError('Every original alias position required')
            yield AddressBatch(self.a['site_position'][sl],coverage[alias],exact[alias],self.a['current_exact'][sl],
                self.a['postcode_code'][sl],self.a['insee_code'][sl],tuple(self.book[int(k)] for k in self.a['number_codes'][sl]))


class AliasSnapshot:
    """Check the whole immutable query-side alias score ONCE, not per site chunk."""
    def __init__(self,coverage,exact):
        aliases=len(coverage);bounded_vector(coverage,aliases,np.float64,'whole-address-alias')
        if (exact.dtype!=np.bool_ or exact.shape!=coverage.shape or coverage.flags.writeable or exact.flags.writeable
            or np.any(exact&(coverage!=1.))):
            raise ValueError('Readonly complete coherent alias query snapshot required')
        self.aliases=aliases;self.coverage=coverage;self.exact=exact
