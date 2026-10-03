"""Complete per-name word postings, scoped-owner access and observation metadata."""
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_facts_v85 import literal
from src.xgb_matcher.retrieval_proof_v56 import signal

PATTERN='(^|[^0-9])([0-9]{14}|[0-9]{9})([^0-9]|$)'
SOURCE={'RNE':1,'BODACC':2,'SIRENE_CURRENT':3}
NAME_KIND={'LEGAL':1,'SIGN':2,'TRADE':3,'USUAL':4}


def masked_sql(column):
    # Two passes handle shared digit boundaries of adjacent identifiers.
    # Only canonical ASCII uppercase input is accepted below; no surrogate
    # SQL Unicode normalizer silently replaces the authoritative V56 signal.
    pattern="'"+PATTERN+"'"
    once='regexp_replace('+column+', '+pattern+", '\\1 \\3', 'g')"
    twice='regexp_replace('+once+', '+pattern+", '\\1 \\3', 'g')"
    return "trim(regexp_replace("+twice+", ' +', ' ', 'g'))"


def build_postings(c,parts,output,progress=lambda value:None):
    output=Path(output); nf=sum(p['facts'] for p in parts); clauses=[]
    for part in parts:
        clauses.append('SELECT CAST(file_row_number+'+str(part['fact_start'])+
            ' AS UINTEGER) fact_id,* EXCLUDE(file_row_number) FROM read_parquet('+literal(part['path'])+',file_row_number=true)')
    c.execute('CREATE TEMP VIEW name_facts AS '+' UNION ALL '.join(clauses))
    if c.execute("SELECT count(*) FROM name_facts WHERE NOT regexp_full_match(name_text,'[A-Z0-9]+( [A-Z0-9]+)*')").fetchone()[0]:
        raise ValueError('Noncanonical source text: authoritative normalization required before SQL mask')
    mask=masked_sql('name_text')
    masked='SELECT *, '+mask+' masked_text FROM name_facts'
    counts=c.execute('SELECT count(*),count(*) FILTER(WHERE scope_code=0),count(*) FILTER(WHERE scope_code<>0 AND masked_text=\'\') FROM ('+masked+')').fetchone()
    if counts[0]!=nf: raise ValueError('Fact denominator drift')
    began=time.monotonic()
    c.execute('''COPY (SELECT fact_id,token,hash(token)%32 token_bucket FROM (
        SELECT fact_id,unnest(list_distinct(string_split(masked_text,' '))) token FROM ('''+masked+''')
        WHERE scope_code<>0) WHERE token<>'') TO '''+literal(output/'word_staging')+
        ' (FORMAT PARQUET,PARTITION_BY(token_bucket),COMPRESSION ZSTD)')
    progress(dict(phase='all_masked_name_word_postings',seconds=time.monotonic()-began))
    postings=0; buckets=[]
    for bucket in range(32):
        folder=output/'word_staging'/f'token_bucket={bucket}'
        if not folder.exists(): continue
        began=time.monotonic(); sorted_path=output/f'words_sorted{bucket:02d}.parquet'
        c.execute('COPY (SELECT token,fact_id FROM read_parquet('+literal(folder/'*.parquet')+
                  ',hive_partitioning=false) ORDER BY token,fact_id) TO '+literal(sorted_path)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
        vocab=output/f'word_vocabulary{bucket:02d}.parquet'
        c.execute('''COPY (SELECT token,CAST(min(file_row_number) AS UINTEGER) posting_start,
            CAST(max(file_row_number)+1 AS UINTEGER) posting_stop,CAST(count(*) AS UINTEGER) postings
            FROM read_parquet('''+literal(sorted_path)+''',file_row_number=true) GROUP BY token ORDER BY token)
            TO '''+literal(vocab)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
        values=pq.read_table(sorted_path,columns=['fact_id'],use_threads=False)['fact_id'].to_numpy(zero_copy_only=False)
        if len(values)>=2**32 or len(values) and np.max(values)>=nf: raise ValueError('Word posting identity range drift')
        np.save(output/f'word_postings{bucket:02d}.npy',values,allow_pickle=False)
        words=pq.ParquetFile(vocab).metadata.num_rows; postings+=len(values)
        buckets.append(dict(bucket=bucket,vocabulary=str(vocab),postings_path=str(output/f'word_postings{bucket:02d}.npy'),
                            words=words,postings=len(values)))
        progress(dict(phase='sorted_word_bucket',bucket=bucket,postings=len(values),words=words,seconds=time.monotonic()-began))
    c.execute('DROP VIEW name_facts')
    return dict(facts=nf,context_only_facts=counts[1],identifier_only_admissible_facts=counts[2],
        word_postings=postings,vocabulary_words=sum(b['words'] for b in buckets),buckets=buckets,
        identifier_mask_applied=True,per_fact_word_dedup=True,whole_source_ngrams_not_indexed=True,
        same_alias_word_bags_only=True,unknown_name_scope_not_used_as_lexical_proof=True)


def build_owners(parts,output,progress=lambda value:None):
    output=Path(output); nf=sum(p['facts'] for p in parts)
    specs={'fact_owners':np.uint64,'fact_parents':np.uint32,'fact_subjects':np.uint8,'fact_scopes':np.uint8}
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(nf,)) for k,d in specs.items()}
    for part in parts:
        table=pq.read_table(part['path'],columns=['owner_key','official_parent','subject_code','scope_code'],use_threads=False)
        lo=part['fact_start']; hi=lo+part['facts']
        for dest,source in zip(specs,table.column_names): arrays[dest][lo:hi]=table[source].to_numpy(zero_copy_only=False)
    stats={}
    for subject,label in ((1,'parent'),(2,'site')):
        chosen=np.flatnonzero(arrays['fact_subjects']==subject)
        order=np.argsort(arrays['fact_owners'][chosen],kind='stable'); refs=chosen[order].astype(np.uint32)
        keys=arrays['fact_owners'][refs]; starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1] if len(keys) else np.array([],dtype=int)
        offsets=np.r_[starts,len(refs)].astype(np.uint32)
        np.save(output/(label+'_owner_keys.npy'),keys[starts],allow_pickle=False)
        np.save(output/(label+'_owner_offsets.npy'),offsets,allow_pickle=False)
        np.save(output/(label+'_owner_facts.npy'),refs,allow_pickle=False)
        stats[label]=dict(owners=len(starts),facts=len(refs),max_facts_per_owner=int(np.max(np.diff(offsets))) if len(refs) else 0)
    for value in arrays.values(): value.flush()
    if sum(v['facts'] for v in stats.values())!=nf: raise ValueError('Owner grain denominator drift')
    progress(dict(phase='all_owner_fact_references',counts=stats))
    return stats


def build_metadata(source,output,progress=lambda value:None):
    output=Path(output); file=pq.ParquetFile(source); n=file.metadata.num_rows
    specs={**{k:np.int32 for k in ('valid_from_days','valid_to_days','observed_days')},
           **{k:np.uint8 for k in ('source_codes','kind_codes','current_codes','priority_known')},'priorities':np.int16}
    arrays={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=d,shape=(n,)) for k,d in specs.items()}
    columns=['valid_from','valid_to','observed_at','source','name_kind','is_current','source_priority']; offset=0
    for batch in file.iter_batches(batch_size=131072,columns=columns,use_threads=False):
        stop=offset+len(batch); values={k:batch.column(i) for i,k in enumerate(columns)}
        for dest,key in zip(('valid_from_days','valid_to_days','observed_days'),columns[:3]):
            if values[key].type!=pa.date32(): raise ValueError('Observation date unit drift')
            arrays[dest][offset:stop]=pc.fill_null(pc.cast(values[key],pa.int32()),np.iinfo(np.int32).min).to_numpy(zero_copy_only=False)
        try:
            arrays['source_codes'][offset:stop]=[SOURCE[v] for v in values['source'].to_pylist()]
            arrays['kind_codes'][offset:stop]=[NAME_KIND[v] for v in values['name_kind'].to_pylist()]
        except KeyError as exc: raise ValueError('Unknown metadata code; no silent reinterpretation') from exc
        current=pc.if_else(pc.is_null(values['is_current']),0,pc.if_else(pc.fill_null(values['is_current'],False),2,1))
        arrays['current_codes'][offset:stop]=current.to_numpy(zero_copy_only=False)
        arrays['priority_known'][offset:stop]=pc.invert(pc.is_null(values['source_priority'])).to_numpy(zero_copy_only=False)
        arrays['priorities'][offset:stop]=pc.fill_null(values['source_priority'],0).to_numpy(zero_copy_only=False)
        offset=stop
    if offset!=n: raise ValueError('Observation metadata denominator drift')
    for value in arrays.values(): value.flush()
    progress(dict(phase='all_observation_metadata',observations=n))
    return dict(observations=n,source_codes=SOURCE,kind_codes=NAME_KIND,current_codes={'UNKNOWN':0,'FALSE':1,'TRUE':2},
                null_date_sentinel=int(np.iinfo(np.int32).min),original_provenance_strings_still_in_v82_source=True)


class WordPostings:
    """All exact literal/weighted query terms, no cap on matching name facts."""
    def __init__(self,stats):
        self.facts=stats['facts']; self.terms={}; self.postings={}
        for b in stats['buckets']:
            table=pq.read_table(b['vocabulary'],use_threads=False); bucket=b['bucket']
            self.postings[bucket]=np.load(b['postings_path'],mmap_mode='r',allow_pickle=False)
            for row in table.to_pylist():
                if row['token'] in self.terms: raise ValueError('Duplicate global word key')
                self.terms[row['token']]=(bucket,row['posting_start'],row['posting_stop'])

    def scores(self,weighted_terms):
        scores=np.zeros(self.facts,dtype=np.float64); read=0
        for term,weight in sorted(weighted_terms.items()):
            if not np.isfinite(weight) or weight<0 or signal(term)!=term or ' ' in term:
                raise ValueError('Unsafe weighted lexical word')
            if term not in self.terms or not weight: continue
            b,lo,hi=self.terms[term]; ids=self.postings[b][lo:hi]
            scores[ids]+=weight; read+=len(ids)
        return scores,dict(postings_read=read,total_facts=self.facts,fixed_name_fact_pool=None)
