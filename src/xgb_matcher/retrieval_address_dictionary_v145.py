"""Lossless whole-address/site factorization, not a candidate search or pool.

One shared text dictionary; CP/INSEE/number and exact-current flags belong to
an explicit site proof. All raw observations stay reachable via global IDs.
Parent addresses/nonlexical observations stay in the sealed source authority.
"""
from pathlib import Path

import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_address_projection_v142 import SCHEMA
from src.xgb_matcher.retrieval_variant_factor_v119 import literal

COLUMNS=('observation_id','address_text','site_position','parent_position','postcode_code','insee_code',
    'number_text','current_exact','source_current_code','bucket')
KEYS=('address_id','site_position','parent_position','postcode_code','insee_code','number_text')


def parquet_list(paths):
    # DuckDB CREATE VIEW/TABLE statements cannot bind prepared parameters.
    return '['+','.join(literal(p) for p in paths)+']'


def partition(c,source,output,start,expected,eligible,buckets,sites,parents):
    """Every eligible explicit site proof, no source/date/status/current cap."""
    if (any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (start,expected,eligible,sites,parents))
        or not isinstance(buckets,int) or isinstance(buckets,bool) or not 1<=buckets<=254
        or start+expected>=2**32 or max(sites,parents)>=2**32-1):
        raise ValueError('Exact complete observation/namespace/storage range required')
    output=Path(output);f=pq.ParquetFile(source)
    if f.schema_arrow!=SCHEMA or f.metadata.num_rows!=expected:raise ValueError('Whole original address projection required')
    c.execute('CREATE TEMP VIEW original145 AS SELECT * FROM read_parquet('+literal(source)+',file_row_number=true)')
    n,nlex,bad=c.execute('''SELECT count(*),count(*) FILTER(WHERE lexical_eligible),count(*) FILTER(WHERE
        observation_id IS NULL OR observation_id<>file_row_number+? OR lexical_eligible IS NULL OR
        lexical_eligible IS DISTINCT FROM (subject_code=2 AND address_text<>'') OR
        (lexical_eligible AND (regexp_full_match(address_text,'[A-Z0-9]+( [A-Z0-9]+)*') IS NOT TRUE OR
        site_position IS NULL OR site_position>=? OR parent_position IS NULL OR parent_position>=? OR
        number_text IS NULL OR current_exact IS NULL OR postcode_code IS NULL OR insee_code IS NULL))) FROM original145''',
        [start,sites,parents]).fetchone()
    if n!=expected or nlex!=eligible or bad:raise ValueError('Every original address proof/grain/ordinal required unchanged')
    c.execute('''CREATE TEMP TABLE rows145 AS SELECT observation_id,address_text,site_position,parent_position,
        postcode_code,insee_code,number_text,current_exact,
        CAST(CASE WHEN source_is_current IS NULL THEN 0 WHEN source_is_current THEN 2 ELSE 1 END AS UTINYINT) source_current_code,
        CAST(hash(address_text)%? AS UTINYINT) bucket FROM original145 WHERE lexical_eligible''',[buckets])
    if nlex:
        c.execute('COPY rows145 TO '+literal(output/'staging')+
            ' (FORMAT PARQUET,PARTITION_BY(bucket),WRITE_PARTITION_COLUMNS true,COMPRESSION ZSTD)')
        files=sorted((output/'staging').glob('bucket=*/*.parquet'))
        c.execute('CREATE TEMP VIEW staged145 AS SELECT * FROM read_parquet('+parquet_list(files)+',hive_partitioning=false)')
        count,unique,badbucket=c.execute('SELECT count(*),count(DISTINCT observation_id),count(*) FILTER(WHERE bucket<>hash(address_text)%?) FROM staged145',[buckets]).fetchone()
        diff=' OR '.join('o.'+k+' IS DISTINCT FROM s.'+k for k in COLUMNS)
        bad=c.execute('SELECT count(*) FROM rows145 o FULL OUTER JOIN staged145 s USING(observation_id) WHERE '
            'o.observation_id IS NULL OR s.observation_id IS NULL OR '+diff).fetchone()[0]
        if count!=nlex or count!=unique or badbucket or bad:raise ValueError('Every original address field/site/flag must reconstruct through staging')
    else:files=[]
    return dict(original_observations=expected,lexical_observations=nlex,context_observations=expected-nlex,
        files=[str(p) for p in files],all_eligible_rows_unique_and_exact=True,physical_hash_only=True,
        original_source_provenance_and_nonlexical_rows_preserved=True,proof_or_text_or_owner_cap=None,
        identifiers_as_features=False,parent_address_spread_to_sites=False)


def dictionary(c,inputs,output,bucket,buckets,expected):
    """Whole bucket, full text equality (not hash equality), exact proof keys."""
    if (any(not isinstance(v,int) or isinstance(v,bool) for v in (bucket,buckets,expected))
        or not 1<=buckets<=254 or not 0<=bucket<buckets or not 0<=expected<2**32):
        raise ValueError('Exact complete observation/storage range required')
    output=Path(output)
    if inputs:
        c.execute('CREATE TEMP TABLE bucket145 AS SELECT * FROM read_parquet('+parquet_list(inputs)+',hive_partitioning=false)')
    else:
        c.execute('''CREATE TEMP TABLE bucket145(observation_id UINTEGER,address_text VARCHAR,site_position UINTEGER,
            parent_position UINTEGER,postcode_code UINTEGER,insee_code UINTEGER,number_text VARCHAR,
            current_exact BOOLEAN,source_current_code UTINYINT,bucket UTINYINT)''')
    n,unique,bad=c.execute('''SELECT count(*),count(DISTINCT observation_id),count(*) FILTER(WHERE
        bucket IS NULL OR bucket<>? OR bucket<>hash(address_text)%? OR address_text IS NULL OR
        regexp_full_match(address_text,'[A-Z0-9]+( [A-Z0-9]+)*') IS NOT TRUE OR
        site_position IS NULL OR parent_position IS NULL OR postcode_code IS NULL OR insee_code IS NULL OR
        number_text IS NULL OR current_exact IS NULL OR source_current_code IS NULL OR source_current_code>2)
        FROM bucket145''',[bucket,buckets]).fetchone()
    if n!=expected or n!=unique or bad:raise ValueError('Whole bucket and every unique original observation required')
    c.execute('''CREATE TEMP TABLE addresses145 AS SELECT CAST(row_number() OVER(ORDER BY address_text)-1 AS UINTEGER) address_id,
        address_text FROM (SELECT DISTINCT address_text FROM bucket145)''')
    keys=','.join(KEYS)
    c.execute('CREATE TEMP TABLE facts145 AS SELECT CAST(row_number() OVER(ORDER BY '+keys+')-1 AS UINTEGER) fact_id,* '
        'FROM (SELECT '+keys+',bool_or(current_exact) current_exact FROM bucket145 JOIN addresses145 USING(address_text) GROUP BY '+keys+')')
    for name,table,order in (('addresses','addresses145','address_id'),('facts','facts145','fact_id')):
        c.execute('COPY (SELECT * FROM '+table+' ORDER BY '+order+') TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/(name+'.parquet'))])
    join=' AND '.join('b.'+k+'=f.'+k for k in KEYS if k!='address_id')+' AND a.address_id=f.address_id'
    c.execute('COPY (SELECT f.fact_id,b.observation_id FROM bucket145 b JOIN addresses145 a USING(address_text) '
        'JOIN facts145 f ON '+join+' ORDER BY f.fact_id,b.observation_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'observation_links.parquet')])
    for view,name in (('out_addresses145','addresses'),('out_facts145','facts'),('out_links145','observation_links')):
        c.execute('CREATE TEMP VIEW '+view+' AS SELECT * FROM read_parquet('+literal(output/(name+'.parquet'))+')')
    for name,identity,key in (('addresses','address_id','address_text'),('facts','fact_id','('+keys+')')):
        n,ids,k,bad=c.execute('SELECT count(*),count(DISTINCT '+identity+'),count(DISTINCT '+key+'),count(*) FILTER(WHERE '
            +identity+' IS NULL OR '+identity+'<>file_row_number) FROM read_parquet('+literal(output/(name+'.parquet'))+',file_row_number=true)').fetchone()
        if n!=ids or n!=k or bad:raise ValueError('Every address/fact key must be unique, exact and contiguous')
    n,unique=c.execute('SELECT count(*),count(DISTINCT observation_id) FROM out_links145').fetchone()
    if n!=expected or n!=unique:raise ValueError('Every original eligible address observation linked exactly once')
    diff=' OR '.join('b.'+k+' IS DISTINCT FROM f.'+k for k in KEYS if k!='address_id')
    bad=c.execute('SELECT count(*) FROM bucket145 b FULL OUTER JOIN out_links145 l USING(observation_id) '
        'LEFT JOIN out_facts145 f ON l.fact_id=f.fact_id LEFT JOIN out_addresses145 a ON f.address_id=a.address_id '
        'WHERE b.observation_id IS NULL OR l.observation_id IS NULL OR f.fact_id IS NULL OR a.address_id IS NULL '
        'OR b.address_text IS DISTINCT FROM a.address_text OR '+diff).fetchone()[0]
    exact_bad=c.execute('SELECT count(*) FROM facts145 f FULL OUTER JOIN out_facts145 o USING(fact_id) '
        'WHERE f.fact_id IS NULL OR o.fact_id IS NULL OR f.current_exact IS DISTINCT FROM o.current_exact').fetchone()[0]
    if bad or exact_bad:raise ValueError('Every original site/address/CP/INSEE/number/exact flag must reconstruct')
    return dict(bucket=bucket,observations=expected,addresses=pq.ParquetFile(output/'addresses.parquet').metadata.num_rows,
        facts=pq.ParquetFile(output/'facts.parquet').metadata.num_rows,every_observation_linked_once=True,
        every_site_address_and_attributes_reconstructed_exactly=True,current_exact_any_true_only=True,
        all_original_source_flags_and_provenance_retained_in_authorities=True,proof_or_text_or_owner_cap=None,
        identifiers_as_features=False,parent_address_spread_to_sites=False)
