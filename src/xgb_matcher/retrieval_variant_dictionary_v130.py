"""Exact all-observation name/owner factorization, physically hashed by text.

Hashes select storage buckets ONLY; full text remains the equality key. IDs
are joins/array positions only. Original27-field observations stay sealed.
"""
from pathlib import Path
import time

import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variant_factor_v119 import literal
from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA

SCOPES={'LEGAL_PERSON_NAME':1,'LEGAL_PERSON_COMMERCIAL_NAME':2,'EXPLICIT_ENDPOINT_NAME':3}
COLUMNS=('observation_id','name_text','parent_key','owner_key','subject_code','scope_code','current_code','bucket')
FACT_COLUMNS=('name_id','parent_key','owner_key','subject_code','scope_code')


def partition(c,source,output,start,expected,eligible,buckets,progress=lambda value:None):
    """All eligible observations once; every nonlexical row stays in authority.

    Eligibility belongs to source proof scope/mask, NEVER human denominators.
    Original normalization is not replaced here. The future feature encoder
    still applies its identifier/mask guard to the stored dictionary strings.
    """
    output=Path(output)
    if (not isinstance(buckets,int) or isinstance(buckets,bool) or not 1<=buckets<=254
        or not isinstance(start,int) or start<0 or start+expected>=2**32):
        raise ValueError('Typed observation/storage range')
    f=pq.ParquetFile(source)
    if f.metadata.num_rows!=expected or not f.schema_arrow.equals(OBS_SCHEMA):raise ValueError('Every original observation/schema required')
    c.execute('CREATE TEMP VIEW original130 AS SELECT * FROM read_parquet('+literal(source)+',file_row_number=true)')
    count,nlex,bad=c.execute("""SELECT count(*),count(*) FILTER(WHERE lexical_eligible),count(*) FILTER(WHERE
        lexical_eligible IS NULL OR (lexical_eligible AND (
        regexp_full_match(name_text,'[A-Z0-9]+( [A-Z0-9]+)*') IS NOT TRUE OR
        regexp_full_match(siren,'[0-9]{9}') IS NOT TRUE OR
        CASE WHEN subject_kind='SIREN' THEN siret IS DISTINCT FROM '' OR lexical_scope IS NULL OR lexical_scope NOT IN
        ('LEGAL_PERSON_NAME','LEGAL_PERSON_COMMERCIAL_NAME') WHEN subject_kind='SIRET' THEN
        regexp_full_match(siret,'[0-9]{14}') IS NOT TRUE OR substr(siret,1,9)<>siren OR
        lexical_scope IS DISTINCT FROM 'EXPLICIT_ENDPOINT_NAME' ELSE TRUE END))) FROM original130""").fetchone()
    if count!=expected or nlex!=eligible or bad:raise ValueError('Whole source scope/text/count drift')
    scope='CASE lexical_scope '+''.join("WHEN '"+k+"' THEN "+str(v)+' ' for k,v in SCOPES.items())+'END'
    c.execute('''CREATE TEMP TABLE rows130 AS SELECT CAST(file_row_number+'''+str(start)+''' AS UINTEGER) observation_id,
        name_text,CAST(siren AS UINTEGER) parent_key,CAST(CASE WHEN subject_kind='SIREN' THEN siren ELSE siret END AS UBIGINT) owner_key,
        CAST(CASE WHEN subject_kind='SIREN' THEN 1 ELSE 2 END AS UTINYINT) subject_code,
        CAST('''+scope+''' AS UTINYINT) scope_code,
        CAST(CASE WHEN is_current IS NULL THEN 0 WHEN is_current THEN 2 ELSE 1 END AS UTINYINT) current_code,
        CAST(hash(name_text)%'''+str(buckets)+''' AS UTINYINT) bucket FROM original130 WHERE lexical_eligible''')
    began=time.monotonic()
    if nlex:
        c.execute('COPY rows130 TO '+literal(output/'staging')+
            ' (FORMAT PARQUET,PARTITION_BY(bucket),WRITE_PARTITION_COLUMNS true,COMPRESSION ZSTD)')
        files=sorted((output/'staging').glob('bucket=*/*.parquet'))
        c.execute('CREATE TEMP VIEW staged130 AS SELECT * FROM read_parquet(['+','.join(literal(p) for p in files)+'],hive_partitioning=false)')
        count,unique,misbucket=c.execute('SELECT count(*),count(DISTINCT observation_id),count(*) FILTER(WHERE bucket<>hash(name_text)%'+str(buckets)+') FROM staged130').fetchone()
        different=' OR '.join('s.'+k+' IS DISTINCT FROM t.'+k for k in COLUMNS)
        bad=c.execute('SELECT count(*) FROM rows130 s FULL OUTER JOIN staged130 t USING(observation_id) WHERE '
            's.observation_id IS NULL OR t.observation_id IS NULL OR '+different).fetchone()[0]
        if count!=nlex or unique!=count or misbucket or bad:raise ValueError('Every staged observation/full text/owner/current must reconstruct exactly')
    else:files=[]
    progress(dict(phase='partition_and_exact_all_row_audit',seconds=time.monotonic()-began))
    return dict(original_observations=expected,lexical_observations=nlex,
        context_observations=expected-nlex,files=[str(p) for p in files],all_eligible_rows_unique_and_exact=True,
        all_original_fields_preserved_in_authority=True,storage_buckets=buckets,
        physical_hash_only=True,source_current_unknown_preserved=True,name_or_owner_caps=None)


def dictionary(c,inputs,output,bucket,buckets,expected,progress=lambda value:None):
    """Whole bucket from EVERY physical part, globally unique names and facts.

    Each eligible observation maps once. No date/current/source deduplication:
    fact links point to ALL original observations, not only latest records.
    """
    output=Path(output);phases={}
    def execute(phase,sql,params=None):
        began=time.monotonic();r=c.execute(sql,params or []);phases[phase]=time.monotonic()-began
        progress(dict(phase=phase,seconds=phases[phase]));return r
    if inputs:
        c.execute('CREATE TEMP TABLE bucket130 AS SELECT * FROM read_parquet(?,hive_partitioning=false)',[list(map(str,inputs))])
    else:
        c.execute('''CREATE TEMP TABLE bucket130(observation_id UINTEGER,name_text VARCHAR,parent_key UINTEGER,
            owner_key UBIGINT,subject_code UTINYINT,scope_code UTINYINT,current_code UTINYINT,bucket UTINYINT)''')
    count,unique,bad=c.execute('SELECT count(*),count(DISTINCT observation_id),count(*) FILTER(WHERE bucket<>'+str(bucket)+
        ' OR bucket<>hash(name_text)%'+str(buckets)+') FROM bucket130').fetchone()
    if count!=expected or unique!=count or bad:raise ValueError('Every whole-bucket observation/physical hash required')
    execute('unique_global_names','CREATE TEMP TABLE names130 AS SELECT CAST(row_number() OVER(ORDER BY name_text)-1 AS UINTEGER) name_id,name_text '
        'FROM (SELECT DISTINCT name_text FROM bucket130)')
    keys=','.join(FACT_COLUMNS)
    execute('unique_global_owner_facts','CREATE TEMP TABLE facts130 AS SELECT CAST(row_number() OVER(ORDER BY '+keys+
        ')-1 AS UINTEGER) fact_id,* FROM (SELECT '+keys+',bool_or(current_code=2) current_exact '
        'FROM bucket130 JOIN names130 USING(name_text) GROUP BY '+keys+')')
    execute('export_names','COPY (SELECT * FROM names130 ORDER BY name_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'names.parquet')])
    execute('export_facts','COPY (SELECT * FROM facts130 ORDER BY fact_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'facts.parquet')])
    join=' AND '.join('b.'+k+'=f.'+k for k in FACT_COLUMNS if k!='name_id')+' AND n.name_id=f.name_id'
    execute('export_all_observation_links','COPY (SELECT f.fact_id,b.observation_id FROM bucket130 b '
        'JOIN names130 n USING(name_text) JOIN facts130 f ON '+join+' ORDER BY f.fact_id,b.observation_id) '
        'TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'observation_links.parquet')])
    c.execute('CREATE TEMP VIEW out_names130 AS SELECT * FROM read_parquet('+literal(output/'names.parquet')+')')
    c.execute('CREATE TEMP VIEW out_facts130 AS SELECT * FROM read_parquet('+literal(output/'facts.parquet')+')')
    c.execute('CREATE TEMP VIEW out_links130 AS SELECT * FROM read_parquet('+literal(output/'observation_links.parquet')+')')
    for table,identity,key in (('out_names130','name_id','name_text'),('out_facts130','fact_id','('+keys+')')):
        n,ids,k,misorder=c.execute('SELECT count(*),count(DISTINCT '+identity+'),count(DISTINCT '+key+'),'
            'count(*) FILTER(WHERE '+identity+' IS NULL OR '+identity+'<>file_row_number) FROM read_parquet('+
            literal(output/('names.parquet' if identity=='name_id' else 'facts.parquet'))+',file_row_number=true)').fetchone()
        if n!=ids or n!=k or misorder:raise ValueError('Every name/fact dictionary key/id must be unique and contiguous')
    count,unique=c.execute('SELECT count(*),count(DISTINCT observation_id) FROM out_links130').fetchone()
    if count!=expected or unique!=count:raise ValueError('All eligible observation links required once')
    diffs=' OR '.join('b.'+k+' IS DISTINCT FROM f.'+k for k in FACT_COLUMNS if k!='name_id')
    bad=execute('every_observation_owner_name_reconstruction','SELECT count(*) FROM bucket130 b FULL OUTER JOIN out_links130 l USING(observation_id) '
        'LEFT JOIN out_facts130 f ON l.fact_id=f.fact_id LEFT JOIN out_names130 n ON f.name_id=n.name_id '
        'WHERE b.observation_id IS NULL OR l.observation_id IS NULL OR f.fact_id IS NULL OR n.name_id IS NULL OR '
        'b.name_text IS DISTINCT FROM n.name_text OR '+diffs).fetchone()[0]
    current_bad=c.execute('SELECT count(*) FROM facts130 f FULL OUTER JOIN out_facts130 o USING(fact_id) WHERE '
        'f.fact_id IS NULL OR o.fact_id IS NULL OR f.current_exact IS DISTINCT FROM o.current_exact').fetchone()[0]
    if bad or current_bad:raise ValueError('Full original owner/name/current reconstruction drift')
    return dict(bucket=bucket,observations=expected,names=pq.ParquetFile(output/'names.parquet').metadata.num_rows,
        facts=pq.ParquetFile(output/'facts.parquet').metadata.num_rows,
        every_eligible_observation_linked_once=True,every_owner_and_name_reconstructed_exactly=True,
        phases=phases,original27_fields_preserved_in_authorities=True,current_exact_any_true_only=True,
        name_or_owner_or_period_caps=None,site_cartesian_product_materialized=False)
