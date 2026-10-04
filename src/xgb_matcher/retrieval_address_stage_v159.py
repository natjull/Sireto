"""V147 exact reverse staging with bounded concurrent physical writers.

Only COPY fanout changes: all original facts/fields/joins remain audited.
No site/address/proof cap or candidate filter. This kernel does not certify
national completeness, resources, recall or SLO; V154's actual STOP is intact.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .retrieval_address_reverse_v147 import FACT_SCHEMA, STAGE_SCHEMA
from .retrieval_address_dictionary_v145 import parquet_list
from .retrieval_variant_factor_v119 import literal
from .retrieval_v20_workbench import record, verify


def stage(c, source, output, keys, source_bucket, alias_start, aliases, facts,
          site_quantum=65536, writer_group_parts=16):
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in (source_bucket, alias_start, aliases, facts))
            or source_bucket >= 254 or alias_start+aliases >= 2**32 or facts >= 2**32
            or not isinstance(site_quantum, int) or isinstance(site_quantum, bool) or site_quantum <= 0
            or not isinstance(writer_group_parts, int) or isinstance(writer_group_parts, bool)
            or not 1 <= writer_group_parts <= 16):
        raise ValueError('Complete typed storage ranges and bounded physical writer group required')
    output = Path(output)
    if (output/'staging').exists():
        raise ValueError('Immutable reverse storage already started')
    ref = record(source)
    f = pq.ParquetFile(source)
    if f.schema_arrow != FACT_SCHEMA or f.metadata.num_rows != facts:
        raise ValueError('Whole original V145 facts required')
    c.execute('CREATE TEMP VIEW original159 AS SELECT * FROM read_parquet('+literal(source)+',file_row_number=true)')
    n, bad = c.execute('''SELECT count(*),count(*) FILTER(WHERE fact_id IS NULL OR fact_id<>file_row_number
        OR address_id IS NULL OR address_id>=? OR site_position IS NULL OR site_position>=?
        OR parent_position IS NULL OR parent_position>=? OR current_exact IS NULL
        OR postcode_code IS NULL OR (postcode_code<>4294967295 AND postcode_code>=60466176)
        OR insee_code IS NULL OR (insee_code<>4294967295 AND insee_code>=60466176)
        OR number_text IS NULL OR regexp_full_match(number_text,'([A-Z0-9]+( [A-Z0-9]+)*)?') IS NOT TRUE)
        FROM original159''', [aliases, keys.sites, keys.parents]).fetchone()
    if n != facts or bad:
        raise ValueError('Every original address fact/ordinal/attribute required')
    for batch in f.iter_batches(batch_size=131072, columns=['site_position', 'parent_position'], use_threads=False):
        site = batch.column(0).to_numpy(zero_copy_only=False)
        parent = batch.column(1).to_numpy(zero_copy_only=False)
        raw = keys.reconstruct(site, 'site')
        rp = keys.reconstruct(parent, 'parent')
        if not np.array_equal((raw//np.uint64(100000)).astype(np.uint32), rp):
            raise ValueError('Every original explicit address site/parent must reconstruct exactly')
    c.execute('''CREATE TEMP TABLE staged_rows159 AS SELECT site_position,parent_position,
        CAST(address_id+? AS UINTEGER) alias_position,postcode_code,insee_code,number_text,current_exact,
        CAST(? AS UTINYINT) source_bucket,fact_id source_fact_id,
        CAST(site_position//? AS UINTEGER) site_part FROM original159''',
        [alias_start, source_bucket, site_quantum])
    copies = maximum = 0
    if facts:
        parts = [int(row[0]) for row in c.execute('SELECT DISTINCT site_part FROM staged_rows159 ORDER BY site_part').fetchall()]
        # A GROUP IS STORAGE ONLY. Every observed physical part appears in
        # exactly one COPY; write disjoint directories without overwrite.
        for lo in range(0, len(parts), writer_group_parts):
            group = parts[lo:lo+writer_group_parts]
            c.execute('COPY (SELECT * FROM staged_rows159 WHERE site_part IN ('+
                      ','.join(str(p) for p in group)+')) TO '+literal(output/'staging')+
                      ' (FORMAT PARQUET,PARTITION_BY(site_part),WRITE_PARTITION_COLUMNS true,'
                      'APPEND true,COMPRESSION ZSTD)')
            copies += 1
            maximum = max(maximum, len(group))
        files = sorted((output/'staging').glob('site_part=*/*.parquet'))
        c.execute('CREATE TEMP VIEW exported159 AS SELECT * FROM read_parquet('+parquet_list(files)+',hive_partitioning=false)')
        n, unique, badpart = c.execute('''SELECT count(*),count(DISTINCT(source_bucket,source_fact_id)),
            count(*) FILTER(WHERE site_part<>site_position//?) FROM exported159''', [site_quantum]).fetchone()
        diff = ' OR '.join('s.'+k+' IS DISTINCT FROM e.'+k for k in (*STAGE_SCHEMA.names, 'site_part'))
        bad = c.execute('SELECT count(*) FROM staged_rows159 s FULL OUTER JOIN exported159 e '
                        'USING(source_bucket,source_fact_id) WHERE s.source_fact_id IS NULL '
                        'OR e.source_fact_id IS NULL OR '+diff).fetchone()[0]
        if n != facts or unique != facts or badpart or bad:
            raise ValueError('Every staged address fact must reconstruct exactly')
    else:
        files = []
    verify(ref)
    return dict(facts=facts, files=[str(p) for p in files], source=ref,
                every_fact_staged_once=True, every_original_site_parent_and_attribute_exact=True,
                site_quantum=site_quantum, site_or_proof_or_alias_cap=None, identifiers_as_features=False,
                writer_group_parts=writer_group_parts, copy_operations=copies,
                maximum_physical_writers_per_copy=maximum, physical_writer_group_is_not_a_candidate_cap=True)
