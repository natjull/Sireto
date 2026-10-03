"""V120 equivalent eligible-name key enables native hash join, same V119 audit."""
import time
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variant_factor_v119 import (
    audit_roundtrip, compare_dictionaries, literal,
)


def factorize(c, source, output, progress=lambda row: None):
    phases = {}
    def execute(phase, sql, parameters=None):
        began = time.monotonic(); result = c.execute(sql, parameters or [])
        phases[phase] = time.monotonic() - began
        progress(dict(phase=phase, seconds=phases[phase]))
        return result
    execute('materialize_source_once', 'CREATE TEMP TABLE variants119 AS SELECT '
        'file_row_number observation_index,* EXCLUDE(file_row_number) FROM read_parquet(' +
        literal(source) + ',file_row_number=true)')
    execute('name_dictionary', 'CREATE TEMP TABLE names119 AS SELECT '
        'row_number() OVER(ORDER BY name_text)-1 name_id,name_text FROM '
        '(SELECT DISTINCT name_text FROM variants119 WHERE lexical_eligible)')
    keys = 'subject_kind,siren,siret,lexical_scope,name_id'
    execute('fact_dictionary', 'CREATE TEMP TABLE facts119 AS SELECT '
        'row_number() OVER(ORDER BY ' + keys + ')-1 fact_id,' + keys + ' FROM '
        '(SELECT DISTINCT subject_kind,siren,siret,lexical_scope,name_id FROM variants119 '
        'JOIN names119 USING(name_text) WHERE lexical_eligible)')
    execute('export_names', 'COPY (SELECT * FROM names119 ORDER BY name_id) TO ? '
        '(FORMAT PARQUET,COMPRESSION ZSTD)', [str(output / 'names.parquet')])
    execute('export_facts', 'COPY (SELECT * FROM facts119 ORDER BY fact_id) TO ? '
        '(FORMAT PARQUET,COMPRESSION ZSTD)', [str(output / 'facts.parquet')])
    join = ' AND '.join('v.' + key + '=f.' + key for key in
                        ('subject_kind','siren','siret','lexical_scope')) + ' AND n.name_id=f.name_id'
    execute('export_mapped_observations', 'COPY (SELECT v.*,n.name_id,f.fact_id FROM variants119 v '
        'LEFT JOIN names119 n ON CASE WHEN v.lexical_eligible THEN v.name_text END=n.name_text '
        'LEFT JOIN facts119 f ON ' + join + ' ORDER BY observation_index) TO ? '
        '(FORMAT PARQUET,COMPRESSION ZSTD)', [str(output / 'observations.parquet')])
    began = time.monotonic(); audit = audit_roundtrip(c, source, output)
    phases['exact_aligned_roundtrip'] = time.monotonic() - began
    progress(dict(phase='exact_aligned_roundtrip', seconds=phases['exact_aligned_roundtrip']))
    total, eligible, max_obs = c.execute('SELECT count(*),count(*) FILTER(WHERE lexical_eligible),'
        '(SELECT coalesce(max(n),0) FROM (SELECT fact_id,count(*) n FROM read_parquet(' +
        literal(output / 'observations.parquet') + ') WHERE fact_id IS NOT NULL GROUP BY fact_id)) '
        'FROM variants119').fetchone()
    return dict(observations=total, lexical_eligible_observations=eligible,
        names=pq.ParquetFile(output / 'names.parquet').metadata.num_rows,
        facts=pq.ParquetFile(output / 'facts.parquet').metadata.num_rows,
        max_observations_per_fact=max_obs, full_observation_roundtrip=True,
        name_or_period_cap=None, site_cartesian_product_materialized=False,
        audit=audit, phases=phases)


