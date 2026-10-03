"""Same V118 observations/keys, materialized barriers and exact aligned audit.

No semantic change, less repeated wide-row hashing. Ordered output is checked
against its physical row positions, then every field and owner/name join is
compared exactly (including nulls and arrays). This is not a checksum sample.
"""
import time

import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def audit_roundtrip(c, source, output):
    raw, obs = literal(source), literal(output / 'observations.parquet')
    names, facts = literal(output / 'names.parquet'), literal(output / 'facts.parquet')
    expected = pq.ParquetFile(source).metadata.num_rows
    total, unique, displaced = c.execute(
        'SELECT count(*),count(DISTINCT observation_index),count(*) FILTER(WHERE '
        'observation_index IS NULL OR observation_index<>file_row_number) '
        'FROM read_parquet(' + obs + ',file_row_number=true)').fetchone()
    if total != expected or unique != total or displaced:
        raise ValueError('Observation count/order/uniqueness drift')
    for path, identity, key in [(names, 'name_id', 'name_text'),
                               (facts, 'fact_id', '(subject_kind,siren,siret,lexical_scope,name_id)')]:
        count, ids, keys, nulls = c.execute('SELECT count(*),count(DISTINCT ' + identity +
            '),count(DISTINCT ' + key + '),count(*) FILTER(WHERE ' + identity +
            ' IS NULL) FROM read_parquet(' + path + ')').fetchone()
        if count != ids or count != keys or nulls:
            raise ValueError('Nonunique dictionary/fact keys')
    checks = ['v.' + name + ' IS DISTINCT FROM o.' + name for name in OBS_SCHEMA.names]
    checks += [
        'v.lexical_eligible AND (n.name_id IS NULL OR f.fact_id IS NULL OR '
        'n.name_text IS DISTINCT FROM v.name_text OR f.name_id IS DISTINCT FROM n.name_id)',
        'NOT v.lexical_eligible AND (o.name_id IS NOT NULL OR o.fact_id IS NOT NULL)',
        'v.lexical_eligible AND (' + ' OR '.join('v.' + key + ' IS DISTINCT FROM f.' + key
            for key in ('subject_kind','siren','siret','lexical_scope')) + ')',
    ]
    # Raw file positions are unique by definition; output indices have already
    # been proven unique, contiguous and in the same physical order. Counts
    # equal plus aligned join give exactly one comparison for every input row.
    bad = c.execute('SELECT count(*) FROM read_parquet(' + raw + ',file_row_number=true) v '
        'JOIN read_parquet(' + obs + ') o ON v.file_row_number=o.observation_index '
        'LEFT JOIN read_parquet(' + names + ') n ON o.name_id=n.name_id '
        'LEFT JOIN read_parquet(' + facts + ') f ON o.fact_id=f.fact_id '
        'WHERE ' + ' OR '.join('(' + check + ')' for check in checks)).fetchone()[0]
    if bad:
        raise ValueError('Full-field/name/owner reconstruction drift')
    return dict(full_observation_roundtrip=True, compared_observations=expected,
                compared_source_fields=len(OBS_SCHEMA.names), output_position_alignment_proven=True,
                dictionary_and_fact_keys_unique=True, every_owner_and_name_reference_compared=True)


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
        'LEFT JOIN names119 n ON v.name_text=n.name_text AND v.lexical_eligible '
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


def compare_dictionaries(c, baseline, output):
    """Same IDs and full keys make the separately proven roundtrips equivalent."""
    for filename, key, fields in [('names.parquet','name_id',('name_text',)),
        ('facts.parquet','fact_id',('subject_kind','siren','siret','lexical_scope','name_id'))]:
        checks = ' OR '.join('a.' + field + ' IS DISTINCT FROM b.' + field for field in fields)
        bad = c.execute('SELECT count(*) FROM read_parquet(' + literal(baseline / filename) + ') a '
            'FULL OUTER JOIN read_parquet(' + literal(output / filename) + ') b USING(' + key + ') '
            'WHERE a.' + key + ' IS NULL OR b.' + key + ' IS NULL OR ' + checks).fetchone()[0]
        if bad:
            raise ValueError('V118/V119 dictionary/key parity drift')
    return dict(dictionary_keys_and_ids_identical=True,
                observation_equivalence_by_same_source_and_independent_exact_roundtrips=True)
