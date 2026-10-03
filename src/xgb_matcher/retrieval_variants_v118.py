"""Official variant observations, lossless provenance, no CRM or site fanout.

Projection only: no candidate search, learned alias dictionary or label access.
The name dictionary may be shared by many owners; the owner/scope fact and
every source observation remain distinct. Input sampling is a cost experiment,
never a recall corpus or an exhaustive national publication.
"""
from __future__ import annotations

import re
import unicodedata

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


UL_FIELDS = {
    'denominationUniteLegale': 'LEGAL',
    'denominationUsuelle1UniteLegale': 'USUAL',
    'denominationUsuelle2UniteLegale': 'USUAL',
    'denominationUsuelle3UniteLegale': 'USUAL',
    'sigleUniteLegale': 'TRADE', 'nomUniteLegale': 'LEGAL',
    'nomUsageUniteLegale': 'USUAL', 'pseudonymeUniteLegale': 'USUAL',
}
SITE_FIELDS = {
    'enseigne1Etablissement': 'SIGN', 'enseigne2Etablissement': 'SIGN',
    'enseigne3Etablissement': 'SIGN', 'denominationUsuelleEtablissement': 'USUAL',
}
GIVEN_FIELDS = ('prenomUsuelUniteLegale', 'prenom1UniteLegale',
                'prenom2UniteLegale', 'prenom3UniteLegale', 'prenom4UniteLegale')
ROLES = ('current_ul', 'current_site', 'history_ul', 'history_site', 'official_names')


def clean(value):
    return '' if value is None else str(value).strip()


def normalized(value):
    """Keep originals; do not encode mask placeholders or identifier features."""
    value = clean(value)
    if re.search(r'\[\s*ND\s*\]', value, re.I):
        return ''
    value = re.sub(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)', ' ', value)
    value = ''.join(c for c in unicodedata.normalize('NFKD', value)
                    if not unicodedata.combining(c)).upper()
    return re.sub(r'[^A-Z0-9]+', ' ', value).strip()


def scope(subject, parent, site, kind):
    valid_parent = bool(re.fullmatch('[0-9]{9}', parent))
    if subject == 'SIREN' and valid_parent:
        if kind == 'LEGAL':
            return 'LEGAL_PERSON_NAME'
        if kind in ('USUAL', 'TRADE'):
            return 'LEGAL_PERSON_COMMERCIAL_NAME'
    if (subject == 'SIRET' and valid_parent and re.fullmatch('[0-9]{14}', site)
            and site.startswith(parent) and kind in ('LEGAL', 'USUAL', 'TRADE', 'SIGN')):
        return 'EXPLICIT_ENDPOINT_NAME'
    return 'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED'


def observations(row, role, ordinal):
    """No join/current-name exclusion/latest-period choice or per-owner cap."""
    if role not in ROLES:
        raise ValueError('Unknown official source role')
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        raise ValueError('Source ordinal must be a nonnegative integer')
    historical = role.startswith('history_')
    legal = role.endswith('_ul')
    official = role == 'official_names'
    subject = clean(row.get('subject_kind')) if official else 'SIREN' if legal else 'SIRET'
    parent = clean(row.get('siren'))
    site = clean(row.get('siret')) if subject == 'SIRET' else ''
    source = clean(row.get('source')) if official else 'SIRENE_HISTORY' if historical else 'SIRENE_CURRENT'
    diffusion = row.get('statutDiffusionUniteLegale' if legal else 'statutDiffusionEtablissement')
    # Current dateDebut is a period start, not a closure date or necessarily
    # the start of validity of this name. Do not invent a current-name interval.
    valid_from = row.get('valid_from') if official else row.get('dateDebut') if historical else None
    valid_to = row.get('valid_to') if official else row.get('dateFin') if historical else None
    state = row.get('etatAdministratifUniteLegale' if legal else 'etatAdministratifEtablissement')
    base = dict(input_role=role, source_row_ordinal=ordinal, source=source,
                siren=parent, siret=site, subject_kind=subject,
                valid_from=clean(valid_from) or None, valid_to=clean(valid_to) or None,
                diffusion_status=clean(diffusion) or None,
                administrative_state=clean(state) or None,
                source_period_start=clean(row.get('dateDebut')) or None,
                is_current=row.get('is_current') if official else False if historical else True,
                evidence_id=clean(row.get('evidence_id')) or None,
                source_record_id=clean(row.get('source_record_id')) or None,
                observed_at=clean(row.get('observed_at')) or None,
                source_priority=row.get('source_priority') if official else None,
                historical_address_claimed=False, related_target_alias=False)

    def make(raw, fields, kind, derivation):
        name = normalized(raw)
        name_scope = scope(subject, parent, site, kind)
        return dict(base, raw_value=None if raw is None else str(raw), name_text=name, source_fields=fields,
                    name_kind=kind, derivation=derivation, lexical_scope=name_scope,
                    lexical_eligible=bool(name) and name_scope != 'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED')

    if official:
        # Preserve existing normalized evidence verbatim too; lexical masking
        # is performed on raw_value, with normalized_value fallback only when
        # raw is actually absent. Nothing is reclassified as a sibling alias.
        raw = row.get('raw_value')
        result = make(row.get('normalized_value') if not clean(raw) else raw,
                      ['raw_value'], clean(row.get('name_kind')), 'OFFICIAL_OBSERVATION')
        result['original_raw_value'] = raw
        result['original_normalized_value'] = row.get('normalized_value')
        yield result
        return

    emitted = False
    fields = UL_FIELDS if legal else SITE_FIELDS
    for field, kind in fields.items():
        if field in row and clean(row[field]):
            emitted = True
            yield make(row[field], [field], kind, 'ORIGINAL_FIELD')
    if legal and not historical:
        # Diffusion P/unknown does not delete the entity. Only generate personal
        # name combinations from explicitly fully diffusible current records.
        # No attempt to reverse masks or infer names from another period.
        if diffusion == 'O':
            given = [(field,) for field in GIVEN_FIELDS if normalized(row.get(field))]
            all_declared = tuple(field for field in GIVEN_FIELDS[1:] if normalized(row.get(field)))
            if len(all_declared) > 1:
                given.append(all_declared)
            for family in ('nomUniteLegale', 'nomUsageUniteLegale'):
                if not normalized(row.get(family)):
                    continue
                for given_fields in given:
                    for order in ((family, *given_fields), (*given_fields, family)):
                        emitted = True
                        yield make(' '.join(clean(row[field]) for field in order), list(order),
                                   'LEGAL' if family == 'nomUniteLegale' else 'USUAL',
                                   'SAME_ROW_EI_CONCAT')
    if not emitted:
        yield make('', [], 'UNKNOWN', 'CONTEXT_ROW')


OBS_SCHEMA = pa.schema([
    ('input_role', pa.string()), ('source_row_ordinal', pa.uint64()),
    ('source', pa.string()), ('siren', pa.string()), ('siret', pa.string()),
    ('subject_kind', pa.string()), ('valid_from', pa.string()), ('valid_to', pa.string()),
    ('diffusion_status', pa.string()), ('administrative_state', pa.string()),
    ('source_period_start', pa.string()), ('is_current', pa.bool_()),
    ('evidence_id', pa.string()), ('source_record_id', pa.string()),
    ('observed_at', pa.string()), ('source_priority', pa.int32()),
    ('historical_address_claimed', pa.bool_()), ('related_target_alias', pa.bool_()),
    ('raw_value', pa.string()), ('name_text', pa.string()),
    ('source_fields', pa.list_(pa.string())), ('name_kind', pa.string()),
    ('derivation', pa.string()), ('lexical_scope', pa.string()),
    ('lexical_eligible', pa.bool_()), ('original_normalized_value', pa.string()),
    ('original_raw_value', pa.string()),
])


def factorize(connection, source, output):
    """Lossless name→owner/scope→observation joins; no cartesian site expansion."""
    c = connection
    def literal(path):
        return "'" + str(path).replace("'", "''") + "'"
    c.execute('CREATE TEMP VIEW variants118 AS SELECT file_row_number observation_index, * '
              'EXCLUDE(file_row_number) FROM read_parquet(' + literal(source) + ', file_row_number=true)')
    c.execute("COPY (SELECT row_number() OVER(ORDER BY name_text)-1 name_id, name_text "
              "FROM (SELECT DISTINCT name_text FROM variants118 WHERE lexical_eligible)) "
              "TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(output / 'names.parquet')])
    c.execute('CREATE TEMP VIEW names118 AS SELECT * FROM read_parquet(' +
              literal(output / 'names.parquet') + ')')
    keys = ('subject_kind', 'siren', 'siret', 'lexical_scope', 'name_id')
    key_sql = ','.join(keys)
    c.execute('COPY (SELECT row_number() OVER(ORDER BY ' + key_sql + ')-1 fact_id, ' + key_sql +
              ' FROM (SELECT DISTINCT subject_kind,siren,siret,lexical_scope,name_id '
              'FROM variants118 JOIN names118 USING(name_text) WHERE lexical_eligible)) '
              'TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(output / 'facts.parquet')])
    c.execute('CREATE TEMP VIEW facts118 AS SELECT * FROM read_parquet(' +
              literal(output / 'facts.parquet') + ')')
    join = ' AND '.join('v.' + key + '=f.' + key for key in keys[:-1]) + ' AND n.name_id=f.name_id'
    c.execute('COPY (SELECT v.*,n.name_id,f.fact_id FROM variants118 v '
              'LEFT JOIN names118 n ON v.name_text=n.name_text AND v.lexical_eligible '
              'LEFT JOIN facts118 f ON ' + join + ' ORDER BY observation_index) '
              'TO ? (FORMAT PARQUET, COMPRESSION ZSTD)', [str(output / 'observations.parquet')])
    # EXCEPT ALL both directions verifies every complete observation, not just
    # counts, sums or an identity-field equality. Null comparisons are preserved.
    columns = ','.join('v.' + name for name in OBS_SCHEMA.names)
    c.execute('CREATE TEMP VIEW restored118 AS SELECT o.* EXCLUDE(name_text,name_id,fact_id), '
              "CASE WHEN o.lexical_eligible THEN n.name_text ELSE o.name_text END name_text "
              'FROM read_parquet(' + literal(output / 'observations.parquet') +
              ') o LEFT JOIN names118 n USING(name_id)')
    bad = c.execute('SELECT count(*) FROM ((SELECT observation_index,' + columns +
                    ' FROM variants118 v EXCEPT ALL SELECT observation_index,' + columns +
                    ' FROM restored118 v) UNION ALL (SELECT observation_index,' + columns +
                    ' FROM restored118 v EXCEPT ALL SELECT observation_index,' + columns +
                    ' FROM variants118 v))').fetchone()[0]
    if bad:
        raise ValueError('Lossless observation roundtrip failed')
    total, eligible, unmapped = c.execute('SELECT count(*),count(*) FILTER(WHERE lexical_eligible),'
        'count(*) FILTER(WHERE lexical_eligible AND (name_id IS NULL OR fact_id IS NULL)) '
        'FROM read_parquet(?)', [str(output / 'observations.parquet')]).fetchone()
    if unmapped or total != pq.ParquetFile(source).metadata.num_rows:
        raise ValueError('Observation mapping/denominator drift')
    max_periods = c.execute('SELECT coalesce(max(n),0) FROM (SELECT fact_id,count(*) n '
        'FROM read_parquet(?) WHERE fact_id IS NOT NULL GROUP BY fact_id)',
        [str(output / 'observations.parquet')]).fetchone()[0]
    return dict(observations=total, lexical_eligible_observations=eligible,
                names=pq.ParquetFile(output / 'names.parquet').metadata.num_rows,
                facts=pq.ParquetFile(output / 'facts.parquet').metadata.num_rows,
                max_observations_per_fact=max_periods,
                full_observation_roundtrip=True, name_or_period_cap=None,
                site_cartesian_product_materialized=False)
