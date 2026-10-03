import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA, observations, factorize as original
from src.xgb_matcher.retrieval_variant_factor_v119 import audit_roundtrip, compare_dictionaries, factorize


def make(tmp_path):
    rows = []
    for i in range(220):
        rows.extend(observations(dict(siren='123456789', nomUniteLegale='Alpha',
            dateDebut=f'{1800+i}-01-01'), 'history_ul', i))
    rows.extend(observations(dict(siren='123456789', siret='12345678900001',
        enseigne1Etablissement='Alpha', enseigne2Etablissement='[ND]'), 'current_site', 0))
    rows.extend(observations(dict(siren='987654321'), 'current_ul', 0))
    path = tmp_path / 'raw.parquet'; pq.write_table(pa.Table.from_pylist(rows, schema=OBS_SCHEMA), path)
    return path


def test_same_semantics_ids_and_every_field_as_v118(tmp_path):
    source = make(tmp_path); a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir()
    c = duckdb.connect(); original(c, source, a); stats = factorize(c, source, b)
    assert stats['observations'] == 223 and stats['max_observations_per_fact'] == 220
    assert stats['audit']['compared_source_fields'] == len(OBS_SCHEMA.names)
    assert compare_dictionaries(c, a, b)['dictionary_keys_and_ids_identical']
    for filename in ['names.parquet','facts.parquet','observations.parquet']:
        assert pq.read_table(a / filename).equals(pq.read_table(b / filename))
    c.close()


@pytest.mark.parametrize('field,value', [('raw_value','CORRUPTED'), ('valid_from',None),
    ('source_fields',['other']), ('evidence_id','fake'), ('fact_id',999), ('name_id',999)])
def test_independent_audit_detects_single_field_or_reference_corruption(tmp_path, field, value):
    source = make(tmp_path); c = duckdb.connect(); factorize(c, source, tmp_path); c.close()
    path = tmp_path / 'observations.parquet'; table = pq.read_table(path); rows = table.to_pylist()
    rows[0][field] = value; pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    c = duckdb.connect()
    with pytest.raises(ValueError, match='reconstruction drift'):
        audit_roundtrip(c, source, tmp_path)
    c.close()


def test_reordered_output_and_duplicate_observation_indices_rejected(tmp_path):
    source = make(tmp_path); c = duckdb.connect(); factorize(c, source, tmp_path)
    path = tmp_path / 'observations.parquet'; table = pq.read_table(path)
    pq.write_table(table.take(pa.array(list(reversed(range(len(table)))))), path)
    with pytest.raises(ValueError, match='order/uniqueness'):
        audit_roundtrip(c, source, tmp_path)
    c.close()


def test_same_name_cannot_be_reassigned_to_another_owner(tmp_path):
    source = make(tmp_path); c = duckdb.connect(); factorize(c, source, tmp_path); c.close()
    path = tmp_path / 'facts.parquet'; table = pq.read_table(path); rows = table.to_pylist()
    rows[0]['siren'] = '987654321'; pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    c = duckdb.connect()
    with pytest.raises(ValueError, match='reconstruction drift'):
        audit_roundtrip(c, source, tmp_path)
    c.close()
