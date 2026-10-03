import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA, observations, factorize as original
from src.xgb_matcher.retrieval_variant_factor_v119 import factorize as materialized
from src.xgb_matcher.retrieval_variant_hashjoin_v120 import compare_dictionaries, factorize


def test_all_published_tables_equal_v118_and_v119_including_context(tmp_path):
    rows = []
    for i in range(350):
        rows.extend(observations(dict(siren='123456789', nomUniteLegale='Alpha',
            dateDebut=f'{1600+i}-01-01'), 'history_ul', i))
    rows.extend(observations(dict(siren='123456789', siret='12345678900001',
        enseigne1Etablissement='Alpha', enseigne2Etablissement='[ND]'), 'current_site', 0))
    rows.extend(observations(dict(siren='123456789', subject_kind='UNKNOWN',source='RNE',
        raw_value='Alpha',normalized_value='ALPHA',name_kind='LEGAL'), 'official_names', 0))
    source = tmp_path / 'raw.parquet'; pq.write_table(pa.Table.from_pylist(rows,schema=OBS_SCHEMA),source)
    results = []
    for name, implementation in [('old',original),('materialized',materialized),('hash',factorize)]:
        output = tmp_path / name; output.mkdir(); c = duckdb.connect()
        results.append((output, implementation(c,source,output))); c.close()
    c = duckdb.connect(); assert compare_dictionaries(c,results[0][0],results[2][0])['dictionary_keys_and_ids_identical']; c.close()
    for filename in ('names.parquet','facts.parquet','observations.parquet'):
        assert pq.read_table(results[0][0]/filename).equals(pq.read_table(results[1][0]/filename))
        assert pq.read_table(results[0][0]/filename).equals(pq.read_table(results[2][0]/filename))
    assert results[2][1]['max_observations_per_fact'] == 350


def test_true_false_null_join_semantics_and_native_plan():
    c = duckdb.connect()
    c.execute("CREATE TABLE source(lexical_eligible BOOLEAN,name_text VARCHAR)")
    c.execute("INSERT INTO source VALUES (true,'A'),(false,'A'),(NULL,'A'),(true,NULL),(true,'B')")
    c.execute("CREATE TABLE names(name_text VARCHAR,name_id INTEGER)")
    c.execute("INSERT INTO names VALUES ('A',0)")
    before = 'SELECT s.*,n.name_id FROM source s LEFT JOIN names n ON s.name_text=n.name_text AND s.lexical_eligible'
    after = 'SELECT s.*,n.name_id FROM source s LEFT JOIN names n ON CASE WHEN s.lexical_eligible THEN s.name_text END=n.name_text'
    assert c.execute('SELECT * FROM ('+before+') EXCEPT ALL SELECT * FROM ('+after+')').fetchall() == []
    assert c.execute('SELECT * FROM ('+after+') EXCEPT ALL SELECT * FROM ('+before+')').fetchall() == []
    assert 'BLOCKWISE_NL_JOIN' in c.execute('EXPLAIN '+before).fetchall()[0][1]
    assert 'HASH_JOIN' in c.execute('EXPLAIN '+after).fetchall()[0][1]
    c.close()
