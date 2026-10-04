import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_reverse_v147 import FACT_SCHEMA, stage as old_stage
from src.xgb_matcher.retrieval_address_stage_v159 import stage


class Keys:
    sites = 129
    parents = 1
    def reconstruct(self, ids, grain):
        if grain == 'site':
            return np.uint64(8800000)+ids.astype(np.uint64)
        return np.full(len(ids), 88, dtype=np.uint32)


def source(path, n=12001):
    values = dict(fact_id=list(range(n)), address_id=[i%17 for i in range(n)],
                  site_position=[i%129 for i in range(n)], parent_position=[0]*n,
                  postcode_code=[4294967295 if i%2 else 1234 for i in range(n)],
                  insee_code=[4294967295]*n, number_text=['123456789' if i%2 else '7 BIS' for i in range(n)],
                  current_exact=[bool(i%3) for i in range(n)])
    pq.write_table(pa.Table.from_pydict(values, schema=FACT_SCHEMA), path)
    return values


def read(c, files):
    if not files:
        return []
    return c.execute('SELECT * FROM read_parquet(?,hive_partitioning=false) '
                     'ORDER BY source_bucket,source_fact_id', [files]).fetchall()


@pytest.mark.parametrize('group', [1, 7, 16])
def test_all_fields_and_provenance_bit_exact_to_unbounded_copy_with_129_parts(tmp_path, group):
    path = tmp_path/'facts.parquet'
    source(path)
    old = tmp_path/'old'
    new = tmp_path/'new'
    old.mkdir()
    new.mkdir()
    with duckdb.connect() as c:
        before = old_stage(c, path, old, Keys(), 3, 80, 17, 12001, site_quantum=1)
        after = stage(c, path, new, Keys(), 3, 80, 17, 12001, site_quantum=1, writer_group_parts=group)
        assert read(c, before['files']) == read(c, after['files'])
    assert len({str(p.parent) for p in new.glob('staging/site_part=*/*.parquet')}) == 129
    assert after['facts'] == 12001 and after['every_fact_staged_once']
    assert after['maximum_physical_writers_per_copy'] <= group
    assert after['copy_operations'] == (129+group-1)//group
    assert after['site_or_proof_or_alias_cap'] is None
    assert after['physical_writer_group_is_not_a_candidate_cap']


def test_empty_facts_and_immutable_output(tmp_path):
    path = tmp_path/'empty.parquet'
    source(path, 0)
    out = tmp_path/'out'
    out.mkdir()
    with duckdb.connect() as c:
        stats = stage(c, path, out, Keys(), 0, 0, 0, 0)
        assert stats['files'] == [] and stats['copy_operations'] == 0
    (out/'staging').mkdir()
    with duckdb.connect() as c, pytest.raises(ValueError, match='Immutable'):
        stage(c, path, out, Keys(), 0, 0, 0, 0)


@pytest.mark.parametrize('group', [0, 17, True])
def test_writer_bound_is_explicit_and_cannot_relax_implicitly(tmp_path, group):
    with duckdb.connect() as c, pytest.raises(ValueError, match='bounded physical'):
        stage(c, tmp_path/'absent', tmp_path/'out', Keys(), 0, 0, 0, 0, writer_group_parts=group)


def test_original_parent_identity_still_audited(tmp_path):
    path = tmp_path/'facts.parquet'
    source(path, 2)
    class Wrong(Keys):
        def reconstruct(self, ids, grain):
            return super().reconstruct(ids, grain)+1
    with duckdb.connect() as c, pytest.raises(ValueError, match='site/parent'):
        stage(c, path, tmp_path, Wrong(), 0, 0, 17, 2)


def test_cost_probe_accepts_only_sealed_dictionary_not_stopped_reverse_outputs(tmp_path, monkeypatch):
    import scripts.probe_retrieval_v159_address_stage as probe
    import scripts.compile_retrieval_v154_address_store as store
    from src.xgb_matcher.retrieval_v20_workbench import record, write_json
    from test_compile_retrieval_v154_address_store import setup, finish_phase
    source_root, output = setup(tmp_path, monkeypatch)
    store.init(output, record(source_root/'manifest.json'))
    finish_phase(output, 'part')
    store.seal_staging(output)
    finish_phase(output, 'dictionary')
    store.seal_dictionary(output)
    stopped = store.directory(output, 'reverse-stage', 0)
    stopped.mkdir(parents=True)
    write_json(stopped/'status.json', {'status': 'STOP', 'reason': 'RSS'})
    ref = record(output/'dictionary.json')
    _, plan, b = probe.source(ref, 0)
    assert plan['sites'] == 11 and b['facts'] >= 0
    with pytest.raises(ValueError, match='dictionary authority'):
        probe.source(record(stopped/'status.json'), 0)
    # A dictionary phase's own STOP cannot be hidden by its success manifest.
    write_json(store.directory(output, 'dictionary', 0)/'status.json', {'status': 'STOP'})
    with pytest.raises(ValueError, match='STOP_PHASE'):
        probe.source(ref, 0)
