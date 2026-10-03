import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.probe_retrieval_v118_official_variants import fingerprint, sample_batches


def test_row_group_sample_preserves_original_ordinals_and_all_fields(tmp_path):
    path = tmp_path / 'source.parquet'
    pq.write_table(pa.table({'name':['row'+str(i) for i in range(50)],
                            'other':[str(i) for i in range(50)]}), path, row_group_size=10)
    selected = list(sample_batches(path, 7))
    assert [(g, start, batch.num_rows) for g, start, batch in selected] == [(0, 0, 3), (2, 20, 2), (4, 40, 2)]
    assert [batch.column('other').to_pylist() for _, _, batch in selected] == [['0','1','2'],['20','21'],['40','41']]
    assert fingerprint(path) == fingerprint(path)
    assert fingerprint(path)['full_input_content_sha256_verified'] is False


def test_tiny_single_group_and_limit_less_than_three(tmp_path):
    path = tmp_path / 'source.parquet'; pq.write_table(pa.table({'name':['a','b']}), path)
    assert [(g, start, batch.num_rows) for g, start, batch in sample_batches(path, 10)] == [(0, 0, 2)]
    path = tmp_path / 'groups.parquet'
    pq.write_table(pa.table({'name':list('abcdef')}), path, row_group_size=2)
    assert [(g, start, batch.num_rows) for g, start, batch in sample_batches(path, 1)] == [(0, 0, 1)]


@pytest.mark.parametrize('limit', [0, -1, True, 1.1])
def test_bad_sampling_limit_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        list(sample_batches(tmp_path / 'unused.parquet', limit))
