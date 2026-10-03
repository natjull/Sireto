from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_metadata_v100 import build,encode,SPECS
from src.xgb_matcher.retrieval_name_postings_v86 import build_metadata,SOURCE,NAME_KIND


def test_vectorized_metadata_all_arrays_exactly_match_old_original_ordinal_encoding(tmp_path):
    rows=[]
    for i in range(60):
        rows.append(dict(valid_from=None if i%3 else date(2000,1,1),valid_to=None if i%2 else date(2026,1,1),
            observed_at=None if i%4 else date(2025,1,1),source=list(SOURCE)[i%3],name_kind=list(NAME_KIND)[i%4],
            is_current=None if i%3==0 else bool(i%2),source_priority=None if i%2 else 0))
    table=pa.Table.from_pylist(rows)
    for key in ('valid_from','valid_to','observed_at'):
        table=table.set_column(table.column_names.index(key),pa.field(key,pa.date32()),table[key].cast(pa.date32()))
    source=tmp_path/'source.parquet'; pq.write_table(table,source,row_group_size=7)
    old=tmp_path/'old'; old.mkdir(); new=tmp_path/'new'; new.mkdir()
    expected=build_metadata(source,old); actual=build(source,new)
    for key in SPECS: assert np.array_equal(np.load(old/(key+'.npy')),np.load(new/(key+'.npy')))
    for key in ('observations','source_codes','kind_codes','current_codes','null_date_sentinel'): assert expected[key]==actual[key]


@pytest.mark.parametrize('values', [['UNKNOWN'],[None],['RNE','UNKNOWN']])
def test_unknown_metadata_code_cannot_silently_reinterpret(values):
    with pytest.raises(ValueError,match='Unknown metadata'): encode(pa.array(values,type=pa.string()),SOURCE)
