import copy

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.project_retrieval_v93_national_names import project as materialize
from scripts.project_retrieval_v94_virtual_national_names import project,projection_sql,verified_projection
from tests.test_retrieval_scoped_succession_names_v82 import fixture


def test_complete_virtual_projection_equals_all_materialized_columns_and_rows(tmp_path):
    _,names=fixture(tmp_path)
    rows=pq.read_table(names).to_pylist(); rows.append(dict(rows[0],raw_value='',normalized_value='',evidence_id='empty'))
    pq.write_table(pa.Table.from_pylist(rows),names)
    sites=tmp_path/'sites.parquet'
    pq.write_table(pa.Table.from_pylist([dict(siret='11111111100001',siren='111111111',administrative_state='F'),
        dict(siret='33333333300001',siren='333333333',administrative_state='A')]),sites)
    old=tmp_path/'old'; old.mkdir(); new=tmp_path/'new'; new.mkdir()
    c=duckdb.connect(); old_stats=materialize(c,names,sites,old); c.close()
    c=duckdb.connect(); stats=project(c,names,sites,new)
    actual=c.execute(projection_sql(names)).fetch_arrow_table().to_pylist(); c.close()
    expected=pq.read_table(old/'scoped_name_observations.parquet').to_pylist()
    assert sorted(actual,key=lambda r:r['source_observation_index'])==sorted(expected,key=lambda r:r['source_observation_index'])
    assert stats['counts']==old_stats['counts'] and stats['sites']==2 and stats['name_observations']==10
    assert not (new/'scoped_name_observations.parquet').exists()
    assert pq.read_table(new/'national_membership.parquet').to_pylist()==pq.read_table(old/'national_membership.parquet').to_pylist()
    manifest=dict(status='SEALED_COMPLETE_VIRTUAL_NATIONAL_SCOPED_NAMES_AND_MEMBERSHIP',truth_opened=False,crm_opened=False,stats=stats)
    assert verified_projection(manifest)==(names,projection_sql(names))
    for field,value in [('sql','SELECT * LIMIT 1'),('full_source',False),('crm_labels_queries_present',True)]:
        bad=copy.deepcopy(manifest); bad['stats']['name_projection'][field]=value
        with pytest.raises(ValueError): verified_projection(bad)
