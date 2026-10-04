import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_access_v152 import SCHEMA,stage_addresses as native
from src.xgb_matcher.retrieval_address_channels_v166 import stage_addresses


@pytest.mark.parametrize('group',[1,3,4])
def test_every_export_and_feature_identical_despite_bounded_reader_audits(tmp_path,group):
    texts=['RUE TEST','AVENUE DE LA REPUBLIQUE','A','123456789','NUMERO 12345678901234',
           'SAS SA STE ET FILS','CHEMIN LE LONG DES GRANDES PRAIRIES DU PONT DE LA VALLEE']*35
    table=pa.Table.from_pydict(dict(address_id=np.arange(8,8+len(texts),dtype=np.uint32),address_text=texts),schema=SCHEMA)
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    with duckdb.connect() as c:a=native(c,table,old,67,8,64)
    with duckdb.connect() as c:
        b=stage_addresses(c,table,new,67,8,64,group)
        def rows(files):
            return c.execute('SELECT * FROM read_parquet(?,hive_partitioning=false) ORDER BY alias_id,channel,token',[files]).fetchall()
        assert rows(a['files'])==rows(b['files'])
    assert pq.read_table(old/'features.parquet').equals(pq.read_table(new/'features.parquet'))
    for key,value in a.items():
        if key!='files':assert b[key]==value
    assert b['maximum_physical_readers_per_audit']<=group
    assert b['maximum_physical_writers_per_copy']<=group
    assert b['export_audit_groups']>1 and b['all_exported_postings_exact']
    assert b['alias_or_proof_cap'] is None


def test_empty_guarded_feature_not_dropped_and_no_retry(tmp_path):
    table=pa.Table.from_pydict(dict(address_id=[0],address_text=['123456789']),schema=SCHEMA)
    with duckdb.connect() as c:
        s=stage_addresses(c,table,tmp_path,0,0,64)
        assert s['addresses']==1 and s['postings']==0 and s['export_audit_groups']==0
        with pytest.raises(ValueError,match='Immutable'):stage_addresses(c,table,tmp_path,0,0,64)


@pytest.mark.parametrize('group',[0,5,True])
def test_read_write_bound_cannot_expand_implicitly(tmp_path,group):
    table=pa.Table.from_pydict(dict(address_id=[0],address_text=['RUE TEST']),schema=SCHEMA)
    with duckdb.connect() as c,pytest.raises(ValueError,match='bounded physical'):
        stage_addresses(c,table,tmp_path,0,0,64,group)


def test_export_corruption_rejected_by_independent_group_join(tmp_path):
    table=pa.Table.from_pydict(dict(address_id=[0,1],address_text=['RUE TEST','AVENUE DE LA REPUBLIQUE']),schema=SCHEMA)
    class ChangedExport:
        def __init__(self,c):self.c=c;self.changed=False
        def __getattr__(self,name):return getattr(self.c,name)
        def execute(self,sql,*args,**kwargs):
            if sql.startswith('CREATE TEMP VIEW exported131') and not self.changed:
                self.changed=True
                # Replace tokens in the first nonempty physical export while
                # preserving schema and row count: counts alone cannot pass.
                path=next((tmp_path/'staging').glob('term_bucket=*/*.parquet'))
                t=pq.ParquetFile(path).read()
                column=t.schema.get_field_index('token')
                t=t.set_column(column,'token',pa.array(['WRONG']*len(t),type=pa.string()))
                pq.write_table(t,path)
            return self.c.execute(sql,*args,**kwargs)
    with duckdb.connect() as c,pytest.raises(ValueError,match='reconstruct exactly'):
        stage_addresses(ChangedExport(c),table,tmp_path,0,0,64)
