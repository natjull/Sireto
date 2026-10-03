from collections import Counter
from datetime import date,datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_succession_csr_v84 import SITE_CLASSES,SuccessionCSR,build


def fixture(tmp_path):
    def row(p,s,date_value,site):
        return dict(predecessor_siret=p,successor_siret=s,predecessor_siren=p[:9],successor_siren=s[:9],
            predecessor_state='F',successor_state='A',reported_effective_date=date_value,
            source_updated_at=datetime(2026,1,1),site_binding=site,headquarters_transfer=None,
            economic_continuity=True,historical_site_coexistence_proven=False,
            relation_source='SIRENE',relation_type='ESTABLISHMENT_SUCCESSION')
    classes=list(SITE_CLASSES)
    rows=[row('00123456700001','22222222200001',date(2000,1,1),classes[0]),
          row('00123456700001','33333333300001',None,classes[1]),
          row('00123456700002','44444444400001',date(2020,1,1),classes[2])]
    rows.append(dict(rows[0])); path=tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist(rows),path)
    return path,rows


def test_all_bidirectional_observations_and_owner_scope_equal_exhaustive_oracle(tmp_path):
    path,rows=fixture(tmp_path); stats=build(path,tmp_path); csr=SuccessionCSR(tmp_path)
    assert stats['observations']==4 and stats['directed_incidences']==8
    assert stats['endpoint_sites']==5 and stats['legal_owners']==4
    for subject in sorted({r[k+'_siret'] for r in rows for k in ('predecessor','successor')}):
        name=dict(subject_kind='SIRET',name_kind='SIGN',siret=subject,siren=subject[:9])
        result=list(csr.neighbors_name_owner(name))
        expected=[]
        for i,row in enumerate(rows):
            for anchor,target,direction in (('predecessor','successor','FORWARD'),('successor','predecessor','REVERSE')):
                if row[anchor+'_siret']==subject: expected.append((i,row[target+'_siret'],row[target+'_siren'],direction))
        assert Counter((r['observation_index'],r['siret'],r['siren'],r['direction']) for r in result)==Counter(expected)
        assert all(not r['historical_site_coexistence_proven'] for r in result)
    company=dict(subject_kind='SIREN',name_kind='LEGAL',siren='001234567')
    assert len(list(csr.neighbors_name_owner(company)))==4
    own_site=dict(subject_kind='SIRET',name_kind='SIGN',siren='001234567',siret='00123456700001')
    assert len(list(csr.neighbors_name_owner(own_site)))==3 # Not sibling 00002.
    unknown=dict(subject_kind='SIRET',name_kind='SIGN',siren='001234567',siret='00123456799999')
    assert list(csr.neighbors_name_owner(unknown))==[]


def test_dates_nulls_closed_states_and_grade_are_not_erased(tmp_path):
    path,_=fixture(tmp_path); build(path,tmp_path); csr=SuccessionCSR(tmp_path)
    name=dict(subject_kind='SIRET',name_kind='USUAL',siren='001234567',siret='00123456700001')
    result={r['observation_index']:r for r in csr.neighbors_name_owner(name)}
    assert result[1]['reported_effective_day']==np.iinfo(np.int32).min
    assert result[0]['reported_effective_day']==(date(2000,1,1)-date(1970,1,1)).days
    assert result[0]['anchor_state_code']==2 and result[0]['target_state_code']==1
    assert result[0]['headquarters_transfer_code']==0
    assert result[1]['site_binding_code']==2


def test_context_only_parent_sign_and_conflicting_name_owner_are_not_lexical_paths(tmp_path):
    path,_=fixture(tmp_path); build(path,tmp_path); csr=SuccessionCSR(tmp_path)
    with pytest.raises(ValueError,match='Context-only'):
        list(csr.neighbors_name_owner(dict(subject_kind='SIREN',name_kind='SIGN',siren='001234567')))
    with pytest.raises(ValueError,match='scope conflict'):
        list(csr.neighbors_name_owner(dict(subject_kind='SIRET',name_kind='SIGN',siren='999999999',siret='00123456700001')))


@pytest.mark.parametrize('field,value',[('economic_continuity',False),('historical_site_coexistence_proven',True),
                                       ('predecessor_siren','999999999'),('relation_source','CRM')])
def test_invalid_source_semantics_and_membership_fail_before_index(tmp_path,field,value):
    path,rows=fixture(tmp_path); rows[0][field]=value; pq.write_table(pa.Table.from_pylist(rows),path)
    with pytest.raises(ValueError,match='drift'): build(path,tmp_path)
