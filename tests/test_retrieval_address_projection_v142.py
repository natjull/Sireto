import numpy as np
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_address_projection_v142 import CANONICAL,CURRENT_FIELDS,project,audit_batch,scalar_text_oracle,visible_component
from src.xgb_matcher.retrieval_append_catalog_v141 import append_positions
from src.xgb_matcher.retrieval_proof_bridge_v137 import MISSING
from test_retrieval_append_catalog_v141 import ToyKeys


class Keys(ToyKeys):
    def __init__(self):
        super().__init__();self.previous_sites=8;self.previous_parents=5
        self.new_site=np.array([100003,200009,400001,900003],dtype=np.uint64);self.new_parent=np.array([4],dtype=np.uint32)
        self.sites=12;self.parents=6
    def site_positions(self,values):return append_positions(8,self.new_site,values,super().site_positions(values))
    def parent_positions(self,values):return append_positions(5,self.new_parent,values,super().parent_positions(values))
    def reconstruct(self,pos,grain):
        a=np.concatenate((self.site,self.extra_site,self.new_site) if grain=='site' else (self.parent,self.extra_parent,self.new_parent))
        return a[pos]


def canonical(rows):
    return pa.Table.from_pylist(rows,schema=pa.schema([(k,pa.bool_() if k=='is_current' else pa.string()) for k in CANONICAL]))


def current(rows):
    return pa.Table.from_pylist(rows,schema=pa.schema([(k,pa.string()) for k in CURRENT_FIELDS]))


def test_canonical_all_masks_contexts_grain_unknown_and_raw_priority_kept():
    t=canonical([
        dict(siren='000000001',siret='00000000100001',subject_kind='SIRET',source='RNE',raw_value='[ ND ] [ND]',normalized_value='ND ND',is_current=True),
        dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='BODACC',raw_value='\u00a0 ',normalized_value='9 rue côté',postcode='00000',insee='2a004',street_number='①②③④⑤⑥⑦⑧⑨'),
        dict(siren='000000004',siret='00000000400001',subject_kind='SIREN',source='RNE',raw_value='SIEGE AILLEURS',is_current=False),
        dict(siren='000000001',siret='00000000900003',subject_kind='SIRET',source='BODACC',raw_value='RUE AILLEURS',is_current=True),
        dict(siren='000000077',subject_kind='SIREN',source='RNE',raw_value='CONTEXTE SANS MEMBRE')])
    k=Keys();p=project(t,'official_addresses',k,5,100)
    assert p['observation_id'].to_pylist()==list(range(105,110))
    assert p['subject_code'].to_pylist()==[2,2,1,0,1]
    assert p['site_position'].to_pylist()==[0,8,MISSING,MISSING,MISSING]
    assert p['parent_position'].to_pylist()==[0,0,5,MISSING,MISSING]
    assert p['address_text'].to_pylist()==['','9 RUE COTE','SIEGE AILLEURS','RUE AILLEURS','CONTEXTE SANS MEMBRE']
    assert p['number_text'].to_pylist()[1]=='123456789'  # exactly ONCE normalized, never idempotence assumed
    assert p['lexical_eligible'].to_pylist()==[False,True,False,False,False]
    assert p['postcode_code'].to_pylist()[1]==0
    assert p['source_is_current'].to_pylist()==[True,None,False,True,None]
    assert p['current_exact'].to_pylist()==[True,False,False,True,False]
    assert scalar_text_oracle(t,p,'official_addresses')==5
    assert audit_batch(t,p,'official_addresses',k,5,100)


def test_primary_secondary_full_fields_no_mask_erase_or_parent_or_geo_inheritance():
    t=current([dict(siren='000000001',siret='00000000100001',numeroVoieEtablissement='[ND]',typeVoieEtablissement='rue',
        libelleVoieEtablissement='École',complementAdresseEtablissement='Bâtiment A',dernierNumeroVoieEtablissement='20',
        codePostalEtablissement='75001',libelleCommuneEtablissement='PARIS',codeCommuneEtablissement='75101',
        numeroVoie2Etablissement='5',typeVoie2Etablissement='AV',libelleVoie2Etablissement='Lilas',
        codePostal2Etablissement='69001',codeCommune2Etablissement='69381',complementAdresse2Etablissement='[\u00a0ND\u00a0]'),
        dict(siren='000000003',siret='00000000300001')])
    k=Keys();p=project(t,'current_addresses',k,4,100)
    assert p['observation_id'].to_pylist()==list(range(116,124))
    assert p['source_row_ordinal'].to_pylist()==[4]*4+[5]*4 and p['projection_role'].to_pylist()==[1,2,3,4]*2
    assert p['address_text'].to_pylist()[:4]==['RUE ECOLE','BATIMENT A 20 RUE ECOLE 75001 PARIS','5 AV LILAS','5 AV LILAS 69001']
    assert p['address_text'].to_pylist()[4:]==['']*4 and len(p)==8
    assert p['current_exact'].to_pylist()==[True,True,False,False]*2
    assert p['number_text'].to_pylist()[:4]==['','','5','5']
    assert p['site_position'].to_pylist()==[0]*4+[1]*4
    assert p['lexical_eligible'].to_pylist()==[True]*4+[False]*4
    assert scalar_text_oracle(t,p,'current_addresses')==2
    assert audit_batch(t,p,'current_addresses',k,4,100)


def test_every_observation_retained_no_2000_cap():
    row=dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='BODACC',raw_value='RUE HISTORIQUE')
    t=canonical([row]*2003);p=project(t,'official_addresses',Keys(),0,0)
    assert len(p)==2003 and p['site_position'].to_pylist()==[8]*2003
    assert audit_batch(t,p,'official_addresses',Keys(),0,0)


def test_changed_export_or_absent_explicit_site_stops_not_demotes():
    t=canonical([dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value='RUE A')])
    k=Keys();p=project(t,'official_addresses',k,0,0)
    bad=p.set_column(p.schema.get_field_index('site_position'),'site_position',pa.array([3],type=pa.uint32()))
    with pytest.raises(ValueError,match='second source'):audit_batch(t,bad,'official_addresses',k,0,0)
    t=canonical([dict(siren='000000001',siret='00000000100099',subject_kind='SIRET',source='RNE',raw_value='RUE A')])
    with pytest.raises(ValueError,match='admitted'):project(t,'official_addresses',k,0,0)
    with pytest.raises(ValueError,match='grain'):project(current([dict(siren='000000001',siret='００００００００１００００１')]),'current_addresses',k,0,0)


def test_component_masks_unicode_and_identifier_fields_never_joined():
    values=pa.array(['[ND]','[\u00a0nd\u00a0]','  Côte & NORD ','ND','12345678900001'])
    assert visible_component(values).to_pylist()==['','','Côte & NORD','ND','12345678900001']
    t=current([dict(siren='000000001',siret='00000000100001',libelleVoieEtablissement='9 rue du Lac 12345678900001 987654321')])
    p=project(t,'current_addresses',Keys(),0,0)
    assert p['address_text'].to_pylist()==['9 RUE DU LAC']*2+['']*2
    assert scalar_text_oracle(t,p,'current_addresses')==1


def test_bad_role_schema_or_ordinal_range_stops():
    with pytest.raises(ValueError,match='input role'):project(canonical([]),'history_addresses',Keys(),0,0)
    with pytest.raises(ValueError,match='ordinal'):project(canonical([]),'official_addresses',Keys(),True,0)
    t=canonical([dict(subject_kind='SIREN',siren='000000001')])
    t=t.set_column(t.schema.get_field_index('is_current'),'is_current',pa.array(['True']))
    with pytest.raises(ValueError,match='nullable'):project(t,'official_addresses',Keys(),0,0)
