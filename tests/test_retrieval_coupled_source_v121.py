import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_coupled_source_v121 import build_sites, family_batches, source_coordinate


def test_source_families_not_crm_determine_extra_sample(tmp_path):
    path=tmp_path/'names.parquet'
    table=pa.table({'source':['SIRENE_CURRENT']*4+['BODACC']*4+['RNE']*4,'name':['a']*12})
    pq.write_table(table,path,row_group_size=4)
    rows=list(family_batches(path,2))
    assert [(f,g,start,len(batch)) for f,g,start,batch in rows]==[('SIRENE_CURRENT',0,0,2),('BODACC',1,4,2),('RNE',2,8,2)]
    with pytest.raises(ValueError,match='No pure'):
        list(family_batches(path,2,('MISSING',)))


def current(i,parent='123456789',**values):
    return dict(siren=parent,siret=parent+f'{i:05d}',etatAdministratifEtablissement='F',**values)


def test_parent_membership_site_scope_and_graph_hypothesis_are_distinct():
    names=[dict(name_id=0,name_text='PARENT'),dict(name_id=1,name_text='ENSEIGNE LOCALE')]
    facts=[dict(fact_id=0,name_id=0,subject_kind='SIREN',siren='123456789',siret=''),
           dict(fact_id=1,name_id=1,subject_kind='SIRET',siren='123456789',siret='12345678900001')]
    observations=[dict(fact_id=0,is_current=True),dict(fact_id=1,is_current=True)]
    edges=[dict(siretEtablissementPredecesseur='12345678900001',siretEtablissementSuccesseur='98765432100001',continuiteEconomique=True)]
    sites,stats=build_sites([current(1),current(2),current(1,'987654321')],names,facts,observations,edges=edges)
    assert {p.text for p in sites[0].names}=={'PARENT','ENSEIGNE LOCALE'}
    assert {p.text for p in sites[1].names}=={'PARENT'}
    assert all(p.relation_hypothesis and not p.current_exact and p.grade==.5 for p in sites[2].names)
    assert stats['relation_hypothesis_attachments']==2
    assert all(s.state=='F' for s in sites)


def test_parent_addresses_not_spread_but_explicit_site_address_retained():
    addresses=[dict(subject_kind='SIREN',siren='123456789',siret='',raw_value='SIEGE AILLEURS',source='BODACC',source_row_ordinal=0),
               dict(subject_kind='SIRET',siren='123456789',siret='12345678900001',raw_value='ADRESSE HISTORIQUE',source='RNE',source_row_ordinal=1,is_current=False)]
    sites,stats=build_sites([current(1,codePostalEtablissement='[ND]'),current(2)],[],[],[],address_rows=addresses)
    assert len(sites[0].addresses)==1 and not sites[0].addresses[0].current_exact
    assert not sites[1].addresses and sites[0].postcode==''
    assert stats['context_only_addresses_not_spread_to_sites']==1


def test_unknown_headquarters_activity_xy_and_missing_current_name_not_removed():
    sites,stats=build_sites([current(1,etablissementSiege=None)],[],[],[])
    assert len(sites)==1 and sites[0].headquarters is None
    assert sites[0].activity_class==-1 and sites[0].xy is None
    assert stats['member_or_alias_cap'] is None


def test_overseas_or_invalid_coordinates_not_interpreted_as_lambert93():
    assert source_coordinate(dict(codeCommuneEtablissement='97101',coordonneeLambertAbscisseEtablissement='1',coordonneeLambertOrdonneeEtablissement='2')) is None
    assert source_coordinate(dict(codeCommuneEtablissement='75056',coordonneeLambertAbscisseEtablissement='[ND]',coordonneeLambertOrdonneeEtablissement='6000000')) is None
    assert source_coordinate(dict(codeCommuneEtablissement='75056',coordonneeLambertAbscisseEtablissement='650000',coordonneeLambertOrdonneeEtablissement='6860000'))==(650000.,6860000.)
