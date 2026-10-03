from datetime import date
import string

import numpy as np
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_variant_batches_v128 import normalize,project
from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA,observations,normalized


def same(rows,role,start=0):
    source=pa.Table.from_pylist(rows);expected=[]
    for i,row in enumerate(source.to_pylist()):expected.extend(observations(row,role,row.get('source_row_ordinal',start+i)))
    assert project(source,role,start).equals(pa.Table.from_pylist(expected,schema=OBS_SCHEMA))


def test_ascii_unicode_masks_identifiers_all_exact_scalar():
    values=[None,'','[nd]',' A [\x1cNd\x1f] B ','Éva','Straße','αβ café','Å','A\u093eB','A\ufe0fB','A\u0301B',
        'ALPHA 123456789 123456789 BETA','123456789123456789','ABC123456789DEF','①②③④⑤⑥⑦⑧⑨',
        '١123456789','[\u1680ND\u1680]','[\u00a0Nd\u00a0]','ﬂ café',' ＡＢＣ ㍿ ','\x1c abc \x1f']
    rng=np.random.default_rng(128);chars=np.array(list(string.ascii_letters+string.digits+string.whitespace+'[]&-_'))
    values += [''.join(rng.choice(chars,size=int(rng.integers(100)))) for _ in range(3000)]
    values += [' '.join(['123456789']*i) for i in range(1,70)]
    assert normalize(pa.array(values)).to_pylist()==[normalized(s) for s in values]
    assert normalize(pa.chunked_array([values[:20],values[20:]])).to_pylist()==[normalized(s) for s in values]


def test_every_ei_given_subset_both_orders_duplicates_and_masked_diffusion():
    rows=[]
    for combo in range(16):
        for status in ['O','P',None,'']:
            row=dict(siren='123456789',statutDiffusionUniteLegale=status,nomUniteLegale=' Éva ',nomUsageUniteLegale='Durand',
                prenomUsuelUniteLegale='Eva',sigleUniteLegale='E & D',dateDebut=date(2000,1,1),etatAdministratifUniteLegale='F')
            for j in range(4):row[f'prenom{j+1}UniteLegale']=['Éva','Jeanne','Straße','[ND]'][j] if combo&(1<<j) else None
            rows.append(row)
    same(rows,'current_ul',100)


@pytest.mark.parametrize('role',['current_ul','history_ul','current_site','history_site','official_names'])
def test_scopes_empty_context_originals_and_all_metadata(role):
    if role=='official_names':
        rows=[dict(siren='123456789',siret='12345678900001',subject_kind='SIRET',source='BODACC',name_kind='SIGN',
            raw_value=' Café ',normalized_value='CAFE',valid_from=date(2001,1,1),valid_to=date(2004,2,2),
            observed_at=date(2005,1,1),is_current=False,evidence_id='e',source_record_id='r',source_priority=200,
            statutDiffusionEtablissement='P'),dict(siren='123456789',subject_kind='SIREN',source='RNE',name_kind='LEGAL',
            raw_value=None,normalized_value='ALPHA'),dict(siren='123456789',subject_kind='SIREN',source='RNE',name_kind='LEGAL',
            raw_value='[ND]',normalized_value='DO NOT UNMASK'),dict(siren='[ND]',subject_kind='SIREN',source='RNE',
            name_kind='SIGN',raw_value='BETA')]
    elif role.endswith('_ul'):
        rows=[dict(siren='123456789',denominationUniteLegale='Alpha',denominationUsuelle1UniteLegale='Alpha',
            nomUniteLegale='Ancien',prenom1UniteLegale='Éva',statutDiffusionUniteLegale='O',dateDebut=date(2001,1,1),
            dateFin=date(2005,1,1)),dict(siren='123456789'),dict(siren='[ND]',nomUniteLegale='[ND]')]
    else:
        rows=[dict(siren='123456789',siret='12345678900001',enseigne1Etablissement=' Café ',enseigne2Etablissement='[ND]',
            denominationUsuelleEtablissement='Alpha',dateDebut=date(2001,1,1),dateFin=date(2004,1,1),etatAdministratifEtablissement='F'),
            dict(siren='123456789',siret='98765432100001',enseigne1Etablissement='Beta'),dict(siren='123456789',siret='12345678900002')]
    for i,row in enumerate(rows):row['source_row_ordinal']=np.uint64(1000+i*7)
    same(rows,role)


def test_no_period_variant_or_siren_site_fanout_caps():
    rows=[dict(siren='123456789',denominationUniteLegale=f'NAME {i}',dateDebut=f'{1900+i//12:04d}-{i%12+1:02d}-01') for i in range(250)]
    same(rows,'history_ul')
    sites=[dict(siren='123456789',siret=f'123456789{i:05d}',enseigne1Etablissement='Alpha') for i in range(350)]
    same(sites,'current_site')


@pytest.mark.parametrize('ordinal',[-1,True,1.5])
def test_bad_source_ordinals_stop(ordinal):
    with pytest.raises(ValueError):project(pa.Table.from_pylist([dict(siren='123456789')]),'current_ul',ordinal)
