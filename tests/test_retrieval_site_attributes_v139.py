import datetime

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_site_attributes_v139 import encode_code,decode_code,public_codes,number_codes,build,audit,ATTRS,MISSING
from src.xgb_matcher.retrieval_coupled_source_v121 import public_code
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_official_context_v110 import DATE_NULL


def source(tmp_path):
    data=dict(siret=['00000000300001','00000000100001','00000000500001'],siren=['000000003','000000001','000000005'],
        codePostalEtablissement=[' 69001 ','[ND]',None],codeCommuneEtablissement=['2a001','75056','97101'],
        numeroVoieEtablissement=[' 07 bis ','[ND]','①②③④⑤⑥⑦⑧⑨'],etablissementSiege=[False,True,None],
        etatAdministratifEtablissement=['F','A',None],dateCreationEtablissement=[datetime.date(1998,1,1),None,None],
        dateDebut=[datetime.date(2020,1,1),datetime.date(2000,1,1),None])
    schema=pa.schema([(k,pa.bool_() if k=='etablissementSiege' else pa.date32() if k in ('dateCreationEtablissement','dateDebut') else pa.string()) for k in data])
    table=pa.Table.from_pydict(data,schema=schema);p=tmp_path/'stock.parquet';pq.write_table(table,p,row_group_size=2)
    keys=np.array([100001,300001,500001],dtype=np.uint64);parents=np.array([1,3,5],dtype=np.uint32)
    order=[1,0,2]
    ctx=dict(headquarters_codes=np.array([2,1,0],dtype=np.uint8),state_book=[None,'','A','F'],state_codes=np.array([2,3,0],dtype=np.uint16))
    for key,col in (('creation_days','dateCreationEtablissement'),('current_period_start_days','dateDebut')):
        raw=pc.fill_null(pc.cast(table[col],pa.int32()),int(DATE_NULL)).to_numpy(zero_copy_only=False)
        ctx[key]=raw[order]
    return p,keys,parents,ctx,table


def test_exact_public_codes_leading_zeros_corsica_unicode_and_unknown():
    raw=[None,'',' 00000 ','2a001','2B123','[ND]','[ nd ]','1ß01','００００１','\u200369001\u2003','999999','A0001']
    codes=public_codes(pa.array(raw,type=pa.string()),{})
    assert [decode_code(c) for c in codes]==[public_code(s) for s in raw]
    assert encode_code('00000')==0 and encode_code(None)==MISSING
    assert decode_code(encode_code('1ß01'))=='1SS01'
    for bad in (-1,36**5,True,1.2):
        with pytest.raises(ValueError):decode_code(bad)


def test_number_dictionary_once_normalized_no_fixed_string_width_or_id_feature():
    values=[None,'[ND]',' 007 bis ','É-3','①②③④⑤⑥⑦⑧⑨','123456789','4'*1000]
    book=[''];lookup={'':0};codes=number_codes(pa.array(values,type=pa.string()),book,lookup)
    assert [book[i] for i in codes]==[normalized(v) for v in values]
    assert book[codes[4]]=='123456789' # normalization once, not forced idempotence
    assert len(book[codes[-1]])==1000 and codes[5]==0


def test_all_current_sites_exported_and_second_source_pass_exact(tmp_path):
    p,k,parent,ctx,table=source(tmp_path);out=tmp_path/'build';out.mkdir();s=build(p,k,parent,out,batch_rows=1)
    arrays={name:np.load(out/(name+'.npy')) for name in ATTRS}
    import json
    book=json.loads((out/'number_dictionary.json').read_text())['values'];a=audit(p,k,parent,arrays,book,ctx,batch_rows=2)
    assert s['sites']==3 and a['sites']==3 and s['known']==a['known']
    assert s['known']==dict(postcode_codes=1,insee_codes=3,number_codes=2)
    assert [decode_code(v) for v in arrays['insee_codes']]==['75056','2A001','97101']
    assert [book[i] for i in arrays['number_codes']]==['','07 BIS','123456789']
    assert a['every_reused_headquarters_state_and_date_matches_raw_stock']
    assert s['arbitrary_code_or_number_width_cap'] is None and not a['closed_sites_filtered']


def test_any_exported_attribute_or_reused_original_context_drift_stops(tmp_path):
    p,k,parent,ctx,_=source(tmp_path);out=tmp_path/'build';out.mkdir();build(p,k,parent,out)
    import json
    book=json.loads((out/'number_dictionary.json').read_text())['values'];original={name:np.load(out/(name+'.npy')) for name in ATTRS}
    for field in ATTRS:
        arrays={name:a.copy() for name,a in original.items()};arrays[field][1]=MISSING
        with pytest.raises(ValueError,match='reconstruct'):audit(p,k,parent,arrays,book,ctx)
    for field in ('headquarters_codes','state_codes','creation_days','current_period_start_days'):
        wrong={name:v.copy() if isinstance(v,np.ndarray) else v for name,v in ctx.items()};wrong[field][1]=0
        with pytest.raises(ValueError,match='Reused'):audit(p,k,parent,original,book,wrong)


def test_duplicate_or_missing_current_identity_never_silently_removed(tmp_path):
    p,k,parent,ctx,table=source(tmp_path);out=tmp_path/'bad';out.mkdir()
    duplicate=pa.concat_tables([table.slice(0,1),table.slice(0,1),table.slice(2,1)]);pq.write_table(duplicate,p)
    with pytest.raises(ValueError,match='Duplicate'):build(p,k,parent,out,batch_rows=1)


def test_wrong_parent_source_stops_not_cross_joined_to_sibling(tmp_path):
    p,k,parent,ctx,table=source(tmp_path);out=tmp_path/'bad';out.mkdir();parent[1]=99
    with pytest.raises(ValueError,match='parent drift'):build(p,k,parent,out)
