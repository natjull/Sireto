"""Small source parsing fixtures, not CRM training examples."""
import struct

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher import retrieval_activity_catalog_v112 as catalog
from src.xgb_matcher.retrieval_official_context_v110 import codebook,encode


def dbf(rows):
    widths=(7,80);length=1+sum(widths);header=bytearray(97)
    header[0]=3;struct.pack_into('<IHH',header,4,len(rows),97,length)
    for offset,name,width in ((32,b'CODE',7),(64,'LIBELLÉ'.encode('cp850'),80)):
        header[offset:offset+len(name)]=name;header[offset+11]=ord('C');header[offset+16]=width
    header[96]=13
    return bytes(header)+b''.join(b' '+c.encode('cp850').ljust(7)+l.encode('cp850').ljust(80) for c,l in rows)+b'\x1a'


def test_strict_oem_code_and_label_roundtrip(monkeypatch):
    monkeypatch.setitem(catalog.EXPECTED,'NAFRev2',2)
    source=dbf([('01.11Z','Culture de céréales'),('47.64Z','Commerce de détail')])
    parsed=catalog.dbf_labels(source,'NAFRev2')
    assert [(r['code'],r['label']) for r in parsed]==[('01.11Z','Culture de céréales'),('47.64Z','Commerce de détail')]
    assert [r['source_row'] for r in parsed]==[0,1]
    assert all(r['nomenclature']=='NAFRev2' for r in parsed)


@pytest.mark.parametrize('mutation',['deleted','truncated','trailing','version','header','field_type','ambiguous','duplicate','invalid_code'])
def test_corrupt_or_ambiguous_official_table_stops(monkeypatch,mutation):
    monkeypatch.setitem(catalog.EXPECTED,'NAFRev2',2)
    rows=[('01.11Z','Culture de céréales'),('47.64Z','Commerce de détail')]
    if mutation=='ambiguous':rows[0]=('01.11Z','À vérifier')
    if mutation=='duplicate':rows[1]=(rows[0][0],rows[1][1])
    if mutation=='invalid_code':rows[0]=('0111Z',rows[0][1])
    source=bytearray(dbf(rows))
    if mutation=='deleted':source[97]=ord('*')
    elif mutation=='truncated':source=source[:-5]
    elif mutation=='trailing':source+=b'garbage'
    elif mutation=='version':source[0]=4
    elif mutation=='header':source[96]=0
    elif mutation=='field_type':source[43]=ord('N')
    with pytest.raises(ValueError):catalog.dbf_labels(bytes(source),'NAFRev2')


def stock_fixture(tmp_path):
    keys=np.array([100001,100002,200001,300001,400001],dtype=np.uint64)
    parents=(keys//100000).astype(np.uint32)
    raw=['01.11Z','51.3T','51.3T',None,'00.98']
    nomenclatures=['NAFRev2','NAF1993','NAFRev1',None,'NAP']
    book=codebook(set(raw));activity=encode(pa.array(raw,type=pa.string()),book)
    np.save(tmp_path/'keys.npy',keys);np.save(tmp_path/'parents.npy',parents)
    np.save(tmp_path/'activity.npy',np.where(activity==0,1,activity).astype(np.uint16))
    table=pa.Table.from_arrays([pa.array([f'{int(k):014d}' for k in keys]),pa.array([f'{int(p):09d}' for p in parents]),
        pa.array(raw,type=pa.string()),pa.array(nomenclatures,type=pa.string())],names=['siret','siren',
        'activitePrincipaleEtablissement','nomenclatureActivitePrincipaleEtablissement'])
    source=tmp_path/'stock.parquet';pq.write_table(table.take(pa.array([4,0,3,2,1])),source)
    out=tmp_path/'out';out.mkdir()
    return source,book,out,table,raw,nomenclatures


def test_full_original_taxonomy_and_null_recovered_by_key(tmp_path):
    source,book,out,_,raw,nom=stock_fixture(tmp_path)
    stats=catalog.build(source,tmp_path/'keys.npy',tmp_path/'parents.npy',tmp_path/'activity.npy',book,out,batch_size=2)
    original=np.load(out/'original_activity_codes.npy');taxonomy=np.load(out/'nomenclature_codes.npy')
    assert [book[int(c)] for c in original]==raw
    assert [catalog.NOMENCLATURES[int(c)] for c in taxonomy]==nom
    assert original[3]==0 and np.load(tmp_path/'activity.npy')[3]==1
    assert stats['sites']==stats['persisted_roundtrip_rows']==5
    assert stats['original_activity_nulls']==1
    assert sum(sum(row) for row in stats['joint_counts'])==5
    assert not stats['nomenclature_inferred'] and stats['all_sites_preserved']
    # The exact same raw code must have DISTINCT definitions across versions.
    labels=[dict(nomenclature='NAF1993',code='51.3T',label='Old'),dict(nomenclature='NAFRev2',code='01.11Z',label='Current')]
    cov=catalog.coverage(stats,labels)
    assert cov['described_sites']==2 and cov['unresolved_sites']==3
    assert {r['nomenclature'] for r in cov['unresolved']}=={'NAFRev1','NAP',None}
    assert cov['missing_descriptions_do_not_remove_sites']


@pytest.mark.parametrize('mutation',['nomenclature','activity','duplicate','parent'])
def test_original_stock_drift_stops(tmp_path,mutation):
    source,book,out,table,_,_=stock_fixture(tmp_path)
    if mutation=='duplicate':table=table.take(pa.array([0,1,2,3,0]))
    else:
        key={'nomenclature':'nomenclatureActivitePrincipaleEtablissement','activity':'activitePrincipaleEtablissement','parent':'siren'}[mutation]
        values=table[key].to_pylist();values[0]={'nomenclature':'NEW_UNKNOWN_VERSION','activity':'00.98','parent':'000000099'}[mutation]
        table=table.set_column(table.schema.get_field_index(key),key,pa.array(values,type=pa.string()))
    pq.write_table(table,source)
    with pytest.raises(ValueError):
        catalog.build(source,tmp_path/'keys.npy',tmp_path/'parents.npy',tmp_path/'activity.npy',book,out,batch_size=2)


def test_duplicate_definition_never_guessed_or_overwritten():
    stats=dict(sites=1,nomenclatures=['NAFRev2'],activity_book=['01.11Z'],joint_counts=[[1]])
    labels=[dict(nomenclature='NAFRev2',code='01.11Z',label='a'),dict(nomenclature='NAFRev2',code='01.11Z',label='b')]
    with pytest.raises(ValueError,match='duplicate'):catalog.coverage(stats,labels)
