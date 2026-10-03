import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_national_layout_v126 import read_coordinates,layout,spread
from src.xgb_matcher.retrieval_coupled_source_v121 import source_coordinate


def rows():
    rng=np.random.default_rng(126);result=[]
    for i in range(401):
        result.append(dict(siret=f'123456789{i:05d}',codeCommuneEtablissement='97101' if i%7==0 else '75056' if i%11 else '',
            coordonneeLambertAbscisseEtablissement=str(rng.uniform(0,1300000)) if i%3 else '[ND]',
            coordonneeLambertOrdonneeEtablissement=str(rng.uniform(6000000,7300000)) if i%5 else None,
            etatAdministratifEtablissement='F' if i%2 else 'A'))
    return result


@pytest.mark.parametrize('block_size',[1,17,128])
def test_all_source_sites_exact_coordinate_oracle_and_physical_coverage(tmp_path,block_size):
    original=rows();np.random.default_rng(1).shuffle(original);path=tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist(original),path);keys=np.array(sorted(int(r['siret']) for r in original),dtype=np.uint64)
    c=duckdb.connect();xy=read_coordinates(c,path,keys,tmp_path);c.close();stats=layout(xy,tmp_path,block_size)
    for row in original:
        pos=np.searchsorted(keys,np.uint64(int(row['siret'])));expected=source_coordinate(row)
        assert np.array_equal(xy[pos],np.array(expected) if expected is not None else np.array([np.nan,np.nan]),equal_nan=True)
    ids=np.load(tmp_path/'block_sites.npy');offsets=np.load(tmp_path/'block_offsets.npy');lo=np.load(tmp_path/'block_lo.npy');hi=np.load(tmp_path/'block_hi.npy')
    assert np.array_equal(np.sort(ids),np.arange(len(original))) and stats['sites']==len(original)
    for b in range(len(lo)):
        group=xy[ids[int(offsets[b]):int(offsets[b+1])]]
        if np.isfinite(group[:,0]).any():
            assert np.isfinite(group).all() and np.array_equal(lo[b],group.min(axis=0)) and np.array_equal(hi[b],group.max(axis=0))
        else:assert np.isnan(lo[b]).all() and np.isnan(hi[b]).all()
    assert stats['closed_sites_filtered'] is False and stats['score_coordinates_quantized'] is False


def test_morton_bits_exact_and_same_cell_stable(tmp_path):
    for value in [0,1,2,3,1300000,2**21-1]:
        expected=sum(((value>>bit)&1)<<(2*bit) for bit in range(21))
        assert spread(np.array([value]))[0]==expected
    xy=np.array([[1.1,6000002.2],[1.2,6000002.3],[np.nan,np.nan]])
    layout(xy,tmp_path,1);assert np.array_equal(np.load(tmp_path/'block_sites.npy'),[0,1,2])


@pytest.mark.parametrize('kind',['duplicate','missing','wrong','malformed'])
def test_source_identity_drift_never_becomes_filtered_denominator(tmp_path,kind):
    original=rows()[:5];keys=np.array(sorted(int(r['siret']) for r in original),dtype=np.uint64)
    if kind=='duplicate':original[-1]=dict(original[0])
    elif kind=='missing':original.pop()
    elif kind=='wrong':original[-1]['siret']='99999999999999'
    else:original[-1]['siret']='[ND]'
    path=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist(original),path);c=duckdb.connect()
    with pytest.raises(ValueError):read_coordinates(c,path,keys,tmp_path)
    c.close()
