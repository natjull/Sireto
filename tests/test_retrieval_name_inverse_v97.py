import numpy as np
import pytest

from src.xgb_matcher.retrieval_name_inverse_v97 import inverse_part,assemble


def fixture(tmp_path):
    mapping=np.array([2,0,1,2,3,0,4,4,3,1],dtype=np.uint32)
    path=tmp_path/'mapping.npy'; np.save(path,mapping)
    parts=[dict(bucket=0,fact_start=0,facts=2,observations=4),dict(bucket=1,fact_start=2,facts=3,observations=6)]
    return path,mapping,parts


def test_isolated_inverse_and_full_independent_original_ordinal_certificate(tmp_path):
    path,mapping,parts=fixture(tmp_path)
    for part in parts: inverse_part(path,part,tmp_path,len(mapping))
    stats=assemble(path,parts,tmp_path,len(mapping))
    offsets=np.load(tmp_path/'fact_offsets.npy'); ids=np.load(tmp_path/'fact_observations.npy')
    expected=np.argsort(mapping,kind='stable')
    assert ids.tolist()==expected.tolist() and offsets.tolist()==[0,2,4,6,8,10]
    assert stats['all_original_ordinals_exactly_once'] and stats['independent_inverse_mapping_equality']
    assert sorted(ids.tolist())==list(range(10))


@pytest.mark.parametrize('corruption',['duplicate','missing_mapping','wrong_fact','offset'])
def test_corruption_of_inverse_or_mapping_cannot_seal(tmp_path,corruption):
    path,mapping,parts=fixture(tmp_path)
    for part in parts: inverse_part(path,part,tmp_path,len(mapping))
    if corruption=='missing_mapping': mapping[0]=np.iinfo(np.uint32).max; np.save(path,mapping)
    elif corruption=='offset':
        offsets=np.load(tmp_path/'inverse_offsets00.npy'); offsets[-1]-=1; np.save(tmp_path/'inverse_offsets00.npy',offsets)
    else:
        ids=np.load(tmp_path/'inverse_ordinals00.npy'); ids[0]=ids[1] if corruption=='duplicate' else 0
        np.save(tmp_path/'inverse_ordinals00.npy',ids)
    with pytest.raises(ValueError): assemble(path,parts,tmp_path,len(mapping))
