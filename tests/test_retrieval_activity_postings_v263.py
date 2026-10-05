from types import SimpleNamespace
import numpy as np
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_activity_postings_v263 import block_groups,collect_groups,native_proof
from src.xgb_matcher.retrieval_site_attributes_v139 import encode_code,MISSING
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_v20_workbench import record
from test_retrieval_corroborated_initial_v217 import tiny_native


def arrays():
    return (np.array([10000000000001,10000000000002,10000000000003,10000000000004],np.uint64),
        np.array([encode_code('11111'),encode_code('22222'),MISSING,MISSING],np.uint32),
        np.array([MISSING,encode_code('11111'),MISSING,encode_code('11111')],np.uint32),
        np.array([0,1,1,0],np.uint8),np.array([0,0,0,1],np.uint16))


def test_exact_versioned_classes_all_place_union_and_unknown_not_identity():
    a=arrays();q=RetrievalQuery(name='N',postcode='11111',insee='11111')
    g,c=block_groups(*a,q,[.2,.8],np.array([[0,-1],[1,-1]],np.int32))
    assert g=={.2:{'10000000000001'},.8:{'10000000000002'}}
    assert c==dict(scanned=4,at_place=3,known_positive_relevance=2)
    empty,c=block_groups(*a,RetrievalQuery(name='N'),[.2,.8],np.array([[0,-1],[1,-1]],np.int32))
    assert not empty and c['at_place']==0


def test_zero_relevance_and_shared_grades_never_create_extra_sum():
    a=arrays();q=RetrievalQuery(name='N',postcode='11111',insee='11111')
    mapping=np.array([[0,-1],[1,-1]],np.int32)
    assert block_groups(*a,q,[0.,0.],mapping)[0]=={}
    g,_=block_groups(*a,q,[.4,.4],mapping)
    assert g=={.4:{'10000000000001','10000000000002'}}


@pytest.mark.parametrize('bad',[[-.1,.2],[.2,1.1],[float('nan'),.3]])
def test_relevance_range_not_probability_claim(bad):
    with pytest.raises(ValueError):block_groups(*arrays(),RetrievalQuery(name='N'),bad,np.array([[0,-1],[1,-1]],np.int32))


def test_no_reinterpretation_of_invalid_code_or_shape():
    a=list(arrays());a[4][0]=4
    with pytest.raises(ValueError):block_groups(*a,RetrievalQuery(name='N'),[.2,.8],np.array([[0,-1],[1,-1]],np.int32))


@pytest.mark.parametrize('chunk',[1,2,3,10])
def test_complete_real_readonly_scan_is_chunk_invariant_and_hash_bound(tmp_path,chunk):
    vals=arrays();refs=[]
    for i,a in enumerate(vals):
        p=tmp_path/f'a{i}.npy';np.save(p,a);refs.append(record(p))
    files=SourceFiles(lambda:None)
    e=SimpleNamespace(keys_ref=refs[0],sites=4,files=files,structured=dict(
        artifacts={'postcode_codes.npy':refs[1],'insee_codes.npy':refs[2]},
        reused_arrays=dict(nomenclature_codes=refs[3],original_activity_codes=refs[4])))
    q=RetrievalQuery(name='N',postcode='11111',insee='11111')
    mapping=np.array([[0,-1],[1,-1]],np.int32)
    g,c=collect_groups(e,q,[.2,.8],mapping,chunk=chunk)
    assert g==block_groups(*vals,q,[.2,.8],mapping)[0] and c['scanned']==4
    assert len(files.entries)==5
    files.verify_after()


def test_real_native_continuous_grades_unique_memberships():
    b=tiny_native();p=native_proof(b,{.2:{'10000000000001'},.8:{'10000000000002'}})
    got={b._first(b.searcher.doc(loc).to_dict(),'siret'):s for s,loc in b.searcher.search(p,limit=100).hits}
    assert got==pytest.approx({'10000000000001':.2,'10000000000002':.8})
    with pytest.raises(ValueError):native_proof(b,{.2:{'10000000000001'},.8:{'10000000000001'}})
    assert native_proof(b,{}) is None
