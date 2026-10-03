import json

import numpy as np
import pytest

from src.xgb_matcher.retrieval_name_pages_v151 import ScopedNamePages
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from src.xgb_matcher.retrieval_v20_workbench import verify
from test_retrieval_national_names_v138 import setup


def test_all_vectors_bit_exact_across_repeated_remaps_and_raw_alias_collisions(tmp_path):
    ref,_=setup(tmp_path);baseline=NationalNames(ref);scoped=ScopedNamePages(ref)
    try:
        queries=['ALPHA BETA','ALPHA DELTA','COMMUNE DE PARIS','BETA GAMMA','[ND]','123456789','FÖÖ','']*3
        for epoch,q in enumerate(queries,1):
            c,e,s=baseline.coverage(q,chunk=2);nc,ne,t=scoped.coverage(q,chunk=2)
            np.testing.assert_array_equal(nc,c);np.testing.assert_array_equal(ne,e)
            assert nc.flags.owndata and ne.flags.owndata
            assert t['channels']==s['channels'] and t['positive_aliases']==s['positive_aliases']
            assert t['exact_aliases']==s['exact_aliases'] and t['postings_or_alias_cap'] is None
            rot=t['source_page_rotation'];assert rot['epoch']==epoch and rot['maps_reopened']==32
            assert rot['full_alias_and_posting_namespace_unchanged'] and not rot['global_cache_purged']
        assert scoped.posting(3,'ALPHA BETA').tolist()==[0,1]
    finally:baseline.close();scoped.close()


def test_public_postings_do_not_borrow_maps_and_owned_outputs_survive_rotation(tmp_path):
    ref,_=setup(tmp_path);reader=ScopedNamePages(ref)
    old=reader.posting(3,'ALPHA BETA');assert old.flags.owndata
    cov,exact,_=reader.coverage('ALPHA BETA');expected=cov.copy();flags=exact.copy()
    reader.coverage('COMMUNE DE PARIS');reader.close()
    assert old.tolist()==[0,1];np.testing.assert_array_equal(cov,expected);np.testing.assert_array_equal(exact,flags)
    with pytest.raises(ValueError,match='Single live'):reader.coverage('ALPHA')
    with pytest.raises(ValueError,match='Closed'):reader.posting(3,'ALPHA BETA')
    reader.close()


def test_changed_file_epoch_blocks_remap_and_future_queries_never_use_prefix(tmp_path):
    ref,_=setup(tmp_path);reader=ScopedNamePages(ref);m=json.loads(verify(ref).read_text())
    path=verify(m['access'][1]['artifacts']['post_ids.npy'])
    with path.open('ab') as f:f.write(b'changed')
    try:
        with pytest.raises(ValueError,match='epoch drift'):reader.rotate()
        assert not reader.valid
        with pytest.raises(ValueError,match='Single live'):reader.coverage('ALPHA')
        with pytest.raises(ValueError,match='Closed'):reader.posting(0,'ALPHA')
    finally:reader.close()


def test_remap_failure_after_a_prefix_closes_safely_never_claims_full_scope(tmp_path,monkeypatch):
    ref,_=setup(tmp_path);reader=ScopedNamePages(ref);real=np.load;count=0
    def fail(*args,**kwargs):
        nonlocal count
        count+=1
        if count==7:raise ValueError('failed reopening complete source')
        return real(*args,**kwargs)
    monkeypatch.setattr(np,'load',fail)
    with pytest.raises(ValueError,match='failed reopening'):reader.rotate()
    assert not reader.valid and reader.epoch==0
    with pytest.raises(ValueError,match='Single live'):reader.coverage('ALPHA')
    reader.close()


def test_header_or_shape_drift_cannot_be_accepted_as_same_index(tmp_path,monkeypatch):
    ref,_=setup(tmp_path);reader=ScopedNamePages(ref);real=np.load;count=0
    def wrong(*args,**kwargs):
        nonlocal count
        count+=1
        if count==1:return np.zeros(5,dtype=np.uint32)
        return real(*args,**kwargs)
    monkeypatch.setattr(np,'load',wrong)
    with pytest.raises(ValueError,match='Exact unchanged readonly'):reader.rotate()
    assert not reader.valid
    reader.close()
