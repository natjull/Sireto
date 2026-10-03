import numpy as np
import pytest

from src.xgb_matcher import retrieval_compact_coverage_v156 as compact
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from test_retrieval_national_names_v138 import setup


@pytest.mark.parametrize('n,name',[(0,'uint8'),(1,'uint8'),(255,'uint8'),(256,'uint16'),
    (65535,'uint16'),(65536,'uint32'),(2**32-1,'uint32')])
def test_count_bound_always_fits_without_saturation(n,name):
    dtype=compact.counter_dtype(n)
    assert dtype.name==name and np.iinfo(dtype).max>=n


@pytest.mark.parametrize('n',[-1,2**32,True,1.1,None])
def test_invalid_count_is_not_truncated(n):
    with pytest.raises(ValueError,match='distinct query-token'):compact.counter_dtype(n)


def test_all_features_bit_exact_across_channels_and_map_lifetimes(tmp_path):
    ref,_=setup(tmp_path);baseline=NationalNames(ref);new=compact.CompactNamePages(ref)
    try:
        queries=['ALPHA BETA','ALPHA DELTA','COMMUNE DE PARIS','BETA GAMMA','[ND]',
                 '123456789','FÖÖ','','ALPHA BETA 123456789']*3
        for epoch,text in enumerate(queries,1):
            cov,ex,stats=baseline.coverage(text,chunk=2);nc,ne,ns=new.coverage(text,chunk=2)
            assert nc.tobytes()==cov.tobytes() and ne.tobytes()==ex.tobytes()
            assert nc.dtype==np.float64 and ne.dtype==np.bool_ and nc.flags.owndata and ne.flags.owndata
            assert ns['channels']==stats['channels'] and ns['positive_aliases']==stats['positive_aliases']
            assert ns['exact_aliases']==stats['exact_aliases'] and ns['postings_or_alias_cap'] is None
            assert ns['source_page_rotation']['epoch']==epoch
            assert ns['counter_dtype']=='uint8' and ns['counter_bytes']==new.names
        public=new.posting(3,'ALPHA BETA');assert public.flags.owndata
        owned=nc.copy();new.close();assert nc.tobytes()==owned.tobytes() and public.tolist()==[0,1]
        with pytest.raises(ValueError,match='Single live'):new.coverage('ALPHA')
    finally:baseline.close();new.close()


def test_more_than_255_matching_tokens_never_overflow(monkeypatch):
    tokens=frozenset(f'T{i}' for i in range(257))
    monkeypatch.setattr(compact,'channels',lambda text:(tokens,frozenset(),frozenset()))
    class Index:
        names=3
        def posting(self,channel,token):
            return np.array([0,1],np.uint32) if channel==0 else np.array([],np.uint32)
    cov,ex,stats=compact.coverage(Index(),'Q',chunk=1)
    assert cov.tolist()==[1.,1.,0.] and not ex.any()
    assert stats['counter_dtype']=='uint16' and stats['counter_bytes']==6
    assert stats['channels'][0]['postings']==514


def test_real_long_query_uses_wider_counter_with_bit_exact_float64(tmp_path):
    ref,_=setup(tmp_path);baseline=NationalNames(ref);new=compact.CompactNamePages(ref)
    text='ALPHA BETA '+' '.join(f'WORD{i}' for i in range(256))
    try:
        expected,exact,_=baseline.coverage(text)
        actual,flags,stats=new.coverage(text)
        assert stats['counter_dtype']=='uint16' and stats['counter_bound']>255
        assert actual.tobytes()==expected.tobytes() and flags.tobytes()==exact.tobytes()
    finally:baseline.close();new.close()


def test_huge_single_posting_is_not_capped():
    class Index:
        names=10001
        def posting(self,channel,token):
            return np.arange(self.names,dtype=np.uint32)
    cov,ex,stats=compact.coverage(Index(),'ALPHA',chunk=17)
    assert len(cov)==10001 and np.all(cov==1.) and ex.all()
    assert stats['positive_aliases']==10001 and stats['postings_or_alias_cap'] is None


def test_query_guard_does_not_publish_partial_vectors(tmp_path,monkeypatch):
    ref,_=setup(tmp_path);reader=compact.CompactNamePages(ref)
    times=iter([0,2]);monkeypatch.setattr(compact.time,'monotonic',lambda:next(times))
    try:
        with pytest.raises(TimeoutError,match='STOP_OFFICIAL'):reader.coverage('ALPHA',max_seconds=1)
        assert reader.in_coverage is False and reader.epoch==0
    finally:reader.close()
