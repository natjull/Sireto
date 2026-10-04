import duckdb
import numpy as np
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_direct_alias_v172 import evaluate
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized


@pytest.mark.parametrize('query',['RUE JEAN JAURES','JEAN','N','AA AA AAA','', 'CH 390865069 39086506900023',"A & B D'ETE"])
def test_direct_physical_execution_equals_every_scalar_whole_variant(query):
    raw=['RUE JEAN JAURES','RUE JEAN JAURES','A ET B D ETE','JEAN JAURES RUE','JEAN','AA AA AAA',
        'RUEJEAN','3912 RUE JEAN','CH 390865069 39086506900023','390865069','']
    texts=[normalized(x) for x in raw]
    table=pa.table(dict(alias_id=pa.array(np.arange(len(texts),dtype=np.uint32)),feature_text=pa.array(texts)))
    c=duckdb.connect()
    try:ids,cov,exact,stats=evaluate(c,table,query)
    finally:c.close()
    q=channels(query)
    expected=np.array([max((len(a&b)/len(a) if a else 0.) for a,b in zip(q,channels(s))) for s in texts])
    flags=np.array([bool(normalized(query)) and s==normalized(query) for s in texts],dtype=np.bool_)
    np.testing.assert_array_equal(cov,expected);np.testing.assert_array_equal(exact,flags)
    assert ids.tolist()==list(range(len(texts))) and stats['no_alias_or_candidate_cap'] and not stats['recall_measured']


def test_direct_word_boundaries_and_duplicate_occurrences_never_change_coverage():
    c=duckdb.connect();texts=['JEAN','JEAN JEAN JEAN','RUEJEAN','JEANNETTE']
    table=pa.table(dict(alias_id=pa.array(np.arange(4,dtype=np.uint32)),feature_text=pa.array(texts)))
    try:_,cov,_,_=evaluate(c,table,'JEAN')
    finally:c.close()
    # Max-over-char channels can match a substring; word boundary alone is
    # not a new hard filter. Counts remain presence counts, never repetitions.
    assert cov.tolist()==[1.,1.,1.,1.]


@pytest.mark.parametrize('ids',[[0,0],[1,0],[None,1]])
def test_duplicate_reordered_or_unknown_aliases_rejected(ids):
    c=duckdb.connect()
    table=pa.table(dict(alias_id=pa.array(ids,type=pa.uint32()),feature_text=pa.array(['JEAN','JEAN'])))
    try:
        with pytest.raises(ValueError):evaluate(c,table,'JEAN')
    finally:c.close()
