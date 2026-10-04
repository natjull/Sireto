import duckdb
import numpy as np
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_direct_alias_v174 import evaluate,TOKEN_QUANTUM
from src.xgb_matcher.retrieval_direct_alias_v172 import evaluate as previous
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized


@pytest.mark.parametrize('query',['RUE JEAN JAURES','JEAN','N','AA AA AAA','', 'CH 390865069 39086506900023',
    "A & B D'ETE",' '.join('W'+str(i) for i in range(300))])
def test_vector_groups_consume_every_token_and_alias_bit_equal_to_scalar_and_prior(query):
    raw=['RUE JEAN JAURES','RUE JEAN JAURES','A ET B D ETE','JEAN JAURES RUE','JEAN','AA AA AAA',
        'RUEJEAN','3912 RUE JEAN','CH 390865069 39086506900023','390865069','',query]
    texts=[normalized(x) for x in raw]
    table=pa.table(dict(alias_id=pa.array(np.arange(len(texts),dtype=np.uint32)),feature_text=pa.array(texts)))
    c=duckdb.connect()
    try:
        ids,cov,exact,stats=evaluate(c,table,query)
        pi,pc,pe,_=previous(c,table,query)
        # Temp table removed; same connection and source remain usable.
        assert c.execute('SELECT count(*) FROM duckdb_tables() WHERE table_name=\'direct174_base\'').fetchone()[0]==0
    finally:c.close()
    q=channels(query)
    expected=np.array([max((len(a&b)/len(a) if a else 0.) for a,b in zip(q,channels(s))) for s in texts])
    flags=np.array([bool(normalized(query)) and s==normalized(query) for s in texts],dtype=np.bool_)
    np.testing.assert_array_equal(ids,pi);np.testing.assert_array_equal(cov,pc);np.testing.assert_array_equal(exact,pe)
    np.testing.assert_array_equal(cov,expected);np.testing.assert_array_equal(exact,flags)
    assert stats['groups']==sum((len(s)+TOKEN_QUANTUM-1)//TOKEN_QUANTUM for s in q)
    assert stats['all_query_tokens_consumed'] and stats['no_alias_or_candidate_cap']


@pytest.mark.parametrize('ids',[[0,0],[1,0],[None,1]])
def test_invalid_alias_namespace_refused(ids):
    c=duckdb.connect()
    table=pa.table(dict(alias_id=pa.array(ids,type=pa.uint32()),feature_text=pa.array(['JEAN','JEAN'])))
    try:
        with pytest.raises(ValueError):evaluate(c,table,'JEAN')
    finally:c.close()
