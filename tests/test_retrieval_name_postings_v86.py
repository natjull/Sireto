from datetime import date

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_name_facts_v85 import factorize,invert
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings,build_metadata,build_owners,build_postings,masked_sql
from src.xgb_matcher.retrieval_proof_v56 import signal
from tests.test_retrieval_name_facts_v85 import fixture


def test_sql_mask_exactly_matches_authoritative_signal_for_numeric_boundaries():
    rng=np.random.default_rng(86)
    texts=['A 111111111 222222222 B','ABC111111111DEF','ABC111111111111111111DEF','00000000000000',
           '42 00000000000000 12345','111111111 222222222 333333333 444444444 555555555']
    for _ in range(200):
        words=['A'+''.join(str(x) for x in rng.integers(10,size=int(rng.integers(1,24))))+'Z' for _ in range(6)]
        texts.append(' '.join(words))
    c=duckdb.connect()
    for text in texts:
        assert c.execute('SELECT '+masked_sql('?'),[text]).fetchone()[0]==signal(text)
    c.close()


def test_complete_postings_separate_aliases_and_match_whole_fact_oracle(tmp_path):
    source,rows=fixture(tmp_path)
    # Canonical names require masked identifiers even inside letters.
    rows[4]['normalized_value']='CLINIQUE111111111 PORT PORT'
    pq.write_table(pa.Table.from_pylist(rows),source)
    facts=tmp_path/'facts'; facts.mkdir(); out=tmp_path/'index'; out.mkdir()
    c=duckdb.connect(); stats=factorize(c,source,facts); c.close(); invert(facts,stats)
    c=duckdb.connect(); posted=build_postings(c,stats['parts'],out); c.close()
    owners=build_owners(stats['parts'],out); index=WordPostings(posted)
    assert '111111111' not in index.terms and 'NEVER' not in index.terms
    assert owners['parent']['facts']+owners['site']['facts']==posted['facts']
    corpus={}
    for part in stats['parts']:
        for local,row in enumerate(pq.read_table(part['path']).to_pylist()): corpus[part['fact_start']+local]=row
    for query in ({'CLINIQUE':.4,'CENTRE':.6},{'PORT':1.},{'PHARMACIE':1.},{'ABSENT':1.}):
        actual,trace=index.scores(query)
        expected=np.array([sum(w for t,w in sorted(query.items()) if row['scope_code']!=0 and t in set(signal(row['name_text']).split()))
                           for _,row in sorted(corpus.items())])
        assert np.array_equal(actual,expected) and trace['fixed_name_fact_pool'] is None
    score,_=index.scores({'CLINIQUE':.4,'CENTRE':.6})
    site=[score[i] for i,row in corpus.items() if row['subject_code']==2]
    assert max(site)==.6 # No invented CLINIQUE CENTRE alias scoring1.


def test_all_original_observation_dates_and_unknown_priority_preserved(tmp_path):
    source,rows=fixture(tmp_path)
    for i,row in enumerate(rows):
        row['observed_at']=date(2026,1,1) if i%2 else None
        row['is_current']=None if i%3==0 else bool(i%2)
        row['source_priority']=None if i%2 else 0
    table=pa.Table.from_pylist(rows)
    for key in ('valid_from','valid_to','observed_at'):
        table=table.set_column(table.column_names.index(key),pa.field(key,pa.date32()),table[key].cast(pa.date32()))
    pq.write_table(table,source); result=build_metadata(source,tmp_path)
    assert result['observations']==9
    values={k:np.load(tmp_path/(k+'.npy')) for k in ('valid_from_days','observed_days','current_codes','priority_known','priorities')}
    assert values['valid_from_days'][0]==np.iinfo(np.int32).min
    assert values['valid_from_days'][1]==(date(2000,1,1)-date(1970,1,1)).days
    assert values['current_codes'][0]==0 and values['current_codes'][1]==2 and values['current_codes'][2]==1
    assert values['priorities'][0]==values['priorities'][1]==0
    assert values['priority_known'][0]==1 and values['priority_known'][1]==0


def test_noncanonical_text_requires_authoritative_normalizer_not_silent_sql_substitute(tmp_path):
    path=tmp_path/'facts.parquet'; pq.write_table(pa.Table.from_pylist([dict(owner_key=1,official_parent=1,
        subject_code=1,scope_code=1,name_text='ÉCOLE')]),path); c=duckdb.connect()
    with pytest.raises(ValueError,match='Noncanonical'): build_postings(c,[dict(path=str(path),facts=1,fact_start=0)],tmp_path)
    c.close()
