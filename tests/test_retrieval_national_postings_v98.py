import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_national_postings_v98 import prepare,word_bucket
from src.xgb_matcher.retrieval_name_postings_v86 import WordPostings,build_owners,build_postings
from src.xgb_matcher.retrieval_proof_v56 import signal


def test_complete_postings_include_all_aliases_without_empty_context_or_identifier_leaks(tmp_path):
    rows=[dict(subject_code=1,owner_key=111111111,official_parent=111111111,scope_code=1,name_text='CLINIQUE111111111 PORT PORT'),
        dict(subject_code=2,owner_key=11111111100001,official_parent=111111111,scope_code=3,name_text='CENTRE'),
        dict(subject_code=1,owner_key=111111111,official_parent=111111111,scope_code=0,name_text='FORBIDDEN CONTEXT'),
        dict(subject_code=2,owner_key=11111111100002,official_parent=111111111,scope_code=0,name_text=''),
        dict(subject_code=2,owner_key=11111111100003,official_parent=111111111,scope_code=0,name_text=None),
        dict(subject_code=1,owner_key=222222222,official_parent=222222222,scope_code=1,name_text='22222222222222')]
    source=tmp_path/'facts.parquet'; pq.write_table(pa.Table.from_pylist(rows),source)
    parts=[dict(path=str(source),facts=len(rows),fact_start=0)]
    out=tmp_path/'new'; out.mkdir(); c=duckdb.connect(); stats=prepare(c,parts,out)
    buckets=[s for b in range(32) if (s:=word_bucket(c,out,b,len(rows))) is not None]; c.close()
    index=WordPostings(dict(**stats,buckets=buckets)); owners=build_owners(parts,out)
    assert index.facts==len(rows) and owners['parent']['facts']+owners['site']['facts']==len(rows)
    assert stats['context_only_facts']==3 and stats['identifier_only_admissible_facts']==1
    assert 'FORBIDDEN' not in index.terms and '111111111' not in index.terms
    for terms in [{'PORT':1.},{'CLINIQUE':.4,'CENTRE':.6},{'ABSENT':1.}]:
        actual,_=index.scores(terms)
        expected=np.array([sum(w for t,w in sorted(terms.items()) if r['scope_code'] and t in set(signal(r['name_text'] or '').split())) for r in rows])
        assert np.array_equal(actual,expected)
    # V86 parity on the previously admissible nonempty canonical subset.
    old_source=tmp_path/'old_facts.parquet'; old_rows=[r for r in rows if r['name_text']]
    pq.write_table(pa.Table.from_pylist(old_rows),old_source); old_out=tmp_path/'old'; old_out.mkdir(); c=duckdb.connect()
    old_stats=build_postings(c,[dict(path=str(old_source),facts=len(old_rows),fact_start=0)],old_out); c.close()
    old_index=WordPostings(old_stats)
    for terms in [{'PORT':1.},{'CLINIQUE':.4,'CENTRE':.6}]:
        old,_=old_index.scores(terms); new,_=index.scores(terms)
        assert old.tolist()==[new[i] for i,r in enumerate(rows) if r['name_text']]


@pytest.mark.parametrize('scope,text',[(1,''),(3,None),(0,'ÉCOLE'),(3,'LOWER lower')])
def test_noncanonical_lexical_text_is_not_silently_repaired(tmp_path,scope,text):
    path=tmp_path/'facts.parquet'; pq.write_table(pa.Table.from_pylist([dict(scope_code=scope,name_text=text)],
        schema=pa.schema([('scope_code',pa.uint8()),('name_text',pa.string())])),path)
    c=duckdb.connect()
    with pytest.raises(ValueError,match='Noncanonical'): prepare(c,[dict(path=str(path),facts=1,fact_start=0)],tmp_path)
    c.close()
