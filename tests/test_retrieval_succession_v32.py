from datetime import date
import random

import duckdb
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_succession_v32 import build,reference


def edge(seed='11111111100001',target='22222222200001',when=date(2020,1,1),relation='a',forward=True):
    a,b=(seed,target) if forward else (target,seed)
    return dict(predecessor_siret=a,predecessor_siren=a[:9],successor_siret=b,successor_siren=b[:9],
      effective_date=when,relation_id=relation,economic_continuity=True)


def sql(raw,edges,threads=1):
    c=duckdb.connect();c.execute(f'SET threads={threads}')
    c.register('raw',pa.Table.from_pylist(raw));c.register('edges',pa.Table.from_pylist(edges))
    build(c);result=c.execute('SELECT * FROM v32_proof ORDER BY query_id,fold,channel_rank').fetch_arrow_table().to_pylist();c.close()
    return result


def seeds():return [dict(query_id='q',fold=2,siret='11111111100001',union_rank=1),dict(query_id='q',fold=2,siret='11111111100002',union_rank=2)]


def test_seed_first_then_latest_event_row_coherence():
    rows=[edge(target='22222222200003',when=date(2021,1,1),relation='first'),
      edge(target='22222222200002',when=date(2022,1,1),relation='chosen',forward=False),
      edge(seed='11111111100002',target='22222222200001',when=date(2025,1,1),relation='later')]
    actual=sql(seeds(),rows)
    assert actual==reference(seeds(),rows)
    r=actual[0]
    assert r['linked_siret']=='22222222200002' and r['direction']=='SUCCESSOR_TO_PREDECESSOR'
    assert r['relation_id']=='chosen' and r['witness_effective_date']==date(2022,1,1)
    assert r['parent_latest_effective_date']==date(2025,1,1)


@pytest.mark.parametrize('threads',[1,2,8])
def test_permutations_duplicates_and_shards(threads):
    raw=seeds()+[dict(query_id='other',fold=3,siret='11111111100001',union_rank=2)]
    rows=[edge(target=f'222222222{i:05d}',when=date(2022,1,1),relation=str(i),forward=i%2==0) for i in range(1,12)]
    rows.append(dict(rows[0]));expected=reference(raw,rows)
    random.Random(19).shuffle(rows);random.Random(53).shuffle(raw)
    assert sql(raw,rows,threads)==expected
    chunks=[sql([r for r in raw if r['query_id']==q],rows,threads) for q in ('q','other')]
    assert sorted(sum(chunks,[]),key=lambda r:(r['query_id'],r['fold'],r['channel_rank']))==expected


def test_null_dates_known_date_preferred_only_at_equal_seed():
    rows=[edge(when=None),edge(target='22222222200002',when=date(2010,1,1)),
          edge(seed='11111111100002',target='33333333300001',when=None)]
    assert sql(seeds(),rows)==reference(seeds(),rows)
    assert sql(seeds(),rows)[0]['linked_siret']=='22222222200002'


def test_no_direction_preference_or_id_as_relevance():
    # Newer backward event wins despite larger identifier; recency is not numeric ID order.
    rows=[edge(target='22222222200001'),edge(target='22222222299999',when=date(2023,1,1),forward=False)]
    assert sql(seeds(),rows)[0]['linked_siret']=='22222222299999'


def test_continuity_seed_budget_and_parent_cap():
    raw=[dict(query_id='q',fold=2,siret='11111111100001',union_rank=250),dict(query_id='q',fold=2,siret='11111111100002',union_rank=251)]
    rows=[edge(target=f'{i:09d}00001') for i in range(200000000,200000260)]
    rows.append(dict(edge(target='12345678900001'),economic_continuity=False))
    rows.append(edge(seed='11111111100002',target='12345678000001'))
    actual=sql(raw,rows);assert actual==reference(raw,rows) and len(actual)==256


def test_forbidden_fold_rejected():
    with pytest.raises(ValueError,match='Forbidden fold'):
        sql([dict(query_id='q',fold=1,siret='11111111100001',union_rank=1)],[edge()])


def test_duplicate_raw_rejected():
    with pytest.raises(ValueError,match='Duplicate raw'):
        sql([seeds()[0],seeds()[0]],[edge()])
