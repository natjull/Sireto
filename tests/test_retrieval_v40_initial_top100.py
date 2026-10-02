import random

import pytest

from scripts.retrieval_v40_initial_top100 import collect


class Stream:
    def __init__(self,rows): self.rows=list(rows); self.seen=[]
    def peek(self,remaining):
        assert 0<remaining<=100
        return self.rows[0]['score'] if self.rows else None
    def pop(self):
        row=self.rows.pop(0); self.seen.append(row); return row


def test_stream_collector_is_exact_max_view_score_with_duplicates():
    rng=random.Random(50)
    sources={}
    for key in ('national','bodacc'):
        rows=[dict(siret=str(i),siren='parent',score=rng.random()) for i in rng.sample(range(250),180)]
        sources[key]=sorted(rows,key=lambda r:-r['score'])
    expected={}
    for rows in sources.values():
        for row in rows: expected[row['siret']]=max(expected.get(row['siret'],0),row['score'])
    best=sorted(expected,key=expected.get,reverse=True)[:100]
    streams={k:Stream(rows) for k,rows in sources.items()}
    result,trace=collect(streams)
    assert [r['siret'] for r in result]==best
    assert all(r['score']==expected[r['siret']] for r in result)
    assert trace['next_score_upper_bound']<=result[-1]['score']
    assert sum(len(s.seen) for s in streams.values())<=200


def test_empty_sources_short_population_and_conflicting_membership():
    result,_=collect(dict(a=Stream([]),b=Stream([dict(siret='one',siren='parent',score=1.)])))
    assert len(result)==1
    with pytest.raises(ValueError,match='membership'):
        collect(dict(a=Stream([dict(siret='one',siren='p1',score=2.)]),b=Stream([dict(siret='one',siren='p2',score=1.)])))


def test_invalid_cap():
    with pytest.raises(ValueError): collect({},101)


def test_native_pagination_is_unique_and_stable_with_score_ties(tmp_path):
    import json
    import tantivy
    from scripts.retrieval_v40_initial_top100 import NativeStream
    from scripts.retrieval_v39_bodacc_index import schema

    idx=tantivy.Index(schema(),path=str(tmp_path))
    writer=idx.writer(heap_size=16*1024**2,num_threads=1)
    for i in range(150):
        doc=tantivy.Document(); doc.add_text('anchor_names','SAME NAME')
        doc.add_bytes('identity',json.dumps(dict(siret=str(i),siren='parent')).encode())
        writer.add_document(doc)
    writer.commit(); writer.wait_merging_threads(); idx.reload()
    query=tantivy.Query.term_query(idx.schema,'anchor_names','same','basic')
    stream=NativeStream(idx.searcher(),query,lambda d:json.loads(bytes(d['identity'][0])))
    found,_=collect(dict(native=stream))
    assert len(found)==100 and len({r['siret'] for r in found})==100
    assert stream.calls==1 and stream.offset==100
    probe=NativeStream(idx.searcher(),query,lambda d:json.loads(bytes(d['identity'][0])),probe_first=True)
    probed,_=collect(dict(native=probe))
    assert found==probed and probe.calls==2 and probe.offset==100
