import json

import pytest
import tantivy

from scripts.build_hierarchical_retrieval_index import _schema
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,RetrievalQuery,character_ngrams
from src.xgb_matcher.retrieval_v25_conjunctive import search,config_at


@pytest.fixture
def backend(tmp_path):
    (tmp_path/'tantivy').mkdir()
    index=tantivy.Index(_schema(tantivy),path=str(tmp_path/'tantivy'))
    writer=index.writer(heap_size=30_000_000,num_threads=1)
    for i in range(520):
        doc=tantivy.Document()
        for key,value in dict(document_type='siret',siret=f'{i:014d}',siren='999999999',
            number='2' if i==0 else '1',insee='22222' if i==1 else '11111',postcode='11100',
            address_ngrams=' '.join(character_ngrams('1 RUE BLEUE'))).items():
            doc.add_text(key,value)
        writer.add_document(doc)
    writer.commit();writer.wait_merging_threads()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(typed_name_portfolio=True)))
    return TantivyBackend(tmp_path)


def test_conjunction_number_geography_and_stored_parent(backend):
    q=RetrievalQuery('ALPHA','1 RUE BLEUE','1','11100','11111')
    found=search(backend,q,512)
    assert len(found)==512
    assert not {'00000000000000','00000000000001'} & {r['siret'] for r in found}
    assert {r['siren'] for r in found}=={'999999999'}  # never derived from SIRET prefix
    assert [r['candidate_rank'] for r in found]==list(range(1,513))
    assert {r['evidence_rank'] for r in found}=={1}  # IDs do not create evidence ranks
    assert all(found[i]['siret']<found[i+1]['siret'] for i in range(511))


def test_missing_number_or_address_is_empty_not_fabricated(backend):
    assert search(backend,RetrievalQuery('ALPHA','1 RUE BLEUE','','11100','11111'))==[]
    assert search(backend,RetrievalQuery('ALPHA','','1','11100','11111'))==[]
    assert search(backend,RetrievalQuery('ALPHA','1 RUE BLEUE','1','',''))==[]


def test_channel_cap_and_config_are_frozen(backend,tmp_path):
    with pytest.raises(ValueError):search(backend,RetrievalQuery('ALPHA'),513)
    c=config_at('config/retrieval_v25_conjunctive.json');c['site_cap_per_query']=1024
    p=tmp_path/'invalid.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):config_at(p)
