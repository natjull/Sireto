import copy

import pytest

from scripts.audit_retrieval_v92_scoped_evidence import assemble


def test_full128_assembly_requires_all_same_authority_resource_and_exact_hits():
    queries=[dict(query_id=str(i),fold=2+i%3) for i in range(128)]
    rows=[dict(**q,exact_hit=i<114) for i,q in enumerate(queries)]
    batches=[dict(rows=rows[i:i+16],generation={'sha256':'G'},evaluation={'sha256':'E'},
        generation_modified=False,truth_scope='EVALUATOR_ONLY_AFTER_SEALED128',
        resources=dict(pass_resource_policy=True)) for i in range(0,128,16)]
    args=(queries,dict(primary=dict(exact_hits=114)))
    assert assemble(*args,batches,{'sha256':'G'},{'sha256':'E'})==rows
    for mutation in ('partial','order','authority','resource','truth','fold','hits'):
        bad=copy.deepcopy(batches)
        if mutation=='partial': bad.pop()
        elif mutation=='order': bad.reverse()
        elif mutation=='authority': bad[0]['generation']={'sha256':'OTHER'}
        elif mutation=='resource': bad[0]['resources']['pass_resource_policy']=False
        elif mutation=='truth': bad[0]['generation_modified']=True
        elif mutation=='fold': bad[0]['rows'][0]['fold']=1
        else: bad[0]['rows'][0]['exact_hit']=False
        with pytest.raises(ValueError): assemble(*args,bad,{'sha256':'G'},{'sha256':'E'})
