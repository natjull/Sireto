"""Synthetic tiny unit fixtures only, never business sources or recall evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_retrieval_v187_staged_initial as original
from src.xgb_matcher import retrieval_query_population_v192 as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json


def fixture(tmp_path):
    queries = [dict(query_id='CONTROL_ONLY_'+str(i), fold=(3 if i == 0 else 2+i%3),
                    crm_name='NAME_'+str(i), crm_number='', crm_suffix='', crm_street='',
                    crm_postcode='', crm_insee='') for i in range(128)]
    path = tmp_path / 'vectors.npy'
    np.save(path, np.eye(128, 384, dtype=np.float32))
    old = dict(queries=queries, query_sources=[{'control': 'DO_NOT_FOLLOW'}], query_vectors=record(path),
               truth_opened=False, old_candidates_opened=False)
    encoder = tmp_path / 'encoder_launch.json'
    write_json(encoder, dict(queries=deepcopy(queries), query_sources=old['query_sources'],
                            truth_opened=False, old_candidates_opened=False,
                            bundle={'path': 'MUST_NOT_BE_OPENED'}))
    encoded = tmp_path / 'encoded.json'
    write_json(encoded, dict(rows=[dict(query_id=q['query_id'], fold=q['fold'],
                                       legacy_tokens=['MUST_NOT_BE_USED_AS_FEATURE']) for q in queries],
                             vectors=old['query_vectors'], truth_opened=False,
                             resources={'pass_resource_policy': True}))
    return old, record(encoder), record(encoded)


def test_whole_population_own_vectors_and_queries_never_vector0_for_another(tmp_path):
    old, encoder, encoded = fixture(tmp_path)
    pop = workbench.load(old, encoder, encoded)
    assert not pop.vectors.flags.writeable
    for i in range(128):
        row, vector = pop.select(i)
        assert row == old['queries'][i] and vector.flags.owndata
        np.testing.assert_array_equal(vector, np.eye(128, 384, dtype=np.float32)[i:i+1])
        row['crm_name'] = 'UNTRUSTED_COPY'
        vector[:] = 0.
        assert pop.select(i)[0] == old['queries'][i] and pop.vectors[i, i] == 1.
    assert tuple(i for group in workbench.batches() for i in group) == tuple(range(128))


@pytest.mark.parametrize('change', ['last_fold0', 'last_fold1', 'bool_fold', 'string_field',
                                   'missing_row', 'duplicate_id', 'label_field', 'truth', 'old_candidates'])
def test_all128_validated_before_any_file_read_or_query_selection(tmp_path, change):
    old, encoder, encoded = fixture(tmp_path)
    q = old['queries'][-1]
    if change == 'last_fold0': q['fold'] = 0
    elif change == 'last_fold1': q['fold'] = 1
    elif change == 'bool_fold': q['fold'] = True
    elif change == 'string_field': q['crm_number'] = 7
    elif change == 'missing_row': old['queries'].pop()
    elif change == 'duplicate_id': q['query_id'] = old['queries'][0]['query_id']
    elif change == 'label_field': q['siret_humain'] = 'MUST_NOT_USE'
    elif change == 'truth': old['truth_opened'] = True
    else: old['old_candidates_opened'] = True
    with pytest.raises(ValueError):
        workbench.load(old, {'path': 'NO_FILE_MUST_BE_READ'}, {'path': 'NO_FILE_MUST_BE_READ'})


@pytest.mark.parametrize('change', ['encoder_order', 'query_source', 'encoded_order', 'encoded_truth',
                                   'different_vectors', 'last_nan', 'last_not_unit'])
def test_entire_encoder_vector_order_and_content_required(tmp_path, change):
    import json
    old, encoder, encoded = fixture(tmp_path)
    ep = __import__('pathlib').Path(encoder['path'])
    cp = __import__('pathlib').Path(encoded['path'])
    e, c = json.loads(ep.read_text()), json.loads(cp.read_text())
    if change == 'encoder_order': e['queries'][-2:] = e['queries'][-2:][::-1]
    elif change == 'query_source': e['query_sources'] = []
    elif change == 'encoded_order': c['rows'][-2:] = c['rows'][-2:][::-1]
    elif change == 'encoded_truth': c['truth_opened'] = True
    elif change == 'different_vectors': c['vectors']['sha256'] = 'f'*64
    else:
        vectors = np.load(old['query_vectors']['path'])
        if change == 'last_nan': vectors[-1, -1] = np.nan
        else: vectors[-1] *= 2.
        np.save(old['query_vectors']['path'], vectors)
        old['query_vectors'] = record(old['query_vectors']['path'])
        c['vectors'] = old['query_vectors']
    write_json(ep, e); write_json(cp, c)
    with pytest.raises(ValueError):
        workbench.load(old, record(ep), record(cp))


@pytest.mark.parametrize('position', [-1, 128, True, 1., '1'])
def test_never_clip_or_coerce_query_position(tmp_path, position):
    old, encoder, encoded = fixture(tmp_path)
    with pytest.raises(ValueError):
        workbench.load(old, encoder, encoded).select(position)


def activity_fixture(tmp_path):
    old, encoder, encoded = fixture(tmp_path)
    pop = workbench.load(old, encoder, encoded)
    docs = tmp_path / 'docs.npy'; np.save(docs, np.eye(3, 384, dtype=np.float32))
    cat = tmp_path / 'catalog.json'
    label_ref = {'path': 'DO_NOT_OPEN_EXTRA_SOURCES', 'sha256': 'f'*64}
    write_json(cat, dict(official_labels=label_ref))
    old['activity_inputs'] = dict(descriptions=record(docs), semantic_refs={'labels': label_ref},
                                  proofs={'path': 'MUST_NOT_OPEN_TRAINING_PROOFS'})
    models = {}
    for fold in (2, 3, 4):
        p = np.zeros((385, 384), dtype=np.float32); p[:-1] = np.eye(384)
        if fold == 2: p[:3, :3] = np.eye(3)[:, [2, 0, 1]]
        path = tmp_path / ('parameter'+str(fold)+'.npy'); np.save(path, p)
        models[str(fold)] = dict(heldout_fold=fold, training_folds=sorted({2, 3, 4}-{fold}),
            query_ids=['DISJOINT_TRAIN_CONTROL'], features=list(workbench.FEATURES),
            identifiers_as_features=False, raw_activity_without_site_proof=False,
            descriptions=old['activity_inputs']['descriptions'], weights=[1.]*30,
            official_catalog=record(cat), decoder_parameter=record(path), decoder={'family': 'ZERO_SHOT'})
    path = tmp_path / 'models.json'; write_json(path, models); old['models'] = record(path)
    return old, pop, models, SimpleNamespace(metadata={'activity_catalog': record(cat)})


def test_first_query_matches_original187_arithmetic_and_all_other_queries_own_vectors(tmp_path):
    old, pop, models, attrs = activity_fixture(tmp_path)
    assert workbench.activity(pop, old, models, attrs, 0) == original.activity(old, old['queries'][0], models, attrs)
    for i in (1, 2, 64, 127):
        row, v = pop.select(i)
        model = models[str(row['fold'])]
        p = np.load(model['decoder_parameter']['path'])
        docs = np.load(old['activity_inputs']['descriptions']['path'])
        expected = workbench.scores(workbench.query_design(v), workbench.normalized(docs), p, model['decoder']['family'])[0]
        assert workbench.activity(pop, old, models, attrs, i) == {j: float(x) for j, x in enumerate(expected)}
    assert workbench.activity(pop, old, models, attrs, 2) != original.activity(old, old['queries'][2], models, attrs)


@pytest.mark.parametrize('change', ['query_overlap', 'wrong_fold', 'id_feature', 'wrong_descriptions'])
def test_no_unsafe_head_or_training_overlap(tmp_path, change):
    old, pop, models, attrs = activity_fixture(tmp_path)
    m = models['2']
    if change == 'query_overlap': m['query_ids'].append(old['queries'][0]['query_id'])
    elif change == 'wrong_fold': m['heldout_fold'] = 3
    elif change == 'id_feature': m['identifiers_as_features'] = True
    else: m['descriptions']['sha256'] = 'f'*64
    write_json(old['models']['path'], models); old['models'] = record(old['models']['path'])
    with pytest.raises(ValueError):
        workbench.activity(pop, old, models, attrs, 0)
