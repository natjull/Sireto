import inspect
import json
import numpy as np
import pytest
from scripts.profile_retrieval_v106_native_cost import CONFIG,variants,worker,run


def test_ablation_changes_only_diagnostic_components_and_copies_frozen_weights():
    original=np.arange(16)*.1; saved=original.copy(); choices=variants(original)
    assert np.array_equal(original,saved)
    assert np.array_equal(choices['LEXICAL11_PLUS_WORDS'][:11],original[:11])
    assert choices['LEXICAL11_PLUS_WORDS'][15]==original[15]
    assert not choices['LEXICAL11_PLUS_WORDS'][11:15].any()
    assert np.flatnonzero(choices['CURRENT_INITIALISM_ONLY']).tolist()==[11]
    assert np.flatnonzero(choices['CURRENT_INITIALISM_AT_PLACE']).tolist()==[12]


def test_cost_guards_fixed_and_no_quality_evaluator_called():
    cfg=json.loads(CONFIG.read_text())
    assert cfg['queries']==2 and cfg['native_limit']==100
    assert cfg['resources']==dict(max_seconds=80,max_rss_bytes=4294967296,max_swapout_pages_delta=0)
    body=inspect.getsource(worker)+inspect.getsource(run)
    assert 'evaluate(' not in body and 'positive_siret' not in body
    assert "['sidecar']" not in body and "['ledger']" not in body
    with pytest.raises(ValueError,match='weight drift'): variants(np.ones(11))
    with pytest.raises(ValueError,match='weight drift'): variants(np.full(16,-1.))
