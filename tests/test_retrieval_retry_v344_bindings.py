"""Binding/closed-label fixtures only, not national recall or resource proof."""
import inspect
import pytest

from scripts import retrieval_v344_corroborated_population as generation
from scripts import evaluate_retrieval_v346_postseal as evaluation
from src.xgb_matcher import retrieval_evaluation_authority_v345 as authority
from src.xgb_matcher.retrieval_memory_policy_v342 import LIMITS


def test_new_authority_and_evaluator_not_bound_to_stopped_driver():
    assert authority.generation is generation
    assert evaluation.authority is authority
    assert evaluation.config()['resources'] == LIMITS
    assert generation.config()['resources'] == LIMITS
    assert generation.config()['panel_root'].endswith('retrieval_v328_serving_panel/package_20261005_a')
    refs = generation.implementation()
    for suffix in ('retrieval_identifier_surface_v329.py', 'evaluate_retrieval_v346_postseal.py',
                   'retrieval_v346_postseal_evaluation.json', 'retrieval_evaluation_authority_v345.py'):
        assert any(key.endswith(suffix) for key in refs)
    assert 'q = feature_query(raw)' in inspect.getsource(generation.worker)


@pytest.mark.parametrize('stopped', [False, True])
def test_closed_header_before_evaluation_output_or_targets(tmp_path, monkeypatch, stopped):
    root = tmp_path/'generation'; root.mkdir()
    output = tmp_path/'evaluation'
    monkeypatch.setattr(generation, 'config', lambda: {'output_root': str(tmp_path)})
    monkeypatch.setattr(evaluation, 'config', lambda: {'output_root': str(tmp_path)})
    if stopped:
        (root/'status.json').write_text('{"status":"STOP"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run(root, output)
    assert not output.exists()


def test_all_target_specs_and_pure_metrics_unchanged():
    from scripts import evaluate_retrieval_v326_postseal as previous
    assert evaluation.TARGET_LEDGER == previous.TARGET_LEDGER
    assert evaluation.TARGET_EVALUATION == previous.TARGET_EVALUATION
    assert evaluation.metrics is previous.metrics
    assert evaluation.base_metrics is previous.base_metrics
