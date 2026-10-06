"""Closed-target gates and frozen pure metrics only; no production labels."""
import pytest

from scripts import retrieval_v356_checkpointed as generation
from scripts import evaluate_retrieval_v358_postseal as evaluation
from scripts import evaluate_retrieval_v346_postseal as previous
from src.xgb_matcher import retrieval_evaluation_authority_v357 as authority


def test_all_targets_models_metrics_and_worker_resources_unchanged():
    assert authority.generation is generation and evaluation.authority is authority
    assert evaluation.TARGET_LEDGER == previous.TARGET_LEDGER
    assert evaluation.TARGET_EVALUATION == previous.TARGET_EVALUATION
    assert evaluation.metrics is previous.metrics and evaluation.base_metrics is previous.base_metrics
    assert evaluation.config()['resources'] == previous.config()['resources'] == generation.worker.LIMITS
    refs = generation.implementation()
    for suffix in ('retrieval_v344_corroborated_population.py', 'retrieval_checkpoints_v348.py',
                   'retrieval_evaluation_authority_v357.py', 'evaluate_retrieval_v358_postseal.py',
                   'launch_retrieval_v359_resumable.py'):
        assert any(key.endswith(suffix) for key in refs)


@pytest.mark.parametrize('stopped', [False, True])
def test_closed_targets_and_no_evaluation_root_on_partial_or_stop(tmp_path, monkeypatch, stopped):
    root = tmp_path/'generation'; root.mkdir()
    output = tmp_path/'evaluation'
    monkeypatch.setattr(generation, 'config', lambda: {'output_root': str(tmp_path)})
    monkeypatch.setattr(evaluation, 'config', lambda: {'output_root': str(tmp_path)})
    if stopped: (root/'status.json').write_text('{"status":"STOP"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run(root, output)
    assert not output.exists()
