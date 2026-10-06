"""Mechanical-version parity and binding checks, not national recall evidence."""
import inspect
import ast

from scripts import retrieval_v349_checkpointed as historical
from scripts import retrieval_v356_checkpointed as current
from scripts import launch_retrieval_v352_resumable as old_launcher
from scripts import launch_retrieval_v359_resumable as launcher
from scripts import evaluate_retrieval_v351_postseal as old_evaluation
from scripts import evaluate_retrieval_v358_postseal as evaluation
from src.xgb_matcher import retrieval_evaluation_authority_v350 as old_authority
from src.xgb_matcher import retrieval_evaluation_authority_v357 as authority


def rename(text):
    for a, b in ((349, 356), (350, 357), (351, 358), (352, 359)):
        text = text.replace('v'+str(a), 'v'+str(b)).replace('V'+str(a), 'V'+str(b))
    return text


def test_complete_frozen_retry_and_admission_algorithm_same_except_worker_entrypoint():
    for name in ('config', 'implementation', 'load', 'attempt_root', 'checkpoint_path',
                 'execute', 'sealed', 'run'):
        expected = rename(inspect.getsource(getattr(historical, name)))
        if name == 'execute':
            expected = expected.replace('scripts.retrieval_v344_corroborated_population',
                                        'scripts.retrieval_v354_regex_worker')
        assert inspect.getsource(getattr(current, name)) == expected


def test_postseal_authority_and_full_evaluation_unchanged_other_than_namespace_bindings():
    assert rename(inspect.getsource(old_authority.source_scope)) == inspect.getsource(authority.source_scope)
    expected = rename(inspect.getsource(old_authority.authorize)).replace(
        "root, manifest = header(root)  # ALL500 producer exits/resources, not a seal flag.\n"
        "    launch, _, outputs = generation.sealed(root)",
        "root, launch, manifest, outputs = _checked(root)")
    assert ast.unparse(ast.parse(expected)) == ast.unparse(ast.parse(inspect.getsource(authority.authorize)))
    expected = rename(inspect.getsource(old_authority.recheck)).replace(
        'return header(root)', "return root, json.loads(verify(permission['generation']).read_text())")
    assert ast.unparse(ast.parse(expected)) == ast.unparse(ast.parse(inspect.getsource(authority.recheck)))
    assert rename(inspect.getsource(old_evaluation)) == inspect.getsource(evaluation)
    assert evaluation.metrics is old_evaluation.metrics
    assert evaluation.base_metrics is old_evaluation.base_metrics
    assert evaluation.TARGET_LEDGER == old_evaluation.TARGET_LEDGER
    assert evaluation.TARGET_EVALUATION == old_evaluation.TARGET_EVALUATION


def test_entire_detached_launcher_same_transport_and_restart_rules():
    assert rename(inspect.getsource(old_launcher)) == inspect.getsource(launcher)


def test_private_facade_keeps_original_modules_untouched_and_freezes_repair():
    assert current.worker is not historical.worker
    assert current.worker.load is current.repaired.load
    assert current.worker.stage is current.repaired.stage
    assert current.worker.implementation is current.repaired.implementation
    assert historical.worker.load is not current.repaired.load
    cfg = current.config()
    assert {k: v for k, v in cfg.items() if k != 'output_root'} == {
        k: v for k, v in historical.config().items() if k != 'output_root'}
    refs = current.implementation()
    for name in ('retrieval_regex_compiler_v353.py', 'retrieval_v354_regex_worker.py',
                 'retrieval_v356_checkpointed.py', 'retrieval_evaluation_authority_v357.py',
                 'evaluate_retrieval_v358_postseal.py', 'launch_retrieval_v359_resumable.py'):
        assert any(k.endswith(name) for k in refs)
