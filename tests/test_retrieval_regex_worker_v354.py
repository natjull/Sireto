from pathlib import Path

import pytest

from scripts import retrieval_v354_regex_worker as repaired
from src.xgb_matcher.retrieval_regex_compiler_v353 import search


def test_private_worker_bindings_not_historical_monkeypatch():
    assert repaired._execute.__code__ is repaired.original.worker.__code__
    assert repaired._execute.__globals__ is not repaired.original.worker.__globals__
    assert repaired._execute.__globals__['search'] is search
    assert repaired._execute.__globals__['load'] is repaired.load
    assert repaired.original.worker.__globals__['search'] is repaired.original.search
    assert repaired.original.worker.__globals__['load'] is repaired.original.load
    before = repaired.original.worker.__globals__
    for name, value in before.items():
        if name not in ('search', 'load'):
            assert repaired._execute.__globals__[name] is value


def test_frozen_new_policy_and_every_implementation_ref_required(monkeypatch):
    refs = {'a': dict(path='a', sha256='x'), 'b': dict(path='b', sha256='y')}
    launch = dict(regex_compiler_policy=repaired.COMPILER_POLICY, implementation=dict(refs))
    monkeypatch.setattr(repaired.original, 'load', lambda root: launch)
    monkeypatch.setattr(repaired, 'implementation', lambda: refs)
    assert repaired.load(Path('/fixture')) is launch
    launch['regex_compiler_policy'] = 'historical'
    with pytest.raises(ValueError, match='compiler authority'):
        repaired.load(Path('/fixture'))
    launch['regex_compiler_policy'] = repaired.COMPILER_POLICY
    launch['implementation']['b'] = dict(path='b', sha256='drift')
    with pytest.raises(ValueError, match='implementation'):
        repaired.load(Path('/fixture'))


def test_new_stage_still_validates_original_full_exit_and_payload(monkeypatch):
    calls = []
    launch = object()
    monkeypatch.setattr(repaired, 'load', lambda root: launch)
    def stage(root, batch, value):
        calls.append((root, batch, value))
        raise ValueError('partial producer rejected')
    monkeypatch.setattr(repaired.original, '_stage', stage)
    with pytest.raises(ValueError, match='partial producer'):
        repaired.stage(Path('/fixture'), 97)
    assert calls == [(Path('/fixture'), 97, launch)]
