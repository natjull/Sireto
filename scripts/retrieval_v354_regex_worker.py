"""Versioned V344 worker with V353 exact-language native compilation repair.

The historical module/globals/files are not mutated. Its worker code, source
validation, emission, manifest, cleanup and budgets are reused unchanged, with
two explicit private bindings: the stricter launch validator and repaired
search. This module does not authorize resuming a terminal STOP root.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

from scripts import retrieval_v344_corroborated_population as original
from src.xgb_matcher.retrieval_regex_compiler_v353 import POLICY as COMPILER_POLICY, search
from src.xgb_matcher.retrieval_v20_workbench import ROOT, record


def implementation():
    refs = original.implementation()
    for path in (Path(__file__), ROOT/'src/xgb_matcher/retrieval_regex_compiler_v353.py',
                 ROOT/'docs/retrieval_v353_regex_repair_contract.md'):
        refs[str(path)] = record(path)
    return refs


def load(root):
    launch = original.load(root)
    if launch.get('regex_compiler_policy') != COMPILER_POLICY:
        raise ValueError('Explicit new frozen regex compiler authority required')
    expected = implementation()
    if any(launch['implementation'].get(key) != value for key, value in expected.items()):
        raise ValueError('Every original and repaired worker implementation must be frozen')
    return launch


def stage(root, batch):
    return original._stage(Path(root), batch, load(root))


_execute = FunctionType(original.worker.__code__,
    {**original.worker.__globals__, 'load': load, 'search': search},
    name='versioned_regex_worker', argdefs=original.worker.__defaults__)


def worker(root, batch):
    return _execute(Path(root), batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('worker',))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=int)
    args = parser.parse_args()
    worker(args.output, args.batch)
