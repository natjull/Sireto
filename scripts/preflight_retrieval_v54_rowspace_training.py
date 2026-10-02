"""Same nested recipe/resource ceilings, compact full-objective solver."""
import argparse
from pathlib import Path

from scripts.preflight_retrieval_v53_nested_training import run
from src.xgb_matcher.retrieval_rowspace_v54 import solve
from src.xgb_matcher.retrieval_v20_workbench import ROOT


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True,type=Path)
    run(p.parse_args().output,ROOT/'config/retrieval_v54_rowspace_training.json',solve,
        (Path(__file__),ROOT/'src/xgb_matcher/retrieval_rowspace_v54.py'))
