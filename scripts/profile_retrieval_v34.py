"""Bounded truth-free profile of the unchanged v33 legacy path."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats

import pyarrow.parquet as pq
from scripts import retrieval_v33_full_replay as v33
from src.xgb_matcher.retrieval_v20_workbench import Budget,verify,write_json,record


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=32);p.add_argument('--count',type=int,default=8);a=p.parse_args()
    if not 1<=a.count<=16 or a.start<0:raise ValueError('Bounded profile')
    m=json.loads((a.prepared/'manifest.json').read_text());rows=pq.read_table(verify(m['inputs'])).to_pylist()[a.start:a.start+a.count]
    queries=[(r['query_id'],r['oof_fold'],v33.normalize_input(r)) for r in rows]
    a.output.mkdir(parents=True,exist_ok=False);stats=[];original=v33.AllHumanV11Retriever.retrieve
    def measured(self,q):
        result=original(self,q);stats.append(dict(result.strategy_counts));print(json.dumps(stats[-1]),flush=True);return result
    v33.AllHumanV11Retriever.retrieve=measured
    with Budget(v33.RESOURCE) as b:
        profiler=cProfile.Profile();profiler.enable();v33.legacy(queries,a.output);profiler.disable()
    profiler.dump_stats(str(a.output/'profile.pstats'))
    with (a.output/'profile.txt').open('x') as out:pstats.Stats(profiler,stream=out).sort_stats('cumtime').print_stats(70)
    write_json(a.output/'manifest.json',dict(queries=len(rows),strategies=stats,resources=b.report(),prepared=record(a.prepared/'manifest.json'),truth_columns_opened=False))


if __name__=='__main__':main()
