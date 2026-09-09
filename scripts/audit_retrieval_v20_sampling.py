#!/usr/bin/env python3
"""Matched sampling ablation in inner splits; outer labels never evaluate a model."""
from pathlib import Path
import argparse
import gc
import json
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget, load_config, verify, record, write_json, negative_positions, inner_validation, implementation
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation


def run(config_path, output):
    config, sources = load_config(config_path)
    v19 = sources['v19']
    manifest_path = verify(config['feature_manifest'])
    fm = json.loads(manifest_path.read_text())
    features_path = verify({'path': str(manifest_path.parent/'parent_features.parquet'), 'sha256': fm['files']['parent_features.parquet']['sha256']})
    labels = verify(v19['authorities']['cohort_labels'])
    ledger = verify(v19['authorities']['label_ledger'])
    if fm['truth_columns_opened'] is not False or fm['config_sha256'] != config['source_configs']['v19']['sha256']:
        raise ValueError('Historical feature lineage')
    features = fm['feature_order']
    code = implementation(__file__)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, detail = [], []
    with Budget(config['resources']) as budget:
        con = duckdb.connect()
        con.execute("SET memory_limit='4GB'")
        con.execute('SET threads=4')
        truth = con.execute('''SELECT c.query_id,c.fold,substr(c.human_ground_truth_siret,1,9) gt_siren,c.cohort_kind,l.data_origin
            FROM read_parquet(?) c JOIN read_parquet(?) l USING(query_id,fold)''', [str(labels),str(ledger)]).df()
        validate_fold_isolation(truth)
        if len(truth) != 1034:
            raise ValueError('Population drift')
        all_rows = pq.read_table(features_path).to_pandas()
        if set(all_rows.query_id) != set(truth.query_id):
            raise ValueError('Feature query population')
        all_rows = all_rows.merge(truth, on=['query_id','fold'], validate='many_to_one')
        all_rows['is_positive'] = all_rows.siren.eq(all_rows.gt_siren).astype('int8')
        all_rows['inner_val'] = all_rows.gt_siren.map(inner_validation)
        all_rows = all_rows.sort_values(['query_id','rrf_score','channel_count','best_rank_reciprocal','siren'], ascending=[True,False,False,False,True], kind='stable').reset_index(drop=True)
        all_rows['rrf_rank'] = all_rows.groupby('query_id',sort=False).cumcount()+1
        for outer in (2,3,4):
            train = all_rows.loc[all_rows.fold.ne(outer) & ~all_rows.inner_val]
            val = all_rows.loc[all_rows.fold.ne(outer) & all_rows.inner_val].copy()
            assert set(train.gt_siren).isdisjoint(set(val.gt_siren))
            assert outer not in set(train.fold) | set(val.fold)
            split_record = {'outer_excluded': outer, 'train_query_ids': sorted(set(train.query_id)), 'validation_query_ids': sorted(set(val.query_id)), 'split_rule':config['audit']['inner_validation']}
            write_json(output/f'split_outer{outer}.json', split_record)
            Xval = val[features].to_numpy(dtype=np.float32)
            for method in config['audit']['negative_methods']:
                indices, sizes = [], []
                for _, group in train.groupby('query_id', sort=True):
                    positives = group.index[group.is_positive.eq(1)].to_numpy()
                    negatives = group.index[group.is_positive.eq(0)].to_numpy()
                    if len(positives) != 1 or len(negatives) == 0:
                        continue
                    ids = np.concatenate([positives, negatives[negative_positions(len(negatives), method)]])
                    indices.extend(ids)
                    sizes.append(len(ids))
                sampled = all_rows.loc[indices]
                params = dict(v19['xgboost'])
                model = xgb.XGBRanker(**params)
                model.fit(sampled[features].to_numpy(dtype=np.float32),sampled.is_positive.to_numpy(dtype=np.int8),group=np.array(sizes,dtype=np.uint32),verbose=False)
                model.save_model(output/f'outer{outer}_{method}.json')
                val['score'] = model.predict(Xval)
                ordered = val.sort_values(['query_id','score','rrf_score','siren'],ascending=[True,False,False,True],kind='stable').copy()
                ordered['pred_rank'] = ordered.groupby('query_id',sort=False).cumcount()+1
                gt = ordered.loc[ordered.is_positive.eq(1)].set_index('query_id')
                query_rows = truth.loc[truth.query_id.isin(set(val.query_id))].copy()
                query_rows['rank'] = query_rows.query_id.map(gt.pred_rank)
                query_rows['rrf_rank'] = query_rows.query_id.map(gt.rrf_rank)
                query_rows['outer_excluded'] = outer
                query_rows['method'] = method
                detail.append(query_rows.drop(columns='gt_siren'))
                r = dict(outer_excluded=outer,method=method,train_queries=len(sizes),train_rows=len(sampled),validation_queries=len(query_rows),missing_positive=int(query_rows['rank'].isna().sum()),tail_share_predicted_top100=float((ordered.loc[ordered.pred_rank.le(100),'rrf_rank']>512).mean()))
                for kind in ('BASE_TOP1750_MISS','BASE_TOP1750_CONTROL'):
                    q = query_rows.loc[query_rows.cohort_kind.eq(kind)]
                    r[kind] = {'denominator':len(q),'hits':{str(k):int(q['rank'].le(k).sum()) for k in (100,500,2000)}, 'rrf_hits':{str(k):int(q.rrf_rank.le(k).sum()) for k in (100,500,2000)}}
                records.append(r)
                print(json.dumps(r),flush=True)
                del model, sampled, ordered
                gc.collect()
            del train,val,Xval
        pq.write_table(pa.Table.from_pandas(pd.concat(detail,ignore_index=True),preserve_index=False),output/'inner_outcomes.parquet',compression='zstd')
        con.close()
    report = {'schema':'sireto-v20-sampling-audit-1','status':'DIAGNOSTIC_ONLY_NO_PRODUCTION_AUTHORIZATION','config':record(config_path),'feature_manifest':record(manifest_path),'records':records,'resources':budget.report(),'historical_id_sorted_rank_used_only_to_hold_features_constant':True,'new_recipe_feature_compliance':False,'outer_evaluation_performed':False,'fold0_opened':False,'fold1_opened':False,'positive_injection':False,'files':{p.name:record(p) for p in output.iterdir() if p.is_file()}}
    report['implementation'] = code
    write_json(output/'manifest.json',report)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.config,a.output)
