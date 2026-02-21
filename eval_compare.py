import pandas as pd
import numpy as np
import xgboost as xgb

print("Loading test data from Decider samples...")
df = pd.read_parquet('data/samples_v7_decider.parquet')
test_df = df[df['split'] == 'test'].copy()

# 1. Evaluate Ranker (the baseline the Decider started from)
print("\n--- 1. Evaluating Ranker on Top-50 candidates ---")
# The _ranker_score is already inside this parquet file since generate_samples put it there!
def calc_hit1(sdf, score_col):
    sdf = sdf.sort_values(['query_id', score_col], ascending=[True, False])
    sdf['rank'] = sdf.groupby('query_id').cumcount() + 1
    hits = sdf[(sdf['label'] == 1) & (sdf['rank'] == 1)]
    return len(hits) / sdf['query_id'].nunique() * 100

# 1. Evaluate Ranker (the baseline the Decider started from)
print("\n--- 1. Evaluating Ranker on Top-50 candidates ---")
# The _ranker_score wasn't exported, so let's score these top 50 with the Ranker model again
# to see what hit@1 the Ranker gets when it is constrained to these exact 50 candidates
features_ranker = [c for c in list(test_df.columns) if c not in {'query_id', 'label', 'split', 'crm_name', 'crm_address', 'crm_city', 
                'gt_siret', 'loc_key', 'siret', 'insee', 'enseigne', 'denom_ul', 'nom_ul', 'numero', 'type_voie', 'rue', 'ville',
                'siren', 'etat_admin', 'etatAdministratifEtablissement', 'enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement',
                'semantic_enabled', 'decider_score'} and test_df[c].dtype != 'object' and 'semantic' not in c]

bst_ranker = xgb.Booster()
bst_ranker.load_model('models/xgbranker_fast_20260221_193026.json')
# Add back dummy semantic columns just to satisfy the DMatrix
for col in ['name_semantic_max', 'name_semantic_second', 'name_semantic_gap']:
    test_df[col] = 0.0

ranker_features_order = bst_ranker.feature_names
dtest_ranker = xgb.DMatrix(test_df[ranker_features_order])
test_df['_ranker_score'] = bst_ranker.predict(dtest_ranker)

ranker_hit1 = calc_hit1(test_df, '_ranker_score')
print(f"Ranker Hit@1 on these exact 50 candidates: {ranker_hit1:.2f}%")

# 2. Evaluate Decider 
print("\n--- 2. Evaluating Decider on Top-50 candidates ---")
all_cols = list(test_df.columns)
exclude_cols = {'query_id', 'label', 'split', 'crm_name', 'crm_address', 'crm_city', 
                'gt_siret', 'loc_key', 'siret', 'insee', 'enseigne', 'denom_ul', 'nom_ul', 'numero', 'type_voie', 'rue', 'ville',
                'siren', 'etat_admin', 'etatAdministratifEtablissement', 'enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement',
                '_ranker_score', 'semantic_enabled'}
features_cols = [c for c in all_cols if c not in exclude_cols and test_df[c].dtype != 'object']

bst = xgb.Booster()
bst.load_model('models/xgb_decider_20260221_213148.json')

import json
with open('models/xgb_two_stage_meta_20260221_213148.json') as f:
    meta = json.load(f)
feature_names = meta['feature_names']

# Reorder
dtest = xgb.DMatrix(test_df[feature_names])
test_df['decider_score'] = bst.predict(dtest)

decider_hit1 = calc_hit1(test_df, 'decider_score')
print(f"Decider Hit@1 on these exact 50 candidates: {decider_hit1:.2f}%")
