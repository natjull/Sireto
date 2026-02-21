import pandas as pd
import numpy as np
import xgboost as xgb

print("Loading test data...")
df = pd.read_parquet('data/samples_v7_ranker.parquet')
test_df = df[df['split'] == 'test'].copy()

# Features logic mirrored from train_xgb_ranker.py
all_cols = list(test_df.columns)
exclude_cols = {'query_id', 'label', 'split', 'crm_name', 'crm_address', 'crm_city', 
                'gt_siret', 'loc_key', 'siret', 'insee', 'enseigne', 'denom_ul', 'nom_ul', 'numero', 'type_voie', 'rue', 'ville',
                'siren', 'etat_admin', 'etatAdministratifEtablissement', 'enseigne1Etablissement', 'enseigne2Etablissement', 'enseigne3Etablissement'}
features_cols = [c for c in all_cols if c not in exclude_cols and test_df[c].dtype != 'object']

# For Ranker FAST, we exclude semantic features
features_fast = [c for c in features_cols if 'semantic' not in c]
# But the booster expects the columns to be present in the DMatrix!
for col in ['name_semantic_max', 'name_semantic_second', 'name_semantic_gap']:
    test_df[col] = 0.0

# Load model
print("Loading model...")
bst_fast = xgb.Booster()
bst_fast.load_model('models/xgbranker_fast_20260221_193026.json')

# Reorder features exactly as expected by the booster
expected_features = bst_fast.feature_names
features_fast = expected_features

# Score candidates
print("Scoring candidates...")
dtest = xgb.DMatrix(test_df[features_fast])
test_df['score'] = bst_fast.predict(dtest)

# Calculate metrics
print("Calculating metrics...")
def calculate_hits(df, k_list=[1, 5, 10, 50]):
    # Sort by query_id and descending score
    df = df.sort_values(['query_id', 'score'], ascending=[True, False])
    
    # Add rank within each query
    df['rank'] = df.groupby('query_id').cumcount() + 1
    
    total_queries = df['query_id'].nunique()
    
    for k in k_list:
        hits = df[(df['label'] == 1) & (df['rank'] <= k)]
        hit_rate = len(hits) / total_queries * 100
        print(f"Hit@{k}: {hit_rate:.2f}% ({len(hits)} / {total_queries})")

print("\n--- Evaluate XGB Ranker FAST on Test Set ---")
calculate_hits(test_df)
