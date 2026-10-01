"""Bounded nested-OOF SIRET ranking pilot; preserves v33's frozen inventory."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES as PARENT_FEATURES, build_features, make_master

DATA = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
GEN = DATA/'audits/retrieval_v34_fast_resume/full_job/generation/manifest.json'
GEN_SHA = '022b5c4e4436b335e78d217a81319c122390631e5698dd17c9d7db3f36c77e26'
PREP = DATA/'audits/retrieval_v34_fast_resume/prepared/manifest.json'
PREP_SHA = '16dbf6bcfbcd4bff503c1ad90d75b54ece4ac6bb671386fe9115b4ec2c677c45'
MODEL = DATA/'audits/retrieval_v23_oof_master/bb12058773f80d6f/manifest.json'
MODEL_SHA = 'ac4b279be0f30fbee5cfe7fc0c205c31b9cf50fcc23351795ceb55423e05de81'
EVAL = DATA/'audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/manifest.json'
EVAL_SHA = 'b6d7123d2fb33616c76dc14ed4277552a7696d2bdb17b56f540b7f76edfec3c9'
LEDGER = DATA/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
LEDGER_SHA = '24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'
RESOURCE = dict(max_seconds=1200, max_rss_bytes=6*1024**3, max_swapout_pages_delta=0)
PARAMS = dict(objective='rank:ndcg', lambdarank_pair_method='mean',
              lambdarank_num_pair_per_sample=8, n_estimators=200, max_depth=5,
              learning_rate=.06, reg_lambda=5, tree_method='hist', max_bin=128,
              subsample=1, colsample_bytree=1, random_state=20261001, n_jobs=2)
FEATURES = (
    'parent_name_jw', 'parent_name_tokens', 'site_name_jw', 'site_name_tokens',
    'has_parent_name', 'has_site_name', 'street', 'street_tokens',
    'number_exact', 'number_pair_available', 'number_gap_log', 'suffix_ok',
    'suffix_missing', 'same_insee', 'same_postcode', 'local_site',
    'crm_name_missing', 'crm_number_missing', 'crm_street_missing',
    'official_site', 'site_active', 'site_headquarters',
    'direct_rrf', 'parent_rrf', 'base_rank_rrf', 'parent_rank_rrf',
    'joint_site_rank_rrf', 'master_rank_rrf', 'joint_jw', 'joint_tokens',
    'soft_name_street', 'soft_tokens_street', 'parent_site_name_gap',
    'log_parent_sites', 'parent_best_name', 'site_name_gap_to_best',
    'direct_channels', 'parent_channels', 'from_text', 'from_address', 'from_geo',
    'ban_resolved', 'ban_street_similarity', 'ban_certification_commune',
    'ban_declared_weak', 'ban_uncertainty_log', 'geo_available',
    'geo_distance_log', 'geo_lower_log', 'geo_upper_log', 'geo_uncertainty_ratio',
)


def load_manifest(path, sha):
    return json.loads(verify(dict(path=path, sha256=sha)).read_text())


def train_context(outer, query_fold):
    """One single-fold upstream model excludes BOTH target and training query."""
    if outer not in (2, 3, 4) or query_fold not in (2, 3, 4) or outer == query_fold:
        raise ValueError('Invalid nested training context')
    return next(f for f in (2, 3, 4) if f not in (outer, query_fold))


def negative_indices(n):
    if n <= 128:
        return np.arange(n, dtype=np.int64)
    tail = 64 + np.floor((np.arange(64)+.5)*(n-64)/64).astype(np.int64)
    return np.concatenate([np.arange(64), tail])


def sample_training(frame, labels, neutral):
    """Only naturally present exact positives. No truth retrieval or insertion."""
    keep = []
    stats = dict(queries=0, trainable=0, absent_positive=0, neutralized=0)
    for qid, g in frame.groupby('query_id', sort=False):
        stats['queries'] += 1
        truth = labels[qid]
        pos = g.index[g.siret == truth].to_numpy()
        if len(pos) != 1:
            if len(pos) > 1:
                raise ValueError('Duplicate naturally present truth')
            stats['absent_positive'] += 1
            continue
        equiv = neutral.get(qid, set())
        neg = g.index[(g.siret != truth) & ~g.siret.isin(equiv)].to_numpy()
        stats['neutralized'] += int(((g.siret != truth) & g.siret.isin(equiv)).sum())
        keep.extend(pos)
        keep.extend(neg[negative_indices(len(neg))])
        stats['trainable'] += 1
    out = frame.loc[keep].sort_values(['query_id', 'candidate_rank', 'siret']).copy()
    out['target'] = [int(s == labels[q]) for q, s in zip(out.query_id, out.siret)]
    stats['rows'] = len(out)
    return out, stats


def check_model_split(split, kind, fold, scope):
    expected = [fold] if kind == 'inner' else [f for f in (2, 3, 4) if f != fold]
    targets = [f for f in (2, 3, 4) if f != fold] if kind == 'inner' else [fold]
    if split['train_folds'] != expected or split['target_folds'] != targets:
        raise ValueError('Parent split authority drift')
    trained = set(split['trained_query_ids'])
    if any(q in trained for q, f in scope if f in targets):
        raise ValueError('Upstream target identity leaked')


def db(output):
    spill = output/'_duckdb_spill'
    spill.mkdir(exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='3GB'")
    c.execute('SET threads=2')
    c.execute(f'SET temp_directory={sql_path(spill)}')
    c.execute("SET max_temp_directory_size='8GiB'")
    return c


def discover():
    generation = load_manifest(GEN, GEN_SHA)
    if generation['status'] != 'SEALED_FULL_PAIRED_GENERATION':
        raise ValueError('Generation incomplete')
    prepared = load_manifest(PREP, PREP_SHA)
    for authority in prepared['code'].values():
        verify(authority)
    models = load_manifest(MODEL, MODEL_SHA)
    shards = []
    for i, ref in enumerate(generation['batches'][:32]):
        b = json.loads(verify(ref).read_text())
        if b['start'] != i*32 or b['count'] != 32 or len(b['shards']) != 1:
            raise ValueError('Predeclared pilot scope drift')
        s = json.loads(verify(b['shards'][0]).read_text())
        for name in ('queries.parquet', 'raw.parquet', 'direct.parquet', 'structured.parquet',
                     'joint.parquet', 'ban.parquet', 'geo_sites.parquet', 'parent_presence.parquet',
                     'v30_stable/pool.parquet', 'v30_stable/master.parquet'):
            verify(s['files'][name])
        shards.append(s)
    scope = [(r['query_id'], r['fold']) for s in shards
             for r in pq.read_table(s['files']['queries.parquet']['path'], columns=['query_id', 'fold']).to_pylist()]
    if len(scope) != 1024 or len(set(q for q, _ in scope)) != 1024 or any(f not in (2, 3, 4) for _, f in scope):
        raise ValueError('Pilot scope invalid')
    model_paths = {}
    for f in (2, 3, 4):
        for kind, stem in [('outer', f'outer{f}'), ('inner', f'inner_train{f}')]:
            split = json.loads(verify(models['files'][stem+'_split.json']).read_text())
            check_model_split(split, kind, f, scope)
            model_paths[stem] = verify(models['files'][stem+'.json'])
    return prepared, models, shards, scope, model_paths


def cache(c, shards, prepared, output):
    """One projected scan per official source; keys are truth-blind reservoir IDs."""
    paths = [s['files']['v30_stable/pool.parquet']['path'] for s in shards]
    c.execute('CREATE TEMP TABLE keys AS SELECT DISTINCT siren,siret FROM read_parquet(?)', [paths])
    for key in ('sites', 'names'):
        verify(prepared['sources'][key])
    c.execute('''CREATE TEMP TABLE cached_sites AS SELECT s.* FROM read_parquet(?) s
      SEMI JOIN (SELECT DISTINCT siret FROM keys) p USING(siret)''', [prepared['sources']['sites']['path']])
    c.execute('''CREATE TEMP TABLE cached_names AS SELECT DISTINCT n.siren,n.siret,n.subject_kind,n.name_role,
      trim(regexp_replace(upper(strip_accents(n.normalized_value)),'[^A-Z0-9]+',' ','g')) AS name
      FROM read_parquet(?) n WHERE n.selected_for_subject AND n.normalized_value<>'' AND
      ((n.subject_kind='SIREN' AND n.name_role IN ('LEGAL_CURRENT','TRADE_CURRENT')
          AND n.siren IN (SELECT siren FROM keys)) OR
       (n.subject_kind='SIRET' AND n.name_role='SITE_CURRENT' AND n.siret IN (SELECT siret FROM keys)))''',
      [prepared['sources']['names']['path']])
    for name in ('sites', 'names'):
        c.execute(f'COPY cached_{name} TO {sql_path(output/(name+".parquet"))} (FORMAT PARQUET,COMPRESSION ZSTD)')
    return {name: record(output/(name+'.parquet')) for name in ('sites', 'names')}


def evidence(c, s, caches):
    paths = {k: v['path'] for k, v in s['files'].items()}
    c.execute('CREATE TEMP TABLE q AS SELECT * FROM read_parquet(?)', [paths['queries.parquet']])
    c.execute('CREATE TEMP TABLE raw AS SELECT *,union_rank base_rank FROM read_parquet(?)', [paths['raw.parquet']])
    c.execute('''CREATE TEMP TABLE signals AS SELECT query_id,fold,siren,source_siret,channel,channel_rank
      FROM read_parquet(?) UNION ALL
      SELECT query_id,fold,siren,siret,'siret_joint_name_address_char',evidence_rank FROM read_parquet(?)''',
      [[paths['structured.parquet'], paths['direct.parquet']], paths['joint.parquet']])
    c.execute('''CREATE TEMP TABLE parents AS
      WITH support AS (SELECT query_id,fold,siren,sum(1.0/(60+r) ORDER BY channel) parent_rrf
        FROM (SELECT query_id,fold,siren,channel,min(channel_rank) r FROM signals WHERE channel<>'v17_union'
              GROUP BY ALL) GROUP BY query_id,fold,siren),
      membership AS (SELECT query_id,fold,siren FROM signals UNION SELECT query_id,fold,siren FROM raw)
      SELECT m.*,coalesce(s.parent_rrf,0) parent_rrf FROM membership m LEFT JOIN support s USING(query_id,fold,siren)''')
    c.execute('''CREATE TEMP TABLE witnesses AS
      SELECT query_id,fold,siren,siret,'legacy_site_pool' channel,base_rank channel_rank FROM raw UNION
      SELECT query_id,fold,siren,source_siret,channel,channel_rank FROM signals
      WHERE length(source_siret)=14 AND channel NOT IN ('legacy_site_pool','v17_union')''')
    c.execute('''CREATE TEMP TABLE direct_support AS SELECT query_id,fold,siren,siret,
      sum(1.0/(60+r) ORDER BY channel) direct_rrf,count(*) direct_channels
      FROM (SELECT query_id,fold,siren,siret,channel,min(channel_rank) r FROM witnesses GROUP BY ALL)
      GROUP BY query_id,fold,siren,siret''')
    c.execute('''CREATE TEMP TABLE pool AS SELECT u.*,r.base_rank,coalesce(p.parent_rrf,0) parent_rrf,
      coalesce(d.direct_rrf,0) direct_rrf FROM read_parquet(?) u LEFT JOIN raw r USING(query_id,fold,siren,siret)
      LEFT JOIN parents p USING(query_id,fold,siren) LEFT JOIN direct_support d USING(query_id,fold,siren,siret)''',
      [paths['v30_stable/pool.parquet']])
    c.execute('CREATE TEMP TABLE sites AS SELECT s.* FROM read_parquet(?) s SEMI JOIN pool p USING(siret)', [caches['sites']['path']])
    c.execute('''CREATE TEMP TABLE names AS SELECT n.* FROM read_parquet(?) n WHERE
      (n.subject_kind='SIREN' AND n.siren IN (SELECT siren FROM pool)) OR
      (n.subject_kind='SIRET' AND n.siret IN (SELECT siret FROM pool))''', [caches['names']['path']])
    build_features(c)
    c.execute('CREATE TEMP TABLE ban AS SELECT * FROM read_parquet(?)', [paths['ban.parquet']])
    c.execute('CREATE TEMP TABLE geo AS SELECT * FROM read_parquet(?)', [paths['geo_sites.parquet']])
    c.execute('CREATE TEMP TABLE presence AS SELECT * FROM read_parquet(?)', [paths['parent_presence.parquet']])
    c.execute('''CREATE TEMP TABLE parent_channels AS SELECT query_id,fold,siren,count(DISTINCT channel) parent_channels
      FROM signals WHERE channel<>'v17_union' GROUP BY query_id,fold,siren''')
    c.execute('''CREATE TEMP TABLE site_light AS SELECT e.*,tokens(q.crm_street,s.street_normalized) street_tokens,
      coalesce(try_cast(q.crm_number AS DOUBLE) IS NOT NULL AND try_cast(s.street_number AS DOUBLE) IS NOT NULL,false) number_pair_available,
      coalesce(ln(1+abs(try_cast(q.crm_number AS DOUBLE)-try_cast(s.street_number AS DOUBLE))),0) number_gap_log,
      q.crm_suffix='' suffix_missing,q.crm_street='' crm_street_missing,
      coalesce(s.administrative_state='A',false) site_active,coalesce(s.is_headquarters,false) site_headquarters,
      greatest(e.parent_name_jw,e.site_name_jw)*e.street soft_name_street,
      greatest(e.parent_name_tokens,e.site_name_tokens)*e.street soft_tokens_street,
      e.parent_name_jw-e.site_name_jw parent_site_name_gap,
      ln(1+count(*) OVER(PARTITION BY e.query_id,e.siren)) log_parent_sites,
      max(greatest(e.site_name_jw,e.parent_name_jw)) OVER(PARTITION BY e.query_id,e.siren) parent_best_name,
      max(e.site_name_jw) OVER(PARTITION BY e.query_id,e.siren)-e.site_name_jw site_name_gap_to_best,
      coalesce(d.direct_channels,0) direct_channels,coalesce(pc.parent_channels,0) parent_channels,
      coalesce(p.from_text,false) from_text,coalesce(p.from_address,false) from_address,coalesce(p.from_geo,false) from_geo,
      coalesce(b.ban_resolved,false) ban_resolved,coalesce(b.ban_street_similarity,0) ban_street_similarity,
      coalesce(b.ban_certification_commune,false) ban_certification_commune,coalesce(b.ban_declared_weak,false) ban_declared_weak,
      coalesce(ln(1+greatest(b.ban_uncertainty_m,0)),0) ban_uncertainty_log,
      g.siret IS NOT NULL geo_available,coalesce(ln(1+greatest(g.geo_distance_m,0)),0) geo_distance_log,
      coalesce(ln(1+greatest(g.geo_distance_lower_m,0)),0) geo_lower_log,
      coalesce(ln(1+greatest(g.geo_distance_upper_m,0)),0) geo_upper_log,
      coalesce(g.geo_distance_uncertainty_ratio,0) geo_uncertainty_ratio
      FROM site_evidence e JOIN q USING(query_id,fold) LEFT JOIN sites s USING(siren,siret)
      LEFT JOIN direct_support d USING(query_id,fold,siren,siret) LEFT JOIN parent_channels pc USING(query_id,fold,siren)
      LEFT JOIN presence p USING(query_id,fold,siren) LEFT JOIN ban b USING(query_id,fold)
      LEFT JOIN geo g USING(query_id,fold,siren,siret)''')
    if c.execute('SELECT count(*) FROM site_light').fetchone() != c.execute('SELECT count(*) FROM pool').fetchone():
        raise ValueError('Evidence join multiplied candidates')


def master_context(c, paths, outer=False, inner=None):
    import pandas as pd
    frames = []
    for fold in (2, 3, 4):
        if inner == fold:
            continue
        f = c.execute('SELECT * FROM parent_features WHERE fold=? ORDER BY query_id,siren', [fold]).df()
        model = xgb.XGBRanker(n_jobs=2)
        model.load_model(paths[f'outer{fold}' if outer else f'inner_train{inner}'])
        model.set_params(n_jobs=2)
        f['parent_score'] = model.predict(f[list(PARENT_FEATURES)].to_numpy(dtype=np.float32))
        frames.append(f[['query_id', 'fold', 'siren', 'parent_score']])
    c.register('parent_scores', pa.Table.from_pandas(pd.concat(frames), preserve_index=False))
    make_master(c)


def context_features(c):
    # Old rank is an input, never an eligibility rule below the Master ceiling.
    expressions = []
    for name in FEATURES:
        if name == 'base_rank_rrf':
            exp = 'coalesce(1.0/(60+m.base_rank),0)'
        elif name == 'parent_rank_rrf':
            exp = '1.0/(60+m.learned_parent_rank)'
        elif name == 'joint_site_rank_rrf':
            exp = 'coalesce(1.0/(60+m.joint_site_rank),0)'
        elif name == 'master_rank_rrf':
            exp = '1.0/(60+m.candidate_rank)'
        else:
            exp = f'e.{name}'
        expressions.append(f'CAST({exp} AS FLOAT) {name}')
    return c.execute('SELECT m.query_id,m.fold,m.siren,m.siret,m.candidate_rank,'+','.join(expressions)+
                     ' FROM master m JOIN site_light e USING(query_id,fold,siren,siret) ORDER BY m.query_id,m.candidate_rank').fetch_arrow_table()


def build_shard(s, caches, models, output):
    output.mkdir()
    c = db(output)
    try:
        evidence(c, s, caches)
        master_context(c, models, outer=True)
        baseline = s['files']['v30_stable/master.parquet']['path']
        columns = 'query_id,fold,siren,siret,candidate_rank,parent_score,direct_rrf,parent_rrf,joint_jw,joint_tokens,joint_site_rank,learned_parent_rank,rescue_score'
        mismatch = c.execute(f'''SELECT count(*) FROM ((SELECT {columns} FROM master EXCEPT SELECT {columns} FROM read_parquet(?))
          UNION ALL (SELECT {columns} FROM read_parquet(?) EXCEPT SELECT {columns} FROM master))''', [baseline, baseline]).fetchone()[0]
        if mismatch:
            raise ValueError(f'Outer Master parity failed: {mismatch}')
        pq.write_table(context_features(c), output/'outer.parquet', compression='zstd')
        for inner in (2, 3, 4):
            c.execute('DROP TABLE master'); c.execute('DROP TABLE ranked_parents'); c.unregister('parent_scores')
            master_context(c, models, inner=inner)
            pq.write_table(context_features(c), output/f'inner{inner}.parquet', compression='zstd')
        result = dict(status='SEALED_TRUTH_BLIND_FEATURES', files={p.name: record(p) for p in output.glob('*.parquet')},
                      outer_master_parity=True, feature_names=list(FEATURES), source=s['files']['queries.parquet'])
        write_json(output/'manifest.json', result)
        return result
    finally:
        c.close(); gc.collect()


def wilson(hits, n):
    if not n:
        return None
    z = 1.959963984540054
    p = hits/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


def summarize(rows):
    out = {'queries': len(rows)}
    for k in (100, 500, 2000):
        h = sum(r['exact_rank'] is not None and r['exact_rank'] <= k for r in rows)
        op = sum(r['operational_rank'] is not None and r['operational_rank'] <= k for r in rows)
        old = sum(r['baseline_exact_rank'] is not None and r['baseline_exact_rank'] <= k for r in rows)
        gains = sum(r['exact_rank'] is not None and r['exact_rank'] <= k and not
                    (r['baseline_exact_rank'] is not None and r['baseline_exact_rank'] <= k) for r in rows)
        losses = sum(r['baseline_exact_rank'] is not None and r['baseline_exact_rank'] <= k and not
                     (r['exact_rank'] is not None and r['exact_rank'] <= k) for r in rows)
        out[str(k)] = dict(exact_hits=h, exact_recall=h/len(rows), operational_hits=op,
                          operational_recall=op/len(rows), wilson95_lower=wilson(h,len(rows)),
                          baseline_exact_hits=old, gains=gains, losses=losses)
    return out


def fit_evaluate(shards, scope, output):
    # Truth opened only AFTER every feature parquet and recipe is sealed.
    prereg = json.loads((output/'preregistration.json').read_text())
    verify(prereg['code']); verify(prereg['contract'])
    evaluation = load_manifest(EVAL, EVAL_SHA)
    verify(dict(path=LEDGER, sha256=LEDGER_SHA))
    evalroot = EVAL.parent
    # Full evaluator already sealed official label-side evidence. Verify it again.
    refs = evaluation.get('files', {})
    outcomes_path = evalroot/'v30_stable_outcomes.parquet'
    operational_path = evalroot/'acceptable_operational.parquet'
    for name in ('v30_stable_outcomes.parquet', 'acceptable_operational.parquet', 'official_catalog_sidecar.parquet'):
        ref = refs.get(name)
        if ref is None:
            raise ValueError('Missing sealed evaluator file authority: '+name)
        verify(ref)
    ids = {q for q, _ in scope}
    baseline = pq.read_table(outcomes_path, filters=[('fold', 'in', [2, 3, 4])]).to_pylist()
    baseline = {r['query_id']: r for r in baseline if r['query_id'] in ids}
    ledger = pq.read_table(LEDGER, filters=[('fold', 'in', [2, 3, 4])],
                           columns=['query_id','fold','human_ground_truth_siret']).to_pylist()
    labels = {r['query_id']: r['human_ground_truth_siret'] for r in ledger if r['query_id'] in ids}
    if set(labels) != ids or set(baseline) != ids:
        raise ValueError('Human denominator drift')
    acceptable = {}
    neutral = {}
    for r in pq.read_table(operational_path, filters=[('fold','in',[2,3,4])]).to_pylist():
        if r['query_id'] in ids:
            acceptable.setdefault(r['query_id'], set()).add(r['siret'])
            if not r['exact_truth']:
                neutral.setdefault(r['query_id'], set()).add(r['siret'])
    reports, scored = [], []
    for outer in (2, 3, 4):
        t = time.monotonic()
        sampled = []
        stats = []
        for shard in shards:
            for qfold in (f for f in (2, 3, 4) if f != outer):
                inner = train_context(outer, qfold)
                frame = pq.read_table(verify(shard['files'][f'inner{inner}.parquet']), filters=[('fold','=',qfold)]).to_pandas()
                sample, st = sample_training(frame, labels, neutral)
                sampled.append(sample); stats.append(st)
        import pandas as pd
        train = pd.concat(sampled, ignore_index=True).sort_values(['query_id','candidate_rank','siret'])
        if set(train.loc[train.fold == outer, 'query_id']):
            raise ValueError('Outer labels in training')
        values = train[list(FEATURES)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite training features')
        group = train.groupby('query_id', sort=False).size().to_numpy()
        model = xgb.XGBRanker(**PARAMS)
        model.fit(values, train.target.to_numpy(), group=group, verbose=False)
        model.save_model(output/f'site_outer{outer}.json')
        trainids = sorted(set(train.query_id))
        del train, values, sampled; gc.collect()
        for shard in shards:
            frame = pq.read_table(verify(shard['files']['outer.parquet']), filters=[('fold','=',outer)]).to_pandas()
            frame['score'] = model.predict(frame[list(FEATURES)].to_numpy(dtype=np.float32))
            if not np.isfinite(frame.score).all():
                raise ValueError('Nonfinite evaluation score')
            frame = frame.sort_values(['query_id','score','siret'], ascending=[True,False,True])
            frame['site_rank'] = frame.groupby('query_id').cumcount()+1
            # Keep only references/scores, not rich hydration; caps checked directly.
            pq.write_table(pa.Table.from_pandas(frame[['query_id','fold','siret','score','site_rank']], preserve_index=False),
                           output/f'scores_outer{outer}_{len(scored):05d}.parquet', compression='zstd')
            for qid, g in frame.groupby('query_id', sort=False):
                b = baseline[qid]
                exact = g.loc[g.siret == labels[qid], 'site_rank'].tolist()
                op = g.loc[g.siret.isin(acceptable[qid]), 'site_rank'].tolist()
                rank = exact[0] if exact else None
                if bool(rank) != bool(b['exact_rank']):
                    raise ValueError('Master coverage changed during ranking')
                loss = ('HIT' if rank is not None and rank <= 100 else
                        'EVICTED_500_TO_100' if rank is not None and rank <= 500 else
                        'EVICTED_MASTER_TO_500' if rank is not None else b['loss_stage'])
                scored.append(dict(query_id=qid,fold=outer,exact_rank=rank,operational_rank=min(op) if op else None,
                                   baseline_exact_rank=b['exact_rank'],loss_stage=loss,label_kind=b['label_kind'],
                                   qualification_v2=b['qualification_v2'],qualification_v3=b['qualification_v3'],
                                   master_count=len(g),pre500_count=min(len(g),500),top100_count=min(len(g),100)))
        reports.append(dict(outer_fold=outer,train_folds=[f for f in (2,3,4) if f!=outer],
                            trained_query_ids=trainids,sampling={k:sum(st[k] for st in stats) for k in stats[0]},
                            wall_seconds=time.monotonic()-t,model=record(output/f'site_outer{outer}.json')))
        print(json.dumps(dict(stage='outer_fit_evaluated',fold=outer,seconds=reports[-1]['wall_seconds'])),flush=True)
        del model; gc.collect()
    if {r['query_id'] for r in scored} != ids or len(scored) != 1024:
        raise ValueError('Incomplete evaluation denominator')
    pq.write_table(pa.Table.from_pylist(scored), output/'outcomes.parquet', compression='zstd')
    segments = {}
    for col in ('fold','label_kind','qualification_v2','qualification_v3'):
        segments[col] = {str(v): summarize([r for r in scored if r[col] == v]) for v in sorted({r[col] for r in scored}, key=str)}
    losses = {stage:sum(r['loss_stage']==stage for r in scored) for stage in sorted({r['loss_stage'] for r in scored})}
    return dict(status='MEASURED_BOUNDED_NESTED_OOF_SITE_FUSION',decision='PIVOT_GENERATION',
                quality=summarize(scored),segments=segments,losses=losses,fold_models=reports,
                fold0_opened=False,fold1_opened=False,positive_injection=False,
                full_population_measured=False,production_latency_measured=False,ready_for_fold1=False,
                full_source_excluded_missing_human_labels=evaluation['excluded_missing_human_labels'],
                sealed_label_side=evaluation['label_side'],evaluation_authority=dict(path=str(EVAL),sha256=EVAL_SHA))


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        with Budget(RESOURCE) as budget:
            prepared, models, source_shards, scope, modelpaths = discover()
            write_json(output/'preregistration.json', dict(schema='sireto-v35-site-fusion-pilot-1',
                params=PARAMS,features=list(FEATURES),resources=RESOURCE,
                code=record(Path(__file__)),contract=record(ROOT/'docs/retrieval_v35_site_fusion_contract.md'),
                generation=dict(path=str(GEN),sha256=GEN_SHA),prepared=dict(path=str(PREP),sha256=PREP_SHA),
                parent_models=dict(path=str(MODEL),sha256=MODEL_SHA),scope=scope,
                parent_split_authorities={k:v for k,v in models['files'].items() if 'split' in k},
                runtime_versions=dict(xgboost=xgb.__version__,duckdb=duckdb.__version__),truth_opened=False))
            t = time.monotonic(); c = db(output)
            caches = cache(c, source_shards, prepared, output);c.close();gc.collect()
            print(json.dumps(dict(stage='truth_blind_cache',seconds=time.monotonic()-t)),flush=True)
            feature_shards = []
            for i, source in enumerate(source_shards):
                t = time.monotonic()
                feature_shards.append(build_shard(source,caches,modelpaths,output/f'shard{i:05d}'))
                elapsed = time.monotonic()-t
                print(json.dumps(dict(stage='truth_blind_features',done=i+1,total=32,seconds=elapsed)),flush=True)
                if i == 0 and elapsed > 30:
                    raise RuntimeError('STOP_TECHNICAL_PILOT_COST')
            write_json(output/'feature_manifest.json',dict(status='SEALED_ALL_FEATURES_BEFORE_LABELS',
                cache=caches,shards=feature_shards,queries=1024,truth_opened=False))
            result = fit_evaluate(feature_shards,scope,output)
        result.update(resources=budget.report(),seconds=time.monotonic()-start,
                      preregistration=record(output/'preregistration.json'),features_manifest=record(output/'feature_manifest.json'),
                      outcomes=record(output/'outcomes.parquet'))
        write_json(output/'manifest.json',result)
        print(json.dumps(dict(stage='complete',quality=result['quality'],resources=result['resources'])),flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),exception=type(exc).__name__,seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
