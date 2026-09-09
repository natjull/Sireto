"""One convex parent-only calibrator; fixed site evidence fusion, no oracle."""
import json
from pathlib import Path

import numpy as np

from .retrieval_v20_workbench import ROOT, record


def config_at(path):
    c = json.loads(Path(path).read_text())
    if (c['schema'] != 'sireto-v24-nested-fusion-1'
        or c['recipe'] != 'INNER_PARENT_RANK_LIKELIHOOD_MIXTURE_THEN_SITE_SCORE_FUSION'
        or c['outer_folds'] != [2, 3, 4]
        or (c['rank_constant'], c['log_barrier'], c['bisection_iterations']) != (60, .01, 80)
        or (c['master_cap'], c['reserved_prefix']) != (2000, 0)
        or c['national_projection_allowed'] or c['fold1_allowed']):
        raise ValueError('Frozen V24 recipe drift')
    return c


def code_records():
    paths = [Path(__file__), ROOT/'scripts/retrieval_v24_nested_fusion.py',
             ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
             ROOT/'src/xgb_matcher/retrieval_v21_membership.py',
             ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',
             ROOT/'src/xgb_matcher/retrieval_parent_fusion_v19.py',
             ROOT/'scripts/retrieval_v22_global_budget.py']
    return {str(p.relative_to(ROOT)): record(p) for p in paths}


def inner_jobs(outer):
    if outer not in (2, 3, 4):
        raise ValueError('Forbidden outer fold')
    others = [f for f in (2, 3, 4) if f != outer]
    return [(others[0], others[1]), (others[1], others[0])]  # train, target


def check_split(split, train, target, truth):
    if split['train_folds'] != train or split['target_folds'] != target or set(train) & set(target):
        raise ValueError('Model split contamination')
    train_ids = set(truth.loc[truth.fold.isin(train), 'query_id'])
    target_ids = set(truth.loc[truth.fold.isin(target), 'query_id'])
    if (not set(split['trained_query_ids']) <= train_ids
        or set(split['target_query_ids']) != target_ids
        or set(split['trained_query_ids']) & target_ids):
        raise ValueError('Model query split contamination')


def optimum(base, learned):
    base, learned = np.asarray(base, dtype=np.float64), np.asarray(learned, dtype=np.float64)
    if (base.ndim != 1 or base.shape != learned.shape or not len(base)
        or not np.isfinite(base).all() or not np.isfinite(learned).all()
        or (base <= 0).any() or (learned <= 0).any() or (base > 1).any() or (learned > 1).any()):
        raise ValueError('Invalid calibration probabilities')
    delta = learned-base
    def derivative(a):
        return float(-np.mean(delta/(base+a*delta)) - .01/a + .01/(1-a))
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo+hi)/2
        if derivative(mid) < 0:
            lo = mid
        else:
            hi = mid
    a = (lo+hi)/2
    return {'learned_weight': a, 'derivative': derivative(a),
            'conditional_nll': float(-np.log(base+a*delta).mean()),
            'regularized_objective': float(-np.log(base+a*delta).mean()-.01*(np.log(a)+np.log1p(-a)))}


def calibrate(probabilities, truth, outer):
    """Inputs contain per-parent probabilities, never target labels as features."""
    inner_jobs(outer)
    # Filter before the truth join: target label perturbation cannot affect fit.
    train_truth = truth.loc[truth.fold.ne(outer), ['query_id', 'fold', 'gt_siren']].sort_values('query_id')
    if (not set(train_truth.fold) == set((2, 3, 4))-{outer}
        or train_truth.query_id.duplicated().any()):
        raise ValueError('Calibration truth population')
    if (set(probabilities.fold) != set(train_truth.fold)
        or set(probabilities.query_id) != set(train_truth.query_id)
        or probabilities.duplicated(['query_id', 'siren']).any()):
        raise ValueError('Calibration score population/outer contamination')
    joined = probabilities.merge(train_truth, on=['query_id', 'fold'], validate='many_to_one')
    if len(joined) != len(probabilities):
        raise ValueError('Score fold mismatch')
    positive = joined.loc[joined.siren.eq(joined.gt_siren)].sort_values('query_id')
    if positive.query_id.duplicated().any():
        raise ValueError('Duplicate natural positive')
    fitted = optimum(positive.base_probability, positive.learned_probability)
    fitted.update(outer_fold=outer, calibration_folds=sorted(set(train_truth.fold)),
                  calibration_query_ids=sorted(set(train_truth.query_id)),
                  fitted_query_ids=list(positive.query_id), queries=len(train_truth),
                  fitted_queries=len(positive), skipped_absent_parent=len(train_truth)-len(positive))
    return fitted


def parent_probabilities(c, scores):
    """base_parents + explicitly selected OOF scores; no labels in SQL."""
    c.register('current_scores', scores)
    # Join must cover each selected query's complete parent universe exactly.
    if c.execute('''SELECT count(*) FROM (
      SELECT query_id,siren,count(*) n FROM current_scores GROUP BY ALL HAVING n<>1)''').fetchone()[0]:
        raise ValueError('Duplicate parent predictions')
    if c.execute('''SELECT count(*) FROM current_scores s ANTI JOIN base_parents b USING(query_id,fold,siren)''').fetchone()[0]:
        raise ValueError('Foreign parent predictions')
    if c.execute('''SELECT count(*) FROM base_parents b SEMI JOIN
      (SELECT DISTINCT query_id,fold FROM current_scores) q USING(query_id,fold)
      ANTI JOIN current_scores s USING(query_id,fold,siren)''').fetchone()[0]:
        raise ValueError('Missing parent predictions')
    if c.execute('SELECT count(*) FROM current_scores WHERE fold NOT IN (2,3,4) OR parent_score IS NULL OR NOT isfinite(parent_score)').fetchone()[0]:
        raise ValueError('Invalid parent scores')
    return c.execute('''WITH ranks AS (
        SELECT s.*,b.parent_rank,dense_rank() OVER(PARTITION BY s.query_id ORDER BY s.parent_score DESC) learned_rank
        FROM current_scores s JOIN base_parents b USING(query_id,fold,siren)
      ), rr AS (SELECT *,1.0/(60+parent_rank) b,1.0/(60+learned_rank) l FROM ranks)
      SELECT query_id,fold,siren,b/sum(b ORDER BY b) OVER(PARTITION BY query_id) base_probability,
        l/sum(l ORDER BY l) OVER(PARTITION BY query_id) learned_probability FROM rr
      ORDER BY query_id,siren''').df()


def make_master(c):
    """Tables: pool, site_features, parent_scores and fold_weights (no truth)."""
    if c.execute('SELECT count(*) FROM fold_weights WHERE fold NOT IN (2,3,4) OR learned_weight IS NULL OR NOT isfinite(learned_weight) OR learned_weight<=0 OR learned_weight>=1').fetchone()[0]:
        raise ValueError('Invalid fold weights')
    if c.execute('SELECT count(*),count(DISTINCT fold) FROM fold_weights').fetchone() != (3, 3):
        raise ValueError('Fold weights population')
    c.execute('''CREATE TEMP TABLE ranked_parents AS SELECT *,dense_rank() OVER(
      PARTITION BY query_id ORDER BY parent_score DESC) learned_parent_rank FROM parent_scores''')
    c.execute('''CREATE TEMP TABLE scored AS SELECT s.*,p.parent_score,p.learned_parent_rank,w.learned_weight,
      b.rescue_score base_score,s.direct_rrf+CASE WHEN s.local_site THEN
        1.0/(60+p.learned_parent_rank+s.joint_site_rank) ELSE 0 END learned_score,
      s.direct_rrf+(1-w.learned_weight)*(b.rescue_score-s.direct_rrf)
        +w.learned_weight*CASE WHEN s.local_site THEN
          1.0/(60+p.learned_parent_rank+s.joint_site_rank) ELSE 0 END rescue_score
      FROM site_features s JOIN pool b USING(query_id,fold,siren,siret)
      JOIN ranked_parents p USING(query_id,fold,siren) JOIN fold_weights w USING(fold)''')
    n = c.execute('SELECT count(*) FROM pool').fetchone()[0]
    if c.execute('SELECT count(*),count(DISTINCT (query_id,siret)) FROM scored').fetchone() != (n, n):
        raise ValueError('Candidate population drift')
    if c.execute('SELECT count(*) FROM scored WHERE rescue_score IS NULL OR NOT isfinite(rescue_score)').fetchone()[0]:
        raise ValueError('Invalid site score')
    c.execute('''CREATE TEMP TABLE master AS SELECT * EXCLUDE(r),r candidate_rank,
      'NESTED_PARENT_CALIBRATED_FUSION' selection_reason FROM (
        SELECT *,row_number() OVER(PARTITION BY query_id ORDER BY
          rescue_score DESC,direct_rrf DESC,parent_score DESC,siret) r FROM scored
      ) WHERE r<=2000''')
    if c.execute('''SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,
      max(candidate_rank) mx FROM master GROUP BY query_id HAVING n>2000 OR n<>nd OR n<>mx)''').fetchone()[0]:
        raise ValueError('Master integrity')
