"""Full validation census and a frozen correlated-dense family fusion.

No test, labels as features, target lookup, outcome-based selection, or serving
change. The entire 2212-query validation population is retained, including
missing/large communes. This is not a pristine/national/OOF certificate.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_compare as old
from src.xgb_matcher.retrieval_v20_workbench import record, verify

OUTPUT = io.HOME / 'v10'
SOURCE = io.HOME / 'night_20261006_a'
PREVIOUS = io.HOME / 'v8/site_pipeline_20261008_a'


def family_fusion(old_dense, new_dense, lexical, geo=(), geo_weight=1.):
    """Dense max RRF, not additive votes from two correlated encoders.

    All channels are independently generated initial100. No fixed master pool,
    trained post-ranker, minimum channel quota, or label-conditioned promotion.
    """
    if not 0 <= geo_weight <= 1: raise ValueError('Invalid geographic weight')
    dense = {}; sparse = {}; spatial = {}; parents = {}
    for rows, accumulator, weight in ((old_dense, dense, 1.),
            (new_dense, dense, 1.), (lexical, sparse, 1.), (geo, spatial, geo_weight)):
        if len(rows) > 100 or len({r['siret'] for r in rows}) != len(rows):
            raise ValueError('Independent initial100 cap and uniqueness')
        for rank, row in enumerate(rows, 1):
            key = row['siret']; parent = row['siren']
            if key in parents and parents[key] != parent: raise ValueError('Parent conflict')
            parents[key] = parent
            accumulator[key] = max(accumulator.get(key, 0.), weight / (60 + rank))
    scores = {k: dense.get(k, 0.) + sparse.get(k, 0.) + spatial.get(k, 0.) for k in parents}
    return [dict(siret=k, siren=parents[k], score=scores[k])
        for k in sorted(scores, key=lambda k: (-scores[k], k))[:100]]


def wilson(hits, n):
    if not n or not 0 <= hits <= n: raise ValueError('Positive complete denominator')
    p = hits / n; z = 1.959963984540054; denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [center-width, center+width]


def project_population(rows):
    if len({r['query_id'] for r in rows}) != len(rows): raise ValueError('Duplicate query')
    if any(r['fold'] not in (2, 3, 4) for r in rows): raise ValueError('Closed fold')
    # Join/split authority only; identifiers cannot enter the query representation.
    return [{k: r[k] for k in old.CRM} for r in rows]


def census(out):
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    manifest_path = SOURCE / 'manifest.json'; manifest = io.read(manifest_path)
    membership = SOURCE / 'validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']: raise ValueError('Split drift')
    rows = io.read(membership); panel = project_population(rows)
    if len(panel) != 2212: raise ValueError('Full validation required')
    training = SOURCE / 'train_membership.json'
    if io.digest(training) != manifest['split_hashes']['train']: raise ValueError('Train authority drift')
    train_parents = {r['siren'] for r in io.read(training)}
    val_parents = {r['siren'] for r in rows}
    if train_parents & val_parents: raise ValueError('TRAIN SIREN leakage')
    previous = io.read(PREVIOUS / 'launch.json')
    previous_ids = {r['query_id'] for r in io.read(verify(previous['panel']))}
    consumed_parents = {r['siren'] for r in rows if r['query_id'] in previous_ids}
    roles = {r['query_id']: ('PREVIOUS_PANEL_SIREN' if r['siren'] in consumed_parents
        else 'NEW_RETRIEVAL_QUERY_SIREN') for r in rows}
    out.mkdir(parents=True, exist_ok=False)
    io.write(out / 'panel.json', panel)
    io.write(out / 'roles.json', roles)
    backend = TantivyBackend(io.INDEX); inventory = old.inventory()
    if backend.searcher.num_docs != 82772691: raise ValueError('National index drift')
    counts = Counter(q['crm_insee'] for q in panel)
    zones = []; start = time.monotonic()
    for i, (zone, n) in enumerate(sorted(counts.items())):
        sites = backend.searcher.search(old.zone_query(backend, zone), limit=1, count=True).count if zone else 0
        zones.append(dict(zone=zone, queries=n, indexed_sites=sites,
            no_query_dropped=True, missing_zone=not bool(zone)))
        if i % 25 == 0:
            io.write(out / 'state.json', dict(status='QUERY_ONLY_FULL_VALIDATION_CENSUS',
                zones_counted=i+1, zones_total=len(counts), queries=len(panel), seconds=time.monotonic()-start))
    if inventory != old.inventory(): raise ValueError('Index changed')
    # A binomial interval understates dependence among sites of one company;
    # the eventual paired comparison must also bootstrap SIREN groups.
    planning = {str(n): {'if_99_2pct_observed': wilson(round(.992*n), n),
        'if_zero_misses': wilson(n, n)} for n in (80, len(panel), 4323, 8000)}
    summary = dict(status='SEALED_FULL_VALIDATION_CENSUS_NO_RECALL', queries=len(panel),
        siren_groups=len(val_parents), communes=len(counts), total_indexed_commune_sites=sum(z['indexed_sites'] for z in zones),
        missing_zone_queries=counts.get('', 0), zones=zones,
        folds=dict(Counter(str(q['fold']) for q in panel)),
        departments=dict(Counter(q['crm_insee'][:2] or 'UNKNOWN' for q in panel)),
        missing_number_queries=sum(not q['crm_number'] for q in panel),
        missing_name_queries=sum(not q['crm_name'] for q in panel),
        role_counts=dict(Counter(roles.values())), diagnostic_80_query_ids=len(previous_ids),
        population_policy='ENTIRE_PREEXISTING_VALIDATION_NO_EASY_ZONE_OR_STATUS_FILTER',
        train_siren_disjoint=True, used_for_checkpoint_selection=True,
        historical_pipeline_development_population=True, pristine_holdout=False,
        scope='CRM_VALIDATION_POPULATION_NOT_ALL_FUTURE_CRMS_NOT_NATIONAL_DENSE_OOF',
        no_retrieval_truth_opened=True, test_opened=False, fold0_opened=False, fold1_opened=False,
        downstream_models_changed=False, sources={'manifest':record(manifest_path),
            'membership':record(membership), 'train_membership':record(training),
            'previous_panel':previous['panel'], 'panel':record(out/'panel.json'),
            'roles':record(out/'roles.json'), 'implementation':record(Path(__file__))},
        frozen_challenger='RRF60_MAX_OF_OLD_AND_NEW_DENSE_PLUS_LEXICAL_PLUS_GEO',
        geo_weight_policy='1_RELIABLE_0.5_WEAK_UNRESOLVED_EMPTY',
        no_tuning_on_previous80=True, precision_planning_not_measured_recall=planning,
        requires_siren_cluster_bootstrap=True, absolute_certainty_impossible=True,
        seconds=time.monotonic()-start)
    io.write(out / 'census.json', summary)
    io.write(out / 'state.json', dict(status=summary['status'], census=record(out/'census.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated V10 namespace')
    census(args.output)
