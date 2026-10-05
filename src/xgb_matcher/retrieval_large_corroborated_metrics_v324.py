"""Pure post-seal quality report; never authorizes labels or generates candidates.

The separate national caller must prove full2000 supervised admission BEFORE
opening targets. There is no file/network IO or partial-population evaluation.
"""
from collections import Counter
import math

from . import retrieval_large_human_metrics_v324 as metrics
from .retrieval_structural_channels_v210 import identity

CHANNELS = ('joint_proof',)


def quantile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered)-1)*fraction
    lower = int(point)
    upper = math.ceil(point)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(point-lower)


def evaluate(generated, controls, labels, official, acceptable, native_present, available_present):
    """Caller must supply unmodified original outputs and post-seal target data."""
    scope = {r['query_id']: r['fold'] for r in controls}
    rows = metrics.keyed(generated, scope)
    if (set(available_present) != set(scope) or any(type(v) is not bool for v in available_present.values())
            or set(native_present) != set(scope) or any(type(v) is not bool for v in native_present.values())
            or any(native_present[qid] and not available_present[qid] for qid in scope)):
        raise ValueError('Separate exact native/available presence for ALL2000 targets required')
    converted, times, candidates_by_query = [], [], {}
    admitted, represented = Counter(), Counter()
    for control in controls:
        row = rows[control['query_id']]
        candidates = row['results']['primary']
        if len(candidates) > 100:
            raise ValueError('No recapping of oversized initial retrieval output')
        seen, channels = set(), set()
        for rank, candidate in enumerate(candidates, 1):
            siret, _ = identity(candidate)
            score = candidate['source_score']
            if (siret in seen or type(candidate['retrieval_rank']) is not int or candidate['retrieval_rank'] != rank
                    or candidate['channel'] not in CHANNELS or isinstance(score, bool)
                    or not isinstance(score, (float, int)) or not math.isfinite(score) or score < 0):
                raise ValueError('Original unique emission ranks, source channels and scores required')
            seen.add(siret)
            channels.add(candidate['channel'])
            admitted[candidate['channel']] += 1
        represented.update(channels)
        elapsed = row['seconds']['primary']
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
                or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError('Actual finite nonnegative query seconds required')
        times.append(elapsed)
        candidates_by_query[control['query_id']] = candidates
        converted.append(dict(position=row['position'], query_id=row['query_id'], fold=row['fold'], candidates=candidates))
    # Kernel324 validates ALL2000 positions/typed safe folds, immutable targets,
    # every human label in denominator, cap and official same-site equivalence.
    result = metrics.evaluate(converted, controls, labels, official, acceptable, native_present)
    emission_hits, losses = Counter(), Counter()
    for outcome in result['outcomes']:
        qid = outcome['query_id']
        outcome['native_index_catalog_present'] = native_present[qid]
        outcome['available_official_catalog_present'] = available_present[qid]
        if outcome['exact_hit']:
            source = candidates_by_query[qid][outcome['exact_rank']-1]['channel']
            emission_hits[source] += 1
            loss = 'HIT'
        elif not available_present[qid]:
            loss = 'AVAILABLE_OFFICIAL_CATALOG_ABSENT'
        elif not native_present[qid]:
            loss = 'AVAILABLE_BUT_NOT_IN_NATIVE_LEAF_UNIVERSE'
        else:
            loss = 'NOT_RETRIEVED_AT100'
        outcome['structural_loss'] = loss
        losses[loss] += 1
    result.update(status='MEASURED_V324_CORROBORATED_INITIAL100_NEW2000_DEV_PANEL',
        membership_scope='MAIN_CURRENT_PRESENT_MEANS_NATIVE_LEAF_UNIVERSE; AVAILABLE_CATALOG_SEPARATE',
        structural_loss_counts=dict(losses),
        emitted_channel_statistics={channel: dict(candidates_admitted=admitted[channel],
            queries_with_admission=represented[channel], exact_hits_attributed_to_emission=emission_hits[channel],
            all_human_denominator=2000) for channel in CHANNELS},
        channel_statistics_scope='ONE_GLOBAL_JOINT_METRIC_NO_STANDALONE_CHANNEL_ABLATION',
        latency=dict(queries=2000, total_query_seconds=sum(times), mean_seconds=sum(times)/2000,
            p50_seconds=quantile(times, .5), p95_seconds=quantile(times, .95), p99_seconds=quantile(times, .99),
            max_seconds=max(times), cache_state='UNCONTROLLED', production_slo_validated=False),
        generation_admission_verified_by_this_kernel=False, target_io_authorized_by_this_kernel=False)
    return result


