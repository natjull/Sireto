"""Point scores AFTER sealed all128 authority. Never serving or recall."""
import math


def point_scores(native, parsed, ids):
    ids = sorted(set(ids)); scores = dict.fromkeys(ids, 0.)
    tv = native._tantivy
    gate = tv.Query.const_score_query(tv.Query.term_set_query(native.index.schema, 'siret', ids), 0.)
    query = tv.Query.boolean_query([(tv.Occur.Must, gate), (tv.Occur.Must, parsed)])
    seen = set()
    for score, loc in native.searcher.search(query, limit=len(ids), count=False).hits:
        s = native._first(native.searcher.doc(loc).to_dict(), 'siret')
        if s not in scores or s in seen or not math.isfinite(score) or score < 0:
            raise ValueError('Finite unique original diagnostic point required')
        seen.add(s); scores[s] = float(score)
    return scores


def check_scores(scores, emitted):
    for r in emitted:
        if not math.isclose(scores[r['siret']], r['source_score'], rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError('Original V217 emitted native score not reproduced')


def summarize(rows):
    if len(rows) != 10 or len({r['query_id'] for r in rows}) != 10:
        raise ValueError('ALL10 original V218 losses required')
    if any(r['score_diagnostic']['native_score_reproduction_pass'] is not True for r in rows):
        raise ValueError('Actual native frontier reproduction required')
    return dict(misses_audited=10, rows=rows,
        summary=dict(strictly_below_frontier=sum(r['score_diagnostic']['deficit'] > 2e-6 for r in rows),
            tied_at_frontier=sum(abs(r['score_diagnostic']['deficit']) <= 2e-6 for r in rows),
            regressions_vs215_positions=[r['position'] for r in rows if r['position'] in (12,62)]),
        scope='LABEL_SELECTED_DIAGNOSTIC_NOT_RECALL_OR_UNBIASED_ABLATION',
        model_fit=False, outputs_promoted_to_features=False, generation_modified=False,
        ready_for_fold1=False, recall_measured=False, fold0_opened=False,
        fold1_opened=False, downstream_models_changed=False)
