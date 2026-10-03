"""Entity-disjoint vocabulary AND targets for a learned initial query metric.

V59 built its word vocabulary before nested splitting. Here every vocabulary
is derived only from the positive/negative official documents of permitted
fit queries, excluding protected parents even when they occur as negatives.
"""
import numpy as np

from src.xgb_matcher.retrieval_native_v57 import fit as native_fit
from src.xgb_matcher.retrieval_nested_v53 import positions
from src.xgb_matcher.retrieval_sparse_query_v59 import vocabulary, targets, field_bags, solve, controls, summary


def allowed_documents(rows, documents, indices, blocked):
    blocked = set(blocked); wanted = set()
    for i in indices:
        r = rows[i]
        if r['positive_siren'] in blocked:
            raise ValueError('Protected positive in vocabulary fit')
        pairs = [(r['positive_siret'],r['positive_siren'])] + [(n['siret'],n['siren']) for n in r['negatives']]
        for sid,parent in pairs:
            if len(sid)!=14 or len(parent)!=9 or not sid.isdecimal() or not parent.isdecimal() or not sid.startswith(parent):
                raise ValueError('Official membership drift')
            if parent not in blocked:
                wanted.add(sid)
    if not wanted or not wanted.issubset(documents):
        raise ValueError('Missing permitted official documents')
    return {sid:documents[sid] for sid in sorted(wanted)}


def target_matrix(rows, documents, vocab, indices):
    # Crucially, no heldout/validation target is even constructed for the fit.
    result = np.zeros((len(rows),len(vocab)))
    result[indices] = targets([rows[i] for i in indices],documents,vocab)
    return result


def fit_view(rows, documents, indices, blocked, config):
    safe = allowed_documents(rows,documents,indices,blocked)
    vocab = vocabulary(safe,config['vocabulary_cap'],config['minimum_document_frequency'],config['maximum_document_fraction'])
    if len(vocab)<config.get('minimum_vocabulary_words',64):
        raise ValueError('Insufficient safe official vocabulary')
    return vocab,target_matrix(rows,safe,vocab,indices),dict(
        official_fit_documents=len(safe),vocabulary_words=len(vocab),
        target_queries=len(indices),protected_document_overlap=0)


def select(q,rows,scenes,documents,registry,held,config):
    membership=[dict(query_id=r['query_id'],fold=r['fold'],positive_siren=r['positive_siren']) for r in rows]
    train,validation,inner=positions(membership,held,registry)
    outer=set(registry[str(held)]); blocked=outer|inner
    inner_vocab,inner_y,inner_scope=fit_view(rows,documents,train,blocked,config)
    safe_registry={k:list(v) for k,v in registry.items()}; safe_registry[str(held)]=sorted(blocked)
    native_scenes=[dict(s,positive_features=s['positive_features'][:11],
        negatives=[dict(n,features=n['features'][:11]) for n in s['negatives']]) for s in scenes]
    baseline=native_fit(native_scenes,held,safe_registry,config['native_ridge'])
    if set(baseline['query_ids'])!={rows[i]['query_id'] for i in train}:
        raise ValueError('Inner fit query scope drift')
    weights=np.array(baseline['weights']); bags={s:field_bags(f) for s,f in documents.items()}
    zero=np.zeros((q.shape[1],len(inner_vocab)),dtype=np.float32)
    before=summary(controls(q,zero,inner_vocab,rows,scenes,bags,weights,0.,validation,outer))
    candidates=[dict(ridge=None,alpha=0.,inner=before)]
    for ridge in config['ridges']:
        projection=solve(q,inner_y,train,ridge)
        for alpha in config['alphas']:
            candidates.append(dict(ridge=ridge,alpha=alpha,inner=summary(
                controls(q,projection,inner_vocab,rows,scenes,bags,weights,alpha,validation,outer))))
    selected=min(candidates,key=lambda c:(c['inner']['head_error'],c['inner']['mean_outrank_count'],c['alpha'],-(c['ridge'] or 0.)))
    final_indices=train+validation
    final_vocab,final_y,final_scope=fit_view(rows,documents,final_indices,outer,config)
    projection=np.zeros((q.shape[1],len(final_vocab)),dtype=np.float32) if selected['alpha']==0 else solve(q,final_y,final_indices,selected['ridge'])
    final=native_fit(native_scenes,held,registry,config['native_ridge'])
    if set(final['query_ids'])!={rows[i]['query_id'] for i in final_indices}:
        raise ValueError('Final fit query scope drift')
    weights=np.array(final['weights']); held_indices=[i for i,r in enumerate(rows) if r['fold']==held]
    zero=np.zeros_like(projection)
    outer_before=controls(q,zero,final_vocab,rows,scenes,bags,weights,0.,held_indices)
    outer_after=controls(q,projection,final_vocab,rows,scenes,bags,weights,selected['alpha'],held_indices)
    return projection,final_vocab,dict(heldout_fold=held,selected=selected,candidates=candidates,
        alpha=selected['alpha'],native=final,query_ids=[rows[i]['query_id'] for i in final_indices],
        inner_vocabulary_scope=inner_scope,final_vocabulary_scope=final_scope,
        inner_entities=sorted(inner),inner_vocabulary=inner_vocab,
        outer_before=summary(outer_before),outer_after=summary(outer_after),outer_controls=outer_after,
        outer_controls_used_in_selection=False,heldout_documents_used_for_vocabulary=False,
        validation_documents_used_for_inner_vocabulary=False,identifiers_as_features=False,recall_measured=False)
