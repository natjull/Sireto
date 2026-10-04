"""Full frozen128 CRM/encoder alignment, own query vector and held-out head.

No candidate/label/legacy expansion is a feature. Positions/IDs bind controls
only. National callers still certify sources, all sites and all outputs.
"""
from dataclasses import dataclass
import json

import numpy as np

from .retrieval_activity_decoder_v111 import query_design
from .retrieval_activity_semantic_v113 import normalized, scores
from .retrieval_activity_joint_v114 import FEATURES
from .retrieval_coupled_blocks_v121 import weights
from .retrieval_postscore_audit_v187 import canonical
from .retrieval_v20_workbench import verify

FIELDS = frozenset(('query_id', 'fold', 'crm_name', 'crm_number', 'crm_suffix',
                    'crm_street', 'crm_postcode', 'crm_insee'))


def rows(value):
    if (not isinstance(value, list) or len(value) != 128
            or any(not isinstance(q, dict) or set(q) != FIELDS or type(q['fold']) is not int
                   or q['fold'] not in (2, 3, 4)
                   or any(not isinstance(q[k], str) for k in FIELDS-{'fold'}) for q in value)
            or len({q['query_id'] for q in value}) != 128):
        raise ValueError('Every128 typed original CRM query, unique control ID and open folds234 required')
    return tuple(q.copy() for q in value)


@dataclass(frozen=True)
class Population:
    queries: tuple
    vectors: np.ndarray

    def select(self, position):
        if type(position) is not int or not 0 <= position < 128:
            raise ValueError('Exact original query position0..127 required, no clipping')
        # Own storage, unchanged numerical representation; never vector0 for
        # another query. No mutable view can alter the frozen full population.
        return self.queries[position].copy(), np.array(self.vectors[position:position+1], copy=True)


def load(old, encoder_launch, encoded_reference):
    if old.get('truth_opened') is not False or old.get('old_candidates_opened') is not False:
        raise ValueError('Entire closed-label query-only source required')
    population = rows(old['queries'])  # ALL128 before any selection/file read.
    origin = json.loads(verify(encoder_launch).read_text())
    if (origin.get('truth_opened') is not False or origin.get('old_candidates_opened') is not False
            or rows(origin['queries']) != population or origin['query_sources'] != old['query_sources']):
        raise ValueError('Exact same complete CRM order and source-bound encoder launch required')
    encoded = json.loads(verify(encoded_reference).read_text())
    # Only control ID/fold/vector authority is used from this legacy encoder
    # proof. Sparse expansion tokens, models and old outputs are NOT followed.
    if (encoded.get('truth_opened') is not False
            or encoded['resources']['pass_resource_policy'] is not True
            or [(r['query_id'], r['fold']) for r in encoded['rows']] != [(q['query_id'], q['fold']) for q in population]
            or canonical(encoded['vectors']) != canonical(old['query_vectors'])):
        raise ValueError('Entire original encoded128 vector/query order required')
    vectors = np.load(verify(old['query_vectors']), allow_pickle=False)
    if (vectors.shape != (128, 384) or vectors.dtype.kind != 'f'
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-6)):
        raise ValueError('All original finite unit128x384 query vectors required')
    vectors.flags.writeable = False
    return Population(population, vectors)


def activity(population, old, models, attributes, position):
    if (not isinstance(population, Population) or rows(old['queries']) != population.queries
            or models != json.loads(verify(old['models']).read_text()) or set(models) != {'2', '3', '4'}):
        raise ValueError('Same entire query population and frozen original own-fold models required')
    row, vector = population.select(position)
    ids = {q['query_id'] for q in population.queries}
    for fold, model in models.items():
        if (model['heldout_fold'] != int(fold) or set(model['training_folds']) != {2, 3, 4}-{int(fold)}
                or ids & set(model['query_ids']) or model['features'] != list(FEATURES)
                or model['identifiers_as_features'] or model['raw_activity_without_site_proof']
                or model['descriptions'] != old['activity_inputs']['descriptions']):
            raise ValueError('Complete safe held-out heads and no TRAIN/serving overlap required')
        weights(model['weights'])
    model = models[str(row['fold'])]
    refs = old['activity_inputs']
    cat = json.loads(verify(attributes.metadata['activity_catalog']).read_text())
    if (attributes.metadata['activity_catalog'] != model['official_catalog']
            or cat['official_labels'] != refs['semantic_refs']['labels']):
        raise ValueError('Same exact official description/class namespace required')
    descriptions = normalized(np.load(verify(refs['descriptions']), allow_pickle=False))
    parameter = np.load(verify(model['decoder_parameter']), allow_pickle=False)
    values = scores(query_design(vector), descriptions, parameter, model['decoder']['family'])[0]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Same original finite bounded activity grades required')
    return {i: float(v) for i, v in enumerate(values)}


def batches(width=2):
    if type(width) is not int or width not in (1, 2):
        raise ValueError('Physical one/two-query groups only; population stays128')
    return tuple(tuple(range(start, min(start+width, 128))) for start in range(0, 128, width))
