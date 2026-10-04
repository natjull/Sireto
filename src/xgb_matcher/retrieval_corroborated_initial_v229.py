"""V217 native initial100 unchanged, with a TRAIN-only learned place weight."""
from .retrieval_corroborated_initial_v217 import Evidence
from .retrieval_corroborated_initial_v217 import search as original_search


def search(*args, **kwargs):
    found, trace = original_search(*args, **kwargs)
    trace.update(new_fit=True, initial_place_training='V228_OOF_TRAIN64_NATIONAL_CONTRASTS',
                 changed_initial_weight_indices=[4], query_time_training=False,
                 downstream_models_changed=False)
    return found, trace
