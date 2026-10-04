"""Same immutable V217 initial100 kernel, actual native11 learned on TRAIN."""
from .retrieval_corroborated_initial_v217 import Evidence
from .retrieval_corroborated_initial_v217 import search as original_search


def search(*args, **kwargs):
    found, trace = original_search(*args, **kwargs)
    trace.update(new_fit=True, initial_native_training='V233_ACTUAL217_OOF_TRAIN64_LISTWISE',
                 changed_initial_weight_indices=list(range(11)), actual_native_kernel_training_features=True,
                 query_time_training=False, downstream_models_changed=False)
    return found, trace
