"""Label-free authority for complete V224 structural initial100 outputs only."""
import json
from pathlib import Path

from scripts import retrieval_v224_corroborated_population as generation
from .retrieval_memory_policy_v214 import LIMITS
from .retrieval_v20_workbench import record, verify

STATUS = 'PERMITTED_V225_ALL128_STRUCTURAL_POSTSEAL_ONLY'


def header(root):
    root = Path(root)
    launch, manifest, outputs = generation.sealed(root)
    if (type(manifest.get('peak_worker_rss_bytes')) is not int
            or not 0 < manifest['peak_worker_rss_bytes'] <= LIMITS['max_rss_bytes']):
        raise ValueError('Actual full128 generation under authorized resources required')
    return root, manifest


def source_scope():
    """Official namespace specs only; no target table or truth lookup."""
    _, source, _, _, _ = generation.backend.inputs()
    attributes = json.loads(verify(source['structured']).read_text())
    catalog = json.loads(verify(attributes['sources']['catalog']).read_text())
    if attributes['stats']['sites'] != 43896818:
        raise ValueError('Original native leaf identity scope required')
    return source['catalog'], catalog['input_arrays']['endpoint_keys']


def authorize(root):
    root, manifest = header(root)  # ALL16 producer exits/resources, not a seal flag.
    launch, _, outputs = generation.sealed(root)
    controls = [dict(position=i, query_id=q['query_id'], fold=q['fold'])
                for i, q in enumerate(launch['queries'])]
    generation.valid_rows(outputs['rows'], launch['queries'], list(range(128)))
    current, native = source_scope()
    return dict(status=STATUS, generation=record(root/'manifest.json'),
        launch=manifest['launch'], outputs=manifest['outputs'], producers=manifest['pairs'],
        source_epochs=outputs['source_epochs'], controls=controls, queries=128, population=128,
        current_catalog=current, native_keys=native, native_leaf_sites=43896818,
        available_catalog_sites=44403034, full_appended_leaf_universe=False,
        actual_generation_admission_verified=True, all128_before_targets=True,
        evaluation_process_supervision_still_required=True, **generation.FALSE)


def recheck(permission):
    if (permission.get('status') != STATUS
            or any(type(permission.get(k)) is not int or permission[k] != 128 for k in ('queries', 'population'))
            or any(permission.get(k) is not True for k in ('actual_generation_admission_verified',
                    'all128_before_targets', 'evaluation_process_supervision_still_required'))
            or any(permission.get(k) is not False for k in generation.SAFE_FALSE)
            or permission.get('full_appended_leaf_universe') is not False):
        raise ValueError('Typed closed-label complete structural permission required')
    root = verify(permission['generation']).parent
    if authorize(root) != permission:
        raise ValueError('Same original outputs, all producers, epochs and controls required')
    return header(root)
