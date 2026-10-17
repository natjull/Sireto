"""Hash the frozen V142 replay inputs; never invoke retrieval or calibration."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
HOME = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
RUN = HOME/'v142/bano_native_20261013_night'
RESULT = ROOT/'docs/retrieval_frozen_v142.json'


def read(path):
    return json.loads(Path(path).read_text())


def signature(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def main():
    started = time.monotonic()
    paths = {}

    def add(path, role, expected=None):
        path = str(Path(path))
        if '.pending.' in path:
            raise ValueError('Pending artifact is not an authority')
        if path not in paths:
            paths[path] = dict(roles=set(), expected=set())
        paths[path]['roles'].add(role)
        if expected:
            paths[path]['expected'].add(expected)

    def refs(obj, role):
        if isinstance(obj, dict):
            if 'path' in obj and 'sha256' in obj:
                add(obj['path'], role, obj['sha256'])
            else:
                for key, value in obj.items():
                    if key != 'training_shards':
                        refs(value, role)
        elif isinstance(obj, list):
            for value in obj:
                refs(value, role)

    launch = read(RUN/'launch.json')
    refs(launch, 'launch_authority_and_sources')
    for name in ('launch.json', 'recipe.json', 'prepared.json', 'features_generation.json',
                 'calibration/generation.json', 'all_new_sources_max_bano/generation.json',
                 'all_new_sources_max_bano/generate_exit.json'):
        add(RUN/name, 'v142_manifest')
    # Only explicit replay manifests are expanded. No historical training/label
    # sources are traversed; model training_shards are not serving inputs.
    manifests = [RUN/'prepared.json', RUN/'features_generation.json', RUN/'calibration/generation.json',
                 RUN/'all_new_sources_max_bano/generation.json']
    for key in ('parent_prepared', 'supplier_generation', 'address_generation', 'rna_generation',
                'banatic_generation', 'bano_generation', 'baseline_generation', 'upstream_primary_generation'):
        manifests.append(Path(launch[key]['path']))
    for path in manifests:
        refs(read(path), 'replay_cache_or_model')
    for path in (RUN/'all_new_sources_max_bano/native').glob('*.json'):
        add(path, 'native_receipt')
    # Runtime helpers not all listed in the inherited launch helper collection.
    for name in ('embeddinggemma2_v115_streamed_activity_ablation', 'embeddinggemma2_v32_native_features',
                 'embeddinggemma2_v56_corporate_site', 'embeddinggemma2_v23_native_features'):
        add(ROOT/'scripts'/f'{name}.py', 'runtime_code')
    add(Path(__file__), 'sealer_code')
    add(ROOT/'docs/retrieval_site_target_timing_20261015.json', 'cost_reference_not_v142_measurement')
    index_root = Path(launch['index_meta']['path']).parent
    current_inventory = [[p.name, *signature(p)[:2]] for p in sorted(index_root.iterdir()) if p.is_file()]
    if current_inventory != launch['index_inventory']:
        raise ValueError('Tantivy inventory differs from V142')
    for name, _, _ in current_inventory:
        add(index_root/name, 'tantivy_index')
    snapshots = {p: signature(p) for p in paths}
    print(json.dumps(dict(status='HASHING', files=len(paths), bytes=sum(v[0] for v in snapshots.values()))), flush=True)
    artifacts = []
    for i, (path, entry) in enumerate(sorted(paths.items())):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024**2), b''):
                digest.update(block)
        sha = digest.hexdigest()
        if entry['expected'] and entry['expected'] != {sha}:
            raise ValueError(f'Historical hash mismatch: {path}')
        if signature(path) != snapshots[path]:
            raise ValueError(f'Input changed: {path}')
        row = dict(path=path, size_bytes=snapshots[path][0], sha256=sha, roles=sorted(entry['roles']))
        if path.startswith(str(ROOT)+'/') and Path(path).suffix == '.py' and Path(path) != Path(__file__):
            row['last_code_commit'] = subprocess.check_output(
                ['git', 'log', '-1', '--format=%H', '--', path], cwd=ROOT, text=True).strip()
        artifacts.append(row)
        if i % 100 == 0:
            print(json.dumps(dict(hashed=i+1, total=len(paths), elapsed_seconds=time.monotonic()-started)), flush=True)
    if any(signature(p) != s for p, s in snapshots.items()):
        raise ValueError('Frozen inputs changed during hashing')
    cal = read(RUN/'calibration/generation.json')
    assignment = read(cal['assignment']['path'])
    model_info = {}
    for k, ref in cal['models'].items():
        m = read(ref['path'])
        if len(m['coefficients']) != 58:
            raise ValueError('Expected 58 frozen coefficients')
        model_info[k] = dict(path=ref['path'], coefficients=m['coefficients'],
            train_queries=len(m['train_queries']), heldout_queries=len(m['heldout_queries']))
    output_ref = read(RUN/'all_new_sources_max_bano/generation.json')['outputs']
    if not any(r['path'] == output_ref['path'] and r['sha256'] == output_ref['sha256'] for r in artifacts):
        raise ValueError('Current V142 outputs must be explicitly hashed, not just their parent manifest')
    result = dict(status='FROZEN_BY_NATHAN_2026_10_17', variant='learned_native_ooc', output=output_ref,
        recipe_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H', '--',
            'scripts/embeddinggemma2_v142_bano_native_ablation.py'], cwd=ROOT, text=True).strip(),
        rule='No further retrieval experiment or change without a new written Nathan decision',
        scope='Sealed ALL526 replay inputs and their direct source authorities; not a rebuild of all ancestors',
        all_sha256_recomputed=True, file_count=len(artifacts),
        bytes_hashed=sum(r['size_bytes'] for r in artifacts), elapsed_seconds=time.monotonic()-started,
        index_files=len(current_inventory), index_bytes=sum(r[1] for r in current_inventory),
        assignment=dict(path=cal['assignment']['path'], query_count=len(assignment),
            rule="int.from_bytes(sha256((group_seed+'|'+human_siren).encode()).digest()[:8], 'big') % 5",
            group_seed=read(HOME/'v71/frozen_residual_20261013_a/calibration/recipe.json')['group_seed'],
            partitions={str(k): sum(v == k for v in assignment.values()) for k in range(5)},
            internal_partitions_not_closed_population_folds=True),
        models=model_info, unique_production_model_exists_in_frozen_recipe=False,
        production_fit_not_run=True, production_promotion_allowed=False,
        production_requirements=['One V32 plus one V71 residual fit on all eligible folds2/3/4 groups, retaining encoder-control exclusions',
            'Never average the five coefficient sets or route by a human SIREN at serving time',
            'Label-free online feature/source construction, cached-input replacement, memory/SLO and independent validation'],
        production_fit_estimate=dict(minutes=[45, 75], confidence_interval=False,
            assumption='One 284-query two-stage mining run measured 2571.82s; linear scale to375 eligible queries gives3396.12s; caches intact, optimizer alone <0.1s',
            excludes='New sources/features, all-query regeneration and independent production validation'),
        no_training=True, no_generation=True, closed_sets_opened=False, artifacts=artifacts)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status','file_count','bytes_hashed','elapsed_seconds','index_files')}), flush=True)


if __name__ == '__main__':
    main()
