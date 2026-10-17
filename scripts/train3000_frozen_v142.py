"""Read-only train3000 fidelity/OOF preflight; never substitute missing V142 proofs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import uuid

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pyarrow.parquet as pq
from scripts import retrieval_site_target_full as supervision
from scripts.retrieval_site_operational_metric import read, record

H = Path('/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2')
OUT = H/'v174/train3000_frozen_v142_20261017'
RUN = H/'v142/bano_native_20261013_night'
FROZEN = ROOT/'docs/retrieval_frozen_v142.json'
CONTRACT = ROOT/'docs/train3000_frozen_v142_contract.md'
RESULT = ROOT/'docs/train3000_frozen_v142_result.json'
GIB = 1024**3


def publish(path, value):
    """No clobber, no access to existing pending files; retain publication sources."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('Immutable preflight publication already exists: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent/'.train3000_frozen_v142_publications'/uuid.uuid4().hex
    staging.mkdir(parents=True)
    source = staging/'payload'
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n' if isinstance(value, dict) else value
    with source.open('x') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(source, path)


class FrozenInputs:
    def __init__(self, frozen):
        self.artifacts = {r['path']: r for r in frozen['artifacts']}
        self.checked = {}
        self.signatures = {}
        if len(self.artifacts) != frozen['file_count']:
            raise ValueError('Duplicate or incomplete frozen inventory')
        for path, ref in self.artifacts.items():
            if '.pending.' in path:
                raise ValueError('Pending file cannot be a serving authority')
            stat = Path(path).stat()
            if stat.st_size != ref['size_bytes']:
                raise ValueError('Frozen size changed: '+path)
            self.signatures[path] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)

    def path(self, ref):
        path = str(Path(ref['path'] if isinstance(ref, dict) else ref))
        expected = self.artifacts[path]
        if isinstance(ref, dict) and ref.get('sha256', expected['sha256']) != expected['sha256']:
            raise ValueError('Historical reference differs from freeze: '+path)
        if path not in self.checked:
            current = record(path, expected['sha256'])
            self.checked[path] = current
        return Path(path)

    def json(self, ref):
        return read(self.path(ref))

    def unchanged(self):
        for path, before in self.signatures.items():
            stat = Path(path).stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != before:
                raise ValueError('Frozen file changed during preflight: '+path)


def markdown(r):
    o = r['oof_coefficient_audit_before_closed_cohort_admission']
    lines = ['# Train3000 V142 — STOP préflight, 17 octobre 2026', '',
        '**Aucun smoke natif ni run3000 lancé. Aucun nouveau candidat.**', '',
        f'Contrat précommité : `{r["contract_commit"]}`. Gel : `36c7a476`.', '',
        '## Faisabilité entreprise, pas admission du cohort final', '',
        f'- Census projeté du ledger folds2/3/4 : {o["usable_human_queries"]} labels exploitables.',
        f'- Pilote : {o["pilot_queries"]} requêtes, {o["pilot_companies"]} entreprises humaines.',
        f'- Exclus du census par entreprise pilote : {o["pilot_company_queries_excluded"]}, dont {o["additional_same_pilot_company"]} hors query_id du pilote.',
        f'- Restent {o["nonpilot_company_queries"]} dossiers / {o["nonpilot_companies"]} entreprises ; coefficients V32 ET V71 hors entreprise : {o["oof_ok"]} OK, {o["oof_failed"]} échec.',
        '- Les dix modèles et les groupes de fit sont vérifiés ; affectation SHA256 SIREN identique au pilote.',
        '- Ce census n\'est PAS un tirage de3000 ni une admission de test4323/control962. Les exclusions spécifiques des jeux fermés et la stratification commune restent à sceller avant toute sélection.',
        '- Aucune preuve full-OOF de l\'encodeur n\'est déduite de cet audit des seuls coefficients.', '',
        '## Couverture des entrées scellées', '',
        '| Entrée nécessaire | Requêtes scellées | Couverture hors entreprises pilote |',
        '|---|---:|---:|']
    for name, c in r['frozen_query_coverage'].items():
        lines.append(f'| {name} | {c["sealed_queries"]} | {c["covered_nonpilot_queries"]}/{o["nonpilot_company_queries"]} |')
    lines += ['', 'La matrice numérique51 et les shards d\'activité, BANO et OSM sont requis par',
        '`embeddinggemma2_v142_bano_native_ablation.generate` et par le replay V173.',
        'Ils sont indexés par query_id et liés au panneau526 ; ils ne constituent pas',
        'un cache national de preuves applicable tel quel à toute nouvelle requête.',
        'Une absence de clé n\'est pas une preuve de score nul. Aucun fallback lexical,',
        'rattachement à un dossier voisin ou faux heldout n\'a été exécuté.', '',
        '## Hashes et ressources', '',
        f'- {r["frozen_inventory_files"]} tailles/statuts de fichiers du gel contrôlés ; {r["sha256_recomputed_files"]} SHA réellement recalculés ({r["sha256_recomputed_bytes"]} octets).',
        '- Contrôle complet des SHA des dix modèles, code scellé et manifestes utilisés. Contrôle intégral149Go du gel NON exécuté : gate manquant et VM occupée. Il resterait obligatoire avant génération.',
        f'- Préflight réel : {r["elapsed_seconds"]:.2f} s ; pic RSS {r["peak_rss_bytes"]/1024**2:.1f} Mio.',
        f'- hydraulic : `{r["hydraulic"]["status"]}` ; aucune action VM. Calcul lourd non lancé.',
        '- Smoke20 : 0/20 ; s/requête, mémoire du smoke et ETA3000 non mesurés. PID superviseur : aucun.',
        '- Labels, modèles, index, caches et pending préexistants inchangés ; aucune évaluation de recall ni apprentissage.', '',
        '## Blocage et suite nécessaire', '',
        'Construire/adapter les préparateurs query-only pour de nouvelles requêtes,',
        'avec exactement les mêmes règles et autorités officielles, dans v174 sans',
        'modifier les caches526. Cette matérialisation de nouvelles preuves n\'est pas',
        'un fit production ; sa fidélité doit être établie avant de prétendre réutiliser',
        'la recette figée. La clause « mêmes caches, hashes » a été interprétée strictement,',
        'pas comme une autorisation de leur extension silencieuse.',
        'Puis sceller l\'exclusion des jeux fermés, les tailles de communes, le tirage3000,',
        'recalculer tous les SHA du gel, exécuter20 vrais natifs sous supervision,',
        'et seulement alors lancer le complet détaché avec évaluation séparée/commit final.', '',
        f'Statut : `{r["status"]}`. CLI : `.venv/bin/python scripts/train3000_frozen_v142.py --status`.']
    return '\n'.join(lines)+'\n'


def preflight():
    if RESULT.exists():
        raise FileExistsError('Preflight already published; use --status, preserve all artifacts')
    started = time.monotonic()
    frozen = read(FROZEN)
    frozen_bytes = subprocess.check_output(['git', 'show', '36c7a476:docs/retrieval_frozen_v142.json'], cwd=ROOT)
    if FROZEN.read_bytes() != frozen_bytes or frozen['variant'] != 'learned_native_ooc':
        raise ValueError('Freeze differs from 36c7a476')
    contract_commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(CONTRACT)], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'show', contract_commit+':docs/train3000_frozen_v142_contract.md'], cwd=ROOT) != CONTRACT.read_bytes():
        raise ValueError('Contract must be committed before execution')
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Fresh preflight namespace required; no partial artifacts overwritten')
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = FrozenInputs(frozen)
    # Only small replay authorities/code are hashed here. Never start the full
    # 149GB scan or a native worker while hydraulic may be active.
    for ref in frozen['artifacts']:
        if 'last_code_commit' in ref or 'v142_manifest' in ref['roles'] or 'sealer_code' in ref['roles']:
            inputs.path(ref)
    launch = inputs.json(RUN/'launch.json')
    for ref in (frozen['output'], launch['index_manifest'], launch['index_meta']):
        inputs.path(ref)
    fg = inputs.json(RUN/'features_generation.json')
    prepared = inputs.json(RUN/'prepared.json')
    pg = inputs.json(launch['parent_prepared'])
    cal = inputs.json(RUN/'calibration/generation.json')
    panel = inputs.json(launch['panel'])
    ids = {q['query_id'] for q in panel}
    if len(ids) != len(panel) or len(ids) != 526 or any(q['fold'] not in (2, 3, 4) for q in panel):
        raise ValueError('Frozen pilot scope drift')
    baseline = read(ROOT/'docs/step2_baseline_v142.json')
    ledger_ref = baseline['source_refs']['ledger']
    # Same projected census as step2.additional_examples, never opening a
    # held-out membership/feature/label file or constructing a final cohort.
    ledger = record(ledger_ref['path'], ledger_ref['sha256'])
    rows = pq.read_table(ledger['path'], columns=['query_id', 'fold', 'human_ground_truth_siret',
        'human_ground_truth_siren'], filters=[('fold', 'in', [2, 3, 4])], use_threads=False).to_pylist()
    valid = [r for r in rows if re.fullmatch(r'\d{14}', str(r['human_ground_truth_siret']))
             and r['human_ground_truth_siret'][:9] == r['human_ground_truth_siren']]
    labels = {r['query_id']: r for r in valid}
    if len(labels) != len(valid) or not ids <= labels.keys():
        raise ValueError('Duplicate query or missing pilot label')
    pilot_companies = {labels[q]['human_ground_truth_siren'] for q in ids}
    pilot_company_queries = [r for r in valid if r['human_ground_truth_siren'] in pilot_companies]
    extra = [r for r in valid if r['human_ground_truth_siren'] not in pilot_companies]
    from scripts import embeddinggemma2_v71_frozen_residual as residual
    models, train_hashes = {}, {}
    for k, ref in cal['models'].items():
        m = inputs.json(ref)
        base = inputs.json(m['fixed_baseline'])
        if m['coefficients'] != frozen['models'][k]['coefficients'] or len(m['coefficients']) != 58:
            raise ValueError('Frozen V71 coefficients drift')
        if len(base['coefficients']) != 28 or [m['coefficients'][i] for i in residual.BASE_AT] != base['coefficients']:
            raise ValueError('Frozen V32 coefficient projection drift')
        hashed = []
        for model in (base, m):
            if not set(model['train_queries']) <= ids or set(model['train_queries']) & set(model['heldout_queries']):
                raise ValueError('Model fit scope outside frozen pilot')
            actual = {hashlib.sha256(labels[q]['human_ground_truth_siren'].encode()).hexdigest() for q in model['train_queries']}
            if sorted(actual) != model['train_siren_hashes'] or not model['no_group_overlap']:
                raise ValueError('Historical company fit hashes do not match labels')
            hashed.append(actual)
        train_hashes[k] = hashed
        models[k] = dict(v71=inputs.checked[str(inputs.path(ref))],
                         v32=inputs.checked[str(inputs.path(m['fixed_baseline']))],
                         v71_train_companies=len(hashed[1]), v32_train_companies=len(hashed[0]))
    if set(models) != set('01234'):
        raise ValueError('Exactly five frozen internal partitions')
    seed = frozen['assignment']['group_seed']
    def partition(siren):
        return int.from_bytes(hashlib.sha256((seed+'|'+siren).encode()).digest()[:8], 'big') % 5
    assignment = inputs.json(cal['assignment'])
    if any(assignment[q] != partition(labels[q]['human_ground_truth_siren']) for q in ids):
        raise ValueError('Pilot SIREN assignment rule not reproduced')
    routing_counts = {str(k): dict(ok=0, failed=0) for k in range(5)}
    for r in extra:
        k = str(partition(r['human_ground_truth_siren']))
        digest = hashlib.sha256(r['human_ground_truth_siren'].encode()).hexdigest()
        ok = all(digest not in train for train in train_hashes[k])
        routing_counts[k]['ok' if ok else 'failed'] += 1
    universe = {r['query_id'] for r in extra}
    scopes = dict(numeric51=fg['features'], activity_shards=fg['activity_shards'],
                  bano_leaf_shards=prepared['shards'], osm_leaf_shards=pg['shards'])
    coverage = {name: dict(sealed_queries=len(scope), covered_nonpilot_queries=len(universe & scope.keys()))
                for name, scope in scopes.items()}
    if any(set(scope) != ids for scope in scopes.values()):
        raise ValueError('Unexpected frozen query input scope')
    if any(c['covered_nonpilot_queries'] for c in coverage.values()):
        raise ValueError('Unexpected coverage: STOP report assumptions no longer apply')
    inputs.unchanged()
    if record(ledger['path']) != ledger:
        raise ValueError('Ledger changed during projected census')
    vm = supervision.hydraulic()
    result = dict(status='STOP_MISSING_FROZEN_QUERY_INPUTS', observed_unix=time.time(), contract_commit=contract_commit,
        authorization='Nathan sireto-site-target-7.md', freeze_commit='36c7a476',
        frozen_manifest=record(FROZEN), implementation=record(Path(__file__)), contract=record(CONTRACT),
        namespace=str(OUT), oof_coefficient_audit_before_closed_cohort_admission=dict(
            projected_234=len(rows), usable_human_queries=len(valid), invalid_human_queries=len(rows)-len(valid),
            pilot_queries=len(ids), pilot_companies=len(pilot_companies),
            pilot_company_queries_excluded=len(pilot_company_queries),
            additional_same_pilot_company=len(pilot_company_queries)-len(ids),
            nonpilot_company_queries=len(extra), nonpilot_companies=len({r['human_ground_truth_siren'] for r in extra}),
            oof_ok=sum(r['ok'] for r in routing_counts.values()), oof_failed=sum(r['failed'] for r in routing_counts.values()),
            by_model=routing_counts, scope='V32/V71 coefficients only, census not final closed-set-admitted cohort'),
        models=models, frozen_query_coverage=coverage, ledger=ledger,
        frozen_inventory_files=len(inputs.artifacts), frozen_inventory_sizes_unchanged=True,
        sha256_recomputed_files=len(inputs.checked), sha256_recomputed_bytes=sum(r['size_bytes'] for r in inputs.checked.values()),
        full_frozen_sha256_recheck_complete=False, sha256_checked=list(inputs.checked.values()),
        hydraulic=vm, generated_queries=0, selected_queries=0, requested_queries=3000,
        final_closed_set_exclusions_sealed=False, stratified_sample_sealed=False,
        native_smoke_queries=0, native_smoke_seconds_per_query=None, native_smoke_peak_rss_bytes=None,
        full_run_started=False, supervisor_pid=None, eta_seconds=None,
        elapsed_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024),
        full_encoder_oof=False, closed_membership_files_opened=False, closed_queries_generated=0,
        labels_modified=False, retrieval_changed=False, models_trained=0, evaluated_queries=0,
        existing_pending_files_touched=False, source_mutation_observed=False)
    publish(OUT/'preflight.json', result)
    publish(RESULT, result)
    publish(RESULT.with_suffix('.md'), markdown(result))
    return result


def status():
    if not RESULT.exists():
        return dict(status='NOT_STARTED', namespace=str(OUT), supervisor_pid=None)
    r = read(RESULT)
    return {k: r[k] for k in ('status', 'namespace', 'contract_commit', 'generated_queries',
        'selected_queries', 'requested_queries', 'native_smoke_queries', 'supervisor_pid',
        'eta_seconds', 'full_run_started', 'oof_coefficient_audit_before_closed_cohort_admission',
        'frozen_query_coverage', 'full_frozen_sha256_recheck_complete')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true')
    group.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        preflight()
    print(json.dumps(status(), indent=2, ensure_ascii=False))
