"""Read-only ALL526 training-target mapping; no fit, retrieval or recall."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.retrieval_site_operational_metric import (
    address_match, candidate_point, crm_point, distance, threshold,
    fingerprint, index, read, record, verified,
)

CONTRACT = ROOT/'docs/retrieval_site_target_training_contract_20261015.md'
RESULT = ROOT/'docs/retrieval_site_target_mapping_20261015.json'


def contract_commit():
    rel = str(CONTRACT.relative_to(ROOT))
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', rel], cwd=ROOT, text=True).strip()
    if not commit or subprocess.check_output(['git', 'show', f'{commit}:{rel}'], cwd=ROOT) != CONTRACT.read_bytes():
        raise ValueError('STOP: contract must be committed and unchanged')
    return commit


def main():
    started = time.monotonic()
    commit = contract_commit()
    if RESULT.exists() or RESULT.with_suffix('.md').exists():
        raise FileExistsError('Mapping outputs already exist; no overwrite')
    prior = read(ROOT/'docs/retrieval_site_operational_metric_result_20261014.json')
    refs = {k: prior['source_refs'][k] for k in ('panel', 'geometry', 'ledger', 'stock', 'study')}
    for k in ('panel', 'geometry', 'ledger'):
        verified(refs[k])
    panel = index(read(refs['panel']['path']))
    geometry = index(read(refs['geometry']['path']))
    if len(panel) != 526 or set(panel) != set(geometry) or any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('STOP: ALL526 folds2/3/4 required')
    labels = index(pq.read_table(refs['ledger']['path'], columns=['query_id', 'fold', 'human_ground_truth_siret', 'human_ground_truth_siren'],
        filters=[('fold', 'in', [2, 3, 4]), ('query_id', 'in', sorted(panel))], use_threads=False).to_pylist())
    if set(labels) != set(panel) or any(labels[q]['fold'] != panel[q]['fold'] for q in panel):
        raise ValueError('Label scope mismatch')
    signatures = {k: fingerprint(refs[k]['path']) for k in ('stock', 'study')}
    for k, sig in signatures.items():
        if any(refs[k][field] != value for field, value in sig.items()):
            raise ValueError('Source changed since site metric')
    con = duckdb.connect(':memory:', config={'threads': '1', 'memory_limit': '512MB', 'max_temp_directory_size': '0B'})
    con.register('requested', pa.table({'siren': sorted({r['human_ground_truth_siren'] for r in labels.values()})}))
    cols = ['siren', 'siret', 'etablissementSiege', 'etatAdministratifEtablissement',
            'numeroVoieEtablissement', 'indiceRepetitionEtablissement', 'dernierNumeroVoieEtablissement',
            'indiceRepetitionDernierNumeroVoieEtablissement', 'typeVoieEtablissement', 'libelleVoieEtablissement', 'codeCommuneEtablissement']
    sites = index(con.execute('SELECT '+','.join('s.'+c for c in cols)+' FROM read_parquet(?) s SEMI JOIN requested r USING(siren)',
                             [refs['stock']['path']]).fetch_arrow_table().to_pylist(), 'siret')
    con.unregister('requested')
    con.register('requested', pa.table({'siret': sorted(sites)}))
    points = index(con.execute('SELECT s.siret,s.x,s.y,s.qualite_xy,s.epsg,s.distance_precision FROM read_parquet(?) s SEMI JOIN requested r USING(siret)',
                              [refs['study']['path']]).fetch_arrow_table().to_pylist(), 'siret')
    con.close()
    families = defaultdict(list)
    for s in sites.values():
        families[s['siren']].append(s)
    rows = []
    for qid in sorted(panel):
        label = labels[qid]
        original = label['human_ground_truth_siret']
        if len(original) != 14 or original[:9] != label['human_ground_truth_siren']:
            raise ValueError('Invalid human identity')
        cp = crm_point(geometry[qid]['geometry'])
        evidence = []
        for s in families[label['human_ground_truth_siren']]:
            p = candidate_point(points.get(s['siret']))
            d, t = distance(cp, p), threshold(cp, p)
            am = address_match(panel[qid], s)
            dm = d is not None and t is not None and d <= t
            evidence.append(dict(siret=s['siret'], active_state=s['etatAdministratifEtablissement'],
                is_headquarters=str(s['etablissementSiege']).lower() == 'true', address_match=am,
                distance_match=dm, at_crm_by_rule=am or dm, distance_m=d, threshold_m=t,
                candidate_precision=p['precision']))
        le = next((e for e in evidence if e['siret'] == original), None)
        matches = [e for e in evidence if e['at_crm_by_rule']]
        if le and le['at_crm_by_rule']:
            branch, chosen = 1, le
        elif matches:
            branch = 2
            chosen = min(matches, key=lambda e: ({'A': 0, 'F': 1}.get(e['active_state'], 2), not e['address_match'],
                e['distance_m'] if e['distance_m'] is not None else float('inf'), e['siret']))
        else:
            branch, chosen = 3, le
        target = chosen['siret'] if chosen else original
        rows.append(dict(query_id=qid, fold=panel[qid]['fold'], branch=branch, label_siret=original,
            training_target_siret=target, changed=target != original,
            label_is_headquarters=le['is_headquarters'] if le else None,
            target_is_headquarters=chosen['is_headquarters'] if chosen else None,
            label_missing_from_stock=le is None, crm_precision=cp['precision'], label_evidence=le,
            target_evidence=chosen, same_siren_sites_scanned=len(evidence), matching_sites=sorted(matches, key=lambda e: e['siret'])))
    def summary(rs):
        return dict(queries=len(rs), branches=dict(Counter(str(r['branch']) for r in rs)),
            changed=sum(r['changed'] for r in rs), changed_original_headquarters=sum(r['changed'] and r['label_is_headquarters'] is True for r in rs),
            changed_target_headquarters=sum(r['changed'] and r['target_is_headquarters'] is True for r in rs),
            missing_label_sites=sum(r['label_missing_from_stock'] for r in rs))
    for k, sig in signatures.items():
        if fingerprint(sig['path']) != sig:
            raise ValueError('Source mutated during mapping')
    for k in ('panel', 'geometry', 'ledger'):
        verified(refs[k])
    result = dict(status='COMPLETE_READ_ONLY_TARGET_MAPPING_NO_TRAINING_NO_RECALL', contract_commit=commit,
        source_refs=refs, implementation=record(Path(__file__)), site_rule=record(ROOT/'scripts/retrieval_site_operational_metric.py'),
        totals=summary(rows), by_fold={str(f): summary([r for r in rows if r['fold'] == f]) for f in (2, 3, 4)},
        scanned_sites=len(sites), hydrated_points=len(points), labels_modified=False, closed_sets_opened=False,
        resources=dict(elapsed_seconds=time.monotonic()-started, peak_rss_bytes=__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss), per_query=rows)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    lines = ['# Cibles d’apprentissage site — mesure du 15 octobre 2026', '', f'Contrat préalable : `{commit}`.',
        'Mapping lecture seule, aucun apprentissage ni recall ; labels originaux inchangés.', '',
        '| Périmètre | Requêtes | Branche 1 | Branche 2 | Branche 3 | Changées | Dont labels sièges | Nouvelles cibles sièges |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in [('Total', result['totals']), *result['by_fold'].items()]:
        lines.append(f'| {name} | {s["queries"]} | {s["branches"].get("1",0)} | {s["branches"].get("2",0)} | {s["branches"].get("3",0)} | {s["changed"]} | {s["changed_original_headquarters"]} | {s["changed_target_headquarters"]} |')
    lines += ['', f'Scan exhaustif : {len(sites)} établissements, {len(points)} points typés.',
        f'Labels absents du stock : {result["totals"]["missing_label_sites"]}.',
        f'Durée {result["resources"]["elapsed_seconds"]:.2f} s ; RSS pic {result["resources"]["peak_rss_bytes"]/1024**2:.1f} Mio.',
        'Branche 3 conserve une cible sans preuve site ; proximité ne certifie pas un occupant fibre.',
        'Sources grandes vérifiées taille/mtime/inode avant/après ; petites autorités SHA256.',
        'Commande : `.venv/bin/python scripts/retrieval_site_target_mapping.py`.']
    RESULT.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ('totals', 'by_fold', 'resources')}), flush=True)


if __name__ == '__main__':
    main()
