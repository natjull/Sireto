"""Replay real optionD models/targets, then publish an isolated result milestone.

Can watch our owned detached runner. Never fits, changes labels or accesses v174.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import step2_option_d as m

b, d, np, xgb = m.b, m.d, m.np, m.xgb


def verify(out):
    r = b.read(out/'result.json')
    plan = m.load_plan(out)
    Xold, y, views, groups, folds, ids, _, _, _, _, _ = d.load(r['smoke'])
    if r['pilot_ids'] != ids.tolist() or len(r['content_features']) != 85 or len(r['stack_features']) != 21:
        raise ValueError('Pilot/features changed')
    old_v2 = b.read(ROOT/'docs/step2_features_v2.json')
    pack = np.load(b.verified(old_v2['matrix']))
    X = pack['features'][[pack['query_ids'].tolist().index(q) for q in ids]]
    CX = X[:, :, m.CONTENT_COLS]
    cohort = r['cohort']
    for ref in cohort['artifacts'].values():
        b.verified(ref)
    labels = b.read(cohort['artifacts']['label_side.json']['path'])
    queries = b.read(cohort['artifacts']['training_queries.json']['path'])
    n = r['generation']['generated']
    query_order = [q['query_id'] for q in queries[:n]]
    if {labels[q]['human_ground_truth_siren'] for q in query_order} & set(groups):
        raise ValueError('External/pilot company overlap')
    matrix = b.read(out/'matrix.json')
    b.verified(matrix['features']); b.verified(matrix['keys'])
    keys = np.load(matrix['keys']['path'], mmap_mode='r')
    target = (keys == np.array([labels[q]['human_ground_truth_siret'] for q in query_order], dtype='S14')[:, None]).astype(np.float32)
    if m.d.digest(query_order) != matrix['query_ids_sha256'] or np.any(target.sum(axis=1) > 1):
        raise ValueError('Target ordering/uniqueness')
    predicted = np.load(b.verified(r['predictions']))
    lookup = {q: i for i, q in enumerate(ids)}
    model_count, audits, score_rows, confidence_rows, target_roles = 0, 0, 0, 0, 0
    all_warnings = []
    for size, c in r['curves'].items():
        fit = c['fit']; count = int(size)
        assert fit['params'] == m.PARAMS and fit['features'] == m.CONTENT
        if fit['target_sha256'] != hashlib.sha256(target[:count].tobytes()).hexdigest():
            raise ValueError('External original targets changed')
        if fit['positive_queries'] != int(target[:count].sum()) or fit['absent_positive_queries'] != int((target[:count].sum(axis=1) == 0).sum()):
            raise ValueError('Missing positives hidden')
        if fit['train_id_set_sha256'] != m.cohort.digest_ids(query_order[:count]):
            raise ValueError('Learning curve membership changed')
        b.verified(fit['model']); all_warnings += fit['warnings']
        model = xgb.XGBRanker(n_jobs=1); model.load_model(fit['model']['path']); model_count += 1
        external = model.predict(CX.reshape(-1, 85)).reshape(len(ids), 100)
        assert np.array_equal(external, predicted['content_'+size])
        assert np.array_equal(external, np.load(b.verified(fit['scores'])))
        SX = m.stack_cube(Xold, external)
        uses, models = {}, {}
        for receipt in c['stack_fit_receipts']:
            train = np.array([lookup[q] for q in receipt['train_queries']])
            assert receipt['target_sha256'] == hashlib.sha256(y[train].reshape(-1).tobytes()).hexdigest()
            assert receipt['params'] == m.PARAMS and receipt['features'] == m.STACK
            b.verified(receipt['model']); all_warnings += receipt['warnings']
            model = xgb.XGBRanker(n_jobs=1); model.load_model(receipt['model']['path']); model_count += 1
            models[receipt['key']] = model
            for use in receipt['uses']:
                test = np.array([lookup[q] for q in use['scored_queries']])
                assert not set(groups[train]) & set(groups[test]); audits += 1
                uses[use['purpose']] = receipt['key'], test
                if use['purpose'].startswith('outer_'):
                    value = model.predict(SX[test].reshape(-1, 21)).reshape(len(test), 100)
                    assert np.array_equal(value, predicted['stack_'+size][test]); score_rows += len(test)
        score_rows += len(ids)

        def score(method, purpose, idx):
            nonlocal audits
            if method == 'content':
                use = next(u for u in c['external_score_uses'] if u['purpose'] == purpose)
                assert use['scored_queries'] == ids[idx].tolist()
                train = np.array([lookup[q] for q in use['train_queries']])
                assert not set(groups[train]) & set(groups[idx]); audits += 1
                return external[idx]
            key, locations = uses[purpose]
            assert np.array_equal(locations, idx)
            return models[key].predict(SX[idx].reshape(-1, 21)).reshape(len(idx), 100)

        for method, conf in c['confidence'].items():
            for receipt in conf['receipts']:
                fold = receipt['fold']; variant = method+'_'+size
                roles = [np.array([lookup[q] for q in receipt[k]]) for k in ('proper_fit_queries', 'isotonic_queries', 'threshold_queries', 'test_queries')]
                proper, iso_rows, limit, test = roles
                for i, a in enumerate(roles):
                    for z in roles[i+1:]:
                        assert not set(groups[a]) & set(groups[z])
                positions = {i: j for j, i in enumerate(proper)}
                meta = np.full((len(proper), 100), np.nan)
                for k in range(3):
                    purpose = f'{variant}_scene_meta_{fold}_{k}'
                    if method == 'content':
                        use = next(u for u in c['external_score_uses'] if u['purpose'] == purpose)
                        idx = np.array([lookup[q] for q in use['scored_queries']])
                    else:
                        idx = uses[purpose][1]
                    meta[[positions[i] for i in idx]] = score(method, purpose, idx)
                assert np.isfinite(meta).all()
                top = b.select_top(meta)[0]
                meta_target = views[proper, top, m.UNION].astype(int)
                assert hashlib.sha256(meta_target.tobytes()).hexdigest() == receipt['scene_meta_target_sha256']
                joined = np.r_[iso_rows, limit, test]
                raw = score(method, f'{variant}_confidence_common_{fold}', joined)
                pi, pl, pt = np.split(raw, [len(iso_rows), len(iso_rows)+len(limit)])
                assert np.array_equal(pt, predicted[f'{method}_{size}_confidence'][test]); confidence_rows += len(test)
                itop = b.select_top(pi)[0]
                iso_target = views[iso_rows, itop, m.UNION].astype(int)
                assert hashlib.sha256(iso_target.tobytes()).hexdigest() == receipt['isotonic_target_sha256']
                ref = next(ref for ref in conf['model_refs'] if ref['path'].endswith('fold'+str(fold)+'.pkl'))
                b.verified(ref)
                with Path(ref['path']).open('rb') as stream:
                    classifier, isotonic = pickle.load(stream)
                lt, lm = b.select_top(pl); tt, tm = b.select_top(pt)
                lp = isotonic.predict(d.probability(classifier, d.scene(pl, X[limit, :, :46])))
                tp = isotonic.predict(d.probability(classifier, d.scene(pt, X[test, :, :46])))
                for name, cal, value in [('margin', lm, tm), ('scene_isotonic', lp, tp)]:
                    threshold, stats = b.threshold_choice(cal, views[limit, lt, m.UNION])
                    assert threshold == receipt['limits'][name]['threshold']
                    if threshold is not None:
                        assert stats['exact_precision']['percent'] >= 98
                    for j, i in enumerate(test):
                        row = conf['per_query'][i]
                        assert row['scores'][name] == float(value[j])
                        assert row['sur'][name] == bool(threshold is not None and value[j] >= threshold)
                target_roles += 1
    # Actual smoke rows are batch invariant; no fabricated or appended positives.
    small = m.OWN/'smoke_verified'
    matching_cells = 0
    if not r['smoke'] and (small/'result.json').exists():
        small_queries = b.read(small/'training_queries.json')
        indices = [query_order.index(q['query_id']) for q in small_queries]
        all_features = np.load(matrix['features']['path'], mmap_mode='r')
        assert np.array_equal(all_features[indices], np.load(small/'features.npy'), equal_nan=True)
        assert np.array_equal(keys[indices], np.load(small/'candidate_keys.npy'))
        matching_cells = len(indices)*100*85
    if not r['smoke']:
        assert r['references']['V142']['exact']['hits'] == 353
        assert r['references']['R3_v2']['exact']['hits'] == 368 and r['references']['R3_v2']['union']['hits'] == 407
        assert r['G0']['v142_presence']['exact']['hits'] == 507
    for ref in plan['official'].values():
        b.verified(ref)
    assert m.inventory() == plan['index_inventory']
    proof = dict(passed=True, model_hashes_and_original_targets=model_count, company_fit_score_audits=audits,
        primary_query_score_vectors_replayed=score_rows, confidence_queries_replayed=confidence_rows,
        union_target_role_records_verified=target_roles, smoke_full_matching_feature_cells=matching_cells,
        warnings=all_warnings, implementation=b.record(Path(__file__)))
    write_path = out/'verification.json'
    m.write(write_path, proof)
    return r, proof


def publish(out, r, proof):
    if r['smoke']:
        raise ValueError('Smoke must never be published as a full experiment')
    root_json = ROOT/'docs/step2_option_d.json'
    if b.read(root_json) != r:
        raise ValueError('Owned publication changed before verification')
    journal = ROOT/'handover.md'
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'handover.md'], cwd=ROOT).strip():
        raise RuntimeError('Results verified; foreign uncommitted handover edits: no automatic commit')
    expected_md = (out/'preregistered.md').read_text()+'\n'+m.markdown(r)
    if (ROOT/'docs/step2_option_d.md').read_text() != expected_md:
        raise RuntimeError('Owned Markdown changed: do not include foreign edits')
    r['verification'] = b.record(out/'verification.json')
    m.write(root_json, r)
    with (ROOT/'docs/step2_option_d.md').open('a') as stream:
        stream.write('\n## Rejeu indépendant\n\n'+json.dumps(proof, ensure_ascii=False)+'\n')
    last = r['curves'][str(max(map(int, r['curves'])))]
    a, z = last['content']['totals']['top1'], last['stack']['totals']['top1']
    g = r['G0']
    line = (f'\n- 2026-10-17 — OptionD COMPLETE vérifiée : protocole `bf26c2c8`, exécution exacte optimisée `053387a8` ; '
        f'G0 pilote{r["queries"]} exact{g["presence"]["exact"]["hits"]}/SIREN{g["presence"]["siren"]["hits"]}/union{g["presence"]["union"]["hits"]}, '
        f'{g["search_seconds_per_query"]:.4f}s/q, intersectionV142{g["overlap_mean"]:.2f}/100 ; '
        f'apprentissage{r["generation"]["generated"]} dossiers sans405 entreprises pilote/jeux fermés, '
        f'G0{r["generation"]["total_worker_seconds"]:.1f}s, '
        f'contenu final{a["exact"]["hits"]} exact/{a["union"]["hits"]} union, empilement{z["exact"]["hits"]}/{z["union"]["hits"]} contre353/391 et368/407. '
        f'Courbe complète et SUR union/exact/couverture/Wilson dans `docs/step2_option_d.{{json,md}}`. '
        f'Rejeu indépendant{proof["primary_query_score_vectors_replayed"]} vecteurs+{proof["confidence_queries_replayed"]} sorties confiance, '
        f'{proof["model_hashes_and_original_targets"]} SHA modèles/cibles, {proof["company_fit_score_audits"]} audits sans entreprise commune. '
        f'Pas de label modifié/positif injecté, pas de v174/VM/web/Gemma/BANO/OSM/activité/push ; V142 et étrangers/pending préservés, aucune promotion. '
        f'Artefacts/logs `H/step2_option_d_20261017/full/`. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/step2_option_d.json`.\n')
    before = journal.read_text()
    with journal.open('a') as stream:
        stream.write(line)
    if journal.read_text() != before+line:
        raise RuntimeError('Concurrent journal append detected; preserve all edits, do not commit')
    paths = ['docs/step2_option_d.json', 'docs/step2_option_d.md', 'handover.md']
    # --only prevents another agent's staged paths from entering our commit.
    subprocess.run(['git', 'add', '--', 'docs/step2_option_d.json'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--check', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', 'feat: measure external G0 content learning and grouped stacking', '--', *paths], cwd=ROOT, check=True)
    return subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', paths[0]], cwd=ROOT, text=True).strip()


def final_report(r, proof, commit):
    g = r['G0']; size = str(max(map(int, r['curves']))); last = r['curves'][size]
    lines = ['## Fait', f'OptionD mesurée ; commit résultats `{commit or "non demandé"}`.',
        '`scripts/step2_option_d.py`, vérificateur et `docs/step2_option_d.{json,md}` ; handover mis à jour.', '',
        '## Preuve', f'G0 présence@100 sur{r["queries"]} : exact{g["presence"]["exact"]["hits"]}, SIREN{g["presence"]["siren"]["hits"]}, union{g["presence"]["union"]["hits"]} (V142 :507/511/511 sur526).',
        f'Recherche{g["search_seconds_per_query"]:.3f}s/q, workers{g["worker_seconds_per_query"]:.3f}s/q, intersection{g["overlap_mean"]:.2f}/100 ; pic{r["peak_rss_bytes"]/1024**3:.2f}Gio.',
        f'Apprentissage réel{size}, dont{last["fit"]["positive_queries"]} positifs présents et{last["fit"]["absent_positive_queries"]} absents conservés. G0 train{r["generation"]["total_worker_seconds"]/3600:.2f}h.',
        f'Rejeu{proof["primary_query_score_vectors_replayed"]} vecteurs+{proof["confidence_queries_replayed"]} sorties confiance, {proof["model_hashes_and_original_targets"]} SHA modèles/cibles, {proof["company_fit_score_audits"]} audits sans entreprise commune.', '',
        '| Exemples | (a) contenu exact/union | (b) empilement exact/union |', '|---:|---:|---:|']
    for n, c in r['curves'].items():
        a, z = c['content']['totals']['top1'], c['stack']['totals']['top1']
        lines.append(f'| {n} | {a["exact"]["hits"]}/{a["union"]["hits"]} | {z["exact"]["hits"]}/{z["union"]["hits"]} |')
    lines += ['Références sur526 : V142353 exact/391 union ; R3v2 368/407. Cinq vues/folds/IC/paires dans le JSON.', '',
        '| SUR au volume final | Nombre | Union | Exact | Couverture |', '|---|---:|---:|---:|---:|']
    def value(metric):
        return 'non définie' if metric is None else f'{metric["hits"]}/{metric["queries"]}={metric["percent"]:.2f}%'
    for method, conf in last['confidence'].items():
        for policy, metrics in conf['methods'].items():
            t = metrics['totals']
            lines.append(f'| {method}/{policy} | {t["count"]} | {value(t["precision"]["union"])} | {value(t["precision"]["exact"])} | {t["rate"]["percent"]:.2f}% |')
    lines += ['', '## Écarts / choix', '85 features génériques, sans11 dépendances natives ; score/rang G0 non utilisés. Coordonnées déjà cachées seulement.',
        'Content externe aux405 entreprises ; empilement OOF et calibration union60/20/20. Égalités départagées dans l’ordre V142.', '',
        '## Reste / risques', 'Pilote consommé, décalage G0→V142 et amont non full-OOF : aucune certification/promotion. Wilson et McNemar descriptifs.',
        'Aucun label modifié/positif injecté, jeu fermé ouvert, v174/VM/web touché, ni push.', '',
        '**Recommandation unique :** '+m.markdown(r).split('### Recommandation unique\n\n')[1].split('\n\n')[0]]
    if len(lines) > 40:
        raise ValueError('Final report longer than40 lines')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=m.OWN/'full')
    parser.add_argument('--watch-pid', type=int)
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    if not out.resolve().is_relative_to(m.OWN.resolve()):
        raise ValueError('Owned output directory required')
    if args.watch_pid:
        try:
            process = m.psutil.Process(args.watch_pid)
            if not any(Path(arg).name == 'step2_option_d.py' for arg in process.cmdline()):
                raise ValueError('Refuse to watch a foreign process')
            print(json.dumps(dict(stage='WATCH_OWNED_RUN', pid=process.pid)), flush=True)
            while process.is_running() and process.status() != m.psutil.STATUS_ZOMBIE:
                time.sleep(60)
        except m.psutil.NoSuchProcess:
            pass
    m.wait()
    guard = m.Guard(out)
    try:
        r, proof = verify(out)
        commit = publish(out, r, proof) if args.commit else None
        status = dict(status='VERIFIED_COMMITTED' if commit else 'VERIFIED', proof=proof, commit=commit)
        (out/'final_report.md').write_text(final_report(r, proof, commit))
        m.write(out/'publication.json', status)
        print(json.dumps(status), flush=True)
    except Exception as error:
        m.write(out/'publication_stop.json', dict(status='STOP', error=str(error)))
        raise
    finally:
        guard.stop.set()


if __name__ == '__main__':
    main()
