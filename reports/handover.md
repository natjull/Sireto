## Session 2025-12-27T16:02:08Z — Codex Ctx0
- Phase ciblée : Phase 0 (Diagnostic & Plan)
- Objectif : stabiliser diagnostics + détailler plan + protocoles handover
- Changements :
  - scripts/diagnostic_xgb_routing.py : ajout script diagnostic reproductible
  - scripts/fix_diagnostic_report.py : correction coverage + regen report/plot
  - reports/diagnostic_analysis.json : coverage corrigée + meta ajoutée
  - reports/diagnostic_report.md : régénéré avec coverage cohérente
  - reports/diagnostic_plots.png : régénéré
  - reports/entity_matching_audit.md : plan enrichi (handover + training + Ctx0/1/2/3) + v3 comme base
- Tests/commandes exécutées :
  - python scripts/fix_diagnostic_report.py → OK
- Etat : ✅ terminé
- Prochaines étapes immédiates :
  - Ctx1 : Quick Wins (routing sans SHAP + export features + ranker_fast)
  - Lancer diagnostic reproductible si besoin : python scripts/diagnostic_xgb_routing.py

## Session 2025-12-27T16:29:19Z — Codex Ctx1
- Phase ciblée : Phase 1 (Quick Wins)
- Objectif : routing sans SHAP + export features + ranker_fast
- Note importante : **Toujours inclure les établissements fermés** (`--include-closed-candidates`) pour maximiser le recall (le score ou le tri gérera la préférence ouvert/fermé).
- Changements :
  - scripts/infer_xgb_matcher_topk.py : export des features de routing + option `--export-routing-features` + usage ranker_fast si dispo
  - scripts/train_xgb_matcher_v2.py : entraînement + sauvegarde du ranker_fast (features sémantiques zéro) + métadonnées associées
  - scripts/route_xgb_results.py : routing basé sur colonnes directes + règles adresse‑seule + seuils segmentés
- Tests/commandes exécutées :
  - `python scripts/generate_training_samples_v3.py ...` (small align dataset)
  - `python scripts/train_xgb_matcher_v2.py ...` (ranker_fast training + fix code)
  - `python scripts/infer_xgb_matcher_topk.py ...` (validation on subset)
  - `python scripts/route_xgb_results.py ...` (routing validation)
- Etat : ✅ terminé (Ratio AUTO: 78.4%, REVIEW: 21.6% sur subset test avec établissements fermés)
- Prochaines étapes immédiates :
  - Ctx2 : Sprint ML (Hard Negatives, Calibration, New Features)

## Session 2025-12-27T19:58:00Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML)
- Objectif : Reduire les FP via Hard Negatives + Calibrer les scores
- Changements :
  - `scripts/generate_training_samples_v3.py` : implémenter hard negatives via ranker top-K (et non simple heuristic)
  - `src/xgb_matcher/features.py` : ajouter feature `address_density`, `idf_name`, et **gestion stopwords** ("LES", "DU")
  - `scripts/train_xgb_matcher_v2.py` : activer calibration (Isotonic/Platt)
- Points d'attention (Feedback Phase 1) :
  - **FP à tuer (IDs 1, 53)** : Substring sans sémantique (Sabeco/SABE), "LES" qui gonfle le score (LES DOUCEURS/LES MURIERS).
  - **FN à récupérer (IDs 19, 24, 67, 80, 112)** : Acronymes (JNC/J.N.C, ASL/Assoc), Pluriels (PRIMEUR/S), Stopwords (BANQUE/S).
- Etat : ⚠️ à faire

## Session 2025-12-27T21:11:18Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML)
- Objectif : hard negatives via ranker + nouvelles features + calibration
- Changements :
  - `src/xgb_matcher/features.py` : stopwords nom ("LES", "DU", …), feature `idf_name` (overlap IDF), feature `address_density`, routing tokens ajustés
  - `src/xgb_matcher/candidates.py` : calcul IDF global + densité d’adresse par INSEE/CP + injection dans features
  - `scripts/generate_training_samples_v3.py` : hard negatives basés sur un ranker (auto‑détection `xgbranker_fast_*`) + meta ranker
  - `scripts/train_xgb_matcher_v2.py` : calibration Platt/Isotonic (dev set) + Brier + sauvegarde calibrator
  - `scripts/infer_xgb_matcher_topk.py` : chargement calibrator + scores calibrés
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ⚠️ partiel (code prêt, entraînement + génération samples non exécutés)
- Prochaines étapes immédiates :
  - Générer un dataset v3/v4 avec hard negatives via ranker
  - Réentraîner `train_xgb_matcher_v2.py` avec calibration
  - Ré‑inférer + router pour vérifier baisse FP

## Session 2025-12-27T21:29:13Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML)
- Objectif : compléter les manquants de l’audit Phase 2 (Opus 4.5)
- Changements :
  - `src/xgb_matcher/features.py` : feature `numeric_token_match`, `legal_form_category`, doc + defaults
  - `scripts/infer_xgb_matcher_topk.py` : meta-features (score_gap/ratio/top3_avg/pool_size/has_name_evidence) + export routing
  - `scripts/generate_training_samples_v3.py` : negatives "same address / different name"
  - `scripts/train_xgb_matcher_v2.py` : Recall@10/20 (metric + logs)
  - `scripts/route_xgb_results.py` : gating via score_gap/ratio + has_name_evidence
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ⚠️ partiel (code prêt, entraînement + génération samples non exécutés)
- Prochaines étapes immédiates :
  - Générer samples v4 + réentraîner + ré‑inférer

## Session 2025-12-27T21:37:46Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML)
- Objectif : inclure les établissements fermés en training
- Changements :
  - `scripts/generate_training_samples_v3.py` : `include_closed_establishments=True` par défaut
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Régénérer les samples (avec fermés) puis réentraîner

## Session 2025-12-29T20:52:45Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML)
- Objectif : déblocage génération samples + diagnostics temps
- Changements :
  - `scripts/generate_training_samples_v3.py` : respect `XGB_SEMANTIC_ENABLED`, cap pool via `XGB_MAX_POOL_FOR_SCORING`, logs slow queries, `nan_to_num` avant ranker, GC périodique
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Relancer génération avec `PYTHONUNBUFFERED=1` et `XGB_SAMPLE_SLOW_SEC=30` pour identifier les queries lentes

## Session 2025-12-30T16:46:21Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML) → V4 perf
- Objectif : Partitioning + TF‑IDF prefilter (Phase B/C)
- Changements :
  - `scripts/build_candidate_partitions_v4.py` : création store partitionné `data/candidates_v4/` (insee + cp)
  - `scripts/generate_training_samples_v4.py` : génération samples via partitions + TF‑IDF par commune
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Construire les partitions puis générer les samples v4

## Session 2025-12-30T17:40:31Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML) → V4 perf
- Objectif : Enrichissement UL/PM obligatoire dans le store partitionné
- Changements :
  - `scripts/build_candidate_partitions_v4.py` : jointure UL via DuckDB + enrichissement PM dirigeants via SQLite
  - `requirements.txt` : ajout `duckdb` et `scikit-learn`
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Rebuilder les partitions V4 avec UL/PM

## Session 2025-12-30T21:26:22Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML) → V4 perf
- Objectif : tests de validation samples/recall pour expliquer la dégradation
- Changements :
  - `scripts/evaluate_samples_v4.py` : métriques couverture + hard negatives + recall@K ranker
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Lancer `scripts/evaluate_samples_v4.py` sur samples v4

## Session 2025-12-30T21:41:30Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML) → V4 perf
- Objectif : diagnostic des GT manquants
- Changements :
  - `scripts/analyze_missing_gt_v4.py` : analyse des GT absents (mismatch CRM vs SIRENE, présence dans store)
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Lancer l’analyse des GT manquants sur `samples_v4.parquet`

## Session 2025-12-30T21:46:01Z — Codex Ctx2
- Phase ciblée : Phase 2 (Sprint ML) → V4 perf
- Objectif : corriger le mismatch CP/INSEE dans v4
- Changements :
  - `scripts/generate_training_samples_v4.py` : normalisation CP/INSEE + SIRET (évite mismatch store)
- Tests/commandes exécutées :
  - `date -u +"%Y-%m-%dT%H:%M:%SZ"` → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Régénérer `samples_v4.parquet` puis rerun l’analyse des GT manquants

## Session 2025-12-31T00:00:00Z — Codex Ctx2 (Ctx3 handover)
- Phase ciblée : Phase 3 (SOTA / 2‑étages)
- Objectif : préparer passage vers pipeline ranker + decider (SOTA)
- Etat global :
  - ✅ V4 partitions + TF‑IDF prefilter + UL/PM enrichis
  - ✅ Analyse GT manquants (cause principale: CRM_LOC_MISMATCH attendu)
  - ✅ Ranker recall@20 ~99% (sur samples v4)
  - ❌ Decider 2‑étages pas encore implémenté
- Changements clés apportés par Ctx2 :
  - `scripts/build_candidate_partitions_v4.py` : build partitions + UL (DuckDB) + PM dirigeants (SQLite)
  - `scripts/generate_training_samples_v4.py` : TF‑IDF blocking par commune + partitions V4 (output compatible aval)
  - `scripts/evaluate_samples_v4.py` : audit samples + recall@K ranker
  - `scripts/analyze_missing_gt_v4.py` : diagnostic GT manquants
  - `scripts/generate_training_samples_v3.py` : fixes perf (semantic gating, cap pool, slow logs, GC)
  - `scripts/train_xgb_matcher_v2.py` : calibration + recall@10/20
  - `scripts/infer_xgb_matcher_topk.py` : meta‑features + routing evidence
  - `src/xgb_matcher/features.py` : numeric_token_match + legal_form_category + idf_name + address_density + stopwords
- Périmètre V4 actuel :
  - Partitioning par INSEE/CP → coverage ~71% (reste CRM_LOC_MISMATCH ~79% des manquants)
  - GT_NOT_IN_SIRENE ~16% (data issue)
  - CP/INSEE mismatch corrigé via normalisation
- Tests / scripts disponibles :
  - `scripts/evaluate_samples_v4.py --samples data/samples_v4.parquet`
  - `scripts/analyze_missing_gt_v4.py --samples data/samples_v4.parquet --partitions-dir data/candidates_v4`
  - `scripts/diagnostic_xgb_routing.py` pour risk‑coverage comparatif
- Commandes usuelles (à relancer si besoin) :
  - Build partitions:
    `python scripts/build_candidate_partitions_v4.py --training-csv data/entrainements.csv --parquet-path data/StockEtablissement_utf8.parquet --ul-path data/StockUniteLegale_utf8.parquet --harvest-db data/harvest.db --output-dir data/candidates_v4 --code-batch 200`
  - Samples v4:
    `XGB_SEMANTIC_ENABLED=0 python scripts/generate_training_samples_v4.py --output data/samples_v4.parquet --partitions-dir data/candidates_v4 --prefilter-k 500 --max-negatives 50`
  - Training:
    `XGB_SEMANTIC_ENABLED=0 python scripts/train_xgb_matcher_v2.py --samples data/samples_v4.parquet --calibration isotonic`
  - Inference:
    `python scripts/infer_xgb_matcher_topk.py --crm-path data/testcrm/data_56_subset_corbas_decines.csv --output-path reports/xgb_infer_v4.csv --top-k 5 --include-closed-candidates`
  - Routing:
    `python scripts/route_xgb_results.py --input-path reports/xgb_infer_v4.csv --output-path reports/routed_v4.csv`
- Prochaines étapes Ctx3 (SOTA 2‑étages) :
  1) Implémenter pipeline **ranker_fast retrieval → decider calibré** (nouveaux scripts train_ranker/train_decider).
  2) Multi‑blocking (TF‑IDF + address hash + numeric) pour recovery du CRM_LOC_MISMATCH si souhaité.
  3) Décision finale : routing strict (gap, evidence, density) + web/Places uniquement sur REVIEW.
  4) Ré‑évaluation risk‑coverage + taux AUTO (zéro FP).
