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

## Session 2025-12-31T07:55:19Z — Codex Ctx3
- Phase ciblée : Phase 3 (SOTA / 2‑étages)
- Objectif : pipeline ranker+decider + multi‑blocking + silver labels (Places)
- Changements :
  - src/xgb_matcher/blocking.py : utilitaires multi‑blocking (TF‑IDF, address hash, numeric, densité)
  - src/xgb_matcher/partitioned_store.py : store partitionné (INSEE/CP/DEP)
  - scripts/train_xgb_ranker.py : entraînement ranker stage‑1 + meta two‑stage
  - scripts/train_xgb_decider.py : entraînement decider + calibration + top‑K optionnel
  - scripts/infer_xgb_two_stage.py : inférence 2‑étages + multi‑blocking via partitions
  - scripts/build_silver_labels_from_places.py : extraction silver labels Places
- Tests/commandes exécutées :
  - date -u +"%Y-%m-%dT%H:%M:%SZ" → OK
- Etat :
  - ⚠️ partiel (code ajouté, entraînement/inférence non exécutés)
- Prochaines étapes immédiates :
  - Générer samples v4 puis entraîner ranker+decider (scripts/train_xgb_ranker.py + scripts/train_xgb_decider.py)
  - Lancer infer 2‑étages (scripts/infer_xgb_two_stage.py) puis router (scripts/route_xgb_results.py)
  - Extraire silver labels Places via scripts/build_silver_labels_from_places.py

## Session 2025-12-31T12:26:52Z — Codex Ctx3
- Phase ciblée : Phase 3 (SOTA / 2‑étages)
- Objectif : intégrer sémantique fine‑tunée + gating lexical + retrieval sémantique
- Changements :
  - src/xgb_matcher/semantic.py : modèle sémantique fine‑tuné par défaut si dispo
  - src/xgb_matcher/features.py : gating lexical sémantique (jaro/token overlap)
  - scripts/infer_xgb_matcher_topk.py : gating sémantique lors du scoring
  - scripts/infer_xgb_two_stage.py : retrieval sémantique (dept pool) + gating sémantique
  - scripts/route_xgb_results.py : blocage AUTO en “semantic‑only” (jaro<0.5 & tok<0.2)
  - src/xgb_matcher/__init__.py : export semantic_gate_allows
- Tests/commandes exécutées :
  - date -u +"%Y-%m-%dT%H:%M:%SZ" → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Entraîner avec sémantique ON si souhaité (XGB_SEMANTIC_ENABLED=1)
  - Activer retrieval sémantique via --semantic-retrieval-k

## Session 2025-12-31T12:36:07Z — Codex Ctx3
- Phase ciblée : Phase 3 (SOTA / 2‑étages)
- Objectif : corriger feedback (semantic retrieval + tests + train/serve skew)
- Changements :
  - scripts/infer_xgb_two_stage.py : semantic retrieval effectif + warnings pool_mode/semantic
  - scripts/generate_training_samples_v4.py : colonne semantic_enabled (pour alignement)
  - scripts/train_xgb_ranker.py : warning + meta semantic_enabled_samples
  - scripts/train_xgb_decider.py : warning + meta semantic_enabled_samples
  - scripts/infer_xgb_matcher_topk.py : warning mismatch semantic
  - tests/test_semantic_gating.py : tests unitaires gating
  - scripts/check_semantic_model.py : smoke test chargement modèle fine‑tuné
- Tests/commandes exécutées :
  - date -u +"%Y-%m-%dT%H:%M:%SZ" → OK
- Etat :
  - ✅ terminé
- Prochaines étapes immédiates :
  - Régénérer samples v4 avec XGB_SEMANTIC_ENABLED=1
  - Entraîner ranker+decider puis inférer

## Session 2026-01-03T17:30:00Z — Codex Ctx3
- Phase ciblée : Phase 3 (SOTA / 2‑étages) — Audit “obvious” + fixes retrieval (TF‑IDF) + validation locale
- Objectif : résoudre les faux négatifs évidents avant routing (ATOUT GAZ, L2G, TIMCOD, Couleur Primeur, JNC, SABECO).
- Duo ranker/decider utilisé (meta) :
  - Meta : `models/xgb_two_stage_meta_20260103_132351.json`
  - Ranker_fast : `models/xgbranker_fast_20260103_132351.json`
  - Decider : `models/xgb_decider_20260103_132351.json`
  - Calibrator : `models/xgb_decider_calibrator_isotonic_20260103_132351.pkl`
- Performances XGBoost (issues du meta 20260103_132351) :
  - Ranker_fast (test) : hit@1=0.8823, hit@3=0.9517, hit@5=0.9681, recall@10=0.9827, recall@20=0.9936, MRR=0.9208
  - Ranker_fast (dev) : hit@1=0.8893, hit@3=0.9606, hit@5=0.9723, recall@10=0.9872, recall@20=0.9923, MRR=0.9284
  - Decider calibré (test) : AUC=0.99117, AP=0.87339, Brier=0.00553, hit@1=0.9020, hit@3=0.9653, hit@5=0.9777
  - Decider calibré (dev) : AUC=0.99206, AP=0.88261, Brier=0.00527, hit@1=0.9068, hit@3=0.9693, hit@5=0.9805
- Changements (code) :
  - src/xgb_matcher/naming.py :
    - ajout `candidate_tfidf_text()` (concat + dédup bag‑of‑names, cache `_xgb_cached_tfidf_text`).
  - src/xgb_matcher/blocking.py :
    - TF‑IDF basé sur **bag‑of‑names** (au lieu de `primary_name`).
    - normalisation TF‑IDF renforcée (`normalize_text_for_tfidf`) : suppression ponctuation, acronymes compacts (J.N.C → JNC), expansion légère singulier/pluriel.
    - fallback TF‑IDF **char‑ngram** (3‑5) activé **uniquement** si word‑TF‑IDF retourne nnz=0.
  - scripts/infer_xgb_two_stage.py :
    - passage des `cand_names` vers prefilter pour fallback char‑ngram.
    - cache TF‑IDF enrichi (names).
  - scripts/generate_training_samples_v4.py :
    - alignement de la normalisation TF‑IDF (utilise `normalize_text_for_tfidf`).
- Commits poussés (origin/main) :
  - 3cc4bc6 — Improve TF‑IDF retrieval with bag‑of‑names.
  - d99c2f7 — Use TF‑IDF fallback in two‑stage inference.
  - 6a49a33 — Align training TF‑IDF normalization.
- Tests/commandes exécutées :
  - `XGB_SEMANTIC_ENABLED=1 python scripts/infer_xgb_two_stage.py --crm-path data/testcrm/data_56_subset_corbas_decines.csv --partitions-dir data/candidates_v4_fixed --output-path reports/xgb_two_stage_topk_56_with_closed_sem_fixed.csv --top-k 5 --meta-path models/xgb_two_stage_meta_20260103_132351.json`
  - `XGB_SEMANTIC_ENABLED=1 ... --output-path reports/xgb_two_stage_topk_56_with_closed_sem_fixed2.csv`
  - `XGB_SEMANTIC_ENABLED=1 ... --output-path reports/xgb_two_stage_topk_56_with_closed_sem_fixed3.csv`
- Résultats clés (top‑5) :
  - ATOUT GAZ (crm_id 54) → **SIRET 53011730800029 rank 1**.
  - L2G (crm_id 94) → **SIRET 44888664800055 rank 1**.
  - TIMCOD (crm_id 115) → **SIRET 48048580400037 rank 1**.
  - COULEUR PRIMEUR (crm_id 24) → **SIRET 42037022300034 rank 1** (pluriels corrigés).
  - JNC (crm_id 80) → **SIRET 79341104200013 rank 1** (acronymes corrigés).
  - SABECO (crm_id 1) → **SIRET 79422623300037 rank 2** (rank 1 = SABEXTRA, decider préfère un homonyme).
- Observations / alertes :
  - Le modèle sémantique affiche un warning de tokenizer (regex Mistral) mais l’inférence fonctionne.
  - SABECO reste un cas où le **ranking** (decider) préfère un homonyme ; corriger nécessiterait un retrain ou un tie‑breaker dédié (non fait).
  - Aucun retrain effectué sur ces changements (seulement retrieval).
- Prochaines étapes immédiates :
  - Poursuivre vers le **routing** (ROUTING XGBoost v1.0) à partir de `reports/xgb_two_stage_topk_56_with_closed_sem_fixed3.csv`.

## Session 2026-01-03T18:45:00Z — Codex Ctx4 (handover)
- Phase ciblée : Phase 4 (SOTA Routing & Places‑as‑CRM)
- Objectif : aligner la doc d'architecture avec "AUTO/REVIEW uniquement avant Places" + préparer calibration automatique.
- Changements (doc/archi) :
  - `reports/entity_matching_audit.md` :
    - NO_MATCH uniquement **après** Places (AUTO/REVIEW pré‑Places)
    - ajout "Places‑as‑CRM (decider identique)" + calibration automatique Places (`score_min/gap_min`)
    - ajout "Calibration AUTO vs REVIEW (automatique)"
    - commands Phase 4: `--top-k 20` + chemins modèles explicites
  - `AGENTS.md` :
    - routing v1.0 = AUTO/REVIEW only
    - diagramme V7 + texte: NO_MATCH seulement après Places/WEB
    - note legacy ajoutée pour V6
  - `README.md` : note Phase 4 (AUTO/REVIEW pré‑Places)
  - `docs/PRODUCTION_DEPLOYMENT_PROMPT.md` : outputs/metrics ajustés (NO_MATCH post‑Places)
  - `docs/diagrams/pipe_v6_flowchart.mmd` : sortie REVIEW (NO_MATCH après Places)
  - `docs/diagrams/pipe_v6_flowchart.svg` : régénéré (mermaid-cli)
- État : ✅ doc alignée, code non touché
- Prochaines étapes immédiates (code) :
  1) Implémenter `scripts/calibrate_routing_thresholds.py` (calibration AUTO/REVIEW).
  2) Wire `scripts/route_xgb_results.py` pour charger `configs/routing_thresholds.yaml` + AUTO/REVIEW only.
  3) Implémenter `scripts/calibrate_places_thresholds.py` + `scripts/evaluate_places_matching.py`.
  4) Ajouter `address_close()` + seuils Places dans `src/pipe_v6/places_validator.py`.
  5) Mettre à jour `src/pipe_v6/places_orchestrator.py` (pool recall@20 + Places‑as‑CRM).
- Notes :
  - NO_MATCH **uniquement** après Places (pas de NO_MATCH pré‑Places).
  - Modèles decider **chemins explicites** (pas d'auto‑latest). Référence : `models/xgb_decider_20260103_132351.json` + calibrator associé.

## Session 2026-01-04T00:15:00Z — Codex Ctx5 (Phase 4 Implementation)
- Phase ciblée : Phase 4 (SOTA Routing + Places‑as‑CRM) — **Implémentation complète**
- Objectif : implémenter le routing cost-aware + Places‑as‑CRM avec le même decider XGB
- Duo modèles (chemins explicites, pas d'auto‑latest) :
  - Decider : `models/xgb_decider_20260103_132351.json`
  - Calibrator : `models/xgb_decider_calibrator_isotonic_20260103_132351.pkl`
  - Meta : `models/xgb_two_stage_meta_20260103_132351.json`
- Changements (code) :
  - **`configs/routing_thresholds.yaml`** (CRÉÉ) :
    - Seuils segmentés (unique_name_full_addr, common_name_full_addr, short_name, etc.)
    - Règles de certitude (perfect_match, identical_name, model_certainty, contains_match)
    - Résolution same‑SIREN automatique
    - Règles de blocage (no_lexical_evidence, semantic_only, address_only, high_density, weak_gap)
    - Règles de promotion (strong_etab, contains, pm_dirigeant, token_overlap)
    - Budget modes (aggressive/normal/permissive)
  - **`configs/places_thresholds.yaml`** (CRÉÉ) :
    - Chemins modèles explicites (decider, calibrator, meta)
    - Pool config (xgb_topk: 20, arm_a, arm_b)
    - Gate config (min_addr_overlap, min_name_semantic)
    - Seuils promotion Places (score_min: 0.97, gap_min: 0.05)
    - Config address_close (addr_jaro_min, street_number_diff_max, distance_max_m)
  - **`scripts/route_xgb_results.py`** (RÉÉCRIT) :
    - `route_cost_aware()` : routing Phase 4 avec certainty → same‑SIREN → blocking → segment → promotion → gap → threshold
    - `is_absolute_certainty()` : 4 règles FP‑impossible
    - `resolve_same_siren()` : résolution automatique intra‑SIREN (préférence OUVERT)
    - `check_blocking_rules()` : 6 règles → force REVIEW
    - `check_promotion_rules()` : 4 règles → force AUTO
    - `get_segment_config()` : seuils variables par segment
    - `apply_budget_mode()` : multiplicateurs aggressive/normal/permissive
    - `RoutingConfig` + `RoutingMetrics` dataclasses
  - **`src/pipe_v6/places_validator.py`** (MIS À JOUR) :
    - `address_close()` : validation combinée postcode + Jaro + street_number_diff + geo_distance
  - **`src/pipe_v6/places_xgb_rescorer.py`** (RÉÉCRIT) :
    - Chemins modèles explicites (DEFAULT_DECIDER_MODEL, DEFAULT_CALIBRATOR_PATH, DEFAULT_META_PATH)
    - Même gating sémantique que l'inférence (`semantic_gate_allows()`)
    - Classes IsotonicCalibrator / SigmoidCalibrator (pickle compat)
    - `score_candidates()` / `score_candidates_with_features()`
  - **`src/pipe_v6/places_orchestrator.py`** (MIS À JOUR) :
    - Pool = recall@20 + arm_a + arm_b
    - Mini‑gate CRM ↔ Places
    - Integration `address_close()` avant promotion MATCH_PLACES
    - REVIEW devient NO_MATCH après Places
  - **`scripts/calibrate_routing_thresholds.py`** (CRÉÉ) :
    - Calibration seuils AUTO/REVIEW par segment avec target FP rate
    - Génère YAML + rapport JSON
  - **`scripts/calibrate_places_thresholds.py`** (CRÉÉ) :
    - Calibration seuils Places (score_min, gap_min) + address_close params
  - **`scripts/evaluate_routing.py`** (CRÉÉ) :
    - Évaluation routing vs ground truth (precision AUTO, FP/FN rate, cost analysis)
  - **`scripts/evaluate_places_matching.py`** (CRÉÉ) :
    - Évaluation Places matching (MATCH_PLACES precision, pool coverage, source contribution)
  - **`scripts/simulate_places_costs.py`** (EXISTANT) :
    - Simulation coûts API Places selon budget mode
- Tests exécutés (Corbas/Decines, 134 CRM) :
  - `python scripts/route_xgb_results.py --input-path reports/xgb_two_stage_topk_56_with_closed_sem_fixed3.csv --output-path reports/phase4_routed_test.csv --thresholds configs/routing_thresholds.yaml --budget-mode normal`
  - Résultats :
    - AUTO rate : **49.3%** (66/134)
    - AUTO_CERTAIN : 56 (perfect_match: 43, identical_name: 9, contains_match: 3, model_certainty: 1)
    - AUTO_SAME_SIREN : 5
    - AUTO (promoted) : 5 (contains: 4, token_overlap: 1)
    - REVIEW rate : **50.7%** (68/134)
    - Blocked by : address_only (7), high_density (6), no_lexical_evidence (5), weak_gap (5), weak_ratio (4), no_name_evidence (1)
    - Review reason : below_threshold (37), low_gap (1)
    - Estimated Places API calls : 68
    - Estimated cost : $0.07
  - Segment breakdown :
    - short_name : 22 total, 2 AUTO (9.1%)
    - common_name_full_addr : 19 total, 3 AUTO (15.8%)
    - common_name_partial_addr : 2 total, 0 AUTO (0.0%)
- Bugs corrigés :
  - `ValueError: The truth value of a Series is ambiguous` → fix `if resolved_row is not None`
- État : ✅ code complet, tests Corbas/Decines passés
- Prochaines étapes :
  1) Calibrer les seuils sur samples d'entraînement labelisés (ground truth)
  2) Évaluer FP rate / precision avec `evaluate_routing.py` sur données labelisées
  3) Tester le mode Places (`--places-mode`) sur cas REVIEW
  4) Affiner `address_close()` params via `calibrate_places_thresholds.py`
  5) Production : `--budget-mode normal` recommandé pour équilibre coût/qualité

## Session 2026-01-04T11:18:30Z — Codex Ctx6 (Phase 4 Audit + Tests)
- Phase ciblée : Phase 4 (Audit & validation)
- Objectif : auditer l’implémentation Phase 4, corriger les incohérences, exécuter les tests DS sur data d’entraînement
- Changements :
  - `scripts/route_xgb_results.py` : applique `resolved_siret` (same‑SIREN) dans la sortie + champs `resolved_*`
  - `src/pipe_v6/places_orchestrator.py` :
    - `REVIEW_PLACES_FAILED` si API Places en erreur
    - application `ratio_min` + seuils par segment (adresse complète/incomplète)
    - helper `_has_complete_address`
  - `src/pipe_v6/places_validator.py` : ajout check `ratio_min` + evidence `ratio_after`
  - `src/pipe_v6/config.py` : ajout `places_ratio_min`
  - `scripts/infer_xgb_two_stage.py` : auto‑détection séparateur CSV (virgule/point‑virgule)
  - `scripts/evaluate_samples_v4.py` : options `--ranker-model` + `--meta-path`
  - `scripts/evaluate_routing.py` : fix JSON serialization (numpy types)
  - `scripts/evaluate_decider_on_samples.py` : nouveau script d’évaluation decider (AUC/PR/Brier/ECE + hit@k + thresholds)
- Tests/commandes exécutées :
  - `python scripts/evaluate_samples_v4.py --samples data/samples_v4_with_ranker.parquet --ranker-model models/xgbranker_fast_20260103_132351.json --meta-path models/xgb_two_stage_meta_20260103_132351.json`
  - `python scripts/evaluate_decider_on_samples.py --samples data/samples_v4_with_ranker.parquet --model models/xgb_decider_20260103_132351.json --calibrator models/xgb_decider_calibrator_isotonic_20260103_132351.pkl --meta models/xgb_two_stage_meta_20260103_132351.json --output reports/decider_eval.json`
  - `XGB_SEMANTIC_ENABLED=1 python scripts/infer_xgb_two_stage.py --crm-path data/splits/test.csv --output-path reports/xgb_two_stage_topk_test.csv --top-k 20 --partitions-dir data/candidates_v4 --pool-mode insee_then_postcode --prefilter-k 500 --dept-prefilter-k 200 --max-dept-candidates 50000 --meta-path models/xgb_two_stage_meta_20260103_132351.json --ranker-fast-model models/xgbranker_fast_20260103_132351.json --decider-model models/xgb_decider_20260103_132351.json --calibrator-path models/xgb_decider_calibrator_isotonic_20260103_132351.pkl`
  - `python scripts/route_xgb_results.py --input-path reports/xgb_two_stage_topk_test.csv --output-path reports/routed_phase4_test.csv --thresholds configs/routing_thresholds.yaml --budget-mode normal`
  - `python scripts/evaluate_routing.py --routed-path reports/routed_phase4_test.csv --ground-truth-path data/splits/test.csv --output-path reports/routing_evaluation_test.json`
  - `python scripts/calibrate_routing_thresholds.py --inference-path reports/xgb_two_stage_topk_test.csv --ground-truth-path data/splits/test.csv --output-path reports/routing_thresholds_calibrated_test.yaml --report-path reports/routing_calibration_test.json --target-fp-rate 0.001`
- Etat : ⚠️ partiel (Places matching non évalué faute de clé/API; routing precision très basse vs objectif 0 FP)
- Prochaines étapes immédiates :
  - Inspecter `reports/routing_evaluation_test.json` (AUTO precision ~75%) + renforcer règles “certainty”
  - Re‑calibrer routing/thresholds sur dataset élargi + re‑router
  - Lancer évaluation Places si clé Serper dispo
