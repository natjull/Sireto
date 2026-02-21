# SIRETO Handover - 21 Février 2026

## État des Lieux
Nous avons finalisé l'alignement SSOT (Single Source of Truth) complet du pipeline XGBoost, en résolvant les problèmes de fragmentation et de skew train/serve. Le socle est maintenant prêt pour une production industrielle.

## Actions terminées dans cette fenêtre
- **Unification du Retrieval** : Toutes les briques de recherche (train + serve) passent par `src/xgb_matcher/retrieval.py`.
- **Bascule Partitions V6/V7** : Partitions reconstruites avec types `string` forcés. Résolution des bugs Corse (2A/2B) et zéros initiaux. Option Dense traitée (`precompute_embeddings.py`).
- **Validation Partitions** : Création de `scripts/validate_partitions_v5.py` (stop-the-line check).
- **Mode Turbo Samples & Skew Fix** : Analyse approfondie du Train/Serve Skew causé par l'injection forcée du Ground Truth lors de la génération des données Decider.
- **Entraînement Ranker V7 (Étage 1)** : Ranker rapide entraîné avec Hit@50 = 100% (sur les cas qui passent le prisme TF-IDF).
- **Entraînement Semantic Re-Ranker V7 (Étage 2)** : Abandon de l'objectif binaire (`binary:logistic`) pour le Decider au profit d'un pur Re-Ranker sémantique (`rank:ndcg`). Hit@1 boosté à ~80%.
- **Lancement Inférence Globale** : Inférence V7 lancée sur `crm_ok_gt.csv` pour exporter les features de routage et préparer l'entraînement de l'Étage 3.

## Fichiers modifiés
- `src/xgb_matcher/retrieval.py` : Cœur du retrieval unifié.
- `src/xgb_matcher/retrieval_config.py` : Configuration centralisée et signature (hash).
- `src/xgb_matcher/partitioned_store.py` : Store robuste aux types string.
- `scripts/generate_training_samples_v5fast.py` : Générateur Turbo aligné SSOT.
- `scripts/build_candidate_partitions_v5.py` : Builder v6 string-safe.
- `SSOT.md` / `DECISIONS.md` : Documentation technique et historique de design à jour.

## Audit/Fix branche retrieval hybride (réalisé)
- **Merge main <- audit retrieval hybride** : intégration de la couche dense FAISS + cache TF-IDF persistant + instrumentation timing, avec compatibilité descendante conservée. *(commit GitHub: `9ab297e`)*
- **Ablation dense-only corrigée** : ajout du flag `sparse_retrieval_enabled` dans `RetrievalConfigV1` pour permettre un vrai mode dense-only (sans branche TF-IDF), et propagation dans la signature de config. *(commit GitHub: `35fc3a3`)*
- **Retrieval unifié ajusté** : la branche sparse est maintenant conditionnelle (`config.sparse_retrieval_enabled`), avec compatibilité descendante conservée sur `pool_sizes["tfidf"]` et nouveau motif de perte `PRUNED_BY_PREFILTER` quand sparse est désactivé. *(commit GitHub: `35fc3a3`)*
- **Benchmark retrieval sécurisé** : `scripts/benchmark_retrieval.py` n’exécute plus dense/hybrid sans dense store pour éviter des métriques trompeuses; `dense_only` est désormais réellement dense. *(commit GitHub: `35fc3a3`)*
- **Test de non-régression config** : ajout de `tests/test_retrieval_config_sparse.py` (round-trip + hash signature). *(commit GitHub: `35fc3a3`)*
- **Gouvernance technique** : activation progressive du dense (opt-in) et conservation du sparse comme baseline stable. *(référence design: `DECISIONS.md`, 2026-02-08)*

## Travail en cours
- **Extraction des Features Top-K (Inférence V7 en cours)** : `infer_xgb_two_stage.py` tourne sur les 17K requêtes pour encoder l'Étage 2 (Sémantique BERT MPS) et exporter `topk_v7_for_risk.csv`.
- **Préparation de l'Étage 3 (Le Juge)** : Analyse de `scripts/train_routing_risk_model.py` validée pour entraîner un Risk Model XGBoost Isotonique ultra-fiable centré uniquement sur le candidat Top-1.

## Problèmes / Points d'attention
- **Coverage** : Actuellement à ~93%. Le gap restant (7%) est principalement dû à des SIRET réellement absents de SIRENE ou des noms totalement vides.
- **Latence Inférence** : Le mode `full_insee` est coûteux sur Paris/Lyon sans indexation.

## Artefacts principaux (V7 Actifs)
| Artefact | Chemin |
|----------|--------|
| Partitions | `data/candidates_v7_all/` |
| Ranker (Fast) | `models/xgbranker_fast_20260221_224040.json` |
| Ranker (Semantic) | `models/xgbranker_20260221_224040.json` (Utilisé en "Decider_model") |
| Meta | `models/xgb_two_stage_meta_20260221_224040.json` |
| Re-Ranked Demos | `data/topk_v7_for_risk.csv` (en cours de gen.) |

## Prochaines étapes (DS Mode)
1. Attendre la fin de l'inférence dense de `topk_v7_for_risk.csv`.
2. Générer le dataset Risk via `build_routing_eval_dataset.py`.
3. Entraîner le **Risk Model (Stage 3 XGBoost Calibration)** sur le candidat Top-1.
4. Évaluer le taux de routage AUTO final via la Risk Coverage Curve.

---
*Note : Chaque modification de code doit citer le commit GitHub correspondant dans ce document.*
