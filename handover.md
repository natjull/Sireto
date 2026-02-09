# SIRETO Handover - 5 Février 2026

## État des Lieux
Nous avons finalisé l'alignement SSOT (Single Source of Truth) complet du pipeline XGBoost, en résolvant les problèmes de fragmentation et de skew train/serve. Le socle est maintenant prêt pour une production industrielle.

## Actions terminées dans cette fenêtre
- **Unification du Retrieval** : Toutes les briques de recherche (train + serve) passent par `src/xgb_matcher/retrieval.py`.
- **Bascule Partitions V6** : Partitions reconstruites avec types `string` forcés. Résolution des bugs Corse (2A/2B) et zéros initiaux.
- **Validation Partitions** : Création de `scripts/validate_partitions_v5.py` (stop-the-line check).
- **Mode Turbo Samples** : Accélération massive de la génération via sampling basé sur les rangs TF-IDF (calcul features lourdes seulement sur 51 candidats).
- **Policy Mega-Communes** : Passage à `full_insee` pour maximiser le coverage (Nice, Nantes, etc. ne sont plus bridées par le CP CRM).
- **Entraînement Ranker V6** : Premier ranker canonique entraîné avec Hit@1 (test) = 75.6% et Recall@50 = 99.96%.
- **Correctif GT** : `data/crm_ok_gt.csv` nettoyé (INSEE/CP SIRENE injectés pour les cas mismatch non-méga; SIRET absents supprimés).

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
- **Génération Samples Decider (V7 Turbo)** : Utiliser le Ranker V6 pour miner les hard negatives (scène top-50) sur partitions V7.
- **Câblage dense dans le générateur de samples** : ajout des flags CLI (`--enable-dense-retrieval`, `--dense-top-k`, `--dense-store-dir`) et passage du `PartitionEmbeddingStore` à `build_candidate_pool` pour exploiter les embeddings pré-calculés pendant la génération. *(commit GitHub: `a309a7c`)*
- **Bascule V7 + manifest O(1)** : `PartitionedCandidateStore` charge `manifest/insee_counts.parquet` pour éviter `count_rows` sur méga-communes, avec fallback V6 inchangé; defaults de `generate_training_samples_v5fast.py`, `benchmark_retrieval.py`, `precompute_embeddings.py`, `infer_xgb_two_stage.py` et `InferenceProfile` alignés sur `data/candidates_v7_all`. *(commit GitHub: `a309a7c`)*
- **Nettoyage des références V6** : suppression des chemins `candidates_v6_all` dans les scripts Python versionnés de support (`validate_partitions_v5.py`, `build_candidate_partitions_v5.py`) et neutralisation du commentaire V6 dans `blocking.py`. *(commit GitHub: `a309a7c`)*
- **Option A embeddings (prêt à lancer)** : `scripts/precompute_embeddings.py` enrichi avec priorisation manifeste (`--sort-by-size`, `--mega-first`, `--mega-threshold`, `--min-rows`, `--max-partitions`), mode `--dry-run` et logs cumulés pour pilotage des runs longs. *(commit GitHub: `LOCAL_PENDING_PUSH`)*

## Problèmes / Points d'attention
- **Coverage** : Actuellement à ~93%. Le gap restant (7%) est principalement dû à des SIRET réellement absents de SIRENE ou des noms totalement vides.
- **Latence Inférence** : Le mode `full_insee` est coûteux sur Paris/Lyon sans indexation.

## Artefacts principaux (V7 en cours d'adoption)
| Artefact | Chemin |
|----------|--------|
| Partitions | `data/candidates_v7_all/` |
| Ranker | `models/xgbranker_20260205_195700.json` |
| Meta | `models/xgb_two_stage_meta_20260205_195700.json` |
| GT Data | `data/crm_ok_gt.csv` |

## Prochaines étapes (DS Mode)
1. Lancer la génération des samples **Decider** (V7 Turbo).
2. Entraîner le **Decider** Champion.
3. Recalibrer le **Routing (Stage 3)** sur la distribution V7.

---
*Note : Chaque modification de code doit citer le commit GitHub correspondant dans ce document.*
