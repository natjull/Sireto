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

## Travail en cours
- **Génération Samples Decider (V6 Turbo)** : Utiliser le Ranker V6 pour miner les hard negatives (scène top-50).
- **Performance Méga-Communes** : Le mode `full_insee` ralentit Paris/Lyon. L'Option C (Pre-indexation v7) est le prochain jalon technique.

## Problèmes / Points d'attention
- **Coverage** : Actuellement à ~93%. Le gap restant (7%) est principalement dû à des SIRET réellement absents de SIRENE ou des noms totalement vides.
- **Latence Inférence** : À surveiller sur les méga-communes en prod tant que v7 n'est pas implémentée.

## Artefacts principaux (V6 Canoniques)
| Artefact | Chemin |
|----------|--------|
| Partitions | `data/candidates_v6_all/` |
| Ranker | `models/xgbranker_20260205_195700.json` |
| Meta | `models/xgb_two_stage_meta_20260205_195700.json` |
| GT Data | `data/crm_ok_gt.csv` |

## Prochaines étapes (DS Mode)
1. Lancer la génération des samples **Decider** (V6 Turbo).
2. Entraîner le **Decider** Champion.
3. Implémenter l'**Option C** (Build v7 indexé) pour stabiliser la latence prod.
4. Recalibrer le **Routing (Stage 3)** sur la distribution V6.

---
*Note : Chaque modification de code doit citer le commit GitHub correspondant dans ce document.*
