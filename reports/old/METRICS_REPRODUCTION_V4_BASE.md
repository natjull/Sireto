# Audit de Reproduction : Métriques Base V4 (Stage 1 & 2)
**Date :** Dimanche 11 Janvier 2026
**Statut :** ✅ Validé - Base Saine pour Phase Routing

## 1. Contexte
Ce document certifie la reproduction exacte des métriques du meta-modèle de référence après archivage des données obsolètes. L'objectif est de garantir que le pipeline d'inférence et d'évaluation part d'un état connu et stable ("Canonical Baseline").

## 2. Artefacts Canoniques
Les fichiers suivants constituent la source de vérité pour toute future expérimentation.

| Composant | Fichier / Path | Timestamp / Version |
|-----------|----------------|---------------------|
| **Meta-Model (JSON)** | `models/xgb_two_stage_meta_20260103_132351.json` | 20260103_132351 |
| **Ranker (Stage 1)** | `models/xgbranker_20260103_132351.json` | 20260103_132351 |
| **Ranker Fast (No Sem)** | `models/xgbranker_fast_20260103_132351.json` | 20260103_132351 |
| **Decider (Stage 2)** | `models/xgb_decider_20260103_132351.json` | 20260103_132351 |
| **Calibrator (Isotonic)** | `models/xgb_decider_calibrator_isotonic_20260103_132351.pkl` | 20260103_132351 |
| **Dataset (Parquet)** | `data/samples_v4_with_ranker.parquet` | Canonical V4 (13602 queries) |
| **Candidates (Active)** | `data/candidates_v4_active/` | Open-only SIRENE |

## 3. Métriques de Reproduction (Split: TEST)

### Stage 1 : Ranker (Retrieval)
La reproduction est **100% identique** aux valeurs enregistrées dans le meta-modèle.

| Métrique | Meta (Référence) | Reproduit | Statut |
|----------|-----------------|-----------|--------|
| **Hit@1** | 0.9024167807 | 0.9024167807 | ✅ Identique |
| **MRR** | 0.9379803469 | 0.9379803469 | ✅ Identique |
| **Hit@1 (Fast)** | 0.8823529412 | 0.8823529412 | ✅ Identique |

### Stage 2 : Decider (Classification)
Les métriques de classification sont quasiment identiques. Les légères variations sur le `hit@1` proviennent de la gestion des "ties" (scores identiques) par Pandas lors du filtrage top-50.

| Métrique | Meta (Référence) | Reproduit | Écart |
|----------|-----------------|-----------|-------|
| **AUC** | 0.9914929108 | 0.9914792935 | -0.000014 |
| **Brier Score** | 0.0181959197 | 0.0181981027 | +0.000002 |
| **Calibrated Brier** | 0.0055309096 | 0.0055309180 | < 1e-7 |
| **Hit@1** | 0.9019607843 | 0.9019607843 | ✅ Identique |

## 4. Protocole de Validation
Pour reproduire ces chiffres, le protocole suivant doit être respecté :
1. Charger `data/samples_v4_with_ranker.parquet`.
2. Utiliser l'ordre des features défini dans `feature_order` du meta-modèle (44 features).
3. **Filtrage Decider :** Appliquer `ranker_fast` pour sélectionner le Top-50 ET inclure impérativement tous les positifs (`label == 1`), même s'ils sont hors Top-50 (conforme à `scripts/train_xgb_decider.py`).
4. **Tie-breaking :** Notez que `nlargest` peut varier légèrement en cas de scores identiques.

## 5. Conclusion pour le Routing
La base ML est solide et les modèles chargés correspondent exactement aux performances documentées. Nous pouvons désormais implémenter le module de routing (AUTO/REVIEW) avec une confiance totale dans les scores produits par le `decider`.

---
*Document généré par Antigravity le 2026-01-11*
