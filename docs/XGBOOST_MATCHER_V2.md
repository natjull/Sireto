# XGBoost Matcher v2 - Procédure de Refonte

Ce document décrit la procédure complète pour aligner l'entraînement du modèle XGBoost avec l'inférence, résolvant le **train/serve skew** identifié.

## Problème résolu

L'ancien entraînement utilisait un pool de candidats réduit (autres CRM comme négatifs), tandis que l'inférence score tous les établissements du CP/INSEE. Le modèle n'apprenait donc pas à discriminer dans un pool massif réaliste.

## Architecture des scripts

```
src/xgb_matcher/
├── candidates.py       # [NEW] Logique partagée de pooling candidats
├── features.py         # [MODIFIED] +5 nouvelles features
└── naming.py           # Inchangé

scripts/
├── generate_training_samples.py  # [NEW] Génération samples alignés
├── train_xgb_matcher_v2.py       # [NEW] Entraînement v2
├── evaluate_xgb_comprehensive.py # [NEW] Évaluation complète
└── infer_xgb_matcher_topk.py     # Existant (avec policy layer)
```

## Nouvelles features (Phase 3)

| Feature | Description | Règle de reranking correspondante |
|---------|-------------|-----------------------------------|
| `is_siege` | Indicateur siège social | D9 |
| `is_association` | Candidat de type association | B1 |
| `alias_match` | Match alias entre parenthèses | D6 |
| `token_overlap_ul` | Overlap tokens avec UL | D11 |
| `ul_vs_pm_indicator` | Source du meilleur match (UL vs PM) | D7 |

---

## Procédure d'exécution

### Étape 1 : Générer les samples alignés

```bash
cd /path/to/SIRETO
python scripts/generate_training_samples.py
```

**Sortie :**
- `data/samples_aligned.parquet` : Tous les samples avec features
- `data/samples_aligned.json` : Métadonnées
- `data/splits/train.csv`, `dev.csv`, `test.csv` : Splits par SIREN **[DEPRECATED 2026-01-11]**

**Note (2026-01-11)** : Les fichiers CSV `data/splits/*.csv` sont **archivés** dans `data/old/2026-01-11_splits/`. La source canonique est désormais `data/samples_v4_with_ranker.parquet` avec une colonne `split` (train/dev/test).

**Durée estimée :** 30-60 minutes (dépend du volume de données)

### Étape 2 : Entraîner les modèles v2

```bash
python scripts/train_xgb_matcher_v2.py
```

**Sortie :**
- `models/xgbranker_<timestamp>.json` : Ranker LambdaMART
- `models/xgbclassifier_<timestamp>.json` : Classifier binaire
- `models/xgb_matcher_features_<timestamp>.json` : Métadonnées + métriques

### Étape 3 : Évaluer les modèles

```bash
# Évaluation sur test set, avec et sans policy layer
python scripts/evaluate_xgb_comprehensive.py --dataset test --policy both
```

**Sortie :**
- `reports/eval_test_<timestamp>.json` : Rapport complet

---

## Métriques d'évaluation

| Métrique | Description |
|----------|-------------|
| **Hit@1** | % de fois où le bon SIRET est rang 1 |
| **Hit@5** | % de fois où le bon SIRET est dans le top 5 |
| **MRR** | Mean Reciprocal Rank (moyenne de 1/rang) |
| **AUC** | Area Under ROC Curve |

---

## Comparaison attendue

```
SANS policy layer (D1-D11) | AVEC policy layer
---------------------------+------------------
Hit@1: XX.XX%              | Hit@1: YY.YY%
Hit@5: XX.XX%              | Hit@5: YY.YY%
MRR:   X.XXX               | MRR:   Y.YYY
```

Si les nouvelles features sont bien apprises, l'écart **AVEC - SANS** devrait diminuer, car le modèle capture directement les patterns des règles.

---

## Décommissionnement progressif des règles

Une fois le modèle v2 validé :

1. **Désactiver les règles redondantes** une par une
2. **Mesurer l'impact** sur le test set
3. **Conserver uniquement** les règles business non-apprenables (ex: B1 école/association si le modèle ne généralise pas)

```bash
# Test avec policy layer désactivé
python scripts/evaluate_xgb_comprehensive.py --dataset test --policy off
```

---

## Monitoring recommandé

En production, surveiller :
- Hit@1, Hit@5 : Taux de correspondance
- Rate REVIEW, NO_MATCH : Taux d'incertitude
- Déclenchements policy layer : Fréquence des règles D1-D11

---

## Rollback

Si le modèle v2 régresse, les anciens modèles sont toujours dans `models/` avec leur timestamp. L'inférence utilisera automatiquement le plus récent via `find_latest_models()`.
