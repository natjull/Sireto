# Handover Consultant — Audit Pipeline Entity Matching SIRETO

## Contexte général

### Le projet
SIRETO est un système de **matching d'entités** qui associe des entreprises d'un CRM client à leur fiche officielle dans le registre SIRENE (base légale française des entreprises).

Le pipeline utilise une architecture **two-stage XGBoost**:
1. **Ranker (stage 1)**: récupère les top-K candidats parmi des millions d'établissements via TF-IDF + scoring XGBoost
2. **Decider (stage 2)**: re-score les candidats avec un modèle XGBoost calibré pour sélectionner le meilleur match

Après le scoring XGBoost, un **routing** décide si le match est suffisamment fiable pour être validé automatiquement (AUTO) ou s'il doit être vérifié (REVIEW). Les cas REVIEW sont ensuite enrichis via l'API Google Places.

### Objectif business
- Maximiser le taux d'AUTO (réduction des coûts de vérification manuelle et API Places)
- Minimiser les faux positifs (erreurs de matching automatique)

---

## Ce qui a été réalisé dans cette session d'audit

### 1. Bugs corrigés

**a) Normalisation SIRET** dans [scripts/evaluate_routing.py](scripts/evaluate_routing.py)
- Problème: comparaison "7565021800318" vs "07565021800318" comptée comme FP
- Fix: normalisation à 14 caractères avec `zfill(14)`

**b) Règle identical_name sans gap check** dans [scripts/route_xgb_results.py](scripts/route_xgb_results.py)
- Problème: homonymes parfaits (CABANON, SCHLUMBERGER) passaient en AUTO sans vérification du gap
- Fix: ajout de `gap_min: 0.05` dans la règle identical_name

### 2. Tests exécutés

- Évaluation du Recall@K sur le split test (2193 queries avec ground truth)
- Analyse des faux positifs (FP) par catégorie
- Comparaison des stratégies de routing (règles heuristiques vs seuils simples vs Router ML)
- Génération de listes de FP avec noms CRM et noms candidats

### 3. Découverte majeure: écart de précision

**Métriques du fichier meta officiel** (`models/xgb_two_stage_meta_20260103_132351.json`):
```
Decider calibré (test): hit@1 = 90.2%
Ranker (test): Recall@20 = 99.4%
```

**Métriques observées en inférence**:
```
Recall@1 effectif: 72.5%
GT dans le pool top-20: 88.1%
```

**Explication de l'écart**:
- Le hit@1 = 90.2% est calculé sur les **samples d'entraînement** où le ground truth (GT) est **toujours présent par construction** (hard negatives)
- En **inférence/production**, le retrieval (TF-IDF + ranker) ne récupère le GT que dans **88.1%** des cas
- Recall@1 effectif = Recall@20_retrieval × hit@1_decider ≈ 88% × 90% ≈ **79%**
- L'écart restant (79% → 72.5%) suggère un possible train/serve skew

---

## Métriques clés actuelles (split test, 2193 queries avec GT)

| Métrique | Valeur |
|----------|--------|
| Recall@1 (top1 = GT) | 72.5% |
| Recall@20 retrieval (GT dans pool) | 88.1% |
| hit@1 conditionné (GT dans pool) | 82.3% |
| FP total (baseline 100%) | 586 |

### Performance du routing actuel (Phase 4)

| Scénario | Volume AUTO | FP | Précision AUTO |
|----------|-------------|-----|----------------|
| Baseline (100%) | 100% | 586 | 73.3% |
| Routing Phase 4 | 46% | 112 | 89% |
| Score≥0.85 + Gap≥0.10 | 42% | 58 | 93.7% |

---

## Fichiers pertinents

### Modèles SOTA
```
models/xgb_two_stage_meta_20260103_132351.json     # Meta avec toutes les métriques
models/xgb_decider_20260103_132351.json            # Modèle decider
models/xgbranker_fast_20260103_132351.json         # Modèle ranker (sans sémantique)
models/xgb_decider_calibrator_isotonic_20260103_132351.pkl  # Calibrateur isotonic
```

### Données
```
data/samples_v4_with_ranker.parquet    # Source canonique : samples d'entraînement avec splits train/dev/test (colonne `split`)
data/old/2026-01-11_splits/            # [ARCHIVÉ] Anciens CSV splits (train.csv, dev.csv, test.csv) - ne plus utiliser
data/candidates_v4/                    # Store partitionné des candidats SIRENE
data/StockEtablissement_utf8.parquet   # Base SIRENE établissements
data/StockUniteLegale_utf8.parquet     # Base SIRENE unités légales
```

**Important (2026-01-11)** : `data/splits/` a été archivé. Pour extraire un CSV CRM du split test, filtrer `data/samples_v4_with_ranker.parquet` où `split == 'test'` et reconstruire les colonnes CRM source.

### Scripts clés
```
scripts/infer_xgb_two_stage.py         # Inférence 2-étages (TF-IDF → ranker → decider)
scripts/route_xgb_results.py           # Routing (AUTO/REVIEW) avec règles
scripts/evaluate_routing.py            # Évaluation routing vs GT
scripts/evaluate_samples_v4.py         # Évaluation recall ranker sur samples
scripts/generate_training_samples_v4.py # Génération samples (TF-IDF blocking)
scripts/train_xgb_ranker.py            # Entraînement ranker
scripts/train_xgb_decider.py           # Entraînement decider + calibration
```

### Retrieval / Blocking
```
src/xgb_matcher/blocking.py            # TF-IDF blocking, prefilter
src/xgb_matcher/naming.py              # Extraction noms candidats, normalisation
src/xgb_matcher/features.py            # Calcul features XGBoost
src/xgb_matcher/semantic.py            # Modèle sémantique fine-tuné
```

### Configuration
```
configs/routing_thresholds.yaml        # Seuils routing (certitude, segments, blocking)
configs/places_thresholds.yaml         # Seuils Google Places
```

### Rapports générés
```
reports/xgb_two_stage_topk_test.csv              # Résultats inférence sur test
reports/routed_phase4_test_fixed.csv             # Résultats routing
reports/routing_evaluation_from_routed_phase4_test_fixed.json  # Métriques routing
reports/decider_eval.json                        # Métriques decider détaillées
reports/fp_router_ml_905.csv                     # Liste FP du Router ML
reports/handover.md                              # Historique complet des sessions
```

### Documentation
```
AGENTS.md                              # Architecture pipeline V7
reports/entity_matching_audit.md       # Plan de phases et objectifs
```

---

## Pistes d'investigation identifiées

### 1. Retrieval gap (88% → 99%?)
- Le retrieval (TF-IDF + ranker stage 1) perd 12% des GT avant même le scoring decider
- Sources possibles: normalisation TF-IDF, blocking par code postal/INSEE, candidats non indexés
- Fichiers concernés: `src/xgb_matcher/blocking.py`, `scripts/generate_training_samples_v4.py`

### 2. Train/serve skew (90% → 82% hit@1 conditionné)
- Même quand le GT est dans le pool, le hit@1 en inférence (82%) est inférieur au hit@1 training (90%)
- Causes possibles: distribution des features différente, normalisation, features manquantes
- Fichiers concernés: `src/xgb_matcher/features.py`, `scripts/infer_xgb_two_stage.py`

### 3. Stratégie de routing
- Le routing actuel (règles heuristiques) atteint 46% AUTO à 89% précision
- Un simple seuil (score≥0.85, gap≥0.10) atteint 42% AUTO à 94% précision
- Un Router ML sur features de confiance pourrait optimiser le trade-off volume/précision
- Fichiers concernés: `scripts/route_xgb_results.py`, `configs/routing_thresholds.yaml`

### 4. Qualité du ground truth
- 29 FP identifiés sont probablement des erreurs GT (même nom + adresse, SIREN différent)
- Impact: les métriques sont potentiellement sous-estimées de ~1-2%

---

## Clarification split CSV vs parquet (audit 2026-01-11) → RÉSOLU

### Constat
- Le modèle 2-étages référence `data/samples_v4_with_ranker.parquet` dans `models/xgb_two_stage_meta_20260103_132351.json`.
- Le parquet contient ses propres splits (`split` colonne) avec `query_id` uniques: train=9457, dev=1952, test=2193.
- Les CSV `data/splits/*.csv` avaient des volumes différents (train=12590, dev=2607, test=2873) → ils ne correspondaient pas au parquet.

### Résolution (2026-01-11)
- **`data/splits/` a été archivé** dans `data/old/2026-01-11_splits/`
- **Source canonique unique** : `data/samples_v4_with_ranker.parquet` (colonne `split` pour train/dev/test)
- Les métriques officielles du modèle (dans `models/xgb_two_stage_meta_20260103_132351.json`) sont basées sur ce parquet

### Origine des anciens CSV (historique)
- `scripts/generate_training_samples_v4.py` créait `data/splits/{train,dev,test}.csv` en splitant `data/entrainements.csv` par SIREN (70/15/15).
- Des scripts `scripts/old/generate_training_samples*.py` écrivaient aussi ces fichiers → risque d'artefacts obsolètes confirmé.

### Scripts impactés (mise à jour nécessaire)
- Scripts prenant `--ground-truth-path data/splits/test.csv` doivent utiliser `data/old/2026-01-11_splits/test.csv` (backward compat) OU extraire le split test du parquet.
- Pour les nouveaux workflows : toujours partir du parquet canonique.

### Migration
- Pour extraire un CSV CRM du split test historique : filtrer `data/samples_v4_with_ranker.parquet` où `split == 'test'` et reconstruire les colonnes CRM source si nécessaire.

---

## Commandes de référence

### Inférence sur le split test (utiliser CSV CRM réel, pas splits d'entraînement)
```bash
# Pour référence legacy, le split test archivé :
XGB_SEMANTIC_ENABLED=1 python scripts/infer_xgb_two_stage.py \
  --crm-path data/old/2026-01-11_splits/test.csv \
  --output-path reports/xgb_two_stage_topk_test.csv \
  --top-k 20 \
  --partitions-dir data/candidates_v4 \
  --meta-path models/xgb_two_stage_meta_20260103_132351.json
```

### Routing
```bash
python scripts/route_xgb_results.py \
  --input-path reports/xgb_two_stage_topk_test.csv \
  --output-path reports/routed_phase4_test.csv \
  --thresholds configs/routing_thresholds.yaml
```

### Évaluation routing
```bash
python scripts/evaluate_routing.py \
  --routed-path reports/routed_phase4_test.csv \
  --ground-truth-path data/old/2026-01-11_splits/test.csv \
  --output-path reports/routing_evaluation_test.json
```

**Note (2026-01-11)** : Les exemples ci-dessus utilisent le split test archivé pour backward compatibility. Préférer extraire le split test depuis `data/samples_v4_with_ranker.parquet` (filtrer `split == 'test'` et reconstruire CSV CRM si nécessaire).

### Évaluation decider sur samples
```bash
python scripts/evaluate_decider_on_samples.py \
  --samples data/samples_v4_with_ranker.parquet \
  --model models/xgb_decider_20260103_132351.json \
  --calibrator models/xgb_decider_calibrator_isotonic_20260103_132351.pkl \
  --output reports/decider_eval.json
```

---

## Questions ouvertes

1. Pourquoi le retrieval perd-il 12% des GT? Est-ce un problème de blocking (TF-IDF, codes postaux) ou de données (GT absents de SIRENE)?

2. Pourquoi le hit@1 conditionné en inférence (82%) est-il inférieur au hit@1 training (90%)? Y a-t-il un train/serve skew?

3. Quel est le trade-off optimal volume/précision pour le routing? Faut-il privilégier des règles simples ou un modèle ML dédié?

4. Les 29 FP "probables erreurs GT" doivent-ils être corrigés dans les données d'entraînement?

---

## Environnement

- Python 3.11+
- XGBoost, scikit-learn, pandas, numpy
- Modèle sémantique: sentence-transformers fine-tuné (optionnel, `XGB_SEMANTIC_ENABLED=1`)
- Base SIRENE: ~12M établissements partitionnés par code INSEE/postal
