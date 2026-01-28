# Audit Phase 2 (Ctx2) — Entity Matching Expert Review

## 📊 Verdict: Partiel — 3 Items Majeurs Manquants

Ctx2 a bien avancé sur plusieurs fronts, mais **n'a pas terminé l'exécution** (aucun training lancé) et a **omis certaines features critiques** du plan.

---

## ✅ Ce qui a été fait (Code)

| Item du Plan | Implémenté ? | Fichier |
|---|---|---|
| Hard Negatives via Ranker Top-K | ✅ Oui | `generate_training_samples_v3.py:find_latest_ranker`, `score_with_ranker` |
| `address_density` | ✅ Oui | `features.py:L361, L804` |
| `idf_name` (IDF overlap) | ✅ Oui | `features.py:L738` |
| Stopwords (`NAME_STOPWORDS`) | ✅ Oui | `features.py:L52` |
| Calibration (Platt/Isotonic) | ✅ Oui | `train_xgb_matcher_v2.py:calibrate_classifier` |
| Calibrator chargé en inférence | ✅ Oui | `infer_xgb_matcher_topk.py:L742-752, L867-868` |
| Brier Score ajouté | ✅ Oui | `train_xgb_matcher_v2.py` |

---

## ❌ Ce qui manque (Code)

| Item du Plan | Implémenté ? | Impact |
|---|---|---|
| `numeric_token_match` | ❌ Non | Critique pour "APAJH 69" vs "APAJH" |
| `legal_form_category` | ❌ Non | Moyen — utile pour PUBLIC/PRIVE |
| `score_gap` / `score_ratio` / `pool_size` (meta-features) | ❌ Non | Critique pour routing AUTO/REVIEW |
| `same_address_diff_name` sampling | ❌ Non | Critique — c'est le cas #1 de FP |
| `Recall@10` / `Recall@20` monitoring | ❌ Non | Critique pour KPI du ranker |

---

## ⚠️ Ce qui manque (Exécution)

D'après le `handover.md`, les commandes d'entraînement n'ont **pas été exécutées** :

```bash
# Non lancé
XGB_SEMANTIC_ENABLED=0 python scripts/generate_training_samples_v3.py --output data/samples_v4.parquet
XGB_SEMANTIC_ENABLED=0 python scripts/train_xgb_matcher_v2.py --samples data/samples_v4.parquet --calibration isotonic
```

---

## 🔬 Analyse Détaillée (Expert EM)

### 1. `numeric_token_match` — Critique

Le plan demande cette feature pour gérer les codes numériques dans les noms (ex: "APAJH **69**").
Sans elle, le modèle ne peut pas distinguer "APAJH 69" de "APAJH 75".

**Action** : Ajouter dans `features.py` :
```python
def numeric_token_match(a: str, b: str) -> float:
    nums_a = set(re.findall(r'\d+', a))
    nums_b = set(re.findall(r'\d+', b))
    if not nums_a or not nums_b:
        return 0.0
    return len(nums_a & nums_b) / max(len(nums_a), len(nums_b))
```

### 2. Meta-features (`score_gap`, `pool_size`) — Critique

Le plan spécifie des meta-features pour le routing :
> `score_gap`, `score_ratio`, `top3_avg`, `pool_size`, `has_name_evidence`

Ces features sont **essentielles** pour calibrer le seuil AUTO.
Par exemple, si `score_gap` (score_top1 - score_top2) est faible, c'est un signal d'incertitude.

**Action** : Calculer ces features dans `infer_xgb_matcher_topk.py` après le scoring et les exporter dans le CSV.

### 3. `same_address_diff_name` sampling — Critique

Le plan demande :
> Ajouter négatifs "même adresse / nom différent"

C'est la principale cause du FP #53 (LES DOUCEURS vs LES MURIERS). Sans cet échantillonnage, le modèle ne voit **jamais** ces cas pendant l'entraînement.

**Action** : Dans `generate_training_samples_v3.py`, après le sampling des hard negatives, ajouter une boucle qui trouve les candidats avec `addr_jaro >= 0.95` mais `name_jaro_max < 0.5`.

### 4. `Recall@K` Monitoring — Critique

Le plan demande :
> Monitorer **Recall@10** et **Recall@20**.

Actuellement, seuls Hit@1, Hit@3, Hit@5 sont calculés. Pour un **ranker**, le KPI le plus important est le Recall@K (est-ce que le GT est dans le top-K ?). C'est différent de Hit@K qui compte combien de fois le top-K contient *un* positif.

**Action** : Ajouter `compute_recall_at_k` dans `train_xgb_matcher_v2.py`.

---

## 📝 Recommandations

1.  **Priorité 1** : Implémenter `same_address_diff_name` sampling (tue le FP #53).
2.  **Priorité 2** : Ajouter les meta-features (`score_gap`, `pool_size`) pour améliorer le routing.
3.  **Priorité 3** : Ajouter `numeric_token_match` (tue le FN #112 APAJH).
4.  **Exécuter** les commandes d'entraînement et valider sur le set de test.
