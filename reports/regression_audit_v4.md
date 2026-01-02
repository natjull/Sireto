# Audit de Régression Hit@1: V3 (92.4%) → V4 (78.1%)

**Date**: 2 Janvier 2026
**Auditeur**: Claude Code (Opus 4.5)

---

## Résumé Exécutif

La régression de **14.3 points** de Hit@1 entre V3 et V4 est causée par **deux problèmes distincts**:

1. **TF-IDF prefilter trop restrictif** (cause principale): Génère des samples avec très peu de négatifs pour les noms propres
2. **Partitions V4 incomplètes** (cause secondaire): Champs UL manquants + candidats sans noms non filtrés

| Version | Date | Hit@1 (test) | Samples/query (moy) | Samples/query (min) |
|---------|------|--------------|---------------------|---------------------|
| V3 | Dec 24 | 92.4% | 51.0 | 12 |
| V4 | Dec 31 | 78.1% | 31.3 | **1** |

---

## Cause Racine Identifiée

### Le problème: TF-IDF prefilter trop restrictif

**V4** (`generate_training_samples_v4.py` ligne 239-263) utilise un TF-IDF prefilter qui ne retourne que les candidats ayant des **tokens en commun** avec le nom CRM:

```python
def prefilter_candidates_tfidf(...) -> List[List[int]]:
    crm_norm = [normalize_text(n or "") for n in crm_names]
    q = vectorizer.transform(crm_norm)
    sims = q @ cand_matrix.T  # Similarité TF-IDF

    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:   # <-- Fallback SEULEMENT si ZÉRO match!
            top_indices.append([])
            continue
        # ... sinon, retourne seulement les matchs TF-IDF
```

**Problème critique** (ligne 414-421): Le fallback random sampling ne s'active que si `idx_list` est **vide**, pas s'il est court:

```python
if idx_list:
    subset = [candidates[i] for i in idx_list]  # Seulement 2-5 candidats!
else:
    subset = random.sample(candidates, prefilter_k)  # Fallback jamais déclenché
```

### Preuve par l'exemple

| Query ID | CRM Name | TF-IDF matches | Pool total | Samples générés |
|----------|----------|----------------|------------|-----------------|
| 374 | AUGUSTE PERDONNET | 2 | 4,450 | 2 |
| 1315 | SODATEX | 1 | 1,633 | 1 |
| 1382 | EMAG | 1 | 3,578 | 1 |
| 1602 | CESR | 2 | 13,141 | 2 |

Pour ces noms propres, TF-IDF ne trouve que 1-2 candidats avec tokens partagés. Résultat: **samples avec 1-2 candidats au lieu de 51**.

### Statistiques V4 test split

- **399 queries (15.5%)** ont seulement 1-2 samples
- **441 queries (17.2%)** ont 3-10 samples
- **32.7%** des queries ont ≤10 samples (vs V3 minimum de 48)

### Comparaison V3 vs V4

| Aspect | V3 | V4 |
|--------|----|----|
| Prefilter | Jaro (`quick_score`) | TF-IDF |
| Comportement | Score TOUS les candidats, prend TOP-K | Ne retourne que les candidats avec tokens partagés |
| Garantie | Toujours ~500 candidats prefiltrés | Peut retourner 1-2 candidats |
| Samples/query | Min=12, Max=51, Moy=51 | Min=1, Max=51, Moy=31.3 |

---

## Impact sur le Modèle

Le manque de négatifs pour ~33% des queries cause:

1. **Biais d'apprentissage**: Le modèle ne voit pas assez de négatifs difficiles pour ces patterns
2. **Surfit aux patterns communs**: Le modèle excelle sur les noms génériques mais échoue sur les noms propres
3. **Recall@5 moins impacté**: 95.9% (V4) vs 98.7% (V3) car le GT est toujours présent

---

## Cause Racine #2: Partitions V4 Incomplètes

### Problème A: Champs UL manquants

Le script `build_candidate_partitions_v4.py` ne charge pas tous les champs UniteLegale:

| Champ | V3 | V4 | Impact |
|-------|----|----|--------|
| `sigle_ul` | ✅ | ✅ | - |
| `denomination_ul` | ✅ | ✅ | - |
| `denomination_usuelle_ul` | ✅ | ✅ | - |
| `nom_ul` | ✅ | ❌ | **Critique pour EI** |
| `prenom_usuel_ul` | ✅ | ❌ | **Critique pour EI** |
| `nom_usage_ul` | ✅ | ❌ | Minor |
| `pseudonyme_ul` | ✅ | ❌ | Minor |

**Impact quantifié (Lyon 1er, INSEE 69381):**
- 25,540 candidats Person/EI (54% du total)
- 16,379 (64%) n'ont AUCUN nom en V4
- Ces candidats sont **impossibles à matcher** sans `nom_ul + prenom_usuel_ul`

### Problème B: Candidats sans noms non filtrés

| Aspect | V3 | V4 |
|--------|----|----|
| Filtrage sans noms | `drop_unnamed_candidates=True` | **Aucun filtrage** |
| Lyon 1er (69381) | 31,580 candidats | 47,269 candidats |
| Sans noms | 0 (filtrés) | 15,622 (33%) |

Ces candidats "anonymes" polluent le pool TF-IDF et ne peuvent jamais être matchés.

---

## Correctif Minimal Proposé

### Option A: Garantir un minimum de candidats (recommandé)

Dans `generate_training_samples_v4.py`, modifier le comportement du TF-IDF prefilter:

```python
# Ligne 414-421 - Avant
if idx_list:
    subset = [candidates[i] for i in idx_list if i < len(candidates)]
else:
    if candidates:
        k = min(prefilter_k, len(candidates))
        subset = random.sample(candidates, k)

# Après (correctif)
MIN_CANDIDATES = 50  # Garantir au moins 50 candidats
if idx_list and len(idx_list) >= MIN_CANDIDATES:
    subset = [candidates[i] for i in idx_list if i < len(candidates)]
else:
    # Fallback: combiner TF-IDF + random pour atteindre MIN_CANDIDATES
    if candidates:
        tfidf_set = set(idx_list) if idx_list else set()
        remaining = [i for i in range(len(candidates)) if i not in tfidf_set]
        needed = max(0, MIN_CANDIDATES - len(tfidf_set))
        random_extra = random.sample(remaining, min(needed, len(remaining)))
        subset_idx = list(tfidf_set) + random_extra
        subset = [candidates[i] for i in subset_idx if i < len(candidates)]
    else:
        subset = []
```

### Option B: Revenir au prefilter V3 (jaro)

Remplacer le TF-IDF prefilter par le `quick_score` de V3 qui garantit toujours TOP-K candidats.

### Option C: Entraîner avec les samples V3

Utiliser `data/samples_aligned_v3.parquet` pour entraîner les modèles V4 (features V4 + pooling V3).

### Option D: Corriger le build des partitions (recommandé)

Dans `build_candidate_partitions_v4.py`, ajouter les champs UL manquants:

```python
# Ligne 137-143 - Ajouter dans la requête SELECT:
ul.nomUniteLegale AS nom_ul,
ul.nomUsageUniteLegale AS nom_usage_ul,
ul.prenomUsuelUniteLegale AS prenom_usuel_ul,
ul.pseudonymeUniteLegale AS pseudonyme_ul,
```

Puis rebuilder les partitions:
```bash
python scripts/build_candidate_partitions_v4.py \
    --training-csv data/entrainements.csv \
    --parquet-path data/StockEtablissement_utf8.parquet \
    --ul-path data/StockUniteLegale_utf8.parquet \
    --harvest-db data/harvest_full.sqlite \
    --output-dir data/candidates_v4_fixed
```

---

## Commandes de Validation

```bash
# 1. Vérifier la distribution actuelle des samples V4
python3 -c "
import pandas as pd
v4 = pd.read_parquet('data/samples_v4_sem.parquet')
test = v4[v4['split']=='test'].groupby('query_id').size()
print(f'Queries with ≤5 samples: {(test <= 5).sum()} / {len(test)}')
print(f'Min samples/query: {test.min()}')
"

# 2. Régénérer les samples V4 avec le correctif
python scripts/generate_training_samples_v4.py \
    --output data/samples_v4_fixed.parquet \
    --partitions-dir data/candidates_v4 \
    --prefilter-k 500 \
    --max-negatives 50

# 3. Réentraîner le modèle
python scripts/train_xgb_ranker.py --samples data/samples_v4_fixed.parquet
python scripts/train_xgb_decider.py --samples data/samples_v4_fixed.parquet --calibration isotonic

# 4. Valider le Hit@1
# Attendu: Hit@1 ≈ 90-92% après correctif
```

---

## Annexes

### A. Distribution des samples par query

```
V3 samples_aligned_v3.parquet:
- Shape: (699,374, 44)
- Queries: 13,720
- Min samples/query: 12
- Max samples/query: 51
- Mean samples/query: 51.0

V4 samples_v4_sem.parquet:
- Shape: (523,086, 49)
- Queries: 16,724
- Min samples/query: 1
- Max samples/query: 51
- Mean samples/query: 31.3
```

### B. Fichiers impactés

| Fichier | Rôle | Correctif nécessaire |
|---------|------|---------------------|
| `scripts/generate_training_samples_v4.py` | Génération samples | **OUI** - ligne 414-421 |
| `scripts/infer_xgb_two_stage.py` | Inférence V4 | **OUI** - ligne 279-283 |
| `src/xgb_matcher/blocking.py` | TF-IDF prefilter | Non (helper) |

### C. Modèles et métadonnées

- V3 meta: `models/xgb_matcher_features_20251224_111912.json`
- V4 meta: `models/xgb_two_stage_meta_20251231_162239.json`
- V3 samples: `data/samples_aligned_v3.parquet`
- V4 samples: `data/samples_v4_sem.parquet`

---

## Conclusion

La cause racine est **clairement identifiée**: le TF-IDF prefilter de V4 ne garantit pas un minimum de candidats, ce qui génère des samples avec très peu de négatifs pour les noms propres spécifiques.

Le correctif est **minimal et ciblé**: garantir au moins 50 candidats par query en combinant TF-IDF + random sampling.

**Risque du correctif**: Faible. Le changement n'affecte que les queries avec peu de matchs TF-IDF.

**Temps estimé de validation**: Régénération samples (~2h) + Réentraînement (~30min) + Test (~5min)
