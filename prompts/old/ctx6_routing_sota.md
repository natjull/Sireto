# Prompt pour ctx6: Recherche SOTA sur le Routing de Décisions

## Contexte Projet

Tu travailles sur **SIRETO**, un système de **Entity Matching** qui associe des entreprises d'un CRM à leur fiche officielle dans la base SIRENE (registre français des entreprises).

### Architecture actuelle (Pipeline V4)

```
CRM Record → TF-IDF Prefilter → XGBoost Ranker → XGBoost Decider → Routing → AUTO/REVIEW
                                    ↓                  ↓
                              Top-200 candidats   Score calibré [0,1]
```

1. **Prefilter TF-IDF**: Réduit 12M établissements → ~50-500 candidats par query
2. **XGBoost Ranker**: Classe les candidats (Recall@20 = 99.4%)
3. **XGBoost Decider**: Prédit P(match) pour chaque paire (calibré avec isotonic regression, ECE=0.0005)
4. **Routing**: Décide AUTO (automatique) vs REVIEW (validation humaine/API)

### Performances actuelles du modèle

```
Recall@1 (Decider):  90.2%  ← Si on prend toujours le top1, on a raison 90.2% du temps
Recall@5:            97.7%
Recall@20:           99.4%
AUC Decider:         0.99+
Calibration ECE:     0.0005 (excellent)
```

### Le problème du Routing

Le routing actuel utilise des **règles heuristiques complexes** (certainty rules, blocking rules, promotion rules) qui:
- Donnent seulement **42% d'AUTO** avec **89% de précision**
- Sont **trop conservatrices**: 5800 reviews/mois au lieu de ~1000 nécessaires

### Découvertes récentes (ctx5)

1. **Le modèle two-stage est déjà excellent** (90.2% Recall@1)
2. **Les règles heuristiques ajoutent du bruit**, pas de la valeur
3. **Un Router ML simple** (XGBoost sur features de confiance) atteint:
   - 90.5% volume AUTO avec 95% précision
   - Features clés: `dominance`, `gap_1_2`, `score_top1`

```python
# Features de confiance découvertes
dominance = score_top1 - mean(scores[1:])  # Importance: 0.14
gap_1_2 = score_top1 - score_top2           # Importance: 0.14
n_competitors = count(scores >= 0.9 * score_top1)  # Ambiguïté
```

4. **Trade-off fondamental**:

| Scénario | Volume AUTO | Précision | Erreurs/10k |
|----------|-------------|-----------|-------------|
| Pas de routing (tout AUTO) | 100% | 90.2% | 980 |
| Router ML (seuil 0.62) | 90.5% | 95.1% | 443 |
| Routing actuel | 42% | 89% | 461 |

## Fichiers clés

```
models/
├── xgbranker_fast_*.json           # Ranker (étape 1)
├── xgb_decider_*.json              # Decider (étape 2)
├── xgb_decider_calibrator_*.pkl    # Calibration isotonique
├── xgb_two_stage_meta_*.json       # Métadonnées (feature order, etc.)
└── router_confidence_model.pkl     # Router ML prototype

data/
├── samples_v4_with_ranker.parquet  # Dataset avec labels (692k samples)
└── splits/                         # train/val/test splits

scripts/
├── infer_xgb_two_stage.py          # Inférence two-stage
├── route_xgb_results.py            # Routing (à améliorer)
└── evaluate_routing.py             # Évaluation
```

## Questions de recherche SOTA


### 1. Routing optimal sous contrainte

C'est un problème de **Selective Prediction** / **Learning to Defer**:
- Maximiser le volume AUTO sous contrainte P(correct|AUTO) ≥ target
- Littérature: Geifman & El-Yaniv 2017, Mozannar & Sontag 2020

Pistes SOTA:
- **Conformal Prediction**: Intervalles de confiance avec garanties
- **Cost-sensitive learning**: Optimiser directement le trade-off erreur/review
- **Calibration-aware thresholding**: Exploiter la calibration parfaite du decider

### 2. Estimation d'incertitude

Le gap `score_top1 - score_top2` est un proxy d'incertitude. Peut-on faire mieux?
- **MC Dropout** sur le decider
- **Ensemble disagreement**
- **Evidential Deep Learning**

### 3. Same-SIREN Resolution

52 des 112 "erreurs" sont en fait le bon SIREN mais mauvais établissement.
Comment choisir le bon établissement parmi plusieurs du même SIREN?
- Préférer OUVERT vs FERMÉ?
- Matcher sur l'adresse exacte?
- Utiliser la date de création?

### 4. Ground Truth Quality

29 "erreurs" ont le même nom ET la même adresse que le GT.
Ce sont probablement des **erreurs dans le ground truth**.
Comment détecter et corriger automatiquement?

## Contraintes techniques

- Python 3.11+
- XGBoost pour le ML (pas de deep learning, trop lent pour le volume)
- Inférence doit être < 100ms par query
- Budget API Places: minimiser les appels ($0.02/appel)

## Ce que tu dois faire

1. **Analyser** le code et les données existantes
2. **Proposer** des améliorations SOTA pour le routing
3. **Implémenter** les plus prometteuses
4. **Évaluer** rigoureusement sur le split test
5. **Documenter** les résultats dans un rapport

## Objectifs quantitatifs

| Métrique | Actuel | Objectif |
|----------|--------|----------|
| Volume AUTO | 42% | **>85%** |
| Précision AUTO | 89% | **>95%** |
| Recall global | 90.2% | **>92%** |

## Pour commencer

```bash
# Charger les données
import pandas as pd
samples = pd.read_parquet('data/samples_v4_with_ranker.parquet')
test = samples[samples['split'] == 'test']

# Charger le modèle
from xgboost import XGBClassifier
import json
with open('models/xgb_two_stage_meta_20260103_132351.json') as f:
    meta = json.load(f)
decider = XGBClassifier()
decider.load_model('models/xgb_decider_20260103_132351.json')
```

## Références

- Geifman & El-Yaniv (2017): "Selective Classification for Deep Neural Networks"
- Mozannar & Sontag (2020): "Consistent Estimators for Learning to Defer to an Expert"
- Conformal Prediction: Vovk et al., Angelopoulos & Bates (2021)
- Entity Matching SOTA: DITTO (Li et al. 2020), Magellan framework

---

*Prompt préparé par ctx5 le 2026-01-04*
