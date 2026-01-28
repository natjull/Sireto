# SIRETO Routing SSOT (Single Source of Truth)

**Date** : 26 janvier 2026  
**Performance** : **74.5% AUTO @ 99.84% Precision**

## 1. Architecture du Pipeline (Pipe V7)

Le pipeline utilise une architecture XGBoost à 3 stages remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML.

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1** | `xgbranker_20260124_210313.json` | Sélectionne les top-k candidats SIRENE par requête |
| **Stage 2** | `xgb_decider_20260124_210218.json` | Score chaque candidat (probabilité de match) |
| **Stage 3** | `routing_risk_model.pkl` | Décide si la requête peut être AUTO ou doit aller en REVIEW |

## 2. Métamodèle de Risque (Stage 3)

Le métamodèle analyse la "scène" de la requête (68 features) pour décider du routing.

**Seuil de décision** :
```python
if risk_score >= 0.835:
    return "AUTO_RISK"
else:
    return "REVIEW"
```

## 3. Métriques de Référence (Test Set - 2 512 requêtes)

| Métrique | Valeur |
|----------|--------|
| **Taux d'AUTO** | **74.5%** (1 872 / 2 512) |
| **Précision réelle** | **99.84%** (3 vrais FPs après audit) |
| **Couverture CRM** | **97.8%** |

## 4. Artefacts Canoniques

| Artefact | Chemin | Description |
|----------|--------|-------------|
| **Ranker** | `models/xgbranker_20260124_210313.json` | Stage 1 |
| **Decider** | `models/xgb_decider_20260124_210218.json` | Stage 2 |
| **Risk Model** | `models/routing_risk_model.pkl` | Stage 3 Routing |
| **Risk Meta** | `models/routing_risk_meta.json` | Threshold & Features |
| **GT Data** | `data/crm_ok_gt.csv` | Gold Standard |

## 5. Pipeline d'Inférence

```bash
# 1. Inférence Ranker + Decider
python scripts/infer_xgb_two_stage.py --input-path crm.csv --output-path topk.csv

# 2. Routing (AUTO/REVIEW)
python scripts/route_xgb_results.py \
  --input-path topk.csv \
  --risk-meta models/routing_risk_meta.json \
  --disable-certainty-rules \
  --disable-promotion-rules \
  --output-path routed.csv
```

## 6. Places Fallback (REVIEW → MATCH_PLACES ou NO_MATCH)

Pour les cas REVIEW, un fallback Places simple est disponible : **"Places as CRM Repair"**.

### Principe

```python
def fallback_places(crm):
    places = search_places(crm.name, crm.address, crm.city)
    
    if not places:
        return NO_MATCH
    
    # Dept-guard: reject if Google found something in wrong area
    if places[0].postcode[:2] != crm.postcode[:2]:
        return NO_MATCH
    
    # Use Google's top-1 as "repaired CRM" and rerun EXACT same pipeline
    repaired_crm = {
        "crm_name": places[0].title,
        "crm_address": places[0].address,
        "postcode": places[0].postcode,
    }
    
    result = run_full_xgb_pipeline(repaired_crm)  # Stages 1, 2, 3 identical
    
    if result.status == "AUTO":
        return MATCH_PLACES(result.siret)
    else:
        return NO_MATCH  # Post-Places, no more REVIEW possible
```

### Key Insight

Google connaît l'identité des entreprises (successions, changements de nom, fusions). 
On fait confiance à Google pour "quelle entreprise" et à XGBoost pour "quel SIRET".

### Usage

```bash
python scripts/route_xgb_results.py \
  --input-path topk.csv \
  --places-mode \
  --output-path routed.csv
```

### Fichiers

| Fichier | Description |
|---------|-------------|
| `src/pipe_v6/places_fallback.py` | Logique du fallback simplifié |
| `src/pipe_v6/serper_places_client.py` | Client API Serper + cache |
| `src/xgb_matcher/infer.py` | Engine d'inférence réutilisable |

---
*Note : Ce document est la source unique de vérité pour le routing SIRETO. Toute modification des seuils ou des modèles doit être reflétée ici.*
