# Sireto

Pipeline Python pour le rapprochement automatique CRM ↔ SIRENE/SIRET.

## Vue d'ensemble

Sireto est un pipeline de matching d'entités haute performance conçu pour associer des données CRM françaises à la base SIRENE. 

Le pipeline utilise une architecture **XGBoost à 3 stages** (Pipe V7) remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML :

1. **Stage 1 (Ranker)** : Sélectionne les top-k candidats SIRENE par requête.
2. **Stage 2 (Decider)** : Score chaque candidat (probabilité de match 0-1).
3. **Stage 3 (Risk Metamodel)** : Décide du routing AUTO vs REVIEW en analysant la cohérence globale de la requête.

## Performances (SOTA 26/01/2026)

| Métrique | Valeur |
|----------|--------|
| **Taux d'AUTO** | **74.5%** |
| **Précision réelle** | **99.84%** (3 FPs sur 1 872 cas AUTO) |
| **Couverture CRM** | **97.8%** |

## Architecture du Pipeline (Pipe V7)

- **Socle XGBoost** : Les décisions AUTO sont prises par le Risk Metamodel calibré à 0.835.
- **Fallback Places** : Les cas classés en `REVIEW` sont envoyés vers un lookup guidé par l'API Google Places (via Serper.dev).
- **Zéro Faux Positif** : Le système est conçu pour privilégier la sûreté, n'effectuant un MATCH que si l'évidence est multi-sources et cohérente avec SIRENE.

## Installation et Usage

Voir `AGENTS.md` pour les détails techniques et les scripts d'exécution.

```bash
# Exemple d'exécution du routing
python scripts/route_xgb_results.py \
  --input-path reports/benchmark_v6a_topk.csv \
  --risk-meta models/routing_risk_meta.json \
  --output-path reports/routed.csv
```

## Structure du Projet

- `src/xgb_matcher/` : Logique de scoring et features ML.
- `src/pipe_v6/` : Orchestration, clients API et cache.
- `scripts/` : Entraînement, inférence et évaluation.
- `models/` : Modèles XGBoost et métadonnées de routing.
- `reports/` : Benchmarks et rapports de performance.

