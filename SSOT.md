# SIRETO Routing SSOT (Single Source of Truth)

**Date** : 1 février 2026  
**Performance cible** : **74.5% AUTO @ 99.84% Precision**

## 1. Architecture du Pipeline (Pipe V7)

Le pipeline utilise une architecture XGBoost à 3 stages remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML.

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1** | `xgbranker_fast_..._ultima.json` | Sélectionne les top-k candidats SIRENE (ML Pruning) |
| **Stage 2** | `xgb_decider_..._hardneg.json` | Score chaque candidat (probabilité de match) |
| **Stage 3** | `routing_risk_model.pkl` | Décide si la requête est AUTO ou REVIEW |

## 2. Stratégie de Retrieval "Ultima" (Stage 1)

Le retrieval est conçu pour un recall quasi-total sans fallback départemental.

| Paramètre | Valeur | Rationale |
|-----------|--------|-----------|
| **Pool Mode** | `insee_then_postcode` | Strict commune, fallback CP local |
| **TF-IDF Name** | `bag` (No L2 Norm) | Capture toutes les enseignes/SIREN sans dilution |
| **TF-IDF Addr** | `tokens` (No L2 Norm) | **Nouveau** : Repêche via tokens d'adresse (ex: "RUE PAIX") |
| **Rescue** | `addr_hash` + `numeric` | Whitelist systématique pour matchs exacts |
| **Prefilter k** | 500 | Union des canaux Nom + Adresse |
| **Stage 1 top-N** | 200 | Envoyés au Stage 2 (Decider) |

## 3. Stratégie d'Entraînement

L'alignement Train/Serve est garanti par l'usage des mêmes modules de retrieval.

- **Phase 1 : Ranker FAST** (No Semantic)
  - Entraîné sur le pool de retrieval brut (A/B/C Ultima).
  - Objectif : Maximiser le Recall@N.
- **Phase 2 : Decider + Risk**
  - Mining de **Hard Negatives** via le Ranker FAST nouvellement entraîné.
  - Entraînement du Decider sur ces cas complexes.
  - Entraînement du Risk Model sur la distribution d'inférence réelle.

## 4. Artefacts Canoniques (En cours)

| Artefact | Chemin (Ultima) | Description |
|----------|--------|-------------|
| **Ranker** | `models/xgbranker_fast_20260131_v5fast_B_ultima.json` | Stage 1 Champion |
| **Decider** | `models/xgb_decider_20260131_v5fast_B_ultima_hardneg.json` | Stage 2 (Training...) |
| **GT Data** | `data/crm_ok_gt.csv` | Gold Standard (17k sites) |

---
*Note : Ce document est la source unique de vérité. Toute modification doit respecter le principe de "Zero Skew".*
