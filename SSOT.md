# SIRETO Routing SSOT (Single Source of Truth)

**Date** : 5 février 2026  
**Performance cible** : **74.5% AUTO @ 99.84% Precision**

## 1. Architecture du Pipeline (Pipe V7)

Le pipeline utilise une architecture XGBoost à 3 stages remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML.

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1** | `xgbranker_fast_..._v6_turbo.json` | Sélectionne les top-k candidats SIRENE (ML Pruning) |
| **Stage 2** | `xgb_decider_..._v6_turbo.json` | Score chaque candidat (probabilité de match) |
| **Stage 3** | `routing_risk_model.pkl` | Décide si la requête est AUTO ou REVIEW |

## 2. Partitions candidats (point de départ)

Les partitions SIRENE sont la source canonique des candidats et le point de départ du filtrage.

| Élément | Valeur | Rationale |
|---------|--------|-----------|
| **Chemin** | `data/candidates_v6_all/` | Stock candidats partitionné par commune (V6 string-safe) |
| **Partitionnement** | `insee/` et `cp/` (Hive) | Accès rapide par commune ou CP |
| **Génération** | `scripts/build_candidate_partitions_v5.py` | Pipeline unique de build (Types String forcés) |
| **Nettoyage** | Suppression des partitions existantes avant build | Évite l'accumulation de doublons SIRET |
| **Contraintes** | Pas de doublons SIRET dans une partition | Garantie pour TF-IDF + ranker |
| **Mega-communes** | Seuil 100 000 lignes | Policy `full_insee` pour coverage maximal |

## 3. Stratégie de Retrieval "Ultima" (Stage 1)

Le retrieval est conçu pour un recall quasi-total sans fallback départemental.

Standard unique : **Variant B** (Bag-of-names + char fallback).

| Paramètre | Valeur | Rationale |
|-----------|--------|-----------|
| **Pool Mode** | `insee_then_postcode` | Strict commune, fallback CP local |
| **TF-IDF Name** | `bag` + word ngrams (1,2) | Aligné sur l'entraînement v5fast (token_pattern \b\w+\b) |
| **TF-IDF Addr** | word ngrams (1,2), norm=None | Repêche par adresse, robuste aux variantes de voie |
| **Char TF-IDF** | ngrams (3,5) | Fallback acronymes/typos sur le nom |
| **Rescue** | `addr_hash` + `numeric` | Whitelist systématique (matchs exacts) |
| **Prefilter k** | 500 | Union des canaux Nom + Adresse |
| **Stage 1 top-N** | 50 | Envoyés au Stage 2 (Decider) |
| **Rescue post-ranker** | Aucun | Pas d'ajout de candidats hors top-N |

## 4. Stratégie d'Entraînement

L'alignement Train/Serve est garanti par l'usage des mêmes modules de retrieval.

- **Phase 0 : Retrieval (aligné Train/Serve)**
  - Pool strict `insee_then_postcode` + double TF-IDF (nom + adresse) + rescue universel.
  - `prefilter_k=500`, padding deterministe si pool < min_candidates.
  - **Mega-policy** : `full_insee` (V6+) pour supprimer les pertes dues au CP CRM.
- **Phase 1 : Ranker FAST** (No Semantic)
  - Entraîné sur le pool de retrieval (ULTIMA B).
  - Sampling "Turbo" : 50 négatifs choisis via rangs TF-IDF du retrieval.
  - Objectif : maximiser le Recall@50.
- **Phase 2 : Decider**
  - Hard negatives minés via le Ranker FAST (scène = top-50 ranker).
  - Semantic retrieval **désactivé** (aucun ajout de candidats hors pool).
  - Semantic gate **activé par défaut** (`XGB_SEMANTIC_ENABLED=1`) pour le Stage 2.
- **Phase 3 : Risk Model**
  - Entraîné sur la distribution d'inférence réelle (top-k + features de routing).

## 5. Artefacts Canoniques (En cours)

| Artefact | Chemin (V6 Turbo) | Description |
|----------|--------|-------------|
| **Ranker** | `models/xgbranker_20260205_195700.json` | Stage 1 Champion (v6 string-safe) |
| **Ranker Meta** | `models/xgb_two_stage_meta_20260205_195700.json` | Meta Ranker aligné SSOT |
| **Decider** | TBD | En attente de génération des samples Turbo |
| **GT Data** | `data/crm_ok_gt.csv` | Gold Standard (corrigé INSEE/CP) |

## 6. Environnement d'execution (SSOT)

Les developpements et optimisations doivent respecter la contrainte materielle suivante :

- Machine cible : MacBook Pro M4 Pro, 24 GB RAM (latence et memoire doivent rester compatibles)

---
*Note : Ce document est la source unique de vérité. Toute modification doit respecter le principe de "Zero Skew".*
