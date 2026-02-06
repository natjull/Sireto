# SIRETO Routing SSOT (Single Source of Truth)

**Date** : 4 février 2026  
**Performance cible** : **74.5% AUTO @ 99.84% Precision**

## 1. Architecture du Pipeline (Pipe V7)

Le pipeline utilise une architecture XGBoost à 3 stages remplaçant les approches LLM par un traitement 100% déterministe et supervisé par ML.

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1** | `xgbranker_fast_..._clean_500neg.json` | Sélectionne les top-k candidats SIRENE (ML Pruning) |
| **Stage 2** | `xgb_decider_..._hardneg.json` | Score chaque candidat (probabilité de match) - **champion non validé** |
| **Stage 3** | `routing_risk_model.pkl` | Décide si la requête est AUTO ou REVIEW |

## 2. Partitions candidats (point de départ)

Les partitions SIRENE sont la source canonique des candidats et le point de départ du filtrage.

| Élément | Valeur | Rationale |
|---------|--------|-----------|
| **Chemin** | `data/candidates_v5_all/` | Stock candidats partitionné par commune |
| **Partitionnement** | `insee/` et `cp/` (Hive) | Accès rapide par commune ou CP |
| **Génération** | `scripts/build_candidate_partitions_v5.py` | Pipeline unique de build |
| **Nettoyage** | Suppression des partitions existantes avant build | Évite l'accumulation de doublons SIRET |
| **Contraintes** | Pas de doublons SIRET dans une partition | Garantie pour TF-IDF + ranker |
| **Mega-communes** | Seuil 100 000 lignes | Fallback CP filtre INSEE pour stabilite memoire |

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
- **Phase 1 : Ranker FAST** (No Semantic)
  - Entraîné sur le pool de retrieval (ULTIMA B).
  - Objectif : maximiser le Recall@50.
- **Phase 2 : Decider**
  - Hard negatives minés via le Ranker FAST.
  - Semantic retrieval **désactivé** (aucun ajout de candidats hors pool).
  - Semantic gate **désactivé par défaut** (`XGB_SEMANTIC_GATE_ENABLED=0`) pour éviter le skew.
- **Phase 3 : Risk Model**
  - Entraîné sur la distribution d'inférence réelle (top-k + features de routing).

## 5. Artefacts Canoniques (En cours)

| Artefact | Chemin (Ultima) | Description |
|----------|--------|-------------|
| **Ranker** | `models/xgbranker_fast_v5fast_B_clean_500neg.json` | Stage 1 Champion (actuel) |
| **Ranker Meta** | `models/xgb_two_stage_meta_v5fast_B_clean_500neg.json` | Meta Ranker (clean) |
| **Decider** | TBD | Aucun champion validé à ce stade |
| **Decider (candidat)** | `models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json` | Candidat non satisfaisant |
| **Meta Pipeline** | TBD | Bloqué tant que le Decider n'est pas validé |
| **GT Data** | `data/crm_ok_gt.csv` | Gold Standard (17k sites) |

## 6. Environnement d'execution (SSOT)

Les developpements et optimisations doivent respecter la contrainte materielle suivante :

- Machine cible : MacBook Pro M4 Pro, 24 GB RAM (latence et memoire doivent rester compatibles)

---
*Note : Ce document est la source unique de vérité. Toute modification doit respecter le principe de "Zero Skew".*
