# SIRETO — Plan V5 : Data Cleaning + Retrieval Alignment + Experiments A/B/C

Date: 2026-01-21  
Owner: Nathan / Antigravity  
Scope: XGBoost two-stage pipeline (retrieval + ranking + routing)  
Status: Ready to implement (Final Specifications)

---

## 1. Decisions Metier "Verrouillees"

| Point | Decision V5 | Justification |
|-------|-------------|---------------|
| **Faux GT** | Exclure si mismatch INSEE ET CP | Securite geo : zero tolerance sur les erreurs humaines CRM |
| **Scope Partitions** | **S1 (Prod-first)** | Reduire le taux de NOT_IN_PARTITION en production |
| **UL Parquet** | **Obligatoire** | Les champs UL sont critiques pour le matching de haute precision |
| **GT Injection** | **Garder + Flag** | Eviter l'effondrement du training set tout en mesurant le skew |
| **SIREN Siblings** | **Retrieval + Features** | Coherence : si on trouve via le siege, on score via le siege |
| **Knobs Retrieval** | **500 / 100 / 200 / 50** | Prefilter_k / Min_cand / Char_top_k / Sem_top_k |

---

## 2. Source of Truth (SSOT) des artefacts

### 2.1 Artefacts versionnes
- `data/crm_ok_gt.csv` : CRM nettoye (GT valides).
- `data/candidates_v5_all/` : Partitions incluant les etablissements fermes.
- `data/samples_v5_{A,B,C}.parquet` : Datasets d'entrainement.
- `models/xgb_two_stage_meta_v5_{A,B,C}.json` : Pack modele complet (knobs + model paths).

---

## 3. Checklist d'Implementation pour l'Agent Build

### Phase 1 : Environnement & Data Audit
- [ ] **Transfert Local** : Deplacer le repo hors OneDrive pour eviter les corruptions DuckDB.
- [ ] **Audit GT** : Executer `scripts/audit_gt.py` pour generer `data/crm_ok_gt.csv`.
- [ ] **Check UL** : Verifier l'integrite de `StockUniteLegale_utf8.parquet`.

### Phase 2 : Build Partitions V5 (S1 Scope)
- [ ] **Scope** : Extraire tous les codes INSEE/CP de `crm_ok_gt.csv` (et optionnellement du CRM prod).
- [ ] **Build** : `python3 scripts/build_candidate_partitions_v5.py --training-csv data/crm_ok_gt.csv` (Inclus fermes par defaut).

### Phase 3 : Alignement Retrieval & Generation Samples
- [ ] **Unified Code-path** : Creer `src/xgb_matcher/retrieval.py` pour unifier `generate_training_samples` et `infer`.
- [ ] **Variante A** : Baseline (TF-IDF primary).
- [ ] **Variante B** : Bag-of-names + Char fallback (200) + Semantic rescue (50).
- [ ] **Variante C** : Variante B + Enrichment noms par SIREN siblings (Retr + Feat).

### Phase 4 : Training & Eval
- [ ] **Training** : Lancer Stage 1 et Stage 2 pour chaque variante.
- [ ] **Meta** : Generer les 3 fichiers `.meta.json` incluant tous les knobs de retrieval.
- [ ] **Comparison** : Evaluer le Hit@1 sur le split `gt_was_injected == False`.

---

## 4. Parametres de Retrieval (Knobs)

| Parametre | Valeur |
|-----------|--------|
| `prefilter_k` | 500 |
| `min_candidates` | 100 |
| `char_top_k` | 200 |
| `rescue_addr_hash` | True |
| `rescue_numeric_tokens` | True |
| `rescue_semantic_k` | 50 |
| `semantic_min_sim` | 0.3 |
| `include_closed` | True |
| `tfidf_name_mode` | "bag" (pour B et C) / "primary" (pour A) |

