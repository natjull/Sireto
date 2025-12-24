# XGBoost Matcher — Production Deployment Prompt

## Contexte du Projet

Tu travailles sur **SIRETO**, un système de matching entre des entrées CRM (nom d'entreprise + adresse) et la base SIRENE (14M d'établissements) pour identifier le SIRET correspondant.

### Architecture Actuelle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE XGBOOST MATCHER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. FILTRAGE GÉOGRAPHIQUE                                                   │
│     - Index par code postal / INSEE                                         │
│     - Pool candidats: ~5-10K par commune                                    │
│                                                                             │
│  2. STAGE 1: RANKING RAPIDE (sans sémantique)                              │
│     - XGBRanker sur 37 features (Jaro, Levenshtein, token overlap, etc.)   │
│     - Sélection top-200 candidats                                          │
│                                                                             │
│  3. STAGE 2: CLASSIFICATION AVEC SÉMANTIQUE                                │
│     - XGBClassifier sur 40 features (+ 3 features sémantiques)             │
│     - Modèle sémantique: siret-bert-deploy (MiniLM-L12 fine-tuné)          │
│     - Batch encoding GPU (MPS) pour performance                             │
│                                                                             │
│  4. OUTPUT: Top-K candidats avec scores (probabilité 0-1)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fichiers Clés

| Fichier | Description |
|---------|-------------|
| `src/xgb_matcher/features.py` | 40 features (nom, adresse, sémantique, contexte) |
| `src/xgb_matcher/semantic.py` | Embeddings sémantiques, batch encoding, cache LRU |
| `src/xgb_matcher/candidates.py` | Chargement candidats SIRENE optimisé |
| `scripts/infer_xgb_matcher_topk.py` | Script d'inférence two-stage avec TreeSHAP |
| `models/xgbranker_20251224_111912.json` | Modèle Ranker (stage 1) |
| `models/xgbclassifier_20251224_111912.json` | Modèle Classifier (stage 2) |
| `models/semantic/siret-bert-deploy/` | Modèle sémantique fine-tuné |

### Performance Actuelle (sur subset Corbas/Décines)

| Métrique | Valeur |
|----------|--------|
| **Score médian TOP1** | 99.8% |
| **Score moyen TOP1** | 96.3% |
| **Score > 0.8** | 94.8% |
| **Score > 0.5** | 98.3% |
| **Vitesse** | ~2 queries/sec (avec sémantique) |

---

## Objectif Principal

**Définir une stratégie de déploiement "safe" sans faux positifs** en établissant des seuils et des règles de routage entre :

1. **AUTO** : Match automatique → le SIRET est assigné sans intervention humaine
2. **REVIEW** : Match incertain → l'opérateur doit valider manuellement
3. **NO_MATCH** : Aucun candidat viable → le SIRET reste vide

---

## Cas d'Erreurs Identifiés

### Catégorie 1: Adresse CRM incorrecte (irrécupérable)
- **Filmor** : CRM dit "33 Avenue Franklin Roosevelt" mais l'entreprise est rue Emile Zola
- Le modèle matche correctement l'adresse fournie → ce n'est pas une erreur du modèle

### Catégorie 2: Commandes en masse (non-SIRETIsables)
- **CORBAS - ALLENDE** : Maisons du Rhône sans identifiant unique
- **DECINES - MITTERAND** : SIRET attendu 22690001700600 mais score faible

### Catégorie 3: Points de présence sans SIRET
- **POP Corbas** : Point télécom, pas d'établissement SIRENE

### Catégorie 4: Établissements fermés
- Certains matchs sont sur des établissements fermés (etat_admin = "F")
- Question: accepter les fermés ou fallback sur actifs ?

### Catégorie 5: Entrées enchevêtrées / multi-sites
- **Stryker Corporation** : 2 adresses différentes mélangées dans les données

---

## Données Disponibles pour Analyse

### Fichier d'inférence avec SHAP
```
reports/xgb_infer_topk_corbas_decines_with_closed.csv
- 672 lignes (134 queries × top-5)
- Colonnes: crm_name, siret_candidate, score, candidate_state, rank, shap
```

### Features disponibles (40)
```python
FEATURE_NAMES = [
    # Nom (26 features)
    'has_any_name', 'name_count', 'name_jaro_max', 'name_jaro_second', 'name_jaro_gap',
    'name_levenshtein_max', 'name_token_overlap_max', 'name_first_word_match_max',
    'name_contains_crm_max', 'name_crm_contains_cand_max', 'acronym_match_max',
    'name_sim_max_etab', 'name_sim_max_ul', 'name_sim_max_sigle', 'name_sim_max_pm_dirigeant',
    'type_of_max_name', 'is_ul_name_max', 'is_sigle_max', 'name_length_max',
    'has_person_name', 'person_name_jaro_max', 'name_city_overlap_max', 'name_is_city_like_max',
    'name_semantic_max', 'name_semantic_second', 'name_semantic_gap',
    
    # Adresse (7 features)
    'addr_jaro', 'addr_levenshtein', 'postcode_match', 'city_match',
    'street_number_diff', 'addr_token_overlap', 'street_name_jaro',
    
    # Contexte (7 features)
    'name_addr_consistency', 'is_siege', 'is_association',
    'alias_match', 'token_overlap_ul', 'ul_vs_pm_indicator', 'is_crm_school'
]
```

---

## Tâches à Réaliser

### Phase 1: Analyse des Erreurs
1. **Extraire les cas où le bon SIRET n'est pas en rang 1**
   - Comparer avec ground truth si disponible
   - Catégoriser par type d'erreur

2. **Analyser la distribution des scores**
   - Identifier les zones de confusion (score intermédiaire)
   - Trouver le "gap" entre TOP1 et TOP2 comme signal de confiance

3. **Examiner les explications SHAP**
   - Quelles features dominent dans les cas de faux positifs ?
   - Y a-t-il des patterns exploitables ?

### Phase 2: Définition des Seuils
1. **Proposer un seuil AUTO**
   - Score minimum + gap minimum pour acceptation automatique
   - Règles additionnelles (ex: état admin = "A")

2. **Proposer une zone REVIEW**
   - Plage de scores nécessitant validation humaine
   - Signaux visuels pour l'opérateur (features clés, alternatives)

3. **Critères NO_MATCH**
   - Quand déclarer qu'aucun candidat n'est valide

### Phase 3: Implémentation Production
1. **API/Module de matching**
   - Input: (crm_name, crm_address, postcode, city)
   - Output: {siret, confidence, decision: AUTO|REVIEW|NO_MATCH, alternatives}

2. **Interface opérateur pour REVIEW**
   - Affichage des top-5 avec scores et features clés
   - Boutons de validation/rejet

3. **Métriques de monitoring**
   - Taux AUTO/REVIEW/NO_MATCH
   - Taux de correction manuelle
   - Distribution des scores

---

## Variables d'Environnement

```bash
# Sémantique
XGB_SEMANTIC_ENABLED=1
XGB_SEMANTIC_MODEL=models/semantic/siret-bert-deploy
XGB_SEMANTIC_BATCH_SIZE=512

# Inférence
XGB_SHAP_ENABLED=1  # Pour explications TreeSHAP
XGB_INCLUDE_CLOSED=1  # Inclure établissements fermés
```

---

## Commandes Utiles

```bash
# Inférence avec SHAP sur subset
XGB_SEMANTIC_ENABLED=1 XGB_SEMANTIC_MODEL=models/semantic/siret-bert-deploy \
python3 scripts/infer_xgb_matcher_topk.py \
  --crm-path data/testcrm/data_aligned.csv \
  --with-shap \
  --include-closed-candidates \
  --write-csv

# Statistiques rapides
python3 -c "
import pandas as pd
df = pd.read_csv('reports/xgb_infer_topk.csv')
rank1 = df[df['rank'] == 1]
print(f'Queries: {len(rank1)}')
print(f'Score moyen: {rank1[\"score\"].mean():.4f}')
print(f'Score > 0.8: {(rank1[\"score\"] > 0.8).sum()} ({100*(rank1[\"score\"] > 0.8).mean():.1f}%)')
"
```

---

## Contraintes

1. **Zéro faux positif en AUTO** : Un SIRET assigné automatiquement ne doit JAMAIS être incorrect
2. **Maximiser le taux AUTO** : Pour réduire la charge de validation manuelle
3. **Explainability** : Chaque décision doit être justifiable (SHAP, features saillantes)
4. **Performance** : < 3 secondes par query en production

---

## Questions Ouvertes

1. **Établissements fermés** : Les accepter en AUTO avec un flag, ou fallback actifs ?
2. **Multi-établissements même adresse** : Comment gérer 65 SIRET au 4 rue du Mont Blanc Corbas ?
3. **Seuil gap TOP1-TOP2** : Quel écart minimum pour confiance élevée ?
4. **Règles post-hoc** : Réactiver D4/D5/etc. ou tout laisser au modèle ?

---

## Livrables Attendus

- [ ] Matrice de confusion sur ground truth (si disponible)
- [ ] Seuils AUTO/REVIEW/NO_MATCH calibrés
- [ ] Script de production prêt à l'emploi
- [ ] Documentation des décisions et edge cases
- [ ] Dashboard de monitoring (optionnel)
