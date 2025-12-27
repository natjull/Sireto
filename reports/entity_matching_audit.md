# Audit approfondi — XGBoost Entity Matching (SIRETO)

## Résumé exécutif (version amendée)

Cet audit complète et corrige l’analyse initiale avec un regard **data science + production** (train/serve, calibration, pipeline Places). Objectif produit confirmé : **maximiser le taux d’AUTO‑MATCH XGBoost sans faux positifs**, le reste part en **REVIEW** et est **rematché via Google Places** pour retrouver le bon SIRET.

**Constats majeurs (priorité haute)**
1. **Skew train/serve avéré** : le ranker est entraîné avec des features sémantiques, mais l’inférence stage‑1 les désactive → risque de rater le vrai candidat avant la phase de scoring. (`scripts/train_xgb_matcher_v2.py`, `scripts/infer_xgb_matcher_topk.py`)
2. **Routing dépendant de SHAP** : `_route_xgb()` lit des features depuis `shap`, mais le CSV top‑k ne contient pas ces features si `--with-shap` n’est pas activé → promotions AUTO rarement déclenchées. (`scripts/route_xgb_results.py`, `scripts/infer_xgb_matcher_topk.py`)
3. **Calibration très mauvaise** : les scores 0.99+ ont ~10% d’erreur réelle (sur le set diagnostiqué) → seuils AUTO trop optimistes. (`reports/diagnostic_report.md`, `reports/diagnostic_analysis.json`)
4. **Hard negatives annoncés mais pas implémentés** : l’échantillonnage utilise un score heuristique au lieu d’un vrai top‑K ranker → faible apprentissage des confusions critiques (même adresse / nom différent). (`scripts/generate_training_samples_v3.py`)
5. **Risque “adresse‑seule”** : le modèle sur‑pondère l’adresse et produit des faux positifs co‑localisés (centre commercial / ZI), exactement les erreurs observées.

**Conséquence** : la politique actuelle d’AUTO est structurellement risquée. La bonne trajectoire SOTA passe par **(1) alignement train/serve**, **(2) hard negative mining real‑world**, **(3) calibration + seuils segmentés**, **(4) pipeline en 2 étages** (ranker → classif décision) et **(5) boucles de feedback depuis Places**.

---

## Sources & périmètre analysés

- `reports/entity_matching_audit.md` (version initiale)
- `reports/diagnostic_report.md`
- `reports/diagnostic_analysis.json`
- `reports/diagnostic_plots.png`
- Code: `scripts/train_xgb_matcher_v2.py`, `scripts/infer_xgb_matcher_topk.py`, `scripts/generate_training_samples_v3.py`, `scripts/route_xgb_results.py`, `src/xgb_matcher/*`, `src/pipe_v6/places_*`

> **Politique officielle (v3 par défaut)** :  
> - `scripts/generate_training_samples_v3.py` est **la base unique** pour la génération d’échantillons.  
> - `scripts/generate_training_samples.py` et `scripts/generate_training_samples_v2.py` sont **dépréciés** (ne plus utiliser).  
> - Toute nouvelle feature / stratégie d’échantillonnage doit être ajoutée à **v3**.

> **Note critique (corrigée)** : une incohérence de *coverage* existait entre `diagnostic_report.md` et `diagnostic_analysis.json` (dénominateur différent). Elle est **corrigée** via `scripts/fix_diagnostic_report.py`, qui impose :  
> `coverage = auto_count / total_count` avec `total_count = somme(calibration.count)`.  
> Le JSON conserve `coverage_raw` pour traçabilité.

---

## Objectif produit (rappel)

1. **XGBoost doit auto‑matcher le maximum possible** avec un **risque quasi nul** (taux FP proche de 0).
2. Les cas **REVIEW / NO_MATCH** passent dans un **second pipeline Places** pour “rattraper” le bon SIRET.
3. **Jamais de faux positif en sortie automatique** (la revue ou Places absorbe l’incertitude).

---

## Diagnostic quantifié (avec regard critique)

### 1) Taille et erreurs globales
- Taille du set diagnostiqué : **134**
- Erreurs globales (FP) : **12** (~9%)

### 2) Calibration (issues majeures)
D’après `diagnostic_analysis.json` :

| Bin score | N | Erreurs | Taux d’erreur réel | Confiance attendue |
|---|---:|---:|---:|---:|
| (0.98, 0.99] | 8 | 1 | 12.5% | 98.6% |
| (0.99, 1.00] | 111 | 11 | 9.9% | 99.8% |

➡ **Le modèle est sur‑confiant** : il annonce 99.8% mais se trompe ~10%.

### 3) Risk‑coverage (données disponibles)
- Best tradeoff ≤5% erreur (d’après JSON) : **θ ≈ 0.995** avec **4 FP / 98 AUTO**.
- La *coverage* reportée est incohérente dans le JSON → on retient **auto_count** plutôt que coverage.

### 4) Patterns d’erreurs (confirmés par audit initial)
- **Faible overlap lexical du nom** + **adresse forte** → faux positifs co‑localisés.
- **Nom CRM sous‑ensemble** → scores sous‑estimés (faux négatifs).

---

## Problèmes structurels (train/serve & pipeline)

### P1 — Skew train/serve du ranker (critique)
- **Entraînement** : ranker avec features sémantiques.
- **Inférence stage‑1** : `skip_semantic=True`.
➡ Le ranker ne voit pas la même distribution que l’entraînement. Risque : le vrai candidat sort du top‑N.

### P2 — Routing dépendant de SHAP (critique)
- `_route_xgb()` récupère des features depuis SHAP.
- Le CSV top‑k n’exporte pas ces features **sans `--with-shap`**.
➡ En production, la majorité des règles AUTO sont muettes. Il faut **exporter les features directement**.

### P3 — Negative sampling insuffisant
- Script annonce “hard negatives via ranker top‑K”, mais utilise un score heuristique.
- Pas de confusions réalistes “même adresse / nom différent”.
➡ Le modèle ne voit pas ses cas d’erreur critiques → FP persistants.

### P4 — Pas de calibration opérationnelle
- Scores interprétés comme proba → seuils instables.
- Aucune calibration post‑hoc (Platt / isotonic) ni métrique ECE.

### P5 — Distribution du pool candidats incohérente
- Training : union INSEE + CP (tous candidats).
- Inference : INSEE sinon CP.
➡ Définir une stratégie de pool cohérente et **alignée**.

---

## Analyse des features (corrigée + approfondie)

### 1) Adresse sur‑pondérée → faux positifs
La feature `addr_token_overlap` agit comme un “super‑signal” même sans preuve lexicale. C’est la principale cause des FP co‑localisés.

**Action** : gate l’adresse par preuve lexicale minimale.

### 2) Sémantique sans garde‑fou
`name_semantic_max` capture des similarités **thématiques**, pas identitaires. Sans garde lexical, elle crée des FPs (“RUBIX FRANCE” ≈ “FRANCE MECANIQUE”).

**Action** : sémantique **gated by lexical evidence**.

### 3) Features manquantes (SOTA)
- **IDF / TF‑IDF char‑ngrams** pour pénaliser les tokens génériques.
- **address_density** : nb d’établissements à la même adresse → signal de risque.
- **match numérique** : chiffres / codes (ex “APAJH 69”).
- **catégories juridiques** (PUBLIC / PRIVE / ASSO) en feature.
- **commonness_name_idf** : pénaliser les noms/tokens très fréquents (FRANCE, BATIMENT, SOCIETE).

---

## Architecture cible (SOTA adaptée à ton pipeline)

### Étape 1 : Ranker rapide (no‑semantic)
- Optimisé pour **rappeler le bon candidat** dans le top‑K.
- Features “cheap” uniquement (lexical + adresse + tokens).
- **Métrique clé** : Recall@10 / Recall@20 (plafond de performance finale).

### Étape 2 : Classifier (full)
- Features complètes (sémantique + meta‑features).
- Score calibré → décision AUTO/REVIEW.

### Étape 3 : Routing + Places
- AUTO si score calibré + règles “anti‑FP” passent.
- REVIEW sinon → Google Places (ou Web) pour upgrade sans risque.

---

## Routing robuste pour maximiser AUTO sans FP

**Principe** : un AUTO doit combiner **score élevé + preuve lexicale + faible ambiguïté**.

Pseudo‑règle :
```python
# Notation: s=score_calibre, tok=name_token_overlap, jaro=name_jaro, addr=addr_jaro
# density = nb candidats même adresse
# gap = s_top1 - s_top2
if (tok < 0.1 and addr > 0.8) or density > 5:
    return "REVIEW"

# seuils segmentés (nom court plus risqué)
if name_word_count <= 2:
    thr = 0.995
elif name_word_count <= 4:
    thr = 0.99
else:
    thr = 0.98

# exigence de preuve lexicale minimale
if jaro < 0.6 and tok < 0.3:
    return "REVIEW"

# exigence d’écart entre top1 et top2
if gap < 0.05:
    return "REVIEW"

return "AUTO" if s >= thr else "REVIEW"
```

---

# Schéma directeur d’implémentation (Codex)

## Protocoles de handover (obligatoires)
**Objectif** : garantir des transitions parfaites entre fenêtres de contexte.

**Règles générales pour chaque session Codex**
1. **Lire** `reports/entity_matching_audit.md` avant toute action.
2. **Tracer** le travail dans un court “testament” (voir format ci‑dessous).
3. **Ne jamais** démarrer une nouvelle phase tant que la phase précédente n’est pas au moins “Phase READY”.

**Format du testament (à laisser à chaque fin de session)**
Créer / mettre à jour : `reports/handover.md`
```
## Session <date_utc> — Codex Ctx <N>
- Phase ciblée : <Phase 0/1/2/3>
- Objectif : <objectif précis>
- Changements :
  - <fichier> : <résumé>
- Tests/commandes exécutées :
  - <commande> → <résultat court>
- Etat :
  - ✅ terminé / ⚠️ partiel / ❌ bloqué
- Prochaines étapes immédiates :
  - <1>
  - <2>
```

---

## Découpage par fenêtre de contexte (Codex = Ctx0)
**Principe** : chaque fenêtre a une mission unique, des entrées/sorties claires, et un critère d’arrêt.

### Ctx0 — Diagnostic & Plan (actuel)
**Mission** : stabiliser les diagnostics + détailler le plan + protocoles handover.  
**Entrées** : `reports/diagnostic_*`, `reports/entity_matching_audit.md`  
**Sorties attendues** :
- Diagnostic cohérent (`coverage` correct) + report/plot régénérés
- Plan d’implémentation détaillé + training schedule par phase
- Protocoles de handover ajoutés
**Critère d’arrêt** : les sections “Entraînement” + “Handover” sont présentes et `diagnostic_report.md` cohérent.

### Ctx1 — Quick Wins (Phase 1)
**Mission** : routing indépendant de SHAP + export features + ranker_fast.  
**Entrées** : `scripts/infer_xgb_matcher_topk.py`, `scripts/route_xgb_results.py`, `scripts/train_xgb_matcher_v2.py`  
**Sorties attendues** :
- CSV top‑k contient features de routing
- `_route_xgb()` utilise features directes
- Ranker stage‑1 aligné (no‑semantic)
**Critère d’arrêt** : Phase 1 “Definition of Done” OK + `reports/handover.md` mis à jour.

### Ctx2 — Sprint ML (Phase 2)
**Mission** : hard negative mining + nouvelles features + calibration.  
**Entrées** : `scripts/generate_training_samples_v3.py`, `src/xgb_matcher/features.py`, `scripts/train_xgb_matcher_v2.py`  
**Sorties attendues** :
- dataset v4 avec hard negatives réalistes
- nouvelles features intégrées
- calibrateur sauvegardé
**Critère d’arrêt** : Phase 2 “Definition of Done” OK + `reports/handover.md` mis à jour.

**Note importante** : v1/v2 de génération d’échantillons sont dépréciées. Tout changement va dans v3.

### Ctx3 — SOTA (Phase 3)
**Mission** : pipeline 2‑étapes, multi‑blocking, silver labels.  
**Entrées** : nouveaux scripts ranker/decider, blocking, Places feedback  
**Sorties attendues** :
- pipeline 2‑étapes stable
- multi‑blocking actif
- silver labels intégrés
**Critère d’arrêt** : Phase 3 “Definition of Done” OK + `reports/handover.md` mis à jour.

---

## Phase 0 — Pré‑requis (1–2 jours)
- **Corriger la génération des diagnostics** : coverage, risk‑coverage, calibration ECE.
- **Exporter les features nécessaires au routing** directement dans le top‑k CSV.
- **Aligner train/serve** du ranker (features identiques).

### Détails de code (Phase 0)
**Objectif** : rendre les diagnostics reproductibles et cohérents.

- **Script de correction** (déjà ajouté) : `scripts/fix_diagnostic_report.py`
  - Entrée : `reports/diagnostic_analysis.json`
  - Sorties : `reports/diagnostic_analysis.json` (corrigé), `reports/diagnostic_report.md`, `reports/diagnostic_plots.png`
  - Logique : `coverage = auto_count / total_count` avec `total_count = somme(calibration.count)`
  - Ajout de `meta.total_count` + `coverage_raw` pour traçabilité

- **Script de diagnostic reproductible** (implémenté) :
  - **Fichier** : `scripts/diagnostic_xgb_routing.py`
  - **Entrées** :
    - CSV avec ground truth (`ground_truth_siret`) : ex `data/splits/test.csv`
    - Modèles XGB (ranker + classifier)
    - Paramètres : `--pool-mode` (insee_then_postcode / union), `--thresholds`
  - **Sorties** :
    - `reports/diagnostic_analysis.json` (recalculé)
    - `reports/diagnostic_report.md`
    - `reports/diagnostic_plots.png`
  - **Règle de coverage** :
    - `coverage_total = auto_count / total_count`
    - `coverage_eligible = auto_count / eligible_count` (eligible = ground truth dans pool)
  - **Validation** :
    - Vérifier `auto_count = somme` et cohérence avec precision@1
  - **Commandes** :
    - `python scripts/diagnostic_xgb_routing.py --input-path data/splits/test.csv`
    - Variante rapide : `python scripts/diagnostic_xgb_routing.py --limit 200`

## Phase 1 — Quick Wins (1–3 jours)
**Objectif : améliorer l’AUTO immédiatement sans casser le recall**

1. **Route XGB indépendant de SHAP**
   - Ajouter les colonnes lexicales dans `infer_xgb_matcher_topk.py` (name_jaro_max, name_token_overlap_max, name_sim_max_etab, name_crm_contains_cand_max, name_sim_max_pm_dirigeant, addr_jaro, addr_token_overlap, street_number_diff, name_length_max).
   - Modifier `_route_xgb()` pour utiliser ces colonnes en priorité.

2. **Aligner ranker stage‑1**
   - Entraîner un *ranker_fast* avec `skip_semantic=True`.
   - Charger ce ranker pour le stage‑1.

3. **Seuils segmentés immédiats**
   - Implémenter le routing segmenté du diagnostic (nom court/long, adresse complète).
   - Ajouter une règle “adresse‑seule → REVIEW”.

### Entraînement (Phase 1) — quand & comment (MacBook M4 Pro, 24GB)
**But** : valider rapidement le pipeline sans lancer un entraînement complet.

- **Quand** :
  - Dès que le routing est indépendant de SHAP **et** que le ranker_fast existe.
- **Quoi** :
  - Entraîner uniquement le **ranker_fast** (no‑semantic), pour aligner le stage‑1.
  - Pas de re‑training complet du classifier à ce stade (sauf bug).
- **Modalités** :
  - Désactiver la sémantique pour accélérer : `XGB_SEMANTIC_ENABLED=0`
  - Limiter les samples (si besoin) : `--max-negatives 50` dans `generate_training_samples_v3.py`
  - Utiliser un dataset réduit pour smoke tests (ex. `--limit 200` sur diagnostic)
  - Threads recommandés : `OMP_NUM_THREADS=6` (évite la saturation CPU sur macOS)

**Commandes rapides** :
```
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/generate_training_samples_v3.py --output data/samples_aligned_qw.parquet --max-negatives 50
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/train_xgb_matcher_v2.py --samples data/samples_aligned_qw.parquet
python scripts/diagnostic_xgb_routing.py --limit 200
```

### Détails de code (Phase 1)
**Fichiers impactés + contrat attendu**

- `scripts/infer_xgb_matcher_topk.py`
  - Ajouter l’export des features de routing dans `rows_out` :
    - `name_jaro_max`, `name_token_overlap_max`, `name_sim_max_etab`,
      `name_crm_contains_cand_max`, `name_sim_max_pm_dirigeant`,
      `addr_jaro`, `addr_token_overlap`, `street_number_diff`, `name_length_max`
  - Ajouter option `--export-routing-features` (bool) pour activer/désactiver

- `scripts/route_xgb_results.py`
  - Modifier `_route_xgb()` pour utiliser les colonnes directes si disponibles
  - Garder fallback SHAP (si présent) uniquement en dernier recours
  - Ajouter logique “adresse‑seule” : si `name_token_overlap_max < 0.1` et `addr_token_overlap > 0.8` ⇒ REVIEW
  - Ajouter seuils segmentés (nom court / long, adresse complète)

- `scripts/train_xgb_matcher_v2.py`
  - Ajouter un **ranker_fast** :
    - same params, mais features `skip_semantic=True`
    - nom modèle : `xgbranker_fast_<timestamp>.json`
  - Référencer ce modèle en phase 1 dans l’inférence

**Definition of Done Phase 1**
- Top‑k CSV contient toutes les features de routing (vérif : colonnes présentes)
- `_route_xgb()` n’est plus dépendant de SHAP
- Ranker stage‑1 = modèle dédié no‑semantic
- AUTO rate stable ou en hausse, FP rate en baisse sur set diagnostic

**Validation Phase 1 (commandes)**
- `python scripts/infer_xgb_matcher_topk.py --crm-path data/testcrm/data_56_subset_corbas_decines.csv --output-path reports/xgb_infer_topk_phase1.csv --with-shap`
- `python scripts/route_xgb_results.py --input-path reports/xgb_infer_topk_phase1.csv --output-path reports/routed_phase1.csv`
- `python scripts/diagnostic_xgb_routing.py --input-path data/splits/test.csv --limit 200`

## Phase 2 — Sprint ML (1–2 semaines)
**Objectif : réduire massivement les FP et stabiliser l’AUTO**

1. **Hard negative mining real‑world**
   - Générer négatifs depuis top‑K ranker (pas heuristique).
   - Ajouter négatifs “même adresse / nom différent” + “nom proche / autre ville”.

2. **Nouvelles features robustes**
   - `address_density`
   - `idf_name_score` (TF‑IDF char‑ngrams)
   - `numeric_token_match`
   - `legal_form_category`

3. **Calibration**
   - Platt scaling ou isotonic sur set dev.
   - Sauvegarder calibrateur pour l’inférence.

4. **Meta‑features pour décision**
   - `score_gap`, `score_ratio`, `top3_avg`, `pool_size`, `has_name_evidence`.

5. **Métriques Ranker (obligatoires)**
   - Monitorer **Recall@10** et **Recall@20**.
   - Ne pas utiliser Precision@1 comme KPI principal du ranker.

### Entraînement (Phase 2) — quand & comment (MacBook M4 Pro, 24GB)
**But** : entraînement complet après features + hard negatives + calibration.

- **Quand** :
  - Une fois **hard negatives** en place + nouvelles features intégrées.
  - Une fois la **calibration** ajoutée dans le pipeline d’entraînement.
- **Quoi** :
  - Re‑entraîner **ranker + classifier** avec les nouvelles features.
  - Entraîner le calibrateur (Platt ou isotonic) sur le split dev.
- **Modalités (adaptées Mac)** :
  - Entraînement CPU (XGBoost n’utilise pas MPS).
  - Limiter threads : `OMP_NUM_THREADS=6` / `MKL_NUM_THREADS=6`
  - Sémantique : **cohérence stricte** train/infer
    - Soit **tout OFF** (`XGB_SEMANTIC_ENABLED=0`) pour un modèle “no‑semantic” stable
    - Soit **tout ON** (`XGB_SEMANTIC_ENABLED=1`) *si et seulement si* tu acceptes le coût (embeddings)
  - Si sémantique ON : `XGB_SEMANTIC_DEVICE=mps` + `XGB_SEMANTIC_BATCH_SIZE=128`

**Commandes recommandées** :
```
# Génération dataset complet (hard negatives inclus)
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/generate_training_samples_v3.py --output data/samples_aligned_v4.parquet

# Entraînement complet
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/train_xgb_matcher_v2.py --samples data/samples_aligned_v4.parquet

# Diagnostic complet
python scripts/diagnostic_xgb_routing.py --input-path data/splits/test.csv
```

### Détails de code (Phase 2)
- **Base officielle** : `scripts/generate_training_samples_v3.py`
  - Remplacer la logique de hard negatives actuelle par **ranker top‑K**
  - Ajouter sampling “same_address_diff_name” et “same_name_diff_city”
  - Garantir que le GT est inclus même si le pré‑filtre l’écarte
  - Journaliser les cas **GT absent du pool** + taux d’exclusion

- `src/xgb_matcher/features.py`
  - Nouvelles features :
    - `address_density` (nb candidats à la même adresse)
    - `idf_name_score` (TF‑IDF char‑ngrams)
    - `numeric_token_match`
    - `legal_form_category`
    - `commonness_name_idf` (pénalise tokens fréquents : FRANCE, BATIMENT, SOCIETE)
    - **Features issues des règles de reranking (à intégrer au modèle)** :
      - `name_evidence_max` = max(jaro, token_overlap, acronym, contains, UL/PM)
      - `addr_perfect_strict` = addr_jaro>=0.97 & street_name_jaro>=0.95 & street_number_diff<=2
      - `addr_strong_no_num` = addr_jaro>=0.97 & street_number_diff==9999
      - `addr_only_risk` = addr_perfect_strict & name_evidence_max < 0.3
      - `semantic_only_risk` = name_semantic_max>0.7 & name_jaro_max<0.4 & token_overlap<0.2
      - `holding_mismatch` (bool, mots HOLDING/GROUPE/CORPORATION en cand mais pas en CRM)
      - `alias_perfect_addr` = alias_match & addr_perfect_strict
      - `name_uniqueness_gap` = best_name_score_top1 − best_name_score_top2 (pool-level)
      - `addr_density_is_high` = address_density > seuil (ex: 5)

- `scripts/train_xgb_matcher_v2.py`
  - Ajouter calibration post‑hoc (Platt / Isotonic)
  - Sauver calibrateur : `models/xgb_calibrator_<ts>.pkl`

- `scripts/infer_xgb_matcher_topk.py`
  - Charger calibrateur si présent pour scorer AUTO
  - Calculer meta‑features top‑K (gap, ratio, pool_size)

**Definition of Done Phase 2**
- Hard negatives réalistes dans le dataset
- Calibration activée (ECE < 2–3% visée)
- AUTO rules basées sur score calibré + meta‑features

**Validation Phase 2 (commandes)**
- `python scripts/generate_training_samples_v3.py --output data/samples_aligned_v4.parquet`
- `python scripts/train_xgb_matcher_v2.py --samples data/samples_aligned_v4.parquet`
- `python scripts/diagnostic_xgb_routing.py --input-path data/splits/test.csv`

## Phase 3 — SOTA (2–4 semaines)
**Objectif : pipeline robuste avec garanties quasi‑zéro FP**

1. **Pipeline 2‑étapes**
   - Ranker (pairwise/listwise) pour top‑K
   - Classifier calibré pour AUTO/REVIEW

2. **Multi‑blocking**
   - Ajout de blocs (nom+trigrammes, phonétique, rue)
   - Pour récupérer erreurs CP/INSEE sans explosion du pool

3. **Risk‑coverage dynamique**
   - Courbes par segment + seuils adaptatifs
   - Rejector sur incertitude

4. **Boucle Places → retraining**
   - Cas Places “très sûrs” → silver labels
   - Feedback pour améliorer hard negatives et recall

### Entraînement (Phase 3) — quand & comment (MacBook M4 Pro, 24GB)
**But** : produire le modèle final “prod‑ready”.

- **Quand** :
  - Quand la pipeline 2‑étapes + multi‑blocking + calibration sont stables.
  - Quand les diagnostics sont satisfaisants sur un set test dédié.
- **Quoi** :
  - Entraînement **final** ranker + classifier + calibrateur.
  - Génération d’une model card à partir des métriques et versions.
- **Modalités** :
  - Ré‑entraîner sur dataset complet (hard negatives + silver labels Places).
  - Sémantique ON/OFF **cohérente** avec l’inférence en prod.
  - Sur Mac M4 : threads limités + batch size raisonnable.

**Commandes recommandées** :
```
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/train_xgb_ranker.py
XGB_SEMANTIC_ENABLED=0 OMP_NUM_THREADS=6 python scripts/train_xgb_decider.py
python scripts/diagnostic_xgb_routing.py --pool-mode union --input-path data/splits/test.csv
```

### Détails de code (Phase 3)
- **Ranker 2‑étapes** : `scripts/train_xgb_ranker.py` + `scripts/train_xgb_decider.py`
- **Multi‑blocking** : `src/xgb_matcher/candidates.py` + nouveau module `blocking.py`
- **Risk‑coverage par segment** : `scripts/diagnostic_xgb_routing.py` + seuils per‑segment
- **Silver labels** : pipeline d’extraction depuis Places + intégration dans `generate_training_samples_v3.py`

**Definition of Done Phase 3**
- AUTO FP rate ≤ 0.1% (sur set test dédié)
- Coverage AUTO ≥ 80% (sur segments fiables)
- Reste routé en REVIEW/Places

**Validation Phase 3 (commandes)**
- `python scripts/train_xgb_ranker.py` + `python scripts/train_xgb_decider.py`
- `python scripts/diagnostic_xgb_routing.py --pool-mode union --input-path data/splits/test.csv`
- `python scripts/route_xgb_results.py --places-mode --input-path reports/xgb_infer_topk_phase3.csv --output-path reports/routed_phase3.csv`

---

## Plan SOTA complet (Blueprint opérationnel)
**Objectif** : converger vers un matching industriel, traçable, calibré, avec ~0 faux positif en AUTO.

### 1) Candidate Generation (Multi‑Blocking)
- **Bloquages simultanés** :
  - INSEE / CP (baseline)
  - trigrammes nom (TF‑IDF char‑ngrams)
  - tokens rue (street_name + typeVoie normalisés)
  - phonétique (Soundex/Metaphone FR)
- **Fusion + dédup** par SIRET
- **Outputs** : pool candidat enrichi + stats pool (taille moyenne, densité adresse)

### 2) Ranker (Recall‑first)
- **But** : Recall@10/20 très élevé (plafond de la perf finale)
- **Features** : cheap lexical + adresse (pas de sémantique si trop coûteuse)
- **KPI obligatoire** : Recall@10 ≥ 98%, Recall@20 ≥ 99% (sur set test)

### 3) Decider (Calibrated Classifier)
- **Features** : full + meta‑features (gap, ratio, pool size, name_evidence_max)
- **Calibration** : Platt/Isotonic sur dev + ECE < 2%
- **Routing** : seuils segmentés + garde‑fous (addr_only_risk, semantic_only_risk)

### 4) Fallback Places / Web (safe‑upgrade)
- **Uniquement** pour REVIEW / NO_MATCH
- **Upgrade** uniquement si preuves multi‑sources + validation SIRENE stricte
- **No false positives** comme invariant

### 5) Feedback Loop (Silver Labels)
- **Collecte** : cas Places “très sûrs”
- **Injection** : dans dataset v5 (avec tag provenance)
- **Monitoring** : drift (nom/adresse), taux d’upgrade Places

### 6) Monitoring & Reporting
- Risk‑coverage par segment (nom court/long, adresse complète/incomplète)
- ECE + calibration bins
- FP sentinel set (cases “fragiles” récurrents)
- **Model card auto‑générée** à partir des métriques + versions

### 7) Gates de mise en prod
- AUTO FP rate ≤ 0.1%
- Coverage AUTO ≥ 80% (sur segments fiables)
- Recall@20 ranker ≥ 99%
- ECE ≤ 2%
- Zéro régression sur le “sentinel set”

---

## Checklist “Nouvelle fenêtre de contexte”
1. Ouvrir `reports/entity_matching_audit.md` pour le plan en cours.
2. Vérifier les diagnostics :
   - `python scripts/fix_diagnostic_report.py`
   - Consulter `reports/diagnostic_report.md`
3. Identifier la phase active (Quick Wins / Sprint ML / SOTA)
4. Lister les fichiers à toucher (voir détails ci‑dessus)
5. Valider avec un run minimal (top‑k + routing)

---

# Model Card (à remplir après entraînement)

## 1) Informations générales
- **Nom du modèle** : XGB SIRETO Matcher
- **Version** : vX.Y.Z
- **Date d’entraînement** : YYYY‑MM‑DD
- **Owner** : SIRETO Team
- **Code / commit** : <hash>

## 2) Architecture
- **Stage 1** : XGBoost Ranker (pairwise/listwise)
- **Stage 2** : XGBoost Classifier (calibré)
- **Features** : lexicales + adresse + sémantiques + meta‑features
- **Calibration** : Platt / Isotonic (préciser)

## 3) Données d’entraînement
- **Source** : `data/entrainements.csv` + SIRENE parquet + UniteLegale
- **Taille** : N positives / M negatives
- **Stratégie négatifs** : hard negatives top‑K + random + co‑location
- **Split** : par SIREN (train/dev/test)

## 4) Évaluation
- **Metrics** : Precision@1, Recall@K, MRR, AUC, ECE, Risk‑Coverage
- **Résultats** :
  - Precision@1 = …
  - Recall@5 = …
  - ECE = …
  - Risk‑Coverage (θ=0.99, 0.995) = …

## 5) Seuils de décision
- **AUTO** : seuils segmentés (nom court/long, adresse complète/incomplète)
- **REVIEW** : zone d’incertitude
- **NO_MATCH** : score faible / conflits majeurs

## 6) Limites connues
- Adresses co‑localisées (centres commerciaux, ZI)
- Noms très courts ou génériques
- Mauvais CP/INSEE dans le CRM

## 7) Sécurité & usage
- Usage prévu : matching CRM → SIRET (France)
- Usage interdit : décisions légales/financières sans validation
- Process de revue : Google Places pour REVIEW

---

## Conclusion

Le système actuel est prometteur mais **structurellement fragile** à cause du train/serve skew, de la mauvaise calibration et d’un routing dépendant de SHAP. Les Quick Wins permettent d’améliorer immédiatement le **taux d’AUTO fiable**. Le Sprint ML et la phase SOTA rendent le système robuste, traçable et sûr, tout en maximisant l’automatisation.
