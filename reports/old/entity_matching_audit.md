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
2. Les cas **REVIEW** passent dans un **second pipeline Places** pour “rattraper” le bon SIRET.
   - **Pas de NO_MATCH avant Places** : la décision est **AUTO vs REVIEW** en sortie XGB.
   - **NO_MATCH n’existe qu’après Places** si aucune promotion n’est possible.
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
d**Mission** : hard negative mining + nouvelles features + calibration.  
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

### Ctx4 — SOTA Routing (Phase 4)
**Mission** : Maximiser le taux AUTO sans faux positif, en conscience du coût Places aval.
**Entrées** : `scripts/route_xgb_results.py`, `src/pipe_v6/places_orchestrator.py`, ground truth élargi
**Sorties attendues** :
- Routing cost‑aware (minimise appels Serper)
- Règles de certitude absolue implémentées
- Segmentation + seuils appris sur ground truth diversifié
- Résolution same‑SIREN automatique
- Métriques : AUTO rate, FP rate, Places call rate, coût estimé
**Critère d'arrêt** : Phase 4 "Definition of Done" OK + `reports/handover.md` mis à jour.

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
    - CSV avec ground truth (`ground_truth_siret`) : **LEGACY** : `data/old/2026-01-11_splits/test.csv` OU extraire depuis `data/samples_v4_with_ranker.parquet` (split='test')
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
  - **Commandes (legacy)** :
    - `python scripts/diagnostic_xgb_routing.py --input-path data/old/2026-01-11_splits/test.csv`
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
- `python scripts/diagnostic_xgb_routing.py --input-path data/old/2026-01-11_splits/test.csv --limit 200` (legacy; préférer extraire split test du parquet)

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
- `python scripts/diagnostic_xgb_routing.py --input-path data/old/2026-01-11_splits/test.csv` (legacy; préférer extraire split test du parquet)

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
- `python scripts/diagnostic_xgb_routing.py --pool-mode union --input-path data/old/2026-01-11_splits/test.csv` (legacy; préférer extraire split test du parquet)
- `python scripts/route_xgb_results.py --places-mode --input-path reports/xgb_infer_topk_phase3.csv --output-path reports/routed_phase3.csv`

## Phase 4 — SOTA Routing & Cost‑Aware Decision (1–2 semaines)
**Objectif : Maximiser AUTO sans FP, minimiser les appels Places (coût Serper)**

**Note clé** : Le fallback Places adopte un **mode Places‑as‑CRM**.
Le **decider XGB est réutilisé tel quel** (mêmes features + calibrator),
sur un pool **recall@20 + arm_a + arm_b**, avec promotion MATCH_PLACES
pilotée par une **calibration automatique**.

### Contexte : Pipeline Places aval et coûts

Le routing détermine ce qui part en **AUTO** (gratuit, instantané) vs **REVIEW** (appel Serper Places payant).

**Architecture post‑routing** :
```
┌──────────────┐     ┌─────────────────────────────────────────────────────────┐
│   XGBoost    │     │                    ROUTING                              │
│   Inference  │────►├────────────────────────────────────────────────────────►│
│   (top‑k)    │     │  AUTO ────► Résultat final (coût = 0)                   │
└──────────────┘     │                                                         │
                     │  REVIEW ───► Places API (Serper) ───► Rescoring ───►    │
                     │              │                          │               │
                     │              │ ~0.001$/req              │               │
                     │              ▼                          ▼               │
                     │         Validation     MATCH_PLACES ou NO_MATCH        │
                     │                                                         │
                     │  (pas de NO_MATCH avant Places)                        │
                     └─────────────────────────────────────────────────────────┘
```

**Coûts Serper.dev** (janvier 2026) :
- Plan gratuit : 2 500 requêtes/mois
- Plan payant : ~0.001$ / requête (1 000 req = 1$)
- Rate limit : 1 req/sec (respecté dans `serper_places_client.py`)

**Équation économique** :
```
Coût_total = nb_REVIEW × coût_par_requête
Objectif = Maximiser(AUTO_rate) sous contrainte FP_rate ≈ 0 ET Coût_total < Budget
```

### Stratégie Phase 4 : Routing Cost‑Aware

**Principe** : Ne pas envoyer en REVIEW des cas qui ne bénéficieront pas de Places.

| Cas | Places utile ? | Décision optimale |
|-----|----------------|-------------------|
| Match parfait (lexical + adresse) | Non | AUTO |
| Ambiguïté same‑SIREN | Non (même entreprise) | AUTO (prendre OUVERT) |
| Score élevé, faible gap | Peut‑être | REVIEW si budget |
| Nom court/générique, score moyen | Oui | REVIEW |
| Aucun candidat viable | Non | REVIEW (Places tentée, NO_MATCH seulement après échec) |
| Adresse incomplète/incorrecte | Limité | REVIEW avec flag "low_places_value" |

### Stratégie Phase 4B : Places‑as‑CRM (Decider identique)

**Objectif** : utiliser le **decider XGB tel quel** (mêmes features, même ordre, même calibrator)
en remplaçant l'entrée CRM par le **résultat Places** (pseudo‑CRM).  
Pas de roue réinventée : c'est le **même pipeline d'inférence** que `infer_xgb_two_stage.py`.

**Invariants (obligatoires)** :
- `preprocess_crm_row()` + `make_features_from_preprocessed()` inchangés
- `feature_order` issu du meta decider
- même modèle decider + calibrator (chemins **explicites**)
- même gating sémantique (`XGB_SEMANTIC_ENABLED` + `semantic_gate_allows`)
- mêmes filtres candidats (drop_unnamed/exclude_closed/dedupe)
- pool large : **recall@20** + arm_a + arm_b

**Différences acceptées** :
- CRM remplacé par Places (`crm_name = places.title`, `crm_address = places.address`)
- mini‑gate CRM↔Places conservé (CP/ville + overlap minimum)

### 1) Règles de certitude absolue (FP impossible)

Ces règles ont priorité maximale et bypassent tous les autres checks.

```python
def is_absolute_certainty(row: pd.Series) -> bool:
    """
    Règles où un faux positif est théoriquement impossible.
    Ces cas vont directement en AUTO sans passer par les seuils.
    """
    score = row['score']
    jaro = row['name_jaro_max']
    tok_overlap = row['name_token_overlap_max']
    addr_jaro = row['addr_jaro']
    street_diff = row['street_number_diff']
    semantic = row['name_semantic_max']

    # R1: Match lexical quasi‑parfait + adresse parfaite
    # Ex: "GE FRUITS" → "GE FRUITS" @ même adresse
    if (jaro >= 0.98 and tok_overlap >= 0.90 and
        addr_jaro >= 0.98 and street_diff == 0 and
        score >= 0.90):
        return True

    # R2: Nom identique (jaro=1) + ville identique + score élevé
    # Ex: "CHAMPILYON" → "CHAMPILYON" à Corbas
    if (jaro == 1.0 and score >= 0.95):
        return True

    # R3: Score parfait (1.0) + preuve lexicale forte
    # Le modèle est absolument certain ET on a une preuve indépendante
    if (score >= 0.999 and (jaro >= 0.90 or tok_overlap >= 0.70)):
        return True

    # R4: Contains match parfait + score élevé
    # Ex: "TIMCOD RHONE-ALPES" contenu dans nom candidat
    if (row['name_crm_contains_cand_max'] >= 0.95 and
        score >= 0.95 and addr_jaro >= 0.80):
        return True

    return False
```

### 2) Résolution automatique Same‑SIREN

Quand les top candidats appartiennent au même SIREN, c'est la même entreprise avec des établissements différents.

```python
def resolve_same_siren(top_k: List[dict]) -> Tuple[dict, str]:
    """
    Quand top1 et top2 ont le même SIREN, résoudre automatiquement.
    Retourne (candidat choisi, décision).
    """
    if len(top_k) < 2:
        return top_k[0], None

    siren_top1 = top_k[0]['siret'][:9]
    siren_top2 = top_k[1]['siret'][:9]

    # Pas d'ambiguïté si SIREN différents
    if siren_top1 != siren_top2:
        return top_k[0], None

    # Même SIREN : trier par (état, score) et prendre le meilleur
    same_siren = [c for c in top_k if c['siret'][:9] == siren_top1]

    # Priorité : OUVERT > FERME, puis score décroissant
    def sort_key(c):
        state_priority = 0 if c['candidate_state'] == 'OUVERT' else 1
        return (state_priority, -c['score'])

    same_siren.sort(key=sort_key)
    best = same_siren[0]

    # Conditions pour AUTO : score suffisant + preuve lexicale
    if (top_k[0]['score'] >= 0.90 and
        top_k[0]['name_jaro_max'] >= 0.80):
        return best, "AUTO_SAME_SIREN"

    return best, "REVIEW_SAME_SIREN"
```

### 3) Segmentation et seuils cost‑aware

Adapter les seuils selon la "valeur attendue" d'un appel Places.

```python
def get_segment_config(row: pd.Series) -> dict:
    """
    Retourne la configuration de routing par segment.
    Inclut le seuil AUTO et un flag "places_value" (utilité de Places).
    """
    name_words = count_words(row['crm_name'])
    name_idf = row.get('idf_name', 0)  # Rareté du nom
    addr_complete = has_complete_address(row)
    pool_size = row.get('pool_size', 100)
    score_gap = row.get('score_gap', 0)

    # Segment 1: Nom unique + adresse complète → Places peu utile
    if name_idf >= 5 and addr_complete:
        return {
            'threshold': 0.95,      # Seuil bas (confiance élevée)
            'places_value': 'low',  # Places n'apportera rien
            'gap_min': 0.03,        # Gap relaxé
        }

    # Segment 2: Nom courant + adresse complète → Places moyennement utile
    if name_idf < 5 and addr_complete:
        return {
            'threshold': 0.98,
            'places_value': 'medium',
            'gap_min': 0.05,
        }

    # Segment 3: Nom unique + adresse partielle → Places peut aider
    if name_idf >= 5 and not addr_complete:
        return {
            'threshold': 0.97,
            'places_value': 'medium',
            'gap_min': 0.05,
        }

    # Segment 4: Nom courant + adresse partielle → Places très utile
    if name_idf < 5 and not addr_complete:
        return {
            'threshold': 0.995,     # Seuil très strict
            'places_value': 'high', # Places peut vraiment aider
            'gap_min': 0.08,
        }

    # Segment 5: Nom très court (1-2 mots) → Toujours risqué
    if name_words <= 2:
        return {
            'threshold': 0.995,
            'places_value': 'high',
            'gap_min': 0.10,
        }

    # Défaut
    return {
        'threshold': 0.99,
        'places_value': 'medium',
        'gap_min': 0.05,
    }
```

### 4) Routing principal cost‑aware

```python
def route_cost_aware(row: pd.Series, top_k: List[dict], budget_mode: str = "normal") -> str:
    """
    Routing principal avec conscience du coût Places.

    budget_mode:
    - "aggressive": Minimise les appels Places (seuils stricts pour REVIEW)
    - "normal": Équilibre AUTO/REVIEW
    - "permissive": Maximise le recall Places (plus de REVIEW)
    """

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 0: Certitude absolue (bypass tout)
    # ─────────────────────────────────────────────────────────────
    if is_absolute_certainty(row):
        return "AUTO_CERTAIN"

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 1: Résolution same‑SIREN
    # ─────────────────────────────────────────────────────────────
    resolved, siren_decision = resolve_same_siren(top_k)
    if siren_decision:
        return siren_decision  # "AUTO_SAME_SIREN" ou "REVIEW_SAME_SIREN"

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 2: Règles de blocage (force REVIEW)
    # ─────────────────────────────────────────────────────────────

    # B1: Aucune preuve lexicale
    if row['name_jaro_max'] < 0.50 and row['name_token_overlap_max'] < 0.20:
        return "REVIEW"  # Places peut trouver le bon nom

    # B2: Semantic‑only match
    if is_semantic_only_match(row):
        if row['score'] >= 0.998:
            pass  # Bypass si modèle très confiant
        else:
            return "REVIEW"

    # B3: Address‑only match (risque co‑location)
    if row['name_token_overlap_max'] < 0.10 and row['addr_token_overlap'] > 0.80:
        return "REVIEW"

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 3: Configuration par segment
    # ─────────────────────────────────────────────────────────────
    config = get_segment_config(row)
    threshold = config['threshold']
    gap_min = config['gap_min']
    places_value = config['places_value']

    # Ajuster selon budget_mode
    if budget_mode == "aggressive":
        threshold *= 0.995  # Légèrement plus permissif pour AUTO
        gap_min *= 0.8      # Gap relaxé
    elif budget_mode == "permissive":
        threshold *= 1.005  # Plus strict, plus de REVIEW
        gap_min *= 1.2

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 4: Règles de promotion (force AUTO)
    # ─────────────────────────────────────────────────────────────

    score = row['score']

    # P1: Strong establishment match
    if score >= 0.95 and row['name_sim_max_etab'] >= 0.70:
        return "AUTO"

    # P2: Contains match
    if score >= 0.90 and row['name_crm_contains_cand_max'] >= 0.90:
        return "AUTO"

    # P3: PM dirigeant match
    if score >= 0.95 and row['name_sim_max_pm_dirigeant'] >= 0.70:
        return "AUTO"

    # P4: High token overlap
    if score >= 0.98 and row['name_token_overlap_max'] >= 0.50:
        return "AUTO"

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 5: Vérification gap (ambiguïté)
    # ─────────────────────────────────────────────────────────────

    score_gap = row.get('score_gap', 0)
    score_ratio = row.get('score_ratio', 1)

    if score_gap < gap_min:
        # Ambiguïté forte — mais Places utile seulement si les candidats sont vraiment différents
        if places_value == "low":
            # Places n'aidera pas, on REVIEW quand même pour sécurité
            return "REVIEW_LOW_PLACES_VALUE"
        return "REVIEW"

    if score_ratio < 1.02 + (0.02 if budget_mode == "aggressive" else 0):
        return "REVIEW"

    # ─────────────────────────────────────────────────────────────
    # ÉTAPE 6: Décision finale par seuil
    # ─────────────────────────────────────────────────────────────

    if score >= threshold:
        return "AUTO"

    # Score sous seuil → REVIEW (pas de NO_MATCH avant Places)
    if budget_mode == "aggressive" and places_value == "low":
        return "AUTO_BUDGET"  # Flag pour audit
    return "REVIEW"
```

### 4B) Calibration AUTO vs REVIEW (automatique)

**Principe** : le routing ne sort **que AUTO ou REVIEW**.  
La calibration cherche des seuils (par segment) qui maximisent l’AUTO **sous contrainte FPR**,
et tout le reste devient REVIEW (pas de NO_MATCH pré‑Places).

Inputs :
- set GT diversifié (holdout)
- features de routing + scores calibrés

Outputs :
- `configs/routing_thresholds.yaml` avec seuils AUTO par segment
- metrics : auto_rate, fp_rate, places_call_rate

### 5) Promotion MATCH_PLACES (calibration automatique)

**But** : convertir un score decider (Places‑as‑CRM) en MATCH_PLACES **sans faux positif**,
en calibrant automatiquement les seuils sur un set GT.

```python
# Conditions (toutes obligatoires)
# 1) Gate CRM↔Places OK (mini‑gate)
# 2) address_close == True (CP + rue + numéro OU distance <= seuil)
# 3) score_places_top1 >= score_min  (calibré)
# 4) gap_places >= gap_min           (calibré)
#
# Calibration :
# - grid search sur score_min × gap_min
# - objectif : FP rate <= 0.1% (ou target_fpr)
# - segmentation optionnelle : adresse complète vs partielle
```

### 6) Métriques et monitoring

```python
ROUTING_METRICS = {
    # Volume
    'total_records': 0,
    'auto_count': 0,
    'auto_certain_count': 0,
    'auto_same_siren_count': 0,
    'auto_budget_count': 0,
    'review_count': 0,
    'review_low_places_value_count': 0,
    'no_match_final_count': 0,  # uniquement après Places

    # Qualité (requiert ground truth)
    'auto_correct': 0,
    'auto_incorrect': 0,  # FP !
    'review_would_be_correct': 0,  # AUTO manqués

    # Coût
    'estimated_places_calls': 0,  # = review_count (pas de NO_MATCH pré‑Places)
    'estimated_cost_usd': 0.0,    # = review_count × 0.001

    # Efficacité
    'auto_rate': 0.0,             # auto_count / total
    'fp_rate': 0.0,               # auto_incorrect / auto_count
    'places_call_rate': 0.0,      # review_count / total
    'cost_per_record': 0.0,       # estimated_cost / total
}
```

### Entraînement (Phase 4) — quand & comment

**But** : Valider les seuils sur un ground truth diversifié, pas Corbas/Décines.

- **Quand** :
  - Une fois le ground truth élargi disponible (voir section "Élargir le ground truth")
  - Après avoir collecté des labels sur d'autres villes/secteurs
- **Quoi** :
  - Pas de re‑training XGBoost (modèle fixé)
  - Calibration **automatique** des seuils routing + Places (score_min/gap_min)
  - Validation des règles de certitude absolue (FP = 0 garanti)
- **Modalités** :
  - Cross‑validation par ville/région pour éviter l'overfitting géographique
  - Métriques séparées par segment
  - A/B testing des budget_modes

**Commandes recommandées** :
```bash
# Générer ground truth élargi (en utilisant le parquet canonique)
python scripts/build_evaluation_dataset.py \
    --sources crm_history,places_validated \
    --min-samples-per-segment 100 \
    --output data/evaluation_dataset.parquet

# Calibrer les seuils par segment
python scripts/calibrate_routing_thresholds.py \
    --eval-data data/evaluation_dataset.parquet \
    --output-config configs/routing_thresholds.yaml \
    --target-fpr 0.001

# Calibrer les seuils Places (Places‑as‑CRM)
python scripts/calibrate_places_thresholds.py \
    --eval-data data/evaluation_dataset.parquet \
    --decider-model models/xgb_decider_20260103_132351.json \
    --calibrator-path models/xgb_decider_calibrator_isotonic_20260103_132351.pkl \
    --output-config configs/places_thresholds.yaml \
    --target-fpr 0.001

# Valider sur holdout
python scripts/evaluate_routing.py \
    --eval-data data/evaluation_holdout.parquet \
    --thresholds configs/routing_thresholds.yaml \
    --output reports/routing_evaluation.json

# Valider Places‑as‑CRM sur holdout
python scripts/evaluate_places_matching.py \
    --eval-data data/evaluation_holdout.parquet \
    --decider-model models/xgb_decider_20260103_132351.json \
    --calibrator-path models/xgb_decider_calibrator_isotonic_20260103_132351.pkl \
    --places-thresholds configs/places_thresholds.yaml \
    --output reports/places_evaluation_phase4.json

# Simuler les coûts Places
python scripts/simulate_places_costs.py \
    --routed-csv reports/routed_phase4.csv \
    --cost-per-call 0.001 \
    --monthly-volume 10000
```

**Note (2026-01-11)** : `data/splits/` a été archivé dans `data/old/2026-01-11_splits/`. La source canonique pour splits train/dev/test est désormais `data/samples_v4_with_ranker.parquet` (colonne `split`).

### Détails de code (Phase 4)

**Fichiers impactés + contrat attendu**

- `scripts/route_xgb_results.py`
  - Refactoring de `_route_xgb()` en `route_cost_aware()`
  - Ajout des règles de certitude absolue
  - Ajout de la résolution same‑SIREN
  - Ajout de `--budget-mode` (aggressive/normal/permissive)
  - Ajout d’options explicites pour modèles decider Places (`--decider-model`, `--calibrator-path`)
  - Export des métriques de routing

- `scripts/calibrate_routing_thresholds.py` (nouveau)
  - Entrée : ground truth élargi
  - Sortie : `configs/routing_thresholds.yaml`
  - Logique : Trouver seuil par segment où FPR ≤ 0.001

- `scripts/calibrate_places_thresholds.py` (nouveau)
  - Entrée : ground truth élargi + résultats Places‑as‑CRM
  - Sortie : `configs/places_thresholds.yaml`
  - Logique : grid‑search `score_min` × `gap_min` sous contrainte FPR ≤ 0.001

- `scripts/evaluate_places_matching.py` (nouveau)
  - Entrée : holdout + thresholds Places + modèles decider explicites
  - Sortie : `reports/places_evaluation_phase4.json`

- `src/pipe_v6/places_xgb_rescorer.py`
  - Ajouter `crm_mode="places|original"` (default: places pour Phase 4)
  - Réutiliser **exactement** la logique d'inférence du decider (feature order + calibrator)
  - Chemins de modèles **explicites** (pas d’auto‑latest)

- `src/pipe_v6/places_orchestrator.py`
  - Pool **recall@20** + arm_a + arm_b
  - Mini‑gate CRM↔Places (CP/ville + overlap minimum)
  - Promotion MATCH_PLACES si `address_close` + `score_min` + `gap_min`

- `src/pipe_v6/places_validator.py`
  - Ajouter `address_close()` (CP + rue + numéro OU distance <= seuil)
  - Paramètres configurables : `places_addr_jaro_min`, `places_distance_max_m`

- `scripts/evaluate_routing.py` (nouveau)
  - Entrée : holdout + thresholds
  - Sortie : métriques détaillées par segment
  - Inclut simulation de coût

- `scripts/build_evaluation_dataset.py` (nouveau)
  - Collecte de labels depuis : historique CRM validé, Places API (high confidence), validation manuelle
  - Stratification par : ville, secteur, longueur nom, complétude adresse

- `configs/routing_thresholds.yaml` (nouveau)
  ```yaml
  segments:
    unique_name_full_addr:
      threshold: 0.95
      gap_min: 0.03
      places_value: low
    common_name_full_addr:
      threshold: 0.98
      gap_min: 0.05
      places_value: medium
    # ... autres segments

  certainty_rules:
    enabled: true
    min_jaro_perfect: 0.98
    min_addr_jaro_perfect: 0.98

  same_siren_resolution:
    enabled: true
    min_score: 0.90
    prefer_ouvert: true

  budget:
    mode: normal  # aggressive/normal/permissive
    max_monthly_calls: 2500
    alert_threshold: 0.8
  ```

- `configs/places_thresholds.yaml` (nouveau)
  ```yaml
  promotion:
    score_min: 0.97
    gap_min: 0.05
    addr_jaro_min: 0.90
    distance_max_m: 80
  gating:
    require_postcode: true
    min_addr_overlap: 0.35
  ```

### Definition of Done Phase 4

- [ ] Règles de certitude absolue implémentées et validées (FP = 0 sur test set)
- [ ] Résolution same‑SIREN automatique active
- [ ] Seuils calibrés sur ground truth diversifié (pas seulement Corbas/Décines)
- [ ] Routing XGB = **AUTO/REVIEW uniquement** (NO_MATCH après Places)
- [ ] Seuils Places calibrés automatiquement (`score_min`, `gap_min`) + `address_close`
- [ ] Métriques de coût intégrées au reporting
- [ ] `budget_mode` fonctionnel (aggressive/normal/permissive)
- [ ] Documentation des seuils dans `configs/routing_thresholds.yaml`
- [ ] Documentation Places dans `configs/places_thresholds.yaml`
- [ ] Chemins de modèles decider explicites (pas d’auto‑latest) pour Places‑as‑CRM
- [ ] Tests unitaires pour chaque règle de routing
- [ ] AUTO FP rate ≤ 0.1% sur holdout diversifié
- [ ] Réduction du Places call rate d'au moins 20% vs Phase 3

### Validation Phase 4 (commandes)

```bash
# Inférence standard (utiliser un CSV CRM réel, pas les splits d'entraînement)
XGB_SEMANTIC_ENABLED=1 python scripts/infer_xgb_two_stage.py \
    --crm-path data/evaluation_holdout.csv \
    --partitions-dir data/candidates_v4_fixed \
    --output-path reports/xgb_infer_phase4.csv \
    --top-k 20 \
    --decider-model models/xgb_decider_20260103_132351.json \
    --calibrator-path models/xgb_decider_calibrator_isotonic_20260103_132351.pkl

# Routing cost‑aware
python scripts/route_xgb_results.py \
    --input-path reports/xgb_infer_phase4.csv \
    --output-path reports/routed_phase4.csv \
    --budget-mode normal \
    --thresholds configs/routing_thresholds.yaml

# Évaluation (requiert ground truth)
python scripts/evaluate_routing.py \
    --routed-csv reports/routed_phase4.csv \
    --ground-truth data/evaluation_holdout.csv \
    --output reports/routing_evaluation_phase4.json

# Évaluation Places‑as‑CRM (requiert Places thresholds)
python scripts/evaluate_places_matching.py \
    --eval-data data/evaluation_holdout.parquet \
    --decider-model models/xgb_decider_20260103_132351.json \
    --calibrator-path models/xgb_decider_calibrator_isotonic_20260103_132351.pkl \
    --places-thresholds configs/places_thresholds.yaml \
    --output reports/places_evaluation_phase4.json

# Simulation coût mensuel
python scripts/simulate_places_costs.py \
    --routed-csv reports/routed_phase4.csv \
    --monthly-volume 10000 \
    --cost-per-call 0.001
```

**Note (2026-01-11)** : Les splits train/dev/test sont définis dans `data/samples_v4_with_ranker.parquet` (colonne `split`). Pour tester sur le split test historique, utiliser `data/old/2026-01-11_splits/test.csv` (archivé).

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
- **Uniquement** pour REVIEW (pas de NO_MATCH pré‑Places)
- **Places‑as‑CRM** : decider XGB identique (mêmes features + calibrator)
- **Upgrade** uniquement si `address_close` + seuils Places calibrés
- **NO_MATCH** uniquement après échec Places

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

## Checklist "Nouvelle fenêtre de contexte"
1. Ouvrir `reports/entity_matching_audit.md` pour le plan en cours.
2. Vérifier les diagnostics :
   - `python scripts/fix_diagnostic_report.py`
   - Consulter `reports/diagnostic_report.md`
3. Identifier la phase active (Quick Wins / Sprint ML / SOTA / SOTA Routing)
4. Lister les fichiers à toucher (voir détails ci‑dessus)
5. Valider avec un run minimal (top‑k + routing)
6. Pour Phase 4 : vérifier le budget Places restant avant de lancer des REVIEW

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
- **NO_MATCH** : uniquement après Places si aucune promotion n’est possible

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

Le système actuel est prometteur mais **structurellement fragile** à cause du train/serve skew, de la mauvaise calibration et d'un routing dépendant de SHAP. Les Quick Wins permettent d'améliorer immédiatement le **taux d'AUTO fiable**. Le Sprint ML et la phase SOTA rendent le système robuste, traçable et sûr, tout en maximisant l'automatisation.

**Phase 4 (SOTA Routing)** complète le pipeline avec une vision **cost‑aware** :
- Les règles de certitude absolue capturent les "easy wins" sans risque
- La résolution same‑SIREN élimine les faux positifs d'ambiguïté intra‑entreprise
- La segmentation par utilité Places évite les appels API inutiles
- Le budget_mode permet d'ajuster le compromis AUTO/coût selon les contraintes

**Objectifs finaux** :
- AUTO rate ≥ 60% avec FP rate ≤ 0.1%
- Places call rate ≤ 30% du volume total
- Coût Places mensuel < budget alloué
