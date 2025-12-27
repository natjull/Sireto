# Audit approfondi — XGBoost Entity Matching (SIRETO)

## Résumé exécutif (version amendée)

Cet audit complète et corrige l’analyse initiale avec un regard **data science + production** (train/serve, calibration, pipeline Places). Objectif produit confirmé : **maximiser le taux d’AUTO‑MATCH XGBoost sans faux positifs**, le reste part en **REVIEW** et est **rematché via Google Places** pour retrouver le bon SIRET.

**Constats majeurs (priorité haute)**
1. **Skew train/serve avéré** : le ranker est entraîné avec des features sémantiques, mais l’inférence stage‑1 les désactive → risque de rater le vrai candidat avant la phase de scoring. (`scripts/train_xgb_matcher_v2.py`, `scripts/infer_xgb_matcher_topk.py`)
2. **Routing dépendant de SHAP** : `_route_xgb()` lit des features depuis `shap`, mais le CSV top‑k ne contient pas ces features si `--with-shap` n’est pas activé → promotions AUTO rarement déclenchées. (`scripts/route_xgb_results.py`, `scripts/infer_xgb_matcher_topk.py`)
3. **Calibration très mauvaise** : les scores 0.99+ ont ~10% d’erreur réelle (sur le set diagnostiqué) → seuils AUTO trop optimistes. (`reports/diagnostic_report.md`, `reports/diagnostic_analysis.json`)
4. **Hard negatives annoncés mais pas implémentés** : l’échantillonnage utilise un score heuristique au lieu d’un vrai top‑K ranker → faible apprentissage des confusions critiques (même adresse / nom différent). (`scripts/generate_training_samples.py`)
5. **Risque “adresse‑seule”** : le modèle sur‑pondère l’adresse et produit des faux positifs co‑localisés (centre commercial / ZI), exactement les erreurs observées.

**Conséquence** : la politique actuelle d’AUTO est structurellement risquée. La bonne trajectoire SOTA passe par **(1) alignement train/serve**, **(2) hard negative mining real‑world**, **(3) calibration + seuils segmentés**, **(4) pipeline en 2 étages** (ranker → classif décision) et **(5) boucles de feedback depuis Places**.

---

## Sources & périmètre analysés

- `reports/entity_matching_audit.md` (version initiale)
- `reports/diagnostic_report.md`
- `reports/diagnostic_analysis.json`
- `reports/diagnostic_plots.png`
- Code: `scripts/train_xgb_matcher_v2.py`, `scripts/infer_xgb_matcher_topk.py`, `scripts/generate_training_samples.py`, `scripts/route_xgb_results.py`, `src/xgb_matcher/*`, `src/pipe_v6/places_*`

> **Note critique importante** : `diagnostic_report.md` et `diagnostic_analysis.json` sont **partiellement incohérents** sur la *coverage*. Ex. à θ=0.995 :
> - `diagnostic_report.md` indique **73.1% couverture (98/134)**
> - `diagnostic_analysis.json` indique **coverage=0.4066** avec **auto_count=98**
> 
> **Interprétation probable** : la colonne *coverage* du JSON est calculée contre un dénominateur différent (peut‑être l’ensemble CRM total), ou bug de calcul. Les *comptes* (auto_count=98, errors=4) sont cohérents avec 98/134=73.1%. Il faut vérifier la logique du script de diagnostic avant d’utiliser la coverage comme KPI.

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

---

## Architecture cible (SOTA adaptée à ton pipeline)

### Étape 1 : Ranker rapide (no‑semantic)
- Optimisé pour **rappeler le bon candidat** dans le top‑K.
- Features “cheap” uniquement (lexical + adresse + tokens).

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

## Phase 0 — Pré‑requis (1–2 jours)
- **Corriger la génération des diagnostics** : coverage, risk‑coverage, calibration ECE.
- **Exporter les features nécessaires au routing** directement dans le top‑k CSV.
- **Aligner train/serve** du ranker (features identiques).

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
