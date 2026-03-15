# Approches alternatives pour Sireto

## Contexte

Exploration de deux visions radicalement différentes du projet Sireto :
1. **Approche Karpathy** : end-to-end neural, autoresearch, petits LLM locaux
2. **Approche INSEE** : rigueur statistique, méthodes publiques éprouvées

## Approche actuelle : Sireto V7

Pipeline ML classique en 3 étages :
- Retrieval TF-IDF → XGBoost ranker → XGBoost risk routing
- 74.5% AUTO à 99.84% de précision
- ~68 features artisanales, modèles supervisés, déterministe et auditable

---

## 1. L'approche Karpathy : "Let the model figure it out"

Philosophie : remplacer le code heuristique par des réseaux de neurones entraînés
end-to-end, itérer vite avec des boucles d'évaluation automatiques.

### 1.1 Small local LLM comme matcher direct

Au lieu de 3 étages XGBoost + 68 features artisanales, un petit LLM fine-tuné
(Mistral 7B, Phi-3, Qwen2.5-3B) sur les ~2500 paires annotées + hard negatives
synthétiques. Format LoRA/QLoRA, tourne sur le M4 Pro en local.

Le modèle apprend *implicitement* les features (Jaro-Winkler, token overlap, etc.)
sans les coder. Avantage : zéro feature engineering, le modèle généralise sur les
cas bizarres (abréviations, erreurs de saisie, noms commerciaux vs dénominations).

### 1.2 Autoresearch / self-improving loop

La philosophie "Software 2.0" poussée à l'extrême :

```
while not converged:
    1. Le LLM matche un batch de CRM
    2. Un "juge" (LLM plus gros ou humain) évalue les résultats
    3. Les erreurs deviennent du training data
    4. Fine-tune le modèle sur les erreurs
    5. Goto 1
```

Eval-driven development : la boucle `eval → train → eval` remplace l'ingénierie
manuelle. Le modèle s'auto-améliore sur les cas difficiles.

### 1.3 Embeddings appris en contrastif

Au lieu de TF-IDF + BERT pré-entraîné :
- Contrastive fine-tuning (SimCSE/InfoNCE) sur les paires CRM↔SIRENE
- Le retrieval devient un simple ANN search (FAISS) dans l'espace appris
- Un seul modèle fait retrieval + ranking

### 1.4 Architecture concrète

```
┌──────────────────────────────────────────────┐
│  Embedding model fine-tuné (e5-small, 33M)   │
│  Entraîné en contrastif sur CRM↔SIRENE       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  FAISS index sur 10M établissements SIRENE   │
│  Retrieval top-100 par cosine similarity     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Cross-encoder local (Phi-3-mini fine-tuné)  │
│  Rerank top-100 → score de confiance         │
│  Routing AUTO/REVIEW par seuil calibré       │
└──────────────────────────────────────────────┘
```

### 1.5 Forces et faiblesses

| Aspect | Pro | Contra |
|---|---|---|
| Généralisation | Gère les cas non couverts par les features | Besoin de plus de données |
| Simplicité du code | ~200 lignes vs ~5000 | Boîte noire, moins auditable |
| Itération | Boucle eval→train automatique | Coût GPU pour fine-tuning |
| Hardware | Phi-3/Qwen2.5-3B tourne sur M4 | Latence ~500ms/query vs ~100ms |

---

## 2. L'approche INSEE : rigueur statistique et méthodes publiques

Un data scientist de l'INSEE aborderait ça avec la culture des statisticiens
publics : reproductibilité, méthodes éprouvées, documentation exhaustive.

### 2.1 Le modèle Fellegi-Sunter (1969)

L'INSEE utilise historiquement le modèle probabiliste de Fellegi-Sunter :

```r
library(reclin2)
pairs <- pair_blocking(crm, sirene, on = "code_postal")
pairs <- compare_pairs(pairs,
  by = c("nom", "adresse", "ville"),
  comparators = list(jaro_winkler(), jaro_winkler(), jaro_winkler())
)
model <- problink_em(pairs)  # EM algorithm
pairs <- predict(model, pairs, type = "mweights")
linked <- select_greedy(pairs, threshold = 8.5)
```

Pas de ML au sens moderne : un modèle EM estime les probabilités de match/non-match.
Les poids sont interprétables (log-likelihood ratios).

### 2.2 Blocking par Code Officiel Géographique (COG)

- Utilisation du COG comme clé de blocking stricte
- Table de correspondance code postal → commune(s) maintenue par l'INSEE
- Gestion explicite des fusions de communes
- Pas de fallback département

### 2.3 Qualité des données en amont (60% du temps)

- Normalisation RNVP (Restructuration, Normalisation, Validation Postale)
- Standardisation des formes juridiques
- Détection des doublons CRM en amont
- Utilisation du FANTOIR (fichier des voies)
- Distinction explicite enseigne / dénomination / sigle

### 2.4 Validation par sondage stratifié

- Plan de sondage stratifié par secteur × taille × zone géographique
- Estimation de la précision avec intervalles de confiance
- Redressement par calage sur les marges connues
- Note méthodologique publiée

### 2.5 Architecture concrète

```
┌─────────────────────────────────────────────┐
│  Nettoyage RNVP + normalisation COG         │
│  (standardisation adresses + formes jurid.) │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Blocking exact: code commune COG           │
│  + blocking phonétique (Soundex FR)         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Fellegi-Sunter (EM) sur paires bloquées   │
│  Variables: nom (JW), adresse (JW),         │
│  code postal (exact), enseigne (JW)         │
│  → poids composite (log-likelihood ratio)   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Triple seuillage:                          │
│  - Score > 12 → MATCH AUTO                  │
│  - Score 6-12 → REVUE MANUELLE              │
│  - Score < 6  → NON-MATCH                   │
│  Seuils calibrés par validation croisée     │
└─────────────────────────────────────────────┘
```

### 2.5 Forces et faiblesses

| Aspect | Pro | Contra |
|---|---|---|
| Interprétabilité | Chaque poids a un sens probabiliste | Moins performant que XGBoost |
| Robustesse | Méthode éprouvée depuis 50 ans | Ne capture pas les interactions |
| Reproductibilité | Documentation INSEE-grade | Rigide, peu adaptable |
| Données | Nettoyage RNVP supérieur | Lent, manuel, coûteux |

---

## 3. Synthèse comparative

| Dimension | Sireto V7 | Karpathy | INSEE |
|---|---|---|---|
| Philosophie | Feature engineering + ML supervisé | End-to-end neural, eval-driven | Statistique probabiliste classique |
| Retrieval | TF-IDF multi-signal | Embeddings contrastifs + FAISS | Blocking exact COG + Soundex |
| Ranking | XGBoost 68 features | Cross-encoder LLM local | Fellegi-Sunter EM |
| Routing | Metamodel XGBoost | Seuil sur confiance du LLM | Triple seuillage log-LR |
| Auditabilité | Moyenne | Faible (boîte noire) | Excellente |
| Effort code | ~5000 lignes | ~500 lignes | ~200 lignes (R) |
| Performance estimée | 74.5% AUTO, 99.84% prec | ~78-82% AUTO potentiel | ~65-70% AUTO probable |
| Latence | ~100ms | ~500ms | Batch (non temps réel) |

---

## 4. Ce que Sireto pourrait emprunter

### De Karpathy

- Fine-tuner un petit embedding model en contrastif sur les paires annotées
  → remplacerait TF-IDF et BERT générique d'un coup
- Mettre en place une boucle `eval → error analysis → retrain` automatique
- Tester un cross-encoder léger (deberta-v3-small) comme Stage 2

### De l'INSEE

- Investir dans le nettoyage RNVP des adresses CRM en amont (ROI énorme)
- Utiliser le COG et FANTOIR pour la normalisation géographique
- Ajouter des intervalles de confiance sur les métriques de précision
- Gérer explicitement les fusions de communes dans le blocking
