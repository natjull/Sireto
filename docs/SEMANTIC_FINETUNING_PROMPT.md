# 🧠 SIRET-BERT: Fine-tuning Sémantique pour le Matching SIRENE

## Mission

Créer un modèle d'embeddings sémantiques spécialisé pour le matching des dénominations sociales françaises,
en partant du modèle `paraphrase-multilingual-mpnet-base-v2` et en l'adaptant au domaine SIRENE/CRM.

---

## Contexte du Projet SIRETO

### Architecture Actuelle

```
CRM Query → Blocking (CP/INSEE) → XGBoost Ranker → XGBoost Classifier → Reranking Rules → Top-K
              ↓                        ↓                   ↓
         ~1000 candidats          37 features         Probabilité
```

### Données Disponibles

| Fichier | Description | Volume |
|---------|-------------|--------|
| `data/entrainements.csv` | Paires CRM-SIRET validées | **23 607 lignes** |
| `data/StockEtablissement_utf8.parquet` | SIRENE complet | ~40M établissements |
| `data/StockUniteLegale_utf8.parquet` | Unités Légales | ~12M UL |
| `data/harvest_full.sqlite` | Dirigeants PM | ~15M entrées |

### Colonnes CRM (entrainements.csv)

```
SITE             → crm_name (nom commercial CRM)
SITE_CLI_ADRESSE → crm_address
SITE_CLI_COMMUNE → crm_city
CODE_POSTAL      → postcode
CODE_INSEE       → insee
SIRET            → ground_truth_siret (14 digits)
```

### Module Sémantique Existant

Fichier: `src/xgb_matcher/semantic.py`

```python
# Activation via env var
XGB_SEMANTIC_ENABLED=1

# Modèle par défaut
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Fonctions clés
embed_text(text) → np.ndarray  # Avec preprocessing
max_semantic_similarity(source, candidates) → float
_normalize_for_embedding(text) → str  # CamelCase split, etc.
```

### Preprocessing Existant

`src/xgb_matcher/naming.py` et `features.py` contiennent :
- `normalize_text()` : Unicodedata, lowercase, strip
- `strip_location_from_crm_name()` : Supprime ville/CP du nom
- `primary_name()` : Extrait le nom principal d'un candidat
- Mapping abréviations : STE→SOCIETE, ETS→ETABLISSEMENTS, etc.

---

## Objectif du Fine-Tuning

### Problèmes à Résoudre

1. **Équivalences métier** : "Fournil" ≈ "Boulangerie", "Garage" ≈ "Automobiles"
2. **Abréviations françaises** : "Ste" = "Société", "Ets" = "Etablissements"
3. **Noms de personnes** : "Jean Dupont" ≈ "Dupont Jean" ≈ "J. DUPONT"
4. **Formes juridiques** : "SAS", "SARL", "EURL" doivent être proches
5. **Variations géographiques** : "Rhône-Alpes" ≈ "Lyon" ≈ "69"

### Dataset d'Entraînement

Créer à partir de `entrainements.csv` :

```python
# Structure des paires positives
{
    "anchor": crm_name (normalisé),
    "positive": official_sirene_name (dénomination SIRENE),
    "hard_negative": nom_similaire_mais_faux_siret  # Optionnel
}
```

### Sources de Hard Negatives

1. **Même commune, nom proche** : top-K Jaro des autres SIRET de la commune
2. **Même SIREN, autre établissement** : Autre établissement de la même UL
3. **Homonymes géographiques** : "Boulangerie Durand Paris" vs "Boulangerie Durand Lyon"

---

## Spécifications Techniques

### Environnement

- **Hardware** : MacBook M4 Pro, 24GB RAM unifié
- **Backend PyTorch** : `mps` (Metal Performance Shaders)
- **Python** : 3.14 (déjà installé)

### Modèle de Base

```python
# Option 1 : MPNet (recommandé pour la qualité)
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Option 2 : CamemBERT (spécialisé français)
model = SentenceTransformer("dangvantuan/sentence-camembert-large")
```

### Hyperparamètres

| Paramètre | Valeur |
|-----------|--------|
| Batch size | 64-128 |
| Époques | 2-3 |
| Learning rate | 2e-5 |
| Warmup steps | 100 |
| Loss | `MultipleNegativesRankingLoss` |

### Structure du Code

```
semantic_finetuning/
├── prepare_dataset.py      # Génère paires + hard negatives
├── train.py                # Fine-tuning avec sentence-transformers
├── evaluate.py             # Métriques: Recall@K, MRR, Hit Rate
├── export.py               # Sauvegarde modèle pour semantic.py
└── config.yaml             # Hyperparamètres
```

---

## Livrables Attendus

### 1. Script de Préparation des Données

```python
# prepare_dataset.py
# Input: data/entrainements.csv + data/StockEtablissement_utf8.parquet
# Output: data/semantic_train.jsonl

# Format JSONL:
{"anchor": "boulangerie durand", "positive": "DURAND BOULANGERIE SAS", "negative": "DURAND PATISSERIE SARL"}
```

### 2. Script d'Entraînement

```python
# train.py
# Utilise sentence-transformers avec MNRL loss
# Sauvegarde checkpoints et logs TensorBoard
```

### 3. Script d'Évaluation

```python
# evaluate.py
# Métriques sur le set de test:
# - Recall@1, Recall@5, Recall@10
# - MRR (Mean Reciprocal Rank)
# - Amélioration vs modèle de base
```

### 4. Intégration avec semantic.py

```python
# Modifier src/xgb_matcher/semantic.py pour charger le modèle fine-tuné
XGB_SEMANTIC_MODEL=models/siret-bert-fr-v1
```

---

## Contraintes et Bonnes Pratiques

### Preprocessing Cohérent

**CRITIQUE** : Utiliser **exactement** la même normalisation que dans `semantic.py` :

```python
from src.xgb_matcher.semantic import _normalize_for_embedding
# Split CamelCase, sépare chiffres, normalise espaces
```

### Gestion de la Mémoire

- Utiliser `DataLoader` avec `pin_memory=False` sur MPS
- Libérer le cache avec `torch.mps.empty_cache()`

### Versioning

- Sauvegarder le modèle avec métadonnées (date, métriques, config)
- Structure: `models/semantic/siret-bert-v{VERSION}/`

---

## Métriques de Succès

| Métrique | Baseline (sans FT) | Objectif |
|----------|-------------------|----------|
| Recall@1 | ~40% | >70% |
| Recall@5 | ~65% | >90% |
| MRR | ~0.50 | >0.80 |

### Test sur Cas Difficiles

Vérifier sur les cas connus :
- "Timcod Rhône-Alpes" → SIRET correct
- "DigitBoxing and coall" → SIRET correct
- "Groupe ADF" → SIRET correct

---

## Workflow Recommandé

```bash
# 1. Créer branche
git checkout -b feature/semantic-finetuning

# 2. Préparer dataset
python semantic_finetuning/prepare_dataset.py

# 3. Fine-tuner
python semantic_finetuning/train.py

# 4. Évaluer
python semantic_finetuning/evaluate.py

# 5. Intégrer
cp models/semantic/siret-bert-v1 models/
# Update semantic.py default model

# 6. Re-évaluer le pipeline complet
python scripts/evaluate_xgb_comprehensive.py --dataset test --policy both
```

---

## Fichiers de Référence

```
src/xgb_matcher/semantic.py      # Module sémantique existant
src/xgb_matcher/naming.py        # Normalisation des noms
src/xgb_matcher/features.py      # Features XGBoost (FEATURE_NAMES)
scripts/generate_training_samples_v3.py  # Génération samples alignés
data/entrainements.csv           # Dataset CRM-SIRET
```

---

## Questions Ouvertes

1. **Choix du modèle de base** : MPNet multilingue vs CamemBERT français ?
2. **Ratio hard negatives** : 1 par ancre ou plus ?
3. **Augmentation** : Ajouter des variations synthétiques (typos, abréviations) ?
4. **Ville dans l'embedding** : Concaténer `nom + ville` pour l'ancre ?

---

**PRÊT À LANCER LE FINE-TUNING !** 🚀
