# SIRETO V9 — Guide d’implémentation

V9 est une convergence modulaire, pas une réécriture monolithique. Le chemin
nominal est :

```text
CRM → retrieval multicanal/50 → ranker candidat → scène query-level
    → accepteur sélectif → AUTO_MATCH | REVIEW
```

`AUTO_NO_MATCH` n’existe pas dans cette version.

## 1. Construire le retrieval et le dataset

Quatre configurations reproductibles sont fournies dans `configs/` :

- `v9_retrieval_sparse_50.json` : baseline V7 sparse, sortie à 50 ;
- `v9_retrieval_sparse_dense_local_50.json` : sparse+dense local, sortie à 50 ;
- `v9_retrieval_sparse_global_siren_50.json` : sparse local+dense SIREN
  global, expansion SIRET puis sortie à 50 ;
- `v9_retrieval_hybrid_100_ablation.json` : ablation latence/recall à 100.

Le benchmark compare les variantes à budget identique :

```bash
python scripts/benchmark_retrieval.py \
  --crm-gt data/crm_ok_gt.csv \
  --partitions-dir data/candidates_v7_all \
  --dense-dir data/dense_index \
  --candidate-budget 50 \
  --output-csv reports/v9_retrieval_50.csv
```

La variante dense globale requiert d’abord un index SIREN :

```bash
python scripts/build_v9_siren_dense_index.py \
  --source data/StockUniteLegale_utf8.parquet \
  --output-dir data/v9_indices/siren_dense_<snapshot> \
  --model models/semantic/siret-bert-deploy
```

Puis ajouter au benchmark `--global-siren-dense-dir` et
`--siren-geo-index`. Les chemins correspondants doivent également être
présents dans la configuration d’inférence.

Une fois les candidats réellement retrouvés et leurs 54 features brutes
exportés, le bundle canonique est construit de façon immuable :

```bash
python scripts/build_v9_dataset.py \
  --queries <queries.csv> \
  --labels <labels.csv> \
  --candidates <candidates.parquet> \
  --sirene-snapshot-id <snapshot> \
  --sirene-snapshot <StockEtablissement.parquet> \
  --tokenizer-model models/semantic/siret-bert-deploy \
  --retrieval-config configs/v9_retrieval_sparse_50.json
```

Le dossier de sortie est adressé par hash. Un modèle refuse un manifeste dont
le snapshot, le tokenizer, l’ordre de features ou la signature de retrieval
est incompatible.

La baseline historique corrigée est mesurée sans changer ses modèles avec :

```bash
python scripts/evaluate_v9_baseline.py \
  --topk <export_topk_v7.csv> \
  --labels <labels_v9.csv> \
  --output reports/v9_baseline.json
```

Le rapport sépare recall candidat, Hit@1 SIRET, Hit@1 SIREN et
risque-couverture. Une vérité absente du pool est une erreur end-to-end.

## 2. Benchmark open-set

Créer une feuille de 500 requêtes stratifiées :

```bash
python scripts/build_v9_open_set_template.py \
  --source <historical_topk_or_reviews.csv> \
  --output data/v9_adjudication/open_set_500.csv
```

Les colonnes `llm_preannotation` et `llm_evidence_summary` sont uniquement
informatives. Les champs `validator`, `evidence_refs`, `validated_at`,
`sirene_snapshot_id` et `reference_date` sont obligatoires. Après validation
humaine :

```bash
python scripts/freeze_v9_open_set.py \
  --source data/v9_adjudication/open_set_500_validated.csv \
  --output-root data/v9_open_set
```

`NO_MATCH` signifie qu’aucun SIRET éligible n’existe dans le snapshot indiqué
à la date de référence. `AMBIGUOUS` signifie qu’aucun SIRET unique ne peut être
choisi avec les preuves conservées.

## 3. Ranker et accepteur

Le ranker unique émet des prédictions OOF pour toutes les scènes train et des
prédictions holdout pour dev/test :

```bash
python scripts/train_v9_ranker.py \
  --dataset data/v9/<build_id> \
  --output-dir models/v9/ranker_<build_id>
```

Les positifs absents du retrieval peuvent être fournis avec
`--training-positive-rows`, mais ces lignes sont limitées au fit du ranker.
Elles ne sont jamais réinjectées dans les prédictions OOF, dev ou test.

L’accepteur est ensuite entraîné sur ces prédictions :

```bash
python scripts/train_v9_acceptor.py \
  --dataset data/v9/<build_id> \
  --predictions models/v9/ranker_<build_id>/ranker_predictions.parquet \
  --output-dir models/v9/acceptor_<build_id> \
  --target-precision 0.998
```

La moitié déterministe de dev calibre les probabilités ; l’autre moitié fixe
le seuil. Le test final est lu seulement après sélection du modèle et du
seuil. Le rapport contient les nombres bruts, la courbe risque-couverture et
la borne de précision unilatérale à 99 %.

## 4. Cross-encoder

Le cross-encoder est une ablation optionnelle. Une révision de modèle est
obligatoire :

```bash
python scripts/run_v9_cross_encoder_ablation.py \
  --dataset data/v9/<build_id> \
  --ranker-predictions models/v9/ranker_<build_id>/ranker_predictions.parquet \
  --output-dir models/v9/cross_<build_id> \
  --revision <commit_huggingface>
```

Il est évalué sur le top 20. Le script entraîne les folds cross-encoder et
ranker injecté en OOF, puis produit les trois fichiers de prédictions appariés :
sans cross-encoder, cross-encoder seul et score cross-encoder injecté dans
XGBoost. Exécuter l’accepteur sur chacun. Le composant n’est promu que si les
gates mesurées passent : +1 point de couverture à précision cible, aucune
régression segmentaire supérieure à 2 points et latence p95 inférieure à 2×.

## 5. Promotion

`scripts/evaluate_v9_gates.py` applique les critères retrieval,
cross-encoder et déploiement à un JSON de mesures. Il sort avec un code non nul
si la précision SIRET exacte baisse ou si une famille critique régresse de
plus de 2 points.

La revendication « 99,8 % garantie » reste interdite avant un audit indépendant
d’environ 2 300 décisions AUTO sans erreur. Les rapports parlent d’estimation
observée tant que ce volume n’est pas atteint.
