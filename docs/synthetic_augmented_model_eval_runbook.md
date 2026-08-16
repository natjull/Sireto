# Runbook — effet du corpus synthétique sur XGBoost et BGE

Ce protocole répond à une seule question : à modèle et évaluation constants,
l'ajout du corpus synthétique améliore-t-il le classement SIRET sur le fold de
développement réel ? Il n'autorise ni calibration, ni risk model, ni seuil
AUTO, ni ouverture du fold 1 ou du test final.

## Protocole gelé

- apprentissage réel : folds 2/3/4 exclusivement ;
- évaluation : fold 0 réel exclusivement ;
- fold 1 et test final fermés ;
- deux scènes réelles pour une scène synthétique, sélection synthétique
  déterministe et proportionnelle à `difficulty × augmentation_stratum` ;
- poids d'une variante synthétique : `0.5/k`, où `k` est le nombre de
  variantes éligibles du même SIRET cible dans le corpus complet ;
- les poids XGBoost des scènes réelles restent les `ranker_weight` historiques
  gelés ; les scènes BGE réelles conservent leur poids 1 ;
- poids BGE appliqué à la loss groupwise par scène, jamais par paire ;
- vérité absente du top 100 : variante synthétique non entraînable, jamais
  injectée ; elle reste comptée dans le diagnostic upstream ;
- métrique primaire : Hit@1 SIRET exact ; vue même-SIREN/même-site publiée
  séparément comme métrique opérationnelle secondaire.

Le plan machine est
`config/synthetic_augmented_model_eval_v1.json`.

## Bundle synthétique préalable

La génération des 20 000 JSON CRM n'est pas directement consommable par les
modèles. Le retrieval top 100 gelé et les projections V4.12-L doivent publier
un bundle immutable `sireto-synthetic-gt-model-features-1` contenant :

- `labels.parquet` : `query_id`, vérité SIRET/SIREN/état, fold 2/3/4 assigné
  par SIREN, difficulté et strate ;
- `candidates_business.parquet` : candidats naturellement retrouvés et les
  129 features `BUSINESS_FEATURE_ORDER` ;
- `training_groups.parquet` : texte CRM/candidat, un positif naturel et au
  plus quinze négatifs par scène ;
- `manifest.json` : hashes des trois fichiers, `candidate_ceiling <= 100`,
  `positive_injection=false` et consommateurs risk/calibration/seuils AUTO à
  `false`.

Les SIREN synthétiques doivent être disjoints de tous les SIREN exacts réels,
y compris les folds 0 et 1. Le préparateur s'arrête avant fit au moindre écart.

### Construction exécutable après la 20 000e promotion

Les commandes suivantes rejouent exactement les deux audits de canaux k=5000
du retrieval V4.12-L puis l'admission gelée à 100. Elles n'entraînent aucun
modèle. Les caches et stores sont ceux des runs historiques épinglés.

```bash
SOURCE=$(python3 scripts/prepare_synthetic_gt_model_retrieval_input.py \
  --minimum-variants 20000)
SOURCE_ID=${SOURCE##*/}

V7_CHANNELS=/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/synthetic_gt_v7_channels_${SOURCE_ID}
OVERLAY_CHANNELS=/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/synthetic_gt_overlay_channels_${SOURCE_ID}

python3 scripts/audit_retrieval_channels.py \
  --benchmark "$SOURCE/benchmark.parquet" \
  --benchmark-manifest "$SOURCE/manifest.json" \
  --partitions-dir data/candidates_v7_all \
  --output-dir "$V7_CHANNELS" \
  --cache-dir /Volumes/CATNAT_DATA/SIRETO_RECALL100/cache/tfidf_sparse500_c33b80855f560074 \
  --split synthetic_train \
  --cutoffs 50 100 200 500 1000 5000 \
  --per-channel-k 5000

python3 scripts/audit_retrieval_channels.py \
  --benchmark "$SOURCE/benchmark.parquet" \
  --benchmark-manifest "$SOURCE/manifest.json" \
  --partitions-dir /Volumes/CATNAT_DATA/SIRETO_RECALL100/stores/legacy_closed_overlay_c33b80855f560074_e39fddd \
  --output-dir "$OVERLAY_CHANNELS" \
  --cache-dir /Volumes/CATNAT_DATA/SIRETO_RECALL100/cache/tfidf_closed_overlay_c33b80855f560074 \
  --split synthetic_train \
  --cutoffs 50 100 200 500 1000 5000 \
  --per-channel-k 5000

SYNTHETIC_BUNDLE=$(python3 scripts/build_synthetic_gt_model_features.py \
  --source "$SOURCE" \
  --v7-channels "$V7_CHANNELS" \
  --overlay-channels "$OVERLAY_CHANNELS")

printf '%s\n%s\n' "$SOURCE" "$SYNTHETIC_BUNDLE"
```

Le premier script découvre tous les `P*_promoted/promoted.jsonl`, vérifie
chaque manifeste de promotion et refuse de démarrer sous 20 000 variantes. Il
assigne 2/3/4 par hash stable du SIREN, vérifie la disjonction contre tous les
labels réels et produit le benchmark sans regarder le retrieval.

Le finaliseur vérifie que les deux audits couvrent exactement le corpus et
applique `select_candidates` avec les poids et quotas gelés. Il construit les
features candidat, les 129 features BUSINESS et les textes BGE à partir des
snapshots SIRENE. Le ranker réel publié `BUSINESS_LEARNED/full` intervient
uniquement pour ordonner les négatifs BGE, jamais pour qualifier ou injecter
un positif. Un sibling même SIREN avec le même identifiant d'adresse officiel
est explicitement exclu des hard negatives.

Sur la base des runs historiques (11 837 requêtes), l'extrapolation prudente à
20 000 variantes est : 35 minutes pour les canaux V7, 21 minutes pour
l'overlay, puis 10 à 25 minutes pour les features et groupes. Prévoir environ
6 Go d'artefacts temporaires/finaux ; le SSD externe dispose actuellement de
plus de 480 Go libres. Le bundle modèle final devrait rester sous 600 Mo, les
listes de canaux k=5000 représentant l'essentiel du volume.

## Commandes de la nuit

```bash
SYNTHETIC_BUNDLE=/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_model_features/<build_id>

MIX=$(python3 scripts/prepare_synthetic_augmented_model_mix.py \
  --synthetic-bundle "$SYNTHETIC_BUNDLE")

XGB_RUN=$(python3 scripts/evaluate_synthetic_augmented_xgb.py \
  --mix "$MIX")

BGE_RUN=$(python3 scripts/train_v412_bge_groupwise.py \
  --groups "$MIX" \
  --train-folds 2,3,4 \
  --target-fold 0 \
  --device mps \
  --output-root /Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/synthetic_augmented_bge_v1)

BGE_COMPARISON=$(python3 scripts/evaluate_synthetic_augmented_bge.py \
  --candidate "$BGE_RUN")

printf '%s\n%s\n%s\n' "$XGB_RUN" "$BGE_RUN" "$BGE_COMPARISON"
```

XGBoost reconstruit un contrôle `REAL_ONLY` et un candidat
`REAL_PLUS_SYNTHETIC` avec exactement les mêmes folds, features et
hyperparamètres. BGE réutilise comme contrôle le run réel publié
`01e1049c16af2600`; seul le candidat augmenté est réentraîné.

## Gate développement

Un modèle augmenté reçoit `GO_SYNTHETIC_AUGMENTATION_*` uniquement s'il :

- gagne au moins 10 bonnes réponses exactes face à son contrôle apparié ;
- atteint au moins 2 452/2 797 exacts, 33/38 difficiles, 2 164/2 391 actifs
  et 246/406 fermés ;
- conserve le plafond de 100, l'absence d'injection et la fermeture du fold 1
  et du test.

Un échec ne déclenche ni recherche d'hyperparamètres ni second essai opportuniste.
La vue opérationnelle même-site n'altère jamais ce gate exact.

## Temps et ressources attendus sur le Mac M4 Pro

- préparation/mix : quelques minutes après matérialisation des features ;
- XGBoost, deux fits : environ 1 à 3 minutes ;
- BGE augmenté : environ 8 heures, dont environ 4,4 heures d'apprentissage et
  3,6 heures de scoring fold 0, extrapolées du run réel publié ;
- pic RAM attendu : inférieur à 6 Go d'après les quatre runs BGE précédents ;
- aucun GPU loué, aucun appel externe et aucune dépense.

Le run BGE peut donc tenir dans une nuit sur le Mac. Les trois runs BGE
cross-fittés nécessaires à un futur stack XGBoost+BGE ne sont pas lancés dans
ce premier verdict : ils ne deviennent utiles que si le BGE direct ou le XGB
direct justifie de poursuivre.
