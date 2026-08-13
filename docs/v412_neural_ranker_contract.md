# Contrat préenregistré — reranker neuronal V4.12-N

Date de gel : 13 août 2026  
Statut au gel : aucun score des nouveaux modèles lu sur les folds de sélection
ou de confirmation ; test final historique fermé.

## Question expérimentale

Le retrieval V4.12-L conserve déjà le SIRET exact dans 13 604 des 13 704
requêtes identifiables, avec 100 candidats au maximum. Cette expérience ne
modifie ni le CRM, ni les labels, ni le retrieval. Elle cherche à remplacer la
décision candidat du ranker XGBoost par un modèle de texte capable de choisir
le bon établissement dans le pool existant.

Le modèle reçoit le CRM brut et les données SIRENE brutes de chaque candidat.
Il ne reçoit pas le SIRET vérité, le rang du positif, une règle de promotion ou
un positif réinjecté. Le SIRET n'est conservé que comme identifiant de sortie.

## Splits gelés

Les cinq folds existants sont séparés par composantes SIREN. Aucun SIREN d'un
fold ne peut apparaître dans un autre.

- entraînement : folds 2, 3 et 4 ;
- sélection des modèles, du format de texte et des hyperparamètres : fold 0 ;
- confirmation unique du gagnant : fold 1 ;
- test final historique : non ouvert.

Ces folds ont déjà servi à des expériences V4.12-L antérieures. Les résultats
seront donc des résultats de développement comparatifs, jamais une
certification indépendante ou une garantie de production.

## Pool et métriques

- Pool gelé :
  `datasets/v4_12_learned_candidate_features/e22aa96feb6ac16f` ;
- plafond : 100 candidats par CRM, sans présélection par le ranker XGBoost ;
- une vérité absente du pool reste une erreur ;
- métrique principale : Hit@1 SIRET exact sur toutes les requêtes
  `MATCH_EXACT` du fold ;
- métriques secondaires : Hit@1 sur les cas difficiles audités, actifs et
  fermés ; latence totale et p95 par CRM ; mémoire maximale observée ;
- baseline de comparaison uniquement : sorties OOF BUSINESS_LEARNED de
  `experiments/v4_12_learned_oof_rankers/839ef55308d5077e`.

## Modèles épinglés

| Nom court | Modèle | Révision Hugging Face | Usage |
|---|---|---|---|
| QWEN_RERANKER | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | candidat principal |
| GTE_RERANKER | `Alibaba-NLP/gte-multilingual-reranker-base` | `8215cf04918ba6f7b6a62bb44238ce2953d8831c` | candidat principal |
| CAMEMBERT_FR | `antoinelouis/crossencoder-camembert-large-mmarcoFR` | `8636e2f548bfce7576808c40b454606c7a881d31` | spécialiste français |
| MMINILM_REF | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` | référence historique |
| BGE_REF | `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | référence historique |

La configuration GTE appelle le code distant `Alibaba-NLP/new-impl`. Ce code
est lui aussi épinglé, à la révision
`40ced75c3017eb27626c9d4ea981bde21a2662f4`; aucun code `main` mouvant n'est
autorisé pendant le benchmark.

Les cinq modèles sont d'abord évalués zéro-shot avec le même texte et les 100
candidats. Les deux meilleurs candidats parmi QWEN_RERANKER, GTE_RERANKER et
CAMEMBERT_FR sont retenus par Hit@1 global sur le fold 0 ; le Hit@1 difficile,
puis la latence, départagent une égalité.

## Apprentissage par scène

Une scène est ici un CRM avec son groupe de candidats concurrents. À chaque
pas d'apprentissage, le modèle voit le positif réellement retrouvé et des
négatifs difficiles issus du même groupe : premiers candidats du retrieval,
homonymes, mêmes adresses et autres établissements du même SIREN. Le score du
positif est comparé simultanément aux scores des négatifs par une entropie
croisée de groupe. Il ne s'agit donc pas d'apprendre des paires indépendantes.

Configuration initiale gelée :

- un positif et jusqu'à 15 négatifs par scène ;
- un passage sur les folds 2/3/4 ;
- longueur maximale 256 tokens par paire ;
- LoRA si le modèle l'accepte de manière fiable sur MPS, sinon couches basses
  gelées ;
- seed 42 ;
- aucune recherche au-delà de deux learning rates (`1e-5`, `3e-5`) et deux
  tailles de groupe (`8`, `16`) ;
- sélection uniquement sur le fold 0.

## Gates préenregistrés

Un modèle n'accède au fold 1 que s'il satisfait simultanément sur le fold 0 :

1. au moins 15 bonnes réponses nettes de plus que BUSINESS_LEARNED ;
2. aucune baisse du nombre de bons cas difficiles ;
3. aucune baisse supérieure à 1 point absolu sur actifs ou fermés ;
4. toutes les requêtes sont scorées avec au plus 100 candidats ;
5. aucune fuite de SIREN ou injection positive détectée.

Le gagnant est ensuite exécuté exactement une fois sur le fold 1. Le verdict
`GO` exige au moins 10 bonnes réponses nettes supplémentaires, aucune baisse
sur les cas difficiles, aucune baisse supérieure à 1 point sur actifs ou
fermés, et une exécution locale reproductible. Une amélioration sur le fold 0
mais pas sur le fold 1 vaut `PIVOT`; aucun gain matériel sur le fold 0 vaut
`STOP` pour cette famille.

## Pilote génératif conditionnel

Si aucun reranker spécialisé ne franchit le gate du fold 0, un seul pilote
QLoRA `Qwen/Qwen3-1.7B` à la révision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` est autorisé. Il reçoit une liste
numérotée de candidats et doit rendre un identifiant présent dans la liste.
Le pilote porte sur un échantillon gelé du fold 0 contenant tous les cas
difficiles et au plus 1 000 requêtes. Il sert à décider si l'approche listwise
mérite un cycle séparé ; il ne peut pas être promu en production ni ouvrir le
fold 1 dans ce cycle.

## Arrêts et ressources

- Mac M4 Pro et `/Volumes/CATNAT_DATA` uniquement ;
- aucun GPU loué, API payante ou service d'annotation ;
- arrêt d'un modèle en cas d'erreur mémoire répétée ou d'estimation supérieure
  à 18 heures pour une seule variante ;
- artefacts immuables avec manifeste, hashes, révisions et mesures de
  ressources ;
- un commit isolé pour le contrat, le corpus, le benchmark, l'apprentissage et
  le handover.
