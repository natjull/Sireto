# V4.12-L — conclusion du goal appris

Date : 13 août 2026  
Verdict final : **PIVOT**. Le test final n'a pas été ouvert.

## Résultat par étage

| Étage | Périmètre | Résultat | Gate | Verdict |
|---|---|---:|---:|---|
| Population | CRM unifié | 17 097 requêtes, 13 704 exactes | traçable, SIREN-disjoint | GO |
| Retrieval V4.12-L | exactes identifiables | 13 604/13 704 = 99,270 % Recall@100 | ≥99,0 % | GO |
| Qualification | population totale | 13 704/17 097 = 80,154 % | ≥80,0 % | GO |
| Ranker métier appris | développement difficile | 220/241 | ≥225/241 | **PIVOT** |
| Ranker métier appris | exactes unifiées | 11 939/13 704 = 87,121 % | — | insuffisant |
| Accepteur XGBoost nested OOF | population totale | 743 AUTO, 99,596 %, couverture 4,35 % | 99,8 % et 88–92 % | **PIVOT** |
| Accepteur, audités ouverts | 31 ambigus + 7 non résolus | 0 AUTO/38 | 0 | GO local |

Les métriques retrieval gelées sont publiées ensemble :

| Vue | Couverture identifiable | Recall@100 exact |
|---|---:|---:|
| Historique | — | 16 390/17 054 = 96,106 % |
| V2 | 15 853/17 054 = 92,958 % | 15 286/15 853 = 96,423 % |
| V3 | 13 658/17 054 = 80,087 % | 13 558/13 658 = 99,268 % |
| V4.12-L | 13 704/17 097 = 80,154 % | 13 604/13 704 = 99,270 % |

## Ce que sont les scènes

Une scène est une ligne par CRM, construite après le classement candidat. Ses
259 variables décrivent le top1 et ses concurrents : écart top1/top2, meilleur
autre SIREN, densité des scores, nombre de SIREN proches, différences de nom,
adresse, activité, rôle, siège et signaux d'admission. Elle ne contient ni la
vérité terrain ni une règle de promotion.

L'accepteur apprend sur ces scènes s'il faut rendre le top1 en `AUTO_MATCH` ou
abstenir en `REVIEW`. Chaque fold évalué est exclu à la fois de l'entraînement
et de la calibration du seuil. Il ne change jamais le SIRET du ranker.

## Pourquoi l'objectif AUTO est impossible avec ce ranker

Sur la population complète, le ranker ne produit que 11 939 bonnes réponses
sur 17 097 requêtes. Même un accepteur oracle ne pourrait donc dépasser
69,831 % de couverture AUTO exacte. Sur les seules 13 704 requêtes exactes, sa
borne est 87,121 %, encore inférieure au plancher visé de 88 %.

La calibration séparée le confirme : chaque fold de calibration avait zéro
erreur parmi ses AUTO, mais le fold externe produit au total trois erreurs —
un `MATCH_EXACT` mal classé et deux labels ouverts V3 non audités. Baisser ou
rechoisir le seuil sur ces folds externes constituerait une fuite.

## Ablations fermées

- 129 features candidat, incluant les anciennes logiques métier sous forme de
  variables apprises : gain réel, de 211 à 220/241, mais gate manqué ;
- pondération humaine x2/x4 et objectif `rank:ndcg` : aucun gain ;
- spécialiste appris sur les seuls cas humains : 221/241 au mieux avec baisse
  globale ;
- petit cross-encoder et BGE zéro-shot, seuls ou dans un reranker XGBoost :
  221/241 au mieux en mélange, 219/241 dans le reranker ;
- vraie décomposition SIREN puis SIRET : 219/241 et baisse globale ;
- fine-tuning pairwise local du petit cross-encoder, fold pilote strictement
  exclu : +1/2 797 au global du fold, +0/38 sur le difficile ; les quatre
  autres folds n'ont pas été lancés.

Ces résultats excluent raisonnablement une nouvelle boucle de réglages sur la
même représentation. Ils ne démontrent pas que le matching est impossible ;
ils démontrent que cette famille n'atteint pas la North Star avec les labels et
preuves actuels.

## Orientation après PIVOT

1. Conserver en production la baseline historique gelée ; V4.12-L ne la
   remplace pas.
2. Conserver le retrieval V4.12-L : son gate est franchi et il résout bien le
   problème des SIRET éliminés avant le ML.
3. Ne pas ajouter de nouvelles règles déterministes ni réajuster l'accepteur.
4. Le prochain investissement utile est un corpus ciblé de preuves de rôle et
   d'identité d'établissement, à une échelle supérieure aux 241 cas : marques
   exploitées par un autre SIREN, réseaux/franchises, établissements publics,
   co-localisations et changements temporels. Sans nouvelle information
   supervisée, un nouveau modèle réapprendra les mêmes ambiguïtés.
5. Une reprise devra être un nouveau cycle expérimental préenregistré avec un
   développement vierge. Les vues actuelles sont entièrement consommées et ne
   peuvent plus certifier un progrès.

## Artefacts principaux

- population : `datasets/v4_12_learned_unified_population/2d29be3ccd8fcc3e` ;
- retrieval : `evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f` ;
- features métier : `datasets/v4_12_learned_business_features/8800ef53f6927215` ;
- ranker : `experiments/v4_12_learned_oof_rankers/839ef55308d5077e` ;
- scènes : `datasets/v4_12_learned_scenes/2f2bb2b0208241e0` ;
- accepteur : `experiments/v4_12_learned_acceptor/13997088931181ba` ;
- pilote cross-encoder :
  `experiments/v4_12_learned_cross_encoder_pilot/d517650eb9951cf9`.

Tous les chemins sont sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100`. Les
résultats sont des mesures de développement consommé, ni SOTA ni garantie de
production.
