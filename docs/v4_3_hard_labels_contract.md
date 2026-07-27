# Contrat V4.3 — labels représentatifs difficiles

Statut : préenregistré avant analyse détaillée des 542 cas non résolus.

## Objectif

Construire la matière d'apprentissage qui manquait à V4.1 : des scènes CRM
sales et difficiles permettant d'apprendre quand refuser un top-1 plausible
mais faux.

Ce milestone ne modifie ni le retrieval V4.2, ni le ranker, ni l'accepteur.
Il décide seulement si les labels disponibles sont assez fiables et assez
nombreux pour autoriser un nouvel entraînement.

## Population figée

La population contient exactement les 542 lignes `UNRESOLVED` publiées dans :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_evidence/e696f22d68c0210f/provisional_adjudications.parquet`

La décision, la prédiction et la confiance V4.1 ne sont jointes qu'après le
gel des preuves SIRENE. Elles servent à prioriser l'adjudication et à définir
la justesse du top-1, jamais à créer une vérité.

Les 242 `MATCH_EXACT` et 16 `AMBIGUOUS` existants restent physiquement
disponibles mais ne sont pas requalifiés dans ce milestone.

## Sorties possibles

Chaque cas non résolu reçoit exactement un statut :

- `MATCH_EXACT` : un SIRET actif unique est démontré ;
- `WRONG_TOP1` : les preuves démontrent que le SIRET prédit ne désigne pas
  l'entité CRM, même si le bon SIRET reste inconnu ;
- `AMBIGUOUS` : plusieurs SIRET restent plausibles ;
- `UNRESOLVED` : aucune conclusion sûre.

Le statut conserve deux dimensions séparées :

- `adjudication_status` : `PROVISIONAL` ou `HUMAN_VALIDATED` ;
- `training_eligible` : vrai uniquement si la règle de preuve est autorisée
  ci-dessous ou si un humain a validé.

Un jugement produit par un LLM, par une heuristique lexicale ou par le score
du modèle reste `PROVISIONAL` et `training_eligible=false`.

## Preuves autorisées

- snapshot SIRENE complet épinglé ;
- unité légale et établissements actifs ou fermés du même SIREN ;
- nom, enseigne, sigle, adresse, commune et état administratif ;
- top-1 V4.1 uniquement après gel des preuves ;
- sources publiques officielles, si leur URL et leur date sont conservées ;
- validation humaine explicite avec identifiant de validateur.

Une incompatibilité de nom ou de type d'entité peut prioriser une ligne, mais
ne devient pas à elle seule une vérité d'entraînement.

## File d'adjudication

Les 542 cas sont ordonnés sans suppression :

1. décisions AUTO avec contradiction forte entre CRM et top-1 ;
2. autres décisions AUTO non résolues ;
3. REVIEW proches du seuil ;
4. autres REVIEW.

Chaque ligne doit réunir dans un même artefact :

- CRM brut ;
- SIRET d'entrée et état actuel ;
- prédiction V4.1 et décision ;
- nom et adresse SIRENE du top-1 ;
- établissements actifs du SIREN d'entrée ;
- candidats locaux soutenus par les preuves ;
- indicateurs lisibles de concordance nom/adresse ;
- raison de priorité ;
- champs d'adjudication vides ou provisoires.

## Gates

`GO_RETRAIN` uniquement si le corpus contient au minimum :

- 100 `MATCH_EXACT` difficiles `training_eligible=true` ;
- 100 `WRONG_TOP1` difficiles `training_eligible=true` ;
- 50 `AMBIGUOUS` `training_eligible=true` ;
- au moins 100 cas utilisables issus du tirage aléatoire ;
- zéro label créé à partir d'un score ou d'un rang ;
- chaque label traçable à une preuve ou un validateur.

`PIVOT_VALIDATION` si la file est construite mais que les seuils de labels
fiables ne sont pas atteints.

`STOP` si même une adjudication assistée ne peut raisonnablement produire les
preuves nécessaires.

Le gate ne peut pas être abaissé après lecture des résultats.

## Livrables

- file complète de 542 cas sur le SSD ;
- sous-file prioritaire des AUTO à risque ;
- labels provisoires séparés des labels entraînables ;
- synthèse par décision, état d'entrée et famille de difficulté ;
- rapport `reports/v9/v4_3_hard_labels_results.md` ;
- verdict explicite et handover à jour.
