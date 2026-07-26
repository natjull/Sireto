# Contrat V4 — vérité SIRET active au snapshot

## Objet

V2 et V3 ne pouvaient que conserver ou écarter le SIRET historique. Elles ne
pouvaient pas désigner une entité active d'un autre SIREN, même lorsque son
nom et son adresse correspondaient directement au CRM.

V4 répond à une autre question :

> Quel SIRET actif du snapshot SIRENE correspond de manière unique au nom et à
> l'adresse actuellement présents dans le CRM ?

Cette politique est adaptée à un enrichissement actuel du CRM. Elle ne prétend
pas reconstruire un exploitant historique : les requêtes ne contiennent pas de
date de référence exploitable.

## Périmètre

- splits autorisés : `train` et `dev` uniquement ;
- split `test` refusé par le builder ;
- snapshot établissement et unité légale déjà liés au benchmark V9 ;
- univers candidat : toutes les lignes de la partition géographique SIRENE,
  avant tout classement ;
- calcul local sur le Mac et `/Volumes/CATNAT_DATA` ;
- aucune lecture d'un hit, rang, score de retrieval, score XGBoost ou décision
  historique.

La partition principale est l'INSEE. Si l'INSEE est absent, le code postal est
utilisé. Dans les mégapoles dont la partition INSEE dépasse 100 000 lignes,
l'univers est limité à l'intersection code postal–INSEE, jamais à un top-k.

## SIRET actif

Un SIRET est actif si `etat_admin == "A"` dans le snapshot. Un établissement
fermé ne peut jamais devenir `MATCH_EXACT` V4.

Le SIRET historique, son SIREN, son statut V3 et sa raison de qualification
restent conservés comme provenance.

## Correspondance directe

Le CRM et un candidat actif doivent fournir une preuve forte de nom et une
preuve forte d'adresse, selon les fonctions déterministes V3.

Pour éviter qu'une comparaison approximative sur toute une commune ne crée
une nouvelle vérité fragile, au moins une ancre exacte est obligatoire :

- adresse canonique exacte, puis nom fort ; ou
- nom normalisé exactement égal à l'un des noms SIRENE, puis adresse forte.

La clé d'adresse canonique neutralise la casse, les accents, les types de voie
et la ponctuation parasite, notamment les apostrophes remplacées par `?` dans
certains exports CRM. Elle conserve le numéro de voie.

Les noms SIRENE autorisés sont ceux déjà utilisés par les features partagées :
enseignes, dénomination établissement, sigle, noms usuels et dénomination
d'unité légale. La sémantique neuronale est exclue.

## Attribution du label

Après déduplication exacte des SIRET :

| Candidats actifs avec correspondance directe | Label V4 |
|---:|---|
| exactement 1 | `MATCH_EXACT` sur ce SIRET |
| 2 ou plus | `AMBIGUOUS` |
| aucun | `UNRESOLVED` |

V4 ne produit pas automatiquement `NO_MATCH` : l'absence de preuve directe
n'établit pas l'absence d'entreprise.

Une ligne V2/V3 `AMBIGUOUS` ou `UNRESOLVED` peut devenir `MATCH_EXACT` si un
unique SIRET actif satisfait la règle ci-dessus. Il s'agit d'une nouvelle
qualification mécanique au snapshot, pas d'une validation humaine.

## Garde-fous

- aucune requête supprimée ;
- `query_id` unique ;
- zéro SIRET fermé parmi les `MATCH_EXACT` ;
- exactement un candidat direct actif pour chaque `MATCH_EXACT` ;
- aucun SIRET dans les labels ouverts ;
- aucune colonne de rang, hit, score ou décision modèle acceptée par le
  builder ;
- ordre déterministe par `query_id` et `candidate_siret` ;
- manifeste avec hashes V3, snapshot, politique et univers de partitions ;
- train et dev produits séparément ;
- zéro SIREN V4 exact partagé entre train et dev ; tout conflit éventuel doit
  être retiré du fit avant un nouvel apprentissage ;
- aucune métrique de retrieval calculée pendant la qualification.

## Sorties obligatoires

- benchmark V4 complet ;
- labels V4 ;
- preuves candidates actives examinées ;
- résumé des transitions V3 → V4 ;
- volumes exacts, ambigus et non résolus ;
- changements de SIRET, passages fermé → actif et conflits de SIREN ;
- manifeste immuable.

## Gate de viabilité

La qualification permet de poursuivre le chantier aval si :

- la couverture `MATCH_EXACT` est au moins 50 % sur train et sur dev ;
- train contient au moins 5 000 `MATCH_EXACT` ;
- aucun SIREN V4 exact n'est partagé entre train et dev ;
- tous les garde-fous ci-dessus passent.

Ce gate ne certifie pas la justesse humaine des labels. Le dev a déjà été
inspecté et ne peut plus valider une revendication finale. Un nouveau holdout
indépendant, qualifié avec la même politique gelée, restera obligatoire.

## Conséquence pour l'accepteur

Lors de la reconstruction aval :

- `MATCH_EXACT` fournit les positifs et les mauvais choix exact-SIRET ;
- `AMBIGUOUS` peut fournir des exemples à envoyer en `REVIEW` ;
- `UNRESOLVED` est exclu du fit : inconnu ne signifie pas faux ;
- le retrieval et le ranker E1 restent gelés jusqu'à la mesure de compatibilité
  avec les nouveaux SIRET V4.
