# Politique V2 de qualification des labels SIRET

## Objet

Le benchmark historique `c33b80855f560074` reste immuable. La V2 ajoute une
couche de qualification qui sépare :

- les labels SIRET exacts encore évaluables ;
- les sites non identifiables de façon unique avec les champs CRM ;
- les références historiques devenues incohérentes avec le snapshot SIRENE
  courant.

Cette qualification est indépendante du résultat du retrieval. Une requête
n'est jamais retirée parce que son SIRET est absent du top 100 ou mal classé.

## Statut

La V2 est une qualification mécanique et rétrospective du train/dev déjà
observé. Elle ne constitue ni une nouvelle vérité terrain certifiée, ni un
nouveau test vierge. Elle sert à mesurer honnêtement le retrieval et à préparer
une adjudication humaine ciblée.

Le test historique n'est ni lu par une nouvelle variante, ni requalifié avant
le gel d'une politique finale.

## Règle de qualification

La comparaison porte uniquement sur les autres établissements du SIREN
historique et utilise la clé d'adresse canonique déjà auditée.

| Classe d'audit | Qualification V2 | Motif |
|---|---|---|
| `ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING` | `AMBIGUOUS` | le label et au moins un autre SIRET actif du même SIREN partagent l'adresse CRM |
| `MULTIPLE_ACTIVE_EXACT_SIBLINGS` | `AMBIGUOUS` | plusieurs SIRET actifs alternatifs du même SIREN partagent l'adresse CRM |
| `CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING` | `UNRESOLVED` | le label est fermé et un autre SIRET actif du même SIREN correspond exactement à l'adresse CRM |
| toute autre classe | `MATCH_EXACT` | aucune contradiction structurelle suffisante n'est établie par cet audit |

Pour les lignes `AMBIGUOUS` et `UNRESOLVED`, le SIRET historique est conservé
comme provenance, mais n'est plus utilisé comme vérité exacte dans la couche
V2. Aucun SIRET alternatif n'est promu automatiquement.

## Ce qui n'est pas exclu

- un cas difficile ou faiblement ressemblant ;
- un établissement fermé sans alternative active exacte établie ;
- un sibling inactif à la même adresse ;
- une adresse CRM manquante ;
- un échec du retrieval, du ranker ou d'un autre modèle.

Ces cas restent dans `MATCH_EXACT` tant que l'audit mécanique ne démontre pas
une contradiction. Cette règle volontairement conservatrice évite d'améliorer
artificiellement la métrique en retirant les erreurs difficiles.

## Métriques

Deux résultats doivent toujours être publiés ensemble :

1. Recall@100 historique sur toutes les lignes, sans changement ;
2. Recall@100 V2 sur les seules lignes qualifiées `MATCH_EXACT`.

Les volumes `AMBIGUOUS` et `UNRESOLVED` sont publiés séparément. Ils ne
disparaissent pas du produit : leur comportement cible est `REVIEW`, mais un
artefact de retrieval seul ne permet pas encore de mesurer ce routing.

Un résultat supérieur ou égal à 99 % sur la V2 ne peut être revendiqué que
sur le périmètre exact restant et avec son dénominateur explicite. Il ne
remplace pas la métrique historique et ne vaut pas certification open-set.

## Traçabilité

Chaque build V2 doit :

- vérifier les hashes du benchmark, du snapshot et de l'audit source ;
- conserver la classe d'audit, le label historique et les SIRET alternatifs ;
- produire un manifeste avec version de politique, commit et hashes ;
- écrire dans un nouveau répertoire immuable ;
- ne jamais modifier le benchmark fermé d'origine.
