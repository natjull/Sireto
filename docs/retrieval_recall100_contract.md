# Contrat d'exécution — Retrieval SIRET Recall@100

## Objectif

Construire un générateur de candidats qui place le **SIRET exact** de la vérité
terrain dans au moins **99,0 %** des sorties, avec **100 candidats maximum par
requête**.

Le retrieval est évalué indépendamment du ranker et du routing. Tant que ce
gate n'est pas franchi, le ranker, le decider, le risk model et l'accepteur
restent gelés.

## Corpus et séparation

- benchmark fermé : `c33b80855f560074` ;
- source :
  `/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks/closed/c33b80855f560074`;
- train : 11 837 requêtes ;
- dev : 2 565 requêtes ;
- test : 2 652 requêtes ;
- aucun SIREN partagé entre train, dev et test ;
- labels CRM historiques, non réaudités comme benchmark open-set.

Le test possède déjà une mesure sparse historique. Aucune nouvelle variante ne
doit y être exécutée avant gel de sa configuration sur train/dev. Une fois la
configuration gelée, elle est évaluée une seule fois sur le test.

## Métriques

La métrique principale est le Recall candidat au SIRET exact :

```text
Recall@K = requêtes dont le vrai SIRET appartient aux K candidats / requêtes
```

Publier systématiquement :

- Recall SIRET à K = 50, 100, 200 et 500 ;
- Recall SIREN aux mêmes K comme métrique secondaire ;
- nombres bruts et intervalles de Wilson à 95 % et 99 % ;
- latences p50, p95 et p99 ;
- nombre maximal, moyen et p95 de candidats retournés ;
- résultats actifs, fermés, mégapoles, multi-sites et localisation CP seule.

Les courbes à 200 et 500 sont exclusivement diagnostiques. Aucune configuration
de production ne peut dépasser 100 candidats.

## Attribution obligatoire des pertes

Chaque miss doit recevoir la première cause applicable :

1. `PARTITION_MISS` : vérité absente de la partition géographique chargée ;
2. `FILTER_MISS` : présente avant les filtres, absente après filtres métier ;
3. `DEDUPE_MISS` : présente après filtres, absente après déduplication ;
4. `PRUNED_AT_50` : présente après déduplication, rang final supérieur à 50 ;
5. `PRUNED_AT_100` : rang final supérieur à 100 ;
6. `PRUNED_AT_200` : rang final supérieur à 200 ;
7. `PRUNED_AT_500` : rang final supérieur à 500 ;
8. `UNEXPLAINED` : incohérence à traiter comme un défaut d'instrumentation.

La déduplication doit être auditée avant et après sur le SIRET normalisé. Elle
ne peut jamais faire disparaître l'unique ligne portant le vrai SIRET.

## Audit des canaux

Mesurer séparément, puis en unions cumulatives :

- nom TF-IDF mots ;
- nom TF-IDF caractères ;
- adresse TF-IDF ;
- égalités normalisées et clés déterministes ;
- rescue adresse exacte et tokens numériques ;
- recherche SIREN ;
- géographie INSEE, CP et élargissements éventuels ;
- dense uniquement comme canal diagnostique tant qu'il n'améliore pas le
  rappel sous plafond.

Chaque candidat conserve sa provenance, son rang par canal et ses scores. Une
union ne peut jamais évincer silencieusement un candidat déjà admis : toute
troncature à 100 doit être explicite, déterministe et auditable.

## Tuning et gates

Le tuning utilise exclusivement train/dev. Aucun positif n'est injecté dans un
pool ou une évaluation.

Gate dev :

- Recall@100 SIRET >= 99,0 % ;
- zéro requête avec plus de 100 candidats ;
- aucune régression supérieure à 2 points face à la baseline sparse@100 sur
  actifs, fermés, mégapoles et multi-sites ;
- latence p95, mémoire et disque publiés ;
- zéro perte `UNEXPLAINED`.

Si le gate dev passe :

1. figer configuration, code et hashes dans un manifeste ;
2. exécuter une seule fois le test ;
3. déclarer `GO` uniquement si Recall@100 test >= 99,0 % et plafond respecté.

Sinon :

- `PIVOT` si une cause précise possède une solution étayée mais non validée ;
- `STOP` si aucun chemin crédible ne permet d'atteindre la cible sous plafond.

## Ressources et traçabilité

- MacBook Pro M4 Pro, 24 Go ;
- SSD `/Volumes/CATNAT_DATA` ;
- aucune location de GPU, API payante ou infrastructure cloud ;
- artefacts volumineux sous
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/` ;
- chaque run conserve commande, commit, hashes, configuration, résultats bruts,
  résumé, durée et erreurs ;
- chaque milestone est un commit isolé référencé dans `handover.md` ;
- aucun artefact historique n'est déplacé ou supprimé.
