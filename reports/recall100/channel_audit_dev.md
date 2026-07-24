# Audit des canaux de retrieval — dev

## Contrat

- benchmark gelé : `c33b80855f560074`, split `dev`, 2 565 requêtes ;
- vérité principale : SIRET exact ;
- aucun positif injecté ;
- artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_dev_c33b80855f560074_69f6f7e` ;
- le canal `current_sparse` reproduit exactement les 2 565 listes de la
  baseline gelée, soit zéro divergence.

## Résultats individuels

| Canal | Recall@50 | Recall@100 | Recall@500 | Misses sparse@100 récupérées |
|---|---:|---:|---:|---:|
| Sparse actuel | 2 317 / 2 565 = 90,33 % | 2 379 / 2 565 = 92,75 % | 2 457 / 2 565 = 95,79 % | — |
| Nom TF-IDF mots | 85,38 % | 86,39 % | 87,64 % | 59 |
| Nom TF-IDF caractères | 84,41 % | 86,86 % | 90,72 % | 65 |
| Adresse TF-IDF mots | 72,98 % | 76,88 % | 82,14 % | 6 |
| Nom normalisé exact | 35,83 % | 35,83 % | 35,83 % | 15 |
| Adresse exacte | 47,91 % | 48,42 % | 48,77 % | 2 |
| Rescue numérique du nom | 0,78 % | 0,82 % | 0,90 % | 0 |

Le taux individuel d'un canal n'indique pas son utilité dans une union. Le nom
en caractères est légèrement inférieur au nom en mots à 50, mais il continue à
progresser jusqu'à 500 et récupère le plus de misses du sparse à 100.

## Plafonds observés

- Pool géographique V7 : 2 503 / 2 565 = 97,58 % au SIRET.
- Sparse actuel au bon SIREN à 100 : 2 452 / 2 565 = 95,59 %.
- Bon SIREN mais mauvais SIRET à 100 : 73 requêtes.
- Au moins un canal sparse individuel voit le bon SIRET dans son propre
  top-100 : 2 452 / 2 565 = 95,59 %, soit 77 récupérations par rapport au sparse
  actuel.
- En autorisant top-500 par canal, cet oracle n'atteint que
  2 478 / 2 565 = 96,61 %.

Cet oracle n'est pas une configuration éligible : l'union brute de plusieurs
top-100 dépasse 100 candidats. Il montre la complémentarité maximale disponible
avant de construire une admission à budget strict.

## Lecture architecturale

1. Le score `max(word, char, address)` suivi d'un RRF sparse/rescue écrase de
   l'information utile. Les listes mots et caractères voient 59 et 65 misses
   complémentaires, mais elles sont fusionnées trop tôt dans un score commun.
2. L'adresse est utile pour ordonner les établissements d'un SIREN ou confirmer
   un candidat, mais son retrieval seul apporte peu de nouveaux SIRET à 100.
3. Le rescue numérique ne justifie aucun quota réservé dans sa forme actuelle.
4. Une union lexicale mieux budgétée peut gagner, mais elle ne peut pas atteindre
   99 % avec ce store : le plafond du pool est déjà inférieur à la cible.
5. Les 62 SIRET absents sont tous fermés et existent dans
   `StockEtablissement_utf8.parquet`. Le builder V7 dit inclure les fermés mais
   écarte en réalité ceux dont `dateDebut < 2016-01-01`. `dateDebut` est une date
   de début de période, pas une date de fermeture : cette optimisation legacy
   est une perte de source à corriger avant tout tuning final.
6. Les ablations denses gelées restent des diagnostics : dense local seul fait
   70,29 % à 50 et sa fusion RRF dégrade le sparse. Aucun nouveau calcul dense
   n'est nécessaire pour décider l'étape suivante.

## Décision

L'audit des canaux est clos. La prochaine expérience éligible doit :

1. reconstruire un store candidat incluant réellement tous les établissements
   fermés du snapshot dans le périmètre du benchmark ;
2. réutiliser séparément les rangs mots, caractères, adresse et clés exactes ;
3. sélectionner au plus 100 candidats sans fusion précoce destructrice ;
4. régler les quotas/poids sur train, puis appliquer le gate une seule fois sur
   dev.

