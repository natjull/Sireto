# V4.11 — Retrieval input-blind aligné

## Verdict

**`GO_TRAIN_INPUT_BLIND_RANKER`**

Le retrieval sparse V4.11 conserve le bon SIRET dans 100 % des requêtes
exactes du dev historique et dans 99,9786 % du fit, avec un plafond strict de
100 candidats. Le gate préenregistré Recall@100 est franchi. Le ranker C peut
donc être entraîné ; le ranker, l'accepteur et leurs seuils n'ont pas été
utilisés pour obtenir ce verdict.

Artefact immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d`

## Population et pools

| Mesure | Valeur |
|---|---:|
| Requêtes | 7 003 |
| Labels | 7 003 |
| Candidats | 698 892 |
| Taille minimale d'un pool | 23 |
| Taille médiane | 100 |
| Taille maximale | 100 |
| Pool vide | 0 |
| Candidat fermé | 0 |
| Doublon `(query_id, SIRET)` | 0 |
| Positifs exacts présents | 5 882 / 5 883 |

## Recall exact SIRET

| Split | Recall@1 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|
| Fit | 4 121 / 4 666 = 88,3198 % | 4 611 / 4 666 = 98,8213 % | 4 663 / 4 666 = 99,9357 % | **4 665 / 4 666 = 99,9786 %** |
| Dev historique | 1 075 / 1 217 = 88,3320 % | 1 211 / 1 217 = 99,5070 % | 1 217 / 1 217 = 100 % | **1 217 / 1 217 = 100 %** |

L'unique miss Recall@100 est la requête fit `6818`. Il n'est ni réinjecté ni
transformé en positif. Il reste une erreur end-to-end et sa scène sera
conservée comme miss par les étapes aval.

Les métriques @1 sont celles du retrieval avant ranking ML. Elles ne mesurent
pas la performance attendue du ranker C.

## Comparaison contextuelle V4.2-B

Le V4.2-B complet atteint 100 % à Recall@100 sur fit et dev, mais il contient
la branche d'identifiant désormais interdite. Son sous-ensemble sparse
historique obtient 4 665/4 666 sur fit et 1 217/1 217 sur dev, soit le même
Recall@100 que V4.11. Cette comparaison est descriptive : V4.11 a reconstruit
ses pools sans voir le SIRET/SIREN CRM et ne reprend pas les rangs du V4.2-B.

## Intégrité et reproductibilité

- `input_siret` et `input_siren` sont exclus dès la projection physique du
  parquet de requêtes et absents des sorties ;
- aucun argument de vérité n'est transmis au retrieval ;
- les labels et les candidats V4.2-B ne sont hashés et ouverts qu'après
  fermeture du pool V4.11 final sans labels ;
- le snapshot SIRENE est hydraté par une seule jointure bulk ;
- le cache TF-IDF RAM est plafonné à 20 partitions ;
- le cache disque est lié à la configuration et à la signature des
  partitions, puis vérifié par SHA-256 avant désérialisation ;
- 1 409 misses cache ont construit les partitions et 2 632 hits les ont
  réutilisées ; aucun cache non vérifié n'a été accepté ;
- les 508 081 SIRET uniques demandés existent tous dans le snapshot ;
- les hashes de tous les fichiers de sortie ont été recomputés
  indépendamment ;
- le SHA-256 du builder dans le manifeste correspond exactement au blob Git
  du commit `fc8c848` ;
- `is_ground_truth` a été recomputé indépendamment sans aucun écart ;
- le validateur officiel de l'artefact passe.

Hashes de contenu canoniques :

- pools ordonnés :
  `873c1109c5384cfedcbae0e7d4fbb186351e09962d4bd111248b16db65a5b3a8` ;
- candidats complets :
  `3680f399a1f212ac287c34b9bc293911910e81efddfb136b66ed09035cbcf3bc`.

## Limites

- Le dev historique est déjà consommé : ce `GO` autorise le développement
  du ranker C, pas une certification produit finale.
- Les 225 lignes inédites restent fermées.
- Le test final historique et le holdout V4-Fresh ne sont pas réutilisés.
- Le Recall@100 ne préjuge ni du Hit@1 du ranker, ni de la précision ou de la
  couverture `AUTO_MATCH` de l'accepteur.
- La preuve finale nécessitera toujours un nouvel export CRM indépendant.

