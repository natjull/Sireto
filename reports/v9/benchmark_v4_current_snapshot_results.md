# Résultats V4 — vérité SIRET active au snapshot

## Verdict

- verdict contractuel : **`STOP_V4`** ;
- test lu : **non** ;
- utilisation d'un rang, hit ou score modèle : **non** ;
- SIRET fermé parmi les labels exacts V4 : **0**.

V4 corrige bien les conflits de vérité terrain observés, mais sa règle directe
est trop stricte pour franchir le gate de couverture pré-enregistré.

## Résultats

| Split | Requêtes | `MATCH_EXACT` | Couverture | `AMBIGUOUS` | `UNRESOLVED` |
|---|---:|---:|---:|---:|---:|
| train | 11 837 | 4 060 | 34,299 % | 803 | 6 974 |
| dev | 2 565 | 872 | 33,996 % | 163 | 1 530 |

Le gate demandait :

- au moins 50 % de couverture sur chaque split ;
- au moins 5 000 exacts train ;
- aucun SIREN exact partagé entre train et dev.

Les deux seuils de volume échouent. Quatorze SIREN actuels sont en outre
partagés entre train et dev, sur 19 requêtes train et 22 requêtes dev. Cette
fuite vient du fait que l'ancien split était construit sur les SIREN
historiques, avant le changement de vérité.

## Ce que V4 a corrigé

- 620 SIRET train et 139 SIRET dev diffèrent du SIRET historique ;
- 280 SIREN train et 71 SIREN dev diffèrent du SIREN historique ;
- 382 labels V3 ouverts train et 83 labels V3 ouverts dev deviennent
  `MATCH_EXACT` actifs ;
- les cinq conflits qui bloquaient les 320 premiers scores E2b sont tous
  corrigés :

| Requête | SIRET V4 |
|---|---|
| VISSELECT, `14355` | `62820158400024` |
| IMD OPTIQUE, `16826` | `82213655200020` |
| SCI AVOCATS DU PLATEAU, `2446` | `51518433100020` |
| LMP SANTE, `11265` | `75394095600018` |
| PGDIS Dardilly, `10353` | `90032220700011` |

Chaque `MATCH_EXACT` V4 possède exactement une preuve candidate dans
`direct_evidence.parquet`, et chaque candidat correspondant porte l'état
SIRENE actif `A`.

## Compatibilité avec le système actuel

Cette mesure est postérieure à la qualification et n'a pas servi à créer les
labels.

| Split | Vérité V4 présente dans le top-100 | Hit@1 du ranker E1 |
|---|---:|---:|
| train | 4 058/4 060 = 99,951 % | 3 899/4 060 = 96,034 % |
| dev | 872/872 = 100 % | 828/872 = 94,954 % |

Le retrieval et le ranker ne sont donc pas le problème principal sur le
périmètre V4 strict. En revanche, l'ancien accepteur est incompatible avec la
nouvelle cible : il a été entraîné à considérer les `UNRESOLVED` comme des
erreurs et avec les anciens SIRET. Aucun point à 99 % n'est retrouvé lorsque
son score est relu contre V4 ; ce diagnostic ne constitue pas un nouveau
tuning.

## Lecture

V4 établit un noyau de 4 932 dossiers actuels très défendables. Ce noyau
représente environ 34 % du CRM, donc davantage que la couverture AUTO minimale
de 25 % visée pour l'accepteur. Mais il ne satisfait pas le contrat de
qualification qui exigeait 50 % et 5 000 exemples train.

Il ne faut pas assouplir maintenant les seuils sur le même dev. Deux problèmes
doivent être traités avant un nouvel apprentissage :

1. refaire la séparation des données à partir des SIREN V4 afin d'éliminer les
   14 chevauchements ;
2. décider sous un nouveau contrat si le noyau strict de 4 060 exemples train
   suffit, via une expérience d'apprentissage bornée, ou fournir de nouvelles
   lignes CRM indépendantes pour dépasser 5 000.

Les `UNRESOLVED` restent exclus du fit. Ils ne doivent pas être transformés en
négatifs ni automatiquement réétiquetés par une règle plus souple observée sur
ce dev.

## Artefact

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759`

- contrat pré-enregistré : `ce82b01` ;
- builder et tests : `799c32d` ;
- suite complète : 129 tests passants.
