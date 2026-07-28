# V4.11 — Qualification mécanique du challenge inédit

## Verdict

**`GO_FREEZE_LABELS`**

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_qualification/4f9ef46516b89ab8`

Les labels et leurs preuves sont gelés avant toute inférence du stack
V4.11.

## Populations

| Cohorte | Total | `MATCH_EXACT` | `AMBIGUOUS` | `UNRESOLVED` |
|---|---:|---:|---:|---:|
| Aveugle principale | 222 | 73 | 17 | 132 |
| Exposée | 3 | 1 | 0 | 2 |
| Total | 225 | 74 | 17 | 134 |

La couverture identifiable mécanique est de 74/225, soit 32,889 %. Les
17 ambiguïtés représentent 7,556 % et les 134 non-résolus 59,556 %.

Ce faible volume identifiable confirme que les 225 lignes sans `SERVICE ID`
forment une population atypique. La précision SIRET exacte ne pourra être
mesurée que sur les 74 `MATCH_EXACT`.

## Règle appliquée

La politique gelée `active-direct-current-v4.0` a examiné l'univers
géographique complet du snapshot :

- un unique établissement actif direct : `MATCH_EXACT` ;
- plusieurs établissements actifs directs : `AMBIGUOUS` ;
- aucun établissement actif direct : `UNRESOLVED` ;
- aucun `NO_MATCH`, secours web ou départage opportuniste.

Les 74 vérités exactes sont donc des vérités actives dans le snapshot
SIRENE gelé, pas une reconstitution historique ni une garantie d'état en
juillet 2026.

## Intégrité

- 225 labels uniques ;
- 138 preuves d'établissement actif ;
- cardinalités preuve/label cohérentes ;
- chaque vérité exacte égale son unique preuve ;
- taxonomie de labels et raisons conformes au contrat ;
- politique, six sources transitives, snapshot et partitions épinglés ;
- aucun registre source, SIRET CRM, retrieval, modèle, score ou prédiction
  consulté par le qualificateur ;
- qualification atomique et content-addressée ;
- validateur officiel et contre-audit indépendant réussis ;
- hash du manifeste :
  `17c7915725cea978278f1699832e5c17405dbab8cd21ef407f6d96916a5c89e7`.

## Suite autorisée

Implémenter et valider par parité le runner one-shot. Les 225 prédictions
doivent être scellées avant que les labels ci-dessus soient désérialisés.
Cette future mesure restera `DESCRIPTIVE_UNSEEN_225`, sans gate de
promotion.
