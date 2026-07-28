# V4.12-G — Résultat du gate historique

## Verdict

`GO_V412_HISTORICAL_GATE`

Ce verdict couvre uniquement les gates entiers et segmentaires sur
`comparison_dev`. Il ne certifie ni la latence de service, ni la production,
ni la précision sur une population indépendante.

## Artefact

- chemin :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/v4_12_guard_historical/fedcd1d512bfd269`
- build id : `fedcd1d512bfd269`
- manifeste :
  `0bd3024d519b2f363aed40b19c9b7902024376f4a6b3db7c26d1000ecfc02ba4`
- verrou :
  `be1a2422ec576a0e843fa0d43622324673465e0fd47e9860fe388b1eca87f639`

## Gate `comparison_dev`

| Mesure | V4.11 | V4.12-G |
|---|---:|---:|
| Population | 746 | 746 |
| AUTO | 614 | 614 |
| Couverture AUTO | 82,3056 % | 82,3056 % |
| AUTO exacts | 614 | 614 |
| Erreurs AUTO | 0 | 0 |
| `AMBIGUOUS` AUTO | 0 | 0 |
| Précision observée | 100 % | 100 % |

Le gate effectif `A >= 600`, `E == 0`, `B == 0` passe. Les onze segments
prédéfinis sont publiés, dont sept bloquants de taille au moins 100. Aucun ne
perd un AUTO et tous passent la non-infériorité de couverture.

## Contrôle descriptif hors gate

La garde a néanmoins un effet sur les autres populations historiques :

| Population | AUTO V4.11 | AUTO V4.12-G | Retraits |
|---|---:|---:|---:|
| fit OOF | 4 557 | 4 554 | 3 |
| threshold dev | 564 | 563 | 1 |
| comparison dev | 614 | 614 | 0 |

Les quatre retraits sont des `AMBIGUOUS` que V4.11 aurait automatisés. Ils
possèdent chacun deux candidats directs forts appartenant à deux SIREN et
sont refusés avec `MULTIPLE_STRONG_DIRECT_CANDIDATES`. La garde supprime donc
les quatre erreurs historiques observables hors population de gate, sans
perte d'un AUTO exact dans ces populations.

Ces résultats hors `comparison_dev` sont descriptifs. Ils ne servent à
modifier ni la règle, ni le seuil.

## Intégrité

- baseline V4.11 reproduite exactement à `614 / 0 / 0` ;
- 7 003 décisions uniques et populations canoniques `5 547 / 710 / 746` ;
- modèle, ranker, accepteur et seuil inchangés ;
- garde exclusivement en veto ;
- pool ranker inchangé et plafonné à 100 ;
- aucune colonne `is_ground_truth` du ranker ni label retrieval ouverte ;
- aucun challenge consommé ouvert ;
- décisions, métriques et segments recalculés par le validateur ;
- pic RSS : 673 038 336 octets, inférieur à 8 Gio ;
- validation officielle et contre-audit indépendant concluants.

Deux préflights techniques ont échoué avant scoring et sans artefact : le
premier traitait à tort le `dev_partition` vide du fit comme une divergence ;
le second comparait le fold de composante des scènes dev au fold OOF nul du
ranker dev. Les corrections `7d70249` et `e8b052f` ont été testées,
contre-auditées et reverrouillées avant l'exécution publiée.

## Limite d'interprétation

Les labels historiques V4.11 dérivent de la même politique de preuve directe.
Le résultat confirme donc la cohérence et l'intérêt de la garde sur
l'historique ; il ne fournit pas une preuve indépendante de précision
juridique.

Le manifeste fixe explicitement :

```text
latency_gate_evaluated = false
production_certified = false
```

## Suite

1. figer la règle V4.12-G et son bundle de dépendances ;
2. valider la parité du même calcul entre batch et chemin d'inférence ;
3. mesurer séparément la latence appariée V4.11/V4.12-G ;
4. seulement ensuite préparer un nouvel export CRM indépendant ;
5. sceller les prédictions V4.11 et V4.12-G avant ouverture de labels
   indépendants.
