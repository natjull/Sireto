# Certification finale — Retrieval sélectif SIRET Recall@100

## Verdict

**Décision contractuelle : `PIVOT`.**

Le retrieval franchit tous les gates globaux du test final :

- 2 128 dossiers sur 2 652 ont un SIRET exact soutenu par une preuve directe,
  soit **80,241 % de couverture** ;
- le bon SIRET est présent dans les 100 candidats pour 2 116 de ces 2 128
  dossiers, soit **99,436 % de Recall@100** ;
- les 2 128 vérités sont toutes visibles dans l'union interne des canaux ;
- aucune requête ne dépasse 100 candidats.

Le verdict n'est pas `GO` parce que deux gates de stabilité pré-enregistrés
échouent : la couverture des établissements fermés et celle des mégapoles
baissent de plus de cinq points entre dev et test. Les recalls de ces deux
segments restent au-dessus de leurs planchers.

Ce résultat ne doit donc être interprété ni comme un échec du retrieval, ni
comme une certification de l'auto-match. Il établit qu'avec la qualification
gelée, le retrieval conserve le bon SIRET dans 99,436 % des dossiers
identifiables, lesquels représentent 80,241 % du test.

## Résultats globaux

| Périmètre | Succès | Total | Recall@100 |
|---|---:|---:|---:|
| Historique, toutes requêtes | 2 547 | 2 652 | 96,041 % |
| V2, labels structurellement exacts | 2 371 | 2 458 | 96,461 % |
| V3, SIRET exact identifiable | 2 116 | 2 128 | **99,436 %** |

Sur V3, l'intervalle de Wilson est :

- à 95 % : **[99,017 % ; 99,677 %]** ;
- à 99 % : **[98,838 % ; 99,727 %]**.

Le sparse gelé seul atteint 2 059/2 128, soit 96,758 %. L'admission
multicanal gelée apporte donc 57 succès supplémentaires dans le même plafond
de 100 candidats.

Les 524 autres dossiers restent présents dans le benchmark :

- 105 sont `AMBIGUOUS` ;
- 419 sont `UNRESOLVED`.

Ils sont destinés à `REVIEW`; leur routage produit n'est pas évalué par ce
test de retrieval.

## Gates globaux

| Gate pré-enregistré | Seuil | Observé | Résultat |
|---|---:|---:|---|
| Couverture V3 exacte | ≥ 80,0 % | 80,241 % | PASS |
| Recall@100 V3 exact | ≥ 99,0 % | 99,436 % | PASS |
| Vérités V3 invisibles dans l'oracle interne | 0 | 0 | PASS |
| Nombre maximal de candidats | ≤ 100 | 100 | PASS |
| Sorties au-dessus de 100 | 0 | 0 | PASS |

## Stabilité par segment

Les gates segmentaires s'appliquent aux segments contenant au moins 100
requêtes V3 exactes sur le test.

| Segment | N test | N exact | Couverture dev | Couverture test | Recall dev | Recall test | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Actifs | 2 181 | 1 833 | 84,986 % | 84,044 % | 99,887 % | 99,618 % | PASS |
| Fermés | 471 | 295 | 69,405 % | **62,633 %** | 97,929 % | 98,305 % | **FAIL couverture** |
| Mégapoles | 174 | 135 | 87,273 % | **77,586 %** | 98,611 % | 99,259 % | **FAIL couverture** |
| Multi-sites | 716 | 558 | 79,479 % | 77,933 % | 99,385 % | 99,462 % | PASS |
| Localisation INSEE | 2 611 | 2 104 | 82,185 % | 80,582 % | 99,566 % | 99,430 % | PASS |

Les établissements fermés ratent leur plancher de couverture de 1,772 point.
Les mégapoles le ratent de 4,687 points. Aucun des deux segments ne rate son
plancher de Recall@100.

## Lecture métier

Le premier problème historique n'était pas seulement de « retrouver plus de
SIRET ». Une partie du benchmark demandait au système de reproduire un SIRET
historique alors que le CRM pointait vers un autre établissement du même SIREN,
un établissement actif concurrent, ou ne contenait aucune preuve directe
suffisante.

La qualification V3 ne corrige pas ces labels à la main et ne choisit pas un
nouveau SIRET. Elle sépare avant retrieval :

- les dossiers où le nom ou l'adresse soutiennent directement le SIRET exact ;
- les dossiers ambigus ou non résolus, qui doivent rester en revue.

Sur le premier groupe, la cible de 99 % est franchie. Le prochain problème
n'est donc pas d'ajouter un modèle dense ou un cross-encoder au retrieval. Il
est de rendre la qualification des cas fermés et des mégapoles plus stable,
avec des preuves versionnées indépendantes du test.

## Ce qui est certifié — et ce qui ne l'est pas

Ce test certifie :

- le Recall@100 SIRET exact du retrieval gelé sur le périmètre V3 ;
- le plafond absolu de 100 ;
- la couverture V3 calculée indépendamment des résultats du retrieval ;
- l'absence de vérité V3 invisible dans l'union interne.

Il ne certifie pas :

- la précision `AUTO_MATCH` ;
- le taux final d'automatisation ;
- le traitement des 524 dossiers `REVIEW` ;
- un niveau SOTA ou une garantie générale à 99 % sur tous les CRM.

## Orientation recommandée

La suite doit rester un `PIVOT` ciblé, pas une refonte du retrieval :

1. figer définitivement ce test et ne plus l'utiliser pour choisir des règles ;
2. analyser sur train/dev la stabilité de la preuve directe pour les fermés et
   les mégapoles ;
3. construire, si nécessaire, un registre versionné d'alias, d'enseignes et
   d'historique d'établissement, toujours sourcé et daté ;
4. geler une V4 de qualification sur train/dev puis l'évaluer sur un nouveau
   holdout indépendant ;
5. seulement après stabilité de la couverture, rouvrir le ranker et
   l'accepteur pour mesurer la couverture `AUTO_MATCH` à précision contrôlée.

Il ne faut ni assouplir après coup les deux gates échoués, ni lancer une
nouvelle variante sur le test actuel.

## Reproductibilité

- Contrat : `docs/retrieval_selective_recall100_contract.md`
- Qualification test V3 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v3/72cc411a916c4814`
- Admission test gelée :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_diagnostic_test_c33b80855f560074_eb0e6a3`
- Certification :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/certification/selective_test_c33b80855f560074_6fab035`
- Commit de certification : `d1c0fc9`
- Correction d'instrumentation sans changement de métrique, seuil ou label :
  `6fab035`
- Suite complète : **105 tests passants**

Le premier lancement du certificateur s'est arrêté avant le calcul des
métriques à cause d'une colonne de segment présente dans les deux entrées.
La correction retire uniquement ce doublon avant jointure et possède un test
de régression. Aucun résultat intermédiaire n'a servi à modifier la
qualification ou le retrieval.
