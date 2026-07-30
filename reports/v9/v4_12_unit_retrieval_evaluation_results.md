# V4.12 — Évaluation oracle du retrieval unitaire

## Verdict

**`GO_V412_UNIT_RETRIEVAL_EVALUATION`**

Les deux gates développement sont franchis :

- couverture SIRET exact identifiable supérieure ou égale à 80 % ;
- Recall@100 SIRET exact supérieur ou égal à 99 %.

Deux audits post-run indépendants concluent
`GO_EVALUATOR_ARTIFACTS_1` et `GO_EVALUATOR_ARTIFACTS_2`. Ils ont
reconstruit les 1 456 outcomes depuis l'oracle, les statuts et les candidats,
sans écart avec la publication.

Cette mesure porte sur le dev historique. Elle n'est ni indépendante, ni une
certification de production, ni le test final.

## Résultats V4.12

| Population | Total | Exact | Ambigu | Couverture | Recall@1 | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Global | 1 456 | 1 217 | 239 | 83,585 % | 88,332 % | 99,507 % | 100 % | 100 % |
| Threshold dev | 710 | 583 | 127 | 82,113 % | 90,566 % | 99,314 % | 100 % | 100 % |
| Comparison dev | 746 | 634 | 112 | 84,987 % | 86,278 % | 99,685 % | 100 % | 100 % |

Nombres bruts globaux :

- couverture : 1 217 / 1 456 ;
- Recall@1 : 1 075 / 1 217 ;
- Recall@10 : 1 211 / 1 217 ;
- Recall@50 : 1 217 / 1 217 ;
- Recall@100 : 1 217 / 1 217 ;
- vérités absentes du pool : 0.

Intervalle de Wilson bilatéral à 99 % :

- couverture : [80,933 % ; 85,932 %] ;
- Recall@100 : [99,458 % ; 100 %].

Le gate utilise les taux observés préenregistrés, pas les bornes Wilson :

| Gate | Seuil | Observé | Verdict |
|---|---:|---:|---|
| Couverture identifiable | ≥ 80 % | 83,585 % | PASS |
| Recall@100 exact | ≥ 99 % | 100 % | PASS |

## Références gelées publiées ensemble

Ces références portent sur un autre échantillon de 2 565 requêtes. Elles ne
constituent pas une comparaison appariée avec V4.12.

| Référence | Couverture | Recall@100 |
|---|---:|---:|
| Historique toutes requêtes | 2 565 / 2 565 = 100 % | 2 495 / 2 565 = 97,271 % |
| V2 exact | 2 400 / 2 565 = 93,567 % | 2 343 / 2 400 = 97,625 % |
| V3 exact identifiable | 2 104 / 2 565 = 82,027 % | 2 095 / 2 104 = 99,572 % |

## Plafond et coût

- 145 236 candidats ;
- pools de 46 à 100 candidats ;
- 13 pools sous 100 ;
- aucun pool vide ;
- plafond absolu de 100 respecté partout.

Durée worker agrégée : 1 030,16 secondes, soit 0,708 seconde par requête en
moyenne dérivée. Aucun temps individuel n'existe : p95 et SLA ne sont donc
pas publiés.

## Preuve d'exécution

Identités :

- evaluator build :
  `50cbc46e54fbef158a21250bd83f3a0c0ffddf85124293e8c195033626532e7c` ;
- measurement slot :
  `9cf7f6d335da1362abca7afe82d6de2eba494af2db0a6ae416fedcb41a4721b7` ;
- attempt :
  `01260473ef791bd7232d922474d3e279924139788edee6e3deb5e8161208c2ed` ;
- computed attestation :
  `e360d92da957d0fd04ff178ba5ac649e8e16ee1cc4ca00f5c9c2773d415198b2`.

La chaîne possède sept événements et termine `FINAL`. Les 16 entrées ont été
relues conformément au verrou ; les 12 lignes du ledger concordent avec
l'attestation. Aucun modèle, challenge ou test final n'a été ouvert.

La première invocation relative du CLI s'est arrêtée avant création du reçu
et avant oracle, car le contrat exige des chemins absolus. L'invocation
canonique suivante constitue l'unique tentative officielle.

## Décision et prochain geste

Le retrieval V4.12 mérite **GO sur développement** : pour les dossiers
identifiables, il ne perd aucun SIRET exact avant le ranking dans ce jeu.

Ce GO ne dégèle pas encore le ranker ou l'accepteur. Selon la directive
active, le prochain geste est le contrat puis l'unique évaluation retrieval
sur le test final fermé. Le verdict produit final restera `GO`, `PIVOT` ou
`STOP`.
