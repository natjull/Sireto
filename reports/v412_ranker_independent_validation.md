# V4.12 — Validation indépendante du ranker candidat

Date : 31 juillet 2026  
Docket préenregistré : commit `c39dfb1`, antérieur à toute consultation des vérités.

## Résultat

| Requête | CRM | Vérité adjudiquée | Baseline | Candidat | Verdict |
|---|---|---:|---:|---:|---|
| `10395` | PROMOTRANS LYON | `AMBIGUOUS` | `77568013501063` | `77568013501071` | aucun gagnant |
| `1495` | PHB CREATION | `90370122500018` | `53465464500025` | `90370122500018` | candidat |
| `3165` | Centre Hospitalier Dufresne Sommeiller | `26740017400012` | `26740017400046` | `26740017400012` | candidat |
| `4522` | UNIVERSITE D ARTOIS | `19624401600016` | `19624401600255` | `19624401600016` | candidat |
| `5708` | ALMERYS | `43270163900069` | `88414992300010` | `43270163900069` | candidat |
| `fresh:FR031148` | ROCHA SA | `30202053200073` | `44415907300014` | `30202053200073` | candidat |
| `fresh:FR031197` | BIGGIE GROUP | `91359061800033` | `49283100300057` | `91359061800033` | candidat |

Sur les six dossiers exacts :

- baseline : **0/6** ;
- candidat pondéré `0,5` : **6/6** ;
- ambiguïtés forcées en positif : **0** ;
- vérités absentes des 100 candidats : **0**.

Ce lot est exhaustif parmi les 196 REVIEW vierges où les deux modèles ne sont pas d'accord. Il valide donc les bascules effectivement produites sur cette population ; il ne mesure pas la précision générale du système et reste beaucoup trop petit pour une autorisation produit.

## Preuves dossier par dossier

### PROMOTRANS LYON — `AMBIGUOUS`

L'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/775680135) publie deux établissements actifs, `77568013501063` et `77568013501071`, créés le même jour, à la même adresse et avec la même activité. Le premier porte l'enseigne `PROMOTRANS DCR SUD`, le second aucun nom d'établissement. Le CRM générique « PROMOTRANS LYON » ne fournit aucune preuve permettant un choix SIRET exact.

### PHB CREATION — `90370122500018`

Le registre de [PHB CREATION MAINTENANCE](https://www.pappers.fr/entreprise/phb-creation-maintenance-903701225) associe l'entité, son adresse exacte et son activité d'aménagement paysager. Le portfolio [PHB CRÉATION](https://www.phb-creation.com/wp-content/uploads/2025/12/Book-PHB-2025-.pdf) publie la même adresse et la même marque. Le baseline `53465464500025` est une société d'architecture `PHB`, co-localisée mais distincte.

### Centre Hospitalier Dufresne Sommeiller — `26740017400012`

La fiche [FINESS officielle](https://finess.esante.gouv.fr/fininter/jsp/actionDetailEtablissement.do?noFiness=740000286) rattache le centre hospitalier générique à `26740017400012`. L'avis [INSEE de l'autre candidat](https://api-avis-situation-sirene.insee.fr/identification/pdf/26740017400046) indique explicitement l'enseigne spécialisée `USLD`. Le CRM ne désigne pas cette unité de soins de longue durée.

### UNIVERSITE D ARTOIS — `19624401600016`

Le [site officiel](https://www.univ-artois.fr/contacts) décrit le 9 rue du Temple comme siège de l'Université d'Artois. Un annuaire de formation publie le [SIRET `19624401600016`](https://www.intercariforef.org/formations/universite-dartois/organisme-20_100000800.html). Le baseline correspond à l'UFR Économie-Gestion présente à la même adresse, alors que le CRM désigne l'université entière.

### ALMERYS — `43270163900069`

Les [mentions légales officielles d'Almerys](https://www.almerys.com/accueil/legal-information) donnent le SIREN `432701639` et le 46 rue du Ressort. Le registre conserve `43270163900069` comme établissement actif à cette adresse. Le baseline est la filiale distincte `ALMERYS SOFTWARE`.

### ROCHA SA — `30202053200073`

Le site d'emploi Rocha publie dans ses [mentions légales](https://jobs.rocha.fr/mentions-legales/) le RCS `302020532` pour « ROCHA SA ». Les actes de 2024 montrent que cette société, désormais [ROCHA HOLDING](https://www.pappers.fr/entreprise/rocha-holding-302020532), a apporté sa branche opérationnelle à `ROCHA E.V.A.`. La politique actif-au-snapshot conserve l'identité juridique issue du CRM, pas la société bénéficiaire de l'apport.

### BIGGIE GROUP — `91359061800033`

L'[Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/913590618) publie exactement `BIGGIE GROUP`, le SIRET `91359061800033` et l'adresse CRM. Le baseline `49283100300057` est la société opérationnelle `BIGGIE`/`REPEAT`, dirigée par le groupe mais juridiquement distincte.

## Décision

Verdict : **`GO_RANKER_CANDIDATE_FOR_ACCEPTOR_DEVELOPMENT`**.

Le candidat pondéré `0,5` a franchi son premier lot indépendant sur toutes les décisions exactes où il diverge. Il peut maintenant produire les scènes hors échantillon nécessaires au développement de l'accepteur sélectif. Il reste interdit en production : la North Star concerne la précision des décisions `AUTO_MATCH`, pas seulement le top 1 du ranker, et six décisions exactes ne fournissent aucune certification à 99,8 %.
