# V4.12 — audit complémentaire de 30 REVIEW historiques

## Verdict métier

| Résultat | Nombre |
|---|---:|
| `MATCH_EXACT` fiable | **27** |
| `AMBIGUOUS` | **3** |
| `UNRESOLVED` | **0** |
| Labels exacts utilisables | **27** |
| Anciennes qualifications à corriger | **26 / 30** |

Les cinq premiers dossiers sont détaillés dans
[`v412_remaining_review_audit_first5.md`](v412_remaining_review_audit_first5.md)
et les 25 suivants dans
[`v412_remaining_review_audit_next25.csv`](v412_remaining_review_audit_next25.csv).

## Ce que le pipeline faisait réellement

Sur les 27 dossiers exacts :

- le ranker avait déjà le bon SIRET en tête dans **24 cas** ;
- l'accepteur les envoyait tous en REVIEW ;
- le ranker se trompait dans **3 cas** : centre hospitalier restructuré, CESI
  Association contre CESI SAS, et Institut Lemonnier contre sa filiale CFC.

Les trois `AMBIGUOUS` sont justifiés : changement d'exploitant LG Alès,
succession de structures Constructys sans date CRM, et deux entités Promotrans
actives sous la même marque à la même adresse.

## Tableau métier des 25 dossiers suivants

| ID | CRM | Décision | Cause principale |
|---:|---|---|---|
| 1264 | INTEVA PRODUCTS FRANCE | `60201069600147` | Top 1 exact, fausse ambiguïté liée à une société proche. |
| 1401 | CH Aunay-sur-Odon | `26140092300320` | Vérité historique obsolète ; le site hospitalier actuel était retrouvé mais mal classé. |
| 1739 | JM PRESTATIONS | `49011913800025` | Nom et adresse exacts. |
| 1772 | GRANDEUR NATURE SERVICES | `80499169300016` | Entité liée co-localisée, mais dénomination CRM discriminante. |
| 2036 | GEEMARC TELECOM SA | `40089260000026` | Société Geemarc voisine, mais nom juridique complet discriminant. |
| 2413 | BNP PARIBAS JARVILLE | `66204244922034` | Agence et adresse exactes. |
| 2423 | FIDAL NANCY | `52503152200143` | Prédécesseur fermé ; établissement Fidal actif exact. |
| 2483 | VANDIS | `50760857800021` | Siège VANDIS / Centre Leclerc exact. |
| 2499 | CONSTRUCTYS | `AMBIGUOUS` | Deux structures Constructys temporellement plausibles, pas de date CRM. |
| 2525 | ALLIANCES INFORMATIQUE GESTION | `41871093500036` | Nom, enseigne AIG et adresse exacts. |
| 2682 | IMARA Saint-Julien | `44275189700061` | Centre médical exact ; SCI co-localisée non pertinente. |
| 2999 | AGENCE OLIVIER | `78886354600016` | Dénomination exacte malgré un quasi-homonyme voisin. |
| 3201 | EHPAD LES COULEURS DU LAC | `26740006700018` | EHPAD exact ; autre NIC de dotations non affectées. |
| 3381 | CAMPUS DE GROISY | `30023144600018` | Dénomination et adresse exactes. |
| 3481 | IMARA 74 Faverges | `44275189700087` | Centre médical exact ; SCI co-localisée non pertinente. |
| 3673 | RITSCHARD | `42007266200018` | Dénomination et adresse exactes. |
| 4598 | SATCOMS | `32506104200025` | Société nue exacte parmi plusieurs sociétés sœurs qualifiées. |
| 4655 | CESI ASSOCIATION | `77572257200051` | Ranker trompé par CESI SAS à la même adresse. |
| 4692 | CROUS HSC | `13002442500600` | Entité CROUS actuelle exacte ; ancien SIRET obsolète. |
| 4720 | MBC DISTRIBUTION | `39000406700034` | Nom et adresse exacts. |
| 4759 | FIDAL CAEN | `52503152200713` | Variation d'adresse `1` / `1 ter`, même établissement. |
| 4769 | IBC DIALOG | `78980027300011` | Entité actuelle exacte ; ancien IBC Dialog fermé. |
| 5020 | INSTITUT LEMONNIER | `78071394700015` | Ranker trompé par la filiale CFC co-localisée. |
| 5052 | PROMOTRANS | `AMBIGUOUS` | Association et FPC actives sous la même marque au même lieu. |
| 5139 | CROIX-ROUGE CALAIS | `77567227232127` | IFSI exact ; abstention excessive de l'accepteur. |

## Causes agrégées

1. **Labels historiques faux ou trop prudents** : 24 anciens `AMBIGUOUS`
   deviennent exacts sur l'ensemble des 30, tandis qu'un ancien exact devient
   ambigu et un ancien exact change de SIRET.
2. **Accepteur entraîné sur des faux négatifs** : il apprend à refuser des
   top 1 parfaitement exacts parce que la cible historique les classe comme
   ambiguës.
3. **Co-localisation d'entités liées** : le ranker doit mieux exploiter les
   qualificatifs `ASSOCIATION`, `CFC`, `SAS`, ainsi que l'absence d'un suffixe
   spécialisé.
4. **Temporalité absente** : les restructurations hospitalières, changements
   d'exploitant et prédécesseurs rendent certains dossiers impossibles à
   arbitrer sans date de référence.

## Décision pour la suite

Les échecs des expériences de pondération et de variables relationnelles ne
prouvent pas que l'accepteur est irréparable : elles ont été conduites avec un
lot de contrôle encore fortement contaminé. Le prochain travail utile est de
réétiqueter un volume suffisant des REVIEW restants, puis de reconstruire le
jeu de développement avec les labels corrigés. Réentraîner avant ce nettoyage
reproduirait le même biais.
