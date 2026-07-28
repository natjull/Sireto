# V4.11 — Challenge descriptif inédit exécuté une seule fois

Date : 28 juillet 2026

## Verdict

**`PIVOT_ACCEPTOR_EVIDENCE_GATE`**

L'intégrité de l'unique exécution est validée, mais la North Star produit
n'est pas démontrée.

- Le retrieval conserve le bon SIRET dans 74/74 cas exacts observés, avec un
  plafond strict de 100 candidats.
- Le ranker place le bon SIRET en première position dans 74/74 cas exacts.
- L'accepteur automatise un cas `AMBIGUOUS`.
- Sur les 75 décisions AUTO évaluables, 74 sont correctes et une est une
  erreur confirmée : 98,667 % de précision observée.
- 73 autres décisions AUTO portent sur des labels `UNRESOLVED` et restent
  invérifiables. Elles ne sont comptées ni comme succès ni comme erreurs
  certaines.
- La qualification ne fournit un SIRET exact que pour 74/225 lignes
  (32,889 %). Cette population sans `SERVICE ID` est atypique et ne peut pas
  servir de certification produit.

Ce résultat ne justifie ni un `GO` produit ni un `STOP` du projet. Il déplace
le prochain travail du retrieval vers la preuve d'unicité et l'accepteur.

## Artefact et intégrité

Artefact immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_execution/ddb7336e8c2e042d`

- Ledger global : `COMPLETED`
- Run ID : `ddb7336e8c2e042d`
- Hash du manifeste :
  `37f4957052493b3aa1e8b2e3ba5f156816cb33121aa5915f88c9b581306c71e6`
- Hash des prédictions aveugles scellées :
  `ac4e9d2ee8cb112039f4242d51bb10c7eee95b771f8ef1d2a358bb8d8fa1b392`
- Verrou d'exécution :
  `ae4fea1cec771f092b31179706866accb1e31599a93bbb07dab4385cc286094b`

Le contre-audit indépendant confirme :

- les huit sorties, leurs tailles et leurs hashes ;
- les 225 requêtes uniques et les cohortes 222/3 ;
- 22 483 candidats actifs, uniques, aux rangs contigus ;
- un maximum strict de 100 candidats par requête ;
- la concordance ledger, manifeste, lock, modèles, sources et prédictions ;
- l'ouverture des labels uniquement après scellement des prédictions ;
- l'absence de réentraînement et de sélection de seuil.

## Résultats

| Cohorte | Exact / Ambigu / Non résolu | Recall@100 exact | Hit@1 exact | AUTO | Correct / erreur confirmée / invérifiable | Précision évaluable |
|---|---:|---:|---:|---:|---:|---:|
| Aveugle 222 | 73 / 17 / 132 | 73/73 = 100 % | 73/73 = 100 % | 146 | 73 / 1 / 72 | 73/74 = 98,649 % |
| Exposée 3 | 1 / 0 / 2 | 1/1 = 100 % | 1/1 = 100 % | 2 | 1 / 0 / 1 | 1/1 = 100 % |
| Total 225 | 74 / 17 / 134 | 74/74 = 100 % | 74/74 = 100 % | 148 | 74 / 1 / 73 | 74/75 = 98,667 % |

Sur la cohorte aveugle :

- couverture AUTO brute : 146/222 = 65,766 % ;
- couverture parmi les seuls `MATCH_EXACT` : 73/73 = 100 % ;
- AUTO parmi les cas non `UNRESOLVED` : 74/90 = 82,222 % ;
- qualification `MATCH_EXACT` : 73/222 = 32,883 %.

Les estimations ponctuelles à 100 % ne constituent pas une preuve à 99 % ou
99,8 %. Avec 73 succès sur 73, la borne basse Wilson bilatérale à 95 % est
seulement voisine de 95,0 %.

## Cause de l'erreur confirmée

Le retrieval et le ranker ne sont pas en cause :

- les deux candidats directement plausibles étaient présents ;
- ils étaient classés premier et deuxième ;
- ils appartenaient à deux SIREN différents ;
- chacun possédait des preuves fortes de nom et d'adresse.

Le premier candidat avait toutefois une ressemblance de nom nettement
supérieure. Le ranker a créé un écart très fort et l'accepteur a interprété
cet écart comme une preuve d'identité unique. Son score `0,876593` a dépassé
le seuil gelé `0,872092` de seulement `0,004501`.

La scène actuelle représente correctement la concurrence des scores et la
concurrence entre établissements d'un même SIREN. Elle ne représente pas
explicitement la situation différente rencontrée ici : plusieurs identités
absolument plausibles appartenant à plusieurs SIREN.

## Orientation V4.12

La prochaine variante doit être développée sur les anciennes populations
fit/dev uniquement, puis gelée avant tout nouvel export indépendant.

1. Ajouter une garde déterministe, sans label :
   si au moins deux SIREN distincts satisfont individuellement la politique
   de rapprochement direct nom + adresse, forcer `REVIEW`.
2. Ajouter à la scène :
   le nombre de candidats forts, le nombre de SIREN forts, la force absolue
   du meilleur concurrent, l'indicateur de localisation forte partagée entre
   SIREN et l'accord d'activité des concurrents.
3. Miner dans les anciennes données les ambiguïtés à grand écart de ranker,
   sans réutiliser ce challenge pour choisir une règle ou un seuil.
4. Comparer d'abord la garde seule au candidat V4.11, puis seulement un
   accepteur réentraîné avec les nouvelles features.
5. Geler politique, modèle et seuil avant un nouvel export CRM indépendant.

Il est interdit de relever simplement le seuil au-dessus de `0,876593` :
ce serait un réglage direct sur un challenge consommé et ne corrigerait pas la
cause architecturale.
