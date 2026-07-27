# V4.3 — Labels représentatifs difficiles

Date : 27 juillet 2026  
Verdict : **`PIVOT_VALIDATION`**

## Conclusion directe

La file de cas difficiles est maintenant construite, complète et priorisée.
Elle démontre aussi pourquoi il serait incorrect de réentraîner immédiatement
les modèles : **aucun nouveau label n'a aujourd'hui une autorité suffisante
pour l'entraînement**.

Les 542 cas `UNRESOLVED` de l'audit V4.1 sont tous présents :

- 172 décisions `AUTO_MATCH` ;
- 370 décisions `REVIEW` ;
- 144 cas issus du tirage aléatoire ;
- cinq `WRONG_TOP1` déjà documentés, encore `AI_PROVISIONAL` ;
- 537 cas toujours `UNRESOLVED` ;
- zéro cas `training_eligible`.

La V4.3 ne bloque pas sur le code ou le calcul. Elle bloque sur la vérité
terrain indépendante. Inventer cette vérité avec une règle, le score du modèle
ou un avis IA recréerait le biais qui a produit les scores V4.1 trop optimistes.

## File de priorité

| Priorité | Cas | Interprétation |
|---|---:|---|
| Erreurs provisoires déjà démontrées | 5 | mauvais nom, établissement ou type |
| AUTO portés par l'adresse avec nom faible | 35 | risque élevé de cohabitation d'entités |
| AUTO différents d'un SIRET d'entrée actif | 28 | migration possible ou mauvais top-1 |
| Autres AUTO non résolus | 104 | doivent tous être vérifiés |
| REVIEW proches du seuil | 137 | meilleurs candidats à une future couverture |
| REVIEW sans candidat actif | 48 | problème de preuve ou de source |
| Autres REVIEW | 185 | priorité plus faible |

Les signaux de nom et d'adresse servent uniquement à ordonner la file. Un
sigle, une fusion, une reprise d'activité ou un changement de raison sociale
peut expliquer une faible ressemblance. Aucun signal n'est donc converti
automatiquement en label d'entraînement.

## Audit du « gold standard » historique

Le fichier `data/crm_ok_gt.csv` ne constitue pas une validation humaine. Le
script historique `scripts/audit_gt.py` l'a produit en conservant le SIRET CRM
dès que la commune **ou** le code postal du SIRET correspondait au CRM.

Sur les 542 cas difficiles :

- 313 figurent dans ce fichier historique ;
- 116 des 172 AUTO y figurent ;
- pour 40 de ces 116 AUTO, le top-1 V4.1 diffère du SIRET historique ;
- parmi ces 40 divergences, 19 SIRET d'entrée sont encore actifs et 21 sont
  fermés.

Certaines divergences sont des corrections légitimes vers un établissement
actif ou une nouvelle entité ; d'autres sont clairement mauvaises. Le filtre
géographique historique ne permet pas de les départager. L'utiliser comme
vérité pour réentraîner reviendrait à apprendre les erreurs du CRM.

## Livrables

L'artefact final contient :

- `hard_label_queue.parquet` : les 542 cas et toutes les preuves ;
- `auto_priority.csv` : les 172 AUTO à auditer en premier ;
- `human_adjudication_template.csv` : la file complète ;
- `human_adjudication_batch250.csv` : le premier lot opérationnel de 250 cas ;
- `summary.json` et `manifest.json`.

Le lot de 250 commence par les 172 AUTO, puis les REVIEW les plus informatives.
Chaque ligne contient le CRM, le SIRET d'entrée, son état, le top-1, les noms et
adresses SIRENE, les signaux de risque et les autres preuves candidates.

## Ce qu'un spécialiste fait maintenant

Il ne change pas les features et ne baisse pas le seuil. Il fait adjuger le lot
de 250 par une source indépendante :

1. vérifier si le top-1 désigne réellement l'entité CRM ;
2. renseigner `MATCH_EXACT`, `WRONG_TOP1`, `AMBIGUOUS` ou `UNRESOLVED` ;
3. conserver le SIRET exact et les références de preuve lorsqu'ils existent ;
4. enregistrer le validateur ;
5. faire contrôler les cas litigieux par une seconde personne ;
6. geler le fichier avant de relancer un entraînement.

Le gate préenregistré exige au minimum 100 positifs difficiles, 100 mauvais
top-1 et 50 ambigus, tous entraînables. Il n'est pas atteint.

## Limites rencontrées

Le classeur historique `reports/old/révisionhumaine.xlsx` n'a pas été utilisé :
le lecteur tableur contrôlé n'était pas disponible dans la session. Son contenu
ne peut donc pas être revendiqué comme validation ou joint à la file sans
vérification.

Le disque interne est par ailleurs saturé. Seuls les caches Python/pytest
régénérables du dépôt ont été supprimés, libérant environ 270 Mo. Aucun
dataset, modèle ou résultat n'a été supprimé. Les nouveaux artefacts V4.3 sont
sur `/Volumes/CATNAT_DATA`, qui dispose d'environ 1,2 To libre.

## Décision

**`PIVOT_VALIDATION`** :

- retrieval V4.2 validé ;
- architecture aval toujours plausible ;
- nouveaux exemples difficiles identifiés ;
- vérité indépendante insuffisante pour réentraîner honnêtement.

Le statut reste **`STOP_DEPLOYMENT`** et **`NO_RETRAIN`** jusqu'à validation
du lot.

## Artefact

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_3_hard_labels/0f832305ab199267`

## Provenance Git

- contrat : `a2232bf` ;
- constructeur de file : `c3c5944` ;
- correction des sigles et noms soudés : `3388649` ;
- export du lot de 250 : `b16ce8b` ;
- suite complète : 221 tests passants.
