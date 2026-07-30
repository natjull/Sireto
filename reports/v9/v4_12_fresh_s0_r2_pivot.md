# V4.12 — Verdict de l'unique exécution synthétique S0-R2

## Verdict

`PIVOT_R2_WORKER_IDENTITY`

L'unique tentative R2 est consommée et ne doit jamais être relancée. Elle
n'a ouvert aucun CRM réel, n'a produit aucune sortie métier et n'autorise
toujours pas l'étape de qualification fraîche.

## Autorités immuables

- `synthetic_run_id` :
  `bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo`
- `attempt_id` :
  `dhlmigejpmdehbjbppcfmlnkbcehcmgagojmnmibhcdnliicljifmjegieiogmmm`
- commit d'autorisation : `5dbb2ff`
- receipt terminal :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r2/audit/bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo/parent/launch_receipts/dhlmigejpmdehbjbppcfmlnkbcehcmgagojmnmibhcdnliicljifmjegieiogmmm.json`
- SHA-256 du receipt :
  `6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e`
- SHA-256 du claim :
  `cc825f40fe8c9596767891a7fac14b338c87a0a2dc6d2ff288ad0d1aa44cff17`
- SHA-256 du message terminal worker :
  `b6c2a5d27e8e00a1168a802e1dc1e2f2f3fcec1eb21bfe368c266205df15075e`

Le receipt est un objet JSON canonique valide : UTF-8 strict, clés uniques,
ordre canonique, représentation compacte et exactement un saut de ligne
final. Les observations des autorités parent sont identiques avant et après
l'enfant.

## Ce qui a fonctionné

- le worker Python a réellement démarré ;
- les onze canaris interdits ont tous été refusés avec `errno=1` ;
- le même processus a conservé les mêmes cinq FD de payload pendant
  `60.005023459` secondes ;
- stdout et stderr sont restés vides ;
- la fixture, le contrôle et les cinq payloads passent leurs validations
  déterministes en diagnostic de lecture seule ;
- le CSV est valide, contient six lignes et doit suivre la branche
  synthétique `INGESTED`.

Il ne s'agit donc ni d'une violation du bac à sable, ni d'un rejet normal de
la fixture, ni d'un résultat du scanner métier.

## Cause reconstruite

Le builder R2 et le worker n'emploient pas la même dérivation d'identité.

Le builder R2 calcule le nouveau run avec :

1. le domaine R2
   `SIRETO-V412-FRESH-SYNTHETIC-S0-R2-RUN-ID\0` ;
2. le hash du fragment `fixture` ;
3. le hash du plan cœur ;
4. le hash du receipt R1 prédécesseur.

Cette formule produit le run R2 autorisé `bjpoib...`.

Le worker copié dans le runtime recalcule encore l'ancienne identité du cœur
avec le domaine R1 et seulement :

1. le hash du fragment `fixture` ;
2. le hash du plan cœur.

Cette formule produit `komapn...`, puis un attempt différent de l'attempt R2.
La première condition métier de `_process` échoue donc nécessairement sur
`worker spec/control deterministic identity mismatch`, avant la création de
la moindre autorité de sortie.

Cette reconstruction est déterministe et en lecture seule. Elle concorde
avec l'état physique : aucun fichier n'existe sous `sealed`, `scan`,
`quarantine` ou `tmp`; seul le rapport de canaris a été écrit.

## Pourquoi le receipt ne donne pas cette cause

Le `main` du worker capture toute `Exception` sans conserver son objet, puis
émet toujours le même code `WORKER_CONTROLLED_STOP`. Cette perte
d'observabilité masque l'étape et la cause interne. Le diagnostic précis
provient donc de la comparaison des deux dérivations épinglées et de
l'exécution en lecture seule des validations antérieures, pas d'un message
d'erreur présent dans le receipt.

Les tests ayant précédé le GO validaient séparément le builder, le schéma du
spec et le protocole du launcher. Aucun n'appelait `_process` avec l'identité
R2 réelle. Ils ne pouvaient donc pas détecter cette incohérence.

## Condition minimale d'un successeur

Avant de préenregistrer un R3 :

1. le worker doit dériver explicitement l'identité successeur depuis une
   autorité fermée et épinglée ;
2. le worker doit publier dans tout STOP un `phase` et un `reason_code`
   appartenant à des ensembles fermés, sans contenu CRM ;
3. un test d'intégration jetable doit exécuter le vrai `_process` avec
   l'identité successeur et atteindre `INGESTED` sous le même bac à sable ;
4. le même test doit démontrer que l'identité R1 est rejetée ;
5. seulement après contre-audit de ce gate, une nouvelle racine, un nouveau
   run et un nouvel attempt pourront être préenregistrés.

R2 reste une preuve utile : la frontière système et la stabilité fonctionnent.
Le pivot porte uniquement sur la cohérence d'identité et l'observabilité du
worker.
