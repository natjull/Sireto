# Contrat final — Retrieval sélectif SIRET Recall@100

## Décision de périmètre

Le benchmark historique contient des relations CRM → SIRET qui ne sont pas
identifiables avec les champs disponibles : sites concurrents du même SIREN,
SIRET fermé contredit par un site actif et relations sans preuve directe de
nom ni d'adresse.

Le gate final devient donc double :

1. conserver au moins 80 % des requêtes dans le périmètre SIRET exact
   identifiable ;
2. placer le SIRET exact dans les 100 candidats pour au moins 99 % de ce
   périmètre.

Les autres requêtes ne sont pas supprimées. Elles deviennent `AMBIGUOUS` ou
`UNRESOLVED` et leur comportement produit cible est `REVIEW`.

## Qualification gelée

La qualification est calculée avant le retrieval et ne reçoit aucun hit, rang
ou score de modèle.

- V2 structure intra-SIREN : `docs/benchmark_v2_label_policy.md` ;
- V3 preuve directe : `docs/benchmark_v3_evidence_policy.md` ;
- politique V3 : commit `09b9d46`, SHA-256
  `f67d2aa5b9c691a41cc2e94751fda336f8d68e6202fda10c0921777bcbd3db44` ;
- builder V3 : logique commitée dans `cf7133c`, ouverture technique du split
  test dans `c6c8186`, SHA-256 final
  `9ebf636101de6cd73e4079fbcc14b012e655fdd6ff08910e00127ee915718dcc` ;
- snapshot établissement :
  `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` ;
- snapshot unité légale :
  `5c17354cdabe707beffaa965896e81d98c51bfc6ca150ceed59e4404c924ced8`.

Aucun seuil, mapping ou source d'alias ne peut être modifié après lecture du
test.

## Retrieval gelé

La sortie utilise l'admission déterministe déjà sélectionnée sur dev :

- fusion réciproque pondérée :
  `current_sparse=2`, `name_word=1`, `name_char=1`,
  `address_word=0,5`, `siren_head=1`, `name_exact=2`,
  `address_exact=2` ;
- quotas overlay : `name_word=1`, `name_char=10` ;
- profondeur interne maximale : 5 000 par canal ;
- tie-break : SIRET croissant ;
- sortie : 100 candidats maximum ;
- évaluateur : commit `5a0e67f`, fichier SHA-256
  `b24ee3f52ab5d713c92114ac13d3b1e99498bb40a3ca6cca015bb991dd237c45`.

Le ranker métier, le decider, le risk model et l'accepteur restent gelés.

## Référence dev

Build V3 : `ab8343817551c0a5`.

| Périmètre | Volume | Couverture | Recall@100 |
|---|---:|---:|---:|
| Toutes requêtes | 2 565 | 100 % | 97,271 % historique |
| V2 exact | 2 400 | 93,567 % | 97,625 % |
| V3 exact identifiable | 2 104 | 82,027 % | 99,572 % |

Sur le périmètre V3, l'oracle interne voit 2 104/2 104 vérités. Neuf sont
ensuite éliminées par l'admission à 100.

Ce résultat dev est rétrospectif : il autorise l'évaluation finale, mais ne
constitue pas à lui seul une validation indépendante.

## Gates du test final

Les conditions globales sont toutes obligatoires :

- couverture `MATCH_EXACT` V3 ≥ 80,0 % ;
- Recall@100 SIRET exact V3 ≥ 99,0 % ;
- zéro vérité V3 absente de l'oracle interne ;
- zéro sortie au-dessus de 100 candidats ;
- métriques historique, V2 et V3 publiées ensemble ;
- nombres bruts et intervalles de Wilson à 95 % et 99 % publiés.

Pour les segments contenant au moins 100 requêtes V3 exactes sur le test :

- couverture ne baissant pas de plus de 5 points absolus face au dev ;
- Recall@100 ne baissant pas de plus de 2 points absolus face au dev.

Références dev :

| Segment | Couverture | Recall@100 V3 |
|---|---:|---:|
| actifs | 84,986 % | 99,887 % |
| fermés | 69,405 % | 97,929 % |
| mégapoles | 87,273 % | 98,611 % |
| multi-sites | 79,479 % | 99,385 % |
| localisation INSEE | 82,185 % | 99,566 % |

Les segments plus petits sont publiés sans gate dur.

## Procédure unique sur le test

1. Produire les audits V2/V3 du test, sans aucun artefact de retrieval.
2. Geler et publier leurs manifests, hashes, couverture et volumes.
3. Exécuter les canaux V7 et overlay à la configuration gelée.
4. Appliquer une seule fois l'admission déterministe gelée.
5. Joindre les résultats à la qualification déjà produite.
6. Publier `GO`, `PIVOT` ou `STOP`.

Après l'étape 3, aucune nouvelle variante, correction de seuil ou seconde
lecture du test n'est autorisée. Un défaut d'instrumentation reproductible peut
être corrigé, mais le premier résultat reste conservé et documenté.

## Interprétation produit

Un `GO` signifie :

> Pour au moins 80 % des dossiers, les données disponibles soutiennent un
> SIRET exact et le retrieval le conserve dans 100 candidats au moins 99 % du
> temps.

Il ne signifie pas que 99 % de tous les CRM sont automatiquement matchés. Les
dossiers sans preuve directe restent à traiter par réparation CRM, registre
d'alias versionné ou revue.
