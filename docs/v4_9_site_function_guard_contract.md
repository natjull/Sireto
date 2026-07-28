# Contrat V4.9 — garde-fou déterministe de fonction de site

Statut : préenregistré après `STOP_RETRAIN` V4.8 et avant toute construction
de feature V4.9, toute mesure rétrospective et toute sélection fraîche.

Identifiant : `V49_SITE_FUNCTION_GUARD`.

## 1. Question

Un contrôle déterministe de la fonction exacte du site peut-il refuser les
confusions sémantiques à très haut score, sans perdre plus de deux points de
couverture AUTO ?

V4.9 ne modifie initialement ni retrieval, ni ranker, ni accepteur, ni seuil.
Elle ajoute uniquement un garde-fou `AUTO → REVIEW` après l'accepteur.

## 2. Point de départ et interdictions

Le verdict V4.8 est `STOP_RETRAIN`. La réserve random est définitivement
consommée :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout/f1ac35f4f7450b6a`

Hashes :

| Artefact | SHA-256 |
|---|---|
| registre global | `0f09e84d05d9b7f790544d4505a6288345bf338a361dd59f8dfa69edf49bb117` |
| `manifest.json` | `86aa84fe1084fadc416e151a2a46c12bd523aa66cc4d328e944e822ff78a6712` |
| `random_report.json` | `e253fee5a2263f102b2e905345d460f414c032301185ed78f6c83548818cd568` |
| `random_predictions.parquet` | `9d236f7d53c167a2a970c77fac9e34ef9540cd60418e917d5dd4fb468fcc6e79` |

Il est interdit :

- de retuner le seuil `0.3617231974526733` sur ces 52 cas ;
- de présenter une mesure sur ces cas comme validation V4.9 ;
- de rouvrir le registre ou de créer un autre « random » à partir des mêmes
  dossiers ;
- de promouvoir `HARD_W1` ;
- d'ouvrir le test final historique.

Les 150 labels courants fiables peuvent servir au développement V4.9, puisque
leur rôle de validation est terminé. Toute conclusion doit cependant venir
d'une population fraîche.

## 3. Sources et calcul local

Le garde-fou utilise seulement :

- nom CRM brut et normalisé ;
- nom, enseigne, dénomination usuelle et activité principale du top-1
  SIRENE ;
- SIRET/SIREN et état administratif ;
- mêmes champs pour les candidats du même SIREN présents dans le top-100.

Snapshot SIRENE épinglé :

`data/StockEtablissement_utf8.parquet`, SHA-256
`c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845`.

Aucune API payante, aucun GPU et aucun LLM en inférence ne sont autorisés.

## 4. Taxonomie V4.9 à geler avant mesure

La taxonomie doit être un fichier de configuration versionné contenant :

- règles positives de détection par tokens ou expressions ;
- codes d'activité utilisables comme corroboration, jamais comme preuve
  unique lorsqu'ils sont trop génériques ;
- matrice de compatibilité entre fonctions ;
- version et tests unitaires.

Familles minimales :

- `ADMIN_MAIRIE` ;
- `EDU_MATERNELLE`, `EDU_PRIMAIRE`, `EDU_COLLEGE`, `EDU_LYCEE` ;
- `CHILDCARE_CRECHE` ;
- `MED_FAM`, `MED_MAS`, `MED_EAM`, `MED_IME`, `MED_EHPAD`,
  `MED_FOYER` ;
- `HEALTH_HOSPITAL`, `HEALTH_CLINIC`, `HEALTH_PHARMACY` ;
- `UNKNOWN` et `MULTI_ROLE`.

Une expression spécifique prévaut sur une expression générique. Les accents,
tirets, pluriels et abréviations sont normalisés de manière déterministe.

## 5. Décisions du garde-fou

Le garde-fou ne s'applique qu'à une décision que l'accepteur aurait mise en
AUTO.

Il produit `REVIEW` avec :

- `SITE_FUNCTION_CONFLICT` : CRM et top-1 portent deux fonctions connues et
  incompatibles ;
- `SITE_FUNCTION_AMBIGUOUS` : le CRM porte plusieurs fonctions incompatibles
  ou plusieurs candidats du même SIREN portent ces fonctions sans résolution
  unique ;
- `SITE_FUNCTION_INSUFFICIENT` : une fonction CRM très spécifique est connue,
  mais aucune fonction compatible n'est prouvée pour le top-1.

`UNKNOWN` seul ne suffit jamais à refuser. Une fonction compatible conserve
l'AUTO. Le garde-fou ne change jamais le SIRET proposé et ne crée jamais
d'AUTO.

## 6. Étape A — diagnostic rétrospectif

Après gel du lexique et avant toute population fraîche :

- reconstruire les fonctions sur les 172 scènes V4.7 ;
- mesurer sur les 150 labels fiables combien de mauvais, ambigus et corrects
  seraient refusés ;
- publier séparément les cas V4.4 transportés, V4.7 nouveaux et random
  consommés ;
- publier chaque bascule ligne à ligne avec la règle déclenchée.

Cette étape est exploratoire. Aucun taux rétrospectif n'autorise une
promotion.

`GO_FRESH_V49` exige seulement un signal minimal permettant de justifier le
coût d'une nouvelle adjudication :

- au moins cinq `TOP1_WRONG` ou `AMBIGUOUS` refusés ;
- au moins une erreur random V4.8 refusée ;
- au plus 5 % des `TOP1_CORRECT` rétrospectifs refusés ;
- zéro règle utilisant un identifiant de dossier ou un libellé complet
  propre à un exemple.

Sinon : `PIVOT_TAXONOMY` ou `STOP_SITE_FUNCTION_GUARD`.

## 7. Étape B — évaluation fraîche

Si `GO_FRESH_V49` :

1. geler code, taxonomie, matrice, accepteur et seuil ;
2. tirer avant scoring au moins 300 lignes CRM absentes de toutes les
   populations V4.3–V4.8 ;
3. reconstruire retrieval V4.2-B et ranker A sans positif injecté ;
4. appliquer baseline et garde-fou gelés ;
5. adjudiquer autonomement les top-1 avec au moins deux groupes de preuves
   indépendants, sans demander de validation à l'utilisateur ;
6. conserver `UNRESOLVED` hors précision mais dans le compte de couverture
   identifiable ;
7. ne modifier aucune règle après la première lecture des labels frais.

Le gate de faisabilité fraîche exige :

- zéro erreur parmi les AUTO V4.9 adjudiqués fiables ;
- au moins une erreur baseline fraîche correctement refusée, si la population
  en contient ;
- perte de couverture AUTO ≤ 2 points ;
- aucun segment critique perdant plus de 5 points de couverture ;
- tous les nombres bruts et intervalles de Wilson publiés.

Avec 300 dossiers, un zéro erreur reste une observation de faisabilité et non
une certification à 99,8 %.

## 8. Suite après faisabilité

`GO_SHADOW_V49` autorise uniquement un shadow sans écriture CRM. La North
Star à 99,8 % exigera ensuite une cohorte indépendante beaucoup plus grande ;
les quelque 2 300 AUTO sans erreur nécessaires à une borne unilatérale 99 %
ne sont pas remplacés par le petit gate V4.9.

Autres verdicts :

- `PIVOT_TAXONOMY` ;
- `STOP_SITE_FUNCTION_GUARD` ;
- `STOP_FRESH_INTEGRITY` ;
- `STOP_FRESH_SAFETY`.

Chaque milestone est un commit isolé cité dans `handover.md`.
