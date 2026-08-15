# Politique métier — équivalence opérationnelle de SIRET sur un même site

Date de décision métier : 15 août 2026  
Statut : active pour les futurs jeux de labels, entraînements, évaluations et
sorties produit ; sans effet rétroactif sur les cycles déjà gelés.

## Décision

Pour l'usage CRM de SIRETO, une prédiction est **opérationnellement correcte**
si elle désigne :

1. le SIRET exact de la vérité ; ou
2. un autre SIRET du même SIREN situé à la même adresse physique que le CRM.

Le second cas est nommé `OPERATIONAL_EQUIVALENT_SAME_SITE`. Il est correct du
point de vue métier même si le SIRET diffère de la vérité historique. Lorsque
la vérité historique est fermée et que le candidat équivalent est actif, le
candidat actif est en outre le résultat opérationnel préféré ; le motif publié
est `ACTIVE_SUCCESSOR_SAME_SITE`.

Cette décision ne prétend pas que deux numéros SIRET sont juridiquement
identiques. Elle définit quels résultats sont acceptables pour rattacher une
fiche CRM à la bonne personne morale et au bon site physique.

## Preuve minimale de « même adresse »

L'équivalence exige une adresse CRM exploitable et une concordance forte du
site :

- numéro de voie concordant, y compris suffixe `bis/ter` lorsqu'il est fourni ;
- voie concordante après normalisation déterministe des accents, ponctuation
  et types de voie ;
- code postal concordant, ou code INSEE concordant lorsque le code postal est
  absent ou correspond à une commune multi-CP ;
- même SIREN à neuf chiffres.

Un code postal, une commune, une zone d'activité ou une similarité floue seuls
ne suffisent jamais. Les compléments d'adresse (bâtiment, étage, résidence)
peuvent différer sans invalider le même site, mais cette différence doit être
publiée. Une adresse non exploitable ne peut pas produire cette équivalence.

## États administratifs et préférence de sortie

Les états ne changent pas l'appartenance à l'ensemble acceptable, mais fixent
la préférence opérationnelle :

1. candidat actif au même site lorsque le SIRET exact est fermé ;
2. SIRET exact actif ;
3. autre SIRET actif équivalent au même site ;
4. SIRET fermé lorsque aucun équivalent actif n'existe.

Si plusieurs SIRET actifs du même SIREN partagent le site, ils sont tous
opérationnellement acceptables selon cette politique. La sortie doit toutefois
publier `MULTIPLE_ACTIVE_EQUIVALENTS` et conserver la liste des SIRET
acceptables pour l'audit.

## Schéma de labels requis

Une ligne qualifiée doit conserver deux cibles distinctes :

- `ground_truth_siret_exact` : vérité historique inchangée ;
- `acceptable_sirets_operational` : ensemble contenant le SIRET exact et les
  SIRET du même SIREN prouvés au même site.

Champs dérivés recommandés :

- `exact_siret_correct` ;
- `operational_siret_correct` ;
- `operational_equivalence_reason` ;
- `predicted_siret_state` et `ground_truth_siret_state` ;
- `same_site_evidence` et version de la normalisation d'adresse ;
- `multiple_active_equivalents`.

La vérité exacte n'est jamais réécrite silencieusement par le modèle, par une
règle de score ou par le générateur synthétique.

## Métriques obligatoires

Tous les résultats futurs publient ensemble :

1. Hit/Recall et précision **SIRET exacts**, selon les contrats historiques ;
2. Hit/Recall et précision **opérationnels** selon la présente politique ;
3. nombre de réponses promues uniquement par
   `OPERATIONAL_EQUIVALENT_SAME_SITE` ;
4. sous-total `ACTIVE_SUCCESSOR_SAME_SITE` ;
5. nombre de cas `MULTIPLE_ACTIVE_EQUIVALENTS`.

La métrique opérationnelle ne peut jamais être présentée comme une métrique
SIRET exacte. Les gates exacts déjà préenregistrés restent inchangés.

## Conséquences pour BGE, CamemBERT, XGBoost et FusionSet

Dans tout nouveau cycle visant la justesse opérationnelle :

- un sibling du même SIREN au même site ne doit pas être étiqueté comme hard
  negative ;
- les losses groupwise/pairwise doivent accepter plusieurs positifs, ou retirer
  les équivalents du dénominateur négatif ;
- les datasets candidats doivent porter séparément `is_exact_positive` et
  `is_operational_positive` ;
- l'accepteur peut automatiser un équivalent seulement si la preuve de même
  SIREN et de même site passe de manière déterministe ;
- le choix préférentiel d'un candidat actif face à une vérité fermée est une
  feature/règle produit explicite, jamais une correction cachée du label.

Les modèles et évaluations déjà gelés ne sont ni réentraînés ni rescored par la
présente décision. Toute comparaison rétroactive est publiée comme une analyse
secondaire séparée.

## Conséquences pour le corpus synthétique Luna

Le générateur et l'assembleur de hard negatives doivent séparer :

- `SAME_SIREN_SAME_SITE_EQUIVALENT` : positif opérationnel, jamais négatif ;
- `SAME_SIREN_OTHER_SITE` : véritable hard negative de site ;
- `ACTIVE_SUCCESSOR_SAME_SITE` : positif opérationnel préféré lorsque la
  vérité exacte est fermée.

L'éligibilité et la preuve d'adresse restent établies sans utiliser les scores,
rangs ou hits des modèles. Les erreurs OOF peuvent ensuite servir uniquement à
prioriser les familles et les seeds autorisés.

