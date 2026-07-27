# Contrat V4.5 — réentraînement expérimental de l'accepteur

Statut : préenregistré avant tout réentraînement et avant tout calcul de seuil
sur les adjudications V4.4.

## But

Tester si les erreurs difficiles prouvées en V4.4 permettent à l'accepteur de
rejeter davantage de mauvais top-1, sans réduire la précision SIRET exacte ni
détruire la couverture sur les scènes historiques.

Ce contrat n'autorise ni déploiement, ni revendication de précision à 99,8 %.
Il autorise uniquement une expérience locale lorsque le gate V4.4 vaut
`GO_RETRAIN_AUTO`.

## Composants gelés

- retrieval V4.2, variante B, plafond absolu de 100 candidats ;
- ranker V4.1 `a11b1356b8526165` ;
- ordre et calcul des features candidat et scène ;
- snapshot SIRENE et modèles sémantiques épinglés ;
- test final historique, qui reste fermé.

Le premier essai ne modifie que l'accepteur logistique. Aucun XGBoost
supplémentaire et aucun nouveau ranker ne sont entraînés dans cette étape.

## Liaison des labels aux scènes

Une adjudication V4.4 juge le top-1 figé du shadow V4.1. Elle ne peut pas être
appliquée aveuglément à une autre prédiction.

Pour chaque dossier :

1. rejouer le retrieval V4.2 et le ranker gelé, sans positif injecté ;
2. reconstruire les 80 features de scène avec les fonctions train/serve
   communes ;
3. vérifier que le top-1 rejoué est exactement le top-1 adjudiqué ;
4. classer tout changement de top-1 en `SCENE_DRIFT` et l'exclure de
   l'entraînement jusqu'à une nouvelle adjudication de cette prédiction.

Un `TOP1_WRONG` sans SIRET alternatif exact reste une cible négative valide
pour l'accepteur, mais jamais pour le ranker.

## Données d'expérience

- socle : scènes OOF `fit` et scènes hors échantillon `dev` du bundle V4.1 ;
- ajout : scènes V4.4 validées, après contrôle `SCENE_DRIFT` ;
- groupes anti-fuite : composantes construites avec le SIREN d'entrée, le
  SIREN prédit et tous les SIREN du pool figé ;
- aucun composant ne peut être partagé entre fit et évaluation ;
- les cas `UNRESOLVED` sont absents de toute cible ;
- les cas `AMBIGUOUS` peuvent entraîner l'abstention, jamais le ranker.

Les cas V4.4 ciblés servent au fit difficile. Les cas issus du tirage aléatoire
sont conservés pour une évaluation de dérive et ne servent pas à choisir les
features. Si un même composant contient les deux provenances, tout le composant
va du côté évaluation.

## Variantes appariées

1. accepteur V4.1 gelé ;
2. même régression logistique, réentraînée sur le socle plus les cas difficiles
   ciblés ;
3. variante 2 avec pondération des cas difficiles choisie dans
   `{1, 2, 4}` uniquement sur le dev historique.

Le seuil est choisi sans test final. Il doit satisfaire une précision SIRET
exacte observée d'au moins 99,8 % sur le dev historique et ne produire aucune
erreur AUTO sur la portion aléatoire V4.4 réservée. Les nombres bruts et les
intervalles de confiance sont toujours publiés ; zéro erreur sur un petit
échantillon n'est pas une garantie.

## Gate expérimental

`GO_SHADOW_V45` exige simultanément :

- aucune régression de précision SIRET exacte sur le dev historique ;
- aucune erreur AUTO sur l'évaluation aléatoire V4.4 ;
- plus de mauvais top-1 V4.4 rejetés que par l'accepteur gelé ;
- couverture historique au moins égale à celle du baseline moins 2 points ;
- artefacts, splits, features, seuil et hashes entièrement reproductibles.

Sinon :

- `PIVOT_FEATURES` si les labels sont suffisants mais la logistique ne sépare
  pas les scènes ;
- `PIVOT_SCENE_DRIFT` si trop de top-1 changent avec le retrieval V4.2 ;
- `STOP_RETRAIN` si le nouveau modèle réduit la sécurité.

Même en cas de `GO_SHADOW_V45`, la prochaine étape est un shadow frais sans
écriture CRM. Une certification à 99,8 % exige ensuite un volume indépendant
beaucoup plus grand.
