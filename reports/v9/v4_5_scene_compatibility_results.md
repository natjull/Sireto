# V4.5 — compatibilité des scènes difficiles

Date d'évaluation : 28 juillet 2026
Verdict : **`PIVOT_SCENE_DRIFT`**

## Résultat

Les 172 décisions V4.4 ont été rejouées avec le retrieval V4.2-B, limité à
100 candidats, puis avec le ranker V4.1 gelé. Le label d'une décision n'a été
transporté que lorsque le SIRET top-1 rejoué était strictement identique au
SIRET top-1 qui avait été adjudiqué.

| Mesure | Compatible | Total | Gate |
|---|---:|---:|---:|
| Toutes les scènes | 135 | 172 | informatif |
| Labels aléatoires fiables | 46 | 53 | 53 requis |
| Négatifs aléatoires | 2 | 6 | 6 requis |
| `TOP1_WRONG` ciblés | 16 | 37 | au moins 30 |
| `AMBIGUOUS` ciblés | 1 | 5 | au moins 4 |
| `TOP1_CORRECT` ciblés | 64 | 67 | au moins 55 |

Trente-sept scènes changent de top-1. Le seul critère franchi est celui des
bons top-1 ciblés. Quatre négatifs du tirage aléatoire dérivent et moins de la
moitié des erreurs ciblées restent liées à la même prédiction.

Le gate interdit donc tout entraînement de l'accepteur V4.5. Entraîner malgré
ce résultat reviendrait à appliquer des jugements portés sur une entreprise à
une autre entreprise proposée par le pipeline.

## Contrôles

- Les cinq artefacts canoniques V4.4 et leur gate
  `STOP_AUTONOMOUS_LABELING` sont vérifiés par hash et par comptes.
- Les 172 dossiers sont reliés à la file V4.3 gelée par
  `audit_case_id`, service, top-1 et strate d'échantillonnage.
- Le retrieval est exactement V4.2-B et ne dépasse jamais 100 candidats.
- Aucun positif n'est injecté et aucun candidat fermé n'est produit.
- Le ranker V4.1 est chargé après vérification de toute sa chaîne de hashes.
- La divergence entre sa signature d'entraînement A et la signature de
  replay B est publiée comme `EXPERIMENTAL_CROSS_RETRIEVAL`.
- Les 64 features candidat et les 80 features de scène empruntent les
  constructeurs communs au train et au service.
- Une scène `SCENE_DRIFT` ne porte ni cible accepteur ni label transporté.
- Aucun accepteur n'est chargé, aucun score ou seuil n'est calculé, aucun
  modèle n'est entraîné et le test final reste fermé.

La suite complète contient 296 tests passants.

## Artefacts immuables

- Scènes et candidats rejoués :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_5_hard_scenes/21f8c0b0b172b907`
- Gate de compatibilité :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/gates/v4_5_scene_compatibility/5c8b87fd8e226157`

L'artefact de scènes contient 172 lignes et 17 116 candidats, avec un maximum
de 100 candidats par dossier. Le gate séparé porte explicitement
`training_authorized=false`.

## Décision

V4.5 s'arrête avant entraînement avec `PIVOT_SCENE_DRIFT`. Ce résultat ne
remet pas en cause le retrieval V4.2 : il montre que les labels produits sur
les top-1 de l'ancienne pile A ne sont pas assez transportables vers les
top-1 de la pile B.

La prochaine expérience doit d'abord aligner la pile candidate et son ranker,
ou constituer des labels directement sur une pile préalablement gelée. Elle
ne peut pas recycler silencieusement les 37 décisions dont le top-1 a changé.
