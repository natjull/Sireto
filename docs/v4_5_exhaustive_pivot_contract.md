# Addendum V4.5 — pivot exhaustif vers une étude de faisabilité de l'accepteur

Statut : préenregistré après le verdict exhaustif V4.4 et avant tout
réentraînement, calcul de score V4.5 ou choix de seuil.

Identifiant de l'expérience :
`V45_EXHAUSTIVE_PIVOT_FEASIBILITY`.

## 1. Décision scientifique et portée

V4.4 est définitivement close avec le verdict
`STOP_AUTONOMOUS_LABELING`. Sa population de 172 `AUTO_MATCH` a été épuisée :
114 `TOP1_CORRECT`, 42 `TOP1_WRONG`, six `AMBIGUOUS` et dix `UNRESOLVED`.
Le minimum préenregistré de 50 erreurs n'a pas été atteint.

Le présent addendum :

- ne remplace pas ce verdict ;
- ne transforme pas 42 erreurs en un succès au gate V4.4 ;
- n'abaisse pas rétroactivement le minimum de 50 ;
- n'autorise pas le réentraînement prévu par `GO_RETRAIN_AUTO` dans le
  contrat V4.4 ;
- ouvre une nouvelle étude de faisabilité, plus limitée, avec ses propres
  hypothèses et ses propres critères d'arrêt.

La question testée est uniquement la suivante :

> Les 162 labels fiables disponibles permettent-ils à une régression
> logistique de mieux reconnaître les mauvais top-1 difficiles, sans réduire
> la sécurité ni la couverture sur les scènes historiques V4.1 ?

Un résultat positif autorise au plus un nouveau shadow sans écriture CRM. Il
n'autorise ni déploiement, ni certification, ni revendication de précision à
99,8 %.

## 2. Relation avec les contrats antérieurs

Le contrat `docs/v4_5_acceptor_retraining_contract.md` reste la référence pour
les composants gelés, la liaison d'un label à un top-1, les cibles accepteur
et l'interdiction d'utiliser le test final. Sa version de référence a pour
SHA-256
`227b6ae55bda5192ffb47355d62eef7336809c34da700687b9000498fd4550c2`.

Cet addendum remplace seulement sa condition d'activation
`GO_RETRAIN_AUTO` par une expérience distincte rendue nécessaire par
l'épuisement de la population V4.4. Il précise en outre :

- la séparation du tirage aléatoire avant toute modélisation ;
- le graphe anti-fuite global ;
- l'évaluation hors échantillon des cas difficiles ciblés ;
- l'ordre exact de choix des variantes et des seuils ;
- les critères quantitatifs de `GO`, `PIVOT` et `STOP`.

En cas de contradiction, cet addendum prévaut uniquement pour l'expérience
`V45_EXHAUSTIVE_PIVOT_FEASIBILITY`. Il ne modifie aucun résultat V4.1, V4.2,
V4.4 ou V4-Fresh.

## 3. Entrées épinglées

### 3.1 Gate V4.4

Source :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_gate/9fb43b4f7bb0919a/gate_report.json`

SHA-256 :
`50227ca89842ebd3f9e4e0d9b8fa9257353203c8256c49df50893332b13a3990`

Assertions obligatoires :

| Population V4.4 | Nombre |
|---|---:|
| Dossiers totaux | 172 |
| Labels fiables accepteur | 162 |
| `TOP1_CORRECT` | 114 |
| `TOP1_WRONG` | 42 |
| `AMBIGUOUS` | 6 |
| `UNRESOLVED` exclus | 10 |
| Cas du tirage aléatoire | 57 |
| Cas aléatoires fiables | 53 |

Les 53 cas aléatoires fiables se composent de 47 `TOP1_CORRECT`, cinq
`TOP1_WRONG` et un `AMBIGUOUS`. Les quatre cas aléatoires `UNRESOLVED`
n'ont pas de cible, mais leur composante anti-fuite reste réservée.

Les 109 labels fiables hors tirage aléatoire se composent de 67
`TOP1_CORRECT`, 37 `TOP1_WRONG` et cinq `AMBIGUOUS`.

Tout écart avec ces nombres ou avec le verdict
`STOP_AUTONOMOUS_LABELING` arrête l'expérience avec
`STOP_INPUT_INTEGRITY`. Il est interdit de reconstruire un gate V4.4 plus
favorable.

### 3.2 Socle de scènes V4.1

Source :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155/acceptor_scenes.parquet`

SHA-256 :
`8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9`

Le socle contient exactement 7 003 scènes produites par des prédictions
ranker hors échantillon :

| Split historique | Scènes | `MATCH_EXACT` | `AMBIGUOUS` |
|---|---:|---:|---:|
| `fit` | 5 547 | 4 666 | 881 |
| `dev` | 1 456 | 1 217 | 239 |
| Total | 7 003 | 5 883 | 1 120 |

Les 80 features et leur ordre restent strictement ceux de V4.1. Aucune
feature, interaction ou règle métier n'est ajoutée ou retirée dans cette
expérience.

Autres composants épinglés :

| Artefact | SHA-256 |
|---|---|
| `split_assignments.parquet` | `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` |
| accepteur V4.1 gelé | `16283b8aba5ed135846a74e9040c79e9f863f7e2bd658ca642ad444174b9a3fa` |
| métadonnées accepteur V4.1 | `73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c` |
| ranker V4.1 gelé | `720b0d2d44971477198112f03606eb303bc2f61c06bfdaf48b576b6df4551080` |
| métadonnées ranker V4.1 | `5f5edd2a342fd4e8e2e3754bc3bca0f24b8dd93aec7f899f9d727cb54195757b` |

Le seuil historique de référence est `0.46313316267954524`. Sur le dev V4.1
complet, il doit reproduire exactement 1 188 `AUTO_MATCH`, dont 1 186
corrects et deux erreurs.

## 4. Composants gelés

Restent inchangés :

- retrieval V4.2, variante B, 100 candidats maximum ;
- ranker V4.1 et son ordre de features ;
- normalisation, features candidat et 80 features de scène ;
- snapshot SIRENE, modèles sémantiques et dépendances épinglés ;
- vérifications déterministes précédant l'accepteur ;
- définition de `AUTO_MATCH` et `REVIEW` ;
- tous les anciens tests et holdouts.

Le seul modèle susceptible d'être appris est une régression logistique
standardisée de même implémentation que V4.1. Aucun ranker, XGBoost,
calibrateur isotonic, cross-encoder ou nouveau traitement du CRM n'entre dans
cette étude.

## 5. Liaison des labels V4.4 aux scènes

Un label V4.4 juge le top-1 figé du shadow V4.1. Il ne juge pas
automatiquement une autre prédiction.

Pour chacun des 172 dossiers :

1. rejouer le retrieval V4.2 et le ranker V4.1 gelé, sans positif injecté ;
2. reconstruire les 80 features avec le chemin train/serve commun ;
3. comparer le top-1 rejoué au `frozen_top1_siret` adjudiqué ;
4. marquer `SCENE_COMPATIBLE` si les deux SIRET sont identiques ;
5. sinon marquer `SCENE_DRIFT` et ne jamais transporter le label vers la
   nouvelle prédiction.

Les cibles sont :

- `TOP1_CORRECT` → `1`, accepter ;
- `TOP1_WRONG` → `0`, rejeter ;
- `AMBIGUOUS` → `0`, rejeter ;
- `UNRESOLVED` → aucune cible et aucune utilisation modèle.

Un `TOP1_WRONG` sans SIRET alternatif reste une cible accepteur valide. Il ne
devient jamais une cible ranker.

Le socle V4.1 conserve ses scènes OOF archivées. Les scènes V4.4 sont
reconstruites sous V4.2. Ce mélange est volontairement visible grâce aux
features de provenance et constitue une limite de l'étude. L'effet des
nouveaux labels est donc mesuré contre un réentraînement `BASE_REFIT` soumis
au même protocole, pas seulement contre le binaire historique.

### Gate de compatibilité des scènes

Avant tout entraînement, il faut conserver après `SCENE_DRIFT` :

- les 53 labels aléatoires fiables, sans exception ;
- donc les six négatifs aléatoires ;
- au moins 30 des 37 `TOP1_WRONG` ciblés ;
- au moins quatre des cinq `AMBIGUOUS` ciblés ;
- au moins 55 des 67 `TOP1_CORRECT` ciblés.

Si un négatif aléatoire dérive, ou si un autre minimum échoue, le verdict est
`PIVOT_SCENE_DRIFT`. Aucun seuil n'est calculé.

## 6. Graphe anti-fuite et partitions

Le graphe est construit avant tout score accepteur. Une requête est reliée à
tous les SIREN suivants :

- SIREN d'entrée ;
- SIREN prédit ;
- SIREN vérité lorsqu'il existe ;
- tous les SIREN présents dans son pool candidat figé.

Les composantes connexes sont indivisibles. Les identifiants SIRET/SIREN bruts
ne sont jamais des features.

L'affectation applique cette priorité :

1. **`random_sealed`** : toute composante contenant l'un des 57 dossiers du
   tirage aléatoire V4.4 ;
2. **`historical_dev`** : toute autre composante contenant une scène du dev
   V4.1 ;
3. **`hard_oof`** : composantes contenant un label V4.4 ciblé compatible ;
4. **`historical_fit`** : toutes les composantes restantes du fit V4.1.

Conséquences :

- aucune ligne d'une composante `random_sealed` n'entre dans le fit, le dev,
  le choix des variantes ou le choix du seuil ;
- une scène fit historique reliée au dev ou au random est retirée du fit ;
- un cas difficile ciblé relié au dev est évaluation seulement et n'entre
  jamais dans le fit ; il reçoit le tag `hard_dev_locked` ;
- aucun composant ne traverse deux partitions.

Les 109 cas ciblés compatibles et non absorbés par `historical_dev` sont
affectés à cinq folds déterministes de composantes. L'ordre est :

`SHA-256("v4.5-hard-oof:42:" + component_id)`.

Un placement glouton, exécuté dans cet ordre, choisit le fold ayant
successivement le moins de `TOP1_WRONG`, le moins de négatifs totaux, le moins
de cas, puis le plus petit numéro. Cette stratification utilise uniquement les
labels et groupes déjà gelés, jamais un score modèle.

Un manifeste publié avant entraînement contient :

- l'affectation de chaque requête et chaque composante ;
- les SIREN ayant créé chaque arête ;
- les comptes par provenance et par cible ;
- les exclusions `SCENE_DRIFT` ;
- les hashes de toutes les entrées.

Toute fuite détectée donne `STOP_LEAKAGE`.

### Gate de reproduction

Avant les variantes :

- l'accepteur V4.1 gelé doit reproduire sur les 1 456 scènes dev originales
  le seuil `0.46313316267954524`, 1 188 AUTO, 1 186 succès et deux erreurs ;
- le constructeur train et le constructeur serve doivent produire les mêmes
  80 valeurs, dans le même ordre, sur un échantillon hashé de scènes ;
- deux exécutions de `BASE_REFIT` avec les mêmes entrées doivent produire les
  mêmes décisions et des coefficients égaux à `1e-12` près.

Un échec donne `STOP_REPRODUCTION`, avant toute consultation des résultats
des variantes difficiles.

## 7. Variantes appariées

Cinq variantes seulement sont autorisées :

| Code | Entraînement | Poids d'une scène V4.4 ciblée |
|---|---|---:|
| `BASE_FROZEN` | accepteur V4.1 non réentraîné | sans objet |
| `BASE_REFIT` | socle V4.1 autorisé uniquement | sans objet |
| `HARD_W1` | socle V4.1 + cas ciblés | 1 |
| `HARD_W2` | socle V4.1 + cas ciblés | 2 |
| `HARD_W4` | socle V4.1 + cas ciblés | 4 |

Toutes les scènes historiques ont un poids 1. `TOP1_WRONG` et `AMBIGUOUS`
partagent la cible 0, sans poids de classe additionnel. Les paramètres,
standardisation, seed et solveur sont ceux de V4.1 et sont enregistrés dans le
manifeste.

Pour `HARD_W1`, `HARD_W2` et `HARD_W4`, les métriques difficiles sont
produites par cinq entraînements groupés : chaque composante difficile est
scorée uniquement par un modèle qui ne l'a jamais vue. Aucun score in-sample
V4.4 ne peut servir au choix.

Les cas `hard_dev_locked` sont scorés par un modèle entraîné sur
`historical_fit` et sur toutes les composantes `hard_oof`, qui leur sont
disjointes. Les métriques difficiles agrègent ces prédictions aux cinq folds
OOF. Chaque cas ciblé doit ainsi disposer d'une et une seule prédiction hors
échantillon.

Le modèle final d'une variante n'est appris sur tous ses cas ciblés qu'après
le choix de la variante. Il ne voit toujours aucun cas `random_sealed`.

## 8. Choix des seuils sans random ni test

Pour chaque modèle autorisé, les seuils candidats sont les scores distincts du
dev historique effectif, complétés par les deux seuils extrêmes. Le seuil
retenu maximise la couverture sous toutes les contraintes suivantes :

- au moins 100 décisions `AUTO_MATCH` ;
- précision SIRET exacte observée ≥ 99,8 % ;
- aucune `AMBIGUOUS` automatisée au-delà de ce qu'automatise
  `BASE_FROZEN` sur le même dev effectif.

En cas d'égalité : précision la plus haute, puis seuil le plus élevé.

Le tirage aléatoire V4.4, l'ancien test, V4-Fresh et le shadow complet ne sont
jamais consultés pour choisir un seuil.

## 9. Gate de développement

Les métriques sont toujours calculées sur les mêmes lignes effectives pour
`BASE_FROZEN`, `BASE_REFIT` et chaque variante `HARD`.

Une variante `HARD` est admissible si elle satisfait simultanément :

1. précision observée sur dev historique ≥ 99,8 % ;
2. précision sur ce dev non inférieure à `BASE_FROZEN` ;
3. couverture dev au moins égale à celle de `BASE_FROZEN` moins 2 points de
   pourcentage ;
4. au moins quatre `TOP1_WRONG` ciblés supplémentaires rejetés hors
   échantillon par
   rapport à `BASE_REFIT` sur exactement les mêmes cas ;
5. taux d'acceptation hors échantillon des `TOP1_CORRECT` ciblés au plus cinq
   points sous celui de `BASE_REFIT` ;
6. aucun `AMBIGUOUS` ciblé supplémentaire automatisé par rapport à
   `BASE_REFIT`.

Les nombres bruts, écarts appariés, intervalles de Wilson à 95 % et matrice
des bascules `AUTO→REVIEW` / `REVIEW→AUTO` sont publiés. Les critères en
nombre de cas, et non les seuls pourcentages, font foi.

Sélection parmi les variantes admissibles :

1. plus grand nombre de `TOP1_WRONG` ciblés rejetés en OOF ;
2. plus grande couverture sur dev historique ;
3. poids le plus faible.

Si aucune variante n'améliore d'au moins quatre erreurs difficiles sans
régression de sécurité, le verdict est :

- `PIVOT_FEATURES` si toutes les variantes restent sûres mais ne séparent pas
  mieux les erreurs ;
- `STOP_RETRAIN` si une amélioration apparente exige une baisse de précision,
  une perte de couverture supérieure à deux points ou plus d'ambiguës
  automatisées.

Le modèle choisi, son seuil, ses dépendances et toutes ses prédictions de
développement sont ensuite hashés et gelés.

## 10. Ouverture unique du tirage aléatoire V4.4

Après gel, un marqueur irréversible est écrit avant toute lecture des cibles
`random_sealed`. Une seule évaluation est autorisée.

Le gate random exige simultanément :

- zéro `AUTO_MATCH` parmi les cinq `TOP1_WRONG` et l'unique `AMBIGUOUS` ;
- zéro erreur parmi toutes les décisions AUTO aléatoires ;
- au moins 20 décisions AUTO parmi les 47 `TOP1_CORRECT` ;
- au plus une décision correcte AUTO de moins que `BASE_FROZEN` sur les 47
  mêmes cas ;
- aucune scène non compatible, non hashée ou issue d'une composante
  d'entraînement.

Cette population ne compte que six négatifs. Zéro erreur est donc un contrôle
de sécurité observé, pas une preuve statistique de 99,8 %.

Une erreur AUTO aléatoire ou un effondrement de couverture donne
`STOP_RETRAIN`. Un problème d'intégrité donne `STOP_RANDOM_INTEGRITY` et ne
peut pas être corrigé par une seconde ouverture.

## 11. Verdicts

`GO_SHADOW_V45` exige :

- tous les contrôles d'entrée, de scène, de groupe et de reproduction ;
- une variante admissible au gate de développement ;
- le gate random complet ;
- un bundle et un manifeste intégralement reproductibles.

Le seul travail autorisé après `GO_SHADOW_V45` est un shadow frais, sans
écriture CRM et sans modification du modèle ou du seuil.

Les autres verdicts sont :

- `PIVOT_SCENE_DRIFT` : les labels ne se transportent pas assez vers le
  pipeline V4.2 ;
- `PIVOT_FEATURES` : la logistique à 80 features ne gagne pas hors
  échantillon sur les erreurs difficiles ;
- `STOP_RETRAIN` : la sécurité ou la couverture se dégrade ;
- `STOP_INPUT_INTEGRITY` : les artefacts ou comptes épinglés divergent ;
- `STOP_LEAKAGE` : une composante traverse plusieurs partitions ;
- `STOP_REPRODUCTION` : le baseline ou le calcul des features n'est pas
  reproductible ;
- `STOP_RANDOM_INTEGRITY` : l'ouverture unique n'est pas reproductible.

Aucun de ces verdicts ne modifie le `STOP_AUTONOMOUS_LABELING` V4.4.

## 12. Test final et interdictions

Le test final historique reste fermé pendant toute l'expérience. Le holdout
V4-Fresh déjà consommé ne peut jamais être réutilisé. Le tirage aléatoire
V4.4 est un holdout de faisabilité interne, pas un nouveau test de
certification.

Même après `GO_SHADOW_V45`, un éventuel test final exige un nouveau contrat,
un snapshot CRM réellement nouveau, un gel préalable et des labels
indépendants. Aucune précision de production n'est revendiquée avec les 53 cas
aléatoires.

Sont interdits :

- ouvrir les 370 `REVIEW` V4.3 pour combler les huit erreurs manquantes ;
- relabelliser un cas V4.4 après observation d'un score V4.5 ;
- utiliser les quatre random `UNRESOLVED` comme négatifs ;
- ajuster poids, features ou seuil après l'ouverture random ;
- entraîner sur une prédiction différente de celle adjudiquée ;
- injecter un positif, dépasser 100 candidats ou entraîner un autre modèle ;
- lire un test fermé, modifier le CRM ou présenter l'expérience comme une
  certification.

## 13. Livrables requis avant toute conclusion

- manifeste d'entrée avec tous les hashes ;
- table `scene_compatibility.parquet` pour les 172 dossiers ;
- graphe et affectations de composantes anti-fuite ;
- prédictions OOF appariées des variantes difficiles ;
- courbes risque-couverture et métriques brutes du dev ;
- manifeste de gel du winner et marqueur d'ouverture random ;
- résultats random ligne à ligne et agrégés ;
- verdict explicite parmi ceux de la section 11.
