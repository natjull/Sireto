# Contrat V4.8 — faisabilité de l'accepteur sur les labels courants

Statut : préenregistré après le gate V4.7
`GO_ACCEPTOR_FEASIBILITY`, avant tout réentraînement V4.8, tout calcul de
score V4.8 et toute ouverture ligne à ligne de la réserve aléatoire.

Identifiant : `V48_CURRENT_ACCEPTOR_FEASIBILITY`.

## 1. Question et portée

La seule question de V4.8 est :

> Les cas difficiles dont le top-1 courant a été validé par des preuves
> traçables permettent-ils à la régression logistique de mieux refuser les
> mauvais top-1, sans dégrader la précision ni la couverture historiques ?

V4.8 ne modifie ni le retrieval V4.2-B, ni le ranker A conservé par V4.6, ni
les 80 features de scène. Elle ne crée aucune nouvelle vérité SIRET pour le
ranker. Elle peut au plus autoriser un shadow frais, sans écriture CRM.

Le test final, V4-Fresh et tout ancien holdout consommé restent fermés. Un
succès V4.8 n'est ni un déploiement, ni une certification à 99,8 %.

## 2. Entrées épinglées

### 2.1 Labels courants V4.7

Répertoire canonique :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_7_current_adjudications/4cc5420fb5da0683`

| Artefact | SHA-256 |
|---|---|
| `manifest.json` | `634ad13c1c2eda0abd7c2921e94ebc1631c070cae8cb3b480514bbfba59e3a8c` |
| `current_labels.parquet` | `e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2` |
| `adjudications.parquet` | `c3ceb30e0186c58c6dc9957658935eb7d5a557e75cae4e83c6f6f2cabfb80b74` |

Le manifeste doit porter le verdict `GO_ACCEPTOR_FEASIBILITY` et reproduire :

| Population courante | Total | Fiables | Corrects | Mauvais | Ambigus |
|---|---:|---:|---:|---:|---:|
| ciblée | 115 | 98 | 70 | 27 | 1 |
| aléatoire | 57 | 52 | 46 | 5 | 1 |
| ensemble | 172 | 150 | 116 | 32 | 2 |

Les 22 `UNRESOLVED` n'ont jamais de cible. Leurs identités peuvent seulement
servir à réserver une composante anti-fuite.

La cible accepteur est :

- `TOP1_CORRECT` → `1` ;
- `TOP1_WRONG` ou `AMBIGUOUS` → `0` ;
- `UNRESOLVED` → aucune cible.

Les 52 cibles aléatoires ont déjà été publiées sous forme de comptes agrégés
par V4.7. Avant le gel du winner, il reste interdit de calculer ou consulter
leurs scores, décisions, bascules ou métriques ligne à ligne.

### 2.2 Socle historique V4.1

Répertoire :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155`

| Artefact | SHA-256 |
|---|---|
| `acceptor_scenes.parquet` | `8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9` |
| `split_assignments.parquet` | `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` |
| `acceptor/acceptor_model.joblib` | `16283b8aba5ed135846a74e9040c79e9f863f7e2bd658ca642ad444174b9a3fa` |
| `acceptor/metadata.json` | `73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c` |

Les identités SIREN nécessaires uniquement au graphe proviennent du dataset
source V4.1, lui aussi épinglé :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`

| Artefact | SHA-256 |
|---|---|
| `manifest.json` | `4023ad37156a8bdfbc1471bad4221885a151fcbb76c89d061b29601605893376` |
| `queries.parquet` | `6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0` |
| `labels.parquet` | `69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04` |

Le fichier contient exactement 5 547 scènes `fit` et 1 456 scènes `dev`.
Deux scènes fit portent `acceptor_eligible=False` et avaient été exclues de
l'apprentissage V4.1. Le socle effectif de tout réentraînement V4.8 contient
donc exactement 5 545 scènes fit et 1 456 scènes dev. Toutes les prédictions
ranker sont hors échantillon. L'ordre des 80 features est celui de
`metadata.json`, sans ajout, retrait ni réordonnancement.

L'accepteur gelé est une pipeline `StandardScaler` puis
`LogisticRegression(C=1, class_weight="balanced", max_iter=3000,
random_state=42)`. Son seuil est `0.46313316267954524`.

Sur les 1 456 scènes dev originales, il doit reproduire exactement 1 188
AUTO, 1 186 corrects et deux erreurs. Tout écart donne
`STOP_REPRODUCTION`.

## 3. Composants définitivement gelés

- retrieval V4.2-B, 100 candidats au maximum et aucun positif injecté ;
- ranker A conservé par le verdict V4.6 ;
- top-1 et 80 valeurs de scène archivés par V4.7 ;
- normalisation, snapshot SIRENE, ordre et définition des features ;
- vérifications déterministes avant accepteur ;
- définition de `AUTO_MATCH` et `REVIEW`.

V4.8 n'exécute aucun nouveau retrieval et ne réentraîne aucun ranker. Elle
apprend seulement des accepteurs logistiques sur les scènes déjà figées.

## 4. Partitions anti-fuite préparées avant les scores

La construction des partitions est une étape indépendante. Elle publie un
artefact immuable avant tout score V4.8.

Chaque `siren_component_id` historique reste un atome indivisible. Ses membres
SIREN sont reconstruits exclusivement avec `input_siren`/`input_siret` de
`queries.parquet` et `ground_truth_siren`/`ground_truth_siret` de
`labels.parquet`, exactement comme le split V4.1. Le SIREN prédit historique
ne crée pas d'arête supplémentaire.

Pour les dossiers V4.7, une composante relie exactement :

- le SIREN dérivé de `input_siret`, lorsqu'il existe ;
- le SIREN dérivé de `current_top1_siret` ;
- un éventuel SIREN exact explicitement validé dans l'adjudication.

Les SIREN simplement présents parmi les 100 candidats ne créent pas d'arête :
leur présence comme alternative locale ne transmet aucun label et les inclure
fusionnerait artificiellement des milliers d'entreprises d'une même commune.

Un SIREN est normalisé en conservant exactement neuf chiffres ; un SIRET
valide fournit ses neuf premiers chiffres. Une ligne sans aucun identifiant
valide reçoit un nœud de repli propre à sa provenance et à son `query_id`.
Les composantes V4.7 sont ensuite reliées aux atomes historiques lorsque ces
SIREN exacts sont identiques. L'identifiant global est constitué des nœuds de
la composante triés, joints par `|`, puis tronqué aux 16 premiers caractères
du SHA-256. Les identifiants bruts ne deviennent jamais des features.

Avant partitionnement, chacun des 98 ciblés et chacun des 52 random dits
fiables doit satisfaire simultanément :

- `current_training_eligible=True` et
  `current_evidence_validated=True` ;
- `current_acceptor_target` présent et égal à `1` pour `TOP1_CORRECT`, à
  `0` pour `TOP1_WRONG` ou `AMBIGUOUS` ;
- `current_top1_siret == replayed_top1_siret` ;
- les 80 features de `metadata.json` présentes, dans cet ordre lors de toute
  extraction, numériques et finies.

Il est interdit de substituer `scene_acceptor_target`, le top-1 gelé V4.4 ou
une ancienne adjudication à ces champs courants. Un échec donne
`STOP_INPUT_INTEGRITY`.

Priorité d'affectation :

1. `random_sealed` : toute composante touchant l'un des 57 cas aléatoires ;
2. `historical_dev` : toute autre composante touchant le dev V4.1 ;
3. `hard_oof` : toute autre composante contenant un cas ciblé fiable ;
4. `historical_fit` : le reste du fit V4.1.

Conséquences :

- toute ligne historique reliée à `random_sealed` est exclue du fit et du
  dev de développement ;
- toute ligne fit reliée au dev est exclue du fit ;
- un cas ciblé relié au dev est `hard_dev_locked`, évaluation seulement ;
- un cas ciblé relié au random est `hard_random_locked` et reste entièrement
  scellé ;
- aucun composant ne traverse deux partitions.

Les composantes `hard_oof` sont placées dans cinq folds déterministes. Elles
sont ordonnées par
`SHA-256("v4.8-hard-oof:42:" + component_id)`, puis placées gloutonnement
dans le fold ayant, dans cet ordre, le moins de mauvais, le moins de négatifs,
le moins de cas et le plus petit numéro.

Le manifeste de partition doit contenir les hashes d'entrée, l'affectation de
chaque requête et composante, les SIREN ayant créé les liens, les comptes par
provenance/cible, et les exclusions. Il doit aussi démontrer :

- 172 identifiants V4.7 uniques ;
- aucun identifiant de requête partagé avec les 7 003 scènes historiques ;
- les 57 cas aléatoires tous scellés ;
- aucune composante partagée entre fit, dev, hard OOF et random.

Un échec donne `STOP_INPUT_INTEGRITY` ou `STOP_LEAKAGE`. Aucun modèle n'est
alors appris.

## 5. Variantes autorisées

| Code | Fit | Poids total des cas ciblés |
|---|---|---:|
| `BASE_FROZEN` | modèle V4.1 non réentraîné | sans objet |
| `BASE_REFIT` | fit historique effectif | 0 |
| `HARD_W1` | fit historique + cas ciblés OOF | 1 |
| `HARD_W2` | fit historique + cas ciblés OOF | 2 |
| `HARD_W4` | fit historique + cas ciblés OOF | 4 |

Toutes les variantes réentraînées utilisent exactement la pipeline et les
paramètres V4.1. Toute scène historique a un poids total de `1`. Une scène
difficile a un poids total exactement égal à `1`, `2` ou `4` selon la
variante, transmis à `model__sample_weight`; il ne s'agit pas d'un poids
additionnel. Ces poids ne changent ni les cibles ni
`class_weight="balanced"`.

Pour chaque poids `HARD`, un modèle complet est appris sur le fit historique
effectif et tous les `hard_oof`. Il score le dev historique et choisit son
seuil. Ce modèle déjà évalué devient le modèle gelé si la variante gagne :
aucun refit post-sélection n'est autorisé.

En parallèle, cinq modèles OOF sont appris sur le fit historique autorisé et
quatre folds difficiles. Chacun score le fold difficile laissé dehors. Chaque
modèle OOF choisit son propre seuil sur le dev historique avec exactement ses
propres scores ; le seuil du modèle complet ne peut pas être appliqué aux
scores d'un autre modèle.

Les `hard_dev_locked` sont publiés descriptivement. Ils sont exclus de tous
les critères ciblés, de toute sélection de variante et de tout gate, car leur
composante participe au dev qui choisit le seuil.

Deux exécutions de `BASE_REFIT` doivent donner les mêmes scores à `1e-12`
près et les mêmes décisions. Sinon : `STOP_REPRODUCTION`.

Sont interdits : XGBoost accepteur, nouvelles features, calibrateur, recherche
d'hyperparamètres, réécriture du CRM et ajout de labels.

## 6. Seuils sans réserve aléatoire ni test

La règle de décision est `AUTO` si et seulement si `score >= seuil`. Pour
chaque variante réentraînée, les seuils candidats sont les scores distincts
du dev historique effectif, plus
`nextafter(max(score), +inf)` et `nextafter(min(score), -inf)`. Le seuil
maximise le nombre d'AUTO sous les contraintes :

- au moins 100 AUTO ;
- précision SIRET exacte observée ≥ 99,8 %, testée sans arrondi par
  `1000 * correct_auto >= 998 * auto_count` ;
- pas plus d'`AMBIGUOUS` historiques automatisés que `BASE_FROZEN` sur les
  mêmes lignes.

Égalités : meilleure précision, puis seuil le plus élevé.

`BASE_FROZEN` conserve son seuil historique. Aucun score ciblé, random ou test
ne choisit un seuil. Le seuil et les métriques sont choisis et mesurés sur le
même dev historique : ce gate est explicitement une étude de faisabilité
interne, pas une estimation hors échantillon de la sécurité.

## 7. Gate de développement et choix unique

Toutes les comparaisons sont appariées sur les mêmes lignes effectives. Une
variante `HARD` est admissible si elle respecte simultanément :

1. précision dev historique ≥ 99,8 % et non inférieure à
   `BASE_FROZEN` ;
2. couverture dev historique ≥ couverture de `BASE_FROZEN` moins
   2 points ;
3. sur les seuls `hard_oof`, au moins quatre `TOP1_WRONG` ciblés
   supplémentaires rejetés hors échantillon par rapport à `BASE_REFIT` ;
4. sur les seuls `hard_oof`, taux d'acceptation des `TOP1_CORRECT` ciblés
   supérieur ou égal au taux `BASE_REFIT` moins 5 points ;
5. aucun `AMBIGUOUS` ciblé supplémentaire automatisé par rapport à
   `BASE_REFIT`, toujours sur les seuls `hard_oof`.

Sélection parmi les variantes admissibles :

1. plus de `TOP1_WRONG` ciblés rejetés ;
2. plus de corrects ciblés acceptés ;
3. plus grande couverture dev historique ;
4. poids le plus faible.

Les nombres bruts, intervalles de Wilson à 95 %, courbes
risque-couverture et bascules appariées sont publiés. Le winner, son seuil,
ses scores de développement, son modèle et ses hashes sont ensuite gelés.
Les résultats sont aussi séparés par strate V4.7 et origine du label. Les
coefficients, l'intercept, les versions `numpy`/`scikit-learn` et le hash des
scores sont publiés.

Si aucune variante n'est admissible :

- `PIVOT_FEATURES` si la sécurité historique tient mais les 80 features ne
  séparent pas quatre erreurs supplémentaires ;
- `STOP_RETRAIN` si le gain exige une perte de sécurité ou de couverture.

La réserve aléatoire reste fermée dans les deux cas.

## 8. Ouverture unique de la réserve aléatoire

Après gel du winner, un marqueur irréversible contenant les hashes du modèle,
du seuil et de la partition est écrit avant le premier calcul de score
aléatoire. Une seule ouverture est autorisée.

Le gate random exige :

- zéro AUTO parmi les cinq `TOP1_WRONG` et l'unique `AMBIGUOUS` ;
- zéro erreur parmi toutes les décisions AUTO ;
- au moins 20 AUTO parmi les 46 `TOP1_CORRECT` ;
- au plus un correct AUTO de moins que `BASE_FROZEN` sur ces 46 cas ;
- aucune scène du random utilisée au fit, au choix du poids ou du seuil.

Ces six négatifs sont un contrôle observé de faisabilité, pas une preuve
statistique de 99,8 %.

## 9. Verdicts et suite autorisée

`GO_FRESH_SHADOW_V48` exige tous les contrôles d'intégrité, un winner
admissible au développement et le gate random complet. La seule suite
autorisée est alors un shadow réellement frais, sans écriture CRM, avec
modèle et seuil figés.

Autres verdicts :

- `PIVOT_FEATURES` : les labels sont exploitables mais les 80 features ne
  séparent pas assez les erreurs ;
- `STOP_RETRAIN` : sécurité ou couverture dégradée ;
- `STOP_INPUT_INTEGRITY` : entrée ou compte divergent ;
- `STOP_LEAKAGE` : composante partagée ;
- `STOP_REPRODUCTION` : baseline ou réentraînement non déterministe ;
- `STOP_RANDOM_INTEGRITY` : ouverture random non unique ou non reproductible.

Le test final ne peut être ouvert par aucun verdict V4.8. La North Star reste
la couverture AUTO à précision SIRET exacte ≥ 99,8 %, mesurée ensuite sur un
flux indépendant suffisamment grand.

## 10. Livrables obligatoires

- manifeste immuable des partitions, produit avant tout score ;
- prédictions OOF appariées de développement ;
- métriques historiques et difficiles de chaque variante ;
- modèle, seuil et manifeste de gel du winner ;
- marqueur d'ouverture random ;
- décisions random ligne à ligne et agrégées ;
- rapport final et verdict explicite ;
- milestone isolé cité dans `handover.md`.
