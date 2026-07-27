# Contrat V4.6 — ablation du ranker aligné sur le retrieval V4.2-B

Statut : préenregistré avant construction du dataset V4.2-B des 7 003
requêtes, avant réentraînement du ranker et avant toute nouvelle
adjudication.

Identifiant de l'expérience :
`V46_ALIGNED_RANKER_V42B`.

## 1. Question scientifique

Le replay V4.5 a appliqué le ranker V4.1 à des pools produits par un retrieval
différent de celui de son entraînement :

- retrieval d'entraînement du ranker A :
  `189aeae6efead3595a586413871fbc388fde900d3b243d338b70d6a9de5a9db3` ;
- retrieval V4.2-B rejoué :
  `021f928e21e2360186217862b4310be90fe0f705c1bfbf43b39a8b41e644e40c`.

Le replay a produit 37 changements de top-1 sur 172 dossiers. Ce constat ne
prouve pas qu'un nouveau ranker serait meilleur. Il démontre seulement une
rupture de distribution entre ses pools d'entraînement et ses pools
d'inférence.

La V4.6 teste une seule hypothèse :

> À population, labels, features, hyperparamètres et retrieval V4.2-B
> identiques, un ranker réentraîné sur les pools V4.2-B classe-t-il mieux le
> SIRET exact qu'un ranker A gelé, entraîné sur les anciens pools puis appliqué
> à V4.2-B ?

Il s'agit d'une ablation de ranker. Aucun accepteur, seuil AUTO, label V4.4 ou
résultat de shadow ne participe à la sélection.

## 2. Décisions préalables et limites

La V4.5 conserve son verdict `PIVOT_SCENE_DRIFT`. La V4.6 :

- ne réinterprète pas les labels V4.4 ;
- ne transporte aucun label vers un nouveau top-1 ;
- ne modifie pas le retrieval V4.2-B ;
- n'ouvre ni test final, ni V4-Fresh, ni ancien holdout consommé ;
- n'autorise aucune nouvelle adjudication avant son verdict ;
- n'autorise ni déploiement, ni écriture CRM, ni revendication de précision
  production.

Un `GO_ALIGN_RANKER` autorise uniquement le gel du ranker B et la création
ultérieure d'un **nouveau** shadow dont les top-1 devront être adjudiqués.

## 3. Population autorisée

La population est constituée des mêmes 7 003 requêtes historiques V4.1 :

| Split historique | Requêtes |
|---|---:|
| `fit` | 5 547 |
| `dev` | 1 456 |
| Total | 7 003 |

Source canonique :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`

Manifest :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155/manifest.json`

Entrées épinglées :

| Artefact | SHA-256 |
|---|---|
| `queries.parquet` | `6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0` |
| `labels.parquet` | `69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04` |
| anciens `candidates.parquet` | `34b526fe49e3451c05248294305e4a8d6ccf4db92277eb36dc03cc6231420c67` |
| affectations V4.1 | `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` |
| snapshot établissement | `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` |

Les anciens candidats ne servent pas à entraîner le ranker B. Leur hash est
conservé uniquement pour identifier le contexte d'entraînement du ranker A.

Les tables canoniques contiennent déjà 1 319 requêtes issues des anciens
segments `fit_addition` et `dev_new`. Elles font partie des 7 003 lignes
gelées et sont donc conservées telles quelles : les retirer changerait la
population après observation. En revanche, le holdout associé et tout autre
jeu V4-Fresh extérieur à ces deux tables restent interdits.

Sont formellement interdits dans le dataset V4.6 :

- les 172 dossiers ou labels V4.4 ;
- l'indicateur d'appartenance au tirage random ;
- toute preuve ou décision d'adjudication V4.4 ;
- le test final, tout holdout et toute nouvelle source V4-Fresh extérieure
  aux 7 003 lignes déjà canoniques ;
- les cas REVIEW ajoutés après la constitution V4.1 ;
- tout rang, score ou décision d'un accepteur.

Le loader n'accepte comme population que les deux tables canoniques dont les
hashes figurent ci-dessus. Il refuse toute source supplémentaire ainsi que
toute colonne de rôle contenant `test`, `holdout`, `v4_4`, `random` ou
`adjudication`. Il ne recherche pas ces chaînes dans les champs métier libres
(nom, commune ou identifiant), où elles peuvent apparaître légitimement.

## 4. Construction unique des pools V4.2-B

Les 7 003 requêtes sont rejouées une seule fois avec la variante V4.2-B
gelée, sans utiliser leur vérité :

- signature retrieval :
  `021f928e21e2360186217862b4310be90fe0f705c1bfbf43b39a8b41e644e40c` ;
- manifeste V4.2 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_2_retrieval_integrity_7c4b957/manifest.json` ;
- SHA-256 du manifeste V4.2 :
  `63b52c3a1466070410881b0ea61b833ff5d413262239920abbc6b04e3f153f54` ;
- plafond absolu : 100 candidats par requête ;
- état administratif issu du snapshot SIRENE complet ;
- `positive_injection=false`.

La vérité n'est jointe qu'après fermeture du pool afin de créer la cible
d'entraînement. Si le bon SIRET est absent, il reste absent : aucune ligne
n'est ajoutée et la requête compte comme un échec end-to-end.

Avant tout entraînement, publier un dataset V4.6 immuable contenant :

- `queries.parquet` ;
- `labels.parquet` ;
- `candidates_v42b.parquet` ;
- `split_assignments.parquet` ;
- `manifest.json`.

Le manifeste doit contenir les chemins, tailles, nombres de lignes et
SHA-256 de chaque entrée et sortie, le hash du snapshot, la configuration
retrieval complète, l'ordre des features, la seed et
`positive_injection=false`. Tout changement crée un nouveau build ; un
artefact existant n'est jamais écrasé.

### Gate d'intégrité des pools

L'expérience s'arrête avec `STOP_INPUT_INTEGRITY` si l'une des conditions
suivantes échoue :

- exactement 7 003 requêtes uniques et une affectation par requête ;
- mêmes identifiants, labels et frontières historiques que le socle V4.1 ;
- zéro candidat au-delà du rang 100 ;
- zéro doublon SIRET dans un pool ;
- zéro candidat déclaré fermé par la source autoritaire ;
- aucune vérité injectée ;
- aucune donnée interdite chargée ;
- deux constructions indépendantes produisent les mêmes pools ordonnés et le
  même hash de contenu.

Le Recall@100 V4.2-B est publié sur `fit` et `dev`, mais ne peut entraîner
aucune modification du retrieval dans cette expérience.

## 5. Frontières anti-fuite SIREN

Les frontières historiques `fit` et `dev` sont conservées. Elles doivent être
réauditées avant apprentissage avec un graphe requête–SIREN construit
exclusivement à partir :

- du SIREN d'entrée lorsqu'il existe ;
- du SIREN de vérité pour les labels `MATCH_EXACT`.

Une composante connexe est indivisible. Aucun SIREN d'entrée ou de vérité ne
peut apparaître à la fois dans `fit` et `dev`, ni dans deux folds OOF.

Les SIREN des candidats négatifs ne créent pas d'arête : ils ne portent
aucune cible d'identité, peuvent être communs à plusieurs communes et leur
inclusion pourrait relier artificiellement une grande partie du corpus. En
contrepartie :

- aucun SIRET ou SIREN brut n'est une feature ;
- les features ne peuvent encoder un identifiant ;
- les seules cibles positives proviennent du SIRET exact gelé ;
- toutes les lignes d'une requête restent dans la même frontière.

Si l'audit découvre une composante traversant `fit` et `dev`, le split n'est
pas réparé après lecture des performances. Le verdict est `STOP_LEAKAGE`.

Les cinq folds V4.1 sont conservés s'ils passent cet audit. Sinon
l'expérience s'arrête : aucun nouveau split plus favorable n'est recherché.

## 6. Deux variantes et aucune autre

### A — ranker historique gelé appliqué à B

- modèle :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155/ranker/ranker.json` ;
- SHA-256 :
  `720b0d2d44971477198112f03606eb303bc2f61c06bfdaf48b576b6df4551080` ;
- métadonnées SHA-256 :
  `5f5edd2a342fd4e8e2e3754bc3bca0f24b8dd93aec7f899f9d727cb54195757b`.

Le modèle A est seulement scoré sur les pools V4.2-B. Il n'est ni modifié ni
réentraîné.

### B — ranker aligné entraîné sur B

Le ranker B utilise exactement :

- le même XGBoost `rank:pairwise` ;
- le même ordre des 64 features candidat ;
- les mêmes normalisations ;
- les mêmes hyperparamètres ;
- la même seed racine `42`.

Hyperparamètres gelés :

```text
objective=rank:pairwise
eval_metric=ndcg@1
n_estimators=800
max_depth=6
learning_rate=0.035
subsample=0.85
colsample_bytree=0.85
min_child_weight=3
reg_lambda=5.0
n_jobs=-1
random_state=42
```

Aucun tuning, early stopping, nouvelle feature ou variante d'hyperparamètres
n'est autorisé après lecture des résultats.

Pour le `fit`, cinq modèles apprennent sur quatre folds et scorent le
cinquième. Chaque requête fit reçoit exactement une prédiction OOF. Un modèle
final apprend ensuite sur tout le `fit` et score le `dev`, qui reste
strictement hors échantillon.

Une requête `MATCH_EXACT` dont la vérité est absente du pool ne contient
aucune paire positive exploitable : elle est exclue du fit ranker, sans
réinjection, mais demeure obligatoirement dans toutes les métriques
end-to-end. Les nombres exclus sont publiés par split et par fold.

## 7. Comparaison équitable

La sélection entre A et B utilise exclusivement les 1 217 labels
`MATCH_EXACT` du `dev` historique :

- A : modèle historique entraîné sur l'ancien fit, appliqué au dev V4.2-B ;
- B : modèle final entraîné sur le fit V4.2-B, appliqué au même dev V4.2-B.

Les pools, candidats, features et tie-breaks sont donc identiques. La seule
variable est l'alignement de l'entraînement du ranker sur le retrieval B.

Les prédictions OOF du fit produites par B sont nécessaires à une éventuelle
étape aval, mais ne servent pas à sélectionner B contre A : A ne dispose pas
d'une prédiction OOF comparable sur les nouveaux pools.

### Métrique nord

La métrique nord est le `Hit@1 SIRET exact` end-to-end :

```text
nombre de requêtes dev MATCH_EXACT dont le top-1 est le SIRET vérité
divisé par 1 217
```

Une vérité absente du pool, une absence de top-1 et un SIRET mal classé sont
tous des échecs. Le SIREN n'est qu'une métrique secondaire et ne peut
compenser une erreur de SIRET.

Le tie-break après égalité de score est unique :

1. score ranker décroissant ;
2. rang retrieval croissant ;
3. SIRET croissant.

Publier pour A et B :

- succès, erreurs et Hit@1 SIRET exact ;
- succès et Hit@1 SIREN ;
- tableau apparié `A correct/B correct`, `A correct/B faux`,
  `A faux/B correct`, `A faux/B faux` ;
- différence absolue en points ;
- intervalle bootstrap apparié à 95 %, seed `42`, 10 000 réplications ;
- test exact de McNemar bilatéral ;
- résultats par segment historique disponible ;
- listes hashées des corrections et régressions ;
- latences de scoring p50/p95 et temps d'entraînement sur le Mac.

Les segments préenregistrés sont exactement `input_siret_state` et
`source_segment`, avec une ligne par valeur observée et une ligne globale.
Aucun autre découpage n'entre dans un gate.

L'intervalle bootstrap rééchantillonne avec remise les 1 217 indices
appariés, calcule à chaque réplication la moyenne de `hit_B - hit_A`, utilise
`numpy.random.default_rng(42)`, 10 000 réplications et les percentiles 2,5 %
et 97,5 %. Le test de McNemar est le test binomial exact bilatéral sur les
seules paires discordantes, avec probabilité nulle `0,5`.

La latence est mesurée requête par requête sur le dev complet : un warm-up
hors mesure, puis trois passages dans le même ordre de requêtes pour chaque
modèle. Les p50 et p95 portent sur les 4 368 durées individuelles par
variante, hors chargement du modèle et lecture parquet.

## 8. Gates de promotion

Le verdict est `GO_ALIGN_RANKER` uniquement si toutes les conditions
suivantes sont satisfaites :

1. `Hit@1 SIRET exact B >= 99,0 %` sur les 1 217 cas dev exacts ;
2. gain absolu de B sur A d'au moins `0,25` point, soit au moins quatre
   succès nets supplémentaires sur 1 217 cas ;
3. borne basse de l'intervalle bootstrap apparié à 95 % strictement
   supérieure à zéro ;
4. test exact de McNemar bilatéral avec `p < 0,05` ;
5. aucune régression SIREN globale ;
6. pour chaque segment préexistant contenant au moins 100 cas exacts, aucune
   régression SIRET supérieure à `1,0` point ;
7. aucune famille de segment, quelle que soit sa taille, ne perd plus de deux
   succès nets sans publication et examen causal des dossiers ;
8. les deux exécutions de B avec les mêmes entrées produisent exactement les
   mêmes top-1 et des scores égaux à `1e-12` près ;
9. p95 du scoring B inférieur ou égal à `1,25 ×` celui de A, mesuré après un
   warm-up identique, sur le même Mac et avec trois répétitions ;
10. temps total de construction, entraînement et scoring compatible avec une
    exécution locale de moins de huit heures ;
11. tous les gates d'intégrité et d'anti-fuite sont verts.

Ces gates sont volontairement conjoints. Une amélioration apparente de
quelques dossiers sans preuve appariée suffisante ne justifie pas de changer
le ranker qui déterminera la prochaine population à adjudiquer.

## 9. Verdicts

### `GO_ALIGN_RANKER`

Tous les gates de la section 8 passent.

Actions autorisées :

1. entraîner une dernière fois B sur tout le `fit` V4.2-B ;
2. sauvegarder modèle, métadonnées, ordre des features, dépendances, dataset
   manifest et hashes dans un bundle immuable ;
3. reproduire son hash et ses prédictions ;
4. seulement après ce gel, produire un **nouveau** shadow sur une population
   explicitement préenregistrée ;
5. adjudiquer les top-1 de ce nouveau shadow sans réutiliser automatiquement
   les verdicts liés au ranker A.

Le random V4.4 reste fermé et ne devient pas le jeu d'évaluation du nouveau
ranker.

### `KEEP_RANKER_A`

L'intégrité est valide mais un ou plusieurs gates statistiques, segmentaires
ou de latence échouent.

Le ranker A reste la référence. Aucun résultat n'est masqué. Une éventuelle
ré-adjudication des 37 scènes dérivées relève d'un contrat ultérieur distinct.

### `PIVOT_DATASET`

Le dataset V4.2-B ne peut pas être construit fidèlement pour les 7 003
requêtes, ou un défaut local reproductible empêche une comparaison appariée.
La cause et tous les artefacts produits sont conservés. Aucun résultat partiel
ne sélectionne un modèle.

### `STOP_INPUT_INTEGRITY` ou `STOP_LEAKAGE`

Une entrée, une frontière, un hash ou l'interdiction d'injection est violé.
Aucun modèle V4.6 n'est promu.

## 10. Ordre d'ouverture obligatoire

1. publier ce contrat et son SHA-256 ;
2. charger uniquement les entrées V4.1 fit/dev autorisées ;
3. construire et hasher les pools V4.2-B sans vérité ;
4. fermer les pools, puis joindre les labels historiques ;
5. auditer les frontières SIREN et les cinq folds ;
6. publier le manifeste du dataset et les gates d'intégrité ;
7. scorer A sur le dev V4.2-B ;
8. entraîner B en cinq folds OOF sur fit, puis le modèle fit complet ;
9. scorer B une fois sur le dev V4.2-B ;
10. calculer les métriques et appliquer mécaniquement les gates ;
11. publier un verdict unique ;
12. en cas de `GO_ALIGN_RANKER`, geler B avant toute création ou ouverture
    d'un nouveau shadow ;
13. conserver fermés V4.4/random, test final, V4-Fresh et anciens holdouts.

Il est interdit de revenir aux étapes 3, 5 ou 8 après lecture de la
comparaison dev. Toute correction indispensable crée une nouvelle expérience
et conserve le premier résultat.

## 11. Livrables minimaux

- dataset V4.2-B des 7 003 requêtes et son manifeste ;
- rapport d'intégrité et d'anti-fuite ;
- prédictions A dev et B OOF/dev, chacune hashée ;
- modèle B et métadonnées, même s'il n'est pas promu ;
- comparaison appariée complète ;
- rapport de latence Mac ;
- verdict machine-readable ;
- en cas de `GO_ALIGN_RANKER`, bundle B immuable destiné au prochain shadow.

Chaque manifeste doit inclure l'identifiant de l'expérience, le SHA-256 du
présent contrat, les versions Python, XGBoost et dépendances critiques, le
modèle de Mac, le nombre de threads effectif, la seed et tous les hashes
d'entrée et de sortie.
