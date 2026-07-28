# Contrat V4.10 — accepteur structuré identité et site

Statut : préenregistré après `STOP_RETRAIN` V4.8,
`STOP_SITE_FUNCTION_GUARD` V4.9 et l'audit statique V4.10, avant construction
du nouveau dataset, entraînement ou ouverture d'une population fraîche.

Identifiant : `V410_STRUCTURED_ACCEPTOR`.

## 1. Question

Un accepteur query-level unique, recevant les informations d'identité et de
site actuellement perdues après le ranker, peut-il augmenter la sécurité des
décisions AUTO sans perdre plus de deux points de couverture ?

V4.10 ne modifie pas :

- le retrieval V4.2-B ;
- le plafond de 100 candidats ;
- le ranker A ;
- les SIRET proposés par le ranker ;
- le test final historique.

Elle remplace expérimentalement la représentation et le modèle de décision
AUTO/REVIEW. Aucun veto lexical indépendant n'est ajouté après l'accepteur.

## 2. Populations consommées

Sont définitivement interdits comme validation V4.10 :

- test final historique ;
- holdout V4-Fresh ;
- random V4.8 ;
- les 172 dossiers V4.7/V4.9 ;
- toute ligne déjà utilisée pour choisir une règle, une feature ou une
  famille d'erreurs V4.10.

Seuls les 94 dossiers non-random portant le rôle V4.8 `hard_oof` peuvent être
employés au fit et à un diagnostic hors pli explicitement nommé
`development_consumed`. Les quatre `hard_dev_locked` restent descriptifs,
hors fit, sélection et gate. Les 57 lignes `random_sealed` sont physiquement
absentes de tout parquet lu par le trainer et ne sont jamais scorées.

Ces données consommées ne peuvent soutenir aucune revendication de
performance, aucun `GO` de déploiement et aucun seuil final.

## 3. Architecture

```text
CRM + top-100 du ranker A gelé
  → features de scène V4.1 non sémantiques
  → features candidat top-1/top-2 conservées
  → relation au SIRET/SIREN d'entrée et état
  → activité/fonction SIRENE
  → constellation des candidats du même SIREN
  → interactions identité/adresse
  → accepteur unique
  → AUTO_MATCH ou REVIEW
```

Les règles déterministes produisent seulement des features. Elles ne peuvent
que renseigner l'accepteur ; aucune règle ne décide seule après le modèle.

## 4. Dataset canonique

Un builder unique produit sur le SSD externe :

- `historical_scenes.parquet` ;
- `consumed_hard_scenes.parquet` ;
- `descriptive_locked_scenes.parquet` ;
- `feature_catalog.json` ;
- `manifest.json`.

Chaque ligne est une requête. Les features candidat sont calculées à partir
des candidats réellement retrouvés et classés. Aucun positif n'est injecté.
Une vérité absente du pool reste une erreur.

Sources historiques épinglées au moment du build :

- scènes accepteur V4.1 ;
- prédictions candidat hors pli V4.1 ;
- candidats V4.1 correspondant exactement à ces prédictions ;
- requêtes V4.1 ;
- candidats et scènes V4.5 ;
- labels courants V4.7 ;
- file CRM V4.3 ;
- partitions et rôles V4.8 ;
- snapshot SIRENE autoritaire.

Gel des entrées du premier build :

| Entrée | SHA-256 |
|---|---|
| scènes accepteur V4.1 | `8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9` |
| prédictions candidat V4.1 | `eea22c58378d8adc232a7f2723c0a84323963db9633a7bb9af2e2485cd6329d2` |
| candidats V4.1 | `34b526fe49e3451c05248294305e4a8d6ccf4db92277eb36dc03cc6231420c67` |
| requêtes V4.1 | `6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0` |
| candidats difficiles V4.5 | `9f48a558bc77bf9db835e7689963989ba99d2914fb1add32be4988ec3cab3242` |
| scènes difficiles V4.5 | `72540dcdba6f33da0eb1875ef4bcdc8c44a2cd10083589b5e1683098cd954a08` |
| labels courants V4.7 | `e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2` |
| file CRM V4.3 | `47af4887769a2edb11f1e629c38077edccd035dd96cb3a6d39620714361fdecc` |
| partitions V4.8 | `f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9` |
| manifeste partitions V4.8 | `f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33` |
| snapshot SIRENE | `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` |
| taxonomie V4.9 | `48bbb7e1795a0731f1f12df41aeb971667c10d03c879bf06d5ba15b65f8b121d` |

Le manifeste contient les hashes, versions, ordre des features, volumes,
taux de jointure, valeurs manquantes et invariants de non-ouverture.

Les 698 428 paires candidat-prédiction V4.1 doivent se joindre exactement par
`(query_id, candidate_siret)` ; deux sentinelles sans candidat sont conservées.
V4.6-B ne doit pas être utilisé pour enrichir ces prédictions V4.1, car les
deux pools ne sont pas identiques. Le fit historique reste donc retrieval
V4.1 hors pli, tandis que les cas difficiles et le futur utilisent V4.2-B.
Cette hétérogénéité est publiée et acceptée uniquement comme étude de
faisabilité.

Le builder charge uniquement les colonnes de features nécessaires. Il lui est
interdit de charger ou d'utiliser `is_ground_truth`, `ground_truth_*`,
`validated_correct_*`, une cible alternative ou une preuve pour construire
les features. La cible est jointe après gel de la matrice.

## 5. Blocs de features

### 5.1 Scène existante

Conserver les 71 colonnes V4.1 utiles. Les neuf colonnes sémantiques
constamment nulles sont exclues du nouvel ordre de modèle et signalées dans
le catalogue.

### 5.2 Candidat top-1 et top-2

Propager les features ranker aujourd'hui perdues, notamment :

- `input_siret_exact_match`, `input_siren_exact_match` ;
- état actif/fermé/inconnu ;
- provenance sparse, SIRET/SIREN d'entrée et alias fermé ;
- forme juridique, siège, association et indicateur école ;
- égalités géographiques, nominales et de numéro de voie ;
- source du meilleur nom ;
- rangs et scores détaillés des canaux.

Une feature numérique produit `top1`, `top2`, `delta` et indicateur manquant.
Une feature catégorielle produit des catégories épinglées, une catégorie
`UNKNOWN` et une égalité top-1/top-2. Aucun encodage ne peut être appris sur
dev ou holdout.

### 5.3 Interactions identité/adresse

Réintroduire les sept interactions V8 déjà définies :

- `addr_unsupported_by_name` ;
- `name_density_penalty` ;
- `addr_jaro_per_density` ;
- `postcode_match_without_addr` ;
- `full_addr_match_score` ;
- `name_jaro_vs_enseigne` ;
- `name_city_suffix_match`.

Ajouter seulement les interactions générales suivantes :

- adresse forte avec nom faible ;
- SIREN différent avec adresse forte ;
- SIREN identique avec ville ou CP incompatible ;
- SIRET d'entrée exact mais fonction incompatible.

Leur formule et leurs seuils de calcul sont versionnés avant tout score de
modèle. Ils ne sont pas des seuils AUTO.

`name_jaro_vs_enseigne` et `name_city_suffix_match`, absents des parquets
gelés, sont recalculés depuis le CRM brut et les enseignes du même snapshot
avec le normaliseur versionné. Les cinq autres interactions se dérivent des
features numériques gelées.

### 5.4 Activité et fonction

Relire `activitePrincipaleEtablissement` dans le snapshot maître pour top-1,
top-2 et candidats du même SIREN :

- section NAF ;
- division NAF ;
- égalité top-1/top-2 ;
- nombre de divisions parmi les frères ;
- rôles de la taxonomie V4.9 ;
- conflit ou pluralité de rôles.

La taxonomie V4.9 reste gelée. Elle devient un signal, pas un garde. Aucun
libellé complet ni identifiant de dossier ne peut entrer dans une règle.
Le snapshot est joint par SIRET normalisé sur 14 chiffres, avec unicité
obligatoire et couverture top-1/top-2 à 100 %. En l'absence d'activité CRM,
un conflit de fonction signifie uniquement rôle lexical CRM contre rôle
SIRENE candidat ; le code NAF CRM ne peut pas être inventé.

### 5.5 Constellation intra-SIREN

Sur les candidats du même SIREN que le top-1 :

- nombre total et nombre de sites actifs ;
- meilleurs et seconds scores nom/adresse ;
- meilleur site par nom, par adresse et par score ;
- nombre de sites compatibles CP, ville et numéro de voie ;
- écarts entre le top-1 et le meilleur frère pour nom et adresse ;
- désaccord entre meilleur site nominal et meilleur site géographique ;
- nombre de fonctions et de divisions NAF distinctes.

Ces features utilisent tous les candidats du pool, jamais un établissement
réinjecté depuis la vérité.

Un frère est un autre candidat réellement présent dans le pool et portant le
même SIREN. Les égalités sont départagées par score ranker décroissant, rang
retrieval croissant, puis SIRET lexical croissant. Aucun SIRET, SIREN, rang
identifiant ou texte brut n'entre dans l'ordre modèle. Les cas sans candidat,
avec un seul site et avec 100 candidats possèdent des tests train/serve.

## 6. Variantes de développement

Comparer exactement dix variantes :

1. `BASE_FROZEN` : accepteur V4.1 et seuil gelés ;
2. `CURRENT80_W1`, `CURRENT80_W2`, `CURRENT80_W4` ;
3. `STRUCTURED_LOGIT_W1`, `STRUCTURED_LOGIT_W2`,
   `STRUCTURED_LOGIT_W4` ;
4. `STRUCTURED_XGB_W1`, `STRUCTURED_XGB_W2`,
   `STRUCTURED_XGB_W4`.

Un `training_plan.json` immuable est écrit et hashé avant le premier fit.
Aucune grille supplémentaire n'est autorisée.

- `STRUCTURED_LOGIT` : standardisation apprise sur le train du pli pour les
  continues seulement, `LogisticRegression(C=1, class_weight="balanced",
  max_iter=3000, solver="lbfgs", random_state=42)` ;
- `STRUCTURED_XGB` :
  `tree_method="hist", n_estimators=300, max_depth=3,
  learning_rate=0.03, min_child_weight=10, subsample=0.8,
  colsample_bytree=0.8, reg_lambda=10, reg_alpha=0,
  objective="binary:logistic", eval_metric="logloss", random_state=42`,
  sans early stopping ;
- poids historiques `1`, poids difficiles exactement `1`, `2` ou `4`.

Pour XGBoost, les facteurs de classe sont calculés sur le seul train du pli
et multipliés par le poids difficile. En cas de parité : plus de mauvais
hors pli refusés, plus de bons hors pli conservés, plus de couverture
historique, famille la plus simple, puis poids le plus faible.

Les colonnes continues manquantes deviennent zéro avec un indicateur de
manque explicite produit par le builder. Les catégories NAF et métier ont un
vocabulaire épinglé et `UNKNOWN`; elles ne sont jamais ordinales ni apprises
sur dev. Aucun `SimpleImputer(add_indicator=True)` à largeur variable n'est
autorisé.

Les 94 cas difficiles évaluables conservent leurs folds par composante SIREN.
Toute prédiction difficile doit être group-OOF. Les poids testés restent
`1`, `2` et `4`, explicitement considérés comme variantes préenregistrées.
Pour chaque fold, les lignes historiques `historical_hard_support` portant le
même `hard_fold` sont elles aussi exclues du train.

Le seuil de développement maximise la couverture sur le dev historique sous
la seule contrainte de précision SIRET exacte ≥ 99,8 %, avec au moins 100
AUTO. `AUTO` signifie `score >= seuil`. Les candidats seuils sont les scores
distincts du dev plus les deux sentinelles `nextafter`. Il n'est jamais
sélectionné sur les labels difficiles, le random consommé ou un futur
holdout. La non-infériorité et la couverture sont vérifiées ensuite comme
gates, pas ajoutées à la recherche du seuil.

## 7. Gate de développement

Une variante peut obtenir `GO_FRESH_DEV_V410` seulement si :

- reproduction exacte de `BASE_FROZEN` ;
- au moins 24/25 `TOP1_WRONG` difficiles hors pli refusés ;
- zéro `AMBIGUOUS` difficile hors pli en AUTO ;
- au moins 58/68 `TOP1_CORRECT` difficiles hors pli conservés AUTO ;
- précision historique observée ≥ 99,8 % ;
- précision historique non inférieure à `BASE_FROZEN` ;
- couverture historique à moins de deux points de `BASE_FROZEN` ;
- aucune ligne random V4.8 lue ou scorée ;
- aucun test final lu.

Ici, « aucune ligne random lue » signifie que le trainer ne reçoit aucun
fichier contenant ces lignes. Le builder peut lire les identifiants et rôles
V4.8 afin d'effectuer un scan parquet filtré ; aucune cible random n'est
matérialisée dans ses sorties.

Ce gate autorise uniquement la constitution d'un dev frais. Il ne valide pas
le modèle.

Si aucune variante ne passe : `PIVOT_STRUCTURED_FEATURES` ou
`STOP_STRUCTURED_ACCEPTOR`.

## 8. Dev frais

Après `GO_FRESH_DEV_V410` :

1. geler le builder, le catalogue, le modèle candidat et son seuil
   provisoire ;
2. tirer avant scoring au moins 300 dossiers CRM absents de toutes les
   populations V4.1–V4.9 ;
3. reconstruire retrieval V4.2-B et ranker A sans injection ;
4. adjudiquer autonomement chaque top-1 avec deux groupes de preuves
   indépendants lorsque possible ;
5. conserver `UNRESOLVED` hors précision ;
6. utiliser ce dev uniquement pour choisir entre les variantes déjà
   préenregistrées et calibrer le seuil ;
7. geler un bundle final avant toute nouvelle cohorte.

Le dev frais doit publier les nombres bruts, couverture identifiable, précision
SIRET exacte, risque-couverture, Wilson et résultats par famille critique.

## 9. Holdout indépendant

Une nouvelle cohorte indépendante, tirée et scellée après le gel du bundle,
est obligatoire pour toute conclusion :

- précision AUTO SIRET exacte observée ≥ 99,8 % ;
- perte de couverture ≤ 2 points contre `BASE_FROZEN` ;
- aucune famille critique ne perd plus de 5 points de couverture ;
- aucune règle ou seuil modifié après ouverture ;
- verdict explicite `GO_SHADOW_V410`, `PIVOT_V410` ou `STOP_V410`.

Un petit holdout mesure une faisabilité, pas une certification. Revendiquer
une borne unilatérale à 99 % compatible avec 99,8 % exigera environ 2 300
AUTO indépendants sans erreur.

## 10. Ressources et gouvernance

- Mac M4 Pro et `/Volumes/CATNAT_DATA` uniquement ;
- aucun GPU loué ;
- aucune API payante ;
- aucun LLM en inférence ;
- LLM autorisé uniquement pour réunir/résumer des preuves, jamais comme
  vérité autonome ;
- chaque milestone dans un commit isolé cité dans `handover.md`.
