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

Les 172 dossiers peuvent être employés au fit et à un diagnostic hors pli
explicitement nommé `development_consumed`. Ils ne peuvent soutenir aucune
revendication de performance, aucun `GO` de déploiement et aucun seuil final.

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
- `feature_catalog.json` ;
- `manifest.json`.

Chaque ligne est une requête. Les features candidat sont calculées à partir
des candidats réellement retrouvés et classés. Aucun positif n'est injecté.
Une vérité absente du pool reste une erreur.

Sources historiques épinglées au moment du build :

- scènes accepteur V4.1 ;
- prédictions candidat hors pli V4.1 ;
- candidats V4.6 alignés ;
- requêtes V4.1 ;
- candidats et scènes V4.5 ;
- labels courants V4.7 ;
- file CRM V4.3 ;
- snapshot SIRENE autoritaire.

Le manifeste contient les hashes, versions, ordre des features, volumes,
taux de jointure, valeurs manquantes et invariants de non-ouverture.

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

## 6. Variantes de développement

Comparer exactement :

1. `BASE_FROZEN` : accepteur V4.1 et seuil gelés ;
2. `CURRENT80_REFIT` : régression logistique V4.8 sur les features actuelles ;
3. `STRUCTURED_LOGIT` : régression logistique sur la matrice V4.10 ;
4. `STRUCTURED_XGB` : XGBoost peu profond sur la même matrice V4.10.

Les preprocessings, seeds et hyperparamètres sont fixés dans le code avant
exécution. Une petite grille est autorisée uniquement si toutes ses valeurs
sont inscrites dans le manifeste avant le premier fit. En cas de parité, le
modèle le plus simple gagne.

Les 94 cas difficiles évaluables conservent leurs folds par composante SIREN.
Toute prédiction difficile doit être group-OOF. Les poids testés restent
`1`, `2` et `4`, explicitement considérés comme variantes préenregistrées.

Le seuil de développement maximise la couverture sur le dev historique sous
la contrainte de précision SIRET exacte ≥ 99,8 %. Il n'est jamais sélectionné
sur les labels difficiles, le random consommé ou un futur holdout.

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
