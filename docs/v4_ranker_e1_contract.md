# Contrat E1 — Ranker V4 exact-SIRET

## Objet

Après le verdict `GO_RANKER_V4`, entraîner un unique ranker candidat XGBoost
sur la vérité V4 courante et mesurer sa capacité à placer le bon SIRET en
première position sur `dev_new`.

Cette étape ne décide pas encore `AUTO_MATCH` et n’ouvre pas le holdout.

## Dataset candidat

- fit : 5 749 requêtes `MATCH_EXACT` dont le positif est réellement présent
  dans le pool à 100 ;
- dev : 305 requêtes `MATCH_EXACT` de `dev_new`, toutes conservées ;
- les requêtes fit `6818` et `8109`, dont le positif est absent de l’ancienne
  liste réutilisée, sont exclues du fit ranker ;
- aucun positif n’est injecté ;
- aucun SIREN exact n’est partagé entre fit et dev ;
- maximum 100 SIRET uniques par requête.

`AMBIGUOUS` et `UNRESOLVED` ne sont pas transformés en négatifs du ranker.
Les scènes `AMBIGUOUS` seront reconstruites séparément pour l’accepteur après
le verdict E1.

## Features et modèle gelés

- 44 features déterministes V7, sans les trois features sémantiques ;
- 11 features de provenance du retrieval gelé ;
- aucune feature V8 ;
- `XGBRanker` pairwise avec les hyperparamètres déjà versionnés dans
  `scripts/train_v9_ranker.py` ;
- cinq folds OOF groupés par SIREN ;
- seed 42 ;
- aucun GPU, dense, LLM ou service externe.

Le modèle final est entraîné sur tout le fit éligible. Le dev est scoré une
seule fois par ce modèle final.

## Comparaisons

Les trois ordres sont mesurés sur les mêmes 305 pools dev :

1. ordre brut de l’admission gelée ;
2. ranker E1 précédent, explicitement épinglé, si son ordre de features est
   strictement compatible ;
3. nouveau ranker V4.

Métriques : Hit@1 SIRET exact, Hit@1 SIREN, Recall candidat et nombres bruts.
Une vérité absente ou une requête non scorée compte comme erreur.

## Gate

- `GO_ACCEPTEUR_V4` si le nouveau ranker a le meilleur Hit@1 SIRET observé,
  ne régresse pas face à l’ancien ranker compatible et respecte tous les
  contrôles de données ;
- `KEEP_OLD_RANKER` si l’ancien ranker compatible est strictement meilleur ;
- `PIVOT_RANKER_V4` si aucun ranker n’améliore l’ordre brut ou si un contrôle
  de données échoue.

Aucun hyperparamètre, feature ou traitement ne sera modifié après lecture du
dev dans cette expérience. Un éventuel essai suivant devra être
préenregistré séparément.

## Holdout

`holdout_sealed` reste fermé : aucune liste candidate, feature, prédiction ou
métrique n’est produite avant le gel conjoint du ranker, de l’accepteur et du
seuil.
