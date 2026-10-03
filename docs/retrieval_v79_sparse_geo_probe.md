# V79 — Résidu géographique de la recherche sémantique initiale

Préenregistrement avant mesure. Objectif initial100 exact≥99,2% inchangé,
aval gelé. Contrastes TRAIN512 ne valident PAS recall/SLO/généralisation.

Préfixe V77 figé (native11 + mots officiels sémantiques, alpha/ridge choisis
dans son inner uniquement, vocab/projection outer protégés). Ajouter8
composantes V56 : proximité 1/(1+d/scale) sur25/100/500/2000m, mêmes4
conditionnées par BAN non déclarée faible. Distance à vol d'oiseau, pas
adresse textuelle ni route. Banque faible garde les4 proximités brutes ;
coordonnées absentes/incompatibles =>8 zéros, pas retrait d'un SIRET ou
requête. Aucun rayon/état/commune comme filtre global.

Calibrer seulement8 poids positifs par InfoNCE égal-query avec tous négatifs
naturels autorisés, ridge0,005,max180 itérations,bounds0..64. Prédictions de
base pour la calibration issues de modèles fit sur L'AUTRE fold source
autorisé : vocabulaire, cibles, projection et poids natifs excluent le fold
de la requête calibrée ET parents outer/inner protégés comme négatifs.
Hyperparamètres sémantiques V77 figés, non resélectionnés dans ces crossfits
(ils ont été choisis dans l'inner de l'outer, pas une double nested sélection
du décodeur). Ce scope doit rester explicite : on crossfit les fits, pas
une nouvelle sélection de tous les hyperparamètres à chaque sous-fit.
La validation inner a déjà servi à choisir les hyperparamètres V77 : elle
est réutilisée pour le choix géographique, donc ce contrôle inner n'est
pas une nouvelle population indépendante. Les contrôles outer restent
hors de tous ces choix et aucun contrôle TRAIN ne vaut recall national.

Familles BASELINE (8 zéros) et ADD_GEO8 seulement. Sélection uniquement
inner entreprises V53, critère erreur de tête puis mean_outrank, tie=>BASELINE.
Fit non convergé exclu AVANT sélection ; refit sélectionné non convergé=STOP.
Refit8 poids sur prédictions crossfit des requêtes nonouter, puis contrôle
outer APRÈS fit, sans choix selon métriques extérieures. Préfixe final V77
reste bit-identique. Pas alias CRM, ID/synthétique comme feature, nouvelle
vérité/candidat injecté ou altération du dénominateur all-human final.

Réutiliser uniquement TRAIN512 V52/V56/V67 et leurs preuves officielles,
registre FULL234, pas les14 pertes V78 comme corpus/features. Scènes/teacher
historiques ne remplacent pas l'all-human final. Tests gradients/crossfit,
protection textes et features outer, parent négatif protégé, préfixe gelé.
Worker80s/4Gio/zéro swapout, supervision90s, threads2. Aucun cloud/dépense,
grand build, nouveau modèle aval ou ouverturefold0/1. Une sélection favorable
nécessitera un VRAI score initial national natif+spatial, bornes globales
jusqu'à100, sans pool fixe/reranking. Collecteur et coût restent à valider.
