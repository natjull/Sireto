# V239/V240 : apprendre pour le top100, pas le premier choix

Hypothèse figée avant fit et contrôle,5 octobre2026. Aucune nouvelle feature
ni source : décomposition réelle217 de232TRAIN64 complète,64 disjoints du128,
négatifs nationaux naturels227 et anciens locaux. Les cibles exactes ne
sont permises que dans ce dataset superviséTRAIN. Aucune injection serving.

Conserver exclusions held-fold/entités positives ET négatives, équivalents
officiels jamais négatifs. Trois modèles OOF234 ; onze poids nonnégatifs
0..64 appliqués dans les postings initiaux AVANT collecteur100, pas ranker
aval/pool fixe. Vocab/projections/alpha77 inchangés. Contributions fixes
217 sparse/propriétaires/successions/XY représentées exactement par232.
Ne PAS annoncer ce modèle calibré pour le nouveau kernel236.

Sur chaque requête TRAIN, z=[0,1+score_negative-score_positive]. Objectif
moyenne query max(0,moyenne des100 plus grands z) +0,005*L1(w-prior228).
Si moins100 classes observées, k=min(100,nneg+1) et compte explicite ;
ne pas retirer ces requêtes ni prétendre un rang national. C'est un surrogate
convexe sur concurrents MINÉS/CENSURÉS, pas le recall national exact.
Résolution LP sparse avec epigraphes tau/u/hinge, vérifier égalité objective
calculée directement et contraintes. HiGHS un thread,20s par fit,guards214
2Gio/80s/quiet30/pression1/free>=50 inchangées. Aucun sweep de k/marge/ridge,
aucun paramètre choisi sur128 ou contrôles OOF extérieurs.

Prior228 réel scellé, seul ancien poids place modifié, autres métadonnées77
identiques. Bundle écrit/hashé AVANT contrôles TRAIN ; contrôles censurés
des entités tenues NON utilisés pour sélectionner/refaire fit. Serving
autorité MODELE-SEUL ne lit pas les cibles232. Tous fichiers sources utilisés
FULLhash/epochs pré/post. Aucun fold0/1, aval gelé, Mac/CATNAT seul.

Inspiration : [Top-k Multiclass SVM](https://proceedings.neurips.cc/paper_files/paper/2015/file/0336dcbab05b9d5ad24f4333c7658a0e-Paper.pdf)
étudie des surrogates convexes top-k en classification. Ici l'adaptation
aux preuves initiales et aux négatifs minés est NOTRE hypothèse ; ni ses
résultats vision ni une meilleure loss TRAIN ne garantissent99,2% ou SLO.
Une recette de génération nationale nouvelle, figée et intégralement128
post-scellée, reste obligatoire avant toute mesure de rappel. Pilote réutilisé
ne vaut jamais READY/fullOOF/généralisation même100% ; pas union oracle.
