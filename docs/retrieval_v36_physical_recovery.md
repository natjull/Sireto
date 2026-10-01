# Reprise physique de l'ablation v36

Le run `pilot_20261002_a` a terminé les trois folds de la recette primaire
et deux folds de l'ablation. La garde globale a ensuite interrompu le
troisième fit de l'ablation avec `STOP_RESOURCE_POLICY`. L'ancien runner
n'enregistre pas le compteur précis responsable dans son exception ; ne pas
présenter une hypothèse RSS ou swap comme un diagnostic établi. Son STOP et
ses artefacts restent intacts.

La reprise utilise un processus neuf par fold, séquentiellement, avec les
mêmes features, groupes complets, paramètres, seeds et labels. Chaque
processus limite sa durée à300s et son RSS à5Gio, sans nouveau swapout ; le
contrôleur borne l'ensemble à900s. Les échecs enregistrent cette fois les
compteurs réels. Les scores des deux folds déjà calculés doivent être
strictement identiques, candidat par candidat, avant adoption.

Les sources, features scellées, modèles et outcomes de la recette primaire
sont vérifiés par hash. La nouvelle racine publie la comparaison complète
et référence le STOP précédent. Aucune modification scientifique ni nouvelle
évaluation du fold1. Le résultat primaire ne peut être remplacé par une
ablation qui obtiendrait une meilleure métrique.
