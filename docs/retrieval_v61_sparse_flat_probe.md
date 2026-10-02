# V61 — optimisation physique V60, pas de nouveaux paramètres

V60 terminal STOP au contrôleur360s après120 sorties, labels non ouverts ;
partiel immuable, pas de recall extrapolé. V61 est une variante physique
distincte, préenregistrée avant nouvelle génération, pas une reprise
élargie du job vivant ni un déverrouillage des labels partiels.

Supprimer le doublon de guard SIRET du score de base imbriqué et aplatir
ses11 clauses dans le score natif, puis ajouter en dernier la même
composante sémantique. Un seul filtre SIRET const0 partagé. Même score
mathématique, poids/projections/vocabulaire/prefix/encoder V59/V60
inchangés, alpha2/ridge10, aucune sélection de variante via labels.
Tests de parité memberships/scores sur corpus miniature320 documents.
L'ordre interne du calcul natif float32 peut varier ; tolérance1e-5,
pas de promesse bit-exacte ni de parité globale des égalités au cutoff.

Même panel128 réutilisé, corpus national complet43896818 sites,
sortie initiale<=100 sans pool/reranker/injection, contrôles OOF entités
et dénominateur all-human. Première garde16 sans labels moyenne<=4s.
Worker80s/6Gio/zéro swap, supervision90s, contrôleur360s INCHANGÉS,
deux threads ; encodage query-only128 inclus budget, aucun gros build.
Pas de relance automatique élargie si STOP.

Les autres protections/limites v60 s'appliquent : mots supplémentaires
seuls gated commune/CP, recette de base globale ; adresses/numéro/site
conservés, XY/graphes/catégories/temporel encore hors de cette ablation.
Évaluation séparée seulement après128 scellées ; pas READY_FOR_FOLD1,
fold0 non lu/fold1 fermé/aval gelé/SLO non validés par le pilote.
