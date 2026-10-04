# V213 — rapport qualité de la recette structurelle, après seal uniquement

Noyau métrique sans IO, entraînement, retrieval ou changement de budget.
Une entrée en mémoire n'est PAS permission d'ouvrir les labels. Le caller
national doit refuser LIVE/STOP/partiel, valider réellement les128 outputs,
leurs producteurs supervisés et leurs hashes AVANT tout label-side. Ce
noyau ne remplace ni cet évaluateur séparé ni le contrat initial100.
V212 STOP reste INADMISSIBLE ; aucun calcul de recall national213 prévu
tant qu'un pilote complet n'a pas réussi sous un protocole autorisé.

Adaptation explicite `results.primary` -> `candidates` ; jamais score aval,
aucun recapping, rescore, filtre statut ou réparation GT. Mêmes128 positions,
folds234,100 max, identités/rangs/canaux/seconde/type stricts. Réutiliser le
noyau200 all-human exact/opérationnel/Wilson/segments et les cibles officielles
scellées. L'équivalence opérationnelle ne vient pas de proximité/succession.

Publier deux présences distinctes pour chaque vérité : index natif43896818
et catalogue disponible44403034. Native absent implique disponible peut
être présent ou absent. Aucune absence ne retire une ligne du dénominateur.
Le caller fournit ces lookups POSTSEAL seulement. Pertes : absence des
sources disponibles, source présente mais site non indexé, non retrouvé
dans100 malgré présence native. Ce n'est pas une attribution fine au score,
filtre, canal ou qualité du label, qui reste non mesurée.

Rapport canaux = nombre de places admises, queries représentées et hits
exacts ATTRIBUES au canal qui a émis le candidat, sur128. PAS recall d'un
canal isolé ou d'une union exhaustive jamais calculée. Les doublons retirés
par l'émission initiale ne sont pas reconstitués artificiellement.

Latence : p50/p95/p99 descriptifs sur les128 secondes enregistrées, cache
non contrôlé ; aucun SLO validé. Pas READY même100%, ni fullOOF ou preuve de
généralisation. Fold0 interdit/fold1 fermé/aval gelé, aucun coût externe.
Code/contrat/tests commités avant mesure nationale puis immuables.
