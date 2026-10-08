# V7 — Lots d'inférence sans changement de recette

V6 a terminé réellement l'ablation native160 :80 requêtes,107682 textes,
938,705s, ownedexit0 et sorties scellées avant labels. Son ablation officielle
512 avance environ45 textes/s ; environ215k textes site+identité par variante
rendent les deux ablations restantes incompatibles avec l'échéance08:43.

V7 ne change aucun poids, texte, contexte, candidat, score mathématique, RRF
ou split. Il réutilise la baseline native scellée et le fit V6 finalisé. Seule
l'inférence des deux ablations officielles passe de lots8 à lots32, ordonnés
par longueur puis restitués dans l'ordre initial. Le padding diminue. Pas de
troncation additionnelle, pas de nouvelle étape de gradient.

Préflight obligatoire avant générations :128 textes officiels choisis par
hash, comparaison ancien/nouveau encodeur, finitude, erreur absolue<=.002 et
erreur cosinus<=1e-5, accélération réellement mesurée. Ces tolérances sont
fixées avant toute lecture de labels ; aucune identité bit-à-bit ni invariance
de tous les rangs à l'arrondi n'est revendiquée. En cas d'échec, STOP, jamais
un seuil assoupli selon les résultats. MPS.4/RSS8Gio et guards V4 conservés.

Namespace neuf, anciennes générations incomplètes conservées, aucun code
historique hashé modifié. Trois variantes complètes avec ownedexit0/scellés
obligatoires avant labels. Géographie V5 réutilisée sans rescan ; exact et
opérationnel séparés.80 requêtes diagnostiques, pas gate nationale99,2%.
Échéance absolue08:43 inchangée ; test4323/folds0/1 et modèles aval gelés.
