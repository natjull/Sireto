# V71 — Conditionnement sans changement de kernel ou d'objectif

Préenregistré avant mesure. V70 a exclu six candidats contrastifs sur limite
180itérations ; ne pas les appeler inutiles ni sélectionner les non convergés.

Une modification : optimisation dans eta où theta=origin+scale*eta,
origin=prior23 natives et decoder ridge10 train-only, scale=1 natives et
max(1,sqrt(dimension_latente/(2*ridge))) pour decoder. C'est le changement
de coordonnées induit par le Hessien ridge seul. Aucun label/contrôle sert
à calculer ce facteur. Loss V70 EXACTEMENT identique, gradient par chaîne,
bounds physiques natives[0,64], decoder[-16,16] transformés exactement.
Résultat reconverti en theta, jamais changement de l'activation ou du score.

NATIVE23/RIDGE_SPARSE délèguent au solver V70 gelé, comparateurs identiques.
Nested entities, vocabulaire1024, bases16+16+biais, top16 continu,
ridge0,01/0,1, FLOPS0,1 et source512 inchangés. Sélection interne uniquement,
non convergés exclus, refit train+inner et contrôle outer ensuite uniquement.
Maxiter180,ftol1e-8,gtol1e-5,budget80s/4Gio/zéro swapout,CPU2inchangés.
Le gradient eta n'est pas le gradient theta : publier aussi norme projetée
physique et raison d'arrêt ; pas de relâchement de tolérance. Facteur>=1
ne rend pas une tolérance absolue sur gradient decoder plus permissive.

Loader commun V71 nouveau, logique V70 reprise sans modifier son code ou
ses manifests : caches hashés, champs vides masqués, alias STOCKÉS, audit
exhaustif exact courant/native, identité et dénominateurs contrôlés.
Tous refits sélectionnés doivent converger, sinon STOP non déployable.

Sortie models/reports/manifest scellée, PAS candidats ou recall national.
Tests égalité loss/gradient, bounds transformés, finite differences et
indépendance outer ; serving identique à V70. Aucun modèle aval modifié,
fold0 jamais lu/fold1 fermé, all-human/sidecar intacts. Pas de grand build,
nouvel encodage ou dépense externe.

Limites V70 inchangées : contraste teacher restreint, documents fixes,
alias en bags et pas source/période cohérente, graphes typés/juridique/activité
non intégrés, exécuteur national conjoint sparse+XY/proofs et SLO non réalisés.
Le dernier recall mesuré V57 sur128 ne change pas par ces contrôles.
