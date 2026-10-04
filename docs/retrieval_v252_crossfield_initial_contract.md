# V252 : apprendre trois nouvelles preuves, pas changer le head229

Avant fit5 octobre2026. Source251 complet64 TRAIN/8 producteurs214 et parité
conditionnelle native vérifiée sur tous les points. Disjoint du pilote128,
folds234 seulement ; fold0interdit, fold1fermé, modèles aval gelés.
Original77safe vocab/projection/alpha et poids11prior228 strictement fixes.
Apprendre UNIQUEMENT les trois coefficients250 nonnégatifs0..64, départ0,
prior0/L2ridge0,005. Chaque query pèse autant ; négatifs sûrs de sa scène
équipondérés. Logistic pairwise avec offset score217228 COMPLET scellé232
(nom/adresse+fixedsparse/owners/relations/XY), pas logsumexp top1 ni grille.

Même registry entités held complet, exclusion folds held ET positif/negatifs
du SIREN held, conservation membershipquery77, équivalents officiels de
TRAIN exclus des faux négatifs. Ne pas construire feature depuisGT ; vérité
supervisée sert uniquement comme cible/référence d'apprentissage.
Modèles écrits/hashés AVANT contrôles OOFTRAIN, aucune sélection ou refit
à partir de ces contrôles censurés. Ce n'est PAS recall national/top100
validé et ne prouve pas que la nouvelle hypothèse sera bénéfique.

Sources/hash/epochs pré/post, autorité réelle exit0/PASS214, quiet30/2GiB/
80s/unthread/pression1/free50. STOP si dérive, aucun retry budget élargi.
Autorité serving modèle-seul : pas ouverture des lignes TRAIN251/cibles.
Poids compilés AVANT collection mondiale100, pas pool puis ranker renommé.
Validation produit ultérieure128intégrale avec métriques postseal distinctes
et ciblesall-human exact/op immuables. Aucun READY/SLO/fullOOF/généralisation.
