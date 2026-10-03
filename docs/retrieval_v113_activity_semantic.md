# V113 — sens des activités officielles, contrôles TRAIN seulement

Comparer un a priori sémantique de libellés aux fréquences TRAIN, pas lancer
un retrieval national ou annoncer du recall. Sources : catalogue V112 et
2140 définitions versionnées NAF2008/2003/1993, NOM MiniLM local figé de V52.
Seul le LIBELLÉ officiel entre dans l'embedding ; pas code activité, version,
SIREN/SIRET, alias CRM ou nom reconstruit à partir d'une vérité.
Même préfixe `nom`, masque IDs9/14 et longueur128 que le cache NOM ;8 premiers
TRAIN noms vérifiés avec maxdiff<=1e-5 AVANT tout encodage de description.

Jointures exactes `(nomenclature originale,code original)` pour23874
documents TRAIN, pas inference par format/date. NAP et couples contradictoires
ou descriptions absentes ont preuve0, PAS exclusion de sites/requêtes ou
application d'un code d'une autre version. Les512 scènes/négatifs naturels
V52 sont conservés ; unknowns et ties publiés. Les tables anciennes restent
distinctes, même si deux libellés ont le même sens/vecteur ; pas de claim
que le nom CRM pourrait prédire la version d'une nomenclature historique.

Sélection par contrôles inner entity-disjoint uniquement entre fréquence
TRAIN, zero-shot dot NOM/libellé, et ridge1/10/100 vers le vecteur de libellé
du positif TRAIN admissible. Projection385x384 avec biais, normalization du
vecteur prédit (zero reste zero), cos borné0..1, PAS probabilité calibrée.
Unknown targets0 mais lignes maintenues. Les familles de fold tenu et inner
sont hors fit, négatifs protégés hors contrôles inner ; refit hors fold tenu.
Rapports outer fréquence/zero-shot/sélection, jamais utilisés pour choix.
Tout document absent du registre ou contrôle sans négatif sûr donne STOP.

Activité-définition top1/top5, paire erreur égale par requête, head error,
ties/même activité/unknowns publiés par fold : PAS Recall SIRET@100. Une
définition partagée entre versions/sites ne fournit aucune identité exacte.
Pas de candidat émis, pool fixe, reranker ou changement des modèles aval.

Deux phases distinctes préparatoires : encodage SOURCE2140 et fit TRAIN512.
Chacune garde80s/4Gio/zéro swap et superviseur90s ; deux threads CPU, local
offline sans modèle téléchargé/dépense externe. Préflight128 descriptions
estime<=60s sinon STOP, pas relance élargie. Hashes avant/après des sources,
code et poids d'encoder ; troncatures publiées. Fold0 jamais lu/fold1 fermé.

Seulement si les contrôles montrent une utilité, tester l'interaction avec
nom officiel/adresse/XY et confiance BAN/relations dans la recherche initiale
à100 via preuves et bornes correctes. Ce probe ne remplace pas ce test et ne
franchit aucun gate99,2% exact all-human/fullOOF/Wilson/généralisation/SLO/
attribution. Dernier vrai recall mesuré V101114/12889,0625% inchangé.
