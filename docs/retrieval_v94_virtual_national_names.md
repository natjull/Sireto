# V94 — même portée nationale sans copie redondante des observations

V93 a STOP_RESOURCE_POLICY en80,361s lors de COPY de tous les noms,
pic2256781312octets, zéro swap. Registre entier validé/exporté mais sortie
globale non scellée : ne pas utiliser sa copie partielle comme corpus complet.
Code et artefacts V93 conservés inchangés.

V94 réalise la MÊME projection logique sur TOUTES162537492 observations
du fichier officiel déjà hashé et immuable. Source, ordinal, rôle, éligibilité,
dates et provenance sont accessibles par une vue SQL déterministe scellée,
pas par une nouvelle copie5,5Go. Aucun filtre/limite/corpus16/label/graphe,
et les43896818 membres du registre sont toujours validés et exportés.
Les lecteurs doivent vérifier l'autorité du manifeste et reconstruire la
vue depuis le code hashé, jamais exécuter une chaîne SQL arbitraire.

Oracle de fixture : égalité de TOUTES colonnes et observations avec l'export
V93, y compris closed/siblings/contexte/grain invalide/doublons/provenance.
En national, cardinalités et comptages par rôle/grain/éligibilité ; pas encore
jointure indépendante exhaustive de chaque ordinal contre toutes les données.
Même budget80s/4Gio/zéro swap, supervision90s, DuckDB1GB/spill2GB/deuxthreads.
Plafond des NOUVEAUX fichiers1Gio, source originale référencée séparément ;
il ne mesure pas la taille logique du corpus, qui reste intégral.

Pas d'index serving ni candidat produit ni nouveau recall ni modèle fit.
Suite : factorisation sur cette vue complète avec contexte original retenu,
postings/membership complets et vrais flux initiaux100, sans pool fixe,
scopes et oracles, puis coût aveugle et évaluation scellée séparée.
Les protections folds2/3/4 uniquement, aval gelé, objectif99,2% exact,
fullOOF/généralisation/SLO et absence d'injection restent inchangées.
