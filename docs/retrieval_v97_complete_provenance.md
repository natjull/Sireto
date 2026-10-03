# V97 — provenance exhaustive, scratch isolé par bucket

V95 a terminé toutes16 phases de tuples :57804962 faits,162537492 observations
assignées. L'inverse global a STOP_RESOURCE_POLICY au deuxième bucket,
2,078s,pic4602052608octets,zéro swap. Aucune source complète scellée V95,
ni utilisation de l'inverse partiel ; code et artefacts arrêtés conservés.

V97 vérifie séparément les16 phases/partitions/tuples achevés, leurs ressources,
codes/sources hashés et cardinalités ; référence originale mapping162m reste
lecture seule. Seul inverse refait dans NOUVEAUX fichiers, chaque bucket en
processus indépendant pour libérer scratch et pages, sans recomputation des
noms ni restriction de scope. Assembly final vérifie exhaustivement chaque
mapping[inverse] contre tous IDs de faits, bitmap d'ordinals exactly-once,
nombre original162m, toutes offsets et références ; aucune observation capée.

Budgets identiques80s/4Gio/zéro swap par processus, superviseur90s, contrôleur
900s, deuxthreads, nouveaux fichiers2Gio préfixés (anciennes sources/tuples
référencés séparément, pas présentés comme supprimés). Sources originales
et toutes phases avant/après au contrôleur ; mapping650Mo et références de
phase hashés par worker. Pas label/CRM/folds0/1/fit/aval/dépense. Oracle contre
inverse indépendant, y compris doublons de nom/NULL/vide/context et corruption.
Pas recall mesuré. Après scellement : postings et intégration serving initial100.
