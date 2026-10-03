# V100 — mots/owners complets, metadata équivalentes vectorisées

V98 prepare et tous32 mots + owners achevés sous gardes, mais metadata
STOP_RESOURCE_POLICY à80,150s,pic2650292224octets,zéro swap. Aucune source
complète scellée V98, aucune utilisation des mmap metadata partiels.

V100 récupère SEULEMENT les phases prepare/mots/owners complètes, ressources,
codes, source/tuple et artifacts hashés avant/après. Pas nouvelle restriction
de scope. Toutes162537492 metadata sont réencodées dans de NOUVEAUX fichiers,
même dates/null/priorité/source/kind/current que V86, codes Arrow index_in
au lieu de deux listes Python géantes. Oracle égalité de toutes arrays et
codes, fail unknown/date-unit ; aucun champ retiré ni preuve inventée.

Même80s/4Gio/zéro swapout, superviseur90s,deuxthreads, nouveaux fichiers4Gio
préfixés pour metadata18octets×162,5m. Anciennes sources/word/owners restent
référencées, pas prétendument effacées. Fichier original5,5Go hashé avant/après
metadata, phases complètes avant/après contrôleur. Aucun CRM/label/fold0/1,
fit/aval/dépense, aucun recall revendiqué. Étape suivante : vraie génération
initial100 sur noms nationaux directs et une-hop, collecteur exhaustif borné100.
