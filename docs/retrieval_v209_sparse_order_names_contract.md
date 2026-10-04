# V209 — ordre sparse stable uniquement, sans warm

Protocole avant mesure. V208 numériquebit-exact a été plus lent AU TOTAL
dans6comparaisons : initialisation0,68..1,40s, non masquée dans le coût.
V209 supprime complètement ce reset/touchfloat. Construction originale170
fichiersneufs/zerosfilesystem, flags205 inchangés. Pas benchmark modifié
pour masquer warm : mêmes seed208/world30000000, dense1048576/sparse65536,
4calls/3trials alternés et init+maximum total mesuré pour les DEUX modes.

Maximum.at original sur TOUTESréférences ; seules positions opaques d'un
bloc >=4096refs et span>256*refs sont ordonnées STABLEMENT. Doublons et
signedzeros dans leur ordre original, blocs131072, aucun reduceat/skip
ni dtype changé. Les autres blocs restent dans l'ordre original. Aucune
identité ou propriétésource utilisée comme feature/filtre/candidat sélectionné.

Query205 complet inchangé ailleurs, compteur198/ALLaliases, propriétaires/
patches149/parents/sites séparés, relations201 figées1hop/tousliens/deuxsens/
.5/.25/noexactprop. Profil physique uniquement. Timerquery20 global,
flatten20/worker80/RSS1Gio/quiet30/p1/free>=50 inchangés, swapGLOBAL
nonattribué. Aval/folds0/1/GT gelés. STOP206 conservé, jamais retrié/réadmis.

Tests scalar150/138 BYTES toutespositions/gathers/chunks/textes/vides,
forçage sparse, signedzero/doublons, référencesinvalides, spy interdit
reset208 et groupedmaximum204. FULLhash toujours requis : metadata seules
ne détectent pas forcément une écriture mmap. Fixturesnumériques seulement,
pas features métier synthétiques ni label national.

Bench fresh sur CATNAT dansretrieval_v209_numeric_storage/benchmark.*,
samefilesystem publié, cache noncontrôlé. ALL30000000floats comparés pour
chaque essai avant coût admis ; seules12npy privées regenerables supprimées
ensuite, report/preflight gardés. Pas source/STOP/autreapp/purge/cloud/dépense.
Code/tests/contrat/bench commité et handover avant mesure, ensuite immuables.
Si gain crédible, NEWq24probe nationalentier et FULLaudit avant tout128long.
Un gain numérique ne vaut ni recall ni preuve de latence/généralisation.
Dernier89,0625%exact inchangé ; objectif99,2% initial<=100 reste actif.
