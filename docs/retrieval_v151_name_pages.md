# V151 — Résidence mmap limitée à une requête, pas données limitées

Nouveau contrat avant mesure. V150 STOP au troisième nom sur RSS3273408512
octets>3Gio,26,327550s,swap0. Deux traces partielles ne valident pas les4
requêtes ; root/STOP/code V150 restent IMMUTABLES. Aucun recall ou SLO.

V151 conserve EXACTEMENT mêmes sources V134/V149,4 premiers CRM V115 et
V150 ReboundNames/scatter/snapshot/relations. Le validateur/protocole V150
reste inchangé. Seul le lecteur de pages lexicales est spécialisé : après
chaque calcul COMPLET, coverage/exact sont des vecteurs possédant leurs
octets ; toutes les anciennes mmap de ce lecteur sont fermées puis toutes
rouvertes readonly, même namespace/shape/dtype. Aucun posting/alias/owner
cap, filtrage, quantification, recalcul approximatif ni pool candidat.

Cela libère les pages file-backed du processus, PAS les fichiers, PAS un
cache global ni une application. Les appels publics posting renvoient une
copie qui reste valide après rotation. Reader mono-requête ; tout remap
partiel échoué bloque les requêtes suivantes, jamais un index tronqué.
Inode/size/mtime contrôlés à chaque remap ; SHA complets toujours AVANT et
APRÈS les4 sorties, pas seulement une comparaison de timestamps. Chaque
rotation publie RSS réel avant/après ; aucune baisse mémoire présumée.

Même Budget80s/3GioRSS/ZÉRO swapout global, superviseur90s,quiet30s/max55s/
min50% libre, output CATNAT, root neuf/no retry. PAS garde1Go/pression-only
ni autorisation implicite pour reprendre adresses V144. Pas dépenses/GPU/
purge/app fermée. Sources/hashes et modèles aval inchangés.

Tests de parité bit-exacte avec V138 sur toutes variantes, collisions,
retours possédant leur mémoire/posting indépendant/rotations répétées,
dérive fichier/phase partielle STOP et validateur query/folds V150. Sonde
réelle toujours coûts/comptes de preuves, pas recall/initial100/READY.
