# V185 — corriger le raccordement, préserver le STOP184

Protocole avant mesure, 4 octobre 2026. V184 est arrêté sur AttributeError
RotatedKeys.remap AVANT tout score/candidat, pas sur la mémoire ou la qualité.
Tous ses fichiers restent immuables. Nouvelle version : la méthode statique
remap170 manque au wrapper ; elle est explicitement raccordée. Aucun score,
poids, source, normalisation, feature ou budget n'est changé.

V185 reprend UNIQUEMENT les phases noms et adresses184 réellement réussies,
hashes de launch/manifests fixés, vrais parent_exit0/report153, epochs et
artefacts vérifiés via le validateur ORIGINAL184. Configuration commune,
source/query/model/BAN identiques, code original hashé ; aucun candidat
ni fichier du score STOP n'est importé. Le root184 entier reste STOP, jamais
requalifié comme complet. Les vecteurs seuls restent readonly et rehashés.

Trois groupes physiques complets puis réduction100, mêmes kernels183/182/
181/180 et collecteur146 sans rescore. V185 dans un root fresh, pas retry184.
Temps global de worker = les deux preuves184 + tous les workers185, publié ;
temps du nouveau run séparé, aucune latence artificiellement réduite en
omettant les préfixes. Chaque worker 1Gio/80s/pression1/free50/quiet30 sous
V153/V164, contrôle de toutes les sources avant/après, Mac/CATNAT seuls.

Une requête ne mesure aucun recall. Tous les labels/folds0/1 et candidats
historiques fermés, aval gelé. Objectif initial10099,2% exact all-human,
fullOOF/Wilson/généralisation/SLO1s/2s restent non certifiés. Le pilote128
ne pourra être évalué qu'après toutes ses sorties scellées, jamais partiel.
