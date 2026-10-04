# V184 — première sortie nationale intégrée, une requête sans labels

Protocole avant mesure, 4 octobre 2026. Première requête du panel gelé V115,
ordre préexistant indépendant de la vérité, après validation de TOUTES les
128 requêtes typées OOF2/3/4. Ni GT ni anciennes sorties ouverts. Ce test de
coût/intégration ne mesure aucun recall, ne remplace pas le pilote128 et ne
donne jamais READY_FOR_FOLD1. Sortie initiale absolue <=100, avant aval gelé.

Sources entières hashées : noms134/joins149/catalog141, adresses177/faits160
via181, attributs/XY139 via180. Tous les sites44403034, alias34346446 et
faits98804872 restent dans le calcul. Même30poids cross-fit gelés114/115,
même BAN45/46 et prior de descriptions d'activité113/114. Aucune retune.
Numéro par égalité de dictionnaire182, parité146 ; kernels183 inchangés.

Workers successifs : noms+flattenTOUSsites ; adresseTOUSalias ; trois groupes
physiques contigus de CSR160 ; réduction100. Les sorties locales sont des
collecteurs exacts du même score sur chaque feuille, PAS un pool2000/500 ni
un pré-ranker. Aucun rescore à la réduction. Publication finale seulement
après chaque worker réussi, ressources et parent_exit0 réellement confirmés,
source namespace/epochs/hashes et recette identiques, dénominateurs complets.

Chaque worker reste V153/V1641Gio/80s/pression1/free>=50%. Quiet30s avant
chaque phase. Budgets de diagnostic : preuves20s, score parpartition30s,
ce sont des budgets physiques locaux, PAS des SLO globaux relâchés. Le temps
global de calcul publié additionne toutes les phases ; les sources chargées
et contrôles compris sont publiés séparément. Pas de résultat partiel après
STOP ni retry du même root. Les fichiers de preuve scellés restent readonly,
pas de purge globale, autre application tuée, cloud ou dépense.

Les noms et adresses sont contrôlés par hash avant/après leur génération,
les sources lues par chaque partition par hash avant/après son score, tous
les epochs des générations recontrôlés avant/après l'assemblage final. Les
snapshots de requête sont rehashés après lecture et avant sortie. La chaîne
cross-fit114 est vérifiée via le modèle/manifeste gelé et les paramètres, sans
suivre les références de preuves d'entraînement ou lire les labels.

Cette exécution n'est pas un test froid, cacheUNCONTROLLED. Les SLO1s/2s
restent actifs et non certifiés. Après une sortie complète il faudra générer
TOUTES128 sorties avant évaluation séparée, puis traiter les pertes sans
changer le dénominateur. FullOOF/Wilson95bas>=99,0/fold>=98,8 et généralisation
obligatoires pour l'objectif99,2% exact all-human ; exact/opérationnel séparés.
