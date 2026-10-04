# V211 — fermer le moteur terminé avant contrôle fullhash

Protocole avant mesure. Même recette initial100 multicanal V210, implémentée
par `retrieval_structural_channels_v210.py`, contrat210 conservé immutable.
Aucun poids, canal, propriétaire, prédicat, rayon, cap ou score changé.
V210 q24 a produit100 en10,077460458s, mais STOP mémoire pendant le hash
postlecture : ces sorties ne sont ni admises ni évaluées. Ne pas les recycler.

Seule correction : libérer notre propre TantivyBackend et fermer notre
connexion DuckDB après la fin de toute recherche, AVANT rehash complet de
chaque fichier utilisé. gc.collect n'agit que sur les objets de ce processus ;
pas purge globale, cache macOS effacé ou autre application arrêtée. Epoche
original et hashes complets pré/post restent obligatoires. Les résultats
ne sortent du worker qu'après audits et budgetPASS ; exitparent0 réel requis.
Le pic RSS de TOUT le worker, y compris recherche et nettoyage, reste compté.

Nouvelle racine211 indépendante, nouvelle générationq24, code/config/doc/
handover avant mesure puis immuables. Pas retry210 ni preuve/fragment repris.
Même ALL128 inputs avantq24, probe COSTONLY sans permissionGT, pas recall,
pas pilote complet déjà implémenté. Même limites1531Gio/80/query20/quiet30/
p1/free>=50/swapglobalnonattribué, pas mean16 élargi. Fullhash des sources
utilisées, metadata/inventory natifs seulement (pas fullhashsegments), même
limite de43896818sites natifs et catalogue étendu non entièrement intégré.
Exact/allhuman99,2%/OOF/généralisation/sidecar/SLO restent l'objectif, jamais
remplacés par ce probe. Dernier recall reste V10189,0625% sur128 réutilisées.

Les15tests du noyau210 restent valables pour la recette inchangée. Test du
cycle de vie du driver211 vérifie fermeture avant hash, exclusion des modèles
aval/labels et préservation du budget/contrat ; aucun test simulé ne prouve
le RSS national. Si211 STOP, conserver immutable et ne pas lancer128 en le
présentant comme réussi. Une nouvelle demande de budget nécessiterait une
autorisation humaine explicite ; aucune hausse ici.

