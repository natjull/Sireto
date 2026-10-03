# V81 — Compilation nationale des rattachements de succession

Préenregistrement source-only, objectif initial100 exact99,2% inchangé.
V80 a mesuré64 paires diagnostiques ; elles ne servent PAS de corpus ici.
Lire TOUT le stock9730972 lignes et catalogue43896818 sites officiels.
Identifiants uniquement clés de jointure, provenance/groupement/sortie.

Conserver toute observation à continuité économique explicitement vraie,
endpoints14 chiffres distincts, ET deux établissements officiellement présents.
Les états fermés ne sont jamais écartés. Dates et indicateur transfert de
siège restent des preuves distinctes, pas de préférence pour l'actif.
Chaque observation raw conservée : ne pas prétendre autant d'événements
uniques. Les classes de couverture couvrent TOUTES les lignes d'entrée,
y compris continuité fausse/inconnue, autosuccession et endpoints absents.
Les absences de preuve ne peuvent jamais filtrer le dénominateur humain.

Pour chaque paire conserver les deux SIREN officiels, les deux adresses
COURANTES, CP/INSEE, états et numéros. Une classe distingue concordance
d'adresse courante nonvide + géographie concordante, différence connue,
ou preuve incomplète. Les paires à adresses différentes restent stockées,
pas de propagation automatique de nom ni d'assimilation d'identité. Même
adresse courante ne prouve PAS une coexistence historique à la date du lien.
Pas de site frère ajouté, pas de fermeture transitive, UN saut uniquement.

Sortie Parquet nationale immuable de rattachements, aucune sélection CRM,
label/teacher/candidat antérieur. Les noms ne sont PAS encore transférés
dans cette étape ; prochaine recherche devra retrouver les noms officiels
scopés à l'endpoint ou sa personne morale, sans propager enseignes aux frères.
Ces fichiers seront un index de jointure, pas un pool candidat fixe à
reranker ; le futur score initial devra être borné globalement jusqu'à100.

Compilation physique : IDs convertis en entiers UNIQUEMENT pour jointures
étroites, endpoints dédupliqués avant le scan du catalogue, champs source
indispensables seulement, tables libérées après leurs dernières utilisations.
DuckDB1GB/2threads, spill1GB sous output CATNAT, résultat+spill≤2Gio,
worker80s/4Gio/zéro swapout, superviseur90s. STOP si garde dépassée, aucune
relance élargie automatique. Sources complètes SHA avant/après, source-count
et catalogue vérifiés. Aucune dépense, GPU loué, fold0/1, fit ou modèle aval.
La compilation ne mesure NI recall ni SLO et ne peut donner READY_FOR_FOLD1.
