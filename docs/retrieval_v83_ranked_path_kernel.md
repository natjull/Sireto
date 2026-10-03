# V83 — Recherche initiale de sites via jointure classée à un saut

Kernel d'exécution, PAS encore recette nationale/recall. V82 a désormais
conservé toutes les50097324 observations de noms de12531976 endpoints et
leurs6087070 personnes morales. Le kernel permet leur exploitation sans
réservoir fixe ni expansion depuis des parents déjà choisis lexicalement.

Une preuve de nom scopée conduit par un chemin officiel conditionnel vers
un site cible. Score initial d'un SIRET t :
`max(native_gelé(t), poids * max_nom_chemin_admissible(t) * preuve_site(t))`.
Les deux facteurs relationnels sont nonnégatifs bornés ; leur produit demande
des preuves de nom ET de site. MAX sur chemins/observations, jamais addition
de50 annonces répétées. Le nom du prédécesseur n'est pas un alias inventé du
successeur, et un rôle RNE context-only n'est pas une relation admissible.
Le choix de facteurs/poids nationaux reste à apprendre/valider OOF ; ce
kernel ne promeut PAS un poids empirique selon les14 pertes du pilote V78.

Trois flux exhaustifs triés : score natif complet, noms scopés, preuve site.
Chaque nom lu énumère TOUS ses endpoints éligibles (pas topN sites). Chaque
site découvert est complété par accès indexé au score natif et au MAX sur
TOUS ses chemins entrants admissibles, même non encore lus dans le flux.
Cet accès fait partie de la même métrique initiale, pas d'un score aval sur
un pool fixe. Seul le heap classé a100 groupes, pages/completion100..4096
comme unités physiques, sans profondeur maximale du corpus. La table de
groupes déjà vus n'est pas plafonnée artificiellement : gardes RSS/temps
externes obligatoires pour le futur adaptateur national.

Pour un target jamais découvert : ni son site ni aucun nom qui mène à lui
n'ont été consommés. Donc son score relationnel est au plus
`poids * frontière_nom * frontière_site` ; borne native distincte, borne
globale=max des deux. Avancer la frontière de nom seulement APRÈS traitement
de TOUT son fanout. Arrêter quand100 scores dominent STRICTEMENT la borne,
ou borne0 (preuves positives uniquement). Ties canoniques globaux, parfois
coûteux : pas de proclamation p95/SLO sur ce kernel. Timeout retourne STOP
avec trace, aucune sortie noncertifiée. Erreurs d'ordre/range/membership ou
couverture de random-access échouent fermées.

Portée des responsabilités : les adaptateurs doivent prouver exhaustivité
des flux/adjonctions, appartenance officielle SIREN/SIRET, portée des noms,
grade relationnel<=1, dates/visibilité et compatibilité de site REQUÊTE-side.
Pas de filtration dure de tous les sites à distance/commune inconnue dans
la voie native43,9m, ni fusion transitive de composantes. Le certificat du
kernel est conditionnel à ces responsabilités ; il ne les démontre pas.
Source V81 distingue adresse actuelle compatible, différente/incomplète,
et n'affirme jamais co-validité historique. BAN fiable/faible/manquante
reste distinct, les sources de site/adresse peuvent fournir d'autres vues.

Appui primaire : [Deep, Hu, Koutris, ranked joins with projections](https://arxiv.org/html/2201.05566v2)
explique pourquoi matérialiser puis dédupliquer tous les chemins est coûteux
et pourquoi classer des preuves avant projection ne suffit pas. Notre
seuil/fanout/random-access est un kernel propre, pas reproduction de leur
algorithme ni emprunt de ses garanties de délai ou gains expérimentaux.

Tests : oracle exhaustif indépendant sur graphes many-to-many, observations
répétées, fanout>2000, target provenant seulement du graphe, ties, bornes,
timeout DANS expansion. Fixtures algorithmiques uniquement : aucun corpus
synthétique appris ni label positif injecté en mesure CRM. Pas d'ouverture
fold0/1, aval inchangé. Dernier vrai recall reste114/128=89,0625% ; objectif
initial10099,2% exact, OOF complet/généralisation/SLO NON atteint.
