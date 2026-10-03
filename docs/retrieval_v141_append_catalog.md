# V141 — Admettre tous les sites officiels V140 sans déplacer les précédents

V140 a trouvé79745 SIRET officiels supplémentaires, indépendamment du CRM et
des labels, dans les observations d'adresses. Les admettre tous AVANT
l'indexation des adresses. Conserver TOUTES positions V136, y compris les
426471 sites déjà ajoutés par V135 et leurs parents : ne pas trier ensemble
anciens et nouveaux extras. Les preuves noms/graphe V137 restent valides.

Troisième segment de clés triées, append uniquement, lookup typé exact.
Nouveaux parents = ceux explicitement déduits du préfixe officiel SIRET et
absents des deux segments précédents. CSR parent→membres/blocs complet,
blocs physiques <=128 sites, jamais plafond candidats ou fanout. Conserver
les leading zeros à la restitution. Identifiants = joins/sorties/ties,
jamais features. Aucun alias×membre matérialisé.

Reconstituer CHAQUE nouvelle identité, parent, tuple CSR et tableau exporté.
Vérifier CHAQUE position précédente de site ET parent à travers le nouveau
lookup. Sources et sorties entièrement hashées. Champs/XY/état/HQ/activité
des nouveaux sites restent UNKNOWN ; aucune preuve de site héritée de leur
parent ou relation. Les adresses explicites seront indexées séparément,
pas utilisées ici pour attribuer des coordonnées ou une date de fermeture.

Worker80s internes/90s superviseur,3GioRSS,zéro nouveau swapout, sorties16Mio.
STOP si dérive/incomplétude/coût, pas retry au même root ni garde relevée.
Pas de CRM/GT/fold0/fold1/fit/aval/push/dépense. Ce catalogue n'est PAS un
moteur initial100 ni une preuve de hausse de recall ou de SLO. Après cette
étape, index d'adresses site-scopées et collecteur natif100 restent requis.
