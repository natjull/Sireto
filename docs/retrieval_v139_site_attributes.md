# V139 — Attributs de site officiels complets et coordonnées réutilisées

Compléter les preuves de noms V138 par CP, commune INSEE et numéro de voie
de TOUS43896818 sites du stock courant, fermés/masqués/géo inconnue compris.
Exact joint SIRET/SIREN typé vers positions current V96, chaque ligne une
fois. Identités joins seulement, jamais scores. Aucun cap de CP/département,
numéro/état/fanout, ni candidat injecté ou liste de seeds.

CP/INSEE : autorité scalaire V121 `public_code` sur CHAQUE chaîne distincte
source, y compris Unicode/.strip/.upper. Encodage base36 bijectif5 chars
dans uint32, sentinel4294967295 UNKNOWN ;00000 n'est pas NULL, leading zeros
et2A/2B conservés. Codes utilisés uniquement pour ÉGALITÉ query-site, pas
valeur ordinale comme feature. Numéros : normalisation V118/V128 inchangée,
dictionnaire texte SOURCE unique sans troncature ; code0=texte vide, aucune
inférence depuis masques/voisins/nom d'entreprise/GT. Code numérique opaque
pour égalité uniquement, aucun alias CRM appris.

Réutiliser arrays nationaux officiels V110 HQ/état/dates, V112 nomenclature
et code d'activité originaux + descriptions officielles, V126 XY L93
float64 inchangés. Pas de nouveau géocodage Sirene ni modèle appris.
UNE classe descriptive par couple explicite(nomenclature,code), UNKNOWN=-1
sans exclusion/NAP deviné. La classe ne prédit pas d'activité CRM ici.
Métropole seule dans géométrie V126 validée, DOM/inconnus retenus sans XY.
DateDebut reste début de période courante, pas date inventée de fermeture,
et aucun vieux nom/adresse ne reçoit un faux XY historique.

Construction puis processus AUDIT indépendant : deuxième lecture complète
stock original, reconstruction de chaque CP/INSEE/numéro exporté. Contrôler
aussi CHAQUE HQ/état/date réutilisé contre RAW stock, pas seulement le dossier
dérivé (un coalesce NULL pourrait autrement tromper). Sources utilisées et
sorties entièrement rehashées avant/après phases, toutes au seal.

Worker80s interne/90s superviseur/3GioRSS/zéro swapout PAR phase, deux
processus successifs build/audit, sorties768Mio max. Toute dérive/incomplétude/
budget impose STOP sans relance du même root ni relever garde. Reader ouvrant
une fois, preuves structurées en vecteur pour seuls sites visités, autres
426471 sites V136 UNKNOWN (pas héritage des propriétés d'un autre SIRET).

Pas de CRM/label/fold0/fold1/model fit/push/dépense. Pas encore moteur
national initial100, adresses textuelles à raccorder. Ce milestone ne peut
revendiquer gain de recall/SLO/READY_FOR_FOLD1. Les métriques antérieures
restent immuables ; replay128 seulement après sortie initial100 intégrée,
sorties scellées avant labels, puis all-human OOF et tous gates du contrat.
