# V91 — buffer d'accès natif, aucune admission512

Même métrique V87, mêmes champs/scopes/poids64/grades/fiabilité/queries,
same typed joins V90 et bornes V89. V90 a16 recherches certifiées mais
gate coût STOP : moyenne4,731922s>4s, zéro swap, pics4224106496/4841897984.
Le cas le plus lent17,314s comprend75 pointqueries natives13,085s pour7112
targets, natif trié3,948s ; aucune lecture de labels/recall partiel.

Changer seulement la taille du buffer de complétion aléatoire100→512,
flush de doublons256→4096. Chaque target est traité et chaque fanout ENTIER
reste parcouru avant changement de frontière.512 est un lot de traitement
de références légères, PAS pool/admission500/512 ni top512 avant un ranker.
Le heap de sortie reste100 absolu, profondeur/source exhaustive, aucun
target source retiré, aucune hydratation métier riche. Garde deadline dans
fanout et avant/après callbacks inchangée, aucune sortie noncertifiée.

Tests oracle graphes many-to-many/fanout2501/duplications/ties/timeouts et
adaptateurs réels, parité dtype/scope et aucun grade dépassant SON fait.
Budgets et séquence identiques :20squery/4smean16/80sworker6Gio/zéro swap/
90s supervisor/900s contrôleur. Après16 sans labels,128 aveugles seulement
si gate passe, puis évaluation séparée. Pas READY_FOLD1 ni garantie SOTA.
