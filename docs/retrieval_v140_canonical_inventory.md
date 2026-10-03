# V140 — Identités officielles de TOUS noms et adresses, même sans texte

Avant d'indexer les adresses, vérifier TOUTES162537492 observations de noms
et142096607 observations d'adresses canoniques originales. V135 avait lu
seulement les facts-sites nominaux lexicalement éligibles V132 ; une preuve
de nom vide/masquée ou une adresse explicite peut avoir un autre SIRET.
Ne pas sélectionner sources selon CRM/GT/ancien hit/texte/masque/état/date.

Grain explicite : subject_kind=SIRET, SIREN ASCII9 et SIRET ASCII14 de même
préfixe, leading zeros préservés. SIREN explicites = contexte parent, pas
site fictif. Tous autres grains = contexte non vérifié, publié dans comptes,
pas disparition d'observation. Les données d'origine et leurs champs restent
dans fichiers sources entièrement hashés ; seuls les trois champs d'identité
sont lus pour CET inventaire. Il ne projette pas encore les adresses.

Comparer tous SIRET explicites au namespace V136 current+extra. Reconstituer
CHAQUE observation admise via ID-position, publier toutes références et
identités absentes uniques, puis union complète sources. Aucun lookup GT ni
condition de contenu lexical. Tous sites masqués restent comptés. Les SIREN
d'adresses ne sont jamais propagés aux sites. Pas de nouvelle feature ni
preuve physique inventée. Identifiants joins/audit seulement, pas features.

Deux workers séquentiels80s internes/90s superviseur/3Gio/zéro swapout,
sorties64Mio max. Source originale et dictionnaires d'ID entièrement
rehashés avant/après chaque worker ; toutes sorties au seal. STOP si budget/
dérive/incomplétude, pas retry au même root ni garde relevée. Pas de CRM/GT/
fold0/fold1/fit/aval/push/dépense/recall. Catalogue candidat inchangé par la
sonde : si nouveaux IDs, prochaine version doit les admettre depuis sources
seules, en conservant positions précédentes et tous attributs manquants
UNKNOWN. Puis index d'adresses site-scopées et recherche directe100, jamais
pool fixe reranké. Aucune hausse de recall ni READY revendiquée ici.
