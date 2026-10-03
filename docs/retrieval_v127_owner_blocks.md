# V127 — Tous les parents officiels vers leurs blocs physiques

Contrat avant mesure,3 octobre2026. Le score reste celui du moteur de preuves
cohérentes V121 ; aucune recherche ni nouveau modèle dans cette compilation.
V126 couvre déjà toutes43896818 lignes/342945 blocs. Le pont V127 évite
de recopier chaque variante de nom UL pour chacun des sites de son SIREN.

Toutes identités candidates de V96, tous membres et tous blocs V126, sans
filtre d'état/XY/localité/graphe/nom/query/GT. Inverser le layout pour donner
exactement un bloc à chaque site, puis factoriser exactement les couples
distincts (SIREN officiel,bloc physique) en CSR. Garder toutes29906504
entreprises et chaque couple, aucun cap fanout/alias/site. Identifiants comme
jointures seulement ; pas de feature ni de similarité sur leurs chiffres.

La présence d'une entreprise dans plusieurs blocs ne signifie PAS que ses
enseignes d'établissements s'appliquent aux frères. Ce pont sert aux preuves
UL explicitement parent-scoped ; un nom SIRET reste site-scoped et joint au
seul bloc de ce site. Les propriétaires de preuves extérieurs au catalogue
candidat doivent rester dans leurs autorités pour les hypothèses de relation.
Ce pont ne filtre pas leur future source de noms, il indexe les cibles.

Audit bitmap d'inversion complète, dénominateur43,9M, toutes29,9M entreprises
présentes et reconstruction bit-exact de CHAQUE couple distinct à partir
de TOUS les membres d'origine, pas checksum/échantillon. Contexte et
membership originaux V96/V126 immuables. Fermés et XY UNKNOWN conservés.

Budget80s/90s,3GioRSS,zéro nouveau swapout,1Gio sorties. Pas de relance
augmentée automatique. Hash intégral de chaque tableau utilisé et produit,
manifestes sources/implémentations pinés. Aucun scan nouveau ciblé par CRM
ou GT, aucun fold0/fold1/fit/aval/cloud/GPU/coût externe.

Ce résultat est un composant du compiler national, PAS l'index de retrieval
complet : preuves historiques/EI/enseignes et graphe, adresses et activité
doivent encore être rattachés avec leurs scopes/observations. Aucun recall
ni SLO validé ici ; dernier pilote exact@100 V10189,0625% inchangé.
