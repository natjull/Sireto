# V132 — CSR national exact nom→propriétaires

Contrat avant mesure,3 octobre2026. Toutes64 partitions officielles V130,
57530682 noms/105564761 faits,toutes preuves/périodes originales V129/V130.
Pas de sélection selon requêtes,CRM,labels,zone,activité,état ou date.

Chaque bucket physique conserve l'ordre original de tous tuples uniques
(nom,parent,owner,subject,scope). Les positions fact locales sont implicites
dans cet ordre,avec préfixe global original dans le manifeste. Tableaux mmap
parentuint32,owneruint64,subject/scopeuint8,current_exactbool et offsets CSR
par nomuint64. Un nom peut pointer vers autant de parents/sites que la source
le dit : aucun cap. Ne jamais matérialiser un produit aliasUL×établissements.
Une enseigne d'établissement reste SITE-locale,une dénomination personne
morale reste attachée à son SIREN. Les owners hors membership candidat
restent intégralement présents,pour les relations futures. Les identifiants
servent de joints/positions,ne sont jamais des features de similarité.

Audit de toutes positions/ranges/tuples/scope/ordre et chaque frontière de
batch. Puis seconde lecture source,reconstruction de CHAQUE nom via CSR et
comparaison EXACTE de CHAQUE champ owner/scope/current,non échantillonnée.
Le booléen current_exact est l'agrégat V130 TRUE explicite seulement ; les
preuves originales UNKNOWN/FALSE/TRUE ne sont pas remplacées ni supprimées.
Tous liens observation et27 champs restent dans les autorités immuables.

80s interne/90s superviseur,3GioRSS,zéro nouveau swapout,sorties128Mio par
bucket/4Gio cumulés. Parties immuables,entrées/sorties/code hashés ; arrêt
au premier échec,pas de reprise ni d'élargissement automatique d'un STOP.
Seal seulement après tous64 buckets terminaux,audits exacts et hashes.

Ce CSR ne vaut ni index riche complet,ni recherche initiale100/recall/SLO,
ni READY_FOR_FOLD1. Pas de fit,embeddings,dense,RRF ou modèles aval modifiés.
Fold0 jamais lu,fold1 fermé ; dernière mesure V10189,0625% exact inchangée.
La recherche doit encore réunir noms/postings,adresses,attributs,géométrie,
relations explicites et collecte initiale100 avant toute nouvelle métrique.
