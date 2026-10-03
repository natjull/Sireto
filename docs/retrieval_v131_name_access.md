# V131 — Accès complet noms : words/char3/char4/exact-feature

Contrat avant mesure,3 octobre2026. Source V130 entière scellée,57530682 noms
bruts distincts,toutes les preuves/owners/périodes gardées dans V130/V129.
512 rowgroups physiques couvrant exactement tous les noms, sans sélection
CRM/GT/requête/état/zone. Les sources historiques ne sont pas modifiées.

Chaque nom reçoit le texte FEATURE au normaliseur scalaire V118/V121,
notamment nouvelle garde IDs après compatibilité Unicode devenue ASCII.
Tous les noms bruts restent dans la projection feature (texte vide possible,
pas une suppression de source/candidat/label). Chaque texte exact peut
pointer vers plusieurs raw alias_ids, cas d'ID supprimé explicitement testé.
Pour chaque nom,sans échantillon, texte feature compare au scalaire et les
cardinalités de tous quatre ensembles se calculent au scalaire V121.

Postings natifs : mots distincts,grammes3/4 sur espaces remplacées par '_',
exact non vide. Certificat de tous ensembles : unicité de chaque
(alias_id,channel,token), appartenance de chaque token au mot/sous-chaîne/
texte entier, égalité de chaque cardinalité au scalaire. Ces trois propriétés
prouvent chaque ensemble EXACT, pas une comparaison de checksum/sample.
Audit supplémentaire de chaque posting exporté contre la relation native.
Jamais concaténation des sacs de tokens de variantes/owners différents.

64 hashes de tokens physiques seulement, DuckDB1.4.3 épinglé. Puis256 phases
channel×bucket depuis TOUTES leurs parties. Sort streaming en vocabulaire
bytes ASCII/offsetsuint64,CSR post_offsetsuint64,alias IDs globauxuint32.
Pas d'objet Python par terme/site dans le format de serving, pas de copie
Parquet triée supplémentaire. Chaque posting/texte/boundary CSR est
reconstruit EXACTEMENT sur seconde lecture native ordonnée. Aucun cap de
popularité/alias/site/postings ; IDs internes et officiels ne sont jamais
features. Les assertions de range concernent l'encodage, pas des quotas.

Budgets par phase :80s interne/90s superviseur,3GioRSS,zéro nouveau swapout,
1thread DuckDB768Mio,spill<=2Gio,sorties<=512Mio/phase et32Gio cumulés.
Tous handles terminaux avant seals complets ; pas de seal avec groupe ou
channel/bucket manquant, ni reprise/élargissement automatique d'un STOP.
Parties immuables,entrées/sorties/implémentations hashées et vérifiées.

Pas de query/CRM/GT/fold0/fold1,fit/dense/ranker/decider/risk/accepteur.
Même accès aux noms entièrement compilé ne vaut PAS moteur initial100,
index riche complet,recall ou SLO. Adresses,attributs,membership/graphe et
couplage des preuves/scoring natif restent à intégrer. Dernier pilote
V10189,0625% exact inchangé ; pas READY_FOR_FOLD1/garantie99,2%.
