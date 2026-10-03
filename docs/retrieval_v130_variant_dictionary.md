# V130 — Dictionnaire global de noms et faits propriétaire

Contrat avant exécution,3 octobre2026. Entrée obligatoire : V129 COMPLET
scellé,403577216 lignes source,622697477 observations dont338540853 avec
texte/scope admissible. Les autres observations restent dans leurs autorités
originales, aucun label humain retiré du dénominateur et aucun nom masqué
révélé. Aucun CRM/GT/fold0/fold1/requête/modèle/candidat dans ce compiler.

Plan de TOUTES les553 parties établi avant dérivation. Ordinal d'observation
global uint32 déterministe par ordre V129, sans filtre/reclassement. Tous
les27 champs et les ordinals des sources d'origine restent dans les fichiers
V129 scellés. Pas de copie wide/objet par site ni de perte de provenance.

64 buckets purement physiques hash(name_text)%64, DuckDB1.4.3 épinglé. Le
hash ne sert PAS d'identité/features : égalité et déduplication sur le TEXTE
ENTIER, collisions conservées séparées. Une chaîne est dans exactement un
bucket, dédupliquée entre tous propriétaires/sources/périodes avant tout
éventuel encodage. IDs globaux par prefix des tailles de buckets+ID local,
positions internes uniquement. Les identifiants officiels servent aux
jointures/explications/sorties, jamais aux features de similarité.

Stage pour chaque partie : tous tuples observation/name/parent/owner/subject/
scope/current, avec code courant UNKNOWN0/FALSE1/TRUE2. Vérification de
grain/scope/texte source, puis audit EXACT de tous tuples/ordinals (pas
checksum/sample). Contextes non lexicaux comptés et laissés dans V129,
jamais promus à preuve. Aucune période/enseigne/owner/alias supprimée.

Après seal de TOUTES les parties, compiler chaque bucket depuis TOUTES ses
parties : dictionnaire textes uniques, faits uniques name+parent+owner+
subject+scope, bool current_exact seulement si au moins une observation
explicitement TRUE. Toutes observations/périodes sont gardées via liens
fact→ordinal global ; l'agrégat current ne remplace pas leurs métadonnées.
Audit EXACT de chaque observation/nom/propriétaire, unicité et position de
chaque ID local, toutes références une fois. Pas d'enseigne transférée aux
frères, pas aliasUL×sites ; ni graphe requis pour garder un nom officiel.

Budgets par worker :80s interne/90s superviseur,3GioRSS,zéro nouveau swapout,
DuckDB768Mio/1thread,spill<=2Gio,sorties<=512Mio/phase et32Gio cumulés.
Parties/buckets immuables, arrêt au premier échec, pas élargissement/reprise
automatique de STOP. Entrées/sorties/implémentations hashées. Tous workers
terminaux avant les seals ; aucun seal complet si une partie/bucket manque.

Ce dictionnaire ne vaut PAS index de retrieval, recall/SLO ou READY_FOR_FOLD1.
Le futur encodeur features applique encore les gardes IDs/masques aux textes
normalisés, y compris les rares chiffres de compatibilité Unicode devenus
ASCII après la première normalisation. Les textes V129 ne sont pas réécrits.
Postings words/char3/char4, adresses, graphe et runtime national restent à
compiler/intégrer. Aval gelé ; dernier recall pilote V10189,0625% inchangé.
