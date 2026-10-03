# V84 — Accès inversé exhaustif aux successions et propriétaires de noms

Préenregistrement source/index physique, pas recherche évaluée. Kernel V83
nécessite une énumération complète des endpoints par nom, pas un seed de
parents ni une jointure nation-wide relancée pour chaque observation lue.
Compiler7512820 observations V81 en incidence CSR bidirectionnelle
15025640refs. Doublons/date/source conservés, row index0-based au Parquet
V81 hashé = provenance stable, pas autant d'événements uniques revendiqués.

Endpoints triés par SIRET, parents officiels séparément vérifiés, offsets
vers observations ; accès inversé parent→TOUS ses endpoints réellement
présents dans le graphe. Les IDs servent exclusivement au stockage/lookup,
jamais features lexicales. Deux vues d'une même observation, pas deux
supports indépendants. Un seul saut ; pas de fermeture transitive, pas de
nom transféré, pas d'exclusion des fermés ou des emplacements différents.

Dates effectives int32 jours et observation source int64 microsecondes,
NULL représenté par le minimum de chaque dtype, distinct du jour0. États
A/F/inconnu/autre, transfert siège unknown/false/true,3 grades V81 conservés
explicitement. Les chaînes originales et les adresses restent disponibles
dans chaque observation V81 via son ordinal et hash ; pas de preuve de
co-validité historique inventée. Ne pas traiter un grade actuel comme une
équivalence ou une admission automatique.

Lookup nom : enseigne/usage SIRET seulement à son endpoint ET parent déclaré
officiel ; nom légal/commercial/usuel SIREN vers endpoints de CE parent
seulement. Enseigne SIREN/kind inconnu context-only refusé en entrée lexicale.
Les résultats du lookup sont des relations brutes, pas des alias/candidats
admis : adaptateur query-side V83 doit vérifier dates/portée/site et garder
les observations/provenances séparées. Source-SIREN légal reste preuve
de personne morale, pas enseigne universelle de tous établissements.

Guards80s/4Gio/zéro swapout, output≤1Gio, supervision90s, pas de DuckDB/spill,
pas de cloud. Hash intégral des sources utilisées avant/après ; égalité de
TOUS les12531976 couples SIRET/SIREN avec endpoint_scope V82. Aucune source
choisie via CRM/labels. Mêmes6087070 parents attendus. Tests d'énumération
vs oracle exhaustif (doublons/directions/dates/états/zeros), owner scopes,
conflits et sources illégitimes. L'index conserve TOUS les records V81.

Ce CSR ne score pas encore les50097324 observations lexicales de V82 et ne
valide aucun recall ni SLO. Les43896818 sites de la voie native demeurent
nécessaires : ce graphe est une branche complémentaire, jamais corpus de
substitution ou filtre GT. Dernier recall114/128=89,0625%, objectif initial100
99,2% exact/all-human/OOF/généralisation inchangé, fold0/1 non ouverts, aval
gelé. STOP conservé sans agrandissement automatique de budget si nécessaire.
