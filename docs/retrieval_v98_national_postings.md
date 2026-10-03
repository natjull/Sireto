# V98 — mots, owners et métadonnées de tous les noms nationaux

Tous57804962 facteurs V97 et162537492 observations originales. Même masque
signal V56/V86 (IDs9/14 retirés de la similarité) et mots d'un SEUL alias,
pas bag inter-alias ni nombre de sources répétées transformé en pertinence.
NULL/vide/context-only conservés comme facteurs et provenance, sans posting
lexical ; source ASCII canonique exigée, pas normaliseur substitutif.

Posting(word,fact) unique, tous mots et tous faits éligibles. Owner site et
personne morale -> TOUS facteurs, CSR sans cap. Toutes dates/source/kind/
current/priority/NULL des162m observations dans mmap, IDs de preuve et texte
brut restent accessibles au vrai ordinal de source V94. Co-validité temporelle
reste UNKNOWN ; présence de metadata n'est pas une feature prétendument utilisée.

Prepare puis32 buckets de stockage mot, owners et metadata, processus
successifs80s/4Gio/zéro swap, supervision90s, contrôleur900s, DuckDB1GB/
spill2GB/deuxthreads, sortie totale10Gio préfixée. Ces32 ne sont jamais des
pools de candidats. Sources/tuples/inverse complets hashés au contrôleur
avant/après et phases consommant tous facteurs ; chaque tri de mot vérifie
son partitionnement source-only hashé. Metadata vérifie source originale
5,5Go avant/après. Les phases ne consommant pas cette source ne prétendent
pas la rehasher. Code vérifié par phase. Aucune lecture CRM/labels/folds0/1,
aucun fit ni modèle aval changé. Aucune mesure recall avant vrai run initial100.
