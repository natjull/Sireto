# V86 — Postings compacts de noms, grain et périodes accessibles

Préenregistrement source-only après V85 COMPLETE :14162541 faits de noms
pour50097324 observations, dont90441 faits context-only SIREN SIGN. Tous
faits V85 sont canoniques ASCII uppercase [A-Z0-9] avec espaces simples,
constaté sur le corpus ENTIER. Réaffirmer sinon STOP, pas d'approximation de
normalisation Unicode. À ce stade, ne pas lancer un index3-5grams de14m faits
sans budget ; construire d'abord l'accès entier WORD par fait et propriétaire.
La voie native43,9m conserve ses ngrams ; l'accès mots aux noms relationnels
ne prétend pas couvrir déjà fautes de frappe/fuzzy, sigles ou SOTA.

Masque IDs9/14 avant split/features, équivalent V56 signal sur ce domaine
canonique : regexp digits avec bornes nonnumeric, deux passages pour traiter
les nombres adjacents partageant un séparateur, trim/collapse espaces.
Tests aléatoires/contrefactuels sur runs numériques dans des mots et chaînes
adjacentes ;18digits nonidentifiant ne doit pas être découpé en9+9 arbitraire.
Ne jamais indexer owner/parent/factID comme termes lexicaux. Empty après
masque nonservable en lexical, mais faits/observations conservés V85.

FULL KEY (token,factID) postings uniques par UN nom. Context-only scope0
nonservable en lexical, enregistré séparément, jamais choisi autour de GT.
COPY32 shards par hash(token)%32 puis tri token/factID, vocabulaire exact
et CSR disque. Le hash n'est qu'un routage de stockage, pas une feature ou
clé probabiliste du mot. Pas topK aliases, noms ou parents ; toutes correspondances
positives de tokens parcourues pour la même métrique initiale. Dot product
query nonnégatif pondéré sur14,2m faits, pas reranking d'un pool fixe SIRET.
Score par nom séparé, MAX futur entre noms/chemins, pas addition de annonces.
Poids de qualité et facteurs V83 nationaux restent à figer/apprendre OOF,
aucune promotion selon pertes V78. Ce build n'entraîne rien.

Accès inverse owner SIREN/SIRET→TOUS ses faits, champs grain/parent/scope
mmap, pas de collisions entre SIREN et SIRET à zéros initiaux. Source IDs
et ordinals ne sont pas features. Metadata mmap sur TOUS50097324 records
V82 : trois dates jour32(NULL minimum, pas jour0),source/kind/current
UNKNOWN/FALSE/TRUE/priorité+flagknown. Provenance complète textuelle et raw
restent dans V82, ordinals CSR V85 incluent chaque répétition/période. Le
serving doit vérifier scope/date/site dans les preuves et ne pas affirmer
identité/co-validité sur une succession ou rôle non vérifié.

Guards80s/4Gio/zéro swapout,DuckDB1GB/2threads/spill1GB,totaloutput2Gio,
supervision90s. Hash complet de chaque source utilisée avant/après. Fichiers
de staging ET tri conservés/countés/hashés, pas de suppression implicite.
STOP sans augmenter limites si nécessaire. Tests réels SQL/NP/Parquet vs
score exhaustif indépendant : ID mask, scope0, répétitions de mots, aliases
séparés, zéros initiaux, dates/NULL/priorityunknown. Pas CRM/labels/folds0/1,
pas dépense, aval gelé. Dernier recall reste114/128=89,0625%, objectif
99,2% exact initial100/OOF/généralisation/SLO intact et NON validé ici.
