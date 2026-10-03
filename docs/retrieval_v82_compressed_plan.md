# V82 — Compression physique sans réduction des preuves

Préenregistrement après STOP_OUTPUT_SIZE de la projection `_a` : les deux
Parquets complets occupent2173183735octets, soit25700087octets au-dessus de
la garde2Gio (hors petits logs). Aucun manifeste complet n'a été publié ;
le résultat STOP et ses sources/code/config restent immuables.

Changer UNIQUEMENT le niveau de compression ZSTD du COPY Parquet, niveau6.
Mêmes colonnes/types/valeurs, SELECT/SEMI JOIN, duplications d'observation,
dates/provenances, scopes et flags. Aucun cap, filtre supplémentaire, alias
transféré, nom supprimé ni budget augmenté. Les12,5m endpoints et50,1m noms
de la tentative STOP ne sont pas des seeds/positifs ou un pool produit.

Tests de parité multiset et schéma avec le compilateur logique, dont dates,
sources, doublons et enseignes SIREN context-only. Gardes source80s/4Gio,
zéro swapout, DuckDB1GB/2threads, spill1GB, totaloutput2Gio, supervision90s
inchangées. Si compression ne suffit pas ou devient trop lente : STOP.

Pas de CRM/label/fold0/1, pas de rappel/SLO mesuré, pas de modèle ni index.
Dernière mesure recall initial100 reste V78,114/128=89,0625%. L'objectif
99,2% exact initial100 et validation OOF/généralisation restent à réaliser.
