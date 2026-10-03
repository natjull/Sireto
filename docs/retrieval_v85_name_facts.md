# V85 — Faits de noms séparés et observations intégralement référencées

Préenregistrement source-only avant mesure. V82 contient50097324 observations
officielles scopées, V84 donne les liens à un saut exhaustifs. Ne pas indexer
50 annonces répétées comme50 preuves indépendantes ni créer un bag mélangeant
plusieurs enseignes. Mesurer/construire la représentation factorisée complète
avant de budgéter un index lexical lourd ; pas sélectionner un sous-corpus
autour de CRM/GT et le présenter comme retrieval national.

Tuple de fait EXACT : (subject_kind,owner_id,parent_officiel,lexical_scope,
normalized_value). Pas de hashing probabiliste/collision ou d'approximation
pour le grouping. Owner SIRET ou SIREN selon grain déclaré, parent fourni par
la source et vérifié ; IDs stockage/jointure seulement. Portées différentes
séparées, noms différents jamais réunis en texte. Valeur normalized du source
conservée telle quelle à cette étape : futur index applique impérativement
signal/mask9-14chiffres V56 AVANT tokens/features. Ne pas scorer ces IDs.

Chaque observation source reçoit un ordinal stable du Parquet V82 hashé,
via file_row_number, pas row_number sur un scan parallèle non ordonné.
Mapping observation→fait entier, puis CSR fait→TOUTES observations d'origine.
Les dates,source/kind/priority/état/provenance ne sont PAS effacés : ils restent
dans la source originale V82, et la CSR conserve chaque référence. Ils ne
sont pas encore recopiés dans un mmap serving ; ce manque est publié. Un
fact groupé ne peut pas prendre une date/priorité arbitraire de sa première
observation ; futur score temporel/priorité = MAX sur observations admissibles,
pas mélanger plusieurs preuves de site/nom ou fabriquer une validité.

Plan physique exhaustif : staging16 buckets owner%16, puis DISTINCT exact
sur les cinq colonnes dans chaque bucket. Modulo ID n'est PAS un signal de
matching. Join COMPLET sur ces mêmes cinq colonnes vers chaque observation
originale, mappinguint32 puis inversionstable : contrôle chaque observation
assignée UNE fois, chaque fait référencé, total50097324 inchangé. Context-only
enseigne SIREN conservé comme fait mais pas futur flux lexical admissible.

Guards80s/4Gio/zéro swapout,DuckDB1GB/2threads,spill1GB, output/staging/spill≤2Gio,
supervision90s. Hash complet source avant/après ; sources/implémentations
anciennes inchangées. STOP conservé sans élargissement si le plan est trop
coûteux. Pas suppression du staging dans ce premier résultat scellé : compter
ses bytes et hashes. Pas modèle/index appris ou téléchargé, pas dépenses,
CRM/labels/folds0/1 non ouverts, aval gelé. Dernier vrai recall reste114/128=
89,0625%, objectif initial10099,2% exact/all-human/OOF/SLO/généralisation intact.

Tests SQL réel Parquet + oracle de tuples/ordinals : mêmes noms différents
propriétaires/portées restent différents faits ; observations répétées, dates
différentes, source/kind/provenance conservés via tous leurs ordinals ; aucun
name bag fabriqué, zéro source positive injectée en évaluation. Cette
factorisation et son ratio ne prouvent pas un gain de rappel ou un SLO.
