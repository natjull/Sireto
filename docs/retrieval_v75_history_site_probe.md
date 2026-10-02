# V75 — Anciennes enseignes SIRENE, recherche initiale fédérée, avant mesure

Anomalie vérifiée : l'ancien overlay SIRENE history
`official_evidence_overlay_sirene_history_v1/5bb04044578179db` expose réellement
zéro document, malgré100 partitions officielles de projection disponibles.
Ne pas modifier cet artefact historique ni conclure que tout le retrieval
antérieur n'utilisait aucun nom historique : le principal contient d'autres
preuves historiques RNE/BODACC. Cet essai ajoute UNIQUEMENT les noms de sites
SIRENE disponibles dans la projection nationale indépendante de tout CRM.

Avant mesure :100 préfixes00..99,273796 observations d'identité historique
d'établissement,21,4Mo de Parquets, aucune sélection par requête/label/positif.
Vérifier manifeste labels/synthétiques false, source SIRENE identique, préfixe,
compte et hashes COMPLETS des petits Parquets avant/après. Conserver TOUTES
observations de ces fichiers et leur période, sans nouveau cap d'alias/recency.
Les caps/exclusions de la projection préexistante restent une limite héritée.
Noms d'UL historiques4025541 et successions9727541 NON intégrés ici ; ce n'est
pas le catalogue complet de tous les noms passés de toutes les entreprises.

Un document par SIRET avec noms historiques, provenance/périodes et géographie
COURANTE jointe par la projection. Cette géographie n'est PAS une adresse
historique, et n'établit pas de période commune avec le nom ancien. Adresses,
numéros, CP/INSEE/siege restent des signaux de routage courant explicitement
nommés scaffold. Aucune adresse reconstruite pour une ancienne période ;
aucun filtre par état/date/commune. Contradiction de scaffold => STOP.

Hypothèse FIXÉE : transférer les11 coefficients OOF V57 inchangés à la source
historique, remplaçant noms courants par noms de sites historiques, et les
preuves d'adresse par le scaffold courant. Exact nom signifie ici alias
officiel historique égal, pas autorisation AUTO/exact_protection. Les autres
vues restent couverture words/ngrams + bonus lieu/numéro/siege sans BM25.
Addition de nom ancien et scaffold courant ne prétend pas une preuve de
coexistence temporelle. Champs agrégés pas le kernel cohérent V72–V74.
Pas de nouvelle grille, fit, normalisation par source ou poids alpha réglé.

Score initial final=max(source actuelle V57,source historique), collecteur de
frontières natives ordonnées V40 jusqu'à100 SIRET distincts, aucune profondeur
fixe/réservoir/rescoring/reranker. Chaque source retourne uniquement ses scores
finaux. Les égalités sont celles des moteurs natifs, pas un top100 canonique
par ID prouvé pour toutes les égalités. IDs sortie/jointure/ties seulement.

Build borné80s/6Gio/zéro swap,128Mo heap writer/2 threads, sans grosses sources
relues ni embeddings. STOP si budget dépassé, aucune relance élargie.
Puis128 requêtes CRM query-only dans l'ordre hashé V34 déjà consommé, folds234,
modèles V57 scellés et disjoints des query_ids d'évaluation. Même registre OOF
entités du trainer. Batches8 supervisés90s/80s, contrôle moyen<=4s sur16 avant
suite, total360s. Coût gate expérimental, PAS SLO. Modèles aval toujours gelés.

Après scellement de TOUTES128 sorties, processus évaluateur séparé V49 avec
baseline V57 préfixée ; mêmes ledger/sidecar, all-human, exact/opérationnel,
segments historique/V2/V3/folds, Wilson, absences, gains/pertes appariées.
Anciennes sorties/labels jamais ouverts en génération ni construction. Fold0
non lu/fold1 fermé. Ce pilote sur corpus actuel national + overlay global de
sites peut donner PIVOT_GENERATION/STOP, jamais READY_FOR_FOLD1, même à100%.
99,2% exact, confiance, folds, généralisation, SLO et attribution exhaustive
restent nécessaires sur OOF complet. Ne pas substituer ce volet à l'objectif.
