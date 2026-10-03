# V76 — Jointure initiale des noms légaux historiques

Contrat avant mesure. Objectif final inchangé : recherche initiale nationale
≤100 SIRET, recall humain exact≥99,2%, avant tout modèle aval. Aucun pool fixe
de parents ou sites, aucun reranking des références découvertes.

Ajouter à V75 une vue : égalité du nom CRM normalisé (signal V56, identifiants
retirés) avec un alias légal SIRENE historique officiel. Ce prédicat retourne
TOUS les SIREN correspondants, sans LIMIT ni filtre commune/CP/état. Les IDs
servent seulement à la jointure officielle vers les documents SIRET nationaux.
Cette vue est volontairement exacte sur le nom historique : le flou est hors
de cette hypothèse, pas prétendu intégré. Le nom courant/flou V75 reste intact.

Dans cette vue, preuve de nom=1 et alias exact=1. Les autres9 composantes V57
proviennent du même index national et des mêmes requêtes physiques V56 ; les
11 poids OOF V57 sont inchangés. La requête native retourne directement les
sites triés par leur score final. Fusion MAX des3 fronts jusqu'à100 SIRET
distincts, sans score post-découverte ni nombre fixe d'entreprises sélectionné.
Le plafond porte sur les SIRET émis, pas sur le nombre de liens officiels
satisfaisant le prédicat de jointure. Les états fermés restent admissibles.

Source construite sans CRM/teacher/labels/candidats : les100 projections
historiques UL préexistantes00..99,4025541 observations. Déduplication
(signal du nom,SIREN) pour l'égalité, provenance/périodes conservées dans les
fichiers sources entièrement hashés. Les géographies répétées ne limitent PAS
la recherche des sites. Projections amont excluent les noms courants et les
parents sans INSEE courant ; cette limite est héritée, jamais exhaustivité
revendiquée. Alias légal ancien→membres actuels du même SIREN ne prouve pas
coexistence temporelle. Aucune adresse historique inventée ni traversée de
succession/filiation/BODACC ajoutée. La source SITE V75 est réutilisée scellée.

Construction bornée80s/6Gio/zéro swapout, supervision90s. Génération ensuite
bornée360s, travailleurs80s/6Gio/zéro swapout, supervision90s, threads2.
Gate coût16 mean≤4s, requête≤20s ; STOP sans évaluation si population128
incomplète. Budgets séparés figés pour la nouvelle construction et génération,
pas de relance élargie. Aucun cloud/dépense/téléchargement ni modèle aval changé.

128 mêmes requêtes CRM seules, ordre V34 hashé, disjointes TRAIN512,
folds2/3/4. Sources/implémentations/inventaire scellés avant génération.
Labels et sorties V75 ouverts seulement après128 sorties intégrales scellées,
évaluateur séparé V49, ledger/sidecar all-human. Exact/opérationnel/Wilson,
segments historiques/V2/V3, gains/pertes, latence/caps publiés séparément.
Fold0 jamais lu, fold1 fermé. Pilote de développement réutilisé : ne valide
ni OOF complet23k ni généralisation ni SLO1s/2s ni READY_FOR_FOLD1, même à100%.
Pas de garantie SOTA. Les contrats/résultats antérieurs restent immuables.
