# V322–326 — première mesure élargie de la recherche initiale à 100

Préenregistrement du 5 octobre 2026 après l'accord « Go » à la proposition
d'un panel DEV plus large et distinct des 128. Aucun nouvel essai des STOP
317/320, aucun relâchement mémoire, aucun changement de score.

V322 parcourt toutes les 23 571 requêtes CRM folds 2/3/4 du replay V34 scellé,
mais lit exclusivement les fichiers `queries.parquet` et `ban.parquet`.
Les manifests sont des références, leurs anciens candidats ne sont pas lus.
Toutes les lignes et les doublons sont contrôlés avant la sélection. Les 16
labels historiques inexploitables restent des absences à publier, pas des
échecs que l'on supprimerait du nouveau dénominateur.

Population d'échantillonnage : les 23 571 moins les 128 requêtes du pilote
initial100 consommé et les 512 contrôles TRAIN du bundle V77 (union de ses
trois listes de membres). Les deux exclusions doivent être disjointes et
faire exactement 640 contrôles. Aucun statut, nom, qualité BAN, ancien hit,
SIREN/SIRET humain ou difficulté ne peut intervenir dans la sélection.
Exactement 2 000 requêtes : premières priorités SHA256 de
`initial100-v322-20261005-first-large-panel|query_id`, triées avec query_id
comme départage. Le seed et la taille ne seront pas rechoisis après mesure.
Tous les folds sont autorisés mais 0 et 1 ne sont jamais lus. Les distributions
géographiques/folds/adresses manquantes sont publiées, jamais des exclusions.
Le score principal sera le recall brut de ces 2 000 lignes, pas une métrique
sur toute la base : cette population exclut 640 lignes et n'est pas un nouveau
holdout externe. Le replay V34 était déjà consommé par d'anciennes recettes.
Toute adaptation future après ce test devra le considérer comme DEV consommé.

Réutiliser la BAN query-only V34 exige égalité de TOUS les champs avec la
géométrie gelée du pilote 128 après ajout du même indicateur Lambert93 V43.
Sinon STOP, jamais réparation après sélection ni coordonnées label-side.
V322 scelle la population AVANT toute recherche et avant les labels. Puis
réencode uniquement les noms, avec le même modèle LOCAL V52, prefix `nom`,
max_length 128, projection/vocabulaire propre au fold V77, huit mots V59.
Lots physiques 128 avec un thread, pas de téléchargement ni fit. Chaque
producteur doit avoir son exit0 réel et PASS V214 ; tous les lots avant seal.

V323 interroge les 43 896 818 feuilles natives (fermées incluses) avec EXACTEMENT
la recette V286 retenue en V287/288 : sources officielles, propriétaires,
successions, sigles régionaux, XY/BAN et sparse8. Heads V77/V228 gelés, propre
fold tenu à l'écart, aucun modèle aval. ONE collecteur natif global100, pas
pool fixe ni reranker. Changement physique seulement : lots de quatre
requêtes pour réduire l'accumulation RSS ; mêmes preuves/scores et cap100.
Plafonds V214 inchangés : 2 Gio RSS, 80s/worker, 20s/requête ouverture index
incluse, quiet30, pression1, libre>=50%, un thread. Swap global télémétrique,
non attribué au worker. Coût des 16 premières descriptif, pas un filtre
qualité. Aucun STOP/fragment réutilisé, aucun retry sans nouvel accord humain.

V324 généralise uniquement la cardinalité des kernels exact/op V200/218,
sans changer leurs règles de métrique ni leur politique d'équivalence.
V325 autorise les cibles UNIQUEMENT après contrôle de chacun des 500
producteurs V323, de TOUS les 2 000 outputs, de leurs hashes/epochs et cap100.
V326 évalue dans des processus distincts et supervisés : d'abord permission
sans GT, puis filtres Arrow fold234 ET les 2 000 query_id pour les originaux
humains/sidecar scellé. Aucun positif injecté, réparation, exclusion de statut,
ancien candidat, alias CRM ou identifiant comme feature. Exact/op, Wilson95,
folds, historique/V2/V3, absences catalogue, temps/cache non contrôlé séparés.
Les comparaisons aux 128 ne sont PAS appariées et ne prouvent pas un gain.

Ce test ne franchit ni le gate complet 23 571 OOF ni la généralisation/SLO,
et ne peut être READY_FOR_FOLD1 même à100%. Les bornes Wilson sont nominales
(lignes non nécessairement indépendantes, développement adaptatif). Fold1
toujours fermé, objectif final exact99,2%/Wilsonbas99,0%/chaquefold98,8% et
SLO p95<=1s/p99<=2s, tests de familles/zones/domaines et attribution fine dus.
Milestones isolés et cités dans handover.md ; aucune dépense ni push externe.
