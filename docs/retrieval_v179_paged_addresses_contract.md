# V179 — coût d'une adresse CRM gelée sur tout l'index

Protocole avant mesure, 4 octobre 2026. Pas encore de seal complet V177 à la
rédaction ; ce test ne peut partir qu'après le vrai PASS de toutes256phases
et de son seal supervisé. Le chemin, launchSHA V177 et ses dénominateurs
34346446aliases/44403034sites/98804872facts/2440276150postings sont fixés.
Le hash du futur manifest de seal est acquis une fois au préflight et fixé
dans le launch AVANT le worker ; il doit rester identique avant/après.
Impossible de choisir un autre corpus ou une projection partielle.

Une requête : la PREMIÈRE du même launchquery-only115128 figé (folds2/3/4).
Tous128 schémas/types/folds/ID uniques sont vérifiés avant la sélection du
préfixe1. Aucun filtre de query selon label/ancienhit/nom/adresse/qualité.
Seuls les champs CRM sont lus, aucun activityproof/GT/candidat suivi.
Texte identique125 : numéro + suffixe + voie, puis normalisation121/118.

Le reader178 exige chaque phase complète de la source, hashes/epochs,
actualparentexit et ressources. Snapshot exact float64/bool sur TOUTES les
variantes, file-backed readonly avec transfert de propriété, aucun cap de
token/alias/posting. Gather exhaustive des34346446aliases et digest par
chunks fixes524288, rehash des2vecteurs pour prouver qu'ils n'ont pas changé
après la query. Pas encore de projection des98mfacts sur44msites ni initial100.

Garde TOTALE query+snapshotvalidation+fullgather+rehash/cleanup20s. Pas de
trois budgets indépendants. Init/rehash de source séparés du coût de query
mais INCLUS dans le budget worker80s/1Gio. Toutes1024CSRarrays fullhashées
avant/après, sources originales via mêmes autorités scellées175 et epochs.
Quiet30s/pressionnormale1/free>=50%, swapglobal télémétrie non attribuée.
Superviseur153/164 réel, childexit0 effectivement confirmé nécessaire.

Freshroot CATNAT ; chaque échec laisse STOP et ses fichiers, aucun retry ni
snapshot partiel. Seuls les2fichiers scratch propres à une query réussie
sont enlevés après vérification de leur identité/hash. Aucun purge global,
téléchargement, GPU, dépense, fold0/1 ou modèle aval. Fixtures petites avec
fauxpressure/process ne valent pas preuve nationale.

Résultat COST_ONLY, complete_retrieval_index/recall/SLO/READYfalse : ne pas
annoncer un recall ou une latence de retrieval depuis ce test de composant.
Le but demeure initial100 all-human99,2% avant aval et les SLO1s/2s.
