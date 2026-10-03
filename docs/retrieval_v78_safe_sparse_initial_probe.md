# V78 — Recall initial national de l'expansion sémantique corrigée

Préenregistrement avant sorties. Modèles V77 figés, vocabulaire différent par
outer fold et inner-fit protégé, pas l'ancien bundle V59/V60 partagé. Même
famille de score V60 :11 preuves natives + alpha*couverture des mots officiels
appris, composante ajoutée souplement en commune/CP. Aucun filtre géographique
du score de base. Un collecteur natif retourne directement≤100 SIRET sur les
43896818 sites du principal national, aucune liste fixe/reclassement/reranker.
Les champs primaires historiques restent, mais les nouvelles vues V75/V76
ne sont PAS fusionnées dans cette ablation. XY/BAN/graphes/catégorie juridique
non intégrés ; ne pas prétendre architecture complète ni SOTA.

Export d'un bundle serving sans scènes/labels/contrôles de TRAIN. Encoder
local V52 CPU2 threads, prefixnom/max128 tokens, IDs retirés par surface V52,
128 noms CRM seuls réencodés,≤8 mots pondérés de chaque vocabulaire officiel
du fold correspondant. Projection/poids/vocabulaire vérifiés, aucun refit ou
choix de variante après recall. Aucun corpus de documents réduit autour du GT.

Audit d'autorité préalable sans requête/label, puis gate coût des16 premières
requêtes de la nouvelle génération query-only avant poursuite : mean≤4s,
travailleurs80s/6Gio/zéro nouveau swapout, superviseur90s, contrôleur360s
incluant package/encode/génération. Aucun budget élargi, reprise des120 sorties
V60, ancien candidat ni résultat partiel évalué. STOP immuable si garde échoue.

128 requêtes V34 dans l'ordre hashé préexistant, disjointes TRAIN512,
folds234, panel de développement réutilisé PAS holdout inédit. Labels et
candidats des comparateurs V42/V49/V57/V75/V76 ouverts seulement APRÈS
scellement de toutes128 sorties. All-human, sidecar officiel label-side,
exact/opérationnel/Wilson, segments historiques/V2/V3, gains/pertes appariés
sur chaque baseline ; cache/timing comparateurs historiques non appariés.
Diagnostics natifs limités aux références retournées, postings non instrumentés.
Objectif99,2% et SLO1s/2s inchangés ; aucun READY_FOR_FOLD1 sur un pilote,
même à100%. Fold0 non lu, fold1 fermé, aval gelé, aucune dépense/téléchargement.
