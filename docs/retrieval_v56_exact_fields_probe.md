# V56 — preuves natives de site, réparation des noms exacts

Préenregistrement avant toute sortie/mesure initiale V55 ou V56. Contrat
actif v38, objectifs exact99,2%/100 et SLO inchangés ; aucun modèle aval.
L'hypothèse lexicale v38 est historique, pas une obligation de BM25.

V55 remplace les scores BM25 par couverture constante de mots/ngrams,
nom/adresse exacts, numéro+commune, siège+commune et distances physiques
25/100/500/2000m, avec vues BAN fiable/faible. Pas de filtre d'activité,
aucun alias CRM ni identifiant de vérité comme feature. Les identifiants
servent uniquement aux joins, exclusions d'entités, sorties et égalités.
Les poids non négatifs sont appris en OOF avec les entités heldout exclues
des positifs ET négatifs d'apprentissage. Aucun résultat pairwise ne vaut
recall initial. Les preuves lexicales sont compilées dans l'index national.
Le collecteur100 fusionne deux flux triés et s'arrête par borne supérieure
certifiée, pas après un nombre fixe de candidats à reranker. Les pages
physiques peuvent croître mais ne constituent pas un pool fixe produit.

Défaut vérifié avant mesure : le constructeur v3
`_add_site_evidence_document` déclare mais ne remplit pas `names_exact`.
Il remplit `legal_names_exact`, `trade_names_exact`, `site_names_exact`.
Les512 scènes V55 ont leurs deux features dépendantes du champ ancien
entièrement nulles. Cela n'est pas un comptage global du dictionnaire de
termes. V56 interroge les trois champs courants effectivement remplis ;
les noms historiques/supporting n'ont PAS d'équivalent exact dans cet
index et ne reçoivent aucune preuve exacte inventée. Pas de reconstruction.

V55 immuable : manifeste d'entraînement SHA
`d9a4f8df9e438929920b313b5bf6c4db34b59465729667d5d48af71f50e1ebe5`.
Les512 vecteurs cachés sont réutilisés ; seules les coordonnées2 et10
changent, après assertion de leur nullité antérieure. Ordre, identité,
parents et dénominateur vérifiés. Audit complet lexical contre moteur
natif sur les16 premières scènes de cet ordre scellé, avant refit.
Même ridge0,005/BASE/grille19, aucune sélection selon résultats pilote.
Sources naturelles V52, géométrie et17 autres coordonnées inchangées.

Recherche évaluée sur les128 requêtes du panel v38/V42/V49 réutilisé,
disjoint des512 requêtes d'entraînement ; pas un holdout inédit.
Corpus lexical complet43896818 sites, spatial26951280 sites compatibles
métropolitains seulement. Hors couverture XY : preuve spatiale0,
document et requête conservés. Sources CRM/BAN query-only. Aucun lookup
de vérité/ancienne sortie dans le processus de génération.

Modèles scellés avant lancement, implémentation/index/queries hashés.
16 premières requêtes sans labels : moyenne de recherche primaire<=4s,
garde expérimentale PAS SLO produit. Worker80s/6Gio/zéro swap,
supervision90s, contrôleur360s, recherche individuelle20s, deux threads.
STOP conservé si garde déclenchée, aucun résultat128 annoncé sur partiel,
pas de relance élargie. Si128 scellées, évaluateur séparé v49 ouvre labels
folds2/3/4 et comparateur V42 scellé : exact/opérationnel, all-human,
Wilson, folds, historique/V2/V3, gains/pertes, latences et absences.
Gros segments : inventaire size/mtime, pas de rehash intégral revendiqué.

Même si le pilote réussit, aucun READY_FOR_FOLD1 : validation OOF complète,
généralisation, attribution exhaustive et SLO requis. Fold0 non lu,
fold1 fermé, modèles aval gelés, aucun calcul externe ni dépense.
