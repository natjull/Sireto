# V57 — ablation native avant mesure de qualité

V56 s'est arrêté à16 requêtes sans labels, moyenne primaire6,836s >4s.
Il n'est pas déployable et aucun recall V56 n'est mesuré. Ses preuves de
nom exact réparées sont conservées. V57 isole le noyau natif11 features
pour mesurer réellement qualité/coût sans le parcours croisé spatial.
Ce test ne remplace PAS une architecture complète exploitant coordonnées
et relations officielles ; aucune promesse de99,2% ni de SOTA.

Avant toute sortie initiale V57 : reprendre les512 scènes d'entraînement
V56 scellées, conserver leurs11 composantes lexicales/site, retirer les8
coordonnées géographiques, refit OOF non négatif et ridge0,005, BASE1
nom/adresse seulement. Même registry entités, aucun parent heldout comme
positif ou négatif train. Aucun choix de poids suivant le pilote.

Recherche nationale43896818 sites : une unique requête native const-score
combinant couverture mots/ngrams, nom courant exact légal/enseigne/site,
adresse actuelle exacte, commune/CP, numéro+commune, preuves conjointes
et siège+commune. Pas de TF/IDF/BM25, d'alias CRM, d'identifiant comme
feature, de filtre d'activité, de master2000/500, de lookup GT ou de
reranker. Le moteur retourne directement<=100 SIRET. Le nombre interne
de postings visités par le moteur n'est PAS instrumenté ; ne pas le
confondre avec les<=100 références retournées/hydratées légèrement.
Égalités canoniques uniquement entre résultats effectivement retournés.

Population128 v38/V42/V49, panel réutilisé pas holdout inédit, disjoint des
512 requêtes train. Première garde16 sans labels : moyenne primaire<=4s,
budget expérimental pas SLO. Worker80s/6Gio/zéro swap, supervision90s,
contrôleur360s, contrôle individuel20s au retour de l'appel natif, deux
threads, aucune relance élargie. Seule la supervision worker90s coupe
avec certitude un appel natif bloqué ; ne pas prétendre à une préemption
native individuelle20s.
Si128 scellées, évaluateur séparé v49 : exact/opérationnel, dénominateur
all-human inchangé, Wilson, historique/V2/V3, folds, gains/pertes face
V42 scellé, latences. Aucun label en génération ni old candidate pool.
STOP immuable si coût/intégrité échoue, partiel jamais mesuré comme128.

Index/manifeste/métadonnées/inventaire vérifiés avant/après ; gros segments
size/mtime seulement, pas rehash intégral. Fold0 non lu, fold1 fermé,
modèles aval gelés, aucune dépense externe. Aucun READY_FOR_FOLD1 même
à100% pilote : OOF complet, généralisation, attribution et SLO exigés.
