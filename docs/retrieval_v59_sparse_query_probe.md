# V59 — traduction sparse apprise, contrôle préalable avant nouveau run

Hypothèse nouvelle : traduire l'embedding de NOM de la requête en<=8 mots
pondérés d'un vocabulaire OFFICIEL. Le moteur natif existant recherche
ces mots ; pas de nouveaux embeddings des43896818 sites, ni de pool
fixe à reranker. Ce petit décodeur ridge n'est PAS une reproduction
SPLADE et aucune performance de la littérature n'est transférée ici.
Référence de principe :
https://research.google/pubs/learning-sparse-lexical-representations-over-expanded-vocabularies-for-retrieval/

Préenregistrement avant exécution. Réutiliser les512 requêtes V52 et
leurs23874 documents officiels d'apprentissage, embeddings nom384 déjà
calculés, preuves lexicales V56 hashées. Recueillir les5 champs de noms
effectivement indexés pour reproduire le score multichamp, ne pas
apprendre sur un bag global puis servir une autre représentation.
Vocabulaire cap2048, fréquence documentaire>=3 et<=15% des documents,
stopwords linguistiques, lettres uniquement ; pas d'identifiant ou
d'alias CRM stocké. Le vocabulaire est issu du corpus officiel TRAINING,
pas du vocabulaire national complet. Aucun contenu V58 sélectionné via
vérité n'est incorporé en source de modèle ou d'index.

Décodeur ridge1/10/100 vers les mots du positif officiel ; nom CRM
comme entrée, vérité uniquement comme cible TRAINER. Entités du fold
externe2/3/4 exclues des positifs ET négatifs de la recette native.
Sélection IMBRIQUÉE par SIREN, split v53 existant : apprendre décodeur
et recette native baseline seulement hors des entités internes validées.
Inclure identité/alpha0 ; alphas0,5/1/2, jamais choisi via contrôle
externe. Critère interne : erreur de tête, puis nombre de concurrents
devant le positif, puis alpha moindre/ridge plus fort. Aucun contrôle
source (une cinquantaine de concurrents) ne vaut un recall@100 national.

Les termes originaux ne sont pas dupliqués par expansion, masse positive
normalisée, max8. Ajouter uniquement la composante NOM+commune/CP.
Ce gate ne filtre PAS le corpus : la recette native11 preuves demeure
globale, adresses/numéro/site toujours disponibles, sites fermés conservés.
BAN/XY, catégorie juridique, temporalité et relations ne sont PAS
incorporés dans cette ablation ; ils restent à intégrer explicitement
dans l'architecture complète, pas à oublier ou remplacer par du texte.

Contrôle de parité du score ajouté natif/entraîneur sur les4 premières
scènes scellées par fold. Worker80s/4Gio/zéro swap, superviseur90s,
deux threads. Aucune dépense, nouveau modèle téléchargé ou gros build.
STOP immuable si garde échoue. Si tous folds choisissent alpha0 ou si
les contrôles externes n'étayent pas un gain, ne pas déclencher un long
build/run sous prétexte que la loss train baisse. Toute recherche
ultérieure exige son protocole et sorties initiales100 réellement
scellées avant labels ; aucune validation automatique par ces fits.

Fold0 non lu, fold1 fermé, aval gelé, all-human intact, objectif
initial100 exact99,2% et SLO inchangés, pas READY_FOR_FOLD1.
