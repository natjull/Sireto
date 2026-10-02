# V63 — identité et site appris conjointement, objectif de tête

Préenregistrement avant exécution. Ne pas confondre contrôle de modèle et
recall initial100. Objectif final inchangé : exact humain>=99,2%, sortie
initiale<=100, SLO et généralisation, folds0/1 interdits, aval gelé.

Hypothèse : le bag-of-fields et l'objectif moyen par paire ne ciblent pas
suffisamment les concurrents dangereux. Apprendre un noyau conjoint : deux
vues NOM/ADRESSE distinctes et19 preuves exactes/textuelles/site/géographiques
V56. Numéro et adresse exacts ne sont pas confiés à un embedding. Distances
25/100/500/2000m et leur variante BAN fiable sont conservées. Aucun site fermé
exclu. Contextes query-only : nom court/long, présence numéro/voie, coordonnées
compatibles et fiabilité BAN. Pas de lookup vérité à l'inférence.

Réutiliser512 scènes V52/V56 déjà encodées,23874 documents officiels. Aucun
réencodage national, téléchargement ou build. Tous négatifs naturels sûrs
participent à log(1+sum exp(score_negatif-score_positif)), moyenne par requête.
Ni injection de positif dans une évaluation ni nouveau score après reranker.
Le positif officiel intervient uniquement dans ce TRAINER supervisé.

Sélection interne par entité V53, excluant les parents externes ET internes
des positifs et négatifs. Bases SVD NOM/ADRESSE uniquement sur requêtes et
documents de ce train sûr ; directions q8/document32 au plus. Pas de base
apprise sur les requêtes externes. Trois familles : HEAD19, JOINT_STATIC,
JOINT_ADAPTED (preuves pondérées par contexte query-only plus correction
bilinéaire faible rang, séparée pour nom/adresse). BASELINE11 V57 refitée
sur ce seul train incluse, pas de poids baseline ayant vu le contrôle interne.

Poids preuves/contextes et cosines>=0, plafond64 ; corrections signées
[-64,64]. Ridge preuves/cosines0,005, correction0,0001/0,001/0,01. L-BFGS-B
180 itérations ; candidat non convergé publié mais inadmissible. Refit final
non convergé impose STOP, pas un changement silencieux d'hyperparamètres.
Choix : erreur de tête interne, puis nombre moyen de concurrents devant le
positif, puis capacité moindre et ridge plus fort. Mesures externes après
choix uniquement, baseline refit V57 versus recette sélectionnée ; strates
same-parent/adresse exacte/numéro+commune publiées sans sélectionner dessus.
Tests de gradient, équivalence algébrique query-kernel et non-influence des
données externes sur paramètres/choix.

Un modèle peut se servir comme deux vecteurs de requête transformés dot des
vecteurs officiels gelés + preuves natives non négatives. Cette égalité n'est
PAS un moteur national top100 ni une preuve de coût. Aucun pool fixe ou
ranker aval ne doit servir à le déployer. Un build dense national ne sera pas
déclenché sur la seule loss train ; d'abord étayer head/marges sur chacun des
folds, puis protocole de recherche indexée/coût et vraie sortie100 scellée.

Limitations explicites : les embeddings V52 ont concaténé les alias de chaque
vue et peuvent perdre source/période ; temporalité causale CRM indisponible.
Relations/filiations BODACC/RNE, catégorie juridique/activité, preuves
cohérentes nom-adresse et noms de prédécesseurs attachés au site ne sont PAS
ajoutés par ce probe. L'audit V58 choisi par vérité ne peut pas être recyclé
comme index ou feature. Ces sources restent nécessaires à l'architecture
complète, pas considérées implicitement couvertes par les embeddings.

Garde80s/4Gio/zéro nouveau swap, superviseur90s, deux threads, STOP immuable.
Artefacts modèles et diagnostics seulement, zéro candidat produit et aucune
mesure recall nationale. Contrôles réutilisés, non holdout inédit. Inspiration
apprentissage contrastif de retrieval, pas reproduction SC-Block ni garantie
de ses résultats : https://arxiv.org/abs/2303.03132.
