# Proximité géographique quand le CRM nomme une autre rue

Le cas GEDIMAT montre une limite réelle de notre mesure : **440 mètres
jusqu'au point moyen BAN ne signifient pas que le magasin est à 440 mètres
de la rue demandée**. Le point SIRENE du site est à **88,70 mètres de l'axe
OpenStreetMap de la rue du Champ de Tir**. Ces chiffres portent sur des
géométries différentes ; ils ne prouvent pas à eux seuls l'identité du lieu.

## Ce que les positions établissent

| Mesure pour le point SIRENE du site | Distance |
|---|---:|
| Au point BAN moyen retenu auparavant | 439,59 m |
| À la position d'adresse BAN la plus proche de la rue CRM | 146,19 m |
| À l'axe OSM réel de la rue CRM | 88,70 m |

La rue du Champ de Tir et la rue du Marais Cantimpré se rejoignent dans le
secteur. L'ancienne distance pénalise surtout un décalage **le long de la
rue**, alors que le site est proche d'un autre tronçon. Elle ne suffit donc
pas à conclure que l'adresse CRM est fausse.

La carte utilise les géométries OSM observées et le point SIRENE, sans
déplacer le point vers le magasin ni inventer un contour parcellaire. Le
numéro CRM 9012 reste non résolu précisément. Les bâtiments voisins ne
deviennent pas automatiquement le bâtiment de l'entreprise.

## Correctif général testé

Pour une adresse CRM reconnue seulement au niveau d'une rue, la première
version locale mesure la distance à **toutes les positions d'adresse BAN de
cette rue**, plutôt qu'à leur seule moyenne. Le signal de proximité devient
le maximum entre l'ancien signal et cette nouvelle proximité, pondérée comme
une preuve faible. Les numéros précisément reconnus ne sont pas modifiés.

Cela évite les appels cartographiques en production : la source est chargée
une fois puis les distances se calculent localement. Les points d'adresse
BAN restent un ensemble discret, pas une route interpolée ni une parcelle.
L'axe de voie OSM sert ici au contrôle géométrique du cas ; il n'est pas
remplacé silencieusement par ce nuage de points dans les résultats.

La source couvre les **62 requêtes avec un repli explicite à la rue**, soit
**942 positions uniques cumulées**. Les 464 autres requêtes restent
inchangées. Ni GEDIMAT, ni son SIRET, ni aucun autre nom commercial ne
sélectionne une position ou ne règle une distance.

La comparaison conserve Gemma, les calibrateurs et leurs poids. Les preuves
de proximité du site et leurs interactions existantes sont recalculées
avant l'unique collecteur national de 100 SIRET. Les positions de branches
ne sont pas copiées au siège ; le contexte d'entreprise reste distinct de
la position du site. La recherche lexicale nationale reste ouverte.

Les requêtes non modifiées réutilisent leurs sorties antérieures scellées,
car leur score est identique par construction. Les 62 requêtes modifiées
sont recalculées. L'évaluation porte toujours sur les **526 requêtes**, avec
les gains et les pertes appariés, après toutes les sorties et l'exit possédé.

La limite de cette première ablation est importante : elle conserve
l'univers numérique complet déjà préparé. Elle ne prétend pas encore ajouter
tous les établissements nationaux voisins d'une longue rue au-delà de cet
univers. Elle ne change pas non plus la règle métier d'équivalence des labels.

## Résultat de retrieval

La comparaison complète a terminé avec une sortie possédée 0 : **504/526
SIRET exacts (95,8175 % brut ; 94,7553 % pondéré)**, 508/526 SIREN et 505/526
opérationnels. **Zéro gain et zéro perte** par rapport à la baseline V115.
GEDIMAT reste absent des 100. Corriger la géométrie seule ne suffit donc pas
dans cette ablation ; aucun poids n'a été ajusté après ces résultats.

## Le pont géographique enseigne–établissement

Une seconde expérience indépendante teste une autre information : les
**commerces nommés présents dans le quartier**. La requête source demande tous
les commerces OSM autour de la géométrie CRM, sans nom d'entreprise ni SIRET.
Elle trouve 35 commerces, dont un point « Gedimat » situé à **50,06 mètres du
point SIRENE de Léon Bracq**.

Le mécanisme générique recherche les points dont le nom couvre les tokens
informatifs du nom CRM, puis apporte une preuve faible aux établissements
publics voisins. Cette preuve est distincte d'un alias propre de l'entreprise :
la proximité ne certifie pas l'exploitant. Le signal vaut
`0,5 / (1 + distance_m / 300)` et utilise le coefficient géographique déjà
figé, sans entraînement ni ajustement après lecture du résultat. Les
établissements à diffusion protégée sont exclus de ce nouveau signal.

La recherche conserve l'univers numérique complet préexistant de 390 003
références pour cette requête, la branche lexicale nationale non restreinte
et **un seul collecteur initial de 100**. Aucun positif n'est injecté.
Le SIRET humain n'est lu qu'après génération complète, sortie outil 0 et
scellé des candidats. **Résultat : le SIRET 68552038900030 apparaît au rang
46**, alors qu'il était absent dans la comparaison précédente.

C'est une preuve de faisabilité sur un cas déjà diagnostiqué, **pas une
mesure de recall ni une validation de généralisation**. Le score complet
reste 504/526 ; il n'est pas augmenté à partir de cette expérience ciblée.
Les données et coefficients hérités gardent aussi leurs limites de cutoff
et d'OOF encodeur. Aucun déploiement ni certification de latence n'en découle.

La direction est désormais précise : tester une couche locale de lieux
nommés et de géométries de voies sur une cohorte complète, avec une recette
figée avant les labels, en publiant gains **et pertes**. Les homonymes,
enseignes voisines d'une autre entreprise, centres commerciaux, points OSM
mal placés et couverture inégale doivent être audités. Un snapshot local
éviterait la dépendance aux appels Overpass en ligne ; sa préparation n'a
pas été lancée ici. La simple proximité ne doit jamais devenir une décision
de matching automatique.

## Sources et traçabilité

Le contrôle OSM archive 1 322 voies, 6 684 bâtiments et sept tronçons portant
le nom normalisé de la rue CRM. La requête source ne contient aucun nom
d'entreprise ou SIRET. OSM est une source communautaire actuelle : elle ne
constitue pas une preuve officielle d'exploitant ou une archive au cutoff.
Les appels groupés suivants ont échoué par timeout, sans métrique partielle.
Le test local uniforme utilise donc uniquement la source BAN existante.

Le résultat reste diagnostique : la BAN du 14 août et les sources héritées
du pipeline ne sont pas admises rétroactivement au benchmark du 1er août.
Aucun score ici ne peut certifier les 99,2 % nationaux ni les SLO.

Artefacts sous `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` :

- contrôle géométrique `v123/street_geometry_probe_20261013_a`, commit
  `4eb4032a` ; carte locale vérifiée dans Edge, projection D3 des coordonnées
  réellement observées, sans tuile réseau ni géographie dessinée à la main ;
- tentatives de compilation OSM `v124/query_street_shapes_20261013_a` et `_b`,
  commits `2ec5aa99` et `59b197ec`, conservées comme échecs incomplets ;
- source locale `v126/ban_street_clouds_20261013_a`, commit `c9d6e627`,
  génération complète et sortie outil 0, session 2283 ;
- comparaison `v125/street_cloud_retrieval_20261013_b`, commits `7c2e7e4d`,
  `59b197ec`, `c9d6e627` et `873c3372`, recette et code figés avant génération,
  sortie outil 0 session 74149, 526 sorties couvertes ;
- tentative groupée de lieux nommés `v127/geo_named_pois_20261013_a`, commit
  `647de24f`, premier appel Overpass 504, sortie outil 1 session 26571,
  aucune génération complète ni mesure ;
- faisabilité `v128/named_geo_feasibility_5130_20261013_b`, commits
  `97280e9f` et `3ceaee89`, sortie outil 0 session 16456 et diagnostic
  post-scellé rang 46. Namespace `_a` conservé après erreur SQL : source
  complète réutilisée à requête strictement identique, pas de candidats ni
  de métrique dans ce premier namespace.

Tous les labels, modèles aval, serving et jeux fermés restent inchangés.
Pas de nouveau training, unit test, installation de dépendance, cloud ou
nettoyage de fichiers utilisateur.
