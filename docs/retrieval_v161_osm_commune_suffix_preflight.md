# V161 — qualification communale du nom CRM, préflight sans GT

Recette et code pré-déclarés au commit `5b2f8a57` avant exécution. Root
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v161/osm_commune_suffix_preflight_20261014_night`.
Initialize réel 0, puis session `79071`, chunk `47e3f2`, exit réel 0 en 8,358 s.
Aucun label, rang, SIRET candidat, modèle, index, réseau ou PBF relu.
Artefacts/source/COG/tokenizer/STOP hashés avant le préflight ; contexte
OSM complet de 504 742 499 octets lu en streaming puis vérifié de nouveau.

## Hypothèse unique et figée

Le matcher V132 exige actuellement tous les tokens CRM hors articles STOP
dans une seule variante OSM publiée. Il ne retire pas les noms de commune,
les CP ou les codes INSEE. Le panel n'a pas `crm_city`.

V161 autorise seulement le **libellé communal officiel entier normalisé,
en séquence contiguë finale** du nom CRM, routé par son code INSEE vers le
COG 2026 courant ou un parent délégué déjà publié. Pas de mot partagé retiré
isolément, pas de préfixe, quartier, ville manuelle ou suppression de CP.
Le reste informatif doit être non vide. Les articles STOP sont appliqués
après cette décomposition, sans nouvelle liste de termes.

Matching = ancien nom complet **OU** reste décomposé. Toutes les variantes
POI/brand publiées restent séparées ; aucun mélange de mots entre variantes.
La requête CRM originale reste inchangée dans son panel scellé et chaque
ligne conserve son binding ainsi que les tokens communaux retirés. Aucun
alias d'entreprise n'est construit, aucune commune n'est une preuve de maison.

## Résultats query-only sur toutes les 526 entrées

- 521 codes ont un libellé communal routé non ambigu ; 95 noms ont un suffixe
  communal complet et une décomposition informative non vide.
- 46 de ces 95 restent à un seul token informatif : attention à la fragilité
  des noms courts/publics/génériques. Ce constat ne sert pas à un nouveau filtre.
- 6 requêtes décomposées n'ont pas de géométrie compatible et leur source POI
  demeure vide. La décomposition ne répare pas cette absence de source locale.
- Tous les 54 anciens objets POI appariés sur 41 requêtes sont conservés.
  L'union compte 67 objets sur 49 requêtes : 13 nouveaux objets, 11 requêtes
  affectées, dont 8 passent de zéro à au moins un objet apparié.
- Sur les 95 décomposées : 81 n'ont aucun objet apparié, 11 en ont un ; les
  trois autres en ont respectivement 2, 3 et 4. Une enseigne/catégorie n'est
  donc pas automatiquement unique, même dans le contexte local.
- Les 13 nouveaux objets ont des points numériquement valides ; trois portent
  un tag `operator`. Aucun texte d'opérateur/personne/adresse n'est exporté
  ni utilisé pour matcher. Ce tag ne certifie pas un opérateur officiel.

Ces objets sont des **mentions communautaires OSM**, pas des candidats SIRET
validés ni des gains de recall. La couverture des anciens objets est monotone
par construction de l'OU, mais un futur score ajouté pourrait changer la
compétition entre SIRET : aucun engagement de recall monotone n'est établi.

## Décision permise par ce préflight

Il existe un blocage de formulation générique mesurable sans labels : le
qualificatif d'adresse impose à tort un mot supplémentaire au nom du POI.
L'hypothèse unique est désormais scellée ; ne pas tester d'autres suppressions
ou choisir des villes après lecture des résultats humains.

Une éventuelle projection vers SIRET/runs relève d'une autorisation distincte,
avec géométrie communautaire faible et modèle/coefficients/union MAX gelés,
génération ALL526 et exits réels avant labels. Le préflight seul ne décide pas
si les huit nouveaux appariements sont corrects ni si un des ratés est récupéré.
Jeux fermés et dénominateur humain demeurent intacts.
