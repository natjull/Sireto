# V231 : features TRAIN alignées avec la recherche réelle

Préenregistrement4 octobre2026, avant mesure. V228 ne change que place,
mais son fit utilise les11 primitives56, pas tous les bonus217. Il ne faut
pas apprendre toute une nouvelle tête sur ce décalage puis espérer que
cela calibre exactement la recette produit.

Ce module extrait les11 composantes EFFECTIVES du kernel217, compilé
avec un coefficient unitaire à la fois. Cela inclut containment d'alias
et corroboration parent/site avec les bonnes max/conditions/constantes.
Une extraction séparée avec onze poids nuls donne les contributions fixes :
expansion sparse OOF, propriétaires, succession et XY.

Le produit reste une seule recherche native avant collecteur100. Cette
extraction POINT n'est autorisée que pour TRAIN supervisé disjoint (après
227 intégral), ou diagnostic explicitement post-scellement. Les IDs
sélectionnent les observations supervisées, jamais une feature. Aucun
positif ne peut être ajouté à la sortie initiale, aucune génération de
candidats produit via ce module.

L'apprentissage futur devra exclure folds/entités held aux deux côtés,
exclure les équivalents officiels des négatifs, conserver les labels,
geler loss/paramètres AVANT contrôle et servir ces mêmes composantes.
Les géocodages TRAIN proviennent exclusivement des parquets BAN query-only
historiques55 (source512) ; jamais inférés depuis la position du positif.
Les vecteurs de nom TRAIN52 peuvent être réutilisés, normalisés sur leur
moitié nom384, ordre et texte nom vérifiés, projections/vocab77 fold-safe.

Comparer le score natif217 complet au produit scalaire features11 + fixe
pour TOUS les points utilisés, tolérance float32 bornée2e-5abs/2e-6rel.
Un échec STOP, jamais correction de score ni admission du positif.

Politique214 inchangée, aucun nouveau run physique autorisé par cette
simple bibliothèque ; un driver/contrat/scellement dédiés sont requis.
Tests sur petit index synthétique prouvent la sémantique, PAS la qualité.
Pas de recall national, SLO, READY, fold0/1 ou modèles aval modifiés.
