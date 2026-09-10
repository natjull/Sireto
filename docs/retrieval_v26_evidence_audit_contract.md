# V26 — proximité physique et pertes de noms officiels

Préenregistrement du 10 septembre 2026. Deux audits **label-side**, jamais
des générateurs, sur les1034 requêtes folds2/3/4 (522 difficiles +512 contrôles).
Ils réutilisent les labels/segments scellés v25; aucun label modifié ou exclu.
Pas de modèle, candidats nouveaux, score produit ou ouverture fold0/fold1.

## Géographie

Résoudre le point CRM avec la recette BAN v16 inchangée, à partir des seuls
champs CRM. Les labels servent ensuite uniquement à rejoindre **tous les
établissements du SIREN humain**, sans plafond et sans restriction de commune.
Comparer la distance au SIRET exact, au site du parent le plus proche et au
site actif le plus proche. Calculer aussi le site le plus proche dont la voie
normalisée diffère du CRM, et la présence antérieure du parent dans les voisins
v16. Les SIREN sont des clés officielles, jamais dérivés du SIRET.

Coordonnées valides selon l'emprise Lambert93 v16; DOM/hors-emprise, absences
et coordonnées invalides sont publiés, jamais interprétés comme éloignement.
BAN faible/non résolue distincte de BAN non déclarée faible; cette dernière
n'est pas une garantie de précision. Incertitude v16 = proxy heuristique,
pas intervalle de confiance. Le score BAN reste inéligible en production.
Le géocodeur impose déjà un numéro exact côté BAN : cette limite demeure,
aucune récupération de coordonnée depuis le label.

Distances descriptives aux seuils fixes50/100/250/500 m; pas de sélection d'un
rayon après résultat. **Proche ≠ même bâtiment/site ≠ bonne entreprise**.
L'angle entre rues et l'identité du bâtiment ne sont pas établis par ces seules
distances. Aucun élargissement de la politique opérationnelle. Ne persister
aucun X/Y ou longitude/latitude BAN brut, seulement distances/qualité et clés
d'audit. La comparaison exact/opérationnel antérieure reste inchangée.

## Noms

Comparer pour chaque parent humain : portfolio officiel complet; noms présents
dans les documents matérialisés SIREN/SIRET globaux; noms présents dans les
documents admis à la géographie effective de la requête; champs correspondants
réellement stockés dans Tantivy. Les noms de tous les établissements du parent
participent au diagnostic global, pas seulement ceux du SIRET humain.

Rôles primaires LEGAL_CURRENT/TRADE_CURRENT/SITE_CURRENT/HISTORICAL séparés.
SUPPORTING exclu de cet audit de similarité : ce rôle ne vaut pas autorisation
d'utiliser un nom de personne comme feature. Mesurer l'exactitude textuelle,
Jaccard des tokens et Jaro-Winkler sans seuil choisi pour les labels; publier
notamment les cas où une correspondance exacte CRM disparaît et les cas où
une simple intersection de tokens apparaît uniquement dans les noms perdus.
Un token commun peut être générique; une similarité n'est pas un rescue.

Distinguer absence du portfolio (hors portée), sélection/matérialisation,
géographie et différence réelle source→index. Lire tous les documents locaux
du parent par pages si nécessaire; vérifier les ensembles de noms normalisés
par rôle, type et géographie contre les documents sources. Une limite de noms
dans le portfolio matérialisé ne signifie pas que toutes les sources officielles
brutes ont été auditées. Aucun alias CRM, aucune preuve transférée entre sites
pour fabriquer un match conjoint, aucun positif injecté.

## Exécution et limites

Préflight technique ≤16, puis code/config committés avant les audits complets.
Mac/CATNAT, trente minutes par audit, RSS14 Gio, zéro swapout, temporaires
externes. Hashes sources/manifeste/résultats; indexmeta et inventaire taille/
mtime avant/après. Pas de prétention de rehash intégral du grand index.
Historique/V2/V3, folds, origines, statuts et contrôles toujours publiés.
Les audits de plafond voient les vérités; ils ne sont pas des résultats OOF
de génération et ne rendent pas vierge ce panel réutilisé.

Verdict `PIVOT_GENERATION`; intégrité non satisfaite = `STOP`. Pas de national,
de projection, de gate2000/500/100/Wilson/SLO ou de modification aval.
