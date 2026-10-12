# Géolocalisation des adresses publiques des établissements

## Portée du diagnostic

Cette expérience étudie si compléter la localisation des établissements
SIRENE depuis leur propre adresse publique améliore la recherche initiale à
100 candidats. Elle appartient au mandat de huit heures se terminant le
13 octobre 2026 à 07 h 45 min 24 s Europe Paris. Les modèles aval restent gelés.

La BAN autorise les correspondances d'adresses mais son catalogue déclare
`score_eligible=false`. Le statut de toute mesure utilisant ces points est
donc `DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE`. Un gain de recall expérimental
n'autorise ni une mise en production, ni la validation du gate de production.
Les contrats historiques et leur catalogue restent inchangés.

## Source et géométrie

Le stock national SIRENE du 1 août 2026 fournit les adresses publiques propres
des sites O, actifs comme fermés. La BAN du 14 août fournit des correspondances
de référence ; elle ne prouve pas qu'un établissement occupait ce lieu à sa
date de fermeture. L'ensemble source est indépendant des requêtes et labels.

Seules les coordonnées originales absentes ou incompatibles sont complétées.
La commune et la rue normalisées doivent correspondre exactement. Un numéro
et un suffixe identiques permettent une vue maison ; sinon la vue reste une
empreinte complète de rue, de qualité réduite et de dispersion explicite.
Certification, type et source de position contribuent à l'incertitude.
Un point de siège ou d'un autre site n'est jamais copié.

La construction conserve les références et hashes des preuves ainsi que les
points projetés en Lambert 93, pas la longitude ou latitude BAN brute. Elle ne
reconstitue aucune adresse masquée et ne produit aucun fichier de prospection.
Une absence BAN n'est jamais une preuve d'inexistence ni une pénalité apprise.
Les distances ne peuvent servir de filtre dur à l'ensemble national lexical.

## Mesure

L'atlas de géométrie doit être complet, scellé et accompagné d'un owned exit0
avant son utilisation. Son nombre de points n'est pas une mesure de recall.
L'intégration au score requiert un nouveau namespace et une recette figée
avant nouvelles sorties. Le collecteur initial retourne au plus 100 SIRET,
sans pool fixe ni reranker aval. Toutes les requêtes avec label humain
exploitable restent au dénominateur ; exact et opérationnel sont séparés.

Le pilote de 526 requêtes a déjà servi au développement. Une amélioration sur
ce pilote devra être confrontée à un contrôle disjoint ; elle ne certifie pas
99,2 pour cent national. Les folds humains 0 et 1 et le test de 4323 restent
fermés. Aucune sortie partielle ni point ciblé par la vérité n'est admissible.
