# V4.12-L — population unifiée pour apprentissage OOF

## Résultat

Le build autoritatif est :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_learned_unified_population/2d29be3ccd8fcc3e`

Il contient exactement 17 097 requêtes :

- 17 054 lignes historiques liées ligne à ligne à `data/crm_ok_gt.csv` ;
- 236 labels historiques remplacés par leur adjudication locale ;
- 43 dossiers frais audités ajoutés ;
- deux corrections de contrôle appliquées à des lignes historiques existantes ;
- deux contrôles frais conservés hors entraînement.

| Label | Nombre |
|---|---:|
| `MATCH_EXACT` | 13 704 |
| `AMBIGUOUS` | 625 |
| `UNRESOLVED` | 2 768 |

Parmi les labels exacts, 11 619 établissements sont actifs et 2 085 fermés.
Les fermés restent utilisables pour apprendre l'identité avec un poids 0,5 ;
ils ne sont pas éligibles au chemin opérationnel qui préfère un établissement
actif.

## Contrôles d'intégrité

- aucun candidat, hit, rang ou score de retrieval n'entre dans la
  qualification ou les plis ;
- `queries.parquet`, `labels.parquet` et `fold_assignments.parquet` contiennent
  chacun 17 097 identifiants uniques et parfaitement alignés ;
- les cinq plis contiennent 3 499, 3 321, 3 539, 3 419 et 3 319 requêtes ;
- zéro composante SIREN traverse deux plis ; une correction changeant de SIREN
  relie l'ancien et le nouveau SIREN dans une même composante ;
- les hashes de tous les outputs concordent avec `manifest.json` ;
- le test ciblé du builder passe.

Les splits historiques train/dev/test et tous les dossiers audités sont déjà
consommés en développement. Ils ne sont donc pas présentés comme une nouvelle
validation indépendante. La nouvelle comparaison sera une mesure OOF groupée
par SIREN. Le test retrieval historique déjà ouvert reste un `PIVOT` : gates
globaux franchis, mais régressions des segments fermés et mégapoles.

## Prochaine étape

Construire ou réunir, sans réinjecter le positif, les pools de 100 candidats
de la politique retrieval gelée pour les 17 097 requêtes. Publier séparément
couverture identifiable et Recall@100 global, actif, fermé, mégapole et
multi-site avant tout entraînement du ranker.
