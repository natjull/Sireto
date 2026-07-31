# V4.12-S — Contrat service, parité et latence

## 1. But

Figer puis éprouver le candidat produit suivant, sans réentraînement ni
réglage :

```text
retrieval sparse V4.12, top 100 strict
  → 45 features Ranker C
  → Ranker C gelé
  → 80 features de scène V4.11
  → accepteur COMPACT_LOGIT gelé, seuil 0.8720916706888049
  → veto V4.12-G sur l'univers géographique actif complet
```

Le jalon doit prouver que le chemin requête par requête reproduit le batch
historique et mesurer son coût sur le Mac. Il ne mesure pas la précision
produit et n'ouvre aucun nouveau test.

## 2. Frontières

La population de parité contient exactement les 1 456 requêtes `dev` déjà
consommées. Elle est utilisable pour la parité et la latence, jamais pour
modifier une feature, une règle, un modèle ou un seuil.

Le worker de service peut ouvrir uniquement :

- le paquet de référence assaini défini en section 3 ;
- les partitions et caches TF-IDF gelés ;
- le lookup DuckDB V4.12 audité ;
- le modèle complet Ranker C ;
- l'accepteur, sa metadata et la taxonomie gelés.

Sont interdits dans le worker et ses sorties :

- `label_kind`, `ground_truth_siret`, `ground_truth_siren`,
  `is_ground_truth`, `acceptor_target` et toute colonne de correction ;
- les challenges V4.11, V4-Fresh, random V4.8, V4.9, les anciens tests et
  tout test final ;
- le réseau, l'injection positive, l'entraînement, le changement de seuil,
  la reconstruction ou l'écriture du cache TF-IDF.

## 3. Paquet de référence assaini

Un builder séparé projette les sources historiques gelées vers un artefact
sans vérité. Il est le seul composant autorisé à lire les fichiers historiques
contenant également des colonnes de vérité, et il ne lit que les colonnes
explicitement autorisées.

L'artefact contient exactement :

- `queries.parquet` : 1 456 requêtes sûres ;
- `candidates_features.parquet` : 145 236 candidats, rangs 1 à 100 et
  45 features Ranker C, sans vérité ;
- `ranker_reference.parquet` : scores `float32`, rangs et top-1 de référence ;
- `scenes_reference.parquet` : SIRET/SIREN prédit et 80 features de scène ;
- `acceptor_reference.parquet` : score, seuil et décision `COMPACT_LOGIT` ;
- `guard_reference.parquet` : preuves agrégées, décision V4.11, décision
  V4.12-G et motifs, sans label ni correction ;
- `query_evidence.parquet` et `candidate_evidence.parquet` : preuves directes
  V4.12 déjà scellées ;
- `manifest.json` : sources, schémas, nombres, hashes et payload logique.

Le builder impose :

- les hashes exacts des neuf sources ;
- l'ensemble exact des 1 456 `query_id` dev ;
- 145 236 candidats, pools de 46 à 100, rangs contigus et SIRET uniques ;
- un seul enregistrement par requête aux étages scène/accepteur/garde ;
- zéro colonne interdite dans chaque sortie ;
- publication atomique dans un répertoire immuable.

## 4. Parité obligatoire

Le moteur persistant doit reproduire, dans l'ordre canonique :

1. les 145 236 couples `(query_id, candidate_siret, retrieval_rank)` ;
2. les 45 features Ranker C après cast `float32` ;
3. les scores `float32`, rangs et top-1 du Ranker C ;
4. les 80 features de scène `float64` et le SIRET prédit ;
5. le score `float64` de l'accepteur à `1e-15` près, le seuil exact et la
   décision exacte ;
6. les agrégats de preuve directe sur l'univers actif complet ;
7. la décision et le motif V4.12-G.

La tolérance `1e-15` est limitée au score logistique calculé sur une ligne :
BLAS peut différer du batch d'environ un ulp. Elle ne s'applique ni aux
features, ni au ranker, ni au seuil, ni aux décisions.

Toute autre divergence, valeur non finie, pool supérieur à 100, clé de cache
absente, SIRET lookup absent ou mutation d'un artefact produit `STOP`.

La garde reste un veto : elle peut transformer `AUTO_MATCH` en `REVIEW`,
jamais créer ou remplacer un match.

## 5. Persistance et latence

Deux mesures appariées utilisent le même ordre de 1 456 requêtes et deux
processus persistants séparés :

- V4.11 : retrieval, features, ranker, scène, accepteur ;
- V4.12-G : même chaîne, preuve directe complète et veto.

Après un warm-up explicitement exclu des mesures, chaque requête produit une
durée complète. Les percentiles utilisent la méthode nearest-rank et publient
`p50`, `p95`, `p99`, maximum, nombre brut et pic RSS.

Gates :

- parité exacte aux sept étages ;
- `sealed_key_miss_count == 0` ;
- `cache_rebuild_count == 0` et `cache_write_count == 0` ;
- `lookup_missing_count == 0` ;
- maximum absolu de 100 candidats ;
- pic RSS de chaque processus inférieur à 8 Gio ;
- `p95(V4.12-G) < 2 × p95(V4.11)`.

Le temps additionnel de la preuve et du veto est publié séparément. Une
latence élevée produit `PIVOT_SERVICE_LATENCY`, pas une modification de la
politique sur ce dev.

## 6. Verdicts

- `GO_V412_SERVICE_FREEZE` : intégrité, parité, persistance, mémoire et
  latence passent ;
- `PIVOT_V412_SERVICE_IMPLEMENTATION` : intégrité et parité passent mais le
  coût ou la persistance échoue ;
- `STOP_V412_SERVICE_INTEGRITY` : fuite, divergence, mutation, dépassement de
  100 ou ouverture interdite.

Même un `GO_V412_SERVICE_FREEZE` ne prouve pas une précision réelle de
99,8 %. La preuve produit reste une évaluation one-shot sur un nouvel export
CRM indépendant, après gel de ce service.
