# V4.11 — Contrat du challenge descriptif inédit

Statut : préenregistré après `GO_FREEZE_V411_CANDIDATE` et avant toute
qualification ou inférence du challenge.

## 1. Portée

Le challenge mesure une seule fois le stack gelé V4.11 sur les 225 lignes
`UNSEEN` du registre V4.11-A. Il est descriptif : ces lignes ont toutes un
`SERVICE ID` absent et ne sont pas représentatives du flux produit.

Il ne peut :

- promouvoir un modèle ;
- modifier retrieval, ranker, accepteur ou seuil ;
- certifier une précision de 99,8 % ;
- remplacer le nouvel export CRM indépendant exigé pour la preuve finale.

## 2. Entrées gelées

| Entrée | Identité |
|---|---|
| registre V4.11-A | build `fd25d1922040d585` |
| manifeste registre | `77711f91fda8dffec3210c49b3df8404e46ff540f30f9597fc7fe7722f2d6962` |
| parquet `unseen` source | `63ff648f6e326721e0646b0101de079f9a6feadb6e02c0474066c1288d8025a3` |
| candidat V4.11 | build `9d23bf3deb6b63de` |
| manifeste candidat | `a7fc765fe439392baec61fa8a35a941bb1f778281ccdbb54b55c699e9f0c11d9` |
| stack gelé | `81279978f47e1e2b1b4a1ea85d595b8dedd8ee8a073e34a19b3ffd340c945d5a` |
| snapshot établissements | `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` |
| politique de qualification | `active-direct-current-v4.0` |

L'implémentation exacte de la politique est également gelée :

| Source | SHA-256 |
|---|---|
| `scripts/build_benchmark_v4_current_snapshot.py` | `b0451766575f0023d42d598caa23aebb0e81cff5fcb60f5071da64c9b3f0b19b` |
| `scripts/build_benchmark_v3_evidence.py` | `9ebf636101de6cd73e4079fbcc14b012e655fdd6ff08910e00127ee915718dcc` |
| `src/xgb_matcher/blocking.py` | `e6a0fded2f6496c9f4e901d8ba4fca1b912f5410c3c506a170c434ec02a55736` |
| `src/xgb_matcher/features.py` | `839f55b0d8c56e22e75758db88647c910fd8158039d1b0175f9c818e5ac0b191` |
| `src/xgb_matcher/naming.py` | `b7ef59a8cb7529179567f6e3ffe3b64757383a9e449a0110886abe640a1b5fc1` |
| `src/xgb_matcher/partitioned_store.py` | `181d1c8a56539f6b36e01d9fc040a7fb4135e28a0b10147775abd5b33837a39f` |

Le builder de qualification doit vérifier ces six hashes avant de lire une
query. Une dérive invalide le protocole.

Le snapshot qualifie l'identité à sa date propre, au plus tard le
1er décembre 2025. Il ne prouve pas l'état réel en juillet 2026.

## 3. Incident d'ouverture

Après le gel du candidat, une inspection mécanique du parquet source a
affiché le SIRET CRM de trois lignes à l'orchestrateur :

```text
source_row_number = 1102, 1169, 1314
```

Les valeurs ne sont pas recopiées. Un registre immuable doit porter :

```text
reason = INPUT_SIRET_EXPOSED_TO_ROOT_CONTEXT
exclude_from_primary_blind_metrics = true
```

Les trois lignes restent dans l'exécution unique, mais les résultats sont
publiés séparément :

- `DESCRIPTIVE_UNSEEN_BLIND_222`, métrique aveugle principale ;
- `EXPOSED_3`, cohorte contaminée ;
- `ALL_225`, total opérationnel non entièrement aveugle.

## 4. Projection CRM assainie

Le builder lit le parquet source par projection physique. Il ne doit jamais
charger une colonne SIRET/SIREN, le fingerprint source dérivé de la ligne, un
label, un candidat, un rang, un score ou une prédiction.

La sortie contient exactement :

```text
query_id
crm_record_id
crm_name
crm_address
crm_postcode
crm_city
crm_insee
```

`query_id` et `crm_record_id` sont des identifiants opaques générés à partir
du numéro de ligne et d'un domaine V4.11, sans identifiant entreprise.

Tout nom de colonne contenant, sans tenir compte de la casse,
`siret`, `siren`, `truth`, `label`, `candidate`, `rank`, `score`,
`prediction`, `service_id` ou `fingerprint` bloque le build.

La projection doit avoir 225 identifiants uniques et des champs CRM non vides
conformes au constat du registre. Le mapping avec `source_row_number` est
scellé dans un fichier séparé, jamais fourni au qualificateur.

## 5. Qualification indépendante du stack

La qualification n'utilise jamais :

- le SIRET CRM source ;
- le retrieval V4.11 ou un autre top-k ;
- le ranker, l'accepteur, leurs rangs, scores ou décisions ;
- un résultat de pipeline antérieur.

Elle applique mécaniquement la politique `active-direct-current-v4.0` à
l'univers géographique complet du snapshot :

1. partition INSEE lorsqu'elle existe, sinon code postal ;
2. pour une mégapole, intersection INSEE et code postal ;
3. établissements actifs uniquement ;
4. mêmes ancres directes nom/adresse que la politique V4 gelée ;
5. aucun top-k : tous les établissements directs compatibles sont conservés.

Labels :

- un unique établissement actif direct : `MATCH_EXACT` ;
- plusieurs établissements actifs directs : `AMBIGUOUS` ;
- aucun établissement actif direct : `UNRESOLVED` ;
- `NO_MATCH` est interdit.

Il n'existe ni secours web, ni départage agentique entre plusieurs sites.
DataGouv, RNE, BODACC, FINESS ou d'autres sources peuvent documenter un audit
ultérieur, mais ne changent jamais le label mécanique.

Chaque preuve conserve au minimum :

```text
query_id
snapshot_sha256
policy_version
partition_kind
partition_key
active_universe_count
candidate_siret
candidate_siren
candidate_state
candidate_names
candidate_address
name_evidence_class
address_evidence_class
direct_match_rule
```

Le fichier de labels conserve :

```text
query_id
label_kind
ground_truth_siret
ground_truth_siren
direct_active_candidate_count
evidence_refs_json
qualification_reason
snapshot_sha256
policy_version
validator
human_validated
```

`AMBIGUOUS` et `UNRESOLVED` ont des vérités nulles.
`validator=AUTONOMOUS_MECHANICAL_V4` et `human_validated=false`.

Les queries, preuves et labels sont fermés atomiquement et hashés avant toute
inférence.

## 6. Exécution one-shot

Un runner dédié est requis ; aucun script de développement ne peut être
réutilisé tel quel.

Ordre obligatoire :

1. preflight du CRM sanitized, des manifests et du ledger ;
2. hash du fichier de labels sans le désérialiser ;
3. retrieval V4.11 label-free, top-100 absolu ;
4. scoring par l'unique modèle Ranker C `full_fit` gelé ;
5. construction des 80 features de scène ;
6. score par l'accepteur `COMPACT_LOGIT` gelé ;
7. application du seuil fixe `0.8720916706888049` ;
8. fermeture et hash de `predictions_blind.parquet` ;
9. écriture durable du ledger `PREDICTIONS_SEALED` ;
10. seulement ensuite, ouverture des labels et évaluation descriptive.

Le runner ne réentraîne rien et ne choisit aucun seuil. Un pool vide produit
`REVIEW / NO_CANDIDATE`.

`UNRESOLVED → REVIEW` ne peut pas faire partie de la prédiction aveugle car
`UNRESOLVED` est un label. La décision brute du stack est toujours publiée.
Un éventuel overlay de reporting est publié séparément et ne compte jamais
comme performance du stack.

Les 225 prédictions sont produites en un seul run. Toute tentative ayant
franchi le premier scoring marque le challenge consommé, même si elle échoue
ensuite. Aucune correction ou seconde exécution n'est autorisée.

## 7. Parité et intégrité avant ouverture

Avant le challenge, le runner doit passer :

- test de schéma interdit ;
- test qu'aucun label n'est désérialisé avant scellement des prédictions ;
- test du ledger exclusif ;
- test top-100 actif, unique et rangs contigus ;
- test de scoring ranker sans `is_ground_truth` ;
- test train/serve reproduisant les scores Ranker C, les 80 scènes, les
  scores accepteur et décisions sur l'artefact historique gelé ;
- suite complète verte.

Le verrou d'exécution épingle le runner commité, le CRM sanitized, les labels
gelés, le retrieval, le ranker, le calcul de scène, la taxonomie, l'accepteur,
le seuil, le snapshot, les partitions et le runtime.

Toute violation donne `STOP_DESCRIPTIVE_INTEGRITY`, sans tuning et sans
nouvelle tentative.

## 8. Publication

Pour `DESCRIPTIVE_UNSEEN_BLIND_222`, `EXPOSED_3` et `ALL_225`, publier :

- volumes `MATCH_EXACT`, `AMBIGUOUS`, `UNRESOLVED` ;
- recall retrieval exact à 100 sur les seuls `MATCH_EXACT` ;
- Hit@1 Ranker C sur les seuls `MATCH_EXACT` ;
- AUTO, corrects AUTO, erreurs AUTO, précision et couverture brutes ;
- intervalles de confiance ;
- décisions brutes sur `AMBIGUOUS` et `UNRESOLVED` ;
- chaque erreur end-to-end et sa phase d'origine.

Les seuils 99,8 % et 80 % sont affichés à titre descriptif, jamais comme gate
de promotion.
