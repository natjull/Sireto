# V4.12 — Contrat de la garde d'unicité par preuve directe

Statut : préenregistré après le challenge descriptif V4.11 consommé et avant
tout build V4.12, calcul historique de métrique ou nouvel entraînement.

## 1. Verdict d'orientation

V4.12 est un pivot étroit de la décision :

- retrieval V4.11 gelé ;
- Ranker C gelé ;
- accepteur `COMPACT_LOGIT` V4.11 et seuil `0.8720916706888049` gelés ;
- ajout d'une garde déterministe, label-free, qui ne peut que transformer un
  `AUTO_MATCH` en `REVIEW`.

La première phase V4.12 n'entraîne aucun nouveau modèle et ne choisit aucun
seuil. Un accepteur enrichi exigera un nouveau contrat et des labels
indépendants ; il n'est pas autorisé par le présent contrat.

## 2. Constat à corriger

Le challenge descriptif V4.11 a identifié une erreur confirmée :

- deux candidats directement plausibles étaient dans le pool ;
- ils étaient classés premier et deuxième ;
- ils appartenaient à deux SIREN différents ;
- le premier avait un avantage lexical assez fort pour tromper l'accepteur.

La scène V4.11 décrit les écarts de scores et la concurrence intra-SIREN,
mais ne prouve pas l'unicité d'une identité directement compatible.

Le challenge consommé
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_execution/ddb7336e8c2e042d`
est diagnostic uniquement. Il est interdit dans :

- la construction des signaux V4.12 ;
- le choix d'une formule, d'une variante ou d'un seuil ;
- l'entraînement ou la calibration ;
- tout gate de promotion.

Ses prédictions et labels peuvent seulement être cités dans le rapport
historique expliquant l'origine de l'hypothèse.

La denylist `config/v4_12_forbidden_artifacts.json` interdit les trois racines
du challenge et chacun de leurs fichiers par SHA-256. Une copie,
re-sérialisation ou relocalisation portant un hash interdit doit bloquer le
build. Les seuls inputs de développement autorisés sont ceux de l'allowlist
`config/v4_12_development_inputs.json`; tout chemin ou hash supplémentaire
donne `STOP_FORBIDDEN_INPUT`.

## 3. Preuve directe gelée

La garde réutilise sans changement la politique
`active-direct-current-v4.0`, implémentée par :

- `find_direct_active_candidates()` dans
  `scripts/build_benchmark_v4_current_snapshot.py` ;
- `classify_direct_evidence()` dans
  `scripts/build_benchmark_v3_evidence.py`.

Sources gelées :

| Source | SHA-256 |
|---|---|
| `scripts/build_benchmark_v4_current_snapshot.py` | `b0451766575f0023d42d598caa23aebb0e81cff5fcb60f5071da64c9b3f0b19b` |
| `scripts/build_benchmark_v3_evidence.py` | `9ebf636101de6cd73e4079fbcc14b012e655fdd6ff08910e00127ee915718dcc` |
| `src/xgb_matcher/blocking.py` | `e6a0fded2f6496c9f4e901d8ba4fca1b912f5410c3c506a170c434ec02a55736` |
| `src/xgb_matcher/features.py` | `839f55b0d8c56e22e75758db88647c910fd8158039d1b0175f9c818e5ac0b191` |
| `src/xgb_matcher/naming.py` | `b7ef59a8cb7529179567f6e3ffe3b64757383a9e449a0110886abe640a1b5fc1` |
| `src/xgb_matcher/partitioned_store.py` | `181d1c8a56539f6b36e01d9fc040a7fb4135e28a0b10147775abd5b33837a39f` |

La parité V4 est stricte : l'état actif provient du champ `etat_admin` des
partitions gelées, de signature
`2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82`.
Le snapshot SIRENE
`c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845`
est épinglé pour l'hydratation et le contexte, mais ne remplace pas cet état.
Toute autre source d'état constitue une nouvelle politique interdite ici.

Un candidat est direct lorsque :

1. son nom et son adresse sont tous deux forts ;
2. il possède au moins une ancre exacte de nom normalisé ou d'adresse
   canonique.

Nom fort :

- nom normalisé exact ; ou
- Jaro supérieur ou égal à `0.85` et recouvrement de tokens supérieur ou
  égal à `0.50` ; ou
- contenance/acronyme avec Jaro supérieur ou égal à `0.75`.

Adresse forte :

- adresse canonique exacte ; ou
- même code postal, Jaro de rue supérieur ou égal à `0.90` et numéros
  compatibles.

La recherche porte sur tout l'univers actif de la partition géographique,
pas seulement sur le top-100. Elle utilise INSEE en priorité, code postal en
secours et l'intersection INSEE/code postal pour une mégapole, conformément à
la politique V4.

Aucun label, vérité CRM, résultat de ranker ou score d'accepteur n'entre dans
ce calcul.

`direct_candidate_count` peut dépasser 100. Les preuves complètes ne sont
jamais injectées dans le pool du ranker : seuls les agrégats de la section 4
atteignent la garde. Le ranker conserve son pool distinct, actif, unique et
plafonné à 100 candidats.

## 4. Sortie de preuve

Pour chaque requête, le build label-free produit :

```text
query_id
partition_key
active_universe_count
direct_candidate_count
direct_siren_count
sole_direct_siret
sole_direct_siren
cross_siren_direct_collision
same_siren_direct_multisite
evidence_refs_json
```

Les preuves candidates, conservées séparément, contiennent :

```text
query_id
candidate_siret
candidate_siren
candidate_state
exact_name_anchor
exact_address_anchor
strong_name_evidence
strong_address_evidence
direct_evidence_class
direct_match_rule
```

`sole_direct_siret` et `sole_direct_siren` sont non nuls uniquement lorsque
`direct_candidate_count == 1`.

Les sorties sont fermées et hashées avant toute ouverture des labels
historiques.

## 5. Règle de décision V4.12-G

La décision brute V4.11 est calculée sans modification.

```text
si decision_v411 != AUTO_MATCH:
    REVIEW avec la raison V4.11
sinon si direct_candidate_count == 0:
    REVIEW / NO_DIRECT_EVIDENCE
sinon si direct_candidate_count >= 2:
    REVIEW / MULTIPLE_STRONG_DIRECT_CANDIDATES
sinon si sole_direct_siret != predicted_siret:
    REVIEW / DIRECT_EVIDENCE_DISAGREES_TOP1
sinon:
    AUTO_MATCH
```

Si le candidat direct unique est absent du top-100, il diffère nécessairement
du top-1 et produit `DIRECT_EVIDENCE_DISAGREES_TOP1`. Aucun candidat de preuve
ne peut être réinjecté dans le pool.

La garde :

- ne remplace jamais le top-1 ;
- ne transforme jamais `REVIEW` en `AUTO_MATCH` ;
- n'utilise ni SIRET CRM source ni label ;
- ne distingue pas artificiellement les collisions inter-SIREN et les
  multi-sites d'un même SIREN : deux candidats directs suffisent à refuser.

## 6. Populations historiques autorisées

Les seules populations de développement sont celles déjà gelées dans les
artefacts V4.11 :

| Population | Total | `MATCH_EXACT` | `AMBIGUOUS` |
|---|---:|---:|---:|
| fit OOF | 5 547 | 4 666 | 881 |
| threshold dev | 710 | 583 | 127 |
| comparison dev | 746 | 634 | 112 |

Identités obligatoires :

| Entrée | SHA-256 |
|---|---|
| plan V4.11 `config/v4_11_training_plan.json` | `49299bd98f350abb90f159915a5991d88af25a307ab73b951826beb49cc571b4` |
| contrat V4.11 | `3b68fa44aaea2ee166688945ca53518e8c6bc0cf7b3c3f2a31ebccab7c73fb8d` |
| manifeste retrieval V4.11 | `445bf15d0a8f950c213764a104c05f8263bcfba7b7391c9df247d0e5873e6280` |
| queries V4.11 | `3a47aef768cee1436ad77a6e114defe50e685b7495f0e75137e9fd06dfe9fc68` |
| candidats V4.11 top-100 | `78b2f78ddeac863ac39ca64301d42312c7fb766ac51e2b5d19dde5c5910aedac` |
| labels historiques V4.11 | `69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04` |
| splits V4.11 | `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` |
| manifeste Ranker C | `1552ab2623580f1ae68e31ec1497be8a93a1bb1f2d33114dd34cfea07a864053` |
| prédictions Ranker C OOF/dev | `f14828aafa146dc4ad0399697c9477e57930ba618a5b2d7d0a903e52c2d879c0` |
| modèle Ranker C complet | `f4b71b49ed4f879b88e05e4fb84229d0306c5e8ca96958ac20ad97fcc04349c0` |
| manifeste scènes V4.11 | `8faaf2761bb280f1ba559ea3f2c579fd5d91531a202b6b54dff79e38f0d2757e` |
| scènes V4.11 | `c0f3d670e50cb43cdd6fed3b976c95e51d70f0313ae07ed0e1e2ed01eca5bed3` |
| manifeste accepteur V4.11 | `a7fc765fe439392baec61fa8a35a941bb1f778281ccdbb54b55c699e9f0c11d9` |
| modèle accepteur V4.11 | `a804feb64f28c417adda4418724f53df50b20d3d308b3e7c778c7189d368e3cf` |
| metadata accepteur V4.11 | `e4b99676e695d19748b71a7657ff5a1f5c7dfa2879754dd2e1b15c8906a61d6b` |
| stack V4.11 | `81279978f47e1e2b1b4a1ea85d595b8dedd8ee8a073e34a19b3ffd340c945d5a` |

Le ranker reste OOF sur le fit. `threshold_dev` n'est pas réutilisé pour
choisir un seuil puisque modèle et seuil restent gelés ; il sert seulement au
contrôle de cohérence. Le gate de développement porte une seule fois sur
`comparison_dev`.

`UNRESOLVED` est absent de ces populations. Aucun `UNRESOLVED` du challenge
consommé ne peut être importé dans le développement.

Le dev est divisé par composante SIREN exactement comme en V4.11 :

- `threshold_dev` si le premier octet de
  `SHA-256("v411-threshold:" + siren_component_id)` est inférieur à 128 ;
- `comparison_dev` sinon.

`siren_component_id` provient du split épinglé. Les trois ensembles de
`query_id` doivent être recalculés et comparés exactement au parquet de scènes,
pas seulement validés par leurs volumes.

Avant fermeture des preuves, le builder ouvre uniquement
`queries.parquet` et les partitions gelées. Il lui est interdit d'ouvrir le
split, les scènes, les sorties du ranker ou l'accepteur.

Après fermeture et hash des preuves seulement :

- il ouvre `split_assignments.parquet`, y compris `siren_component_id` ;
- il projette physiquement les huit colonnes label-free autorisées du parquet
  Ranker C pour contrôler la présence dans le top-100 ;
- il ouvre les scènes et le bundle accepteur pour appliquer les décisions et
  calculer les métriques.

Il est toujours interdit de charger `is_ground_truth` ou les labels séparés
du dataset retrieval. Les phases et projections autorisées sont obligatoires
dans l'allowlist.

## 7. Anti-circularité

Les labels historiques V4 `MATCH_EXACT` et `AMBIGUOUS` sont eux-mêmes issus
de la politique de preuve directe. Par conséquent :

- zéro ambiguïté automatisée sur ces labels est un test de cohérence de la
  garde, pas une preuve statistique indépendante ;
- la garde peut être retenue comme mécanisme de sécurité explicite ;
- ses métriques historiques ne certifient pas la vérité juridique ;
- aucun modèle ne peut apprendre `direct_candidate_count`,
  `direct_siren_count` ou leurs dérivés sous ce contrat ;
- la promotion produit exige un nouvel export avec labels indépendants de la
  garde.

## 8. Gates historiques entiers

Référence V4.11 sur `comparison_dev` :

```text
N = 746
AUTO = 614
erreurs AUTO = 0
AMBIGUOUS AUTO = 0
```

Pour V4.12-G, avec `A` AUTO, `E` erreurs AUTO et `B` ambiguïtés AUTO :

- précision observée au moins 99,8 % : `500 * E <= A` ;
- couverture globale au moins 80 % : `A >= 597` ;
- non-infériorité de couverture à moins deux points : `A >= 600` ;
- non-infériorité de précision face à 614/614 : `E == 0` ;
- ambiguïtés automatisées : `B == 0`.

Gate effectif :

```text
A >= 600
E == 0
B == 0
```

La garde doit aussi :

- refuser toute scène dont `direct_candidate_count != 1` ;
- refuser toute scène où le candidat direct unique diffère du top-1 ;
- reproduire exactement la décision V4.11 lorsque la preuve autorise AUTO ;
- perdre au plus deux points de couverture par segment contenant au moins
  100 requêtes.

Les familles critiques sont celles de V4.11, sur `comparison_dev` uniquement
et en comparaison appariée avec les décisions V4.11 des mêmes `query_id` :

- chaque valeur de `input_siret_state` ;
- chaque valeur de `source_segment` ;
- `top1_siren_candidate_count > 1` et `= 1` ;
- `role_crm_count > 0` et `= 0`.

Pour toute famille `s` de taille `Ns >= 100`, avec `A0s` AUTO V4.11 et `As`
AUTO V4.12-G :

```text
100 * (A0s - As) <= 2 * Ns
```

Les familles de moins de 100 lignes sont publiées mais non bloquantes.

Un échec donne `PIVOT` ou `STOP_V412_GUARD`. Il n'autorise pas
automatiquement un nouvel accepteur.

## 9. Tests obligatoires

- mêmes entrées CRM/candidats : mêmes preuves au build et au serve ;
- parité exacte avec la politique V4 gelée ;
- calcul sur l'univers géographique actif complet ;
- `direct_candidate_count` peut dépasser 100 sans modifier le top-100 ;
- aucune preuve hors top-100 n'entre dans le ranker ;
- seuls les agrégats de preuve entrent dans la garde ;
- aucun positif injecté ;
- aucun accès au challenge consommé, protégé par denylist de hash et chemin ;
- preuve fermée avant labels ;
- avant seal : seules queries et partitions sont désérialisées ;
- split, ranker, scènes et accepteur ouverts uniquement après seal ;
- SIRET direct unique présent dans le top-100 ou refus explicite ;
- plafond retrieval toujours égal à 100 ;
- décisions V4.11 conservées séparément ;
- V4.12-G est un veto pur ;
- build et métriques reproduits bit à bit ;
- allowlist stricte des inputs et denylist chemin + hash du challenge ;
- p95 du chemin complet inférieur à deux fois la p95 V4.11 sur les mêmes
  requêtes ;
- surcoût p95 de la preuve inférieur ou égal à la p95 du retrieval V4.11 ;
- pic RSS inférieur ou égal à 8 Gio ;
- suite complète verte.

Le builder ne peut être exécuté qu'après :

1. commit du contrat, de l'allowlist et de la denylist ;
2. commit du builder et de ses tests ;
3. création d'un verrou externe épinglant ces commits, leurs SHA-256, les
   sources de politique, les inputs, le runtime, le snapshot et les
   partitions ;
4. audit indépendant `GO_BUILD_V412_EVIDENCE`.

## 10. Gel et nouvel export

Si le gate historique passe :

1. figer module de preuve, runner, bundle V4.11, règle de garde, runtime,
   snapshot, partitions et hashes ;
2. valider la parité train/serve ;
3. contre-auditer le bundle sans ouvrir de nouvelle donnée ;
4. seulement ensuite recevoir un nouvel export CRM indépendant ;
5. produire baseline V4.11 et V4.12-G ensemble ;
6. sceller les décisions avant ouverture des labels indépendants ;
7. exécuter une seule évaluation finale.

Le nouvel export doit mesurer séparément :

- couverture de qualification indépendante ;
- Recall SIRET exact à 100 ;
- Hit@1 SIRET exact ;
- couverture AUTO ;
- erreurs AUTO ;
- `AMBIGUOUS` AUTO ;
- `UNRESOLVED` AUTO ;
- raisons de REVIEW.

Les gates retrieval de la directive active restent distincts :

- couverture `MATCH_EXACT` au moins 80 % ;
- Recall@100 SIRET exact au moins 99 % ;
- une vérité absente du pool reste une erreur end-to-end.

La couche de décision vise séparément :

- couverture AUTO au moins 80 % ;
- précision SIRET exacte AUTO au moins 99,8 % ;
- zéro `AMBIGUOUS` et zéro `UNRESOLVED` automatisé.

Une certification statistique à borne unilatérale 99 %
supérieure ou égale à 99,8 % nécessite environ 2 301 AUTO sans erreur, donc
au moins 2 877 dossiers évaluables à 80 % de couverture.

Sans ce volume, publier une estimation observée, jamais une garantie.
