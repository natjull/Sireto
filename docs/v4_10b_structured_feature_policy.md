# Amendement V4.10b — politique de features structurées

Statut : préenregistré après l'audit statistique pré-fit V4.10 et avant tout
entraînement.

Identifiant : `V410B_STRUCTURED_FEATURE_POLICY`.

Ce document amende uniquement les sections 5, 6 et 7 du contrat
`docs/v4_10_structured_acceptor_contract.md`. Le dataset V4.10
`0d6b87fd50fb550c` et son plan d'entraînement sont déclarés `superseded` :
aucun modèle n'a été entraîné avec eux. Toutes les autres interdictions,
populations, folds, gates et règles de gouvernance du contrat V4.10 restent
applicables.

## 1. Motif

L'audit sans labels a détecté :

- des copies sémantiques simultanément présentes dans l'ordre structuré ;
- des signaux propres à l'instrumentation retrieval V4.1 ou V4.2-B, capables
  d'identifier la population au lieu de la justesse du top-1 ;
- une standardisation des compteurs insuffisante pour une régression
  logistique pénalisée.

Le correctif est entièrement déterminé par les formules et provenances. Il
n'utilise ni cible, ni split, ni score de performance.

## 2. Invariant CURRENT80

`CURRENT80` reste strictement inchangé :

- 80 features ;
- ordre SHA-256
  `e50086608ca3e60071e2575fbd8a0ca7c8ba99fe87251894ee04bf9b1b57cfe5` ;
- valeurs bit à bit identiques au build V4.10 ;
- `StandardScaler` sur les 80 colonnes, comme en V4.8.

Les exclusions ci-dessous concernent uniquement `structured_feature_order`.
Toutes les colonnes restent physiquement présentes dans les parquets pour
l'audit.

## 3. Copies sémantiques exclues

Pour chacune des 17 bases suivantes, conserver le triplet `scene_top1_*`,
`scene_top2_*`, `scene_delta_*` et exclure les trois projections
`candidate_top1_*`, `candidate_top2_*`, `candidate_delta_*` :

```text
name_jaro_max
name_token_overlap_max
idf_name
numeric_token_match
name_contains_crm_max
name_crm_contains_cand_max
name_sim_max_etab
name_sim_max_pm_dirigeant
name_length_max
addr_jaro
postcode_match
city_match
street_number_diff
addr_token_overlap
address_density
street_name_jaro
name_addr_consistency
```

Cela exclut 51 colonnes. Exclure également les sept copies ou tautologies
suivantes :

```text
same_siren_candidate_count
same_siren_best_ranker_score
same_siren_best_sibling_ranker_score
top1_is_same_siren_best_ranker_score
candidate_top1_is_sigle_max
candidate_top2_is_sigle_max
candidate_delta_is_sigle_max
```

Le catalogue contient une `alias_of` explicite pour les 58 colonnes et le
builder vérifie leur égalité définitionnelle avec leur représentation
canonique sur chaque ligne. Toute divergence provoque
`STOP_DATASET_INTEGRITY`.

Il est interdit de dédupliquer par corrélation ou égalité observée. Les
deltas, interactions, hiérarchie NAF, agrégats métier, catégories rares et
indicateurs de manque restent présents lorsqu'ils ne figurent pas dans cette
liste.

## 4. Instrumentation retrieval exclue

Les 16 colonnes suivantes deviennent `audit_only` pour le modèle structuré :

```text
scene_top1_retrieval_channel_count
scene_top1_retrieval_agreement
scene_top1_rrf_score
scene_sparse_dense_top1_agreement
scene_retrieval_disagreement
scene_retrieval_miss
candidate_top1_retrieval_channel_count
candidate_top2_retrieval_channel_count
candidate_delta_retrieval_channel_count
candidate_top1_retrieval_agreement
candidate_top2_retrieval_agreement
candidate_delta_retrieval_agreement
candidate_top1_retrieval_channel_count_missing
candidate_top2_retrieval_channel_count_missing
candidate_top1_retrieval_agreement_missing
candidate_top2_retrieval_agreement_missing
```

Elles s'ajoutent aux 75 features de provenance déjà interdites en V4.10.
`ranker_score`, les signaux de contenu, l'activité, les rôles et la
constellation restent autorisés.

Après retrait des 58 copies et des 16 signaux retrieval, l'ordre structuré
contient exactement 641 features, conserve l'ordre relatif V4.10 et possède
le SHA-256 :

`4ff0eb4e8cc33850742bf4d9c0ddb599cc9abb500d6b60bb3e5dc6a80b9cd13b`.

## 5. Catalogue et prétraitement

Le nouveau catalogue publie pour chaque colonne :

- `current80_allowed` ;
- `structured_allowed` ;
- `structured_exclusion_reason` ;
- `alias_of`.

Il publie aussi les listes et hashes des alias, exclusions retrieval,
features structurées standardisées et non standardisées.

Pour `STRUCTURED_LOGIT`, le scaler est appris uniquement sur le train propre
au modèle ou au pli :

- standardiser les kinds `continuous` et `count` ;
- laisser les kinds `binary` inchangés.

Les deux ordres doivent être disjoints et leur union ordonnée doit reproduire
exactement les 641 features. `STRUCTURED_XGB` ne reçoit aucun scaler.

## 6. Précisions du gate et de la sélection

Pour chaque ligne hard OOF, `AUTO` est calculé avec le seuil propre au modèle
du pli qui a produit son score. Les cinq seuils sont publiés. Le seuil du
modèle complet sert uniquement à ses scores historiques et à son futur
bundle.

Les gates historiques utilisent des entiers :

- précision non inférieure au baseline :
  `correct_auto * 1184 >= auto_count * 1182` ;
- perte de couverture maximale :
  `1184 - auto_count <= 29`.

Toutes les variantes franchissant le gate sont gelées comme
`fresh_dev_eligible`. Le tie-break produit seulement un ordre provisoire ;
il n'élimine aucune autre variante éligible. Le futur dev frais choisira
entre ces variantes préenregistrées. `BASE_FROZEN` reste comparateur
seulement.

## 7. Artefact V4.10b

Le builder produit un nouvel artefact immuable et ne modifie jamais
`0d6b87fd50fb550c`. Son manifeste doit publier :

- `supersedes_build_id: 0d6b87fd50fb550c` ;
- `supersession_reason:
  PREFIT_STATISTICAL_AUDIT_INVALIDATED_STRUCTURED_ORDER` ;
- les hashes et comptes CURRENT80, structured, alias, retrieval audit-only,
  scaled et unscaled ;
- zéro fit, zéro seuil, zéro random, zéro fresh et zéro test.

Un nouveau plan V4.10b doit épingler ce manifeste et tous ses hashes avant le
premier fit. L'ancien plan est non exécutable.
