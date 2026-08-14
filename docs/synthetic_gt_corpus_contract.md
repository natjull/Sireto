# Contrat préenregistré — corpus GT synthétique SIRETO

Date de gel : 15 août 2026  
Statut : contrat de construction train-only ; aucun modèle, retrieval ou test
final n'est ouvert par ce cycle.

## 1. Objet et frontières

Ce cycle construit un corpus de développement destiné à des entraînements
ultérieurs BGE, CamemBERT ou FusionSet. Il produit deux objets séparés :

1. des requêtes CRM bruitées dont la vérité `target_siret` est un SIRET connu
   dans une source autorisée ;
2. des paires query–candidat comprenant le positif réellement présent et des
   hard negatives provenant du même snapshot SIRENE.

Le corpus est un artefact d'apprentissage uniquement. Il ne modifie ni le
retrieval, ni le ranker, ni le decider, ni l'accepteur, ni aucun test final.
Il ne peut pas être joint à une évaluation pour injecter un positif, choisir
un rang, qualifier une ligne ou remplacer une vérité absente du pool.

Les branches physiques et sémantiques sont disjointes :

- `SIRENE_SYNTHETIC` : génération déterministe locale depuis `crm_ok_gt` et
  SIRENE ;
- `MAPS_ASSISTED` : branche optionnelle séparée, uniquement après son avenant,
  jamais mélangée aux lignes SIRENE et jamais utilisable pour validation.

## 2. Sources autorisées et pins de ce cycle

Les chemins et hashes qui autorisent une exécution sont ceux du plan
`config/synthetic_gt_corpus_plan.json`. Toute dérive d'octets, de schéma ou de
cardinalité produit `STOP` avant génération.

Sources de vérité et de contexte :

- `data/crm_ok_gt.csv` : 17 054 lignes, clé de jointure `query_id` égale à
  l'index CSV zéro-based, `gt_siret` comme vérité historique fournie par le
  fichier ;
- `data/StockEtablissement_utf8.parquet` : snapshot SIRENE des établissements,
  utilisé pour vérifier le SIRET cible et construire les candidats ;
- `data/StockUniteLegale_utf8.parquet` : snapshot SIRENE des unités légales,
  utilisé uniquement pour les dénominations, sigles et formes juridiques
  lorsqu'une jointure SIREN unique et non vide existe ;
- `fold_assignments.parquet` de la population V4.12-L : uniquement pour
  appliquer l'assignation SIREN-disjointe et la restriction de population.

SIRENE est une source administrative ouverte et gratuite. La copie locale est
traitée comme un snapshot immuable et sa licence/provenance est publiée dans
le manifeste, avec date et URL officielles ; aucune API payante n'est requise.
RNE/INPI, Annuaire des Entreprises et autres sources ouvertes peuvent être
ajoutés dans un cycle ultérieur seulement avec un pin, une licence et un
schéma déclarés. Aucun appel réseau n'est nécessaire pour ce cycle SIRENE.

## 3. Population train-only et étanchéité

La population de seeds est construite par jointure one-to-one entre
`crm_ok_gt.csv` et les assignments, puis conserve exclusivement :

```text
legacy_split == "train"
oof_fold in {2, 3, 4}
gt_siret conforme à ^[0-9]{14}$
```

Les lignes `dev`, `test`, les folds 0 et 1, les 43 ajouts frais et toute ligne
non jointe sont exclus. Les composants SIREN associés aux lignes exclues sont
mis dans une denylist calculée avant la génération. Le builder vérifie qu'aucun
SIREN du seed, du positif ou d'un hard negative ne se trouve dans cette
denylist. La présence d'un composant dans plusieurs folds est une erreur
bloquante ; les composants sont traités comme des atomes.

Pour atteindre le minimum quantitatif sans recycler les mêmes identités, le
builder peut compléter les 7 099 seeds `crm_ok_gt` par des seeds
`SIRENE_ONLY_TRAIN`. Ceux-ci sont sélectionnés directement dans le snapshot
SIRENE, avec SIRET/SIREN, nom et localisation valides, et doivent être absents
de **tous** les SIREN observés dans `crm_ok_gt` (train, dev et test) ainsi que
des composants interdits. Leur composant synthétique est un atome
`SIRENE_ONLY:<siren>` affecté de façon déterministe à 2, 3 ou 4 ; il n'est
jamais présenté comme une ligne historique `crm_ok_gt` et ne peut ouvrir un
fold 0/1 ou le test.

Le test final historique reste fermé. Aucun fichier de test final, oracle,
résultat, hit, rang, score ou sortie de modèle n'est une entrée du builder.
La qualification du seed utilise uniquement les sources et les règles
ci-dessus, jamais un résultat de retrieval.

## 4. Vérité indépendante et positivité

Un positif est admissible seulement si :

- `gt_siret` est un SIRET décimal de 14 caractères ;
- les neuf premiers chiffres forment le SIREN cible ;
- le SIRET existe exactement une fois dans le snapshot SIRENE ;
- le SIREN cible appartient au train autorisé ;
- les champs utilisés pour le rendu sont lus du snapshot SIRENE ou du CRM
  observé et leur provenance est conservée.

La génération ne prend jamais un candidat retrouvé, un score, un rang ou un hit
pour créer une vérité. Elle ne recopie pas le SIRET dans les champs CRM
bruités. Une séquence autonome de 9 ou 14 chiffres dans un champ texte CRM,
après NFKC et avec bornes non décimales, est rejetée. Les numéros de voie
ordinaires restent permis.

Un `SIRET` est une identité de sortie/label, pas une partie de l'entrée texte
transmise au futur modèle. Les builders ultérieurs doivent déposer les colonnes
d'identité avant tokenisation.

## 5. Génération déterministe

Le générateur est versionné dans le dépôt, seedé par `42`, et ne dépend ni de
l'heure, ni de l'ordre d'énumération du filesystem, ni d'un modèle, ni du
réseau. Les choix aléatoires utilisent un flux dérivé de
`SHA256(global_seed, seed_id, family, variant_index)` ; chaque transformation
est donc reproductible et auditée par son nom et ses paramètres.

Les vues de départ autorisées sont `CRM_OBSERVED` et, lorsque présents,
`SIRENE_OFFICIAL_NAME`, `SIRENE_OFFICIAL_ENSEIGNE` et
`SIRENE_OFFICIAL_ADDRESS`. Une vue officielle ne devient pas une preuve
supplémentaire : elle est une variante synthétique issue du même snapshot et
son origine est publiée.

Familles prévues :

- formes juridiques : suffixes usuels, formes longues/courtes et retrait
  contrôlé d'une forme, uniquement lorsque la source contient cette forme ;
- sigles et tokenisation : initiales conservant les tokens informatifs,
  séparateurs espaces/tirets/points, ordre de tokens avec garde de stabilité ;
- accents et ponctuation : retrait des accents, apostrophes, parenthèses,
  tirets, casse et espaces multiples ;
- OCR réaliste : substitutions limitées `O/0`, `I/1`, `S/5`, `B/8` seulement
  dans des tokens alphabétiques assez longs, jamais dans un identifiant ;
- enseigne versus dénomination : remplacement par une enseigne officielle ou
  une dénomination usuelle non vide, avec provenance `SIRENE_OFFICIAL` ;
- adresse : abréviations rue/avenue/boulevard, ponctuation, tokenisation,
  ordre de composants, variantes de voie/commune, accents, numéro et indice ;
- champs manquants : retrait d'un champ non critique, tout en conservant au
  moins un ancrage géographique et un signal nom/adresse ;
- commune et voie : variantes de casse, accents, traits d'union et formes
  administratives observées dans SIRENE ;
- historique officiel : réservé à une future source historisée explicitement
  pinnée ; aucune reconstruction historique n'est inventée depuis le snapshot
  courant.

Chaque variante porte `corruption_family`, `base_view`, `parameters_json` et
`confidence_weight`. Les transformations qui ne s'appliquent pas à une ligne
sont comptées comme non applicables, pas remplacées par une duplication.

## 6. Garde-fous de fidélité et d'identifiabilité

Avant publication d'une variante, le builder vérifie :

1. la validité UTF-8, l'absence de NUL et l'absence de fuite SIRET/SIREN ;
2. au moins un champ géographique parmi INSEE/CP/commune, et au moins un
   signal informatif nom/adresse ;
3. la présence exacte du positif dans SIRENE ;
4. l'absence de collision de la requête bruitée avec une autre requête cible ;
5. une marge lexicale minimale du positif face aux établissements locaux,
   calculée exclusivement avec règles déterministes sur les champs SIRENE,
   sans ranker, retrieval, hit, rang ou score de modèle ;
6. l'absence d'un autre établissement dont les signaux nom/adresse et
   localisation sont simultanément aussi bons ou meilleurs selon la même
   fonction de garde ;
7. le respect de la denylist de composants SIREN.

Une variante douteuse est rejetée avec une raison codée et conservée dans le
ledger de contrôle ; elle n'est jamais réparée en lisant un résultat modèle.
Une cible hors snapshot ou un garde-fou impossible à évaluer est `STOP`, pas
une ligne SILVER.

## 7. Hard negatives et provenance

Pour chaque query publiée, les candidats sont dédupliqués par SIRET, triés par
catégorie puis par SIRET ASCII, et sont limités à un groupe de 16 lignes : un
positif réellement présent et au plus 15 négatifs. Les catégories sont :

1. `SAME_SIREN_OTHER_SIRET` : autre établissement du même SIREN ;
2. `LOCAL_HOMONYM` : nom/dénomination/enseigne fortement égal dans la même
   commune ou le même CP ;
3. `SHARED_ADDRESS` : adresse normalisée partagée avec un autre SIREN ;
4. `ACTIVE_CLOSED_COMPETITOR` : même zone avec état A/F contrasté ;
5. `TOPOLOGICAL_NEARBY` : même voie/CP ou même commune/INSEE, avec adresse
   distincte et géographiquement/topologiquement proche selon les seuls
   champs SIRENE ;
6. `LOCAL_FILL` : complément local déterministe quand les catégories
   prioritaires ne remplissent pas le groupe.

Un négatif n'est pas dérivé d'un score de retrieval. Chaque ligne conserve
`negative_category`, `candidate_siret`, `candidate_siren`, état, date du
snapshot, champs SIRENE projetés, `source_snapshot_sha256`, `seed_query_id`,
`target_siret`, `target_siren`, `oof_fold`, `siren_component_id` et un digest
canonique de provenance. La catégorie est descriptive et ne constitue pas une
règle de décision.

## 8. Schémas et artefacts

Le corpus principal contient au minimum :

```text
example_id, seed_query_id, source_kind, base_view,
crm_name, crm_address, crm_postcode, crm_city, crm_insee,
target_siret, target_siren, target_state,
oof_fold, siren_component_id, corruption_family,
variant_index, confidence_weight, guard_status,
provenance_digest, generator_version
```

Le groupe candidat contient au minimum :

```text
example_id, candidate_siret, candidate_siren, is_positive,
negative_category, candidate_state, candidate_name, candidate_address,
candidate_postcode, candidate_city, candidate_insee,
target_siret, source_kind, source_snapshot_sha256,
candidate_provenance_digest
```

Les identités sont séparées des textes d'entrée. Les sorties lourdes sont
publiées exclusivement sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/`.
Chaque build possède un `build_id` dérivé des hashes du plan, du générateur,
des tests et des sources ; il ne remplace jamais un build existant.

Le manifeste ferme : plan, code, tests, versions runtime, sources, counts,
hashes d'octets, hashes logiques, seed, quotas, familles, rejets, denylist,
licences et résultat des audits. Le manifest est non auto-référent.

## 9. Pilote, montée en volume et audit distributionnel

Le pilote est exécuté avant tout build complet. Il utilise un échantillon
déterministe de seeds train autorisés, couvre toutes les familles applicables,
et publie : taux de production/rejet, causes de rejet, diversité, doublons,
états A/F, hard-negative categories, marges de garde et provenance.

Le rapport compare les distributions du train source et du corpus par :

- présence nom/adresse/CP/ville/INSEE ;
- longueurs et nombres de tokens ;
- accents, ponctuation, formes juridiques, enseigne/dénomination ;
- états actifs/fermés, CP/INSEE et types de seed ;
- familles de corruption et nombre de variantes uniques par seed.

Les critères de montée sont préenregistrés : zéro fuite, zéro overlap de
composant, zéro positif injecté, 100 % de positivité et de provenance valides,
déduplication exacte, au moins 80 % des familles applicables couvertes,
duplicate rate ≤ 5 %, et taux de variantes uniques ≥ 85 %. La diversité
marginale doit rester ≥ 40 % de nouvelles signatures structurelles au pilote
et ≥ 20 % à chaque extension ; sinon le volume est plafonné.

`GO_PILOT` autorise une extension limitée si tous les invariants passent.
`PIVOT_PILOT` signale un corpus exploitable mais une famille ou une garde à
revoir. `STOP_PILOT` couvre toute fuite, contamination de fold/test,
provenance manquante, mutation de source, collision non expliquée ou
positivité injectée.

Il n'existe aucun objectif de volume fixe : la taille finale s'arrête quand la
diversité utile marginale échoue, pas quand une duplication artificielle est
atteinte.

## 10 bis. Objectifs quantitatifs préenregistrés

Après les contrôles du pilote, les objectifs minimaux sont :

- au moins **20 000 SIRET seed distincts**, tous train autorisés et hors de
  tout SIREN des folds 0/1 ou du test ;
- au moins **3 variantes CRM réalistes, distinctes et non triviales par
  SIRET**, soit au moins **60 000 couples positifs** ;
- de **5 à 10 hard negatives par positif**, soit une cible de **300 000 à
  600 000 paires négatives**, avec quotas publiés par famille.

Le stretch est de 100 000 couples positifs uniquement si les mêmes gates de
diversité, identifiabilité, fidélité, déduplication et anti-fuite passent. Les
permutations superficielles, répétitions ou variantes quasi identiques ne
comptent pas. Le rapport doit publier le nombre de seeds distincts, positifs
uniques, négatifs par famille, rejet, doublons et transformations. Si le
minimum n'est pas atteignable proprement, le verdict est `PIVOT` avec le
maximum valide obtenu.

## 11. Ressources et interdits

Calcul : Mac M4 Pro et `/Volumes/CATNAT_DATA` uniquement. Aucun GPU loué,
service d'annotation, API payante ou dépense externe n'est requis par la
branche SIRENE. Réseau, Google Maps et contenu Google sont interdits dans cette
branche. Le builder ne modifie aucune source existante ni aucun artefact sale.

Chaque milestone de code ou de métier est livré dans un commit isolé et cité
dans `handover.md`.
