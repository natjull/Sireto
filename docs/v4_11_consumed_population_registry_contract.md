# Contrat V4.11-A — registre des populations CRM consommées

Statut : préenregistré avant toute sélection de cohorte V4.11.

Identifiant : `V411_CONSUMED_POPULATION_REGISTRY`.

## 1. Objet

V4.11 doit aligner le retrieval V4.2-B, le ranker et l'accepteur. Une future
preuve de généralisation ne peut toutefois réutiliser aucun dossier ayant déjà
servi à qualifier, entraîner, calibrer, sélectionner, diagnostiquer ou évaluer
une version antérieure.

Ce contrat construit donc un registre canonique des lignes CRM consommées. Il
n'utilise aucun score, rang, hit ou résultat de modèle.

## 2. Entrées épinglées

| Rôle | Chemin | SHA-256 |
|---|---|---|
| CRM source | `data/entrainements.csv` | `f770215cd0d0fcc654b750b90dbba835acbf4efb5c74ed269d339e046c2b049d` |
| benchmark fermé historique | `/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks/closed/c33b80855f560074/benchmark.parquet` | `4c533813218dced6627da238b885db47e45745d784ae9078a4aaa836680308b6` |
| pool V4-Fresh complet | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f/pool/benchmark.parquet` | `0effe19ae7f649ee6a03e73c0858d8f87710015f630971495a8cc5a2461b8279` |

Le benchmark fermé est exclu dans son intégralité, y compris son ancien test.
Le pool V4-Fresh est exclu dans son intégralité, y compris `fit_addition`,
`dev_new` et `holdout_sealed`. Les artefacts V4.1 à V4.10 sont des dérivés de
ces deux sources et ne créent pas une troisième population.

Toute divergence de hash produit `STOP_INPUT_DRIFT`.

## 3. Identité canonique d'une ligne

Les huit champs CRM sont lus comme chaînes, sans interprétation automatique :

```text
SITE
CODE_POSTAL
CODE_INSEE
SERVICE ID
COMMUNE
SIRET
SITE_CLI_ADRESSE
SITE_CLI_COMMUNE
```

Pour chaque ligne, publier :

- `source_row_number`, index de la ligne de données à partir de 1 ;
- `service_id_norm`, chaîne trimée, vide conservé comme valeur absente ;
- `input_siret_norm`, exactement 14 chiffres si possible, sinon valeur
  absente ;
- `row_fingerprint_sha256`, SHA-256 de la sérialisation JSON canonique des
  huit champs après normalisation Unicode NFKC, trim, réduction des espaces et
  conversion en majuscules ;
- `source_key`, `SERVICE:<id>` si l'identifiant de service existe, sinon
  `ROW:<numéro>:<fingerprint>`.

Le numéro de ligne et l'empreinte empêchent que les 659 identifiants de
service vides soient confondus.

## 4. Règle d'exclusion

Une ligne source est `CONSUMED` si au moins une condition est vraie :

1. son `service_id_norm` non vide apparaît comme `crm_record_id` dans une
   entrée épinglée ;
2. son `input_siret_norm` apparaît comme SIRET historique/source dans une
   entrée épinglée ;
3. son empreinte CRM canonique est retrouvée dans une entrée qui transporte
   assez de champs bruts pour la reconstruire.

Les correspondances par SIRET sont nécessaires : le benchmark fermé contient
434 `crm_record_id` vides et le CRM source 659 `SERVICE ID` vides. Ici, le
SIRET CRM sert seulement de clé de lignée pour interdire une réutilisation ;
il ne devient ni vérité terrain, ni feature modèle, ni preuve de matching.

Une ligne n'est `UNSEEN` que si aucune règle ne la relie aux deux populations
consommées.

## 5. Sorties

Le build immuable écrit sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/<build_id>/` :

- `source_registry.parquet`, une ligne par ligne du CRM source ;
- `consumed.parquet` ;
- `unseen.parquet` ;
- `manifest.json`.

Le manifeste contient les hashes d'entrée et de sortie, le hash du code, les
volumes, les motifs de consommation, les recouvrements entre sources et les
invariants ci-dessous.

Le `build_id` est constitué des 16 premiers caractères du SHA-256 d'une
spécification canonique comprenant le schéma, les hashes d'entrée, la version
de normalisation et le hash du script.

## 6. Invariants attendus

Le build est valide uniquement si :

- le CRM contient exactement 23 609 lignes ;
- chaque ligne source a un numéro, une empreinte et une clé ;
- le benchmark fermé contient exactement 17 054 lignes ;
- V4-Fresh contient exactement 6 330 lignes ;
- les deux populations ne se chevauchent pas au niveau de la ligne source ;
- leur union consomme exactement 23 384 lignes source ;
- exactement 225 lignes restent inédites ;
- aucune ligne n'est silencieusement perdue ;
- `CONSUMED` et `UNSEEN` sont disjoints et leur union reproduit le CRM ;
- les 225 lignes inédites ont toutes un `SERVICE ID` absent, constat publié
  comme biais de sélection et jamais utilisé comme critère d'admission.

Tout écart produit `STOP_REGISTRY_INTEGRITY`.

## 7. Conséquence expérimentale

Les 225 lignes `UNSEEN` peuvent servir à un challenge de robustesse
descriptif après gel du candidat V4.11. Elles ne peuvent pas être présentées
comme une validation représentative ni comme une certification à 99,8 %.

Une preuve finale V4.11 exige un nouvel export CRM indépendant des 23 609
lignes actuelles. Ce nouvel export devra être scellé et contrôlé contre ce
registre avant tout retrieval, score ou adjudication.

