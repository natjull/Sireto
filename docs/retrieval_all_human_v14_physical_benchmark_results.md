# Retrieval all-human v14 — résultat du benchmark physique W3

Date d'exécution : 25 août 2026
Verdict : **`PASS_V14_PHYSICAL_BENCHMARK`**

## Portée

L'unique tentative W3 v14 a exécuté le DAG complet sur les 512 requêtes
scellées des folds 2/3/4 : trois acteurs build, trois fits LambdaMART
séquentiels, trois acteurs replay, compaction du top500, réhydratation
indépendante et contrôles logiques/I/O. Elle ne constitue pas une évaluation
finale de la qualité et n'a ouvert ni fold 0, ni fold 1, ni OOF national.

La seule modification par rapport à la science v12 est le protocole de hash
logique NULL-canonique v14. Sources, candidats, 319 features, modèles, caps,
gates et ordre des requêtes sont inchangés. Aucun artefact W3 v13 n'a été
repris.

## Résultat des gates

| Gate | Mesure W3 | Seuil | Verdict |
|---|---:|---:|---|
| Temps mur | 739,521 s (12 min 19,5 s) | <= 872,267 s | PASS |
| Accélération vs W1 | 2,5949x | implicite dans le gate temps | PASS |
| RAM machine maximale | 11,941 Gio | <= 14 Gio | PASS |
| Swapout | 0 octet | 0 | PASS |
| Stockage national projeté | 6,694 Gio | <= 20 Gio | PASS |
| Row-groups uniques | 368 | union W1 exacte | PASS |
| Chevauchement inter-acteurs | 0 | 0 | PASS |
| Amplification I/O logique | 1,0 | 1,0 | PASS |
| Réhydratation top500 | exacte | exacte | PASS |
| Copie physique `merged` | aucune | aucune | PASS |

La référence W1 a pris 1 918,988 s (31 min 59,0 s). W3 matérialise
156 137 585 octets (148,904 Mio), soit un ratio de 0,33623 par rapport aux
464 378 945 octets W1. Les 368 row-groups représentent 2 696 076 099 octets
logiques compressés.

Les neuf checks publiés sont vrais : `logical_bridge_exact`,
`no_physical_merge_files`, `top500_compact_top100_full`,
`compact_rehydration_exact`, `logical_io_pass`, `ram_pass`, `swap_pass`,
`speed_pass` et `storage_pass`.

## Parité fonctionnelle

Les 16 racines logiques W3 sont exactement égales à la projection W1 v12.
Elles couvrent les trois matrices train et leurs groupes, le spool heldout,
les trois modèles, outcomes, top100, top500 commun et payload complet, les
319 valeurs de features et le bridge PREFIT. La réhydratation top500 a été
reconstruite une seule fois après le chronomètre et correspond aux preuves
scellées.

Les ledgers actual-read publient 126 row-groups pour fold2, 117 pour fold3 et
125 pour fold4. Leur union est disjointe et identique à celle de W1. Le hash
intégral de la source a été recalculé après la mesure, jamais avant le marker.

## Autorités publiées

- manifeste benchmark :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v14_w3/manifest.json`
- SHA-256 benchmark :
  `d3e05b59f42aa534803573ac8423232d4b6231f4ccb60b96df063aee1fd90774`
- manifeste trial W3 SHA-256 :
  `2e5d916bcdc85117dc8a5e96a39f4357de90dd9dfe71feb1daa030c486a2147d`
- marker one-shot SHA-256 :
  `4de9fd36954982bc0b828046b70c17274ccbaffcb16324e89beed5e4d43fdb2a`
- autorisation nationale :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v14_w3/execution_authorization_v14.json`
- SHA-256 autorisation :
  `0684d350afa717a74644647f7ce7e6b43d81943e28e77e5b76d5df0c4539cfe6`
- authorization ID :
  `5ae43d08438e8666272ac1c3abeab1a01ac85a85357291885249def3c1ed1128`
- scope : `NATIONAL_OOF_FOLDS_2_3_4_V14_ONLY`

Le loader strict a rouvert le benchmark, l'autorisation, le marker, W1, W3,
les projections, la source, les gates, le code et l'environnement sans
divergence. `automatic_national_launch=false` : le PASS autorise le run
national, mais ne le lance pas.

## Prochaine décision

Le benchmark physique rapide est validé. Le prochain jalon est l'OOF national
v14 sur 23 571 requêtes, toujours limité aux folds 2/3/4. Il doit faire l'objet
d'une instruction utilisateur distincte; son runtime estimé reste de 4 à
6 heures. Les métriques de Recall décisionnelles ne seront publiées qu'après
ce calcul national et sa certification stricte.
