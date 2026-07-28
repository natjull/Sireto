# V4.12 — Résultats des entrées sûres du moteur unitaire

## Verdict

`GO_V412_UNIT_INPUTS`

Contre-audit indépendant : `GO_V412_UNIT_INPUTS_AUDIT`.

Ce verdict autorise uniquement le préenregistrement d'un oracle physiquement
séparé. Il n'autorise encore ni le store runtime, ni le moteur unitaire, ni
le benchmark de latence.

## Exécution

- build ID :
  `ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e` ;
- durée murale observée : 24,55 secondes ;
- commit audité : `c97c737d7d49482ea034217310eee710067b4d55` ;
- verrou :
  `e794c60fe31348b8422815f48514acad41843c0ee9ab0453feac1cc9e9def315`.

Paquet runtime :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e
```

Preuve d'audit séparée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_inputs/
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e
```

## Contenu runtime

| Fichier | Lignes | SHA-256 |
|---|---:|---|
| `queries_all.parquet` | 7 003 | `2f84eea594ed209042eed7b758f1c2390089cbb32cbe5dd832eff7dd80272fb9` |
| `queries_dev.parquet` | 1 456 | `b1fef6ba72e4a557175a60c7e21e658f8fbb739b1d6607b77e5f70da873a031f` |
| `partition_inventory.parquet` | 4 119 | `d17621743fad9c9c18ab46d2ba137521e4f0cfb1ccb44de8d08fca243c925bfe` |
| `tfidf_inventory.parquet` | 1 454 | `845e332df405b3e9d631923ccf17661c5420be8b9518fe68583766b7b4f6881d` |

- `integrity.json` :
  `195083c8fdc26a72db47b7deaebe106004a625afb904de4bcdebaffd15932aa1` ;
- `runtime_manifest.json` :
  `34e7a8d1c97c962b7f2295a99cd45b68c37afa9c866a5eb22e302330cb9010c4`.

Les six champs CRM sont égaux cellule par cellule à la projection physique
des sources. Les schémas sont non nullables, sans metadata Arrow, et les
ordres/payloads LF sont exacts.

## Preuve indépendante

L'audit a été réalisé en lecture seule sans importer le builder. Il a passé
21 161 assertions de recomputation :

- 4 119 partitions rehashées, 27 594 915 lignes déclarées et inventaire
  `680f1884...4463` exact ;
- 1 454 paires pickle/sidecar, soit 2 908 fichiers et 6 730 554 690 octets,
  avec inventaire `589360b1...83ce` exact ;
- ledger exhaustif de 7 029 fichiers, tailles et hashes avant/après égaux ;
- build ID, verrou, sources courantes, blobs Git, manifests et provenance
  redérivés ;
- répertoires `0555`, fichiers réguliers `0444`, même filesystem ;
- aucune colonne, fichier ou valeur runtime de label, oracle, modèle,
  résultat candidat ou chemin absolu sensible.

La preuve séparée est scellée par :

- `data_inputs.parquet` :
  `84d763e5d3fcb5af22515b7856e960b367a2626a7b463f124ae12f87e71618e8` ;
- `provenance.json` :
  `74ac9c83f5522038ef2b07884580786ec14020341bca01e0dd97174f6d7386bc` ;
- `manifest.json` :
  `c88c24c4a42188b3afff074f4d5519f4a65a8a7910d3063765a9d519430383f1`.
