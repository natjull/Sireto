# V4.12 — Résultats de l'oracle unitaire séparé

## Verdict

`GO_V412_UNIT_ORACLE`

Contre-audit indépendant : `GO_V412_UNIT_ORACLE_AUDIT`.

Build ID :
`c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70`.

## Population

| Population | MATCH_EXACT | AMBIGUOUS | Total |
|---|---:|---:|---:|
| threshold_dev | 583 | 127 | 710 |
| comparison_dev | 634 | 112 | 746 |
| Total | 1 217 | 239 | 1 456 |

La couverture identifiable descriptive est 1 217 / 1 456, soit 83,5852 %.
Cet oracle est historique, non indépendant et non certifiant.

## Artefacts

Oracle :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/oracles/v4_12_unit_engine/
c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70
```

Preuve séparée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_oracle/
c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70
```

Hashes oracle :

- `oracle_dev.parquet` :
  `e0f8c66756aec95e9f54cfe180b59609869a927985f51853b676f93bbe409d6d` ;
- `integrity.json` :
  `54b1a8c624e06e03d306a26ec2ad2532b5c1490160731f180810e2907a8ad497` ;
- `manifest.json` :
  `e201a407e968566e5f75c2072c0d441581c9a6d1314698804fb916bd3c36fd25`.

Hashes audit :

- ledger huit entrées :
  `57b5b423b30b07493655dc9c1097bb9086b0ade785f54e9c15d2bb7bbbfbff0d` ;
- provenance :
  `58e4027d052723e891d250afed5e1dc68a4947d60b23280a8b9069e11cd2dbe7` ;
- manifeste :
  `a10fd6e4054a8d6e067c75e7f4c853c3ec795c0715ca705b9692992667a705b7`.

## Audit indépendant

Sans importer le builder, 4 430 contrôles ont redérivé :

- le plan, le verrou et cinq blobs Git ;
- les quatre inputs et les six fichiers du runtime sûr ;
- les 1 456 vérités, leur ordre, leurs nulls et leurs payloads ;
- le build ID ;
- le ledger ordonné de huit ouvertures ;
- les schémas, keysets, manifests, provenance et permissions.

Aucun candidat, rang, score, décision, modèle, preuve directe, challenge ou
test final n'a été utilisé pour construire l'oracle.

Ce GO autorise seulement le préenregistrement du store et du moteur unitaire,
avec sandbox macOS interdisant les racines oracle/audit au worker.
