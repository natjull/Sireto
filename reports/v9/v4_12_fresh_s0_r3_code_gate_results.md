# V4.12 — Résultat du gate code S0-R3

## Snapshot

- commit d'implémentation :
  `8d8e0a327983c80b35f7a4e1630b4335fab48fea` ;
- plan R3 :
  `ce7f8ed4a9d6236e61cffca72b92a1043d414afc69571ae79c94f191e6def1e2` ;
- contrat R3 :
  `247b41f60a39211f85431d141625bf0d8321ae88c701d17ffd380a04ef7a9353`.

## Gate jetable

Artefact :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/diag-r3-successor-gate.sfzj9buk/audit/kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl/parent/gate_results/successor_gate_result.json
```

SHA-256 :

```text
556558d4372b003d23190b86ff8163e021e0a937b83539b0fcc1e4828b53185b
```

Résultat :

- `GO / INGESTED_SYNTHETIC_SCANNER_SEALER_V412` ;
- identité run et attempt R3 exactes ;
- 11/11 canaris refusés avec `EPERM` ;
- stabilité monotone réelle : `60.010024083` secondes ;
- mêmes descripteurs de payload ;
- stdout et stderr vides ;
- arbres entrée/sortie scellés ;
- trois générations de journal chaînées ;
- identité R1 rejetée avec
  `IDENTITY/RUN_DERIVATION_MISMATCH`, sans mutation de sortie.

Le worker SHA-256
`a68c0ea73ccab927677cd40adb588bff5eaa2d9504875f20b01a8e8b57e6c110`
est exactement le blob du commit d'implémentation. Les 1 525 fichiers du
runtime privé et sa fermeture
`9c5a8eb5ddb69337b398bf33c5f49657af556068c3507054f8215f53db199444`
ont été rehashés.

## Audits et tests

Deux audits indépendants concluent :

```text
GO_R3_CODE_BUNDLE
GO_R3_CODE_BUNDLE
```

- tests R3 ciblés : `106 passed` ;
- suite complète sur le SSD : `1078 passed, 62 skipped`, zéro échec.

La racine autoritative R3 est restée absente pendant le gate et les audits.
Le prochain geste autorisé est sa construction unique par le builder R3,
avant sealer, audit du lock et autorisation séparée.
