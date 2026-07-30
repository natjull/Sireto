# V4.12 — Audits du préenregistrement S0-R3

## Snapshot audité

- commit : `a48310781c6d07ab6e0afb9f69d97754a565ebb7` ;
- plan : `config/v4_12_fresh_s0_r3_plan.json` ;
- SHA-256 plan :
  `ce7f8ed4a9d6236e61cffca72b92a1043d414afc69571ae79c94f191e6def1e2` ;
- contrat : `docs/v4_12_fresh_s0_r3_contract.md` ;
- SHA-256 contrat :
  `247b41f60a39211f85431d141625bf0d8321ae88c701d17ffd380a04ef7a9353`.

La racine
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r3`
était absente pendant les deux audits.

## Verdicts indépendants

Les deux audits post-commit concluent :

```text
GO_R3_IMPLEMENTATION
GO_R3_IMPLEMENTATION
```

Les audits ont recomposé l'overlay depuis le plan R2 épinglé : 30 overrides
et quatre suppressions. Ils ont vérifié que :

- les cinq schémas remplacés matérialisent champs, nullabilité et types ;
- l'identité, l'attempt, le contrôle, le receipt prédécesseur et le gate sont
  cohérents ;
- `SANDBOX_EXEC` reste une source système et n'est pas un blob Git ;
- aucune autorité d'identité R2 interdite n'alimente R3 ;
- seule la frontière runtime R2 explicitement déclarée est héritée ;
- seul `INGESTED` peut ouvrir le chantier CRM frais.

## Tests

- tests du plan R3 : `11 passed` ;
- suite complète sur le TMPDIR du SSD : `1106 passed` en 38,10 secondes.

## Portée du GO

Ce gate autorise l'implémentation du builder, du sealer, du launcher, du
worker et des tests R3. Il n'autorise pas encore la création de la racine R3,
le run autoritatif, la lecture d'un CRM réel, le retrieval, les modèles ni le
test final.
