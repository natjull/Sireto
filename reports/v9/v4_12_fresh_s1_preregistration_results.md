# V4.12 — Préenregistrement S1 de l’intake CRM frais

## Verdict

`GO_S1_IMPLEMENTATION`

Deux audits indépendants et hostiles ont validé le commit exact
`288a1def19880712b3a24ab88407ccc0d4f62e92`.

Ce verdict autorise uniquement :

- la construction et le scellement des catalogues source et evidence ;
- l’implémentation S1 sur des fixtures exclusivement synthétiques ;
- le gate synthétique multi-batch et ses audits.

Il n’autorise ni création/ouverture de l’inbox CRM réelle, ni autorisation
one-shot réelle, ni retrieval frais, ni dégel des modèles.

## Autorités

- Contrat :
  `docs/v4_12_fresh_s1_contract.md`
- SHA-256 contrat :
  `42c5c639de4dd7566d19af02f8874dbfb01ff6612ad5c6a298b7c2f2348afdc5`
- Plan canonique :
  `config/v4_12_fresh_s1_plan.json`
- SHA-256 plan :
  `2a618c9db7ea3f92c5674b4e9432a5bb06241dd1060066a9f78cb8b0a2675b8c`
- Tests :
  `tests/test_v412_fresh_s1_plan.py`
- Commit autoritaire :
  `288a1def19880712b3a24ab88407ccc0d4f62e92`

## Architecture fermée

S1 impose quatre frontières :

1. admission manifest-only, sans payload ;
2. Worker Q : CRM + registre de compatibilité, sans evidence/oracle ;
3. Worker E : bridge minimal + evidence + SIREN consommés, sans nom/adresse
   CRM ;
4. scorer : requêtes scellées uniquement.

Les catalogues utilisent un payload canonique et un seal séparé,
non autoréférentiel. Les trois manifests sont exactement typés et signés
Ed25519 via `producer_id + producer_key_id`.

La collection ne peut pas être choisie librement : un producteur unique, un
ledger signé, son hash de tête et le prochain numéro d’export attendu sont
épinglés. Le broker n’accepte aucun chemin cible et ne peut sélectionner que
le prochain export signé. Un trou attend sans payload ; un doublon arrête.

Claim, receipt, lock dynamique, marqueur pré-payload, checkpoints, événements,
seals et receipts terminaux suivent tous une écriture `O_EXCL`, puis
`fsync/F_FULLFSYNC` fichier et répertoire avant transition. La reprise ne
relance ni payload déjà potentiellement ouvert, ni retrieval déjà commencé.

Le gate de conception `GO_S1_IMPLEMENTATION` est distinct du futur
`GO_S1_REAL_CRM_OPEN`. Le retrieval one-shot exige `OPENING.json` avant query,
scellement résultats/candidats avant oracle, événements chaînés, receipt
terminal et aucun rescoring.

## Contrôles

- 12 tests S1 ciblés : verts ;
- suite complète : `1152 passed` ;
- pins matériels des deux registres : rehashés et conformes ;
- racines réelles inbox/control/sealed/audit/ready/temp/evaluation : absentes ;
- catalogues futurs et autorisation réelle : absents.

## Historique des corrections

- tests R3 adaptés à l’autorité désormais certifiée : `421ec40` ;
- préenregistrement initial S1 : `6cbd80a` ;
- fermeture schémas, catalogues, anti-rejeu et évaluation one-shot :
  `f7079ed` ;
- identité producteur et ordre d’export signé : `288a1de`.

## Étape suivante

Construire d’abord les catalogues source et evidence, puis les sceller et les
auditer. Ensuite seulement, implémenter les deux workers sur une fixture
synthétique multi-batch. Aucun CRM réel ne doit être déposé ou ouvert avant
deux verdicts futurs `GO_S1_REAL_CRM_OPEN`.
