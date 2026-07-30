# V4.12 — Gate jetable du successeur S0-R3

## Verdict

`GO_PREREG_R3`

Ce verdict autorise uniquement la préinscription du contrat et du plan R3. Il
n'autorise ni la construction de la fixture R3, ni son lock, ni son
autorisation, ni son exécution.

## Artefact probant

- racine jetable :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/diag-r3-successor-gate.yww2qf5m`
- résultat persistant :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/diag-r3-successor-gate.yww2qf5m/audit/pmkokoegjlcielgbjknnddllkpffomhbjkogilihcdmcilikdccpppcfnkmohamf/parent/gate_results/successor_gate_result.json`
- SHA-256 du résultat :
  `c86ad8bf1a4b8af0525c6870e05ddabb2f27c4208f9f07c8be07edebb52e212b`
- run jetable :
  `pmkokoegjlcielgbjknnddllkpffomhbjkogilihcdmcilikdccpppcfnkmohamf`
- attempt jetable :
  `cfbbdbafbhelmgheanbegkjfpaomcidjpcdleighdlokodbmnhjempgffjjmfilb`

Les racines antérieures `diag-r3-successor-gate.zi2oynzm` et
`diag-r3-successor-gate.9fy69i1l` ne sont pas promotables. Le premier gate
n'était pas persistant ; le deuxième acceptait des `errno` trop larges et
n'exerçait pas le canari d'écriture dans l'audit parent.

## Preuves positives

Le gate probant utilise le vrai Python privé, PyArrow 23.0.1, le profil
Seatbelt `deny default` et le vrai `_process` du worker candidat.

- worker courant et copie privée :
  `26241b29020e6a2135e268181927b45d83a6903cb6d6ce7c0a06e58e274e6732`
- fermeture runtime : 1 525 records, SHA
  `6e729d467c0f0f53e6d0264e0133581c68e74198a98eef3790436ffca1427af0`
- profil effectif :
  `0c8533be2b66d36b940a62ffd5f7e059bc0eca90abcd1d94c62e1efaf956d09a`
- durée monotone : `60.003147917` secondes ;
- mêmes FD de payload avant et après ;
- stdout et stderr vides ;
- onze canaris sur onze refusés avec `errno=1` (`EPERM`) ;
- résultat métier : `INGESTED_SYNTHETIC_SCANNER_SEALER_V412`.

Autorités de sortie :

- sealed manifest :
  `7137abc46329cf697da807016e8cd63123e874e81d20a0f51d830cdc6a9bfac4`
- sealed seal :
  `7c048d273ca2fcdfd41647d7c995c5b15acb22f37b2f4ed4c59ce96d243665c1`
- scan manifest :
  `20d4b6bc507dfab60dc84bffb4e7d72c4ca44fab59ac18915835dbec893ec12c`
- scan seal :
  `1f91d9e83b863ee869a2cbe54010c43bf6f5a4d516a64a01b238ffc9b08f9d18`
- journal : trois générations ;
- génération terminale :
  `b62e7186dfc709cd638a097e051052769a0f859840882f6dd97ccf83106515dc`
- événement terminal :
  `681b8dbc44a1fa440c588b287eb591a69919d7833dae0699721779aca5ac2525`.

Les Parquet contiennent exactement deux lignes sûres, quatre preuves de
quarantaine et six lignes de table d'identité, conformément à la fixture
préenregistrée.

## Preuve négative

Le même runtime et le même profil reçoivent ensuite une identité R1 complète
et cohérente :

- domaine et projection R1 ;
- run R1 exact `komapn...` ;
- aucune simple corruption du seul champ `result`.

Le worker répond :

```text
IDENTITY / EXECUTION_IDENTITY_SCHEMA_INVALID
```

L'arbre de sortie est identique avant et après ce rejet. Aucun texte
d'exception, chemin libre ou contenu CRM ne traverse le canal de diagnostic.

## Évolutions validées par le gate

- le worker ne recalcule plus le run avec `plan["ids"]["run"]` ;
- une autorité fermée `execution_identity` porte les dérivations run et
  attempt ;
- le worker utilise `worker-spec-2` ;
- le terminal utilise `control-result-2` et un `worker_failure` fermé ;
- le launcher valide la matrice phase/code ;
- le vrai `main` propage l'identité du spec vers `_process` ;
- les mutations de schéma, run, attempt, hash contrôle et cohérence spec sont
  rejetées avant toute écriture.

La suite complète compte `1095 passed`.

## Conditions restantes avant toute construction R3

Le plan R2 reste incompatible avec ces schémas. Le plan et le contrat R3
doivent encore :

1. pinner le receipt R2 terminal, SHA
   `6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e` ;
2. imposer une racine, un run et un attempt R3 distincts ;
3. ajouter `execution_identity` au lock et au spec ;
4. déclarer `worker-spec-2`, `control-result-2` et la matrice
   `worker_failure` ;
5. pinner ce résultat de gate, le worker, le gate et les tests ;
6. couvrir réellement le chemin `_load_lock → _worker_spec → main`.

Jusqu'à ce nouveau gate de code, `GO_PREREG_R3` n'est pas un
`GO_BUILD_R3`.
