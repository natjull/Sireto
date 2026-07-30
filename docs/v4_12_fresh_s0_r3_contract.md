# V4.12 — Contrat du successeur synthétique S0-R3

## 1. Portée

S0-R3 est l'unique successeur autorisé de l'exécution consommée S0-R2. Il
sert exclusivement à démontrer, sur la fixture synthétique fermée de six
lignes, que le scanner/sealer peut atteindre son état terminal sous la
frontière macOS réelle.

S0-R3 :

- ne lit aucun CRM réel ;
- ne qualifie aucune ligne métier ;
- n'exécute aucun retrieval ni modèle ;
- ne déplace, ne répare et ne réutilise aucun artefact R1 ou R2 ;
- n'autorise pas le test final ;
- n'autorise aucune dépense ou ressource autre que le Mac et
  `/Volumes/CATNAT_DATA`.

Un succès R3 permet seulement de passer à la construction de l'entrée CRM
fraîche physiquement aveugle définie par les contrats V4.12. Ranker, decider,
risk model et accepteur restent gelés.

### Héritage fermé du plan R2

Le plan R3 est un overlay fermé du plan R2 canonique SHA-256
`2ab9a1d5954588c01de22c54e21c721aa0e9da9a9e7f140d9f93950cb8b1abf4`.
L'algorithme effectif est :

1. charger et valider byte-for-byte ce plan R2 ;
2. en faire une copie profonde ;
3. supprimer exactement les chemins déclarés dans `inheritance.removals`,
   tous obligatoirement présents dans la base ;
4. appliquer exactement les couples `source` → `target` déclarés dans
   `inheritance.overrides` du plan R3 ;
5. refuser toute clé d'overlay non déclarée, toute source absente, tout chemin
   parent cible absent, toute collision de type sur une cible existante et
   tout sentinel ; le dernier composant cible ne peut être créé que s'il est
   explicitement déclaré ;
6. hériter littéralement de toute valeur non supprimée ou override.

Cet héritage conserve notamment, sans réécriture permissive :

- authorization, claim, lease et règle claim-sans-receipt ;
- protocole FD et contrôle ;
- matrice complète des canaris ;
- fermeture du runtime Python/PyArrow/Mach-O ;
- profil sandbox et limites macOS ;
- observations parent avant/après ;
- autorités d'arbres, seals et journal ;
- politique de recovery et d'absence de rerun ;
- ensembles exacts du receipt, ses types, nulls et écriture durable.

Les seuls usages autorisés de `r2_successor` après overlay sont les pins
matériels immuables de `runtime_boundary_amendment` et les preuves
historiques. Aucune identité, racine, fixture, autorisation ou séquence R2 ne
peut piloter R3.

## 2. Prédécesseur immuable

Le seul prédécesseur reconnu est :

- racine :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r2` ;
- run :
  `bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo` ;
- attempt :
  `dhlmigejpmdehbjbppcfmlnkbcehcmgagojmnmibhcdnliicljifmjegieiogmmm` ;
- receipt :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r2/audit/bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo/parent/launch_receipts/dhlmigejpmdehbjbppcfmlnkbcehcmgagojmnmibhcdnliicljifmjegieiogmmm.json` ;
- SHA-256 :
  `6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e` ;
- verdict : `STOP` ;
- reason : `WORKER_CONTROLLED_STOP` ;
- terminal : `STOP_SYNTHETIC_SCANNER_SEALER_V412`.

Le builder R3 doit relire ce receipt par FD ancré, vérifier ses octets
canoniques et toutes ces constantes. Le receipt R2 lie déjà R1
transitivement ; R3 ne relit pas R1.

## 3. Nouvelle autorité R3

La racine R3 est exactement :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r3
```

Elle doit être absente au moment de la préinscription et ne peut être créée
qu'après `GO_R3_IMPLEMENTATION`.

Le run R3 est dérivé par :

```text
domain = "SIRETO-V412-FRESH-SYNTHETIC-S0-R3-RUN-ID\0"
projection = [
  fixture_spec_sha256,
  core_plan_sha256,
  predecessor_receipt_sha256
]
```

Valeurs :

```text
fixture_spec_sha256 =
  6f917f98b7a8b42e34af390b21e63ef4cb33051aa5a0bf7154f5351c4337d33e
core_plan_sha256 =
  e8a55a999035183363c0bf7711280b09553a305434173286e41c696ea3e4772f
predecessor_receipt_sha256 =
  6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e
```

Résultat exact :

```text
kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl
```

La fixture R3 régénère uniquement les manifestes portant le nouveau run. Le
CSV et le Parquet d'évidence restent byte-identiques à la fixture cœur.

Le manifeste de contrôle canonique R3 doit avoir le SHA-256 :

```text
bae57c4f207f2637574a2872169a311a9199f3fc6ba89c2694ddd123b245ac18
```

L'attempt utilise le domaine cœur
`SIRETO-V412-FRESH-SYNTHETIC-ATTEMPT-ID\0` et la projection :

```text
[
  synthetic_run_id,
  fixture_control_manifest_sha256,
  logical_time_utc
]
```

Valeurs :

```text
synthetic_run_id =
  kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl
fixture_control_manifest_sha256 =
  bae57c4f207f2637574a2872169a311a9199f3fc6ba89c2694ddd123b245ac18
logical_time_utc = 2026-07-30T00:00:00Z
```

Résultat exact :

```text
afjgbfncbfdbcakcjiclhmlnmgmemcjmllkhdfgogjjncompjojcnbkelopdklgp
```

## 4. Autorité `execution_identity`

Il n'existe qu'une seule autorité d'identité. Elle a exactement les clés :

```text
schema_version
algorithm
run
attempt
```

`run` et `attempt` ont chacun exactement :

```text
domain
projection
values
result
```

Le plan contient l'objet littéral complet. Le lock le recopie
byte-sémantiquement sans le recalculer. Le launcher valide que le lock est
égal au plan, puis le recopie dans `worker-spec-2`. Le worker recalcule les
deux digests et vérifie :

- fixture et plan cœur ;
- hash du receipt R2 ;
- hash exact du manifeste de contrôle reçu par FD ;
- temps logique ;
- égalité run/attempt entre plan, lock, spec et contrôle.

Le worker n'utilise jamais `core_plan["ids"]["run"]` pour une identité
successeur. Le domaine R1 est rejeté avant toute autorité de sortie.

## 5. Schémas fermés

Le lock R3 utilise :

```text
sireto-v4.12-fresh-s0-r3-authoritative-execution-lock-1
```

Il ajoute obligatoirement `execution_identity` à l'ensemble exact des champs.

Le spec worker utilise :

```text
sireto-v4.12-fresh-s0-worker-spec-2
```

Il ajoute obligatoirement `execution_identity`, littéralement égal au lock.

Le résultat de contrôle utilise :

```text
sireto-v4.12-fresh-s0-control-result-2
```

Il ajoute `worker_failure` :

- `null` sur `RESULT` ;
- objet obligatoire sur `STOP` ;
- exactement `schema_version`, `worker_phase`, `worker_reason_code` ;
- aucun message libre, chemin, traceback, valeur CRM ou représentation
  d'exception.

Le schéma de l'objet est :

```text
sireto-v4.12-fresh-s0-worker-failure-1
```

Matrice initiale autorisée :

| `worker_phase` | `worker_reason_code` |
|---|---|
| `IDENTITY` | `EXECUTION_IDENTITY_SCHEMA_INVALID` |
| `IDENTITY` | `RUN_DERIVATION_MISMATCH` |
| `IDENTITY` | `ATTEMPT_DERIVATION_MISMATCH` |
| `IDENTITY` | `SPEC_CONTROL_IDENTITY_MISMATCH` |
| `WORKER_RUNTIME` | `INTERNAL_ERROR` |

Toute autre paire transforme le terminal en
`WORKER_CONTROL_INVALID`. Le verdict externe reste
`STOP / WORKER_CONTROLLED_STOP`.

Les définitions remplacées `control_result-2`, `worker-spec-2`,
`execution_identity`, `worker_failure` et `runtime_smoke_attestation` sont
matérialisées intégralement dans le plan R3 : champs exacts, nullabilité et
types. Les annotations `inherited_from_r2` sont documentaires et ne remplacent
jamais une contrainte exécutable.

Les rôles de sources globales et les blobs Git sont deux ensembles distincts.
`SANDBOX_EXEC` appartient aux sources globales et reste épinglé comme binaire
système dans le runtime hérité ; il est interdit dans
`execution_lock.implementation_blob_roles`. Chaque rôle de ce dernier ensemble
doit correspondre à un blob du commit d'implémentation.

Le receipt parent R3 passe au schéma :

```text
sireto-v4.12-fresh-s0-authoritative-launch-receipt-3
```

Les validateurs R1/R2 historiques restent attachés à leurs schémas et octets
immuables ; aucun validateur permissif multi-version n'est introduit dans le
run R3.

## 6. Gate jetable déjà acquis

Le gate préalable autorisant cette préinscription est :

- racine :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/diag-r3-successor-gate.yww2qf5m` ;
- résultat :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/diag-r3-successor-gate.yww2qf5m/audit/pmkokoegjlcielgbjknnddllkpffomhbjkogilihcdmcilikdccpppcfnkmohamf/parent/gate_results/successor_gate_result.json` ;
- résultat SHA-256 :
  `c86ad8bf1a4b8af0525c6870e05ddabb2f27c4208f9f07c8be07edebb52e212b` ;
- verdict : `GO_PREREG_R3`.

Il prouve :

- `INGESTED` sous le vrai Python privé et Seatbelt ;
- 11/11 canaris refusés avec `EPERM` ;
- stabilité réelle de `60.003147917` secondes ;
- stdout/stderr vides ;
- trois générations de journal ;
- rejet d'une identité R1 complète sans mutation de sortie.

Il ne prouve pas encore le chemin complet du launcher R3.

## 7. Gate d'implémentation obligatoire

Avant `GO_BUILD_R3`, les tests doivent couvrir :

1. plan R3 → lock R3 → `_load_lock` → `_worker_spec` → vrai `main` ;
2. égalité littérale de `execution_identity` à chaque frontière ;
3. mutations du schéma, du domaine, de chaque projection, valeur et résultat ;
4. mutations directes de `control.synthetic_run_id` et du temps logique ;
5. rejet R1 et rejet de toute identité non égale au plan R3 ;
6. STOP fermé traversant worker, launcher et receipt ;
7. onze canaris avec seulement `EPERM` ou `EACCES` ;
8. vrai `_process` atteignant `INGESTED` sous le profil R3 jetable ;
9. arbres, seals, journal et autorités parent revalidés ;
10. suite complète verte sur le TMPDIR SSD ;
11. receipt R2 prédécesseur : path, hash, JSON canonique, champs, run,
    attempt, verdict, reason et terminal, avec mutations hostiles de chacun.

Deux audits indépendants doivent conclure `GO_R3_IMPLEMENTATION`. À défaut :
`PIVOT` ou `STOP`.

## 8. Ordre irréversible

1. committer ce contrat et son plan canonique ;
2. obtenir deux audits `GO_R3_IMPLEMENTATION` ;
3. implémenter builder R3, sealer, launcher, worker et tests ;
4. committer l'implémentation ;
5. vérifier que la racine R3 est encore absente ;
6. construire une fois la fixture R3 ;
7. construire runtime, profil, smoke et lock R3 ;
8. auditer le lock sans worker ;
9. committer une autorisation R3 exacte ;
10. lancer une seule fois ;
11. écrire le receipt terminal immuable ;
12. conclure `INGESTED`, `QUARANTINED` ou `STOP`.

Un claim sans receipt, un receipt terminal, une collision ou toute dérive
consomme l'attempt. Aucun rerun sous le même run ou attempt n'est autorisé.

## 9. Sortie du jalon S0

Seul un receipt R3 terminal valide `INGESTED`, avec toutes les autorités
attendues, peut ouvrir le chantier CRM frais. La fixture préenregistrée est
valide et attend `INGESTED` ; `QUARANTINED` conclut donc `PIVOT`, jamais
succès. `STOP` conclut `STOP`.

Même après succès :

- aucun label n'est lu pendant la qualification ;
- aucune qualification n'utilise un hit, rang ou score retrieval ;
- le gate retrieval reste couverture identifiable ≥ 80 % et Recall@100 SIRET
  exact ≥ 99 % ;
- les modèles restent gelés jusqu'à ce gate.
