# V4.12 — Résultat autoritatif S0-R3

## Verdict

```text
INGESTED_R3_CERTIFIED
INGESTED_R3_CERTIFIED
```

Les deux audits indépendants ont certifié l'unique exécution S0-R3 en lecture
seule, sans relancer le launcher.

## Autorités

- commit d'implémentation :
  `8d8e0a327983c80b35f7a4e1630b4335fab48fea` ;
- commit d'autorisation canonique : `b64133fc2f5850107dc6d1164f038d297c5ecd57` ;
- autorisation SHA-256 :
  `f686ffd946c8729220bb0caba0e36303c2cac25b498ef0856b2f945da3b0c385` ;
- lock SHA-256 :
  `de5456874ccee990b17321203a572185517f489271978f14d6aa9d93be423289` ;
- claim SHA-256 :
  `c6a1c5b5d386d2b59c5f8311d27da71b47c56d0d141a185399a6b43354ad0ddd` ;
- receipt terminal :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic_r3/audit/kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl/parent/launch_receipts/afjgbfncbfdbcakcjiclhmlnmgmemcjmllkhdfgogjjncompjojcnbkelopdklgp.json
```

- receipt SHA-256 :
  `8061247794f403f52a692e41f19549dcf2803a6db744c74e9719cb824ad96a08`.

## Résultat

- schéma : `sireto-v4.12-fresh-s0-authoritative-launch-receipt-3` ;
- verdict : `INGESTED` ;
- terminal : `INGESTED_SYNTHETIC_SCANNER_SEALER_V412` ;
- reason : `OK` ;
- worker : exit `0`, signal `null`, stdout/stderr vides ;
- stabilité : `60.002720750` secondes, même processus et mêmes cinq FDs ;
- canaris : 11/11 refusés avec `EPERM` ;
- observations parent : 14/14 identiques avant/après ;
- sorties : arbres sealed et scan valides, zéro quarantaine terminale ;
- journal : trois générations chaînées ;
- scénario : six sources, deux sûres et quatre preuves de quarantaine
  synthétiques attendues ;
- cardinalité : un claim, un lease, un spec et un receipt pour l'unique
  attempt.

Le couple claim+receipt ferme désormais l'attempt et conduit le launcher au
chemin idempotent avant tout spawn. Aucun processus R3 ne reste actif.

## Portée

S0-R3 prouve que l'intake peut fonctionner sous la frontière macOS réelle.
Il ouvre le jalon de construction et qualification du CRM frais. Il
n'autorise toujours ni lecture des labels pendant la qualification, ni
retrieval pendant l'admission, ni test final, ni réentraînement des modèles.
