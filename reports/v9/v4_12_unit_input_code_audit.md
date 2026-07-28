# V4.12 — Audit indépendant du builder d'entrées sûres

## Périmètre

- contrat : `docs/v4_12_unit_input_contract.md` ;
- plan : `config/v4_12_unit_input_plan.json` ;
- code audité : commit `18eb76e` ;
- fichiers : `scripts/build_v412_unit_inputs.py` et
  `tests/test_build_v412_unit_inputs.py`.

Aucun build réel et aucune lecture d'oracle n'ont été exécutés pendant cet
audit.

## Premier verdict : STOP_CODE

La première revue a refusé l'implémentation malgré 18 tests réussis. Elle a
notamment détecté :

- la comparaison des sidecars TF-IDF avec le mauvais hash ;
- un contournement possible du plan canonique par `--internal-run` ;
- l'absence de revalidation des quatre Parquet runtime ;
- une signature historique publiée sans être recalculée ;
- un verrou attestant seulement l'audit du contrat ;
- une fenêtre TOCTOU permettant un remplacement puis une restauration des
  sources CRM ;
- l'absence de liaison entre le blob Git annoncé et le fichier exécuté.

Ces défauts ont été reproduits par de nouveaux tests puis corrigés avant le
commit audité.

## Contre-audits

Deux relectures indépendantes en lecture seule ont ensuite rendu :

- `GO_CODE` ;
- `GO_CODE_2`.

Preuves :

- `python3 -B -m pytest -q tests/test_build_v412_unit_inputs.py` :
  **27 tests réussis** ;
- suite complète : **645 tests réussis** ;
- inventaire partitions réel :
  `680f1884879bfa5b8cf2c335a0658604010e3d4c546ed6eaeb2e2ef34c954463`,
  4 119 fichiers, 27 594 915 lignes et 1 969 745 065 octets ;
- inventaire cache réel :
  `589360b10fa65d190bae9a2521e05d8e60e71c2cbd1fc0c5843c044332a183ce`,
  1 454 clés et 6 730 554 690 octets ;
- signature historique partitions recalculée :
  `2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82`.

Les contre-audits ont confirmé la projection physique limitée aux six/deux
colonnes, le plan canonique, la liaison commit/worktree/verrou, les
inventaires ancrés, la reprojection avant chaque promotion, la validation
des quatre Parquet et du ledger, ainsi que la publication audit-first.

## Verdict

`GO_CODE_V412_UNIT_INPUTS`

Ce verdict autorise la création et le contre-audit du verrou d'exécution. Il
ne vaut ni `GO_V412_UNIT_INPUTS`, ni autorisation du store, du worker ou du
benchmark.

## Correctif de séquencement du verrou

Avant la création du verrou, une dernière revue a constaté qu'exiger
`HEAD == git_commit` rendait impossible le commit ultérieur du verrou. Le
correctif `c97c737` autorise un commit audité antérieur tout en maintenant,
pour chaque source, l'égalité stricte entre le fichier de travail, son hash
dans le verrou et le blob binaire du commit audité.

Le contre-audit rend `GO_LOCK_SEQUENCING`. Les tests ciblés passent désormais
à **29/29** et la suite complète à **647/647**. Une source modifiée,
non suivie ou absente du commit audité reste refusée.
