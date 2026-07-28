# V4.12 — Audit du builder d'oracle unitaire

## Verdict

`GO_CODE_ORACLE`

Code audité : commit `7eafad8`.

## Refus successifs

L'audit indépendant a refusé quatre versions pourtant couvertes par des
tests verts :

1. une vérité modifiée pouvait être entièrement rescellée sous le même
   `build_id` ;
2. un fichier sibling du paquet runtime pouvait être modifié à taille égale
   avec son mtime restauré ;
3. le ledger déclarait quatre fichiers alors que huit étaient ouverts ;
4. le ledger pouvait être réordonné puis rescellé.

Chaque défaut a été reproduit par un test avant correction.

## Preuves finales

- 23 tests ciblés réussis ;
- 670 tests du dépôt réussis ;
- identité redérivée depuis plan, verrou, blobs Git et quatre inputs ;
- vérité, ordre, comptes et hashes entièrement reconstruits ;
- six fichiers du paquet runtime sûr rehashés ;
- ledger exact de huit lignes, rôles et ordre UTF-8 vérifiés ;
- aucun inventaire ni `queries_all` transmis à la projection de vérité.

Trois PoC indépendants finaux ont confirmé :

- rescellation coordonnée complète : `STOP` ;
- mutation same-size/mtime restaurée : `STOP` ;
- permutation du ledger et rescellation : `STOP`.

Ce verdict autorise uniquement la création et le contre-audit du verrou
d'exécution. Aucun build réel n'a été lancé pendant l'audit.
