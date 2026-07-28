# V4.12 — Résultats du moteur unitaire et de la parité

## Verdict

**`GO_V412_UNIT_RETRIEVAL_PARITY`**

Le moteur sparse unitaire a terminé sur les 1 456 requêtes dev et reproduit
exactement les deux payloads préenregistrés. Deux audits post-run indépendants
concluent `GO_ARTIFACTS_1` et `GO_ARTIFACTS_2`.

Ce verdict certifie l'exécution label-blind, le plafond de 100 et la parité
avec la référence sparse. Il ne mesure pas encore le Recall@100 : l'oracle
est resté fermé pendant le worker et ses audits.

## Volumes et intégrité

| Mesure | Résultat |
|---|---:|
| Requêtes | 1 456 |
| Candidats | 145 236 |
| Taille minimale d'un pool | 46 |
| Taille maximale d'un pool | 100 |
| Pools sous 100 | 13 |
| Pools vides | 0 |
| SIRET lookup manquants | 0 |

Les rangs sont contigus, chaque SIRET comporte 14 chiffres, les SIRET sont
uniques dans chaque requête et aucune requête ne dépasse 100 candidats.

Payload candidats :

- 3 629 947 octets ;
- SHA-256
  `1689a2f363cac7385dcfec10606c32e89d3f904e3990cda00989c13feb87ab00`.

Payload statuts :

- 16 110 octets ;
- SHA-256
  `65e662c0df3df6bd022da55cde1e2d6b13254e16314c71dad5cc063dde6d5518`.

Ces deux valeurs sont exactement celles du contrat préenregistré.

## Exécution

| Mesure | Résultat |
|---|---:|
| Durée worker totale | 1 030,16 s, soit 17,17 min |
| Temps retrieval cumulé | 620,34 s |
| Temps lookup cumulé | 398,38 s |
| Pic mémoire | 3 643 670 528 octets, soit environ 3,39 Gio |

Le ledger worker contient 1 980 lignes uniques. Les tailles et hashes avant
et après sont identiques partout. Les déclarations et tests sandbox
confirment : aucun oracle, label, historique, modèle ou réseau ouvert ; aucune
injection positive, reconstruction de cache ou écriture hors staging.

## Artefacts

Worker :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/runs/v4_12_unit_retrieval/
d2915fe7747b9b219e7a0dce400052847c913417bd240c7d94df6fb8bafedd1a
```

Audit worker :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_retrieval_worker/
d2915fe7747b9b219e7a0dce400052847c913417bd240c7d94df6fb8bafedd1a
```

Parité :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_retrieval_parity/
d587937b60f3b4c8f43003bc384d4e61b5933999cba7cf557debebfef02205f5
```

## Prochain geste autorisé

Préenregistrer et auditer le contrat de l'évaluateur séparé. Lui seul pourra
ensuite joindre les candidats scellés à l'oracle des 1 217 cas
`MATCH_EXACT`, publier Recall@100 avec les nombres bruts et mesurer la
latence selon une règle gelée. Le ranker, le decider, le risk model,
l'accepteur et le test final restent fermés.
