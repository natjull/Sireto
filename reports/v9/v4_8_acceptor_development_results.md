# V4.8 — faisabilité de l'accepteur sur les cas difficiles courants

Date : 28 juillet 2026  
Verdict de développement : **`GO_RANDOM_OPEN_V48`**  
Winner gelé : **`HARD_W1`**

## Réponse directe

Oui, les labels difficiles courants apportent un signal utile aux 80 features
existantes. La régression logistique `HARD_W1` rejette 23 des 25 mauvais
top-1 ciblés en prédiction hors pli, contre 13/25 pour `BASE_REFIT`. Elle
évite donc dix erreurs supplémentaires.

Ce gain coûte trois bons cas automatiques : 58/68 contre 61/68, soit une
baisse de 4,412 points, sous la limite préenregistrée de 5 points.

Sur le dev historique effectif, `HARD_W1` automatise 1 186/1 452 scènes,
dont 1 184 correctes et deux erreurs : précision observée 99,831 % et
couverture 81,680 %. Le baseline gelé automatise 1 184 scènes, dont 1 182
correctes : précision 99,831 % et couverture 81,543 %. Le nouveau modèle ne
dégrade donc ni la précision observée ni la couverture sur ce dev.

Ces mesures choisissent et évaluent le seuil sur le même dev historique.
Elles prouvent une faisabilité interne, pas une précision de production.

## Reproduction du baseline

Sur les 1 456 scènes dev originales et au seuil historique
`0.46313316267954524`, le bundle V4.1 reproduit exactement :

| Mesure | Résultat |
|---|---:|
| AUTO | 1 188 |
| corrects AUTO | 1 186 |
| erreurs AUTO | 2 |
| précision observée | 99,832 % |
| couverture | 81,593 % |

Deux réentraînements `BASE_REFIT` ont produit des scores et coefficients
identiques à `1e-12` près.

## Comparaison sur le dev historique effectif

| Variante | AUTO | Corrects | Erreurs | Précision | Couverture |
|---|---:|---:|---:|---:|---:|
| `BASE_FROZEN` | 1 184 | 1 182 | 2 | 99,831 % | 81,543 % |
| `BASE_REFIT` | 1 184 | 1 182 | 2 | 99,831 % | 81,543 % |
| `HARD_W1` | 1 186 | 1 184 | 2 | 99,831 % | 81,680 % |
| `HARD_W2` | 1 186 | 1 184 | 2 | 99,831 % | 81,680 % |
| `HARD_W4` | 1 185 | 1 183 | 2 | 99,831 % | 81,612 % |

Les trois variantes `HARD` satisfont le gate historique. Le seuil complet
gelé de `HARD_W1` vaut `0.3617231974526733`.

## Comparaison hors pli sur 94 cas difficiles

| Variante | Mauvais rejetés | Mauvais AUTO | Bons AUTO | Taux bons AUTO | Ambigus AUTO |
|---|---:|---:|---:|---:|---:|
| `BASE_REFIT` | 13/25 | 12 | 61/68 | 89,706 % | 0/1 |
| `HARD_W1` | **23/25** | **2** | 58/68 | 85,294 % | 0/1 |
| `HARD_W2` | 22/25 | 3 | 58/68 | 85,294 % | 0/1 |
| `HARD_W4` | 20/25 | 5 | 58/68 | 85,294 % | 0/1 |

`HARD_W1` gagne le tie-break principal avec dix erreurs supplémentaires
rejetées par rapport à `BASE_REFIT`, contre neuf pour W2 et sept pour W4.

Par origine de label, `HARD_W1` rejette :

- 14/16 mauvais cas transportés sans dérive depuis V4.4 ;
- 9/9 mauvais top-1 nouvellement adjudiqués par V4.7.

Le seul mauvais cas de la strate `AUTO_HIGH_SCORE` reste AUTO. Les 23
rejets viennent de la strate proche du seuil. Ce point devra être surveillé
dans le shadow frais.

## Discipline expérimentale

- Les 94 cas ciblés ont chacun été scorés par un modèle qui n'avait vu ni
  leur label ni leur composante.
- Chaque modèle de fold a choisi son propre seuil sur le dev historique.
- Les quatre `hard_dev_locked` sont seulement descriptifs et n'ont participé
  à aucun gate.
- Aucun identifiant random n'apparaît dans les prédictions dev, OOF ou
  descriptives.
- Zéro ligne random a été lue ou scorée par le runner.
- Aucun retrieval ou ranker n'a été réentraîné.
- Le test final est resté fermé.

## Artefact et gel

Répertoire :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_acceptor_development/f2ea5be7c1a40647`

| Artefact | SHA-256 |
|---|---|
| `manifest.json` | `a232ff17fd708321a3129f9411626cdfea5c5d46f8c69a9776ace474f23888d4` |
| `development_report.json` | `e8848c7f4c8ecbdb532194519d8c90eba39011813c139a09c9266f70c568b2e2` |
| `hard_oof_predictions.parquet` | `e29e7a8bc1a86de0d6511fb31fbe19108e9e2f5f20ab0328d2423fbdca0785aa` |
| `dev_predictions.parquet` | `6abf546b9d858dfab1130c21e1f50bbff24ccc6d6525a9e0456ab72c58338f8c` |
| `winner/acceptor_model.joblib` | `2423033ef5e003112481fb58926611dbfbaf71b8562aea848545c5ab098e487c` |
| `winner/metadata.json` | `41b84f05fe846db9362b1eff5f362b075bec08aee3af1bd1c5ee553d5d56abfc` |
| `winner_freeze.json` | `5d7344b2e4b2fa256f05e75420a5c16edaf52a530f6e9486000aeaec74c8bcbc` |

## Prochaine étape

Le winner, son seuil et ses partitions doivent maintenant être épinglés dans
le contrat. Ensuite seulement, un marqueur irréversible peut ouvrir une fois
les 52 labels random fiables. Le gate exige zéro AUTO parmi leurs cinq
mauvais et leur ambigu, zéro erreur AUTO au total, au moins 20/46 bons AUTO
et au plus un bon AUTO de moins que le baseline gelé.

Commits de référence : contrat et partitions `a15dd07`, runner `3f4671b`,
correctif de lecture stricte `dab961d`.
