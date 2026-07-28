# V4.10b — développement de l'accepteur structuré

Date : 28 juillet 2026

Verdict : **`PIVOT_STRUCTURED_FEATURES`**

## Résultat

Aucune des six variantes structurées ne franchit le gate préenregistré.
Aucun bundle n'est donc produit et aucune population fraîche n'est ouverte.

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_10b_structured_acceptor_development/71e067f75536180b`

### Dev historique effectif

| Variante | AUTO corrects / AUTO | Précision observée | Couverture |
|---|---:|---:|---:|
| `BASE_FROZEN` | 1 182 / 1 184 | 99,831 % | 81,543 % |
| `CURRENT80_W1` | 1 184 / 1 186 | 99,831 % | 81,680 % |
| `CURRENT80_W2` | 1 184 / 1 186 | 99,831 % | 81,680 % |
| `CURRENT80_W4` | 1 183 / 1 185 | 99,831 % | 81,612 % |
| `STRUCTURED_LOGIT_W1` | 1 188 / 1 190 | 99,832 % | 81,956 % |
| `STRUCTURED_LOGIT_W2` | 1 187 / 1 189 | 99,832 % | 81,887 % |
| `STRUCTURED_LOGIT_W4` | 1 187 / 1 189 | 99,832 % | 81,887 % |
| `STRUCTURED_XGB_W1` | 1 180 / 1 182 | 99,831 % | 81,405 % |
| `STRUCTURED_XGB_W2` | 1 178 / 1 180 | 99,831 % | 81,267 % |
| `STRUCTURED_XGB_W4` | 1 174 / 1 176 | 99,830 % | 80,992 % |

Le baseline original est aussi reproduit exactement sur les 1 456 lignes :
1 188 AUTO, 1 186 correctes et deux erreurs.

### Diagnostic difficile group-OOF

| Variante | Mauvais refusés | Bons conservés AUTO | Ambigus AUTO | Gate |
|---|---:|---:|---:|---|
| `CURRENT80_W1` | 23 / 25 | 58 / 68 | 0 / 1 | échec : minimum 24 mauvais |
| `CURRENT80_W2` | 22 / 25 | 58 / 68 | 0 / 1 | échec |
| `CURRENT80_W4` | 20 / 25 | 58 / 68 | 0 / 1 | échec |
| `STRUCTURED_LOGIT_W1` | 18 / 25 | 59 / 68 | 1 / 1 | échec |
| `STRUCTURED_LOGIT_W2` | 17 / 25 | 59 / 68 | 1 / 1 | échec |
| `STRUCTURED_LOGIT_W4` | 16 / 25 | 59 / 68 | 1 / 1 | échec |
| `STRUCTURED_XGB_W1` | 21 / 25 | 55 / 68 | 0 / 1 | échec |
| `STRUCTURED_XGB_W2` | 20 / 25 | 58 / 68 | 0 / 1 | échec |
| `STRUCTURED_XGB_W4` | 20 / 25 | 58 / 68 | 0 / 1 | échec |

Le contrôle `CURRENT80_W1` reste le plus proche du gate mais reproduit son
échec connu à 23/25. Les features structurées ne l'améliorent pas :

- les logits conservent un bon de plus, mais laissent passer davantage de
  mauvais cas et l'ambigu ;
- les XGBoost refusent au mieux 21 mauvais, et W1 perd trois bons sous le
  minimum ; les trois sont aussi sous la précision historique du baseline.

## Intégrité et reproductibilité

- 54 fits logiques, chacun répété : 108 fits physiques ;
- écart maximal de score entre répétitions : zéro ;
- 54 seuils recalculés exactement sur les 1 452 lignes dev historiques ;
- 14 520 prédictions dev, 940 prédictions hard et 77 842 points de courbe ;
- toutes les décisions hard entraînées sont group-OOF et utilisent le seuil
  du pli correspondant ;
- 10 variantes présentes dans le registre, baseline compris ;
- zéro ligne random V4.8, fresh dev, locked ou test final lue ou scorée ;
- quatre lignes historiques exclues lues uniquement pour reproduire le
  baseline original, zéro usage par les modèles entraînés, seuils ou gates ;
- hashes d'entrées, de sorties, du lock, du runner et du scaler recomputés
  sans divergence ;
- absence de `bundles/` conforme à `fresh_dev_eligible_variants=[]`.

Deux audits indépendants ont recomputé les métriques, seuils, gates, hashes,
comptes de lignes et le build ID.

## Limite découverte après le gel

La politique V4.10b a bien exclu les 58 alias préenregistrés, mais elle a
omis trois autres copies définitionnelles déjà signalées lors de l'audit
initial :

- `candidate_top1_ranker_score` / `scene_score_top1` ;
- `candidate_top2_ranker_score` / `scene_score_top2` ;
- `candidate_delta_ranker_score` / `scene_score_gap`.

Les diagnostics le rendent visible : les deux représentations du gap et du
score top-2 reçoivent des coefficients égaux et dominent les modèles
structurés. Cela n'invalide pas le résultat par rapport au plan gelé de 641
features, mais empêche de conclure qu'une représentation réellement
canonique a été testée.

Ces trois colonnes ne doivent pas être retirées puis réévaluées comme si les
94 cas constituaient encore une validation. Toute nouvelle architecture doit
être gelée puis jugée sur une population indépendante.

## Décision

V4.10b est close par `PIVOT_STRUCTURED_FEATURES`.

- aucun modèle V4.10b n'est promu ;
- aucun seuil opérationnel ne change ;
- le retrieval V4.2-B et le ranker A restent gelés ;
- le random V4.8 et le test final restent fermés ;
- le prochain contrat doit traiter l'alignement homogène
  retrieval–ranker–accepteur et prévoir une preuve sur une nouvelle cohorte,
  plutôt qu'une nouvelle règle ajustée aux 94 cas consommés.

Commits :

- plan : `5ed1ba3` ;
- runner/scaler/tests : `6ae4cf7` ;
- verrou d'exécution : `fb33c76`.
