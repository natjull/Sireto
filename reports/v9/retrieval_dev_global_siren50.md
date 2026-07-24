# V9 — Gate 2 retrieval dense global SIREN sur dev gelé

## Verdict

**FAIL — ne pas promouvoir l’hybride dense global SIREN.**

À budget final 50 identique, l’ajout du canal dense global SIREN suivi de
l’expansion SIRET dégrade le Recall@50 SIRET de **2,61 points**. La régression
est statistiquement nette et touche notamment les établissements actifs et les
cas multi-sites.

Gate 2 est donc fermée : les deux variantes denses pré-enregistrées ont échoué.
Conformément au contrat, Gate 3 (ranker/accepteur), Gate 4 (500 adjudications)
et le cross-encoder ne sont pas ouverts.

## Contrat expérimental

- Benchmark : `c33b80855f560074`, split `dev`, 2 565 requêtes.
- Baseline : sparse local SIRET, 500 candidats maximum par sous-canal et
  sortie limitée à 50.
- Modèle dense : `paraphrase-multilingual-MiniLM-L12-v2`, révision
  `86741b4e`, fingerprint
  `69cc9197d5f24b529ca8a0e3ef391e6709038e8d0bd3a8ad9772a4c624748ab5`.
- Index global : 28 982 797 SIREN, FAISS IVFPQ 4096/48, recherche top-50.
- Expansion : store DuckDB v2 de 14 378 332 SIRET, top-40 géographique par
  SIREN en SQL puis cap métier de 20.
- Fusion : RRF pré-enregistrée à poids égal, sortie au cutoff 50.
- Exécution retrieval : commit `5123516`, manifeste SHA-256
  `e227ca956b958ecb76df7032927601346be77b9fe77c875862a397e61fd038b6`.
- Audit de budget corrigé : commit `bc49918`.
- Artefacts immuables :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/dev_global_siren_v2_c33b80855f560074_5123516`
  et
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_global_siren_v2_c33b80855f560074_bc49918`.

Le contrôle intrinsèque de l’ANN, effectué avant l’évaluation métier sur 200
SIREN répartis dans le snapshot, mesurait un self-recall de 54,5 % à 1 et
86,0 % à 50. L’index était donc exécutable et rapide, mais pas assimilable à
une recherche exacte.

## Résultats

| Variante | Recall@50 SIRET | Recall@50 SIREN | Latence p95 | Budget |
|---|---:|---:|---:|---:|
| Sparse | 2 317/2 565 = **90,33 %** | 2 397/2 565 = 93,45 % | 1 452 ms | 0 violation |
| Sparse + dense global SIREN | 2 250/2 565 = **87,72 %** | 2 325/2 565 = 90,64 % | 1 566 ms | 0 violation |

Comparaison appariée sparse → hybride global :

- delta : **−2,612 points** ;
- IC95 bootstrap apparié, 100 000 tirages :
  **[−3,509 ; −1,715] points** ;
- test exact de McNemar bilatéral : **p = 1,47 × 10⁻⁸** ;
- 37 misses sparse récupérés ;
- 104 hits sparse déplacés ;
- 2 213 hits communs et 211 misses communs ;
- ratio de latence p95 : **1,079×**.

## Hit@1 du retrieval

L’ordre brut RRF s’améliore fortement malgré la baisse du Recall@50 :

| Niveau | Sparse | Sparse + dense global | Delta | IC95 apparié |
|---|---:|---:|---:|---:|
| SIRET | 929/2 565 = 36,22 % | 1 219/2 565 = 47,52 % | **+11,31 pts** | [+9,67 ; +12,98] |
| SIREN | 1 075/2 565 = 41,91 % | 1 383/2 565 = 53,92 % | **+12,01 pts** | [+10,29 ; +13,72] |

Le gain Hit@1 est statistiquement net (McNemar p < 1,5 × 10⁻⁴⁰). Il ne valide
pas le pool hybride, puisqu’un ranker ne peut pas récupérer les 67 vérités
terrain supplémentaires sorties du top-50. Il établit toutefois que le signal
dense mérite une nouvelle ablation comme score de classement appliqué à un
pool sparse inchangé.

Deux lignes avaient initialement été signalées à tort comme violations de
budget : la partition locale contenait 41 ou 18 candidats et le canal global
avait complété la sortie jusqu’à 50. Les deux sorties respectaient bien le
cutoff. Le contrôleur accepte désormais un canal additionnel qui complète un
pool local court, tout en refusant une sortie supérieure à 50 ou inférieure au
minimum local disponible.

## Segments critiques

| Segment | n | Delta hybride − sparse | Récupérés | Déplacés |
|---|---:|---:|---:|---:|
| Actifs | 2 078 | **−3,27 pts** | 15 | 83 |
| Fermés | 487 | +0,21 pt | 22 | 21 |
| CP seul | 39 | **−5,13 pts** | 0 | 2 |
| INSEE | 2 526 | **−2,57 pts** | 37 | 102 |
| Mégapoles | 165 | 0,00 pt | 10 | 10 |
| Multi-sites | 614 | **−2,28 pts** | 6 | 20 |

Le léger gain sur les fermés n’est pas significatif : IC95 du delta
[−2,46 ; +2,87] points et McNemar p = 1. Les régressions sur les actifs,
l’ensemble INSEE et les multi-sites sont statistiquement nettes.

## Diagnostic architectural

Le canal global trouve une information différente, mais sa fusion symétrique
est destructrice au cutoff 50 :

- chaque récupération coûte en moyenne 2,8 hits sparse déplacés ;
- le Recall@50 baisse aussi au niveau SIREN, avant même la difficulté de choisir
  le bon établissement ;
- l’ANN global approximatif et un encodeur générique de nom ne produisent pas
  un signal assez précis pour concurrencer le sparse local ;
- l’expansion géographique bornée a supprimé le goulet d’exécution historique,
  sans modifier ce constat qualitatif.

Le gate ne permet pas de transformer après coup ces résultats en justification
d’un tuning de poids RRF, d’un ranker ou d’un accepteur sur le pool rejeté.
Le Hit@1 constitue en revanche un signal précis pour un **PIVOT** : garder le
pool sparse intact et tester séparément le dense comme score de classement.
Cette ablation constituerait une nouvelle hypothèse, avec un nouveau protocole
pré-enregistré.
