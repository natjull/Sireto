# V9 — Gate 2 retrieval dense local sur dev gelé

## Verdict

**FAIL — ne pas promouvoir l’hybride dense local.**

L’ajout du MiniLM multilingue générique par RRF à poids égal dégrade le
Recall@50 SIRET de **1,83 point** sur le split dev SIREN-disjoint. La
régression est statistiquement nette et dépasse aussi le seuil de deux points
sur plusieurs segments critiques.

Ce résultat ne ferme pas encore Gate 2 : l’expérience dense global SIREN reste
justifiée par les 168 misses sparse au niveau SIREN et doit être évaluée sans
ouvrir le test final.

## Contrat expérimental

- Benchmark : `c33b80855f560074`, split `dev`, 2 565 requêtes.
- Modèle : `paraphrase-multilingual-MiniLM-L12-v2`, révision `86741b4e`,
  fingerprint
  `69cc9197d5f24b529ca8a0e3ef391e6709038e8d0bd3a8ad9772a4c624748ab5`.
- Modèle générique non fine-tuné sur SIRETO, afin d’éviter la contamination
  identifiée dans le modèle local historique.
- Store dense : 871 partitions INSEE + 14 CP, 10 216 448 candidats, budget
  final strict de 50.
- Fusion : RRF pré-enregistrée, mêmes 500 candidats maximum par canal.
- Artefacts immuables :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/dev_local_minilm867_c33b80855f560074_fa19430`
  et
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_local_minilm867_c33b80855f560074_fa19430`.

## Résultats

| Variante | Recall@50 SIRET | Recall@50 SIREN | Latence p95 | Budget |
|---|---:|---:|---:|---:|
| Sparse | 2 317/2 565 = **90,33 %** | 2 397/2 565 = 93,45 % | 1 445 ms | 0 violation |
| Sparse + dense local | 2 270/2 565 = **88,50 %** | 2 343/2 565 = 91,35 % | 1 709 ms | 0 violation |
| Dense seul, diagnostic | 1 803/2 565 = **70,29 %** | 1 880/2 565 = 73,29 % | 1 344 ms | 0 violation |

Comparaison appariée sparse → hybride :

- delta : **−1,832 point** ;
- IC95 bootstrap apparié : **[−2,729 ; −0,936] points** ;
- test exact de McNemar bilatéral : **p = 0,0000731** ;
- 45 misses sparse récupérés ;
- 92 hits sparse déplacés ;
- 2 225 hits communs et 203 misses communs ;
- ratio de latence p95 : **1,183×**.

## Segments critiques

| Segment | n | Delta hybride − sparse | Récupérés | Déplacés |
|---|---:|---:|---:|---:|
| Actifs | 2 078 | **−2,26 pts** | 26 | 73 |
| Fermés | 487 | 0,00 pt | 19 | 19 |
| CP seul | 39 | **−2,56 pts** | 0 | 1 |
| INSEE | 2 526 | −1,82 pt | 45 | 91 |
| Mégapoles | 165 | **−3,03 pts** | 4 | 9 |
| Multi-sites | 614 | **−2,28 pts** | 8 | 22 |

Le gate échoue sur deux critères :

- le Recall@50 n’est pas strictement supérieur ;
- plusieurs familles critiques régressent de plus de deux points.

Le budget fixe et la latence inférieure à 2× passent.

## Diagnostic architectural

Le dense local générique apporte bien une information différente, mais elle
est trop bruitée pour une fusion RRF symétrique :

- l’oracle `sparse ∪ dense-seul` atteindrait 2 356/2 565 = **91,85 %** ;
- l’oracle `sparse ∪ hybride` atteindrait 2 362/2 565 = **92,09 %** ;
- le dense seul ne récupère que 39 des 248 misses SIRET sparse tout en perdant
  553 de ses hits ;
- parmi les 248 misses SIRET sparse, 80 conservent le bon SIREN dans le top 50
  et 168 manquent déjà au niveau SIREN ;
- le dense local récupère 25 de ces 168 misses SIREN.

Il existe donc un faible signal complémentaire, mais pas une amélioration
exploitable par la fusion locale prévue. Aucun tuning opportuniste du poids RRF
n’est autorisé après lecture de ces résultats.

## Décision suivante

Poursuivre uniquement l’expérience pré-enregistrée **dense global SIREN →
expansion SIRET** sur dev. Elle cible directement les 168 misses au niveau
SIREN et peut retrouver des entités hors de la partition locale. Si elle
échoue également, Gate 2 sera fermée et aucun ranker/accepteur V9 ne sera
entraîné sur ce retrieval.

