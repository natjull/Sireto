# BGE groupwise V4.12 — résultat fold 0

## Périmètre

- modèle : `BAAI/bge-reranker-v2-m3`, révision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` ;
- apprentissage : folds 2/3/4, 8 192 scènes et 131 072 paires ;
- évaluation : fold 0 entièrement exclu, 2 797 requêtes et 279 511
  candidats, sans injection du positif ;
- artefact immuable :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_groupwise/01e1049c16af2600`.

## Mesures

| Variante / segment | Correct | Total | Hit@1 |
|---|---:|---:|---:|
| BUSINESS_LEARNED, exact | 2 437 | 2 797 | 87,129 % |
| BGE zero-shot, exact | 2 171 | 2 797 | 77,619 % |
| CamemBERT fine-tuné, exact | 2 353 | 2 797 | 84,126 % |
| **BGE groupwise fine-tuné, exact** | **2 400** | **2 797** | **85,806 %** |
| BGE groupwise, difficiles | 32 | 38 | 84,211 % |
| BGE groupwise, actifs | 2 159 | 2 391 | 90,297 % |
| BGE groupwise, fermés | 241 | 406 | 59,360 % |

BGE fine-tuné gagne 229 bonnes réponses sur sa version zero-shot et 47 sur
CamemBERT fine-tuné, mais reste 37 réponses derrière BUSINESS_LEARNED. Il
n'est donc pas retenu comme remplacement autonome.

La complémentarité justifie néanmoins le stack préenregistré :

- corrects par les deux modèles : 2 321 ;
- corrects par BUSINESS_LEARNED seulement : 116 ;
- corrects par BGE seulement : 79 ;
- faux pour les deux : 281 ;
- union oracle BUSINESS_LEARNED + BGE : 2 516/2 797.

Ces 79 corrections BGE sont un potentiel, pas une sélection oracle autorisée.
Le verdict dépend du méta-ranker XGBoost entraîné uniquement avec des scores
BGE cross-fittés.

## Ressources et intégrité

- temps d'apprentissage : 10 492,56 s ;
- temps de scoring : 12 819,86 s ;
- RSS maximal publié : 3 150 594 048 octets ;
- paramètres totaux : 567 755 777 ; paramètres entraînables : 51 435 521 ;
- huit sorties du manifeste revérifiées, zéro mismatch ;
- SHA-256 `evaluation.json` :
  `a54ca871d5950700a8c4d2a54bd85ace522bd3566ffa032d7602f3f8fa4710c8` ;
- SHA-256 `manifest.json` :
  `b23c57ead205fd97eeea03b7495cc85887158d0f1208a0c2e9dcbe9439526da4` ;
- SHA-256 `target_scores.parquet` :
  `b3b99d61a46855a877ff98897116e7a93cb2a579b6f43d48f3fd317189a42430` ;
- SHA-256 `model/model.safetensors` :
  `c73a308c048d32ba821688b5d60ad98ee39ce20b4902682ee7cd6a677c317f6a`.

Le fold 1 de confirmation et le test final restent fermés. Le premier des
trois entraînements BGE cross-fittés nécessaires au stack a démarré après la
matérialisation de cet artefact.
