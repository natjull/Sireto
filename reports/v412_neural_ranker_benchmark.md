# V4.12-N — benchmark de rerankers neuronaux

Date de clôture : 14 août 2026  
Verdict : **`STOP_PURE_NEURAL_REPLACEMENT`**

## Périmètre

Le cycle a comparé des rerankers de texte au ranker `BUSINESS_LEARNED` sur le
fold 0 gelé, avec les 100 candidats V4.12-L au maximum. Le fold 1 de
confirmation et le test final historique n'ont pas été ouverts. Aucun positif
n'a été injecté et aucun résultat de retrieval ou score XGBoost n'a été écrit
dans le texte présenté aux modèles.

Baseline fold 0 : **2 437/2 797 = 87,129 %** au SIRET exact, dont **33/38**
cas difficiles.

## Zéro-shot

| Modèle | Exact | Difficiles | Actifs | Fermés | Pic RSS | Temps / CRM |
|---|---:|---:|---:|---:|---:|---:|
| BGE v2-m3 | 2 171/2 797 (77,619 %) | 32/38 | 1 941/2 391 | 230/406 | 2,07 Go | 4,88 s |
| CamemBERT mMARCO-FR | 1 846/2 797 (65,999 %) | 25/38 | 1 634/2 391 | 212/406 | 1,58 Go | 4,65 s |
| Qwen3-Reranker-0.6B | 1 782/2 797 (63,711 %) | 23/38 | 1 579/2 391 | 203/406 | 3,12 Go | 5,91 s |
| mMiniLM | 1 606/2 797 (57,419 %) | 18/38 | — | — | — | — |
| GTE multilingual | 62/2 797 (2,217 %) | 0/38 | — | — | — | — |

Artefacts principaux :

- BGE : `v4_12_neural_zero_shot/bge_ref/d02db8f3a7ab68ba` ;
- CamemBERT : `v4_12_neural_zero_shot/camembert_fr/0a98a0999ddb3713` ;
- Qwen : `v4_12_neural_zero_shot/qwen_reranker/08688ac0c2892d56` ;
- mMiniLM : `v4_12_neural_zero_shot/mminilm_ref/cf471d30d0cf42ae` ;
- GTE : `v4_12_neural_zero_shot/gte_reranker/71e1c9c92e43d2f3`.

Tous sont sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/`.

## Fine-tuning CamemBERT

CamemBERT a été entraîné sur les folds 2/3/4 avec 8 192 scènes, chacune
contenant un positif réellement retrouvé et quinze négatifs du pool. La loss
est une entropie croisée groupwise ; un passage, longueur 256, seed 42,
learning rate `1e-5`, quatre couches supérieures entraînables.

Résultat fold 0 :

- **2 353/2 797 = 84,126 %** au SIRET exact ;
- **32/38** difficiles ;
- **2 116/2 391** actifs ;
- **237/406** fermés ;
- 9 484,6 s d'entraînement, 13 398,3 s de scoring et 2,12 Go de pic RSS.

Le fine-tuning apporte 507 bonnes réponses à CamemBERT zéro-shot, ce qui
confirme que l'apprentissage fonctionne. Il reste néanmoins **84 réponses
derrière XGBoost** et 99 réponses sous le gate préenregistré de 2 452. Le fold
1 reste donc fermé. Artefact :
`v4_12_neural_groupwise_cross_encoder/camembert_fr/6512c349738294ac`.

La seconde variante CamemBERT `3e-5` n'a produit aucun artefact final et n'est
pas utilisée. Elle a été abandonnée lors du pivot vers BGE.

## Arrêt de Qwen

Le fine-tuning complet Qwen a été interrompu à 3 000/8 192 scènes sur
instruction explicite. Il n'a publié ni modèle final ni métrique de sélection.
Seul un smoke de deux scènes existe sous
`v4_12_neural_groupwise_qwen/147dbf7084ed3f52`; il ne constitue pas une
évaluation. Le pilote Qwen 1,7B est annulé.

## Conclusion

Aucun reranker neuronal testé seul ne remplace proprement
`BUSINESS_LEARNED`. Le gate du fold 0 n'est pas franchi ; ouvrir le fold 1
serait contraire au protocole. Le verdict de cette famille est donc
**`STOP_PURE_NEURAL_REPLACEMENT`**.

Le signal neuronal reste cependant complémentaire : BGE et CamemBERT
corrigent chacun 74 erreurs XGBoost, avec seulement 33 corrections communes.
La suite légitime est un nouveau cycle préenregistré : fine-tuning de BGE,
puis utilisation de ses scores cross-fittés comme features d'un stack
déterministe avec XGBoost. Ce pivot ne modifie aucun des résultats ci-dessus.
