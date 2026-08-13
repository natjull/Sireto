# V4.12-L — ranker appris OOF

Date : 13 août 2026  
Périmètre : développement consommé, sans ouverture du test final.

## Résultat

Le gate `>=225/241` n'est pas franchi. Le meilleur ranker propre atteint
`220/241` sur les cas difficiles et `11 939/13 704` sur tous les labels exacts
du dataset unifié. Verdict : **PIVOT_RANKER**.

| Variante | Global exact | Difficile | Actifs | Fermés |
|---|---:|---:|---:|---:|
| Baseline 45 features | 11 501/13 704 | 211/241 | 88,78 % | 56,88 % |
| Métier appris, 129 features | **11 939/13 704** | **220/241** | **91,75 %** | 61,29 % |
| Métier + faibles labels ouverts | 11 923/13 704 | 217/241 | 91,48 % | **62,06 %** |
| Labels humains pondérés x2 | 11 915/13 704 | 220/241 | 91,62 % | 60,91 % |
| Labels humains pondérés x4 | 11 910/13 704 | 220/241 | 91,69 % | 60,29 % |
| Objectif NDCG | 11 920/13 704 | 218/241 | 91,65 % | 60,96 % |

Le premier run limité par erreur aux 40 premiers négatifs a été rejeté : le
positif se trouve souvent aux rangs d'admission 41–100 et cette troncature
créait un décalage train/inférence. Tous les résultats publiés ci-dessus sont
réentraînés avec les 100 candidats.

## Ce qui a été appris

Les anciennes règles métier ont été converties en variables candidat : nom
opérationnel, rôle et activité, siège, employeur/effectif, comparaisons au
meilleur candidat de la requête, du même SIREN et de la même adresse. XGBoost
reste seul responsable du classement ; aucune variable ne promeut directement
un SIRET.

Les ablations bornées suivantes n'ont pas franchi le gate : augmentation du
poids humain, objectif LambdaMART NDCG, spécialiste entraîné sur les seuls cas
humains, petit cross-encoder multilingue, BGE et reranker appris combinant les
deux scores texte. Les deux modèles texte ont été exécutés sur le GPU intégré
du Mac, sans location ni service payant.

## Limite structurelle pour l'accepteur

Sur les 17 097 requêtes, le ranker ne produit que 11 939 top-1 exacts. Un
accepteur peut refuser une réponse mais ne peut pas la corriger : sa couverture
AUTO exacte est donc bornée avant calibration. Les scènes et l'accepteur OOF
doivent être construits pour mesurer la sélectivité réelle, mais ils ne peuvent
pas rendre atteignable une couverture de 88–92 % sur toute la population avec
ce ranker.

## Artefacts

- features métier 129 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_learned_business_features/8800ef53f6927215` ;
- ablation principale :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_oof_rankers/839ef55308d5077e` ;
- pondérations humaines :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_oof_rankers/ed06ca38cb669291` ;
- NDCG :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_oof_rankers/46803026b12aae59`.

Les 100 misses retrieval restent des erreurs end-to-end. Toutes les
prédictions publiées sont OOF par composante SIREN. Aucun résultat n'est une
certification indépendante.
