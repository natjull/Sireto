# Contrat préenregistré — BGE fine-tuné et stack XGBoost V4.12-BGE

Date de gel : 15 août 2026  
Statut au gel : BGE zéro-shot déjà lu ; aucun score BGE fine-tuné ni score du
nouveau stack lu. Fold 1 et test final fermés.

## Question expérimentale

Ce cycle teste deux hypothèses séparées :

1. le fine-tuning groupwise de `BAAI/bge-reranker-v2-m3` peut-il battre seul
   `BUSINESS_LEARNED` sur les 100 candidats V4.12-L ?
2. même s'il reste moins bon seul, son signal complémentaire peut-il améliorer
   un ranker XGBoost déterministe qui ne reconsidère que le top 10 XGBoost ?

Le retrieval, le CRM, les labels et les résultats V4.12-N déjà publiés ne sont
pas modifiés. Une vérité absente du pool ou du top 10 du stack reste une erreur
end-to-end. Aucune sélection oracle, promotion par `query_id`, règle par dossier
ou injection du positif n'est autorisée.

## Entrées immuables

- corpus texte top 100 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_neural_text_corpus/02b8668f8050c5e9` ;
- features métier candidat :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_learned_business_features/8800ef53f6927215` ;
- prédictions OOF et modèles de référence `BUSINESS_LEARNED` :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_oof_rankers/839ef55308d5077e` ;
- modèle : `BAAI/bge-reranker-v2-m3`, révision Hugging Face
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` ;
- seed globale : `42` ;
- plafond absolu : 100 candidats pour BGE seul, 10 candidats pour le stack.

Les manifests et tous les fichiers consommés sont vérifiés par SHA-256 avant
chaque build ou entraînement. Les artefacts finaux publient les hashes, la
révision du modèle, les paramètres, les versions Python/PyTorch/Transformers/
XGBoost et le commit source.

## Splits et étanchéité

Les folds existants sont groupés par composantes SIREN.

- apprentissage et cross-fitting : folds 2, 3 et 4 uniquement ;
- sélection et unique tuning autorisé : fold 0 ;
- confirmation : fold 1, ouvert une seule fois seulement si le gate fold 0
  est franchi ;
- test final historique : fermé pendant tout ce cycle.

Pour produire les scores BGE d'apprentissage du stack, chaque fold parmi
2/3/4 est scoré par un BGE entraîné uniquement sur les deux autres folds. Le
modèle BGE utilisé sur le fold 0 est entraîné sur 2/3/4. Le stack XGBoost est
entraîné uniquement sur les scores cross-fittés des folds 2/3/4. Aucun score
in-sample BGE ne peut entrer dans le stack.

Les scores et rangs `BUSINESS_LEARNED` utilisés pour le top 10 sont les
prédictions OOF gelées : le modèle de base n'a jamais appris sur la requête
qu'il classe. Ils servent au minage, au tronquage et comme features ; aucune
vérité n'est utilisée pour choisir le top 10.

## Groupes BGE et négatifs difficiles

Une scène contient exactement un positif déjà présent dans le top 100 et
jusqu'à quinze négatifs réellement retrouvés. Le positif n'est jamais ajouté
à un pool qui ne le contenait pas.

Les négatifs sont dédupliqués par SIRET puis pris selon cet ordre déterministe :

1. cinq premiers négatifs du ranker `BUSINESS_LEARNED` OOF ;
2. autres SIRET du même SIREN que la vérité ;
3. concurrents à la même adresse (`same_address_count > 1`) ou portant un
   nom/une enseigne fortement similaire
   (`max(source_name_score, name_jaro_max) >= 0.90`) ;
4. concurrents dont l'état SIRENE `A/F`, lu dans le texte candidat gelé,
   diffère de `ground_truth_state` ;
5. meilleurs candidats restants par rang XGBoost, puis rang retrieval et
   SIRET lexical.

Les catégories sont publiées pour chaque ligne (`xgb_top`, `same_siren`,
`homonym_or_same_address`, `state_competitor`, `retrieval_fill`). Elles ne
constituent pas une règle de décision.

## Fine-tuning BGE gelé

Une seule configuration pleine est autorisée :

- loss : entropie croisée groupwise ;
- un passage sur les groupes disponibles ;
- un positif + quinze négatifs maximum ;
- longueur maximale 256 tokens par paire ;
- couches basses gelées, quatre dernières couches et tête entraînables ;
- AdamW, learning rate `1e-5`, weight decay `0.01` ;
- warm-up linéaire sur 10 % des pas ;
- clipping du gradient à `1.0` ;
- une scène par batch ;
- aucun sampling stochastique à l'inférence ;
- tie-break : score décroissant, rang retrieval croissant, SIRET lexical.

Un smoke de ressources peut utiliser au plus seize scènes et seize requêtes
d'un fold d'apprentissage. Il ne permet aucune sélection de performance.

BGE fine-tuné seul est ensuite scoré une fois sur les 100 candidats du fold 0.
Il est comparé à BGE zéro-shot et à `BUSINESS_LEARNED`, mais son résultat ne
modifie pas les paramètres du stack.

## Stack déterministe XGBoost + BGE

Pour chaque requête, le ranker de base fournit son top 10 OOF. BGE score ces
dix candidats. Le méta-ranker reçoit les 129 features métier gelées et les
features suivantes :

- score et rang OOF `BUSINESS_LEARNED` ;
- score BGE brut ;
- rang BGE et son inverse ;
- écart entre le meilleur score BGE de la scène et le candidat ;
- écart top 1 / top 2 BGE, recopié au niveau candidat ;
- accord `top1 XGBoost = top1 BGE` ;
- indicateurs candidat top 1 XGBoost et candidat top 1 BGE ;
- différence des rangs XGBoost et BGE.

Il utilise les hyperparamètres `BUSINESS_LEARNED` déjà gelés : objectif
`rank:pairwise`, 600 arbres, profondeur 6, learning rate `0.035`,
`min_child_weight=3`, `subsample=0.85`, `colsample_bytree=0.85`,
`reg_lambda=5`, `tree_method=hist`, seed 42. Aucun tuning supplémentaire du
stack n'est autorisé.

Le classement final est déterministe : score méta décroissant, score
`BUSINESS_LEARNED` décroissant, rang retrieval croissant, SIRET lexical.

## Mesures et matrices obligatoires

Pour BGE seul et pour le stack :

- Hit@1 SIRET exact global et nombres bruts ;
- cas difficiles audités, actifs et fermés ;
- toutes les requêtes, y compris vérité hors pool/top 10 ;
- corrections XGBoost→juste, régressions XGBoost→faux, faux→faux et accords ;
- matrice croisée XGBoost/BGE/stack ;
- latence totale, moyenne, p95 par lot et pic RSS ;
- nombre de candidats par requête et couverture de scoring ;
- tests de fuite SIREN, cross-fitting, unicité du positif et absence
  d'injection.

## Gate fold 0

Le stack peut ouvrir le fold 1 uniquement s'il satisfait simultanément :

1. au moins **2 452/2 797** SIRET exacts ;
2. au moins **33/38** cas difficiles ;
3. au moins **2 164/2 391** établissements actifs ;
4. au moins **246/406** établissements fermés ;
5. les 2 797 requêtes ont une décision et aucun pool source ne dépasse 100 ;
6. zéro score BGE in-sample dans le train du stack, zéro SIREN de vérité
   partagé entre folds et zéro injection positive.

BGE seul n'est pas requis pour franchir ce gate : il peut rester sous XGBoost
si le stack gagne proprement.

Si le stack améliore XGBoost d'au moins dix bonnes réponses mais reste entre
2 447 et 2 451, un unique enrichissement CamemBERT cross-fitté est autorisé,
avec la configuration CamemBERT déjà publiée. Dans tous les autres cas, aucun
score CamemBERT supplémentaire n'est calculé.

## Confirmation fold 1 et verdict

Si le gate fold 0 passe, la politique et tous les poids sont gelés puis le
fold 1 est ouvert exactement une fois. `GO` exige :

- au moins dix bonnes réponses nettes de plus que `BUSINESS_LEARNED` sur le
  même fold ;
- aucune baisse sur les cas difficiles ;
- aucune baisse supérieure à un point absolu sur actifs ou fermés ;
- toutes les requêtes scorées et les mêmes contrôles de fuite/intégrité.

Un gain fold 0 qui ne se confirme pas vaut `PIVOT`. Un stack qui n'améliore
pas matériellement XGBoost sur fold 0 vaut `STOP`. Le test final reste fermé
quel que soit le verdict.

## Ressources et arrêts

- machine : Mac M4 Pro, 24 Go de mémoire ;
- stockage et artefacts : `/Volumes/CATNAT_DATA` ;
- aucun GPU loué, API payante ou service externe ;
- arrêt d'un fit après deux OOM MPS sur la même configuration ;
- arrêt si une extrapolation issue du smoke dépasse 9 heures pour un fit ou
  si un fit dépasse effectivement 12 heures ;
- plafond du cycle BGE cross-fitté : 36 heures de calcul cumulé ;
- arrêt si le pic RSS dépasse 18 Go ;
- aucun effacement des résultats historiques ; les sorties incomplètes sont
  non publiées ou mises en quarantaine traçable.

Chaque milestone — contrat, groupes, fine-tuning, stack, verdict et handover —
est livré dans un commit isolé cité dans `handover.md`.
