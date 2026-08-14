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

## Accepteur pré-Maps et objectif produit

Le gate ranker reste une précondition technique. Un candidat qui le franchit
n'est toutefois pas `GO` produit tant qu'il n'améliore pas l'automatisation
sûre avant Google Maps.

L'accepteur ne classe et ne remplace jamais un SIRET. Il reçoit le top 1 du
ranker gelé et rend exclusivement `AUTO_MATCH` ou `REVIEW`. Toute ligne
`REVIEW` est un appel Maps théorique ; aucune API Maps n'est appelée dans ce
cycle. Un futur rematching après réparation Maps sera un second passage
séparé, avec ranker et accepteur gelés.

### Scènes et prédictions nested OOF

Après promotion technique d'un ranker, des prédictions candidat complètes
sont produites hors apprentissage pour chaque fold :

- folds 0 et 1 : modèles BGE et stack appris sur 2/3/4 ;
- fold 2 : modèles appris sur 3/4 ;
- fold 3 : modèles appris sur 2/4 ;
- fold 4 : modèles appris sur 2/3.

Chaque requête `MATCH_EXACT`, `AMBIGUOUS` ou `UNRESOLVED` reçoit exactement
une scène OOF. Les cibles de l'accepteur sont `1` uniquement si le top 1 est
le SIRET exact ; les erreurs de ranker, les ambiguïtés et les non-résolus sont
des négatifs. Aucune scène n'est supprimée.

La scène reprend les agrégats query-level V4.12-L compatibles et ajoute :

- accord top 1 XGBoost/BGE et, si la branche conditionnelle existe,
  CamemBERT ;
- rangs top 1 réciproques et rang de chaque modèle pour le candidat retenu ;
- marges top 1/top 2 propres à XGBoost, BGE et stack ;
- écart de score normalisé entre XGBoost et BGE ;
- stabilité du top 1 entre ranker historique, BGE et stack ;
- identité ou différence des SIREN choisis par chaque modèle ;
- nombre de SIREN distincts dans les top 2 et top 10 ;
- différence entre le meilleur concurrent du même SIREN et le meilleur
  concurrent d'un autre SIREN.

Les features ne contiennent ni vérité, ni correction par dossier, ni décision
Maps.

### Apprentissage et calibration de l'accepteur

Deux familles déjà fixées dans V4.12-L sont comparées sans nouveau tuning :

- logistique standardisée : `C=0.2`, `max_iter=1500`, seed 42 ;
- XGBoost : 450 arbres, learning rate `0.025`, profondeur 4,
  `min_child_weight=10`, `subsample=0.85`, `colsample_bytree=0.80`,
  `reg_lambda=10`, `reg_alpha=0.25`, seed 42.

Pour chaque fold externe `f`, le fold de calibration vaut `(f+1) mod 5` et le
modèle apprend sur les trois folds restants. Sur le fold de calibration, le
seuil maximise le nombre d'AUTO sous deux contraintes : précision observée
`>= 99,8 %` et zéro `AMBIGUOUS/UNRESOLVED` humainement audité automatisé.
Le fold externe n'intervient ni dans le fit ni dans le choix du seuil.

Une baseline accepteur est reconstruite avec exactement ce protocole sur le
top 1 `BUSINESS_LEARNED`. Le candidat utilise le nouveau top 1 et les signaux
de désaccord. Dans chaque système, la famille gagnante est celle qui maximise
la couverture OOF agrégée sous la précision observée `>=99,8 %` et zéro cas
ouvert audité AUTO ; une égalité va à la logistique.

### Métriques pré-Maps obligatoires

Les dénominateurs sont toujours écrits avec les nombres bruts :

- Hit@1 exact : bonnes réponses / 13 704 identifiables ;
- AUTO total CRM : AUTO / 17 097 ;
- précision AUTO : AUTO corrects / AUTO ;
- AUTO parmi les identifiables : AUTO exacts / 13 704 ;
- REVIEW et taux théorique d'appels Maps : REVIEW / 17 097 ;
- appels Maps évités : AUTO / 17 097 ;
- bornes de Wilson bilatérales 95 % et 99 % pour précision et couvertures.

La couverture identifiable est **13 704/17 097 = 80,154 %**. Les résultats ne
doivent jamais présenter cette borne comme une couverture automatique
atteignable ou dépassable sans nouvelles preuves pour les 3 393 cas ouverts.

La comparaison appariée publie aussi :

1. erreurs top 1 `BUSINESS_LEARNED` corrigées par le nouveau ranker ;
2. top 1 déjà corrects mais nouvellement AUTO grâce à l'accord neuronal ;
3. décisions auparavant AUTO désormais `REVIEW`/Maps en présence d'un
   désaccord neuronal ;
4. régressions de top 1 et pertes d'AUTO correctes.

### Gate produit pré-Maps

Un `GO_PRODUCT_PRE_MAPS` exige simultanément :

1. gate ranker fold 0 puis confirmation fold 1 franchis ;
2. précision AUTO OOF agrégée `>=99,8 %` ;
3. zéro `AMBIGUOUS/UNRESOLVED` humainement audité en AUTO ;
4. couverture AUTO totale supérieure d'au moins **1,0 point absolu** à la
   baseline accepteur reconstruite avec le même protocole ;
5. aucune baisse de la couverture AUTO parmi les 13 704 exacts ;
6. toutes les scènes produites, scores finis et décisions déterministes.

Une amélioration du Hit@1 sans ce gain AUTO vaut `PIVOT_PRODUCT_ACCEPTOR`.
Un gain AUTO sans confirmation ranker vaut également `PIVOT`. Aucun résultat
de ce cycle n'autorise un appel Maps réel ni une revendication de performance
post-Maps.

## Confirmation fold 1 et verdict ranker

Si le gate fold 0 passe, la politique et tous les poids sont gelés puis le
fold 1 est ouvert exactement une fois. `GO` exige :

- au moins dix bonnes réponses nettes de plus que `BUSINESS_LEARNED` sur le
  même fold ;
- aucune baisse sur les cas difficiles ;
- aucune baisse supérieure à un point absolu sur actifs ou fermés ;
- toutes les requêtes scorées et les mêmes contrôles de fuite/intégrité.

Un gain fold 0 qui ne se confirme pas vaut `PIVOT`. Un stack qui n'améliore
pas matériellement XGBoost sur fold 0 vaut `STOP`. Après confirmation, le gate
accepteur ci-dessus décide le verdict produit final. Le test final reste fermé
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
