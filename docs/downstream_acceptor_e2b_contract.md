# Contrat E2b — score de confiance exact-SIRET

## Objet

E2 a validé le nouveau ranker mais son accepteur calibré par isotonic ne
couvre que 33/1 280 requêtes au point de précision observée 99,8 %. E2b teste
une seule hypothèse : la calibration a-t-elle supprimé une information de
classement utile pour décider `AUTO_MATCH` ?

E2b ne modifie ni le retrieval, ni les candidats, ni le ranker, ni les
features, ni les labels.

## Entrées gelées

- dataset :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068` ;
- prédictions du ranker :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b/ranker_predictions.parquet` ;
- scènes OOF train et scènes dev : reconstruites par le code V9 existant ;
- seed : `42` ;
- découpe dev : rôles `calibration` et `threshold` déjà déterminés par
  `split_dev_roles`.

Le split test sélectif consommé est interdit. Aucun holdout final ne doit être
chargé ou évalué pendant E2b.

## Variantes autorisées

Deux modèles seulement :

1. régression logistique avec standardisation des features ;
2. XGBoost avec les hyperparamètres E2 inchangés.

Pour chacun, comparer exactement trois transformations du score :

1. `raw` : probabilité brute du modèle, sans calibration ;
2. `sigmoid` : calibration logistique ajustée sur la moitié dev
   `calibration` ;
3. `isotonic` : calibration isotonic ajustée sur cette même moitié.

Aucun hyperparamètre supplémentaire, sélection de features, règle métier ou
seuil segmentaire ne peut être essayé dans cette expérience.

## Sélection

Pour chaque couple modèle–transformation :

- ajuster le modèle uniquement sur les scènes train OOF ;
- ajuster la calibration uniquement sur la moitié dev `calibration` ;
- sélectionner le seuil uniquement sur la moitié dev `threshold` ;
- publier les points de couverture maximale aux précisions observées 99,0 %,
  99,5 % et 99,8 %, avec nombres bruts et erreurs ;
- retenir le couple offrant la plus grande couverture à 99,8 %, puis la plus
  grande précision, puis l'identifiant lexical pour départager exactement une
  égalité.

Le seuil exige au moins 25 décisions AUTO.

## Verdict

- `PASS_E2B` si la variante retenue atteint au moins 25 % de couverture avec
  au moins 99,8 % de précision SIRET exacte observée sur la moitié
  `threshold` ;
- `STOP_E2B` sinon.

Un `STOP_E2B` ferme la piste « simple calibration » : il ne doit pas entraîner
une succession de petits réglages sur le même dev.

Même en cas de `PASS_E2B`, le résultat reste une estimation de développement,
pas une garantie. Le bundle et son seuil devront être gelés avant de
constituer un nouveau holdout indépendant.

## Contrôles obligatoires

- mêmes 80 features de scène et même ordre au train et à l'inférence ;
- scènes train exclusivement issues des prédictions OOF groupées par SIREN ;
- résultat d'inférence explicitant la méthode de calibration ;
- sauvegarde et rechargement du bundle reproduisant les confiances ;
- aucune métrique test dans le rapport ;
- suite complète de tests verte.
