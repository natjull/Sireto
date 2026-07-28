# V4.10 — dataset de l'accepteur structuré

## Verdict

**`GO_TRAIN_V410`**

Le dataset canonique V4.10 est construit sur le SSD externe. Il conserve la
matrice exacte des baselines, ajoute une représentation structurée de
l'identité et du site, et exclut physiquement le random V4.8 des fichiers lus
par le futur trainer.

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10_structured_acceptor/0d6b87fd50fb550c`

## Volumes

| Population | Lignes |
|---|---:|
| Historique | 7 003 |
| Difficiles `hard_oof` | 94 |
| Verrouillés descriptifs | 4 |
| Candidats historiques joints | 698 428 |
| Candidats difficiles | 9 716 |

Les 94 labels difficiles contiennent exactement 68 `TOP1_CORRECT`, 25
`TOP1_WRONG` et un `AMBIGUOUS`. Les cinq folds contiennent respectivement
26, 14, 25, 12 et 17 requêtes. Les 20 supports historiques liés aux
composantes difficiles sont identifiés et portent le même fold.

## Matrices

- 80 features `current80` exactes pour reproduire `BASE_FROZEN` et les
  variantes `CURRENT80`.
- 715 features structurées autorisées au modèle.
- 75 features de provenance retrieval conservées pour audit mais interdites
  au modèle, car elles révèlent la différence V4.1/V4.2-B.
- 884 colonnes physiques par parquet, métadonnées comprises.

Une vérification indépendante sur toutes les lignes confirme :

- égalité bit à bit des 80 features historiques avec les 7 003 scènes
  source ;
- égalité bit à bit des 80 features des 98 cas difficiles avec V4.5 ;
- 715 noms structurés uniques, présents dans chaque population ;
- toutes les valeurs modèle numériques et finies ;
- aucune métadonnée, identité ou feature d'audit dans l'ordre modèle.

## Jointures et intégrité

| Contrôle | Résultat |
|---|---:|
| Jointure prédiction/candidat V4.1 | 698 428 / 698 428 |
| Sentinelles sans candidat conservées | 2 |
| Couverture CRM historique | 7 003 / 7 003 |
| Couverture CRM difficile | 98 / 98 |
| Couverture SIRENE utile | 100 % |
| Cohérence SIREN snapshot/candidat | 100 % |
| Intersection avec random V4.8 | 0 |
| Pool maximal | 100 |
| Positif injecté | 0 |
| Modèle entraîné | non |
| Test final ouvert | non |

La couverture SIRENE à 100 % porte sur top-1, top-2 et tous les candidats du
même SIREN que le top-1 : 15 665 SIRET historiques et 822 SIRET difficiles.

## Linéage

- contrat final : commit `b19abed`, SHA-256
  `91527a57271e5a9410dc6555b6264c817dd2c20d3ce4af1a1903abb6b1f878c4` ;
- builder initial : commit `2966d2b` ;
- correctif de sérialisation nullable : commit `e10e9af` ;
- SHA-256 du builder exécuté :
  `00a4a2843f612dc05b6ec73bf63572c929408f2fa6ee8f20e734402e890d7ffa`.

Le premier lancement s'est arrêté avant création d'un artefact final sur la
sérialisation d'un entier nullable dans le hash des folds. Le correctif
`e10e9af` ajoute un test dédié ; le second lancement a produit l'artefact
ci-dessus.

Hashes de sortie :

| Fichier | SHA-256 |
|---|---|
| `historical_scenes.parquet` | `23016e2e47df2b06f57f66a7b4ba518689eb22a03ce3c1968d14f58738ecc260` |
| `consumed_hard_scenes.parquet` | `88d364a0c496515d5e2e57d8e13bb5976e0212c7bff21b076aeaa041fd4b2536` |
| `descriptive_locked_scenes.parquet` | `203b57ba2d352cea50791f14b6f8a805d4c00da06a598afae922da84d99439e9` |
| `feature_catalog.json` | `280ee7a7b3d0abc50b8776949ab6ec196122ce9acfd47a34b228f18d03686698` |

## Suite autorisée

Le gate autorise seulement l'entraînement de développement préenregistré :
reproduction `BASE_FROZEN`, variantes `CURRENT80`, `STRUCTURED_LOGIT` et
`STRUCTURED_XGB`, prédictions difficiles group-OOF et seuils choisis
uniquement sur le dev historique. Aucun résultat de ce développement ne
vaudra validation ; un modèle admissible devra ensuite affronter un dev CRM
entièrement frais.
