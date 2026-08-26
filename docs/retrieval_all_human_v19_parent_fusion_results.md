# Retrieval parent v19 — résultats de fusion parent@2000

Date : 27 août 2026

## Verdict

Deux verdicts distincts doivent être conservés :

- **`PASS_RRF_PARENT_2000_MASTER_99_2_DEVELOPMENT`** pour la fusion RRF fixe;
- **`STOP_PARENT_2000_CONTROL_REGRESSION`** pour LambdaMART OOF.

La baseline RRF a été construite et scellée sans ouvrir les labels. Elle
conserve le bon parent pour 374 des 522 pertes Master et les 512 contrôles sous
un plafond strict de 2 000. Sa projection oracle est 23 423 / 23 571, soit
99,3721 %. Le palier 99,2 % est donc faisable à `parent@2000`.

LambdaMART OOF ne conserve que 241/522 pertes et 471/512 contrôles. Il est
rejeté sans retuning. Le manifeste officiel garde volontairement ce STOP,
puisque le modèle était la variante primaire de l'évaluateur préenregistré.
Le succès de la baseline déterministe est publié séparément plutôt que de
réécrire le verdict après observation des résultats.

Ni 99,5 % (405/522) ni le niveau de confiance 464 et 160/158/146 par fold ne
sont atteints. Aucun run Master/site/national n'est autorisé par ce résultat.

## Résultats

| Variante | Parent@2000 | Fold 2 | Fold 3 | Fold 4 | Contrôles | Projection Master |
|---|---:|---:|---:|---:|---:|---:|
| RRF fixe | **374/522** | **143** | **123** | **108** | **512/512** | **99,3721 %** |
| LambdaMART OOF | 241/522 | 103 | 78 | 60 | 471/512 | 98,8079 % |

Courbe de conservation :

| Cap parent | RRF pertes | RRF contrôles | LambdaMART pertes | LambdaMART contrôles |
|---:|---:|---:|---:|---:|
| 500 | 279 | 507 | 24 | 246 |
| 1 000 | 318 | 510 | 49 | 311 |
| 1 500 | 357 | 511 | 139 | 402 |
| 1 750 | 367 | 512 | 188 | 437 |
| 2 000 | **374** | **512** | 241 | 471 |

Le RRF réduit la borne brute v18 de 389 à 374, soit seulement 15 parents
perdus en imposant le cap. LambdaMART se comporte presque comme un ordre peu
informatif sur les candidats externes malgré de bonnes features de rang : le
volume d'apprentissage naturel est limité à 582–620 requêtes positives selon
le fold et les motifs rares ne se transfèrent pas correctement.

## Stabilité par origine

| Origine | RRF@2000 | Baseline v17 | Gain |
|---|---:|---:|---:|
| `REAL_CRM_20260817` | 286/389 | 256 | +30 |
| `REAL_CRM_HISTORICAL` | 88/133 | 73 | +15 |

Le RRF ne régresse dans aucune des deux origines internes. Cela ne constitue
toujours pas une preuve cross-CRM : folds2/3/4 appartiennent à la même famille
de données et ont été utilisés pendant plusieurs cycles de développement.

## Protocole et absence de fuite

- 2 580 005 couples requête-parent matérialisés avant ouverture des labels;
- 36 features numériques de provenance et de rang, sans identifiant;
- RRF `k=60` et cap 2 000 scellés avant vérité;
- LambdaMART externe : F2←F3+F4, F3←F2+F4, F4←F2+F3;
- SIREN vérité disjoints entre les folds;
- aucun positif injecté; requêtes sans positif naturel exclues du fit mais
  conservées comme erreurs d'évaluation;
- fold0/fold1 fermés; aucune sélection post-hoc des hyperparamètres.

## Ressources et autorités

Matérialisation label-blind : 1,497 s, pic RSS 2 644 000 768 octets, zéro
swapout.

- manifeste features :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v19_features/dc0461ac2ea01f7b/manifest.json`;
- SHA-256 :
  `9494ea0ed499290057e894d3a2889bc88ec07947d177af1d0f64a0a3d312b12c`.

Entraînement et évaluation OOF : 8,865 s, pic RSS 2 300 903 424 octets, zéro
swapout.

- manifeste OOF :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v19_oof/9494ea0ed4992900/manifest.json`;
- SHA-256 :
  `da566a6e2696616a03ec086e9d626df6955ab49f0f0698bd8e0d5a0927aaddc1`.

## Décision d'architecture

La prochaine étape doit partir de la sélection **RRF fixe**, pas de
LambdaMART. Le modèle peut rester une expérience négative utile; il ne doit
pas être retuné sur les mêmes folds pour récupérer artificiellement le gate.

Le prochain workbench doit transformer les 2 000 parents RRF en au plus 2 000
SIRET Master, avec des quotas de sites et un score site label-blind. Il doit
d'abord être préflighté sur la cohorte 1 034. Fold1 reste fermé jusqu'à ce que
la recette complète parent→site soit gelée. Un shadow set nouveau reste requis
avant toute revendication de généralisation à un autre CRM.
