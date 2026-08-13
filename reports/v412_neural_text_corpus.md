# V4.12-N — corpus texte et baseline gelés

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_neural_text_corpus/02b8668f8050c5e9`

Le corpus contient 17 097 requêtes et 1 708 184 candidats, avec 100 candidats
au maximum. Aucun positif n'est injecté. Le texte d'entrée contient le CRM brut
et les preuves SIRENE brutes ; le SIRET candidat n'est pas sérialisé dans le
texte.

## Baseline BUSINESS_LEARNED

| Fold | Hit@1 exact | Difficiles | Actifs | Fermés |
|---:|---:|---:|---:|---:|
| 0 — sélection | 2 437/2 797 | 33/38 | 2 187/2 391 | 250/406 |
| 1 — confirmation fermée | 2 313/2 655 | 46/50 | 2 060/2 248 | 253/407 |
| 2 | 2 473/2 850 | 49/51 | 2 212/2 420 | 261/430 |
| 3 | 2 368/2 738 | 44/52 | 2 113/2 301 | 255/437 |
| 4 | 2 348/2 664 | 50/52 | 2 089/2 259 | 259/405 |
| Total | 11 939/13 704 | 222/243 | 10 661/11 619 | 1 278/2 085 |

Le fold 1 n'a pas été scoré par un nouveau modèle. Ses valeurs de baseline sont
publiées avant sélection pour rendre le gate vérifiable, mais ses nouvelles
prédictions restent fermées jusqu'à ce qu'un modèle franchisse le fold 0.
