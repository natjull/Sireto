# Groupes d'apprentissage V4.12-BGE

Artefact immuable :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_bge_training_groups/114b407f2ccf7b40`.

- 8 192 scènes et 131 072 paires ;
- folds 2/3/4 uniquement : 2 826, 2 718 et 2 648 scènes ;
- exactement 16 candidats par scène ;
- exactement un positif, déjà présent dans le pool ;
- zéro SIREN vérité traversant deux folds ;
- zéro injection positive.

Les 122 880 négatifs sont répartis ainsi :

| Catégorie | Lignes | Scènes concernées |
|---|---:|---:|
| top XGBoost OOF | 40 960 | 8 192 |
| autre SIRET du même SIREN | 1 659 | 847 |
| homonyme ou adresse forte | 12 028 | 4 686 |
| concurrent actif/fermé | 16 384 | 8 192 |
| complément par rang | 51 849 | 8 192 |

Un premier build `12b9127e397bbc65` a été rejeté avant entraînement : la
condition `same_address_count > 1` était trop large et absorbait les autres
familles de négatifs. Il reste physiquement présent pour traçabilité mais est
supersédé par `114b407f2ccf7b40`, qui applique les quotas préenregistrés.
