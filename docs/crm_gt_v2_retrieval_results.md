# Retrieval prospectif sur les nouveaux GT CRM humains

## Population et protocole

- Population humaine totale : 37 218 labels.
- Développement prospectif : 3 510 requêtes du fold 0.
- Partitions reconstruites depuis le snapshot SIRENE courant sur les 4 586
  communes et 3 025 codes postaux observés.
- Plafond : 100 candidats par requête.
- Aucune injection du SIRET vérité.
- Vues SIRET exacte et opérationnelle publiées séparément.
- Fold 1/test non ouvert.

## Résultats du retrieval historique sur les partitions courantes

| Mesure | Résultat |
|---|---:|
| Couverture de qualification | 3 510 / 3 510 = 100,00 % |
| SIRET présent dans le pool géographique | 3 510 / 3 510 = 100,00 % |
| Recall@100 SIRET exact | 3 279 / 3 510 = 93,42 % |
| Recall@100 opérationnel | 3 279 / 3 510 = 93,42 % |
| Oracle lexical multicanal @5000 | 3 448 / 3 510 = 98,23 % |
| Maximum de candidats publié | 100 |

Par état, le Recall@100 exact vaut 95,05 % sur les établissements actifs et
86,43 % sur les fermés.

Verdict : **PIVOT_RETRIEVAL**. Le gate contractuel de 99,0 % n'est pas franchi,
donc aucun entraînement XGBoost, BGE, CamemBERT ou FusionSet ne doit encore
consommer ce nouveau cycle.

## Diagnostic utile

- 231 vérités exactes sont absentes du top 100 final.
- 169 de ces 231 sont néanmoins présentes dans l'oracle lexical @5000 : la
  fusion/allocation des canaux peut les récupérer sans nouvelle source.
- 62 vérités sont absentes de tous les canaux @5000 alors qu'elles sont toutes
  présentes dans le pool géographique. Les cas récurrents incluent notamment
  des marques ou organismes dont la surface CRM diffère du libellé SIRENE
  courant (`POLE EMPLOI`, `GROUPAMA`, `CNAM`, etc.).

La prochaine piste est un canal d'alias appris **uniquement sur les folds
train 2/3/4**, qui propose un SIREN connu par la surface CRM puis classe ses
établissements locaux par adresse. Cette piste est évaluée sur le fold 0 et ne
constitue pas une injection du positif de la requête.

## Artefacts

- Benchmark :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_gt_v2_retrieval_input_commercial/5a9c2437a2a3d817/`
- Canaux courants :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/crm_gt_v2_v7_current_channels_5a9c2437a2a3d817_dev/`
- Canaux fermés historiques :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/crm_gt_v2_overlay_channels_5a9c2437a2a3d817_dev/`
- Évaluation combinée :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/crm_gt_v2_retrieval_eval_5a9c2437a2a3d817_dev/`
