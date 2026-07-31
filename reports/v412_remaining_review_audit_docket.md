# V4.12 — audit complémentaire des REVIEW historiques

## Lot gelé avant adjudication

Population source : les 278 décisions `REVIEW` de l'accepteur V4.11 retenu
(`COMPACT_LOGIT`), moins les 90 dossiers déjà adjudiqués dans R30, R53 et la
validation ranker. Il reste exactement 188 dossiers.

La sélection ne consulte ni vérité historique, ni candidat, ni rang, ni score :
elle prend les cinq plus petits `query_id` numériques de cette population.

| Ordre | ID | CRM | Adresse |
|---:|---:|---|---|
| 1 | 344 | COLLEGE LELORGNE DE SAVIGNY - PROVINS | 1 RUE DE SAVIGNY VILLE HAUTE, 77160 PROVINS |
| 2 | 410 | COLLEGE SAINT LOUIS - LIEUSAINT | 124 MAIL DES PEPINIERES, 77127 LIEUSAINT |
| 3 | 896 | CCI Avon | 1 Rue du Port de Valvins, 77210 AVON |
| 4 | 1073 | LFB BIOMANUFACTURING | AVENUE DES CHENES ROUGES, 30100 ALES |
| 5 | 1140 | LG ALES AUTOMOBILES | 157 Chemin du Mas de la Bedosse, 30100 ALES |

Ce lot sert à corriger et qualifier les anciennes étiquettes de développement.
Il ne constitue pas une validation indépendante du modèle et n'ouvre aucun
test final.
