# Contrat de side quest géographique v16

## Statut et objet

Ce volet est un diagnostic borné exécuté en parallèle du travail principal de
recall SIREN. Il étudie si une localisation CRM résolue par la BAN et les
coordonnées Lambert-93 officielles des établissements SIRENE peuvent améliorer
la génération ou l'ordre des candidats. Il ne change aucun gate v12, ne
modifie pas la recette v15 et ne peut jamais conclure `READY_FOR_FOLD1`.

Statut obligatoire des résultats :
`DIAGNOSTIC_ONLY_BAN_SCORE_INELIGIBLE`.

## Population et séparation des autorités

- folds ouverts : 2, 3 et 4 uniquement ;
- 522 absences du top1750 v11 et 512 contrôles déterministes v15 ;
- le générateur ne reçoit que `query_id` et `fold` ;
- les GT et le type de cohorte restent dans le répertoire truth-side v15 ;
- `X` est produit et scellé avant l'ouverture de `y` ;
- aucun SIRET, SIREN, `query_id` ou fold ne peut être une feature apprise ;
- aucun positif n'est ajouté ou forcé dans un pool ou un échantillon.

## Sources et limites BAN

La source locale est la BAN nationale du 14 août 2026, scellée par son
catalogue. Elle contient longitude/latitude, type et source de position, et
certification communale. Son propre catalogue déclare :

- `address_match_claim_allowed=true` ;
- `address_absence_claim_allowed=false` ;
- `score_eligible=false` ;
- `personal_prospecting_export_allowed=false`.

Conséquences non négociables :

- une absence BAN ne devient jamais un signal négatif ;
- une distance ne constitue jamais un filtre dur ;
- certification, source inconnue, type de position et similarité de voie
  deviennent des features d'incertitude ;
- aucune coordonnée BAN brute n'est persistée par le workbench ;
- cette expérience n'autorise pas l'emploi de la BAN dans un score de
  production. Une telle utilisation exigerait un nouveau contrat explicite.

## Génération label-blind

La résolution BAN est déterministe : même INSEE en priorité, sinon même code
postal, numéro exact, puis meilleure similarité Jaro-Winkler de voie. Un match
faible est conservé avec `ban_declared_weak=true` et une incertitude majorée.
La longitude/latitude est projetée en Lambert-93 avec la définition GRS80/LCC
2SP, sans dépendance additionnelle.

Deux surfaces sont produites :

1. features géographiques des 1,724 M candidats v15 ;
2. voisins SIRENE les plus proches dans le même INSEE et parents distincts
   ordonnés par leur site le plus proche.

Le voisinage est plafonné à 512 sites persistés et 512 parents par requête. Le
calcul parent n'observe au plus que les 4 096 sites les plus proches : ses
résultats sont donc un minorant documenté, surtout dans les grandes communes.

## `X/y` et évaluation

Après scellement de la génération, l'évaluateur truth-side construit `y exact`
et `y même parent`, puis entraîne trois modèles XGBoost OOF : pour chaque fold,
les deux autres seulement servent à l'entraînement. Les features autorisées
sont rangs non identifiants, état actif, concordances adresse, distance,
bornes d'incertitude et indicateurs de qualité BAN.

Sont publiés ensemble :

- oracle exact voisin@250 ;
- oracle parent@128 et parent@512 ;
- rang exact par distance seule ;
- rang exact du modèle OOF à 1, 10 et 100 ;
- rescues nouveaux par rapport au pool v15 ;
- segments BAN faible/forte et certifiée/non certifiée ;
- couverture des coordonnées et absences, sans retirer aucune requête du
  dénominateur.

Ces métriques mesurent une piste. Elles ne sont ni une nouvelle baseline
officielle ni une preuve d'atteinte de Master@2000 ou Recall@100.

## Ressources et préflight obligatoire

Avant le pilote 1 034 requêtes, un run `--preflight-query-count 1..16` doit
passer. Son manifeste porte `NON_DECISIONAL_PREFLIGHT` et ne peut pas être
évalué truth-side. Le plafond est 45 minutes, 14 Gio de RSS et zéro swapout.
Tout dépassement donne `STOP_RESOURCE_POLICY`.

Commande de préflight proposée :

```bash
.venv/bin/python scripts/run_retrieval_geo_v16_workbench.py \
  --config config/retrieval_geo_v16_sidequest.json \
  --queries /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_master_v15_feasibility_cohort/72896b6f70d7c9eb/queries.parquet \
  --preflight-query-count 4 \
  --output-dir /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_geo_v16_preflight/<build_id>
```

Le pilote complet et l'évaluation ne peuvent être lancés qu'après examen du
préflight et dans de nouveaux répertoires immuables.
