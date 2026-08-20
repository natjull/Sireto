# Résultat du retrieval temporel SIRENE v3

Date : 21 août 2026  
Verdict : **`PIVOT_ORACLE_BELOW_99_3`**

## Périmètre exécuté

Le replay porte uniquement sur le fold de développement 0 : 7 009 requêtes,
dont 6 307 qualifiées `identifiable_exact`. Le fold 1 est resté fermé. Aucun
positif n'a été injecté, aucun label synthétique n'a été utilisé et aucun
LambdaMART, ranker, decider ou risk model n'a été entraîné.

Trois index ont été interrogés :

- index dossier SIREN courant/RNE/BODACC
  `hierarchical_retrieval_v3/4b332899d8698354` ;
- overlay officiel historique précédent
  `official_evidence_overlay_sirene_history_v1/5bb04044578179db` ;
- nouvel overlay temporel SIRENE
  `sirene_temporal_v1/506bb98cfdcccfbe`.

Le nouvel overlay national contient 19 750 392 documents : 16 727 674
documents SIRET et 3 022 718 documents `(INSEE,SIREN)`. Il ne contient ni
label CRM ni ligne synthétique. Les adresses courantes servant à géolocaliser
un ancien nom sont stockées comme `geo_scaffold` mais ne sont jamais indexées
ou présentées comme des adresses historiques. Les liens de succession sont
directionnels et limités à un saut.

## Résultats

| Mesure | Ancien replay | Temporel v3 | Delta |
|---|---:|---:|---:|
| Couverture identifiable | 6 307/7 009 (89,9843 %) | identique | 0 |
| Oracle exact union ≤2 000 | 6 179/6 307 (97,9705 %) | 6 231/6 307 (98,7950 %) | **+52 / +0,8245 pp** |
| Rang brut top100, sans admission apprise | 5 939/6 307 (94,1652 %) | 6 038/6 307 (95,7349 %) | **+99 / +1,5697 pp** |
| Oracle opérationnel | 6 179/6 307 | 6 231/6 307 | +52 |
| p95 retrieval | 1 101,7 ms | 1 533,3 ms | +431,6 ms |
| p99 retrieval | 1 422,1 ms | 1 887,6 ms | +465,5 ms |

Le nouvel oracle échoue le gate préenregistré de 99,3 %. Il fallait au moins
6 263 hits ; 6 231 sont présents, soit un déficit de 32. Il est également
inférieur au strict minimum informationnel de 6 244 hits nécessaire pour
permettre un Recall@100 exact de 99,0 %, indépendamment de l'admission.
Le p95 échoue aussi le SLO de 1 000 ms ; le p99 reste sous 2 000 ms.

Artefacts :

- union :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/retrieval_dossier_v3_temporal_truthful_dev/raw_union_dev_temporal_v3.parquet` ;
- SHA-256 union :
  `12ca8ca40361490278f207439d65815e86f6ee86609a3da13ac7051af89518a9` ;
- 12 126 438 lignes candidates, zéro requête vide, maximum émis 2 000 ;
- rapport : même répertoire, `oracle_report_v3.json`.

## Origine des gains

Le replay temporel ajoute 52 vérités et n'en perd aucune par rapport au replay
antérieur. Les canaux présents sur les nouvelles vérités sont :

- 37 expansions hiérarchiques d'un ancien nom au grain SIREN vers ses sites
  courants locaux ;
- 14 traversées d'un lien de succession SIRET officiel structuré ;
- 2 rappels directs par ancien nom SIRET/SIREN ;
- un chevauchement entre ces familles explique que le total brut soit 53.

Les deux familles historiques ont donc une valeur mesurée, particulièrement
sur les anciens labels `POLE EMPLOI` désormais exposés sous une identité
courante différente, et sur les prédécesseurs/successeurs explicites.

## Diagnostic du résiduel et PIVOT suivant

Il reste 76 misses : 61 vérités actives et 15 fermées, toutes issues de labels
CRM humains. Pour 30, un autre SIRET du bon SIREN est déjà présent dans
l'union. Un diagnostic read-only sur l'index national montre que, pour 29 de
ces 30 requêtes, le SIRET exact figure dans les 32 sites locaux retournés par
`sites_for_siren` dès lors que le bon SIREN est effectivement développé.

Le défaut architectural précis est donc que la route hiérarchique développe
uniquement les SIREN trouvés par les documents parents `(INSEE,SIREN)`. Un
candidat SIRET direct apporte pourtant une preuve déterministe de son SIREN,
mais cette preuve ne promeut pas encore le parent dans la file d'expansion.
La prochaine version doit unir, avant expansion :

1. les SIREN trouvés par l'index parent ;
2. les SIREN portés par les meilleurs candidats SIRET directs ;
3. les SIREN atteints par une succession officielle compatible.

Cette modification n'est pas évaluée dans le présent cycle : la config impose
un unique passage dev et interdit le réglage automatique après lecture du
résultat. Elle devra être préenregistrée comme nouvelle recette, avec une
sélection parentale bornée et un travail séparé sur la latence. Aucun accès au
fold 1 ni entraînement aval n'est autorisé avant un nouvel oracle ≥99,3 %.

