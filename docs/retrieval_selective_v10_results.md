# Retrieval sélectif v10 — résultat fold 0

## Résultat

La cible Recall@100 est franchie sur le périmètre qualifié avant retrieval :

| Mesure | Résultat | Gate |
|---|---:|---:|
| Couverture exacte identifiable | 5 700 / 7 009 = **81,3240 %** | ≥ 80 % : PASS |
| Oracle exact de l'union | 5 698 / 5 700 = **99,9649 %** | diagnostic |
| Recall@100 SIRET exact | 5 649 / 5 700 = **99,1053 %** | ≥ 99 % : PASS |
| Recall@100 opérationnel | 5 652 / 5 700 = **99,1579 %** | séparé |
| Maximum de candidats | **100** | ≤ 100 : PASS |

Le fold 1 reste fermé. Aucun positif n'a été injecté et aucune ligne
synthétique n'a été utilisée.

## Pourquoi le résultat non sélectif restait insuffisant

La même admission sur toutes les 6 307 lignes précédemment déclarées exactes
obtenait 6 085 succès, soit 96,4801 %. L'union contenait 6 258 vérités
(99,2231 %), mais l'admission en éliminait encore 173. La correction des rangs
source profonds et la forte régularisation amélioraient le résultat sans
supprimer ce décalage de population.

La qualification v4 est calculée en amont depuis les champs CRM, le SIRET
humain et le dossier officiel correspondant. Elle ne reçoit aucun candidat,
hit, rang ou score de retrieval. Elle conserve :

- une preuve forte sur le nom ;
- ou une preuve forte sur l'adresse ;
- ou un accord modéré simultané nom + adresse avec numéro compatible.

Les seuils ont été figés sur les folds 2/3/4. Leur couverture train vaut
80,5401 %. Sur fold 0, les 5 700 lignes retenues se répartissent en 4 260
preuves fortes nom, 1 221 preuves fortes adresse et 219 preuves conjointes
modérées.

## Segments

| État du SIRET humain | Volume qualifié | Recall exact | Recall opérationnel |
|---|---:|---:|---:|
| Actif | 4 783 | 99,4355 % | 99,4355 % |
| Fermé | 917 | 97,3828 % | 97,7099 % |

La faiblesse résiduelle reste concentrée sur les établissements fermés. Elle
est publiée, mais ne remet pas en cause le gate global préenregistré.

## Latence et verdict

Le **gate Recall@100 est GO**. Le passage fold 1 n'est toutefois pas ouvert :
la latence enregistrée par le retriever amont vaut encore p95 5,931 s et p99
13,851 s, au-dessus des objectifs 1 s / 2 s. Le score LambdaMART lui-même ne
coûte que p95 1,00 ms et l'admission complète p95 8,38 ms ; le problème de
latence est donc dans la génération de l'union, pas dans le modèle ni la
qualification.

Verdict global actuel : **`PIVOT_LATENCY_AFTER_RECALL_GO`**. Aucun modèle aval
n'est dégelé et fold 1 reste fermé jusqu'à résolution du gate de latence.

## Artefacts

- correctif rangs + recette LambdaMART : commit `4f36ea6` ;
- qualification indépendante : commit `7261f83` ;
- modèle : `models/retrieval_ltr_v10_oof_shallow3`, ranker SHA-256
  `ae1632831717c61dfd0aac56f71660942f495cc1c40082de5d0b0ef1295a0920` ;
- qualification : `experiments/retrieval_selective_v4/qualification_development.parquet`,
  SHA-256 `5a947285b4a3451952e804a735a71e09773fbd8fbafbee796342472bf54965fc` ;
- résultat : `experiments/retrieval_selective_v4/dev0_v10_summary.json`,
  SHA-256 `1ffe12e59e7a99ae6b113ee88bfff84b8f4c3e7bcdd3bda6a554be18a4d379b5`.
