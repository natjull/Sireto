# Étape 2 — baseline V142, sans reclassement ni apprentissage

`outputs.json["learned_native_ooc"]` ; ordre émis conservé, rangs indexés à1.
**526 requêtes / 52 600 candidats**, folds2/3/4 seulement ; 100 uniques par requête.
`baseline` est une autre variante (481 exacts@100), pas la référence507.

| Périmètre | Vue | Top1 n/N (%) | Wilson95 top1 (%) | Présent@100 n/N (%) | Wilson95 @100 (%) |
|---|---|---:|---:|---:|---:|
| Total | SIRET exact | 353/526 (67.11) | [62.98 ; 70.99] | 507/526 (96.39) | [94.43 ; 97.68] |
| Total | SIREN | 406/526 (77.19) | [73.41 ; 80.57] | 511/526 (97.15) | [95.35 ; 98.26] |
| Total | Opérationnel historique | 377/526 (71.67) | [67.68 ; 75.36] | 508/526 (96.58) | [94.66 ; 97.82] |
| Total | Site opérationnel | 315/526 (59.89) | [55.64 ; 63.99] | 404/526 (76.81) | [73.01 ; 80.21] |
| Total | Union exact ∪ site | 391/526 (74.33) | [70.43 ; 77.88] | 511/526 (97.15) | [95.35 ; 98.26] |
| Fold 2 | SIRET exact | 124/178 (69.66) | [62.55 ; 75.94] | 172/178 (96.63) | [92.84 ; 98.45] |
| Fold 2 | SIREN | 137/178 (76.97) | [70.25 ; 82.54] | 176/178 (98.88) | [96.00 ; 99.69] |
| Fold 2 | Opérationnel historique | 131/178 (73.60) | [66.67 ; 79.52] | 173/178 (97.19) | [93.59 ; 98.79] |
| Fold 2 | Site opérationnel | 116/178 (65.17) | [57.92 ; 71.78] | 145/178 (81.46) | [75.11 ; 86.48] |
| Fold 2 | Union exact ∪ site | 135/178 (75.84) | [69.05 ; 81.54] | 176/178 (98.88) | [96.00 ; 99.69] |
| Fold 3 | SIRET exact | 107/162 (66.05) | [58.46 ; 72.89] | 153/162 (94.44) | [89.78 ; 97.05] |
| Fold 3 | SIREN | 119/162 (73.46) | [66.17 ; 79.66] | 153/162 (94.44) | [89.78 ; 97.05] |
| Fold 3 | Opérationnel historique | 113/162 (69.75) | [62.29 ; 76.30] | 153/162 (94.44) | [89.78 ; 97.05] |
| Fold 3 | Site opérationnel | 92/162 (56.79) | [49.09 ; 64.17] | 120/162 (74.07) | [66.82 ; 80.21] |
| Fold 3 | Union exact ∪ site | 115/162 (70.99) | [63.58 ; 77.43] | 153/162 (94.44) | [89.78 ; 97.05] |
| Fold 4 | SIRET exact | 122/186 (65.59) | [58.51 ; 72.04] | 182/186 (97.85) | [94.60 ; 99.16] |
| Fold 4 | SIREN | 150/186 (80.65) | [74.37 ; 85.68] | 182/186 (97.85) | [94.60 ; 99.16] |
| Fold 4 | Opérationnel historique | 133/186 (71.51) | [64.63 ; 77.51] | 182/186 (97.85) | [94.60 ; 99.16] |
| Fold 4 | Site opérationnel | 107/186 (57.53) | [50.34 ; 64.41] | 139/186 (74.73) | [68.03 ; 80.43] |
| Fold 4 | Union exact ∪ site | 141/186 (75.81) | [69.17 ; 81.40] | 182/186 (97.85) | [94.60 ; 99.16] |

## Rang du SIRET humain exact

| Périmètre | N | 1 | 2–3 | 4–10 | 11–30 | 31–100 | Absent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Total | 526 | 353 | 73 | 41 | 17 | 23 | 19 |
| Fold 2 | 178 | 124 | 21 | 13 | 6 | 8 | 6 |
| Fold 3 | 162 | 107 | 21 | 8 | 6 | 11 | 9 |
| Fold 4 | 186 | 122 | 31 | 20 | 5 | 4 | 4 |

## Définitions et limites

- Exact/SIREN : identité humaine ; opérationnel historique : exact ou SIRET accepté dans le sidecar officiel scellé.
- Site opérationnel : même SIREN ET (adresse concordante OU distance admissible), fonctions du contrat `fb604665` réutilisées sans changement.
- Distance200 m si les deux précisions sont au numéro, sinon500 m pour une précision de voie ; commune/inconnue/absente exclue de cette preuve.
- Union = exact OU site. Exact ne vaut pas automatiquement succès site ; absence de preuve site ne prouve pas une erreur fibre.
- Top1 global = succès/N, y compris les19 absences exactes ; aucun statut/label ne filtre le dénominateur.
- Aucun accepteur appliqué : taux et précision des `SUR` **non mesurés**, pas de seuil inventé.
- Wilson bilatéral95 descriptif sur les comptes bruts ; aucune précision nationale indépendante ni borne pondérée revendiquée.
- Pilote déjà consommé, encodeur non entièrement OOF, affectation amont fondée sur le SIREN humain : défaut de serving non corrigé.
- Sources stock/étude : signatures taille/mtime/inode contrôlées avant/après, SHA hérités non recalculés ; petites autorités hashées avant/après.
- Hydratation label-side des seuls candidats émis du bon SIREN pour la métrique, jamais une feature ni une modification des listes.
- Aucun apprentissage, génération, web, jeu fermé, label modifié ou positif injecté.

Inventaire et première expérience recommandée : `docs/step2_downstream_inventory.md`.
Commande : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_baseline_v142.py`.
JSON voisin : sources/hashes, totaux, folds et526 lignes query_id avec rangs des cinq vues (aucun nom/adresse CRM brut).
