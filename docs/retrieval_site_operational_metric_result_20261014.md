# Mesure supplémentaire « site opérationnel » — 14 octobre 2026

Contrat préenregistré : `fb604665562cb9c4421d6a4ba740fddfae0e0a58`. Run v142 `all_new_sources_max_bano`,
`learned_native_ooc` ; **526/526**, folds 2/3/4, sorties scellées ≤100.
Analyse secondaire, exacte toujours principale ; pas de certification OOF/nationale ni promotion.

| Vue | Succès | % brut | Wilson 95 % brut | % pondéré Hájek |
|---|---:|---:|---:|---:|
| SIRET exact | 507/526 | 96.39 | [94.43 ; 97.68] | 95.68 |
| SIREN | 511/526 | 97.15 | [95.35 ; 98.26] | 96.54 |
| Opérationnel existant | 508/526 | 96.58 | [94.66 ; 97.82] | 95.84 |
| Site opérationnel | 404/526 | 76.81 | [73.01 ; 80.21] | 81.30 |
| Site seuil plat 100 m | 325/526 | 61.79 | [57.56 ; 65.84] | 66.87 |
| Site seuil plat 200 m | 373/526 | 70.91 | [66.89 ; 74.63] | 75.44 |
| Site seuil plat 500 m | 413/526 | 78.52 | [74.81 ; 81.81] | 83.06 |

Pondération et trois vues historiques reproduites (507/511/508 et chaque booléen).
Wilson seulement sur le brut, pas un IC pondéré.

## Folds

| Fold | Vue | Succès | % brut | Wilson 95 % brut | % pondéré Hájek |
|---|---|---:|---:|---:|---:|
| 2 | SIRET exact | 172/178 | 96.63 | [92.84 ; 98.45] | 95.64 |
| 2 | SIREN | 176/178 | 98.88 | [96.00 ; 99.69] | 98.39 |
| 2 | Opérationnel existant | 173/178 | 97.19 | [93.59 ; 98.79] | 96.17 |
| 2 | Site opérationnel | 145/178 | 81.46 | [75.11 ; 86.48] | 85.24 |
| 2 | Site seuil plat 100 m | 118/178 | 66.29 | [59.07 ; 72.83] | 69.91 |
| 2 | Site seuil plat 200 m | 135/178 | 75.84 | [69.05 ; 81.54] | 79.92 |
| 2 | Site seuil plat 500 m | 149/178 | 83.71 | [77.58 ; 88.41] | 88.83 |
| 3 | SIRET exact | 153/162 | 94.44 | [89.78 ; 97.05] | 95.11 |
| 3 | SIREN | 153/162 | 94.44 | [89.78 ; 97.05] | 95.11 |
| 3 | Opérationnel existant | 153/162 | 94.44 | [89.78 ; 97.05] | 95.11 |
| 3 | Site opérationnel | 120/162 | 74.07 | [66.82 ; 80.21] | 81.00 |
| 3 | Site seuil plat 100 m | 98/162 | 60.49 | [52.81 ; 67.70] | 70.26 |
| 3 | Site seuil plat 200 m | 111/162 | 68.52 | [61.01 ; 75.17] | 76.74 |
| 3 | Site seuil plat 500 m | 121/162 | 74.69 | [67.48 ; 80.76] | 81.60 |
| 4 | SIRET exact | 182/186 | 97.85 | [94.60 ; 99.16] | 96.22 |
| 4 | SIREN | 182/186 | 97.85 | [94.60 ; 99.16] | 96.22 |
| 4 | Opérationnel existant | 182/186 | 97.85 | [94.60 ; 99.16] | 96.22 |
| 4 | Site opérationnel | 139/186 | 74.73 | [68.03 ; 80.43] | 78.20 |
| 4 | Site seuil plat 100 m | 109/186 | 58.60 | [51.42 ; 65.44] | 61.28 |
| 4 | Site seuil plat 200 m | 127/186 | 68.28 | [61.28 ; 74.54] | 70.47 |
| 4 | Site seuil plat 500 m | 143/186 | 76.88 | [70.32 ; 82.36] | 79.41 |

## Compteurs et limites

- Ratés exacts devenant succès site : **4/19**.
- Succès exacts dont le point label est loin du CRM (>T) : **86/507** ; 440 comparables, 67 non mesurables.
- Succès exacts sans succès site : **107/507** (absence de preuve ≠ site incorrect).
- Preuves principales : `{"adresse": 241, "distance": 163, "aucune": 122}`.
- Coordonnées candidat : étude statistique SIRENE juillet ; adresses courantes août ; CRM BAN v17.
- 11/21 = numéro identifié, pas précision actuelle certifiée ; pas de transfert de qualité vers XY v142.
- Absences/qualités inconnues conservées, règle principale adresse seule si distance inadmissible.
- Sensibilités plates ignorent les niveaux de précision, y compris commune, mais exigent un XY compatible.
- Sources grandes : SHA hérités, taille/mtime/inode vérifiés avant/après, pas de rehash complet.
- RSS gardée maximale : 308.0 Mio ; durée : 6.59 s.

Commande : `.venv/bin/python scripts/retrieval_site_operational_metric.py`.
JSON voisin : totaux, folds, sources et **526 query_id uniques**, preuves sans noms/adresses CRM bruts.
