# V4.12 — résultats aveugles clean-target B2

Le candidat est strictement celui déjà figé : ranker corrigé, accepteur
XGBoost monotone poids `10`, seuil `0.9940522313117981`. Les 30 labels ont été
scellés dans le commit `a653027` avant ouverture des scores.

| Mesure B2 | Résultat |
|---|---:|
| `MATCH_EXACT` / `AMBIGUOUS` | **28 / 2** |
| Bons top 1 du ranker | **23/28 (82,14 %)** |
| `AUTO_MATCH` | **6/30 (20,00 %)** |
| AUTO corrects | **6/6** |
| Erreurs ou ambiguïtés AUTO | **0** |

Les cinq erreurs du ranker concernent ESPI Levallois, Promotrans
Villeneuve-d'Ascq, Somudimec Lyon, OXYA Loos et Boulanger Villeneuve-d'Ascq.
Elles restent toutes en `REVIEW`, comme les deux dossiers ambigus.

## Cumul aveugle B1 + B2

| Mesure cumulée | Résultat |
|---|---:|
| Dossiers | **60** |
| `MATCH_EXACT` / `AMBIGUOUS` | **56 / 4** |
| Bons top 1 du ranker | **47/56 (83,93 %)** |
| `AUTO_MATCH` | **19/60 (31,67 %)** |
| AUTO corrects | **19/19** |
| Erreurs ou ambiguïtés AUTO | **0** |

Le signal de sécurité est positif, mais 19 AUTO sans erreur ne permettent
pas de certifier 99,8 %. Le candidat et son seuil restent figés pendant l'audit
des 39 REVIEW encore vierges. Aucun réentraînement et aucune ouverture du test
final.

Verdict intermédiaire : **`GO_COMPLETE_REMAINING_39`**.
