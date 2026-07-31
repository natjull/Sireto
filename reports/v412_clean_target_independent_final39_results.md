# V4.12 — résultats du dernier lot aveugle

Le candidat est resté figé : ranker corrigé, accepteur XGBoost monotone poids
`10`, seuil `0.9940522313117981`. Les labels ont été scellés dans `5c77a36`
avant ouverture des scores.

| Mesure dernier lot | Résultat |
|---|---:|
| `MATCH_EXACT` / `AMBIGUOUS` | **33 / 6** |
| Bons top 1 du ranker | **28/33 (84,85 %)** |
| `AUTO_MATCH` | **13/39 (33,33 %)** |
| AUTO corrects | **13/13** |
| Erreurs ou ambiguïtés AUTO | **0** |

Les cinq erreurs du ranker et les six ambiguïtés restent toutes en REVIEW.

## Bilan des 99 dossiers vraiment aveugles

| Mesure | Résultat |
|---|---:|
| `MATCH_EXACT` / `AMBIGUOUS` | **89 / 10** |
| Bons top 1 du ranker | **75/89 (84,27 %)** |
| `AUTO_MATCH` | **32/99 (32,32 %)** |
| AUTO corrects | **32/32** |
| Erreurs ou ambiguïtés AUTO | **0** |

La précision observée est 100 %, mais 32 AUTO sans erreur ne certifient pas
99,8 %. La couverture sur ce stock particulièrement difficile est 32,32 %.
Le test final reste fermé.

Verdict : **`GO_BUSINESS_AUDIT_COMPLETE`** pour terminer l'analyse des 279
REVIEW avant toute décision de réentraînement.
