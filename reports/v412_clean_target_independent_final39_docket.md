# V4.12 — dernier lot aveugle des REVIEW historiques

Ce lot contient les **39 derniers dossiers** parmi les 279 REVIEW V4.12-G. Les
240 autres ont déjà été adjudiqués. Il est gelé avant toute consultation des
scores du candidat clean-target.

Les 39 dossiers restants sont tous conservés. Leur ordre est déterminé par :

```text
SHA256(UTF8("SIRETO-V412-CLEAN-TARGET-INDEPENDENT-FINAL39\0" + query_id))
```

Le SHA-256 de la liste JSON ordonnée des `query_id` est
`ab80c216317736920a501a9b9fe798acfed7f9a072eae28ea14ab509453648fd`.

Le candidat reste figé : XGBoost monotone, poids `10`, seuil
`0.9940522313117981`. Les labels doivent décrire le SIRET actif courant avec
preuves publiques ou conclure `AMBIGUOUS` / `UNRESOLVED`. Aucun réentraînement
et aucune ouverture du test final.
