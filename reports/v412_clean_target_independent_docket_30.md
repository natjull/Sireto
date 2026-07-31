# V4.12 — validation aveugle de la cible accepteur nettoyée

Ce lot contient 30 dossiers tirés des 99 REVIEW V4.12-G encore jamais
adjudiqués après le premier contrôle indépendant. Il est gelé avant tout score
du candidat clean-target.

Après exclusion des 180 dossiers déjà adjudiqués, la sélection ordonne les
REVIEW restants par :

```text
SHA256(UTF8("SIRETO-V412-CLEAN-TARGET-INDEPENDENT-30\0" + query_id))
```

Le SHA-256 de la liste JSON ordonnée des `query_id` est
`a87bc34c0eea71677a762b2056e96ff8404049ef94041e447f441a4586216eb1`.

Le candidat est figé : XGBoost monotone, poids `10`, seuil
`0.9940522313117981`. Les résultats de ce lot ne peuvent modifier aucun de ces
choix. Le test final reste fermé.
