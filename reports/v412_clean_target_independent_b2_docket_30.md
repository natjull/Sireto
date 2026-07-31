# V4.12 — deuxième validation aveugle de l'accepteur clean-target

Ce lot contient 30 dossiers tirés des 69 REVIEW V4.12-G encore jamais
adjudiqués. Il est gelé avant toute consultation des scores du candidat
clean-target.

Après exclusion des 210 dossiers déjà adjudiqués, la sélection ordonne les
REVIEW restants par :

```text
SHA256(UTF8("SIRETO-V412-CLEAN-TARGET-INDEPENDENT-B2-30\0" + query_id))
```

Le SHA-256 de la liste JSON ordonnée des `query_id` est
`8802a3236b284e48926d9f41367bc7b54b406390db01d705aa99ba6b3ceb0c37`.

Le candidat reste figé : XGBoost monotone, poids `10`, seuil
`0.9940522313117981`. Ce lot ne peut modifier ni le modèle ni le seuil. Les
labels doivent décrire le SIRET actif courant avec preuves consultables. Le test
final reste fermé.
