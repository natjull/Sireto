# V4.12 — docket indépendant de 30 REVIEW après correction des labels

Ce lot contient 30 dossiers tirés des 129 REVIEW V4.12-G encore jamais
adjudiqués. Il est gelé avant lecture des scores ou décisions du ranker et de
l'accepteur réentraînés.

La sélection exclut les 150 dossiers déjà adjudiqués, puis ordonne les REVIEW
restants par :

```text
SHA256(UTF8("SIRETO-V412-INDEPENDENT-30\0" + query_id))
```

Les 30 premiers sont inscrits dans
[`v412_corrected_acceptor_independent_docket_30.csv`](v412_corrected_acceptor_independent_docket_30.csv).
Le SHA-256 de la liste JSON ordonnée des `query_id` est
`5e286f63c8ce0da1c523a7bd21e3e2929bc862fe3d1eb2dfad67d7abb4f9db29`.

Le lot sert uniquement à mesurer le candidat déjà choisi. Ses résultats ne
peuvent pas modifier le modèle, le poids `10` ou le seuil
`0.8974587321281433`. Le test final reste fermé.
