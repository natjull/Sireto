# V4.12 — vérification du bundle trusted-label

Le bundle `c2a01c6bca43a468` a été rechargé depuis ses artefacts copiés, puis
exécuté sur l'intégralité des 1 127 contrôles dev non utilisés pour les labels
trusted.

| Contrôle | Résultat |
|---|---:|
| Requêtes reconstruites | 1 127 |
| Candidats réellement scorés | 112 389 |
| Pools au-dessus de 100 | 0 |
| Doublons SIRET dans un pool | 0 |
| Top 1 ranker identiques | 1 127/1 127 |
| Features accepteur identiques | 80/80, bit à bit |
| Scores accepteur identiques | 1 127/1 127, bit à bit |
| Décisions identiques | 1 077 AUTO, 50 REVIEW |

Les hashes du bundle, les ordres de 45 features ranker et 80 features
accepteur, la taxonomie et le contrat retrieval V4.2 ont été revérifiés.
Le test final reste fermé.

Verdict : **`PASS_BUNDLE_TRAIN_SERVE_PARITY`**.

Le résultat machine est dans
[`v412_trusted_bundle_verification.json`](v412_trusted_bundle_verification.json).
