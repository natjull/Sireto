# V4.12 — pivot de l'accepteur sur les scènes trusted-label

Artefact local :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_trusted_acceptor/7bde8fd021ec1915`.

Le ranker poids `0,5`, l'accepteur XGBoost monotone poids `10` et les 80
features V4.11 ont été figés. Les 279 scènes sont produites par ranker OOF :
216 top 1 corrects et 63 cibles négatives, dont 25 ambiguïtés et 38 erreurs du
ranker/retrieval.

| Partition | AUTO | Corrects | Erreurs | Ambiguïtés AUTO |
|---|---:|---:|---:|---:|
| Calibration (147) | 71 | 71 | 0 | 0 |
| Comparaison (132) | 52 | 50 | 2 | 2 |
| Cumul OOF (279) | 123 | 121 | 2 | 2 |

Le seuil `0.9712138175964355`, choisi sans erreur sur la calibration, ne se
généralise pas à la comparaison : précision 96,15 %, très inférieure à
99,8 %.

Les deux erreurs sont des ambiguïtés métier réelles :

- `5052`, Promotrans Mondeville : plusieurs entités Promotrans co-localisées ;
- `9406`, Ligue Auvergne-Rhône-Alpes de Handball : ligue et comité
  départemental co-localisés.

Une garde manuelle fondée sur les deux premiers candidats bloque ces erreurs,
mais supprime entre 55 et 99 AUTO corrects selon son seuil. Une pondération
accrue des cibles négatives laisse encore une ou deux erreurs sur la partition
de comparaison. Ces pistes sont rejetées.

Verdict : **`PIVOT_ACCEPTOR`**. Aucun modèle n'est promu et le test final reste
fermé. La prochaine expérience autorisée est une calibration conservatrice sur
tous les scores OOF de développement consommés, sans nouveau tuning de famille,
features ou poids.
