# V4.12 — bilan métier complet des 279 REVIEW

Les **279 dossiers REVIEW du périmètre de développement ont tous été
adjudiqués** avec une preuve traçable. Aucun dossier n'a été renvoyé à
l'utilisateur.

## Vérité métier

| Cohorte | Dossiers | SIRET exact identifiable | Ambigu | Non résolu |
|---|---:|---:|---:|---:|
| Premier audit | 30 | 27 | 3 | 0 |
| Contre-audit ranker | 53 | 50 | 3 | 0 |
| Contrôle ranker indépendant | 7 | 6 | 1 | 0 |
| Overlay corrigé | 60 | 56 | 4 | 0 |
| Premier lot accepteur | 30 | 26 | 4 | 0 |
| Lot aveugle B1 | 30 | 28 | 2 | 0 |
| Lot aveugle B2 | 30 | 28 | 2 | 0 |
| Dernier lot aveugle | 39 | 33 | 6 | 0 |
| **Total** | **279** | **254 (91,04 %)** | **25 (8,96 %)** | **0** |

Le stock apporte donc **254 labels SIRET exacts utilisables** et **25 labels
d'abstention fiables**. La fiabilité est `HIGH` pour 257 dossiers et `MEDIUM`
pour 22.

## Ce que l'audit démontre

Le principal problème historique n'était pas l'impossibilité métier de
matcher : **218 des 254 dossiers aujourd'hui identifiables avaient été marqués
`AMBIGUOUS` par la construction mécanique V4**. Seulement 36 avaient un ancien
label `MATCH_EXACT`, et seulement 24/254 anciennes vérités SIRET concordaient
avec l'adjudication actuelle.

Les causes concrètes se regroupent en quatre familles :

1. plusieurs sites du même SIREN, avec choix du mauvais établissement ;
2. plusieurs personnes morales ou holdings co-localisées ;
3. transferts, fusions et successeurs juridiques rendant le CRM obsolète ;
4. noms de marque, alias publics ou noms abrégés différents de la raison
   sociale.

Les 25 ambiguïtés restantes sont majoritairement des groupes ou marques dont
plusieurs personnes morales actives partagent exactement la même adresse. Elles
doivent rester en REVIEW tant que le CRM ne fournit pas une information
supplémentaire.

## Mesure honnête du candidat figé

Les 99 derniers dossiers ont été tirés et labellisés avant ouverture de leurs
scores. Ils constituent la seule mesure indépendante du candidat clean-target :

| Mesure sur 99 cas aveugles | Résultat |
|---|---:|
| SIRET identifiables / ambigus | **89 / 10** |
| Bons top 1 du ranker | **75/89 (84,27 %)** |
| `AUTO_MATCH` | **32/99 (32,32 %)** |
| AUTO corrects | **32/32** |
| Erreurs ou ambiguïtés AUTO | **0** |

Le candidat est sûr sur cet échantillon difficile, mais trop conservateur et
le ranker commet encore 14 erreurs sur 89 cas identifiables. Avec seulement 32
AUTO, la précision de 99,8 % n'est **pas certifiée**.

## Données ligne par ligne

- [30 premiers labels](v412_review_adjudication_labels.csv)
- [53 contre-audits](v412_review_rerank_counteraudit_53.csv)
- [7 validations ranker](v412_ranker_independent_validation_labels.csv)
- [60 labels corrigés](v412_corrected_review_overlay_60.csv)
- [30 premiers labels accepteur](v412_corrected_acceptor_independent_labels_30.csv)
- [30 labels aveugles B1](v412_clean_target_independent_labels_30.csv)
- [30 labels aveugles B2](v412_clean_target_independent_b2_labels_30.csv)
- [39 derniers labels aveugles](v412_clean_target_independent_final39_labels.csv)

## Conclusion

Verdict : **`GO_RETRAIN_ON_TRUSTED_LABELS`**.

Le prochain travail utile est un réentraînement OOF du ranker et de l'accepteur
sur ces labels corrigés, sans changer le retrieval et sans ouvrir le test final.
Le gain doit venir d'une meilleure vérité d'apprentissage et d'une meilleure
gestion des cas multisites/co-localisés, pas d'une nouvelle architecture.
