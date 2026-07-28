# V4.11 — Ranker C input-blind

## Verdict

**`GO_RANKER_C`**

Le ranker C atteint le gate préenregistré sur le dev historique, sans fuite de
cible, avec cinq modèles OOF pour le fit et un modèle complet pour le dev.
L'artefact peut être consommé par le builder de scènes de l'accepteur.

Artefact promouvable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/e13eb3ac7498256e`

Le premier artefact `0d2e419158c7a4c0` est conservé mais superseded : ses
scores étaient corrects, son champ `retrieval_miss_count` désignait à tort
les seuls pools vides et sa description du canal sparse ne correspondait pas
à la matrice réellement scorée. Le commit `d2a6f5b` a corrigé ces deux
incohérences avant promotion. Modèles et prédictions sont bit à bit identiques
entre les deux runs.

## Performance end-to-end

| Split | Hit@1 SIRET | Hit@1 SIREN | Vérité absente du pool | Pool vide |
|---|---:|---:|---:|---:|
| Fit OOF | 4 661 / 4 666 = **99,8928 %** | 4 662 / 4 666 = 99,9143 % | 1 | 0 |
| Dev historique | 1 216 / 1 217 = **99,9178 %** | 1 216 / 1 217 = 99,9178 % | 0 | 0 |

Le gate exigeait au moins 99,8 % de Hit@1 SIRET sur le dev et aucune
régression supérieure à deux points sur une grande famille face au ranker B
masqué. Les deux conditions sont franchies.

Les cinq erreurs OOF fit concernent `13369`, `1924`, `6818`, `11731` et
`5953`. `6818` est l'unique vérité absente du pool et reste donc une erreur
end-to-end. L'unique erreur dev concerne `13958`.

## Baseline masquée

Le ranker B historique, évalué en diagnostic avec ses cinq signaux
d'identifiant forcés à zéro, atteint 1 217/1 217 sur le même dev. Il conserve
donc un avantage d'un cas. Cette baseline n'est pas promouvable dans V4.11 :
elle n'a pas été entraînée sur les pools input-blind alignés et ne fournit pas
les prédictions OOF requises pour entraîner honnêtement l'accepteur.

La projection diagnostique est désormais déclarée exactement comme exécutée :

- `admission_rank_recip = 1 / retrieval_rank` ;
- `admission_current_sparse_rank_recip = 1 / retrieval_rank` ;
- `admission_fusion_score = 1 / (60 + retrieval_rank)` ;
- `admission_channel_count = 1` pour l'unique canal sparse actif ;
- cinq signaux d'identifiant à zéro.

## Entraînement et intégrité

- 4 665 requêtes exactes fit ont un positif dans leur pool et entraînent le
  modèle complet ; `6818` est exclu du fit faute de positif ;
- dans chaque pli, `6818` est exclu lorsqu'il appartient au train et reste
  scoré comme erreur lorsqu'il appartient au pli OOF ;
- les 698 892 candidats reçoivent exactement une prédiction ;
- fit : origine unique `ranker_c_oof`, avec numéro de pli identique à
  l'assignation gelée ;
- dev : origine unique `ranker_c_dev`, issue du modèle complet fit ;
- rangs contigus, mêmes populations que le dataset retrieval ;
- deux répétitions : modèles, scores et rangs bit à bit identiques ;
- tous les hashes de sortie et le build ID ont été recomputés ;
- l'audit indépendant conclut `GO` après correction des deux défauts de
  reporting.

Hash du modèle complet :

`f4b71b49ed4f879b88e05e4fb84229d0306c5e8ca96958ac20ad97fcc04349c0`

Hash des prédictions OOF/dev :

`f14828aafa146dc4ad0399697c9477e57930ba618a5b2d7d0a903e52c2d879c0`

## Limites

- Le dev historique est consommé et ne constitue pas une preuve finale.
- Le `GO_RANKER_C` autorise seulement la construction et le développement de
  l'accepteur compact.
- Les 225 lignes inédites, le test final historique et le holdout V4-Fresh
  restent fermés.
- La North Star produit reste la couverture `AUTO_MATCH` sous précision SIRET
  exacte ≥99,8 %, pas le seul Hit@1 du ranker.

