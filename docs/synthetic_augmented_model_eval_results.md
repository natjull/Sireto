# Résultats réel seul vs réel + synthétique

Date d'exécution : 16–17 août 2026. Ressource : Mac M4 Pro et
`/Volumes/CATNAT_DATA` uniquement. Dépense externe : 0 €.

## Verdict

Les deux augmentations s'arrêtent sur le fold de développement 0 :

- `STOP_SYNTHETIC_AUGMENTATION_XGB` ;
- `STOP_SYNTHETIC_AUGMENTATION_BGE`.

Le fold 1 de confirmation et le test final n'ont pas été ouverts. Aucun modèle
de risque, aucune calibration et aucun seuil AUTO n'ont été entraînés ou
sélectionnés. Le résultat ne remet pas en cause la valeur du corpus comme
réservoir d'expériences futures ; il refuse seulement les deux recettes
préenregistrées `XGBoost + 0.5/k` et `BGE groupwise + 0.5/k`.

## Corpus et protocole gelés

Le corpus final contient exactement 20 000 surfaces CRM distinctes associées
à 9 737 SIRET, toutes exactes dans le snapshot SIRENE. Son fichier final
`promoted_20000.jsonl` porte le SHA-256
`9e871f0f3c5a19d28a59619c4fd09c87be5d1e75e54296ff41bf34a4dd5cbcc1`.
Les 20 000 lignes sont disjointes des SIREN du GT réel. L'audit déterministe
final est complet ; la revue humaine bornée de réalisme sur 200 lignes reste
`PENDING_BOUNDED_REVIEW`. Les 100 audits humains antérieurs n'avaient identifié
aucun faux réalisme certain.

Le snapshot modèle immuable `6281c9d1470f3913` assigne 6 656 scènes au fold 2,
6 624 au fold 3 et 6 720 au fold 4. Les replays retrieval sont naturels :
aucun positif n'est injecté, 100 candidats au maximum, et ni score ni hit
modèle n'a servi à qualifier les exemples. L'admission gelée retrouve
naturellement la vérité dans 8 430 scènes sur 20 000, soit 42,15 %. Le bundle
`9f99de01516dde9a` contient 857 439 lignes candidat et 8 430 groupes BGE
éligibles. Les 11 570 autres scènes restent des erreurs end-to-end et ne sont
pas transformées artificiellement en exemples d'entraînement.

Le mix `71ceda354734fb7a` contient 8 192 scènes réelles et 4 096 scènes
synthétiques, soit le ratio 2:1 préenregistré. Les folds d'apprentissage sont
uniquement 2/3/4. Chaque identité synthétique pèse 0,5 au total, réparti en
`0.5/k` sur ses variantes ; les scènes réelles conservent leur poids
historique. Le synthétique est interdit au risk model, à la calibration, aux
seuils AUTO, au fold 1 et au test.

## XGBoost

Les deux bras utilisent les mêmes 129 features BUSINESS, les mêmes
hyperparamètres et le même fold 0 réel de 2 797 scènes exactes.

| Segment | Réel seul exact | Réel + synthétique exact | Écart |
|---|---:|---:|---:|
| Tous | 2 435/2 797 (87,058 %) | 2 430/2 797 (86,879 %) | -5 |
| Difficile | 32/38 (84,211 %) | 32/38 (84,211 %) | 0 |
| Actif | 2 184/2 391 (91,343 %) | 2 179/2 391 (91,133 %) | -5 |
| Fermé | 251/406 (61,823 %) | 251/406 (61,823 %) | 0 |

La vue opérationnelle même SIREN/même site passe de 2 451 à 2 446 bonnes
prédictions, soit également -5. La matrice appariée compte 14 corrections
propres au bras synthétique contre 19 régressions propres à ce bras. Les gates
exact global, difficile et gain minimal ne passent pas. Artefact :
`synthetic_augmented_xgb_v1/bf439dfbad1584c5`.

Durée : 17,45 s pour le fit réel seul, 27,02 s pour le fit augmenté et 55 s
muraux pour le comparateur complet.

## BGE groupwise

Le modèle est `BAAI/bge-reranker-v2-m3`, révision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, avec les quatre dernières couches
entraînables, une époque, loss groupwise pondérée, 12 288 scènes et 196 608
paires. Le contrôle publié respecte lui aussi les folds train 2/3/4 et le
target fold 0.

| Segment | Réel seul exact | Réel + synthétique exact | Écart |
|---|---:|---:|---:|
| Tous | 2 400/2 797 (85,806 %) | 2 403/2 797 (85,913 %) | +3 |
| Difficile | 32/38 (84,211 %) | 32/38 (84,211 %) | 0 |
| Actif | 2 159/2 391 (90,297 %) | 2 166/2 391 (90,590 %) | +7 |
| Fermé | 241/406 (59,360 %) | 237/406 (58,374 %) | -4 |

Dans la vue opérationnelle, le total reste strictement inchangé à
2 418/2 797 (86,450 %). L'actif gagne 5 cas, mais le fermé en perd 5. La
matrice appariée exacte compte 56 corrections propres au bras synthétique et
53 régressions. Le gain global de +3 reste sous le gate préenregistré de +10 ;
les gates exact global, difficile et fermé échouent également. Artefacts :
`synthetic_augmented_bge_v1/47ac65d7f3f4fbf0` et comparateur
`synthetic_augmented_bge_comparison_v1/b01dc7d33958f72f`.

Durée : 15 338,13 s de fit (4 h 15 min 38 s), 10 554,01 s de scoring
(2 h 55 min 54 s), soit 7 h 11 min 32 s hors comparateur final de 4,5 s.
Pic RSS : 3 573 743 616 octets, environ 3,33 Gio. L'inférence a porté sur
279 511 candidats pour 2 797 requêtes exactes.

## Coûts de préparation

- snapshot source : 3 s ;
- replay V7 : 1 013 s ;
- replay overlay fermé : 631 s ;
- bundle features/textes/groupes : 753 s ;
- mix : 4 s ;
- XGBoost complet : 55 s ;
- BGE fit + scoring + comparaison : environ 7 h 12 min.

Les artefacts occupent environ 4,6 Go sur le SSD externe. Aucun GPU loué,
appel Maps ou autre dépense externe n'a été utilisé.

## Décision technique

Ne pas publier les modèles augmentés et ne pas ouvrir le fold 1. La recette
XGBoost régresse sans ambiguïté. BGE produit un petit transfert favorable vers
les actifs, mais le gain global exact est insuffisant, nul en vue
opérationnelle et payé par une régression des fermés. Toute suite doit être une
nouvelle hypothèse préenregistrée, par exemple un sampler ou une pondération
spécifique aux fermés, et non une sélection post-hoc sur ce fold 0.
