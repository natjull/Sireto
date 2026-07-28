# V4.6 — ablation du ranker aligné sur le retrieval B

Date d'évaluation : 28 juillet 2026

Verdict : **`KEEP_RANKER_A`**

## Résultat

Le ranker V4.1 historique A et un ranker B réentraîné sur les pools V4.2-B
ont été comparés sur exactement les mêmes 1 217 requêtes `MATCH_EXACT` du dev
historique. Les vérités absentes du pool auraient compté comme erreurs
end-to-end.

| Mesure dev exacte | Ranker A sur pools B | Ranker B aligné |
|---|---:|---:|
| Succès SIRET top-1 | 1 213 / 1 217 | 1 216 / 1 217 |
| Hit@1 SIRET | 99,671 % | 99,918 % |
| Succès SIREN top-1 | 1 213 / 1 217 | 1 216 / 1 217 |
| Hit@1 SIREN | 99,671 % | 99,918 % |
| Latence p95 du modèle seul | 0,609 ms/requête | 0,618 ms/requête |

La comparaison appariée contient trois corrections et aucune régression. Le
gain observé est de 0,247 point. Il reste inférieur au minimum
préenregistré de quatre corrections nettes et n'est pas statistiquement assez
étayé sur ce dev réutilisé :

- borne basse bootstrap appariée à 95 % : `0` ;
- test exact de McNemar bilatéral : `p = 0,25`.

Les gates de sécurité, segments, latence, déterminisme et intégrité passent.
Les trois gates de preuve du gain échouent. Le verdict mécanique est donc
`KEEP_RANKER_A`, sans abaissement rétroactif des seuils.

## Dataset V4.2-B

Deux builds réellement indépendants ont été exécutés avec des caches et des
répertoires séparés. Ils publient des manifests distincts mais le même hash de
contenu candidat :

`2b92f0ff548d0a186b134db6c3886f96f31a38f791d28bcd68413f11cdb2f731`

| Contrôle | Résultat |
|---|---:|
| Requêtes | 7 003 |
| Candidats | 698 991 |
| Maximum par requête | 100 |
| Doublons SIRET intra-pool | 0 |
| Candidats fermés | 0 |
| Requêtes sans candidat | 0 |
| Positifs injectés | 0 |
| Recall@100 fit | 4 666 / 4 666 = 100 % |
| Recall@100 dev | 1 217 / 1 217 = 100 % |

Le premier passage complet avait échoué après le retrieval, lors de la
création du manifeste : un fichier de métadonnées Parquet vide situé sous
`partitions/manifest/` était compté comme une partition physique. Le correctif
`458dd97` exclut uniquement les métadonnées de ce compteur. Il ne change ni
le retrieval, ni les candidats, ni les features. Les deux builds contractuels
ont ensuite été relancés depuis le début et validés.

## Contrôles modèle

- Le ranker A est chargé avec ses hashes et sa signature d'entraînement A.
- Le ranker B utilise les 64 mêmes features, les mêmes hyperparamètres et les
  cinq folds SIREN gelés.
- Aucun positif manquant n'a été réinjecté ou exclu : le retrieval B retrouve
  toutes les vérités exactes fit/dev.
- Chaque requête fit a une prédiction hors fold et chaque requête dev une
  prédiction hors échantillon.
- Deux entraînements complets de B produisent les mêmes top-1 et des scores
  strictement identiques (`delta max = 0`).
- Le tie-break est `score décroissant`, puis `rang retrieval`, puis `SIRET`.
- Aucun label V4.4, random, holdout, accepteur ou seuil AUTO n'a été lu.
- Le test final reste fermé.
- Le temps contractuel cumulé des deux builds et de l'évaluation est de
  9 012 secondes, sous la limite de huit heures.

Le dev a déjà servi historiquement à sélectionner le ranker A. Les résultats
appariés mesurent donc une ablation de développement, pas une validation
indépendante ni une preuve de précision production.

## Artefacts immuables

- Dataset primaire :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_a/301b24f47820f992`
- Réplique indépendante :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_b/301b24f47820f992`
- Modèle B, prédictions, cas appariés et verdict :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_6_aligned_ranker/421f2cd0cc436af7`

## Décision

Le ranker B n'est pas promu. Le pipeline de référence reste le ranker A gelé
appliqué aux pools V4.2-B, c'est-à-dire exactement la pile qui a produit les
37 `SCENE_DRIFT` de V4.5.

La prochaine expérience propre n'est donc pas un nouvel entraînement. Elle
doit adjudiquer les top-1 actuels de ces 37 scènes, puis reconstruire un
corpus dont chaque label juge réellement la prédiction de la pile conservée.
