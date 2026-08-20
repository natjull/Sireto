# SIRENE temporel — build et valorisation

Cette couche corrige l'hypothèse erronée selon laquelle
`StockEtablissementHistorique` contiendrait des adresses. Elle ne modifie ni le
dossier SIREN v3 ni l'index courant : les historiques et successions forment un
overlay de rescue supprimable indépendamment.

## Sémantique

- Les anciens noms d'établissement restent au grain SIRET.
- Les anciens noms d'unité légale restent au grain `(INSEE,SIREN)` puis sont
  développés vers les sites locaux.
- L'adresse courante jointe à un ancien nom est seulement un `geo_scaffold` :
  elle est stockée pour comparer/classer le site mais n'est pas indexée comme
  adresse historique.
- Les liens prédécesseur/successeur sont dirigés, structurés et parcourus sur
  un seul saut. Une succession cross-SIREN n'est jamais une équivalence
  opérationnelle automatique.

## Exécution

1. Lancer `build_sirene_temporal_projection.py preflight` sur un préfixe réel.
   Le processus échoue si un champ requis manque, si une famille vaut zéro ou
   si plus de 5 % des preuves nominales ne rejoignent pas le stock courant.
2. Lancer `build-all` vers un nouveau répertoire du volume. Les 100 préfixes
   sont content-addressés et reprenables ; un préfixe publié n'est pas recalculé.
3. Construire l'overlay avec `build_sirene_temporal_index.py
   --projection-root ...`. Le manifeste refuse zéro document SIRET/SIREN et
   atteste `historical_addresses_indexed=false`.
4. Ajouter `--temporal-overlay-index` à
   `retrieve_official_evidence_union.py` avec la configuration unique
   `config/retrieval_ltr_admission_temporal_v3.json`.
5. Mesurer l'oracle fold0. Aucun LambdaMART n'est ajusté si l'oracle exact est
   inférieur à 99,3 %. En cas de passage, un fit folds 2/3/4 et une seule
   évaluation dev0 produisent le verdict ; fold1 reste fermé.

## Consommateurs aval

`project_sirene_temporal_features.py` produit un sidecar au grain
`(query_id,candidate_siret)` à joindre aux features dossier existantes. Les
features autorisées par ranker, decider, risk et fusion sont gelées dans
`config/siren_dossier_consumers_temporal_v4.json`. Le texte neural est sourcé et
typé ; les relations, dates et conflits restent structurés.

Les comptes annuels, personnes physiques, textes libres BODACC/RNE, PDF,
aliases CRM et données synthétiques ne sont pas admis par cette couche.
