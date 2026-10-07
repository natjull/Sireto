# EmbeddingGemma2 : premier gain mesuré, pas de validation nationale

7 octobre 2026. Protocole et code figés dans `b377cfa7`,
`docs/embeddinggemma2_comparison_contract.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/comparison_v1/run_20261007_a`.

## Training terminé et informations réellement utilisées

Job cached COMPLETE exit0, deux epochs/7 518 pas. Meilleur adapter choisi par
les64 contrôles contrastifs validation au **pas7 200**, loss.187265 ; dernier
checkpoint.227860, donc non choisi. Baseline contrôle64.674642. Aucun test ouvert.

15 035 paires CRM → document officiel du SIRET humain exact. Textes : noms des
champs natifs legal/trade/site/historical/supporting, adresses courantes/
historiques/supporting, CP. Les noms de l'unité légale présents dans le document
sont inclus ; PAS une nouvelle collecte exhaustive de tous les noms en amont.
Identifiants SIRET/SIREN utilisés seulement pour supervision, joints, split et
masquage, jamais comme signal textuel. SIREN disjoints entre TRAIN et validation.

PAS de XY, distance, coordonnées/qualité BAN, contexte des autres établissements
du SIREN, relations de succession/BODACC dans ce fine-tuning. Pas d'apprentissage
explicite du choix du site parmi ses siblings : les candidats de même SIREN
sont exclus/masqués des négatifs pour éviter des faux négatifs. Ce modèle n'est
donc pas encore un retrieval complet entreprise+sites+géographie.

Concaténation puis troncature160tokens. Audit du tokenizer réel :45/15 035
positifs TRAIN tronqués (0,30%),10/2 212 validation (0,45%), médiane42tokens,
p95 TRAIN74/validation78. La troncature est minoritaire ; ne pas en déduire
que toutes les preuves sources étaient présentes dans les champs natifs.

## Diagnostic sur la validation entière, PAS du recall

2 212 lignes :1 887 avec au moins un négatif local BM25,325 sans négatif
explicitement publiées, aucun filtrage du dénominateur d'une mesure retrieval.
Bon document mieux scoré que tous les négatifs disponibles (au plus2/query) :
brut1 435/1 887=76,05%, tuned1 684/1 887=89,24%, soit+249 succès nets.
Loss contrastive moyenne.687264→.372772. Contrôle développement sur groupes
SIREN non vus au fit ;64 de ces lignes ont choisi le checkpoint, pas holdout
indépendant, et ce n'est pas une recherche face au corpus complet.

`contrastive_report.json` sha256
`8d44d7fb3a65c1317221570136cfaf69803b4ad75141fa31b06e3974f126f8f1`.

## Recherche initiale100 face aux communes complètes

30 requêtes validation, communes71153/77118/38087/47031/54397/16374 sélectionnées
par hash CRM/comptages, minimum4 queries et1 000–12 000 établissements indexés.
33 209 établissements officiels, actifs/fermés, PAS les seules cibles annotées.
30 391 textes documents uniques encodés pour chaque variante ;9 tronqués.
Corpus intégral de chaque commune énuméré avant générations, aucun GT injecté,
ni pool lexical fixe2000/500, ni reranker : recherche dense exacte top100.
Tous les GT du panel présents dans leur partition (zéro exclusion).

| Mesure sur les mêmes30 requêtes | Brut | Fine-tuné |
| --- | ---: | ---: |
| SIRET exact@1 |14/30|18/30|
| SIRET exact@10 |18/30|27/30|
| SIRET exact@100 |20/30 (66,67%)|28/30 (93,33%)|
| SIREN@100 |23/30|28/30|
| Opérationnel officiel@100, séparé |20/30|28/30|

Huit gains exacts appariés, zéro régression sur CE panel. Wilson95 exact tuned
[78,68%;98,15%] : impossible d'en faire une preuve99,2%. Biais vers petites/
moyennes communes ; hard partition CRM, pas de distracteurs hors commune,
pas validation nationale, fullOOF, SLO ou résultat initial100 fusionné.
Génération deux variantes513,37s (chargement/hydratation et diagnostic exclus),
exit0 confirmé avant ouverture labels et sidecar opérationnel officiel hashé.

Deux ratés (tous deux parent absent) : nom CRM INFODESCA vs noms officiels
indexés PROSJET IRRIGARONNE/IRRIGARONNE ; CRM NAOS vs AUBERT FRANCE.
Discordances des preuves actuellement encodées, PAS preuve d'annotation fausse
ni d'absence de lien dans toutes les sources. Ne pas ajouter ces aliases CRM
à l'index ni retirer ces lignes ; vérifier les preuves officielles historiques/
relations et le site géographique avant toute conclusion.

`report.json` sha256
`b7aa42792def7bc54e17b93fa71963a3b0f19113ab0127bdad45a2d714d32e21`.
`generation.json` sha256
`44e61b98a46c3628a76716fb4f693d715bcac6910d77d40fa23420fa734b729b`.

## Décision

**Conserver l'adapter comme canal dense expérimental prometteur** : gain hors
fit réel sur ces deux diagnostics. PAS de déploiement ni garantie générale.
Recall national du pipeline précédent toujours91,15%, objectif99,2% non atteint.
Test interne4 323 et fold1 fermés ; fold0 nonlu ; aval gelé.

La prochaine recette doit distinguer identité entreprise (noms officiels liés
au parent, variantes historiques) et preuve site (adresse, distances, confiance
BAN, alternatives de même SIREN). Ne pas simplement concaténer toutes les
adresses des siblings dans un texte ; préserver leur provenance et les
équivalences opérationnelles pour ne pas créer des faux négatifs de site.
Mesurer une admission directement<=100 et ses pertes ; aucun reranking d'un
nouveau pool fixe sous un autre nom. Une comparaison plus large et les tests
de généralisation restent dus avant une promotion.
