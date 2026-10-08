# V14 — mesure du pipeline complet sur les 2 212 requêtes

Autorisation humaine « Let’s go » du 8 octobre, après demande explicite du
résultat EmbeddingGemma2 fine-tuné + canaux complémentaires. Ce calcul peut
prendre plusieurs jours sur le Mac ; aucune nouvelle nuit programmée ni
dépense externe. Ancienne automatisation nocturne laissée en pause.

Population intégrale V10 : 2 212 requêtes, 1 430 SIREN, folds 2/3/4, 937 communes,
19 097 808 établissements dans les communes complètes. Les trois requêtes sans
INSEE restent au dénominateur avec canal dense vide. Aucun budget de sites,
filtre de qualification, sélection des petites communes ou corpus autour du GT.
Validation utilisée pour sélectionner le checkpoint : pas un test pristine,
pas une certification de généralisation à tous les CRM ni une validation OOF.

## Recette figée avant le score de cette population

- Canal lexical national V12 complet, owned exit0, réutilisé sans modification.
- Ancien encodeur natif160 : adapter historique step7200, batch16 original.
- Nouveau SITE512 : EmbeddingGemma2 adapter best1000 du fit finalisé V6,
  batch8 original. Maximum du cosinus sur toutes les vues SITE officielles,
  sans mélange du score identité parent. Noms légaux/historiques du parent,
  propres noms/enseignes historiques du site et adresses officielles. Aucun
  nom d'un établissement voisin copié comme enseigne du site cible.
- Géographie indépendante : BAN CRM uniquement, XY officiels nationaux,
  nearest100 dans max(300m fiable ou 1000m faible, 3 sigma). Sources et recette
  V5 conservées, couverture XY des établissements ajoutés explicitement limitée.
- Sortie Gemma2+lexical+géo : RRF60, poids1/1/1 et géo faible0,5, plafond100.
- Comparaison fusion de famille V10 : MAX de contribution RRF ancien/nouveau
  dense puis lexical et géo, plafond100. Pas de double vote des deux encodeurs.

Chaque canal émet initialement au plus100 ; recherche dense exacte sur TOUTE
la commune, sans pool fixe2000/500 ni ranker aval. La fusion n'est pas apprise.
Pas d'attribution isolée au fine-tuning : poids et textes diffèrent simultanément
entre ancien et nouveau. BODACC/expansion par filiations ne sont PAS mesurés ici.

## Reprises et intégrité

Encodage avec runtime de la nuit torch2.14.1/transformers5.19.0, pas celui
différent du projet. Packing512 original et aucun nouveau tri/batching32.
Les 2 212 requêtes ont été vérifiées sans labels : max60tokens, aucune>512.
Les vues préparées compatibles V3 sont réutilisées littéralement ; autres
communes enrichies également pour tous les distracteurs depuis tout V130.
Un lookup SQLite par propriétaire rend ces mêmes noms accessibles sans scans
répétés. 64 transactions atomiques, faits et noms originaux vérifiés par hashes.
Ce lookup ne dépend ni des labels ni des requêtes. SIREN/SIRET : joins/sorties
uniquement, jamais features. Noms normalisés/dédupliqués avant encodage.

Vecteurs float32 en blocs1024 hashés, appels d'inférence128 conservant les
batches16/8 d'origine. Binding corpus/poids/texte/ordre/code/paramètres/runtime.
Les textes ne sont pas reclassés. Reprise à binding identique ; les blocs déjà
scellés ne sont pas recalculés. Les sources actives ne sont plus éditées.

Admission free30%, pression0/1/2 ; yield réel free20%/RSS10GiB ; supervision
free15%/RSS12GiB, SIGTERM coopératif puis kill90s seulement si nécessaire.
Reprises automatiques seulement sur ressources/MPS OOM reconnu ; erreurs de
code/données/hashes imposent STOP. Trois reprises sans progrès DURABLE : STOP.
Réserve CATNAT80GiB. Caffeinate seulement pendant le processus ; pas de réglage OS.
Base/lookup entièrement hashés à compilation puis à fin ; pendant l'exécution,
epoch taille/mtime contrôlée. Segments Tantivy : inventaire taille/mtime et
manifest/meta, pas rehash complet des gros segments, limite explicitement publiée.

Tous les canaux, communes, sorties et owned exits0 sont requis avant labels.
Exact/SIREN/opérationnel, Wilson95, bootstrap par SIREN2000tirages, segments
folds/rôle/V2/V3/origine/état et gains/pertes appariés publiés ensemble. Sidecar
officiel label-side scellé vérifié. Aucun recall partiel ni dénominateur réduit.
Test4323 reste fermé jusqu'au gel final après cette mesure ; folds0/1 fermés,
aval/serving gelés. Aucun 99,2% national, OOF ou SLO promis par cette validation.
