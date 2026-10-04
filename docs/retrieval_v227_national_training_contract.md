# V227–228 : apprendre la différence local / homonyme national

Contrat avant mesure du 4 octobre2026. V226 prouve que toutes les paires
effectivement utilisées par chacun des trois modèles77 ont place positif=1
et place négatif=1 (commune OU code postal, pas distance). Le coefficient
place=0 ne peut pas être appris à partir de ces contrastes nuls. Cela ne
prouve pas que cette dimension explique tous les dix ratés de217.

## V227 : TRAIN64, jamais une mesure de rappel produit

Les64 premières requêtes de l'ordre préexistant des512 scènes56, sans choix
sur les labels, ratés, noms ou géographie. Source512 intégrale, folds2/3/4
uniquement, aucun recouvrement de query_id avec le pilote128. La préparation
projette strictement les huit champs query_id/fold/CRM dans un JSON dédié.
Le worker mineur ne lit aucun label, ancienne liste candidate, ledger,
sidecar cible ou référence à la source512 label-side. Son modèle77 est OOF
par rapport au fold de la requête ; protections d'entités77 conservées.

Deux collecteurs nationaux indépendants100 : couverture nom + nom exact,
puis preuve native11 multichamp avec poids77. Aucun filtre commune/CP.
Les onze features sont extraites uniquement pour les références émises
naturellement. Ce minage supervisé futur ne définit ni pool ni sortie du
produit, ne prétend pas à du recall@200 et n'injecte pas de positif.
Sceller les64 sorties et les huit sorties réelles exit0/PASS214 avant fit.

## V228 : un seul coefficient, pas un reranker

Après scellement227, ouvrir exclusivement la source TRAIN512 officielle56,
les64 scènes sélectionnées et le sidecar acceptable_opérationnel filtré
par ces64 IDs ET folds2/3/4. Le positif exact et ses équivalents officiels
ne sont jamais des négatifs. Les features positives et négatifs locaux sont
les onze primitives56 déjà scellées ; les nouveaux négatifs sont ceux227.
Dédupliquer local/national et conserver les concurrences locales.

Pour chaque modèle externe2/3/4 : exclure le fold tenu, tout positif dont
le SIREN est dans le registre tenu et chaque négatif de ces SIREN. Publier
ces exclusions, requêtes et paires. Ne jamais utiliser les128 labels.
Au moins16 requêtes sûres et un contraste de place sont nécessaires.

Seul le poids place (index4) change ; les dix autres poids77 restent fixes.
Logistique pairwise, moyenne par requête et mélange local/national50/50
quand les deux groupes sont présents. Régularisation L2 fixe0,005 autour
du poids77, borne non négative[0,64], L-BFGS-B sans recherche de paramètres.
Le score de fit se limite explicitement aux onze primitives56 : il ne
reproduit pas les bonus propriétaires, fragments, XY et expansion sparse217.
C'est une hypothèse isolée, pas la calibration complète ni une promesse.

Ce coefficient sera appliqué directement aux postings de l'index AVANT
le collecteur100. Aucun hard-filter géographique, pool fixe puis score,
changement aval ou adaptation aux dix ratés. Les protections77 de vocabulaire
et les expansions sparse128 restent identiques. Une génération128 entière
et une nouvelle évaluation séparée post-scellement seront obligatoires.

## Ressources et portée

Politique214 inchangée :2GioRSS,80s/worker,20s/requête ouverture comprise,
30s de calme avant chaque worker, pression normale et50% mémoire libre,
un thread ; aucun achat ou autre application touchée. STOP/partiel non
réutilisable. Sources rehashées et epochs pré/post, index inventorié,
segments natifs non entièrement rehashés comme217. Modèles aval gelés.

Ce test n'atteint pas lui-même l'objectif99,2% exact@100. Aucun READY,
fold0 jamais lu, fold1 fermé. Le pilote128 déjà réutilisé ne prouve pas
la généralisation ; OOF complet, Wilson, chaque fold, contrôles de
généralisation, attribution et SLO restent nécessaires.
