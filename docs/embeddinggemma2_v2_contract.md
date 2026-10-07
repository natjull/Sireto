# V2 — fine-tuning entreprise/site, données officielles et contrôles honnêtes

Autorisation7octobre2026 : mieux paramétrer le fine-tuning avec les données
disponibles. Une recette motivée, pas un optimum promis ni une grille coûteuse.
Baseline7200 conservée immuable, aucun modèle aval réentraîné ou déployé.

Même split SIREN que V1 : TRAIN15 036dont1positif natif absent publié,
validation2 212, test4 323 JAMAIS lu. Folds0/1 nonlus. Le panel2000 est exclu
des exemples, pas garanti disjointSIREN ; ne pas y valider ce modèle. Population
historique déjà utilisée pour développer le pipeline, pas holdout pristine.
Suivi des pertes/gains, exact/opérationnel et généralisation avant promotion.

Préparation : toutes les lignes TRAIN/validation membership et sources V1
rehashées. Tous les sites natifs des parents TRAIN/validation énumérés,
actifs/fermés ; arrêt si plus500k, pas sélection à partir de scores/labels.
Dictionnaire officielV130 complet64buckets fullhash pour enrichir les noms UL,
commerciaux et sites, historiques inclus dans sa projection admissible.
Les noms de siblings restent contexte du parent : jamais copiés comme enseigne
du SIRET cible. Les faits masqués/nonlexicaux restent hors représentation.
Les observations/périodes détaillées restent dans les autorités : le pack ne
prétend pas reconstituer des couples nom/adresse datés sans preuve.

Documents512tokens maximum, padding dynamique/micro2. Déduplication des chaînes,
packs complets qui conservent chaque atome officiel ; découpage tokenisé explicite
avec overlap16 pour un atome individuel trop long. Aucune troncature silencieuse
tokenizer. Corpus de vues conservé, exposition réelle de chaque vue auditée :
ne pas prétendre que toutes furent entraînées si le job s'arrête avant couverture.
Les requêtes CRM trop longues provoquent un STOP auditable, pas un retrait de
label. NFKC avant suppression9/14digits, noms ND/masqués exclus, CP conservé.

Deux objectifs sur le même encodeur/adaptateur :75% SIRET/site nom+adresse,
25% identité entreprise nom-only, pertes contrastives masquant les faux négatifs
de même parent. Même SIRET ou sites équivalents/incertains jamais négatifs.
Same-parent other-site négatif seulement si query confirme une adresse numérotée
du positif, commune commune, toutes adresses officielles parsables, aucune
adresse courante/historique partagée ni suffixe ambigu. Cette règle est stricte
et ne corrige jamais une annotation. Aucun sidecar label-side d'évaluation dans
le TRAIN. Les relations ne sont pas transformées en labels positifs d'identité.

Négatifs : lexical officiel V1 sûr (trois disponibles), plus sites distincts
prouvés et concurrents denses minés uniquement dans la banque TRAIN : tous les
sites des SIREN TRAIN et les anciens négatifs sûrs. Ce corpus d'entraînement
réduit n'est JAMAIS présenté comme une évaluation retrieval. Ne pas lire test
pour filtrer ; les exclusions des négatifs existants héritent du prepareV1
hashé. Pas de query/aliasCRM ajoutés aux documents. XY/BAN restent preuves
numériques d'un futur canal séparé, pas chiffres injectés en texte ; manque de
preuve de marque/succession ne se répare pas par davantage d'epochs.

Paramètres `config/embeddinggemma2_v2.json` : reprise du meilleur adapterV1
rank8/alpha16 Q/V/projection, baseBF16 gelée, vrais batches16/micro2 via gradient
cache, température.05, AdamW LR3e-5, warmup3%, cosine vers10%LR, decay.01,
clip1, deuxepochs/huitheures max, checkpoint25pas/cursor déterministe, validation
256queries choisies par hash avant fit tous200pas, arrêt après3nonaméliorations.
Benchmark de gradient FP64, masque siblings et préflight réel incluant les
documents les plus longs avant job long. Baseline initiale conservée si pas gain.

Mac/CATNAT uniquement : runtime isolé, CPU1, MPS cap.25 jamais désactivé,
RSS5Gio, admission65%libre/yield50%, supervisor libère ownchild avant45% ou
pression anormale. Reprises12 resource-only, erreurs data/code/numerics fatales,
owned exits/checkpoints/traces. Aucun cloud/upload/dépense externe. Source
metadata/epochs contrôlées, gros segments Tantivy pas fullrehashés (limite).

Sélection checkpoint sur diagnostic validation, PAS preuve99,2%. Comparer V1/V2
sur le même corpus officiel complet des zones déclarées, puis élargir les
contrôles/généralisation. Sortie recherche<=100 avant aval, pas pool fixe2000/500
reranké sous un autre nom. SLO/national/fullOOF encore dus, test réservé fermé.

Appuis : [modèle Google](https://huggingface.co/google/embeddinggemma-2),
[loss contrastive et gradient cache](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cachedmultiplenegativesrankingloss).
