# V4.4 — Plan de réentraînement sur labels difficiles validés

Date : 27 juillet 2026  
Statut : design prescriptif, aucun modèle réentraîné

## Verdict

Le premier modèle à réentraîner doit être **l'accepteur query-level**, en
conservant d'abord le retrieval et le ranker gelés.

Les futurs labels V4.4 répondent directement à la question de l'accepteur :
« le top-1 produit par le système courant est-il assez fiable pour être
automatisé ? ». Ils ne fournissent pas tous une cible exploitable par le
ranker :

- `TOP1_CORRECT` donne un positif exact ;
- `TOP1_WRONG` donne toujours un négatif utile à l'accepteur, mais ne donne une
  cible ranker que si un autre SIRET exact a aussi été démontré ;
- `AMBIGUOUS` est un négatif pour `AUTO_MATCH`, mais ne doit jamais devenir un
  positif ou un groupe de négatifs candidat artificiel.

Le contrat interdit précisément de fabriquer le SIRET correct lorsqu'on sait
seulement que le top-1 est faux
([contrat V4.4, lignes 60–67](../../docs/v4_4_autonomous_adjudication_contract.md#interdictions)).
Réentraîner le ranker sur tous les `TOP1_WRONG` serait donc une erreur de cible.

## 1. Ce que les artefacts actuels permettent réellement

### Corpus V4.1

Le dataset canonique
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`
contient :

- 7 003 requêtes et 698 428 paires ;
- 5 883 `MATCH_EXACT` et 1 120 `AMBIGUOUS` ;
- six vérités exactes absentes du retrieval, conservées comme misses ;
- 100 candidats maximum et aucune injection positive.

Le code reconstruit actuellement `is_ground_truth` par égalité exacte entre le
candidat retrouvé et le SIRET labellisé, puis refuse toute réparation ou
injection
([`train_v41_models.py`, lignes 231–255](../../scripts/train_v41_models.py#L231)).
Le builder appelle le retrieval sans lui transmettre la vérité
([`build_v41_training_dataset.py`, lignes 708–745](../../scripts/build_v41_training_dataset.py#L708))
et compte explicitement les misses
([lignes 764–771](../../scripts/build_v41_training_dataset.py#L764)).
Ces invariants doivent rester inchangés.

Le ranker actuel est un `XGBRanker` pairwise entraîné uniquement sur les
requêtes `MATCH_EXACT` dont le positif est réellement dans le pool
([`train_v9_ranker.py`, lignes 41–54](../../scripts/train_v9_ranker.py#L41)).
Il produit cinq folds OOF pour le fit et une prédiction hors-échantillon pour
le dev
([`train_v41_models.py`, lignes 408–467](../../scripts/train_v41_models.py#L408)).

L'accepteur V4.1 est une régression logistique standardisée, avec
`class_weight="balanced"`, entraînée sur la cible binaire
`is_exact_siret_correct`
([`v41_acceptor.py`, lignes 98–155](../../src/xgb_matcher/v41_acceptor.py#L98)).
Son seuil est choisi sur le dev complet à 99,8 % de précision observée
([lignes 156–163](../../src/xgb_matcher/v41_acceptor.py#L156)). La construction
de scène marque un cas positif uniquement lorsque le label est `MATCH_EXACT`
et que le SIRET prédit est exactement la vérité ; un cas ambigu est donc déjà
négatif pour l'automatisation
([`v9_scene.py`, lignes 228–249](../../src/xgb_matcher/v9_scene.py#L228)).

### Corpus V4.4

La collecte officielle courante contient 440 réponses pour 172 cas AUTO :

- 172 recherches par top-1, toutes avec résultat ;
- 96 recherches par SIRET d'entrée, dont 85 avec résultat ;
- 172 recherches nom + géographie, dont 68 avec résultat ;
- zéro adjudication créée à ce stade.

L'artefact `official_evidence.parquet` ne contient que les requêtes, réponses,
URLs et dates. Le collecteur publie explicitement
`adjudications_created: 0`
([`collect_v44_official_evidence.py`, lignes 196–216](../../scripts/collect_v44_official_evidence.py#L196)).
Il ne peut donc pas encore être joint au dataset d'entraînement.

Les 172 cas sont tous issus d'anciens `UNRESOLVED` devenus des décisions
`AUTO_MATCH` V4.1 : 94 `AUTO_NEAR_THRESHOLD`, 57 `RANDOM_POPULATION` et 21
`AUTO_HIGH_SCORE`. Ils ne chevauchent aucune des 7 003 requêtes du dataset
V4.1. Les prédictions shadow sont donc hors-échantillon vis-à-vis du ranker
V4.1, ce qui les rend adaptées à un premier entraînement de l'accepteur.

Ce corpus reste cependant un échantillon conditionné par la décision AUTO et
par l'absence de résolution antérieure. Même ses 57 cas aléatoires sont
aléatoires **parmi les cas de la file**, pas parmi toutes les entrées CRM. Il
est excellent pour apporter des erreurs difficiles au fit ; il ne peut pas
servir seul à mesurer la couverture globale ni à choisir honnêtement un seuil
de production.

## 2. Schéma canonique des adjudications V4.4

Créer une table immuable `adjudications.parquet`, à une ligne par cas, avec au
minimum :

| Champ | Type / règle |
|---|---|
| `audit_case_id` | chaîne, unique |
| `query_id` / `service_id` | identifiants stables, non utilisés comme features |
| `adjudication_label` | `TOP1_CORRECT`, `TOP1_WRONG`, `AMBIGUOUS` ou `UNRESOLVED` |
| `evidence_validated` | booléen ; vrai seulement si le contrat des deux preuves est satisfait |
| `training_eligible` | booléen dérivé, jamais saisi librement |
| `frozen_top1_siret` / `frozen_top1_siren` | proposition réellement jugée |
| `frozen_model_bundle_id` | bundle ayant produit le top-1 |
| `frozen_retrieval_signature` | retrieval ayant produit le pool |
| `frozen_candidate_pool_sha256` | hash du pool ordonné réellement jugé |
| `validated_correct_siret` | nullable ; obligatoire lorsque le bon SIRET exact est démontré |
| `validated_correct_siren` | dérivé du SIRET exact, nullable sinon |
| `evidence_refs_json` | au moins deux références indépendantes |
| `evidence_snapshot_or_date` | snapshot et dates auxquels le jugement s'applique |
| `adjudication_rule_version` | version de la règle appliquée |
| `adjudicated_at` | date UTC |
| `sampling_stratum` / `priority_reason` | sélection de la ligne, pour audit et pondération |

Contrôles bloquants :

1. `training_eligible=true` implique `evidence_validated=true`, deux sources
   indépendantes, zéro contradiction non résolue et un top-1 gelé.
2. `TOP1_CORRECT` implique
   `validated_correct_siret == frozen_top1_siret`.
3. `TOP1_WRONG` implique que les preuves réfutent le top-1.
   `validated_correct_siret` peut rester nul.
4. `AMBIGUOUS` implique qu'aucun SIRET exact unique n'est retenu ;
   `validated_correct_siret` est nul.
5. `UNRESOLVED` est toujours `training_eligible=false`.
6. Les réponses officielles, le label, le SIRET validé et tout champ produit
   après la décision sont interdits dans les features.

La table V4.3 a déjà réservé les champs de validation humaine mais les initialise
à vide et force `training_eligible=False`
([`build_v43_hard_label_queue.py`, lignes 427–450](../../scripts/build_v43_hard_label_queue.py#L427)).
V4.4 doit publier un nouveau schéma explicite plutôt que surcharger silencieusement
ces colonnes provisoires.

### Cibles dérivées

Produire séparément une table de cibles, sans recopier les preuves dans la
matrice de features :

| Adjudication | `acceptor_target` | `acceptor_eligible` | `ranker_target_siret` | `ranker_eligible` |
|---|---:|---:|---|---:|
| `TOP1_CORRECT` | 1 | oui | top-1 validé | oui si le SIRET est naturellement dans le pool |
| `TOP1_WRONG`, remplacement exact connu | 0 | oui | SIRET validé | oui si naturellement dans le pool |
| `TOP1_WRONG`, remplacement inconnu | 0 | oui | null | non |
| `AMBIGUOUS` | 0 | oui | null | non |
| `UNRESOLVED` | null | non | null | non |

Conserver aussi `is_ambiguous` comme attribut d'évaluation. Le modèle de
production peut rester binaire : pour le SIRET exact, automatiser un cas
ambigu est bien une erreur. Une éventuelle tête trois classes ne serait qu'une
ablation diagnostique, pas une nouvelle définition de la cible.

## 3. Construction sans fuite

### Expérience A — accepteur d'abord, ranker V4.1 gelé

Cette expérience est la plus courte et la plus informative.

1. Geler les hashes du retrieval, du ranker, des pools, des scores et des 80
   features de scène avant d'ajouter les adjudications.
2. Reconstituer la scène exacte de chacun des 172 cas avec le bundle V4.1 gelé.
   Vérifier que le top-1 obtenu est identique au `frozen_top1_siret`.
3. Joindre uniquement les adjudications `training_eligible=true`.
4. Ajouter ces scènes au **fit seulement**. Comme ces requêtes ne figuraient
   pas dans le fit du ranker V4.1, leurs scores sont déjà hors-échantillon.
5. Conserver les scènes V4.1 historiques : exactes OOF, ambiguës non vues par
   le ranker et misses sous forme de sentinelles. Le code sait déjà conserver
   une requête sans prédiction
   ([`train_v41_models.py`, lignes 537–558](../../scripts/train_v41_models.py#L537)).
6. Entraîner en premier la même régression logistique standardisée. C'est
   l'ablation causale : seule l'information de supervision difficile change.
7. N'ouvrir une ablation XGBoost accepteur qu'après ce résultat. Ne pas
   sélectionner des features ou hyperparamètres après lecture du dev.

Le code actuel filtre l'entraînement de l'accepteur sur les scènes ayant passé
les préchecks
([`train_v41_models.py`, lignes 641–676](../../scripts/train_v41_models.py#L641)).
Les nouvelles scènes doivent rester dans la métrique end-to-end même lorsqu'un
précheck les envoie en REVIEW ; seules les scènes réellement éligibles doivent
entrer dans le fit de l'accepteur.

### Dev nécessaire

Les 172 AUTO difficiles ne doivent pas être utilisés pour fixer le seuil. Il
faut un dev représentatif gelé avant scoring, comprenant AUTO et REVIEW, tiré
dans la population CRM autorisée puis adjugé avec le même standard de preuve.

À défaut de ce dev, l'expérience A peut mesurer :

- la rétention des `TOP1_CORRECT` difficiles ;
- le rejet des `TOP1_WRONG` ;
- le rejet des `AMBIGUOUS` ;
- le changement de scores sur les scènes V4.1 historiques.

Elle ne peut pas revendiquer une nouvelle couverture globale à 99,8 %. Le bon
statut serait alors `GO_BUILD_REPRESENTATIVE_DEV`, pas `GO_DEPLOY`.

Le dev représentatif doit être divisé avant tout tuning en :

- `calibration`, uniquement si une calibration est préenregistrée ;
- `threshold`, pour choisir le seuil ;
- un futur test final indépendant, ouvert une seule fois.

Le code V9 sait déjà séparer calibration et seuil par hash déterministe
([`v9_scene.py`, lignes 296–304](../../src/xgb_matcher/v9_scene.py#L296)) et
entraîner logistique/XGBoost avec calibration séparée
([`v9_acceptor.py`, lignes 232–366](../../src/xgb_matcher/v9_acceptor.py#L232)).
Compte tenu de la saturation isotonic observée sur le holdout V4, la variante
primaire doit rester **logistique brute**. Une sigmoid peut être
préenregistrée comme ablation si le lot de calibration contient assez
d'erreurs ; isotonic ne doit pas être le choix par défaut sur un petit dev.

### Expérience B — ranker ensuite, conditionnelle

Ne lancer cette expérience que si :

- l'accepteur seul ne donne pas assez de couverture ; et
- V4.4 apporte suffisamment de `TOP1_WRONG` avec un
  `validated_correct_siret` exact présent naturellement dans le pool.

Repartir alors de pools candidats figés et reconstruire tous les splits avant
entraînement :

1. Dédupliquer par `service_id` et par empreinte CRM normalisée.
2. Construire des composantes connexes avec :
   `input_siren`, `validated_correct_siren`, `frozen_top1_siren` et les SIREN
   de l'historique/ligne de continuité explicitement prouvés. Ne pas relier les
   composantes par tous les candidats arbitraires du pool.
3. Interdire qu'une composante traverse fit, dev, test ou deux folds OOF.
   L'implémentation actuelle ne relie que SIREN d'entrée et SIREN cible
   ([`v41_split.py`, lignes 40–80](../../src/xgb_matcher/v41_split.py#L40)) ;
   elle devra donc intégrer le top-1 réfuté et les liens de continuité validés.
4. Assigner les nouveaux cas V4.4 difficiles au fit. Réserver à dev/test des
   échantillons représentatifs gelés séparément.
5. Faire cinq folds dans le fit. Pour chaque fold, entraîner le ranker sur les
   seuls labels exacts des quatre autres folds, puis scorer **toutes** les
   scènes du fold, y compris mauvaises, ambiguës et misses.
6. Entraîner le ranker final sur tous les exacts fit éligibles ; scorer le dev
   une seule fois.
7. Construire ensuite l'accepteur exclusivement sur ces nouvelles prédictions
   OOF/hors-échantillon. Ne jamais réutiliser pour l'accepteur des scores
   in-sample du ranker réentraîné.

Une vérité validée absente du pool reste une erreur retrieval end-to-end et ne
devient pas une ligne injectée. Elle est exclue du fit ranker, mais conservée
comme scène incorrecte pour l'accepteur et dans toutes les métriques.

## 4. Métriques et gate

### Métrique nord

Sur le dev représentatif gelé :

```text
precision_AUTO_exacte =
  AUTO dont le SIRET prédit est exactement valide
  / toutes les décisions AUTO

couverture_end_to_end =
  toutes les décisions AUTO
  / toutes les requêtes du benchmark autorisé
```

`TOP1_WRONG` et `AMBIGUOUS` automatisés comptent comme erreurs. Une absence de
candidat, une erreur retrieval ou un précheck REVIEW reste dans le dénominateur
de couverture. Publier en plus la couverture conditionnelle parmi les scènes
éligibles à l'accepteur, mais ne jamais la substituer à la couverture
end-to-end.

Le seuil maximise la couverture sous :

- précision SIRET exacte observée `>= 99,8 %` ;
- au moins 100 AUTO pour rester compatible avec V4.1 ;
- nombres bruts `N`, `AUTO`, corrects et erreurs ;
- borne de précision unilatérale exacte à 99 %, publiée comme information et
  non comme garantie.

Le sélecteur actuel calcule correctement la courbe empirique et choisit le
point de couverture maximale sous la contrainte de précision
([`selective.py`, lignes 56–133](../../src/xgb_matcher/selective.py#L56)).
Il dispose aussi de la borne Clopper-Pearson unilatérale
([lignes 136–174](../../src/xgb_matcher/selective.py#L136)).

Avec 500 AUTO, une seule erreur donne exactement 99,8 % observé ; avec moins de
500, il faut zéro erreur. Pour soutenir une borne unilatérale à 99 % au-dessus
de 99,8 % avec zéro erreur, il faut environ 2 300 AUTO indépendants. Avant ce
volume, le rapport doit dire « estimation observée ».

### Tableaux obligatoires

Comparer, sur les mêmes lignes :

1. politique V4.1 gelée ;
2. accepteur réentraîné, ranker gelé ;
3. uniquement si autorisé, ranker puis accepteur réentraînés.

Publier globalement et par :

- `TOP1_CORRECT`, `TOP1_WRONG`, `AMBIGUOUS` ;
- `sampling_stratum` et `priority_reason` ;
- état du SIRET d'entrée ;
- même SIREN / autre SIREN entre entrée, top-1 et vérité ;
- mono-site / multi-site ;
- famille de preuve et niveau de preuve ;
- retrieval miss, aucun candidat et précheck ;
- mégapole, commune et segment source.

Pour l'accepteur, publier la couverture à 99,0 %, 99,5 % et 99,8 %, le taux de
rejet des mauvais top-1, le taux de rejet des ambigus et la rétention des bons
top-1.

Pour le ranker conditionnel, publier Recall@100 exact, Hit@1 SIRET, Hit@1
SIREN, nombres de misses, gains sur les mauvais top-1 validés et régressions
par segment. Une variante ne passe pas si Recall@100 régresse, si Hit@1 global
régressse ou si un segment critique régresse de plus de deux points.

## 5. Séquence d'exécution recommandée

1. **Achever V4.4** : publier les adjudications immuables et franchir
   `GO_RETRAIN_AUTO` avec au moins 75 bons top-1, 50 mauvais top-1 et 30 cas du
   tirage aléatoire validés conformément au contrat
   ([contrat V4.4, lignes 69–84](../../docs/v4_4_autonomous_adjudication_contract.md#gate)).
2. **Geler un dataset de scènes V4.4** lié aux hashes du bundle, du retrieval,
   des pools et des preuves.
3. **Réentraîner la logistique de l'accepteur seulement**, en plaçant V4.4 dans
   le fit et en conservant le ranker.
4. **Construire et adjuger un dev représentatif indépendant** si aucun corpus
   adéquat n'est déjà disponible. Sans lui, ne pas fixer de nouveau seuil de
   production.
5. **Comparer la politique gelée et l'accepteur réentraîné** sur ce dev, puis
   geler modèle et seuil.
6. **N'envisager le ranker qu'en deuxième intention**, après comptage des
   `TOP1_WRONG` disposant d'un SIRET exact alternatif naturellement retrouvé.
7. **Ouvrir un nouveau test final une seule fois** après gel complet. Les
   anciens holdouts consommés restent interdits pour la sélection.

Cette séquence teste d'abord l'hypothèse la plus directement soutenue par les
nouvelles données : le système sait souvent proposer le bon SIRET, mais son
accepteur n'a pas appris les scènes CRM sales qui rendent un top-1 plausible
et faux. Elle évite de modifier simultanément le classement et la décision,
ce qui empêcherait d'identifier la cause du gain ou de la régression.
