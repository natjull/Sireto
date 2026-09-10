# Avis brut Claude — rejeu proche production

10 septembre 2026. Avis externe non validé expérimentalement. Les corrections dans
`retrieval_claude_production_replay_assessment.md` prévalent sur les recommandations
non conformes de cet avis; ce document n'est pas un contrat d'exécution.

Modèle principal confirmé : `claude-opus-5`; effort demandé `xhigh`, non attesté
séparément. Code et documents sélectionnés fournis textuellement, outils/MCP/hooks
désactivés, aucun CRM brut transmis. Un appel réussi, durée déclarée 122335 ms.
Coût indicatif CLI 0.35673725 USD, auxiliaire Haiku inclus; ce compteur
ne prouve pas une facturation supplémentaire à l'abonnement. Aucun achat.

---

## Résumé (5 lignes)

1. Vos 80,17 % mesurent la couverture d'un réservoir de génération sur un panel stratifié enrichi en échecs — ce n'est ni un recall, ni un taux produit, ni une mesure de production.
2. Le rejeu proposé corrige surtout **la population** ; il ne corrige presque rien de la **fidélité d'exécution**, qui est le vrai trou : les scripts consomment des artefacts précalculés pour ces 1 034 requêtes.
3. Attendez-vous mécaniquement à un chiffre bien plus haut (le panel sur-représente les échecs) : pré-enregistrez cette attente, sinon vous lirez un artefact d'échantillonnage comme un progrès.
4. Un seul run apparié v23/v30 sur les mêmes entrées, modèles gelés, sans refit, en shards — pas deux campagnes.
5. Avant tout calcul long : prouver que chaque entrée est reconstructible à l'arrivée, et recompter le ledger éligible.

## 1. Ce que mesure réellement 80,17 %

Cinq axes distincts, tous confondus aujourd'hui dans un seul nombre :

**Fidélité d'exécution.** Faible et non mesurée. `discover()` charge une SceneIR indexée par `(query_id, fold)` ; `generate()` lit `structured_signals`, `siret_native_signals`, `v17_parent_presence`, `raw_union` v11, `parent_features.parquet` déjà matérialisés pour ce panel. Ces artefacts sont *label-blind*, ce qui est nécessaire mais pas suffisant : label-blind ≠ reconstructible à l'arrivée d'une ligne CRM. Tant que vous n'avez pas exhibé une fonction `ligne CRM brute → RetrievalQuery → candidats` qui reproduit ces fichiers, vous mesurez un banc d'essai, pas un pipeline.

**Population.** Panel de 522 anciens échecs top1750 + 512 anciens hits déterministes. Il n'est pas un échantillon de la population CRM : il est délibérément sur-pondéré en difficulté. 80,17 % est donc un plancher stratifié. Les 512/512 contrôles le disent explicitement. Corollaire à pré-enregistrer : le rejeu global donnera très probablement 93–97 %, **sans aucun progrès technique**. Ce serait un artefact de dénominateur, pas une victoire.

**Apprentissage / choix de recette.** Les modèles parents v23 sont outer-OOF sur ce même panel, et le panel a servi de terrain de décision sur ~10 versions. Le biais adaptatif est réel, non chiffrable par un intervalle.

**Validation indépendante.** Nulle. Même CRM, mêmes folds dev.

**Latence.** Non mesurée. 229,84 s / 1 034 à 4 workers, p95 1,793 s par requête en concurrence, c'est un coût de recherche à froid, pas un SLO chaud end-to-end.

Le prochain rejeu, tel que vous l'avez formulé, répond à **la population** et un peu à la latence à froid. Il ne répond ni à la validation indépendante, ni au biais adaptatif, et ne répond à la fidélité d'exécution **que si vous ajoutez explicitement la reconstruction par entrée**. C'est la simplification à corriger dans votre réponse.

## 2. Le protocole recommandé : un seul run apparié, modèles gelés

**Modèles : gelés, pas de refit.** Refit mesurerait une nouvelle procédure d'entraînement, coûterait des labels rouverts et détruirait la comparabilité avec v23/v30 scellés. Vous voulez d'abord savoir ce que fait *le système actuel* sur une population complète.

Mais la garde à ne pas lever aveuglément : `predict()` vérifie `set(frame.query_id) <= set(split['target_query_ids'])`. Étendre la population fait **échouer** ce contrôle. Ne le supprimez pas : remplacez-le par la garde qui compte réellement, l'isolation par **SIREN**. Vérifiez que l'assignation de fold du ledger est déterministe par `GT_SIREN` (et non par `query_id`) ; alors une nouvelle entrée du fold *f* est OOF pour `outer{f}` par construction. Partitionnez et publiez séparément : **(A)** entrées dont le SIREN est disjoint des SIREN d'entraînement du modèle concerné → métrique principale ; **(B)** violations éventuelles → segment publié, hors titre.

**v23 seul ou v23+v30 ?** Les deux, mais **pas deux campagnes**. Sur une population nouvelle, les signaux v23 n'existent pas plus que ceux de v30 : il faut de toute façon rejouer toute la découverte. v30 = v23 + un canal additif (l'additivité du réservoir est déjà contrôlée). Donc : une passe de découverte produisant les deux jeux de signaux, puis deux extractions de features / inférences / Masters sur les mêmes entrées. Le surcoût marginal de v30 est le canal joint plus une seconde passe DuckDB — quelques pourcents. Un v30 sans référence v23 sur les mêmes entrées serait de l'information gaspillée : le delta apparié est la seule quantité interprétable.

**Ce que le run doit produire par entrée** : sources et prétraitements identiques à la production (champs CRM disponibles à l'arrivée uniquement), tous les canaux branchés, index Tantivy et snapshot SIRENE/RNE/BODACC scellés en lecture seule (ce sont des caches légitimes, ils existent à l'arrivée), **aucun cache indexé par `query_id`**, vérités ouvertes seulement à l'étape d'évaluation par un programme séparé.

**Invariances exigées** (bloquantes, testées sur 16–64 entrées) : identité des sorties entre 1 et 4 workers ; indépendance à l'ordre et aux frontières de shards ; départage terminal déterministe ; parité shard/monolithe sur un sous-ensemble.

**Erreurs et timeouts** : un budget par entrée, issue typée `OK / CANAL_VIDE / TIMEOUT / ERREUR`, toutes conservées au dénominateur comme échecs, aucune reprise adaptative, aucun abandon silencieux.

## 3. Prérequis avant le calcul long

Trois, courts et décisifs — pas une nouvelle succession de recherches.

**(a) Recompter le ledger.** Nombre exact d'entrées avec SIRET humain exploitable, par fold, publié avant le run. Les 23 571 / 23 049 de v20 sont historiques et ne prouvent rien. Publiez aussi la liste des exclusions (seule cause admise : absence de SIRET humain).

**(b) Test de reconstructibilité, ~16 entrées.** Pour chaque artefact d'entrée (SceneIR, `raw_union` v11, `v17_parent_presence`, `names`, `sites`), démontrez qu'il est régénérable depuis la ligne CRM et les sources scellées, puis prouvez l'**égalité bit-à-bit** avec l'artefact existant sur ces 16 entrées du panel. C'est le test qui n'existe pas aujourd'hui et qui est le plus susceptible d'échouer : si `v17_parent_presence` ou l'union v11 sont des legacy non régénérables, **la recette v23 n'est pas rejouable hors panel** et tout le reste tombe. Découvrez-le en une heure, pas après 20 heures de calcul.

**(c) Dimensionnement.** 3 996 520 paires pour 1 034 requêtes ≈ 3 866/requête. À 20 000 entrées, ~77 M paires : la garde 14 Gio et DuckDB 8 Go ne passeront pas en monolithe. Il faut un runner en shards de requêtes, chaque shard scellé indépendamment, agrégation déterministe, garde RSS et zéro swapout par shard. Écrivez un **nouveau** runner (config v31) ; ne patchez pas le script v30, qui reste scellé avec son `len(ids)!=1034`.

## 4. Livrable, biais restants, décision suivante

**Livrable unique** : un rapport scellé donnant, sur entrées identiques, N éligible par fold ; Master@2000 exact **et** opérationnel pour v23 et v30 ; le tableau apparié gains/pertes ; l'attribution exhaustive des erreurs (catalogue absent / parent absent / site non expansé / éviction) ; les segments A/B d'isolation SIREN, fold, origine, état ; les issues d'exécution ; les temps par entrée étiquetés *froid, hors SLO*.

**Biais qui subsistent, à écrire dans le rapport** : même CRM de développement, donc aucune généralisation cross-CRM ; biais adaptatif accumulé sur des dizaines de versions, qu'aucun Wilson n'efface — si vous publiez un intervalle, faites-le sur le delta apparié (discordances) et étiquetez-le comme borne de bruit d'échantillonnage seulement, jamais comme incertitude totale ; modèles gelés appliqués à une distribution de features déplacée ; index non rehashé intégralement.

**Information vraiment attendue de ce run** : moins le chiffre global que le **taux de pertes par éviction sur la majorité facile**. Les 512 contrôles n'avaient aucune puissance pour le détecter ; c'est là que v30 peut coûter cher à l'échelle.

**Décision suivante, pré-enregistrée** : si le Master@2000 global reste nettement sous 99,5 %, le verdict `PIVOT_GENERATION` est confirmé à l'échelle et la priorité est le parent absent — pas 500/100, pas le fold 1. Si le delta apparié v30−v23 devient nul ou négatif hors panel, v30 reste challenger gelé et n'est pas activé. Aucun `READY_FOR_FOLD1`, aucune promesse de 99,2 %, folds 0 et 1 fermés.

