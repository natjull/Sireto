# Avis brut Claude Opus 5 — revue d’architecture

10 septembre 2026. Avis externe, non validé expérimentalement. Le corps ci-dessous est conservé tel que renvoyé par Claude. Certaines affirmations sont contestées dans `docs/retrieval_claude_architecture_review_assessment.md`; ne pas le traiter comme un contrat ou une autorisation d’exécution.

Modèle principal confirmé dans `modelUsage` : `claude-opus-5`; effort demandé `xhigh` (non attesté séparément). Outils désactivés; code et documents sélectionnés transmis, aucun CRM brut ni identifiant de cas. Appel réussi, durée déclarée 177717 ms. Coût indicatif CLI 0,34899675 USD, dont 0,31940475 pour Opus; appel auxiliaire Haiku4.5 également déclaré. Ce compteur n’établit pas une facturation supplémentaire à l’abonnement Max.

---

# Revue d'architecture retrieval SIRETO — synthèse et analyse

## 1. Synthèse accessible

Trois constats dominent.

**(a) Le score qui décide de la coupe à 2 000 est ordinal, donc compétitif.** Dans `make_master` (v23), le score d'un site vaut `direct_rrf + 1/(60 + learned_parent_rank + joint_site_rank)`, où `learned_parent_rank` est un `dense_rank()` calculé *par requête* sur `parent_score`. Preuve dans le code : ajouter des parents (`replacement_union` en v29 injecte jusqu'à 512 SIREN dans la membership) décale mécaniquement ce rang pour tous les autres, sans qu'aucune preuve ne change. Le micro fourni est exactement cette signature : pour les six cas, `parent_score`, `direct_rrf` et le rang du site dans le parent sont strictement inchangés, seul le rang parent se dégrade (28→41, 125→225…) et le rang global franchit 2 000. Ce n'est pas une erreur d'implémentation : c'est le comportement attendu de cette formule. Conséquence architecturale : **les expériences de génération ne sont pas additives**, ce qui rend tout progrès sur le réservoir illisible (7 nouveaux exacts, 4 survivants, 6 anciens évincés).

**(b) L'objectif appris n'est pas l'objectif produit.** v28 apprend un rang parent (`training_sample` : positif = `siren == gt_siren`), et son critère d'éligibilité (`eligible` : gains parent > pertes parent, zéro perte contrôle) est intra-canal. Le résultat v29 fourni montre que ce critère ne prédisait pas l'issue end-to-end. Surtout : le manifeste v29 publie `site_model_scoring_performed=False`, `rich_hydration=False`, `recall100_measured=False`, `prerank500_measured=False`. **Le LambdaMART SIRET exigé par le contrat v12 §3 n'existe pas.** Toute la controverse porte sur un ordonnancement heuristique que le contrat prévoit précisément de remplacer.

**(c) Une part inconnue mais non négligeable des échecs relève d'un signal absent, pas d'une architecture.** 134 des 217 erreurs v29 (62 %) sont des parents absents. Les résultats V25/V26/V27/V16 fournis indiquent, sur les 132 homologues v23 : zéro égalité de nom CRM dans les champs examinés, 3 avec un token commun, 44 hors de tout canal testé, aucun parent à ≤ 100 m, 61 sans distance exploitable. Aucune réorganisation de la cascade ne crée de la preuve.

Autrement dit : vous mesurez le mauvais objet (réservoir @2000 au lieu de @500/@100), avec un score qui rend les mesures non additives, sur une population dont une fraction est peut-être inatteignable — et vous jugez le tout sur un panel réutilisé où l'écart −5 est, en test des signes sur 13 paires discordantes (4/9), non concluant (calcul analytique, pas une expérience).

## 2. Les trois limites structurelles, reliées au micro

**Limite 1 — fusion ordinale et budget compétitif (recherche qui perd la preuve).** Preuve code : `retrieval_v21_membership.build_tables` fusionne par RRF (`sum(1.0/(60+r))`) et `retrieval_v23_joint_parent.make_master` combine deux rangs. Un candidat n'a jamais de score interprétable indépendamment de ses concurrents. Deux effets : (i) une preuve unique mais faible (les 5 cas avec `direct_rrf = 0` s'appuient exclusivement sur le terme `1/(60+rangs)`) est fongible avec du bruit ; (ii) toute amélioration de rappel se paie en évictions. La « protection 1750 » de v21 était un pansement sur ce défaut ; v22 l'a retirée, v23 ne l'a pas remplacée par une échelle absolue.

**Limite 2 — objectif parent substitué à l'objectif SIRET, et cascade bloquante.** Le seul modèle appris en jeu score des SIREN ; la sélection des 2 000 est heuristique ; le modèle SIRET, qui serait le lieu naturel d'un arbitrage calibré, ne voit que ce qui a survécu à la coupe. Les six évictions sont donc **structurellement invisibles** pour le composant censé les rattraper. C'est le point de contrat le plus contraignant : `Master@2000` n'est pas seulement un diagnostic, c'est un filtre non appris placé avant l'unique apprentissage SIRET.

**Limite 3 — représentation parent-first et identifiabilité.** L'expansion (`address_sites` en v21, `local_site` en v23) exige que le bon SIREN soit d'abord retrouvé et retenu. Le SIREN est un bloc de jointure administratif, pas une unité de preuve : la preuve observable est un couple (nom d'enseigne à une date, adresse). Résultat fourni V25 : 88/132 parents sont atteignables par des canaux *une fois filtrés sur le parent humain* — donc la preuve existe parfois dans la queue d'un canal, mais la formulation de requête et l'élagage ne l'y trouvent pas sans oracle. Hypothèse (non démontrée) : une part des 44 restants et des 61 sans distance relève d'un signal inexistant localement. **Mesure manquante décisive** : un audit d'identifiabilité label-side scellé publiant, par cas manquant, la meilleure similarité nom atteignable sur *toutes* les dénominations historiques du SIREN et la meilleure preuve d'adresse — jamais réinjecté en feature.

## 3. Trois architectures distinctes

**A — Contrôle amélioré : cascade actuelle + admission à score absolu + ranker SIRET OOF.**
Objet indexé/recherché : inchangé (dossiers SIREN, cellules d'adresse, sites locaux). Information : déjà disponible (`site_features`, `parent_score`). Génération/allocation : la coupe 2 000 utilise un score *invariant à la compétition* — produit d'une transformation monotone fixe de `parent_score` par une évidence de site déjà bornée (`joint_tokens`, `joint_jw` ∈ [0,1]) plutôt que `1/(60+rangs)` — puis le LambdaMART SIRET du contrat score les ≤2 000, produit PreRank@500 et top100. Apprentissage : parent en inner OOF (obligatoire dès que ses scores alimentent un fit aval, contrat §2), SIRET en outer OOF 2/3/4. Mécanisme de gain : rend les expériences additives et attaque les 79 évictions + les deux niveaux jamais mesurés. Risque : plafond réservoir (896/1034 sur ce panel) — le gain est borné et ne touche pas les 134. Coût Mac : faible ; le run v29 complet tient en 104 s / 9,4 Gio, un XGBRanker sur ~2 M lignes est du même ordre que le fit parent v28.

**B — Index site-centrique à représentation conjointe (nom×adresse) et parent a posteriori.**
Objet indexé : le SIRET-période, avec une clé conjointe concaténant toutes les dénominations officielles rattachées au sujet (SIREN legal/trade et SIRET site, avec validité) et l'adresse normalisée + BAN. Recherche : directe sur le site ; le SIREN devient une feature d'agrégation, plus un goulot. Mécanisme : supprime la dépendance « parent retrouvé d'abord », qui explique 62 % des erreurs. Risque d'échec sérieux : V25 a testé un canal numéro+adresse dur → 0 nouveau, et n'a trouvé que 2 parents sur 119 avec un site au numéro demandé ; V26/27 constate zéro égalité de nom chez les 132. Une représentation conjointe n'aide que si au moins un des deux termes est exploitable — l'audit d'identifiabilité doit précéder cet investissement. Coût : index Tantivy à l'échelle des établissements + périodes ; taille sur SSD et latence p95 non mesurées (mesure manquante).

**C — Planification de requêtes apprise + allocation de budget par requête.**
Objet : mêmes index, mais on apprend *quoi demander* (nom complet, tokens rares, nom×voie, voie×commune, relation structurée à un saut) et *combien de places allouer à chaque canal pour cette requête*, au lieu de top-K fixes fusionnés par RRF. Mécanisme : cible directement le constat V25 (88 cas atteignables sous filtre oracle) — la preuve est dans la queue, l'élagage la tue. Apprentissage admissible : politique entraînée en outer OOF sur des signaux disponibles en production, sans label ni identifiant en feature. Risque : c'est la proposition la plus susceptible d'explosion de bruit et de surapprentissage au panel 1034 ; sans population fraîche, elle est intestable honnêtement. Coût : plusieurs passes de recherche par requête, tension directe avec le SLO p95 ≤ 1 000 ms.

Je ne recommande ni B ni C en premier : les deux dépendent de mesures que vous n'avez pas (identifiabilité, coût d'index, population indépendante), et B/C **héritent** du défaut d'admission de A — on les mesurerait à travers le même bruit d'éviction.

## 4. Orientation prioritaire et test unique falsifiable

**Priorité : A**, formulée non comme un tweak mais comme un changement d'invariant — l'admission au Master doit être une fonction du candidat, pas de ses concurrents.

**Test (un seul run, deux bras, aucun refit) :** rejouer la génération v23 *et* la génération v29, en remplaçant uniquement l'expression `rescue_score` de `make_master` par un score à échelle absolue préenregistré (une seule constante, fixée avant lecture des résultats, aucune grille), modèles v23 gelés et rechargés à l'identique via `predict(..., reference=True)`. Métrique : exacts au Master@2000 sur le panel 2/3/4, plus le devenir des cas conservés.

- Si le bras v23 tombe sous 822 : le diagnostic « ordinalité » est faux, la formule RRF portait du signal utile — invalidation nette.
- Si le bras v23 tient et que le bras v29 ne redevient pas au moins neutre : l'hypothèse « les nouveaux parents sont additifs, seule la compétition les rendait destructifs » est réfutée ; le problème est la qualité du score parent v28 hors distribution (le modèle v23 n'a jamais vu de parents à `parent_rrf = 0` en masse), pas l'architecture d'admission.
- Cette dichotomie sépare précisément « mauvaise intégration / inférence gelée » de « mauvaise architecture ».

Pas d'oracle, pas de retuning sur les six cas (ils ne servent qu'à formuler l'hypothèse, jamais à choisir la constante). **Changement de contrat requis** : c'est une nouvelle recette, donc un addendum préenregistré ; et si le test réussit, l'étape suivante (ranker SIRET) exige l'inner OOF parent.

## 5. Feuille de route conditionnelle (3 jalons)

**J1 — Rendre les mesures additives et mesurer enfin le produit.** Test ci-dessus, puis, s'il passe, fit du LambdaMART SIRET (outer 2/3/4, parent en inner OOF) et **première publication de PreRank@500 et Recall@100**, avec Wilson. Gate : zéro perte sur les contrôles, Master conservé.

**J2 — Identifiabilité et généralisation, en parallèle.** (i) Audit label-side scellé des 134 absences : meilleure preuve nom/adresse atteignable dans les sources locales, jamais réinjectée. (ii) Nouvelle population échantillonnée par tri de `SHA256(query_id)` comme le fait déjà le contrat pour `runtime_query_sample`, stratifiée mais **sans exclure les labels difficiles** : le dénominateur reste tout label humain exploitable, une vérité absente compte erreur. C'est la seule façon de savoir si 822/1034 signifie quoi que ce soit au national — rappel : 23 571/23 049 du contrat v20 ne valide pas ce pipeline.

**J3 — conditionnel.** Si J2 montre une fraction significative de cas sans aucune preuve locale : la voie est source/temporalité (RNE brut 177 Go non audité, noms historiques indexés) ou une renégociation explicite de la cible, pas B/C. Si au contraire la preuve existe mais n'est pas requêtée : engager C (planification apprise), avec budget latence mesuré.

## 6. Contraintes et hypothèses à remettre en question

- **La coupe 2 000 avant le seul apprentissage SIRET.** C'est la contrainte structurellement la plus coûteuse : le modèle ne peut rattraper que ce qu'une heuristique lui a laissé. Recommandation nécessitant accord : autoriser un *pré-score* bon marché, calibré, sur un pool plus large, distinct du budget produit — sans jamais transformer cela en « ranker sur des millions de paires », que le rapport v29 exclut à juste titre.
- **Le panel 1034.** 512 contrôles ne peuvent que perdre : la métrique est asymétrique par construction, et sa réutilisation invalide la sélection de recettes même sous OOF. Une décision à ±5 sur ce panel n'est pas informative.
- **« Parent d'abord ».** Jamais testée comme hypothèse ; elle porte 62 % des erreurs.
- **Objectif exact vs information disponible.** Il est *possible* que l'exact humain sur des vérités fermées soit partiellement incompatible avec les sources locales — ce n'est **pas démontré** ; l'audit J2 est ce qui trancherait. Je ne promets aucun 99,2 %, et l'opérationnel (mode `EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE`) ne doit pas servir de substitut.
- **Dense.** Rien ici ne justifie de le rouvrir : le pivot v9 portait sur la génération à budget 50 ; un usage éventuel serait l'admission dans ≤2 000, économie et mécanisme différents — mais à ne pas instruire avant J1/J2.

