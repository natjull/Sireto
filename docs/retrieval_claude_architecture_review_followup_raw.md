# Contre-examen brut Claude Opus 5 — revue d’architecture

10 septembre2026. Réponse brute à nos objections, pas un contrat ni une validation expérimentale. Le relecteur reçoit le mandat corrigé, le dossier d’architecture et son premier avis; pas une nouvelle transmission complète du code. Modèle principal confirmé `claude-opus-5`, effort demandé `xhigh`, outils désactivés. Durée déclarée108698 ms; coût indicatif CLI0,160897 USD, dont0,151416 pour Opus, auxiliaire Haiku4.5 déclaré. Pas une preuve de facturation supplémentaire à Max.

Les réserves restantes sont dans `docs/retrieval_claude_architecture_review_assessment.md`.

---

# Contre‑examen — révision de mon avis du 10 septembre 2026

## A. Ce que je retire

**1. « Les expériences de génération ne sont pas additives. »** Retiré comme formulé. Tout top‑K est compétitif par construction : ajouter un candidat meilleur évince nécessairement le 2 000ᵉ, quelle que soit l'échelle du score. Un score invariant à la composition du pool ne rendrait ni la génération additive ni les évaluations « lisibles ». Ce qui reste, plus étroit, est écrit en B1. Et votre point tient : le delta net v29 (4 gains / 9 pertes) est une mesure réelle de la recette complète, vérifiée ; je ne peux pas la déclarer illisible.

**2. « Le LambdaMART SIRET n'existe pas. »** Faux à l'échelle du dépôt, et je le retire sans réserve. Vous citez `retrieval_hierarchical_admission_v12.py` : `_fit_ranker` (1415), `fit_site_ranker` (1493), `fit_one_outer_site_model` (1696), et `retrieval_top100_cascade_v12.py` documentant les 2 000 scorés. Ces fichiers n'étaient pas dans les pièces transmises ; j'ai converti « absent du manifeste v29 » en « inexistant ». L'énoncé correct est : *aucun scoring SIRET n'a été exécuté dans v29, et aucun résultat @500/@100 n'est disponible pour v23 ni v29*. C'est un trou de mesure, pas un trou d'architecture.

**3. « Site‑first serait une architecture nouvelle. »** Retiré. V23 possède déjà dix canaux SIRET directs — SIRET direct, nom courant/site/historique, adresse word/char/supporting/historique — qui alimentent les témoins directs et entrent dans le pool sans passer par une coupe parent. Mon affirmation que « 62 % des erreurs sont causées par parent‑first » est donc une erreur de logique : les 132 (v23) / 134 (v29) parents absents signifient que **les parents *et* leurs sites sont absents de l'union de toutes ces recherches**, pas qu'une étape parent les aurait bloqués. Réécrire l'index en site‑centrique ne crée aucun candidat que ces dix canaux n'atteignent déjà. Seule une *représentation* nouvelle pourrait le faire (C1).

**4. Le mode opérationnel « EXACT_ONLY_PENDING… ».** Retiré comme état courant : c'est le mode initial du contrat v12, historique ; un sidecar same‑site scellé existe depuis v20 et figure dans le rapport v29. Exact et opérationnel restent des métriques distinctes, mais elles ne sont plus nécessairement égales, et je n'ai pas à agiter un risque de substitution qui n'est pas dans vos pratiques actuelles.

**5. Le « pré‑score bon marché sur un pool plus large ».** Retiré. C'était bien le cap de 2 000 contourné sous un autre nom. Je maintiens la *critique* (B3) : le seul apprentissage SIRET est placé derrière un filtre non appris. Mais une revue ne peut pas récrire le budget, et aucune de mes propositions ne doit en dépendre.

**6. Ma borne et ma lecture statistique.** La borne v23 pertinente est **76** évictions, pas 75 (v24) ; en v29, 79. Et surtout : ces évictions ne peuvent en aucun cas être « rattrapées » par un modèle placé *après* la coupe 2 000 — mon jalon J1 les présentait implicitement comme telles, c'est incohérent. Enfin, je retire « −5 non informatif » : un test des signes nominalement non significatif ne transforme pas une baisse mesurée en absence d'information. La formulation juste est : −5 est un résultat réel et suffisant pour justifier le rejet de v29 ; il ne permet pas d'inférer un effet national.

**7. Ma dichotomie causale à deux bras.** Trop forte, vous avez raison. Deux bras sans refit ne permettent pas d'attribuer un échec à une inférence hors distribution : c'est une explication résiduelle parmi d'autres (qualité intrinsèque des parents v28, interaction avec les expansions, variance). Le test que je proposais était falsifiant dans un sens et spéculatif dans l'autre ; je le réécris en D comme test **unilatéral**. J'ajoute que je n'ai jamais démontré qu'un score parent pairwise fournirait une probabilité calibrée — un pairwise OOF donne un ordre, la calibration est une propriété séparée qu'il faudrait établir.

## B. Ce qui reste établi

**B1 — Instabilité de valeur, distincte de la compétition.** Le micro des six cas est solide et ne dépend d'aucun refit : `parent_score`, `direct_rrf` et le rang du site dans le parent sont *strictement inchangés*, et pourtant le score global change, parce que `learned_parent_rank` est un `dense_rank()` recalculé par requête. C'est un mécanisme identifiable : la valeur d'un candidat varie quand ses concurrents changent, en plus du déplacement du seuil. Les deux effets se cumulent ici ; ailleurs seul le second existerait. C'est un défaut réel, mais son ampleur end‑to‑end est **inconnue** — cinq des six cas ont `direct_rrf = 0`, ce qui suggère un régime de preuve faible, non représentatif.

**B2 — Le critère d'éligibilité v28 était intra‑canal.** « Gains parent > pertes parent, zéro perte contrôle » ne prédit pas l'issue au Master. Le résultat v29 le montre. Cela reste vrai indépendamment de toute théorie sur l'ordinalité.

**B3 — Une fraction des absences relève d'un signal manquant, pas d'un ordonnancement.** V25/V26/V27/V16 convergent : 44 des 132 hors de tout canal testé, zéro égalité de nom CRM dans les champs examinés, aucun parent à ≤ 100 m, 61 sans distance exploitable, 12 à ≤ 500 m dont aucun bloqué par une frontière de commune. Aucun tri ne crée cette preuve. En revanche, « existence du label ≠ identifiabilité » reste une hypothèse non quantifiée.

## C. Trois orientations, une seule nouvelle

**C1 — Représentation/preuve (la seule vraiment nouvelle).** Non pas « ajouter un canal de noms » — les dénominations courantes, historiques, site et RNE sont déjà indexées et corrélées entre elles ; les compter comme sources indépendantes serait de l'auto‑illusion. L'hypothèse nouvelle est une **unité indexée conjointe (site, période) portant simultanément dénominations rattachées et adresse normalisée dans un champ unique**, de sorte qu'un appariement *faible sur le nom* et *faible sur l'adresse* s'additionne à l'intérieur d'un même score de recherche, au lieu d'être fusionné après coup par RRF entre canaux qui, séparément, n'atteignent aucun top‑K. C'est précisément le régime que V25 n'a pas testé : le canal dur numéro+adresse (0 nouveau) testait la conjonction *forte×forte*. Risque majeur : si l'audit d'identifiabilité montre qu'un des deux termes est nul chez la plupart des 132, le mécanisme n'a rien à additionner.

**C2 — Sélection.** Rendre le terme parent invariant au pool (transformation monotone bornée de `parent_score` au lieu d'un rang). Gain attendu modeste et local : cela n'annule pas la compétition, cela supprime une source de variation de valeur.

**C3 — Génération.** Rien de nouveau ne me paraît justifié avant C1/mesure. Je n'inclus pas la planification apprise de requêtes : non testable honnêtement sur le panel 1034.

## D. Un seul protocole, conclusion limitée

**Objet :** la sélection (C2) uniquement, à génération strictement identique à v23.

**Design :** un run, deux conditions sur la même génération v23, modèles rechargés à l'identique, aucun refit, aucun changement de budget : (i) `rescue_score` actuel ; (ii) le terme `1/(60 + rang_parent + rang_site)` remplacé par une fonction bornée et préenregistrée de `parent_score` et de l'évidence de site déjà calculée, avec **une seule constante fixée avant lecture des résultats**. Métrique : exacts au Master@2000, et devenir des 76 cas évincés en v23.

**Conclusion possible :** si (ii) tombe sous 822, l'invariance de valeur ne porte pas de bénéfice net à ce budget — hypothèse C2 réfutée pour cette recette. Si (ii) tient ou progresse, on sait seulement que la substitution est *neutre ou favorable en v23*. **Ce test ne conclut rien sur v29**, dont l'échec ne peut être diagnostiqué sans refit ; l'inclure comme bras d'attribution serait le faux test que vous refusez à juste titre. Il ne teste ni C1 ni l'architecture site‑centrique.

**Données manquantes :** définir un test de C1 exigerait (a) l'audit d'identifiabilité label‑side scellé des absences — meilleure similarité de nom atteignable sur toutes les dénominations du SIREN, meilleure preuve d'adresse, jamais réinjectées en feature — et (b) une estimation du coût d'index (taille SSD, latence p95). Les deux manquent ; je ne peux pas spécifier ce test aujourd'hui.

## E. Jalons conditionnels

**J1 — Mesurer @500/@100 sur v23 telle quelle**, avec le ranker SIRET existant (`fit_site_ranker` / `fit_one_outer_site_model`), sous discipline OOF. Justification : c'est la métrique contractuelle, elle n'a jamais été publiée pour v23/v29, et elle ne dépend d'aucune réparation préalable. Énoncé explicitement : **cela ne récupère aucun candidat absent du Master** — ni les 132/134, ni les 76/79 évictions.

**J2 — Conditionnel au protocole D.** Si C2 est neutre/favorable : la branche géo‑parent v28/v29 ne peut être réévaluée qu'avec refit et inner OOF ; sans accord, elle reste abandonnée. Si C2 est réfutée : abandonner la piste sélection et concentrer l'effort sur la mesure d'identifiabilité.

**J3 — Identifiabilité + population fraîche**, dénominateur incluant les labels difficiles. C'est ce qui décide si C1 mérite un index, ou si l'écart résiduel est une limite d'information à renégocier — sans promesse de 99,2 %.

