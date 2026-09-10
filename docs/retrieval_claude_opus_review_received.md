# Avis Claude Opus 5 reçu — 10 septembre 2026

## Provenance et périmètre

Après reconnexion utilisateur, même dossier
`docs/retrieval_claude_opus_review_request.md`, même invocation
`claude -p --model claude-opus-5 --effort xhigh`, aucun fallback demandé.
Claude Code renvoie `is_error=false`, `terminal_reason=completed` et
`modelUsage.claude-opus-5` non vide. L'effort xhigh est le paramètre demandé;
le résultat ne fournit pas une attestation indépendante de l'effort effectif.
Le relecteur a reçu uniquement le dossier agrégé : aucun accès aux artefacts,
aucun outil, aucune exécution retrieval, aucune recherche web.

Durée déclarée :119987 ms. `modelUsage` attribue7730 tokens de sortie à
Opus5 et17 à un appel auxiliaire Haiku4.5; aucun changement de modèle
principal. Coût indicatif CLI total0,13949225 USD, dont0,13531725 pour Opus5;
ce compteur ne prouve pas une facturation supplémentaire à l'abonnement Max.
Le présent document est une **synthèse non verbatim** de l'avis reçu,
suivie de notre examen critique. Aucun résultat expérimental nouveau.

## Diagnostic du relecteur

1. **Mauvaise unité de décision pour v28.** Le rejet préenregistré reste
   correct au regard du critère choisi, mais exiger zéro perte dans un canal
   n'évalue pas son apport au système multicanal sous budget. Un remplacement
   peut perdre localement et gagner globalement. Revenir sur le critère après
   observation expose toutefois au choix opportuniste du critère : conserver
   le rejet v28, ne pas retuner et écrire un protocole end-to-end distinct.
2. **Proxy parent versus cible SIRET.** Un meilleur rang parent ne garantit
   pas la survie du bon site parmi2000. Les nouveaux sites peuvent évincer
   des succès. La calibration parent de v24 et le pilote parent v28 ne
   démontrent donc pas à eux seuls un gain exact Master.
3. **Mesurer la nouveauté réelle.** Comparer les SIRET exacts naturellement
   accessibles derrière la sélection v28 au réservoir initial pour distinguer
   nouveaux candidats accessibles et candidats déjà présents mieux remontés.
   Le relecteur propose cette intersection en lecture seule avant le run.
4. **Panel réutilisé.** L'OOF ne corrige ni la composition522 difficultés+
   512 contrôles, ni le choix adaptatif des recettes sur ce panel. Il demande
   les tailles des strates sources, le mode de tirage et l'historique des
   recettes essayées; aucune extrapolation nationale justifiée par le dossier.
5. **Budgets aval.** Un gain2000 ne prouve pas un gain100. Le relecteur propose
   de suivre500/100 avec des modèles gelés, à condition qu'ils n'aient pas vu
   les labels évalués. Cette proposition est soumise aux réserves ci-dessous.

## Expérience proposée par Claude — pas encore approuvée ni exécutée

- Référence v23, pas la fusion v24 (+1 net et calibration sur objectif parent).
- Remplacer la sélection géo par v28 figée; autres canaux inchangés,
  expansion naturelle et règle site/parent déterminée avant le run.
- Fusion par rangs compatible avec v23, sélection globale≤2000, aucune
  protection ancien/nouveau ni poids scalaire recalibré. Aucun apprentissage
  supplémentaire, aucun LambdaMART SIRET sur le grand réservoir.
- Comparaison appariée SIRET humain exact@2000, gains/pertes par fold et
  difficultés/contrôles; vue opérationnelle séparée. Claude propose un test
  exact de McNemar bilatéral, net positif et p<0,05, avec garde-fou contrôles.
- Une seule recette, pas de variantes retunées après résultat. En cas
  d'échec, se concentrer sur les évictions Master plutôt que multiplier les
  variantes géographiques sur les mêmes labels.

Claude suggère, à titre d'exemples, au moins15 exacts réellement nouveaux
pour justifier l'essai et au plus5 pertes contrôles. **Ces seuils ne sont
ni issus d'une preuve, ni adoptés, ni préenregistrés par le projet.**

## Ancien/nouveau : réponse explicite

Le relecteur considère v22 comme l'ablation la plus nette du dossier : mêmes
scores et même réservoir, seule la protection1750+250 disparaît, net+45.
Il recommande de conserver le tri global. La provenance initiale ne suffit
pas à justifier une priorité. V24 ne justifie pas de relancer une recherche
de poids scalaire parent transféré aux sites. Attention : ces résultats ne
prouvent pas qu'aucune politique de diversité ni aucun poids global ne puisse
jamais être utile sur une autre population.

Le relecteur privilégie les75 évictions v24 comme gisement immédiatement
mesurable sans nouvelle source. Les136 absences naturelles exigent une
amélioration de génération : un tri seul ne peut les corriger. Il ne promet
ni d'atteindre ces plafonds, ni99,2%, ni une généralisation cross-CRM.

## Examen critique par Codex et décision

**À retenir :** mesurer le SIRET exact sous budget global, distinguer
génération et admission, garder v23 comme référence, ne pas réintroduire
la réservation ancien/nouveau et ne pas retuner v28. Une intersection de
réservoirs puis un protocole end-to-end borné constituent une orientation
cohérente; ce n'est pas encore une validation ou une autorisation de run.

**À corriger avant tout protocole :**

- Un critère zéro perte n'est pas mathématiquement impossible pour un modèle
  appris, contrairement à une affirmation trop absolue de l'avis. Il reste
  obligatoire pour le verdict du pilote v28; aucun succès rétroactif.
- Le différentiel389→424 est un **net35**, pas un ensemble de35 gains.
  Il faut reconstruire les ensembles gains/pertes et les intersections par
  requête/SIRET, sans traiter le net comme un nombre d'éléments nouveaux.
- Les cinq exacts accessibles derrière les sept nouveaux parents rendent
  un seuil arbitraire de15 particulièrement peu motivé. Un petit gain réel
  peut être utile; son intérêt dépend aussi des pertes et des coûts. Aucune
  liste de labels auditée ne doit alimenter la génération ou la sélection.
- Les22 pertes géo présentes dans le réservoir ne prouvent pas que les
  **gains** géo y soient présents. Et une absence de nouveauté ne condamne
  pas un reclassement utile sous2000. L'intersection répond à une question
  précise, pas à la valeur finale du canal.
- Un McNemar nominal sur un panel sélectionné et réutilisé ne constitue pas
  un gate national confirmatoire. Le déséquilibre des strates n'implique pas
  que tout gain net soit nécessairement optimiste : cela dépend des gains
  et pertes par strate. Une pondération n'autorise pas un Wilson binomial
  ordinaire; l'évaluation nationale prévue reste distincte.
- L'avis ne connaît que le dossier : les tailles des strates ou modalités
  de tirage peuvent exister ailleurs. « Non fourni » n'est pas « inexistant ».
-500/100 n'est pas une mesure gratuite ni automatiquement admissible avec
  les modèles historiques gelés. Respecter les gates, l'OOF et la cascade
  active avant de revendiquer une mesure aval; ne pas dégeler ranker,
  decider, risk model ou accepteur pour suivre ce conseil.
- Un échec d'une recette ne prouve pas que toute géographie soit inutile.
  Il motive l'arrêt de cette recette, sans conclusion universelle.

**État final : consultation terminée; `PIVOT_GENERATION` inchangé.** Aucun
nouveau run, apprentissage, réglage, Master2000 ou métrique500/100 produit.
Folds0/1 fermés. L'avis apporte une critique externe, pas une preuve de recall.
