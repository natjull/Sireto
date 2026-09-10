# Rejeu complet — avis Claude et arbitrage critique

10 septembre 2026. Consultation demandée par l'utilisateur, avec code et
documents sélectionnés. Aucun run retrieval, fit ou modification de recette
n'a été exécuté pendant cette consultation.

## Conclusion retenue

Mesurer le système gelé sur toute la population CRM de développement admissible
est la prochaine mesure utile. Elle exige deux corrections distinctes : un
dénominateur non sélectionné selon les succès/échecs précédents, et une preuve
que chaque candidat est reconstructible depuis l'entrée CRM et les sources
officielles accessibles au système. Agrandir le panel ne prouve pas la seconde.

Claude recommande une comparaison appariée v23/v30, avec modèles parent gelés,
dans un seul travail partageant les calculs communs. Cela mesure les recettes
existantes; un refit sur toute la population serait une expérience différente.
L'avis ne justifie pas de réentraîner maintenant ni d'ouvrir le fold 1.

## Constats vérifiables dans le code fourni

- `scripts/retrieval_v30_joint_search.py:discover` impose 1 034 identités,
  et reconstruit sa requête depuis une SceneIR existante. `generate` recharge
  les anciens signaux et candidats. Seul le nouveau canal est recherché de
  nouveau pour cette expérience; ce n'est pas le rejeu de toute la chaîne.
- `_load_retrieval_queries` dans le runner v18 projette des champs CRM mais
  aussi `postcode` et `insee` d'une première ligne SceneIR. Leur provenance et
  les règles de fallback devront être suivies jusqu'à l'entrée brute. La seule
  absence de colonne GT dans cette lecture ne démontre pas la disponibilité
  de tous les champs à l'arrivée. Ce constat n'établit pas une fuite avérée.
- `predict` dans v29, réutilisé par v30, refuse les identités absentes de
  `target_query_ids`. Étendre le périmètre nécessite un nouveau manifeste
  d'inférence et des preuves d'isolation, pas la suppression de cette garde.
- Les modèles v23 sont outer-OOF sur le panel enrichi. Les garder fixes isole
  la mesure de ces modèles; leur biais d'entraînement ne disparaît pas.

Ces limites sont établies pour les scripts fournis, pas une preuve qu'aucun
runner ou composant réutilisable n'existe ailleurs dans le dépôt.

## Protocole recommandé, pas encore implémenté ni exécuté

1. Recompter et sceller tous les labels humains exploitables des folds 2/3/4
   depuis le ledger. Publier les absences de label et tous les segments.
   Vérifier les SIREN d'entraînement de chaque modèle contre son nouveau
   périmètre d'évaluation. Une violation impose STOP et correction du protocole,
   jamais le retrait de la ligne du dénominateur. Le mapping de modèle par fold
   appartient au dispositif OOF, pas aux features ou branchements du retrieval.
2. Construire ou réutiliser un chemin entrée CRM → prétraitement → tous les
   canaux → expansion → features → Master2000, sans lookup de candidats par
   identifiant CRM. Index/catalogues officiels scellés autorisés. Les caches
   de prétraitements ne sont acceptables que si leur clé, leur contenu et leur
   constructeur sont reproductibles depuis les entrées autorisées, sans label.
   Sceller les sources et clarifier le snapshot temporel : ce rejeu ne prouve
   pas à lui seul la performance historique ou sur de futures entrées.
3. Avant un calcul long, préflight technique court sans choix de recette :
   parité sémantique avec les artefacts gelés sur des entrées du panel,
   indépendance à l'ordre/aux shards/au nombre de workers, plafonds, budget
   mémoire et coût de tous les canaux. Ne pas exiger une identité des octets
   Parquet si seuls l'encodage ou les métadonnées changent; comparer identités,
   rangs, scores selon une règle de précision fixée à l'avance.
4. Un rejeu apparié v23/v30 par lots, mêmes entrées et modèles gelés, calculs
   communs partagés. Budget, erreurs/timeouts et règles de reprise figés avant
   résultats. Conserver chaque requête au dénominateur; publier séparément
   complétude d'exécution et hits des sorties réellement émises. Une panne
   bloquante rend le run incomplet/STOP, pas une autorisation d'omettre les lignes.
   Ne pas forcer une erreur exacte si un canal échoue mais que la sortie
   contractuellement valide contient tout de même le bon SIRET.
5. Bilan exact et opérationnel séparés, N par fold/origine/état/V2/V3,
   gains/pertes appariés, attribution officielle exhaustive, caps, temps et
   ressources. Distinguer panel ancien et reste de la population pour le
   diagnostic sans choisir rétrospectivement le dénominateur principal.

Les runs scellés v23/v30 restent immuables. Un futur runner complet et son
contrat seront distincts; ni ce document ni l'avis brut ne constituent un
gel suffisant de l'implémentation ou une autorisation de lancer un long calcul.

## Affirmations de Claude non retenues

- « Ce n'est ni un recall » est faux : 829/1 034 est bien un recall exact au
  Master2000 sur ce panel, pas le recall de production. « Plancher stratifié »
  n'est pas démontré. Sa prévision 93–97 % est sans fondement et non reprise.
- Exclure d'éventuelles violations d'isolation de la métrique principale
  contredit le contrat all-human : elles doivent bloquer la validité OOF,
  pas améliorer artificiellement le score publié.
- « Quelques pourcents » de surcoût, impossibilité mémoire du monolithe,
  absence totale de puissance des 512 contrôles et latence « à froid » ne sont
  pas établies. Le coût et le régime de cache doivent être mesurés. Le batching
  est une précaution raisonnable, pas une preuve d'impossibilité autrement.
- Une hausse du taux sur une population différente ne mesure pas un progrès
  technique. Le delta v23/v30 apparié mesure un effet sur cette population,
  mais le recall absolu y est aussi une information utile, pas un chiffre vide.
- Master sous 99,5 % n'est pas un nouveau gate autonome permettant d'interdire
  toute mesure500/100 : le contrat top100-first reste inchangé. Ce bilan est
  limité au2000 à la demande actuelle, et ne vaut jamais READY_FOR_FOLD1.
- Le réservoir v30 n'est pas la sortie Master : 900 exacts avant coupe et829
  après coupe doivent rester distincts malgré le vocabulaire imprécis de l'avis.

## Ce qu'on pourra conclure

Un rejeu fidèle donnera la couverture des recettes gelées sur l'ensemble
évaluable du CRM de développement, avec leurs régressions et coûts. Ce ne sera
ni une validation indépendante, ni une preuve cross-CRM, ni le pipeline aval
complet. Les folds 0/1 restent fermés, les modèles aval gelés et l'objectif
exact99,2%@100 inchangé. Verdict actuel PIVOT_GENERATION; aucun nouveau score.
