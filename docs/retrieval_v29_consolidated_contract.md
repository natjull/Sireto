# V29 — essai consolidé v23 + sélection géographique v28

Préenregistrement du 10 septembre 2026 après accord utilisateur « Go ».
Une recette sur les1034 requêtes de développement, folds2/3/4. Aucun national,
fold0/1, entraînement, poids calibré, grille ni activation produit.

## Intégration exacte et limites

V23 reçoit l'ancien canal géo via l'appartenance `v17_union`, dont le rang
arbitraire par SIREN ne contribue pas au RRF. Remplacer cette appartenance
par l'union des parents v17 marqués `from_text` ou `from_address` et des512
parents retenus par les scores outer-OOF v28 déjà scellés. Ne pas conserver
un parent exclusivement géo ancien absent de la nouvelle liste, sauf s'il
est aussi retrouvé par un autre canal conservé. Pas d'union des deux listes
géo, pas de bonus géographique ou de nouveau poids. Le rang v28 sert à la
coupe512, pas de feature supplémentaire pour le modèle v23.

Les autres signaux v18, dont relations officielles déjà générées, restent
bit à bit inchangés : leurs seeds ne sont pas recalculés. Aucun nouveau
parcours de relation. Des anciennes preuves géographiques indirectes dans
ces sources restent possibles; il ne s'agit pas d'une purge de toute origine
géographique historique.

Reconstruction du réservoir par la règle v21 existante : preuves SIRET
directes + tous les sites des parents soutenus qui sont dans le même INSEE
ou CP que le CRM. Pas de filtre dur de numéro ni de distance à l'expansion,
pas de plafond arbitraire de sites par parent. Noms officiels et features v23
inchangés, modèles outer v23 rechargés sans fit. Le produit nom×voie v28
sans numéro obligatoire agit sur le choix géo; la feature conjointe v23,
qui contient une condition de numéro, demeure inchangée dans cet essai.

Recalculer les16 features parent v23 sur le nouveau réservoir, scorer chaque
parent avec le modèle de son fold externe, puis appliquer exactement
`direct_rrf + 1/(60+rang_parent+rang_site_conjoint)` aux sites locaux.
Top2000 global, mêmes ex aequo v23. Aucune réservation ancien/nouveau.
Le modèle parent n'est pas réentraîné sur les nouveaux signaux : un éventuel
décalage de distribution est une limite explicite de ce test d'intégration.

Pas de v24 (+1 non promu), pas de faux correctif issu des audits v25–27.
V28 reste historiquement rejetée au gate intra-canal; ce test end-to-end
distinct ne modifie ni ses résultats ni son verdict rétroactivement.

## Cloisonnement et preuve avant mesure

Génération et évaluation sont deux commandes séparées. Génération lit
uniquement champs CRM autorisés, sources officielles et scores/modèles OOF;
aucun label, cohorte, hit ou liste de cas auditée. Les modèles v23 et v28
excluent les labels du fold cible. Aucun consommateur appris n'est entraîné
sur les nouvelles sorties, donc pas de nouveau problème de stacking inner OOF.

Rejouer techniquement le réservoir v21 original et comparer ses références,
scores et témoins aux artefacts scellés avant remplacement. Sur ce même
réservoir, recharger les features/modèles v23 et vérifier les scores OOF et
le Master de référence. Ce replay n'est pas une variante expérimentale.
Préflight≤16 sans labels, puis code/config committés avant génération complète.
Hashes de sources, code, config, modèles et résultats; splits contrôlés.

Le grand réservoir reste un outil de recherche label-blind; aucun LambdaMART
SIRET ne le score. Seul le Master≤2000 constitue la sortie de génération
mesurée. Aucune hydratation riche ni sortie500/100, modèles aval gelés.

## Mesure et décision préenregistrées

Primaire : SIRET humain exact@2000 comparé à v23, tous1034 labels conservés.
Publier gains, pertes, net, par fold, difficultés/contrôles, origine, état,
historique, V2/V3. Vue opérationnelle séparée via sidecar same-site scellé.
Aucune réécriture de label; l'absence officielle reste une erreur.

Dans le même bilan : présence exacte avant/après dans les réservoirs,
nouveautés et pertes de génération, nouveaux candidats et évictions Master,
intersection des candidats géo avec le réservoir initial. Reconstruire les
ensembles appariés; ne pas prendre le net389→424 pour35 nouveaux éléments.
Réattribuer les échecs à absence catalogue, parent absent, site non développé
ou éviction au Master à partir du **nouveau** réservoir. Ne pas réutiliser
le drapeau historique `parent_in_evidence` comme présence courante.

Statut descriptif `NET_GAIN_REUSED_PANEL` si gains>pertes, sinon
`NO_NET_GAIN_REUSED_PANEL`. Aucun seuil choisi après lecture, aucune promotion
automatique même si net positif; pertes contrôles et par fold publiées sans
les masquer. Aucun essai corrigé/retuné après résultat. Verdict scientifique
`PIVOT_GENERATION`, ou `STOP` si intégrité/sidecars/ressources manquants.
Pas de Wilson national, pas de preuve99,2% ou de généralisation : panel
522 échecs+512 contrôles réutilisé, nouvelle recette choisie après audits.

Mac/CATNAT uniquement, par étape≤30 min, RSS≤14 Gio, zéro swapout;
DuckDB8 Go/deux threads, prédictions parents séquentielles. Temps inclut
les opérations de l'étape sous garde-fou mais hashes préalables séparés.
Un vérificateur distinct recalcule populations, sélections et métriques.
