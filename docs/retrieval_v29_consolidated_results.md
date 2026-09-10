# V29 — essai consolidé terminé : régression nette, v23 conservée

10 septembre 2026. Code/config/contrat figés avant complet : `74588c2f`;
handover de lancement : `8f72d5ba`. Une recette, aucune variante ni refit.

## Résultat exact à budget2000

| Population | Requêtes | v23 exacts | v29 exacts | Gains | Pertes | Net |
|---|---:|---:|---:|---:|---:|---:|
| Ensemble du panel |1034|822|817|4|9|-5|
| Difficultés initiales |522|310|305|4|9|-5|
| Contrôles |512|512|512|0|0|0|
| Fold2 |353|296|293|1|4|-3|
| Fold3 |346|268|265|1|4|-3|
| Fold4 |335|258|259|2|1|+1|

Opérationnel séparé : **824→820**, quatre gains/huit pertes, net−4.
Trois hits v29 uniquement opérationnels;25 requêtes avec successeur actif
acceptable présent et21 avec plusieurs équivalents actifs présents.
Ce sont des présences parmi les candidats, pas des prédictions top1.
Le SIRET exact absent du catalogue reste une erreur exacte et opérationnelle.

**Décision : `NO_NET_GAIN_REUSED_PANEL` / `PIVOT_GENERATION`.** Ne pas
promouvoir cette intégration. V23 reste la référence de recherche; v24
823/825 demeure un résultat antérieur non promu. Aucun gain produit/top100.

## Ce que le run tranche

Les cinq contrôles perdus dans le seul canal géo v28 sont tous conservés
au Master v29. Le rejet intra-canal ne prédisait donc pas leur perte
end-to-end. En revanche, le résultat complet reste négatif à cause des cas
difficiles. Le verdict historique de v28 n'est pas réécrit.

Le nouveau réservoir contient **896 exacts contre898** : sept nouveaux
accessibles, neuf anciens devenus absents. Parmi les sept nouveaux exacts,
**quatre survivent au top2000**, trois sont évincés. Les quatre gains Master
sont tous de vraies nouveautés de réservoir, pas simplement un meilleur tri
des anciennes références.

Sur les neuf succès v23 perdus au Master :

- trois ont perdu leur parent à la génération;
- six restent naturellement accessibles mais sont évincés au top2000.

La présence dans le réservoir ne suffit donc pas. Ajouter des candidats
modifie aussi le classement et peut déplacer des succès existants.

Le pool spatial derrière les parents v28 contient424 exacts; **six** sont
absents de l'ancien réservoir multicanal. L'expansion locale complète vers
les établissements donne sept exacts nouveaux au total. Le net389→424 du
pilote géo n'était pas35 nouveaux candidats multicanaux.

## Construction effectivement testée

Remplacement de l'appartenance géo dans `v17_union` par la sélection512 v28;
parents v17 texte/adresse conservés. Autres signaux v18 inchangés, y compris
relations déjà générées : pas de recalcul de leurs seeds. Aucun bonus de
distance, aucun nouveau poids. Le rang v28 sert seulement à sa coupe512.

Expansion de tous les sites soutenus dans le même INSEE ou CP CRM, puis
features/règle v23 et modèles outer gelés. Pas de filtre dur de numéro à
l'expansion; la feature conjointe v23 conditionnée par numéro reste inchangée.
Les noms officiels restent liés à leur sujet/site. Aucun alias CRM, positif
injecté, nom personnel ou identifiant comme feature. Les identifiants servent
uniquement aux jointures, dédoublonnages, groupes et ex aequo.

Réservoir expérimental :3900144→3929577 paires requête/site;
346704 paires ajoutées,317271 retirées. Ces mouvements ne sont pas des gains
de recall. Master :1973000 références, maximum **2000 par requête**;
115430 références admises/115602 déplacées par rapport au Master v23.
432443 parents sélectionnés par v28, jamais plus512 par requête. Aucun
LambdaMART SIRET sur le grand réservoir, aucune hydratation riche500/100.

Attribution exhaustive des217 erreurs v29 :134 parents absents,
trois sites non développés,79 évictions au Master, une absence catalogue.
Le drapeau parent courant est recalculé sur les sources v29; les faits
d'existence officiels viennent du sidecar label-side scellé inchangé.

## Vérification et ressources

Préflight16 sans labels puis replay complet exact des candidats/témoins v21,
scores parent et Master v23 avant remplacement. Génération et évaluation
séparées; aucun label cible lu pour générer/sélectionner.

`scripts/audit_retrieval_v29_results.py` retourne
**`PASS_V29_OUTPUT_RECONCILIATION`** : hashes, population1034 réconciliée
avec le ledger filtré2/3/4, splits OOF et séparation SIREN humains, remplacement
des appartenances, égalité exacte des témoins, expansion naturelle,
prédictions des trois modèles rejouées exactement, sélection2000 recalculée,
égalité préflight/complet et métriques exactes/opérationnelles/pertes/segments
recalculées indépendamment. **49 tests passent**, couvrant aussi les doublons, folds
interdits, plafonds, absence de fit et détection d'un faux gain de bilan.

| Étape | Temps sous garde-fou | Pic RSS | Swapout |
|---|---:|---:|---:|
| Génération complète avec replay |103,54 s|9,39 Gio|0|
| Évaluation |1,21 s|2,03 Gio|0|
| Vérificateur |29,68 s|8,18 Gio|0|

Hashes préalables hors chronométrage; ces durées de recherche ne sont pas
des SLO produit. Mac/CATNAT uniquement, zéro nouveau fit ou achat externe.

## Scellement et suite

Config SHA `134f18bd1bae893adcea918ad7819ec4df87369d5998140935d29df94a14667c`.
Sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/` :

- `retrieval_v29_consolidated/134f18bd1bae893a/manifest.json`, SHA
  `c6b828eb825a5dd1b6432b38835e8f970ff3edd589e919ca828dd9b2802ebbc2`;
- `retrieval_v29_evaluation/134f18bd1bae893a/manifest.json`, SHA
  `4e1eb661acce89825998cff0688806cd04568eabdd988523f35da20089e501f9`;
- `retrieval_v29_verification/134f18bd1bae893a/manifest.json`, vérification
  scellée avec empreinte de son implémentation.

Pas de recette corrigée après résultat, ni nouvelle variante géo lancée.
Orientation suivante proposée, non exécutée : travailler l'admission des
références déjà disponibles dans le réservoir v23, sans confondre ce progrès
possible avec la récupération des absences de génération. Un éventuel fit
SIRET devra respecter le plafond2000 et le contrat OOF; pas de modèle SIRET
appris sur les millions de paires de recherche pour contourner ce plafond.

Panel522 difficultés+512 contrôles réutilisé : les proportions ne sont pas
le recall national, ni une validation indépendante. Historique/V2/V3 et
origines publiés dans le manifeste; aucun retrait du dénominateur. Aucun
PreRank500, Recall100, Wilson national, SLO ou généralisation revendiqué.
Folds0/1 fermés, modèles aval gelés, aucune autorisation de fold1.
