# V28 — sélection OOF des voisins avant coupe512

Préenregistrement, 10 septembre 2026. Un pilote de génération, sans activation
produit. Même panel1034 folds2/3/4, mêmes512 contrôles et522 difficultés.
Une seule recette; aucun hyperparamètre, seuil ou feature choisi sur résultats
externes. Folds0/1 fermés, labels intacts. V23 demeure référence de recherche.

## Expérience bornée

1. Générer sans labels les4096 sites géographiques v16 inchangés (BAN CRM,
   même commune, même stock et même ordre). Reproduire les512 parents v16.
   Vérifier les SIREN avec le champ officiel, sans dériver un nouveau parent
   d'un identifiant. Aucun site ajouté et aucune nouvelle source géographique.
2. Calculer des références légères parent/site : similarités des noms SIRENE
   courants autorisés (UL cinq champs, sites quatre), voie/numéro/CP, état,
   distance/rang/nombre de sites et qualité BAN v16. Pas de RNE ambigu ni nom
   de personne; les noms de site ne migrent pas vers des sites frères pour
   fabriquer une concordance. Le produit nom×voie est calculé sur le même
   témoin, sans numéro obligatoire. La distance est un signal, pas un rejet.
3. Vérifier le plafond naturel : requêtes avec un parent correct dans ce
   pool et groupes contenant des négatifs. Si aucun groupe entraînable ou
   aucune variation des indices nom/adresse, STOP. Ce n'est pas une recherche
   de seuils sur les labels; aucune autre recette ne sera essayée.
4. Trois fits parent XGBRanker outer OOF2/3/4, hyperparamètres et sampler
   v23 inchangés. Positifs uniquement déjà présents dans le pool; échantillon
   d'entraînement positif naturel+256 négatifs de tête+256 stratifiés de queue.
   Les groupes sans positif sont non entraînables mais restent au dénominateur
   d'évaluation. Vérifier la séparation des SIREN humains entre train/test.
5. Scorer chaque parent du pool dans son fold tenu à l'écart; garder512
   parents maximum par score puis ancien rang spatial et SIREN en ex aequo.
   Aucune quota protectrice, fusion, repondération ni retuning après résultat.

## Ce que le pilote mesure, et ce qu'il ne mesure pas

Comparer au canal géo v16 : parents retenus, gains/pertes nets, plafond naturel
et segments complets, dont132 parents absents des preuves et les12 proches.
Publier séparément la présence du SIRET exact et des équivalents opérationnels
dans les sites naturels derrière les parents retenus : **plafond d'expansion**,
pas Recall SIRET@512, pas Master2000. Aucun SIRET absent du pool4096 n'est
ajouté pour l'entraînement ou l'évaluation. Les2 proches exclus avant4096
restent impossibles à récupérer par ce pilote de sélection.

Les4096 sont le pool expérimental v16 déjà audité, non une nouvelle limite
Master : aucune sortie produit de4096, aucun LambdaMART SIRET sur ce pool,
aucune hydratation riche. Caps produit2000/500/100 et modèles aval restent
gelés. Aucune réinjection des nouvelles sélections dans le Master pendant
ce pilote. BAN score toujours inéligible à l'activation produit.

Avant toute intégration : résultat net positif sur ce canal, absence de perte
sur les512 contrôles, puis mesure end-to-end distincte à budget Master2000.
Un échec à ces critères rejette la recette sans second essai. Un succès ne
donne ni READY_FOR_FOLD1 ni preuve99,2% : le panel est réutilisé et une vraie
validation indépendante demeure nécessaire. Aucun national/projection.

## Exécution et preuve

Préflight technique≤16, puis code/config committés avant complet. Mac/CATNAT,
DuckDB8 Go/deux threads pour le tri géo, budget par étape30 min/RSS14 Gio/zéro
swapout; modèles séquentiels. Hashes sources/code/config/runtime/résultats,
split manifests et scores OOF conservés. Aucun X/Y brut exporté. Les modèles
parents de ce pilote ne remplacent pas le ranker/decider/risk/accepteur gelés.

Historique, V2/V3, folds, origines, états et étapes de perte publiés; compteurs
Master exact/opérationnel antérieurs inchangés823/825. Sidecars officiels
scellés obligatoires pour les vues de labels; absence/incohérence=STOP.
Verdict final PIVOT_GENERATION (ou STOP intégrité/ressources). Milestones et
commits isolés consignés dans handover.md.
