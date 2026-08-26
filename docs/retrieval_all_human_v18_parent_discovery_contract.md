# Contrat v18 — découverte parent SIRET-first, développement uniquement

Statut : préenregistrement de développement. Ce protocole ne constitue ni une
validation finale, ni une autorisation de run national.

## Question testée

V17 retrouve naturellement le bon SIREN pour 329 des 522 pertes Master. Son
pool sélectionné n'en conserve que 309. L'étape v18 teste exclusivement si des
preuves structurelles, applicables à tout CRM français, portent l'union brute
à 405 parents au moins, niveau qui rend Master@2000 >=99,5 %
mathématiquement possible.

Les folds 2/3/4 et les 1 034 requêtes sont désormais considérés comme un jeu
de développement. Les choix humains ayant été inspirés par les erreurs v15 à
v17, aucun score v18 sur cette cohorte ne prouve une généralisation inter-CRM.
Fold0 et fold1 restent fermés.

## Génération sans vérité

Le générateur reçoit seulement le fichier scellé `query_id,fold`, SceneIR et
les autorités officielles listées dans
`config/retrieval_all_human_v18_parent_discovery_generation.json`. Il ne peut
ouvrir ni `cohort_labels`, ni le label ledger, ni le type de cohorte.

Les familles gelées avant exécution sont :

1. `v17_union`, autorité de base monotone ;
2. tous les SIRET naturels déjà présents dans le pool v11, regroupés par leur
   champ **SIREN officiel stocké** ; aucun `siret[:9]` n'est calculé ;
3. dix recherches SIRET-native à profondeur 512 sur les noms de site,
   enseignes, noms historiques/supporting et adresses courantes/historiques ;
4. une relation RNE dirigée `subject_siren -> related_siren`, bornée à partir
   des huit premiers parents v17 ;
5. les cessions BODACC SIREN, un saut, depuis 32 seeds ;
6. les successions SIRET dans les deux sens, uniquement lorsque
   `economic_continuity=true`, depuis les 250 premiers SIRET naturels.

SIREN et SIRET ne sont que des clés de jointure et de déduplication. Ils ne
sont pas des features. Aucun dictionnaire d'alias CRM, aucune règle propre à
une requête, aucun positif injecté et aucun modèle ne sont autorisés.

La sortie contient seulement deux Parquet de signaux parent. Aucun parent
n'est encore sélectionné à 512 et aucun inventaire complet de sites n'est
développé.

## Évaluation truth-side séparée

Après scellement de la génération, un autre processus ouvre les labels et
publie les ablations suivantes : v17, v17 + ancien pool SIRET, v17 + relations,
SIRET-native seul et union v18 complète. Les métriques sont le nombre de bons
parents parmi les 522 pertes, par fold et par origine CRM, ainsi que les 512
contrôles.

Les seuils exacts, pour 23 571 requêtes et 23 049 hits Master scellés, sont :

- 334 rescues : Master@2000 >=99,2 % ;
- 405 rescues : Master@2000 >=99,5 % ;
- 452 rescues : Master@2000 >=99,7 % ;
- niveau de confiance historique : 464 et 160/158/146 par fold.

Sous 405, aucun LambdaMART, aucune expansion massive de sites et aucun run
national ne sont autorisés. Entre 405 et 451, seule une sélection parent OOF
bornée peut être préenregistrée. À partir de 452, le projet dispose d'une
marge crédible avant les futures réductions 2000 -> 500 -> 100.

Les deux origines `REAL_CRM_20260817` et `REAL_CRM_HISTORICAL` sont toujours
publiées séparément. Une baisse de l'une d'elles ne peut être masquée par la
moyenne globale.

## Portée de généralisation

Les folds sont groupés par SIREN et testent donc de nouvelles entités dans la
même famille CRM. Ils ne constituent ni un split temporel, ni un split client,
ni un autre CRM. Après gel complet de la future recette :

1. le test final same-domain sera exécuté une seule fois selon le contrat
   sélectif actif ;
2. toute affirmation cross-CRM exigera un export futur ou un portefeuille
   distinct, conservé comme shadow test sans retuning.

## Ressources et arrêts

Le préflight est limité à 1..16 requêtes. Le pilote complet est limité à 30
minutes, 14 Gio de RSS et zéro page swapout. Les hashes d'autorité, les 100
shards de succession, les caps et les familles sont validés avant ou pendant
la génération. Toute divergence échoue fermée.
