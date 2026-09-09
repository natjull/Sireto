# V23 — parent appris OOF et preuve conjointe nom/site

Préenregistrement du 9 septembre 2026, avant résultats. Folds2/3/4 seulement.

## Recette unique

Le réservoir v21 de 3 900 144 références exploratoires est inchangé; aucune
injection de vérité ni nouvelle recherche. Comme v22, aucune place réservée
au préfixe initial. Le plafond transmis au futur LambdaMART SIRET est 2000.

Construire les preuves depuis les noms officiels du portfolio local scellé :
LEGAL_CURRENT et TRADE_CURRENT attachés au SIREN, SITE_CURRENT attaché au
SIRET. Champs `selected_for_subject`, aucun alias appris du CRM, aucun nom
de dirigeant ou texte privé. Les noms historiques/supporting sont exclus de
cette recette fixe; cela peut limiter les dossiers anciens, sans les retirer
du dénominateur. Le nom CRM vient uniquement du champ CRM de la scène raw,
identique dans ses lignes, et non des labels ou qualifications.

Calculer Jaro-Winkler et Jaccard de tokens séparément. La preuve conjointe
d'un site est `max(similarité nom parent, similarité nom propre du site)` ×
similarité de sa voie × accord numéro/suffixe lorsque le numéro CRM existe.
Un numéro absent laisse ce facteur inconnu/neutre, avec feature de manque
explicite. L'agrégation parent prend le maximum des preuves conjointes de ses
sites locaux : jamais le nom d'un sibling multiplié par l'adresse d'un autre.
Un nom légal de l'entreprise est en revanche valide pour ses établissements.

Les 16 features du modèle parent sont figées dans `FEATURES` : similarités
nom, meilleure preuve conjointe, accords adresse, soutien parent/direct v21,
nombre de sites logarithmique et indicateurs de manque. Aucun SIREN, SIRET,
query_id, fold, qualification, origine ou rang v17 dérivé d'identifiant.

LambdaMART **parent** : paramètres v19 fixes, threads4 pour le budget local,
aucune grille, aucun early stopping sur un fold externe. Négatifs naturels :
256 premiers +256 représentants stratifiés de queue, même correction v20.
Les groupes sans parent positif naturel ne permettent pas un apprentissage
pairwise; leur nombre est publié et leurs requêtes restent erreurs end-to-end.

Trois modèles externes F2←F3+F4, F3←F2+F4, F4←F2+F3; séparation des SIREN
humains vérifiée. Trois modèles supplémentaires entraînés sur un fold chacun
produisent les prédictions inner-OOF utilisables pour les folds d'apprentissage
des modèles externes. Ces scores internes ne choisissent aucune recette et
ne sont pas utilisés pour produire le Master externe. Les modèles, splits,
ordre des features, sources et prédictions sont scellés.

Le rang parent pour une requête vient uniquement de son modèle externe,
sans son label. Le rang site est déterministe : preuve conjointe tokens puis
Jaro, accord strict adresse, INSEE, numéro, suffixe, voie et actif. Ex-aequo
de preuve = même rang. Le score de sélection conserve la forme précédente :
`direct_rrf + 1/(60 + rang_parent_OOF + rang_site_conjoint)` pour un site
local; un témoin direct hors géographie garde son score direct. Top2000
global, départage par direct_rrf puis score parent puis SIRET terminal.
**Aucun modèle SIRET ne score le grand réservoir.** Pas d'hydratation riche.

## Séparation des phases et limites

1. Préflight technique sans labels (1..16), puis extraction complète sans labels.
2. Entraînement OOF fixe et génération des Masters. Chaque fit lit seulement
   les labels des folds d'entraînement; les labels du fold cible ne servent
   ni au fit, ni au score, ni à la sélection. Les métadonnées de séparation
   GT_SIREN restent dans le contrôle de splits, pas dans les features.
3. Évaluation séparée contre v22 : gains ET pertes, contrôles compris,
   exact/opérationnel, historique/V2/V3, origines/folds/statuts/états. Sidecar
   officiel et ensemble same-site v20/v21 réutilisés, avec leurs limites.

Le code manipule le ledger pour assembler les datasets d'entraînement OOF;
il n'est donc pas décrit comme totalement label-blind. Seule l'extraction
des features l'est. L'absence de labels cibles dans les fits est une propriété
testée et scellée. Les indicateurs de succès sont calculés après le Master.

Le panel a été réutilisé dans plusieurs cycles : OOF empêche une fuite de fit,
mais n'efface pas l'adaptation de stratégie. Aucune preuve cross-CRM, aucun
nouveau découpage prétendument vierge. Pas de projection nationale, le préfixe
n'étant plus garanti. Les absences et qualifications ne filtrent rien.

Après cette unique recette, publier et arrêter; aucun ajustement selon les
erreurs externes. Pas de national ni d'ouverture fold0/fold1. Le modèle SIRET
aval, le decider, le risk et l'accepteur ne sont pas dégelés. Un export nouveau
indépendant reste nécessaire avant toute revendication cross-CRM.

Mac et CATNAT seulement, 30 minutes/RSS14 Gio/zéro swapout par phase,
temporaires externes bornés. Verdict maximal `PIVOT_GENERATION` tant que
PreRank500, Recall100, Wilson et SLO ne sont pas prouvés; défaut d'intégrité
ou d'attribution = `STOP`. Une amélioration Master seule ne vaut pas gate.
