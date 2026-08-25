# Contrat de faisabilité Master@2000 v15

Ce jalon répond à une seule question avant tout calcul national long : une
extension officielle, déterministe et sans labels du pool v11 peut-elle porter
le Master@2000 projeté à au moins 99,5 % sur les 23 571 requêtes des folds
2/3/4 ? Il n'autorise ni top500, ni top100, ni modèle LambdaMART site.

## Population et séparation des labels

Le pilote contient les 522 absences naturelles du top1750 v11 et 512 contrôles
où le vrai SIRET est déjà présent. Les contrôles sont les plus petits
`SHA256(query_id)` par fold (171/171/170). Le générateur ne reçoit qu'un
Parquet à deux colonnes, `query_id` et `fold`. La vérité et le type de cohorte
restent dans un artefact séparé, ouvert uniquement après scellement de la
génération.

## Recette immuable

Le préfixe v11 top1750 est conservé dans son ordre. Chaque backend officiel
retourne au plus 64 SIREN parents. Les listes sont fusionnées par RRF, `k=60`,
puis plafonnées à 512 parents. Le sidecar officiel développe au plus 16 sites
actifs ou fermés par parent, dans son ordre géographique/adresse existant. Les
sites nouveaux sont parcourus par profondeur puis rang parent; au plus 250 sont
ajoutés. Le pool final ne dépasse jamais 2 000 SIRET et aucune vérité n'est
injectée.

L'overlay SIRENE historique publié avec zéro document n'est pas ouvert. L'index
national de base est l'autorité temporelle complète; inclure un overlay vide
serait refusé par le loader sans ajouter le moindre signal.

## Gate préalable au national

Le pilote passe seulement si les 512 contrôles restent présents, si au moins
155/182, 148/175 et 138/165 absences sont récupérées sur folds 2/3/4, et si une
des deux sensibilités préenregistrées reste à 99,5 % globalement et sur chaque
fold. Les variantes sont calculées depuis la même génération, sans nouveau
retrieval : `(384,225,16)` et `(512,200,8)` pour
`(parents,rescues,profondeur)`.

Les verdicts possibles sont `PASS_MASTER_V15_FEASIBILITY`,
`PIVOT_PARENT_RETRIEVAL`, `PIVOT_SITE_EXPANSION` et `STOP_INTEGRITY`. Une
recette échouée est terminale : toute modification exige une nouvelle version
préenregistrée. Le pilote est plafonné à 45 minutes, 14 Gio de RSS et zéro
swapout. Un PASS autorise seulement un futur run national Master@2000.
