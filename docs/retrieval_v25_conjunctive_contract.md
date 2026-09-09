# Addendum V25 — recherche d'adresse contrainte par le numéro

Préenregistrement après audit label-side, **avant génération et mesure**.
L'audit trouve 88 parents absents mais atteignables par un canal v18 dans la
zone indexée, dont 82 par adresse-caractères. Cela n'est pas un gain naturel.

Une seule correction testée : canal SIRET **numéro exact MUST + adresse
caractères MUST**, avec la géographie et les termes caractères v18 inchangés.
Même profondeur 512 SIRET que les canaux v18. Le numéro vient de l'adaptateur
CRM déjà utilisé; ne pas le remplir avec celui du label ou d'un candidat.
Numéro/adresse manquant = canal vide, pas exclusion de la requête. Aucun
filtrage du canal historique, aucune modification de BAN, de la géographie,
des noms, des labels ou des statuts. Le nouveau canal est strictement additif.

Cette intersection peut faire remonter une adresse noyée parmi les autres
numéros d'une commune. Un mauvais numéro peut empêcher ce nouveau canal de
réussir; il ne supprime pas les anciens candidats. Les preuves textuelles
absentes ne sont pas magiquement recréées par une intersection plus stricte.

Générateur séparé sans ledger, labels, statut, cohorte ni fichier de diagnostic.
Il lit les 1034 identités de requête et les champs CRM SceneIR autorisés, puis
l'index officiel scellé. SIRET/SIREN sont extraits des champs officiels stockés,
jamais d'un préfixe; validation et déduplication. Départage SIRET terminal,
rang de preuve dense par score, aucun identifiant comme preuve.

Un préflight technique ≤16 requêtes sans labels précède le run complet.
Quatre workers maximum, RSS14 Gio, zéro swapout, trente minutes, Mac/CATNAT.
Le code/config sont committés avant le run complet. Index : manifeste SHA,
meta.json SHA et inventaire tailles/mtime stables; pas de claim de rehash total.

L'évaluateur distinct ouvre ensuite les labels folds2/3/4 et sidecars officiels
scellés. Publier les nouveaux parents et **SIRET exacts effectivement produits**,
ainsi que la couverture de l'union diagnostique avec le réservoir v21 :
ancien, canal seul, union, gains/pertes, contrôles/folds/origines/statuts/V2/V3.
Les comparaisons sont descriptives, pas une grille de sélection. L'union doit
être monotone et toutes les requêtes restent au dénominateur.

**Cette union exploratoire n'est pas Master@2000**. Aucun top2000/top500/top100
n'est fabriqué, aucun modèle SIRET ne score le grand réservoir, aucun gate
ne peut être annoncé. L'étape suivante devra intégrer la nouvelle preuve dans
une allocation OOF à budget2000, avec ses gains ET pertes. La monotonicité du
réservoir ne garantit pas celle de la future sortie limitée.

Une seule recette, aucun ajustement après résultat. Pas de national, projection,
fold0/fold1, dégel aval ou promesse cross-CRM. OOF ne rend pas vierge ce panel
réutilisé. Verdict `PIVOT_GENERATION` ou `STOP` en cas de défaut d'intégrité.
