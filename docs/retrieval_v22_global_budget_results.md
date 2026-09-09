# V22 — ouvrir les 2000 places : 46 gains, une régression

9 septembre 2026. Recette préenregistrée dans `a71865b7`.
Verdict : **`PIVOT_GENERATION`**, pas de promotion produit.

## Expérience réalisée

Sur le réservoir v21 strictement inchangé, sélectionner les 2000 meilleurs
scores globaux au lieu de protéger jusqu'à 1750 résultats de la recherche
initiale. Tous les scores sont conservés bit à bit. Aucun nouveau poids,
feature, apprentissage, source ou ajustement après évaluation.

« Ancien » désigne ici un candidat issu de la recherche initiale, pas une
entreprise âgée, fermée ou confirmée par un humain. « Nouveau » désigne un
candidat ajouté par les recherches complémentaires ou l'expansion des sites.
La provenance apporte encore ses preuves au score, mais aucune place garantie.

## Résultat apparié, panel de développement inchangé

| Mesure | V21 protégé | V22 global |
|---|---:|---:|
| Exacts difficiles conservés | 216 / 522 | 262 / 522 |
| Contrôles exacts conservés | 512 / 512 | 511 / 512 |
| Total exact | 728 / 1034 | 773 / 1034 |
| Total opérationnel | 730 / 1034 | 773 / 1034 |

**46 nouveaux hits exacts et un ancien hit perdu : +45 nets.** La perte est
un contrôle de l'origine historique, fold4, qualifié UNRESOLVED. Elle reste
pleinement une erreur exacte et opérationnelle; cette qualification n'est
pas une justification pour l'exclure. Côté opérationnel : 44 gains et une
perte, soit +43 nets. Deux précédents hits seulement opérationnels deviennent
exacts; aucune réécriture de vérité n'a été effectuée.

| Segment | Requêtes | Exacts v22 | Opérationnels v22 | Gains exacts / pertes |
|---|---:|---:|---:|---:|
| Fold 2 | 353 | 263 | 263 | 16 / 0 |
| Fold 3 | 346 | 258 | 258 | 17 / 0 |
| Fold 4 | 335 | 252 | 252 | 13 / 1 |
| CRM récent | 672 | 486 | 486 | 39 / 0 |
| CRM historique | 362 | 287 | 287 | 7 / 1 |
| MATCH_EXACT | 874 | 685 | 685 | 39 / 0 |
| AMBIGUOUS | 7 | 7 | 7 | 0 / 0 |
| UNRESOLVED | 153 | 81 | 81 | 7 / 1 |
| V2 non renseignée | 1034 | 773 | 773 | 46 / 1 |
| V3 non renseignée | 1034 | 773 | 773 | 46 / 1 |

Les vues historique all-human et état administratif figurent également dans
le manifeste. Présence d'un successeur actif pour 25 vérités officiellement
fermées; plusieurs actifs acceptables conservés pour 22 requêtes. Il s'agit
de présence dans un Master, jamais de précision top1. La preuve opérationnelle
officielle scellée est celle de v20/v21, avec les mêmes limites.

Les 261 échecs exacts sont tous attribués : **132 parents absents, 3 sites
non développés, 125 évictions vers le Master, 1 absent du catalogue**.
Le réservoir contient toujours 386 exacts parmi les 522 difficiles; 124
restent évincés, auxquels s'ajoute la régression d'un contrôle.

La sélection écarte 234 101 références de l'ancien préfixe protégé, sur 844
requêtes, et les remplace à nombre total constant. Le Master contient toujours
1 973 172 références sur les 1034 requêtes : minimum 205, médiane et maximum
2000. Ce changement n'augmente donc pas le budget de candidats.

## Conclusion permise, et limites

À score et réservoir constants, la réservation de places limitait réellement
le résultat sur ce panel. Son retrait explique les différences observées.
Le gain net est positif dans chaque fold et chaque origine; cela ne constitue
pas une validation sur un CRM indépendant.

**Aucune projection nationale publiée.** L'ancienne formule
`23049 + rescues` devient invalide dès que les anciens hits ne sont plus
garantis. Les 512 contrôles sont présélectionnés; on ne peut pas extrapoler
leur unique régression aux 23 049 anciens hits nationaux. Le taux 74,7582 %
est seulement celui de ce panel enrichi en erreurs. Aucun gate national,
PreRank500, Recall100 ou Wilson n'est franchi par ce résultat.

La méthode n'a pas été réajustée après mesure. Les dossiers sont néanmoins
connus du développement depuis plusieurs cycles : le choix d'architecture
peut s'y adapter, même sans paramètres appris. Une validation indépendante
reste nécessaire, avec recette gelée et export nouveau non choisi selon le
succès du retrieval. Fold0 et fold1 restent fermés; aucun national lancé.

## Vérifications et ressources

35 tests passent, dont 7 tests v22 : compétition globale, invariance au rang
de base et aux colonnes interdites, cap sans remplissage artificiel, folds
interdits, doublons/scores non finis, publication d'une régression de contrôle.
Vérification indépendante depuis le ledger humain : 773 exacts, pool naturel,
scores bit à bit identiques à v21, cap/unicité, transitions exactes et
opérationnelles et reproduction du préflight dans le run complet. Tous les
hashes de sorties sont vérifiés. Aucun fichier historique modifié.

| Phase | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Préflight sans labels, 16 requêtes | 0,218 s | 0,19 Gio | 0 |
| Sélection complète des 1034 | 1,170 s | 1,65 Gio | 0 |
| Évaluation | 0,305 s | 0,50 Gio | 0 |

Ces temps excluent les validations préalables des hashes et la génération
amont du réservoir réutilisé; ils ne représentent pas une latence end-to-end.
Temporaires sur CATNAT; le disque interne reste presque plein. Aucun coût
externe, aucun modèle aval dégelé, aucun LambdaMART entraîné dans ce sprint.

## Scellement et suite

Configuration SHA-256 :
`f9a73fa4db0ce00e1d45720225c9021a44dce6487e28146fc64f324e78cfde09`.
Racine : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

| Manifeste relatif à la racine | SHA-256 |
|---|---|
| `retrieval_v22_global_generation/f9a73fa4db0ce00e/manifest.json` | `2855b3e20d06e2d6639f4c89fa38e8fbe836936479d7dd49ff69caaf395b3a34` |
| `retrieval_v22_global_evaluation/f9a73fa4db0ce00e/manifest.json` | `0cf74394b75c452281f416813054482e91edfea00ae984def6dbc73a0972bf94` |
| `retrieval_v22_global_preflight/f9a73fa4db0ce00e/manifest.json` | `7a1f8ce9c2ec6d5710f9df9f0678899dea54c878680fa5cb7c99533b1f3a1da0` |

La prochaine étape proposée est la preuve conjointe nom de l'entreprise /
adresse du site et le parent appris OOF correctement construit, pas un nouveau
quota réglé sur les erreurs observées. L'abandon de la réservation est une
base expérimentale encourageante, pas une preuve que le score actuel suffit.
Un futur ranker SIRET reste limité à 2000 références et ne peut réparer une
éviction en amont. Les 132 parents absents constituent un problème distinct.

**Décision finale : `PIVOT_GENERATION`.** Aucune promotion après le seul +45,
aucune grille ou nouvelle recette automatiquement essayée après le résultat.
