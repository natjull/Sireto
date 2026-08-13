# Audit qualité des 279 labels canoniques V4.12

## Objet

Cet audit complète le fichier canonique `reports/v412_review_trusted_labels_279.csv` sans le modifier. Il recherche les cas où un label `MATCH_EXACT` ne constitue pas une vérité d'apprentissage honnête pour un modèle local CRM–SIRENE : cible temporelle non datée, plusieurs SIRET indiscernables dans la scène locale ou choix justifiable uniquement par une preuve externe.

L'overlay opérationnel est `reports/v412_trusted_label_quality_overlay.csv`.

## Règles appliquées

- `CORRECT` : le label canonique doit être corrigé ou changé de nature.
- `EXCLUDE_LOCAL` : le dossier n'est pas une vérité SIRET exacte apprenable depuis les champs CRM et le snapshot SIRENE locaux. Il doit être exclu du ranker et de l'accepteur locaux.
- `QUARANTINE_EXTERNAL` : le SIRET peut rester défendable grâce à une preuve externe ou une convention métier, mais ne doit pas mesurer la capacité du modèle local.
- Un transfert d'adresse sans date CRM n'autorise pas à remplacer automatiquement l'ancien SIRET visé par le siège courant.
- Deux établissements actifs que les champs CRM ne permettent pas de distinguer rendent la cible locale ambiguë, même si un annuaire externe favorise l'un d'eux.

## Corrections du canonique proposées

| Query | Label actuel | Recommandation | Confiance | Motif |
|---|---|---|---|---|
| `10613` | `MATCH_EXACT 26860086300115` | `MATCH_EXACT 26860086300016` | Haute | Le CRM générique IDEF 86 ne mentionne pas le service SAEF de `00115`; l'adresse est présentée comme siège et le siège SIRENE est `00016`. |
| `11198` | `MATCH_EXACT 77567227214968` | `AMBIGUOUS` | Haute | Deux établissements Croix-Rouge actifs, de même SIREN et à la même adresse; aucun service n'est précisé dans le CRM. |
| `12428` | `MATCH_EXACT 34738457000573` | `AMBIGUOUS` | Haute | Deux SIRET Boulanger actifs, même SIREN, même adresse et signaux locaux identiques; les sources externes ne sont pas temporellement homogènes. |

## Exclusions fortes du périmètre local

| Query | Recommandation | Confiance | Cause |
|---|---|---|---|
| `12237` OXYA | `UNRESOLVED` | Haute | Ancien site exact au 1 rue de Londres contre site courant au numéro 3; absence de date CRM. |
| `12298` DAMARTEX GROUP | `UNRESOLVED` | Haute | Ancien siège exact avenue de la Fosse aux Chênes contre siège courant boulevard de Fourmies; absence de date CRM. |
| `15470` VAPOSTORE | `UNRESOLVED` | Haute | Ancien SIRET exact au numéro 47 contre site courant au numéro 6; absence de date CRM. |
| `fresh:AC009634` SOMUDIMEC | `UNRESOLVED` | Haute | Aucun établissement lyonnais dans l'historique SIRENE local; le label est le siège courant de Grenoble. |
| `fresh:FR018872` LEA ET LEO | `AMBIGUOUS` | Haute | Multiples entités actives co-localisées; le CRM générique ne désigne pas Grand Ouest. |
| `4692` CROUS HSC | `AMBIGUOUS` | Haute | Plusieurs services et résidences du même SIREN à l'adresse; HSC ne désigne que la commune. |
| `15253` ADAPEI 77 | `AMBIGUOUS` | Haute | Tiers régulateur et foyer de vie actifs du même SIREN à la même adresse; aucun service dans le CRM. |
| `13754` EDEIS | `AMBIGUOUS` | Haute | Plusieurs entités EDEIS co-localisées; le nom de groupe générique ne prouve pas EDEIS Ingénierie. |

Ces huit cas rejoignent les deux corrections en `AMBIGUOUS` (`11198`, `12428`) pour former les dix labels retirés du dénominateur `MATCH_EXACT` local.

## Quarantaine fondée sur preuve externe

| Query | Label conservable hors évaluation locale | Confiance | Motif |
|---|---|---|---|
| `1828` IN CONCEPT | `31804514300027` | Moyenne | Le siège et un secondaire portant exactement l'enseigne CRM sont indiscernables localement. Le siège ne gagne qu'avec une convention métier explicite. |
| `7373` UIMM | `78931279000015` | Haute | Une mention légale UIMM Alsace valide le SIRET, mais UIMM Alsace et UIMM Bas-Rhin sont localement indiscernables à l'adresse. |
| `13820` Cap West Leinster | `45402394600220` | Haute | Un document Cap West valide Leinster I, mais Leinster I et II sont localement indiscernables sans le suffixe ou le hall. |

## Cas contrôlés mais non déclassés

- `8381` RSM EST : documents officiels directs donnant le SIRET retenu et dénomination légale locale exacte.
- `899` CFA UTEC Emerainville : publication officielle CCI reliant explicitement le site au SIRET.
- `11579`, `11171`, `13522`, `fresh:AC019651`, `fresh:FR010135`, `fresh:FR027885` : le numéro de voie, la ville ou la dénomination juridique sépare effectivement les concurrents.
- Les principaux dossiers hôpitaux, communes et écoles examinés restent identifiables par rôle, activité ou statut de siège.

## Impact sur la mesure du ranker

Point de départ canonique :

- `216 / 254 = 85,04 %` de Hit@1 sur les labels `MATCH_EXACT`.

Périmètre local fort :

- correction de `10613`, qui transforme une erreur en succès : `+1` au numérateur;
- retrait de dix labels non exacts localement (`11198`, `12428` et les huit `EXCLUDE_LOCAL`);
- parmi ces dix, quatre étaient comptés comme succès et six comme erreurs;
- résultat : `(216 + 1 - 4) / (254 - 10) = 213 / 244 = 87,30 %`.

Périmètre local strict après quarantaine externe :

- retrait supplémentaire de `1828`, `7373` et `13820`;
- seul `1828` était compté comme succès;
- résultat : `(213 - 1) / (244 - 3) = 212 / 241 = 87,97 %`.

Cette variation n'est pas un gain de modèle. Elle corrige le périmètre pour mesurer le ranker uniquement sur les dossiers dont le SIRET exact est identifiable localement.

## Décision recommandée

Pour relancer le ranker local, utiliser les **241 cas du périmètre strict** : appliquer la correction `10613`, exclure les dix dossiers non identifiables localement et les trois dossiers dépendant d'une preuve externe. Conserver le canonique intact et appliquer l'overlay de façon explicite et traçable lors de la construction du dataset.
