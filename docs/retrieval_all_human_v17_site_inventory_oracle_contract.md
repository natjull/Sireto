# Contrat v17 — inventaire exhaustif des sites et oracle parent/adresse

Date : 26 août 2026  
Portée : étapes 1 à 3 uniquement, cohorte bornée de 1 034 requêtes

## Question

Le pilote v16 plafonne à 316/522 bons parents dans l'union texte et
géographie. Il ne teste toutefois pas deux mécanismes ensemble : utiliser les
adresses de tous les établissements pour faire remonter leur SIREN, puis
développer sans plafond préalable tous les SIRET des parents retenus.

Le présent pilote répond à cette question avant tout LambdaMART et avant tout
run national.

## Population et séparation

La population est la cohorte v15 scellée : 522 absences du Master top1750 et
512 contrôles déterministes, folds 2/3/4 seulement. Le générateur reçoit
uniquement les identifiants de requête et folds ; il rouvre les champs CRM via
SceneIR mais jamais le ledger humain. La vérité et les équivalents
opérationnels sont construits par un évaluateur séparé après scellement.

Fold0 et fold1 sont interdits. Aucun SIRET humain ne peut créer, compléter ou
ordonner un candidat.

## Découverte parent par tous les établissements

Le sidecar officiel `siren_local_sites.parquet` est lu exhaustivement. Les
établissements font remonter leur SIREN par trois preuves label-blind :

- même INSEE en priorité, sinon même code postal ;
- numéro exact lorsque le CRM en fournit un ;
- rue normalisée exacte ou similarité Jaro-Winkler >=0,84.

L'état actif et le siège départagent uniquement des preuves d'adresse déjà
admissibles. Une absence, une distance BAN ou l'état fermé ne rejettent jamais
un parent.

Les parents texte v16, les parents géographiques v16 et ce canal adresse sont
fusionnés selon trois variantes préenregistrées. La primaire réserve 256 places
au texte, 160 à l'adresse et 96 à la géographie, puis remplit les doublons par
un RRF déterministe. Le plafond parent reste 512.

## Expansion exhaustive SIREN vers SIRET

Pour chaque parent de la variante primaire, tous les établissements du sidecar
sont émis. Il n'existe ni profondeur 16, ni quota par parent, ni filtre actif,
ni filtre distance avant cette matérialisation. `source_site_count` et
`emitted_site_count` doivent être égaux pour chaque couple requête/SIREN.

Le plafond Master@2000 n'est pas appliqué dans ce pilote oracle ; il relèvera
du futur ranker site uniquement si les gates de présence passent.

## Vérités exacte et opérationnelle

La vérité exacte reste le SIRET humain historique. L'ensemble opérationnel est
construit sans score modèle conformément à
`docs/siret_operational_equivalence_policy.md` : même SIREN, numéro et rue
normalisée concordants, suffixe concordant lorsqu'il existe, et même INSEE ou
code postal. BAN et distance ne constituent jamais la preuve de même site.

Les métriques exactes et opérationnelles sont toujours distinctes. Un sibling
actif au même site peut être `ACTIVE_SUCCESSOR_SAME_SITE`, mais ne devient
jamais silencieusement le SIRET exact.

## Gates et décision

Avant tout apprentissage, l'évaluateur publie :

- parent oracle brut et parent candidat @512, global et par fold ;
- présence du SIRET exact après expansion exhaustive ;
- présence d'un SIRET opérationnel au même site ;
- contrôles préservés, multiplicités et successeurs actifs ;
- borne Master nationale projetée à partir des 23 049 hits scellés.

Le chemin vers les étapes 4 à 7 n'est ouvert que si la variante primaire
atteint au moins 464 rescues parent, dont 160/158/146 par fold, et si l'oracle
exact projette au moins 99,5 % de Master. Une projection 99,7 % constitue la
marge de confiance souhaitée. Sinon le verdict est
`PIVOT_PARENT_DISCOVERY` et aucun LambdaMART/national n'est lancé.

## Ressources

Un préflight 1..16 requêtes est obligatoire. Le pilote complet est limité à
45 minutes, 14 Gio de RSS et zéro swapout. Tout écart produit un STOP et
interdit l'ouverture de la vérité.
