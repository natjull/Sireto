# V30 — gain modeste et vérifié de la recherche conjointe nom/adresse

10 septembre 2026. **NET_GAIN_REUSED_PANEL / PIVOT_GENERATION**.
Expérience terminée, recette inchangée depuis son préenregistrement.
V23 reste la référence; v30 est un challenger de recherche, pas une activation
produit. Aucun calcul en cours à la clôture.

## Résultat au plafond de 2 000

| Mesure sur les 1 034 requêtes | V23 | V30 | Évolution |
|---|---:|---:|---:|
| SIRET humain exact dans le Master | 822 | 829 | 12 gains, 5 pertes, net +7 |
| Exact ou équivalent opérationnel scellé | 824 | 832 | 12 gains, 4 pertes, net +8 |
| SIRET exact dans le réservoir avant coupe | 898 | 900 | +2 |
| Contrôles exacts dans le Master | 512/512 | 512/512 | Aucun contrôle perdu |
| Cas difficiles exacts dans le Master | 310/522 | 317/522 | +7 |

Le panel comprend 522 difficultés initiales et 512 contrôles. Il a été
réutilisé pour le développement : ses proportions **ne sont pas un recall
national**, ni une validation indépendante de généralisation. Les 1 034
labels restent au dénominateur, y compris l'absence du catalogue officiel.

| Fold OOF | Requêtes | Exact v23 | Exact v30 | Gains / pertes |
|---|---:|---:|---:|---:|
| 2 | 353 | 296 | 298 | 4 / 2 |
| 3 | 346 | 268 | 271 | 6 / 3 |
| 4 | 335 | 258 | 260 | 2 / 0 |

Le net positif dans les trois folds est descriptif, pas une preuve de
significativité ou d'indépendance du panel. Le segment des vérités fermées
régresse de 1 en exact : 172/246, contre 173 auparavant. Les vérités actives
gagnent 5 (551/628); l'état non renseigné gagne 3 (106/160).

### Vues historique, V2 et V3 publiées ensemble

La vue historique all-human donne 829/1 034 exacts et 832/1 034 opérationnels.
Les qualifications V2 et V3 sont toutes non renseignées dans ce panel :
chacune publie explicitement ce segment de 1 034 lignes, avec les mêmes
829 exacts et 832 opérationnels; aucune qualification n'est inventée.
Les segments historiques restent : MATCH_EXACT 723/874 exacts,
AMBIGUOUS 7/7 et UNRESOLVED 99/153. Aucun de ces statuts ne filtre la métrique.
Les vues détaillées et les origines figurent dans le manifeste d'évaluation.

## Ce qui a changé, et ce qui explique le gain

Le nouveau canal cherche conjointement des fragments du nom courant et de
l'adresse courante sur le même document SIRET, avant sa coupe à 512 résultats.
Pas d'égalité obligatoire du numéro, ni de transfert de preuve entre sites.
L'index et tous les anciens canaux sont conservés. Les modèles parent v23
sont rechargés sans entraînement; leurs features sont recalculées et la
même règle globale sélectionne au plus 2 000 références.

- Le canal émet 450 725 références sur 1 033 requêtes éligibles. La requête
  restante est conservée avec un canal vide. Il contient directement le
  SIRET exact pour 545 requêtes et un témoin du parent exact pour 666.
- Le réservoir naturel gagne 96 376 paires, sans en retirer : 3 996 520 au
  total. Cela ne représente que **deux SIRET exacts nouvellement retrouvés**.
- Sur les 12 gains au Master, **11 vérités étaient déjà dans l'ancien
  réservoir** : dix reçoivent un témoin direct du nouveau canal, une bénéficie
  indirectement du reclassement. Une seule provient des deux nouvelles vérités.
- L'autre nouvelle vérité reste évincée au plafond de 2 000. Les cinq pertes
  du Master sont également toujours dans le réservoir, mais évincées.

Donc le progrès observé vient surtout de la sélection de références déjà
disponibles, et modestement d'une nouvelle couverture. Ce n'est pas une
résolution des absences de génération. Un canal de plus, issu du même index,
n'est pas une nouvelle autorité officielle indépendante. Le déplacement
de distribution des features sous des modèles gelés reste une limite.

### Attribution exhaustive des 205 erreurs exactes

| Cause | Requêtes |
|---|---:|
| Parent non retrouvé | 130 |
| Site non développé malgré le parent | 3 |
| Éviction du réservoir vers Master2000 | 71 |
| Vérité absente du catalogue officiel | 1 |

Les faits officiels proviennent du sidecar label-side scellé; les présences
dans les nouveaux parents, réservoir et Master sont réactualisées. L'équivalence
opérationnelle reste séparée : trois hits opérationnels sans exact, sans
retrait du dénominateur exact. Aucun positif n'est injecté.

## Vérification et coût mesuré

- Recette/code/config figés par `52dbe1c7`, avant recherche complète;
  lancement consigné par `c98f85f3`.
- Vérificateur figé par `24c1169b`, avant mesure de qualité. **54 tests passent**,
  incluant preuve conjointe partielle, absence de fabrication entre sites,
  garde des folds, caps et absence d'entraînement.
- Préflight 16 sans labels : 6 685 témoins et 30 276 références Master;
  scores et Master v23 rejoués exactement. Parité avec le complet vérifiée.
- Audit final **PASS_V30_OUTPUT_RECONCILIATION** : hashes, requêtes sans labels,
  ledger limité aux folds 2/3/4, splits, témoins, union/expansion naturelle,
  caps, inférence des trois modèles, classement et métriques rapprochés.
- La recherche complète a été rejouée avec le même code gelé et a reproduit
  exactement références et scores. Ce replay n'est pas une deuxième
  implémentation du moteur de recherche; les tests sémantiques sont distincts.
- Index : SHA manifeste/meta et inventaire tailles/mtime inchangés,
  **pas de rehash intégral de tous ses octets**.

| Étape complète | Temps interne | Pic RSS |
|---|---:|---:|
| Recherche conjointe, quatre workers | 229,84 s | 11,52 Gio |
| Réservoir, features et Master | 73,07 s | 9,88 Gio |
| Évaluation label-side | 1,11 s | 2,01 Gio |
| Vérificateur avec rejeu de recherche | 266,78 s | 11,83 Gio |

Zéro swapout; garde de 14 Gio respectée à chaque étape. Hashes préalables
hors temps. En recherche concurrente, durée par requête p95 1,793 s et
p99 2,526 s : coût supplémentaire à prendre au sérieux, **pas un SLO produit
warm validé**. Mac et CATNAT uniquement, aucun coût externe pour cette expérience.

## Artefacts scellés

Racine : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.
Config : `config/retrieval_v30_joint_search.json`, SHA256
`8cb4ad779f11ac844a7307b3e414e2b5895bdd4c863dc367a5f311acc431666f`.

- `retrieval_v30_discovery/8cb4ad779f11ac84/manifest.json` :
  `b6fa2a8b5911d43fc0e0b0949af3e4bc051c6de4a21d0c678ab8bfd782dd2731`.
- `retrieval_v30_master/8cb4ad779f11ac84/manifest.json` :
  `ef6e0a85e30c440d59d70e4041d0103c95747e38dc25443c9b4c014a3d9aece1`.
- `retrieval_v30_evaluation/8cb4ad779f11ac84/manifest.json` :
  `d7722bf7256d89507abd2349c601ca521a60474b5af625485e9051146e66ff5f`.
- `retrieval_v30_verification/8cb4ad779f11ac84/manifest.json` :
  `faeea8c26e85e49bdd8f5c4d1cc852e5c25d607d2f0e0c3fc7c3502f29f6be6c`.

Le Master totalise 1 973 172 références, maximum 2 000 par requête.
Les manifestes scellent aussi les fichiers, dépendances et sidecars.

## Décision

**PIVOT_GENERATION** reste le verdict scientifique : résultat de recherche
positif mais insuffisant pour le gate. Conserver v30 gelée comme challenger,
sans ajuster sa recette après lecture des gains/pertes. V23 reste la référence
stable. La suite proposée, non exécutée ici, est de distinguer une validation
moins réutilisée de ce canal et un travail séparé sur l'admission; aucun des
deux ne doit masquer les 130 parents encore absents.

Pas de national, de mesure PreRank500/Recall100, de garantie 99,2 %, de SOTA,
de promotion automatique, de fit aval ou de revendication cross-CRM.
Folds 0 et 1 non ouverts. Aucun READY_FOR_FOLD1.
