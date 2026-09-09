# V20 — sampling confirmé, premier Master SIRET mesuré

Date : 9 septembre 2026. Verdict : **`PIVOT_GENERATION`**.

Le protocole, la configuration et les programmes ont été figés dans le commit
`353c63ea` avant les deux exécutions officielles. Aucun réglage n'a été changé
après leurs résultats. Le workbench utilise exclusivement les folds 2/3/4 :
522 pertes du préfixe v11 top1750 et 512 contrôles. Ce jeu enrichi en erreurs
n'est ni représentatif du national, ni une validation cross-CRM.

## 1. Le sampling explique une grande partie de l'échec v19

Comparaison appariée : mêmes groupes d'apprentissage, nombre de lignes,
features, hyperparamètres et seed. Seule la sélection des négatifs change :
512 premiers RRF, contre 256 premiers + 256 représentants stratifiés de la
queue. Les validations internes sont groupées par SIREN et ne comprennent
pas le fold externe exclu. Aucun positif absent n'est ajouté.

| Fold externe exclu | Cas difficiles internes | Head512 @2000 | Sampling représentatif @2000 | RRF fixe @2000 | Contrôles head → représentatif |
|---|---:|---:|---:|---:|---:|
| 2 | 53 | 22 | 35 | 35 | 65 → 69 / 69 |
| 3 | 55 | 18 | 34 | 35 | 70 → 78 / 78 |
| 4 | 52 | 23 | 35 | 36 | 61 → 65 / 65 |

Les ensembles de validation interne peuvent se recouvrir entre contextes :
ne pas additionner ces lignes comme des observations indépendantes.
Les volumes d'apprentissage sont respectivement 476/486/516 groupes et
243 020/247 936/263 004 lignes, identiques entre méthodes dans chaque contexte.
Les requêtes sans positif restent dans les dénominateurs de validation.

La part de candidats issus de la queue RRF >512 dans le top100 appris passe
de 98,05/99,03/98,92 % à 56,93/62,23/61,23 %. L'apprentissage historique
voyait les négatifs de tête mais devait scorer toute la queue à l'inférence.
Le test confirme un défaut important de distribution des négatifs; il ne
prouve ni une supériorité au RRF à 2000, ni la performance d'un ranker SIRET.
Il corrige l'interprétation « LambdaMART échoue donc abandonner le ML ».

**Limite de conformité :** le rang `v17_union` historique est un
`row_number() ... ORDER BY siren`, dans
`scripts/run_retrieval_parent_discovery_v18.py`. Cette dérivation d'identifiant
a été conservée uniquement dans ce diagnostic afin de tenir les features
constantes. Aucun des six modèles produits n'est éligible à une nouvelle
recette ni à la production. L'affirmation historique « aucun identifiant en
feature » était donc insuffisante : l'absence de colonne SIREN explicite
n'empêche pas une feature indirectement dérivée du SIREN. Les artefacts v19
restent inchangés et leur résultat externe n'a pas été rejoué.

## 2. Cette fois, le plafond porte bien sur des SIRET

La génération sans labels protège intégralement le préfixe top1750, conserve
les preuves directes attachées à chaque établissement, développe les sites
locaux des parents soutenus et remplit un budget global jusqu'à 2000 SIRET.
Le rang artificiel v17 est exclu. Aucun modèle SIRET n'a été entraîné; le
réservoir exploratoire est léger et n'est pas l'entrée d'un LambdaMART.

| Mesure | Résultat |
|---|---:|
| Requêtes évaluées, sans filtre de qualification | 1 034 |
| Contrôles exacts conservés | 512 / 512 |
| Pertes initiales récupérées au Master2000 | 213 / 522 |
| Total exact sur ce workbench difficile | 725 / 1 034 = 70,1161 % |
| Total opérationnel, publié séparément | 727 / 1 034 = 70,3095 % |
| Couples requête/SIRET dans le réservoir | 3 239 218 |
| Couples requête/SIRET au Master | 1 957 117 |
| Taille minimum / médiane / maximum du Master | 197 / 2 000 / 2 000 |

Parmi les 213 rescues exacts, 120 disposent d'une preuve directe et 93 sont
des siblings sans preuve directe. Cette répartition décrit la provenance,
pas un gain causal isolé de chaque canal.

La projection conditionnelle est `(23049 + 213) / 23571 = 98,6891 %`, sous
l'hypothèse de conservation du même préfixe sur toutes les requêtes nationales.
**Ce n'est pas un recall national mesuré.** Le 99,3721 % v19 était une
projection oracle de présence du **parent** : il ne constituait pas un
Master2000 SIRET réalisé, et ne se compare pas directement à cette mesure.

### Segments exacts et opérationnels

| Segment | Requêtes | Exact | Opérationnel |
|---|---:|---:|---:|
| Fold 2 | 353 | 247 | 249 |
| Fold 3 | 346 | 239 | 239 |
| Fold 4 | 335 | 239 | 239 |
| CRM récent | 672 | 444 | 446 |
| CRM historique | 362 | 281 | 281 |
| MATCH_EXACT | 874 | 643 | 645 |
| AMBIGUOUS | 7 | 7 | 7 |
| UNRESOLVED | 153 | 75 | 75 |
| Qualification V2 non renseignée | 1 034 | 725 | 727 |
| Qualification V3 non renseignée | 1 034 | 725 | 727 |

Les champs V2/V3 sont vides dans le ledger scellé de ces 1034 lignes : aucune
qualification n'est inventée et aucune ligne n'est retirée. La vue historique
all-human est également publiée dans le manifeste. Les rescues exacts par
fold sont 76/182, 68/175 et 69/165; leur présence dans le réservoir avant cap
est 111/182, 113/175 et 101/165.

L'équivalence opérationnelle apporte deux hits supplémentaires, sans changer
les gates exacts. Le Master contient un équivalent actif de vérité officiellement
fermée pour 25 requêtes; cela ne représente pas 25 nouveaux hits. On compte
22 requêtes avec plusieurs SIRET actifs dans l'ensemble acceptable, dont 20
avec plusieurs actifs effectivement conservés au Master. Ces métriques
mesurent la présence, pas une précision de sélection top1. Les 13 équivalents
à CP différent (12 requêtes) ont été contrôlés : leur INSEE correspond à une
commune multi-CP dans le snapshot officiel. Les différences de compléments
d'adresse ne sont pas mesurées dans ce workbench; la sortie produit et sa
preuve complète restent à implémenter.

## 3. Attribution officielle exhaustive des 309 pertes

Le sidecar label-side est scellé depuis le snapshot officiel SIRENE, séparé
de la génération. Une absence reste une erreur end-to-end.

| Cause | Nombre | Conséquence |
|---|---:|---|
| SIRET absent du catalogue officiel | 1 | Erreur publiée, pas de retrait |
| Parent absent de toutes les preuves brutes | 132 | Découverte parent nécessaire |
| Parent présent mais SIRET non développé | 64 | Expansion à revoir |
| SIRET présent au réservoir mais évincé à 2000 | 112 | Allocation du Master à revoir |

Diagnostic secondaire en lecture seule, après scellement :

- **61 des 64 non-expansions** ont un parent présent uniquement dans le canal
  `v17_union`. Leur SIRET est pourtant compatible avec la géographie CRM.
  V20 exclut le canal entier du soutien parent pour neutraliser son rang
  artificiel : cette précaution jette aussi une preuve de présence valide.
  Une future recette peut garder l'appartenance au canal, sans exploiter ce
  rang ni lui inventer un rang de pertinence. Aucun résultat v20 n'a été
  retouché à cette fin.
- Les **3 autres** non-expansions ont un parent soutenu par un autre canal,
  mais leur SIRET exact est hors INSEE/CP CRM et sans témoin direct retenu
  dans le réservoir. Une géographie CRM stricte peut manquer un label exact.
- Sur les **112 évictions**, 91 sont des siblings sans preuve directe,
  10 des témoins directs nouveaux et 11 des candidats du raw v11 au-delà du
  préfixe protégé. La somme RRF favorise les preuves directes; les données
  montrent où examiner la compétition entre sources, sans établir à elles
  seules un nouveau score optimal.

Le réservoir ne contient que **325/522** exacts difficiles. Même un choix
parfait dans ce réservoir plafonnerait conditionnellement à
`23374/23571 = 99,1642 %`. Il ne suffit donc pas de retrouver les 112 évincés.
Il faut au minimum 334 rescues pour 99,2 %, et 381 pour 99,4 %, avant toute
perte lors de Master→500→100. Récupérer aussi les 61 présences v17 donnerait
une borne théorique de 386 rescues, soit seulement cinq cas de marge au-delà
de 99,4 % : ce n'est ni une mesure de la prochaine recette ni une garantie.

## 4. Suite concrète, sans long run

1. Préenregistrer une nouvelle recette qui conserve la présence v17 sans son
   rang dérivé d'identifiant. Vérifier par tests que permuter les identifiants
   hors départages exacts ne change pas les scores.
2. Revoir l'allocation déterministe des places de rescue entre témoins directs
   et siblings avec preuve d'adresse, avant le cap. Garder les preuves
   attachées aux sites et le préfixe protégé. Ne pas faire scorer plus de
   2000 références par un futur LambdaMART SIRET.
3. Évaluer une recette figée sur le workbench, avec les mêmes contrôles et
   pertes officielles. Ne pas sélectionner une grille sur ses résultats.
   Si le plafond ou sa marge reste insuffisant, enrichir la découverte des
   parents et les voies d'expansion hors géographie CRM.
4. Seulement après une marge crédible au Master : développer le préclassement
   SIRET OOF avec négatifs représentatifs et features conformes, puis mesurer
   PreRank500 ≥99,4 %. Recall100, Wilson, latence et généralisation restent
   des preuves distinctes à obtenir.

Le rang artificiel retiré et le sampling corrigé sont deux problèmes
différents : le premier relève de la conformité des features, le second de
la distribution d'apprentissage. Aucun modèle de l'audit n'est promu.

## 5. Ressources, contrôles et scellement

| Phase officielle | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Audit de sampling, six entraînements | 18,817 s | 2,55 Gio | 0 |
| Génération Master, 1034 requêtes | 10,236 s | 5,13 Gio | 0 |
| Évaluation truth-side | 2,463 s | 1,56 Gio | 0 |

Le chronométrage couvre le bloc de calcul, pas la validation préalable des
hashes d'entrée; il ne s'agit pas d'une mesure de latence produit. Tous les
budgets locaux sont respectés, sans GPU loué ni service payant. Les 16 tests
v18/v19/v20 passent. Un recalcul indépendant confirme les 725 exacts, le
préfixe intégralement préservé, l'unicité des SIRET, l'inclusion des hits
exacts dans les hits opérationnels et les hashes de tous les artefacts.

Configuration : `config/retrieval_v20_evidence_workbench.json`, SHA-256
`3ccb3e44a19a1769f1b555acb1e3e26eab94e9d902c442120fb0051274ad6899`.

Racine des artefacts : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

| Manifeste relatif à cette racine | SHA-256 |
|---|---|
| `retrieval_v20_sampling/3ccb3e44a19a1769/manifest.json` | `7c39c3d60bdd3b90172fd9789e73b90742eb83a9fcf547ef00b9666782e295bd` |
| `retrieval_v20_master/3ccb3e44a19a1769/manifest.json` | `02be8082bd5e54f6c74b2025ea391cc5ce138ed85afb0b9e1d01ee2bde6445e4` |
| `retrieval_v20_master_eval/3ccb3e44a19a1769/manifest.json` | `87cf5cd94b20411b56f82d30c62f43e7eb52c13928ff51d5be040f659caac3ae` |

Le sidecar `official_loss_sidecar.parquet` porte le SHA-256
`77653fb8022f897faf2843dfe40d6fb7d50088d87a83f7dcd18799a7332e012d`.
Chaque manifeste référence aussi ses fichiers, sources et implémentations.

**Décision finale : `PIVOT_GENERATION`.** Ni PreRank500 ni Recall100 ne sont
mesurés. Fold0 et fold1 restent fermés; aucun run national, aucune promotion
du ranker/decider/risk/accepteur, aucune revendication SOTA ou cross-CRM.
