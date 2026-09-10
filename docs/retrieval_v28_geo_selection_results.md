# V28 — le tri appris aide, mais perd des contrôles : recette non promue

10 septembre 2026. **`PIVOT_GENERATION` / `REJECT_NO_RETUNING`.**
Le pilote OOF est terminé et vérifié. Aucun modèle produit remplacé, aucun
Master recalculé, aucun label modifié. Préenregistrement et pilote :
`99a763aa`; reprise du bilan depuis les fits scellés : `0b44dd0b`.

## Résultat principal du canal géographique

| Population | Ancien top512 parents | Nouveau top512 parents OOF | Gains | Pertes | Plafond naturel |
|---|---:|---:|---:|---:|---:|
| Toutes les1034 requêtes | 457 | **509** | 74 | 22 | 553 |
| 522 difficultés historiques | 131 | 170 | 56 | 17 | 209 |
| 512 contrôles | 326 | 339 | 18 | **5** | 344 |

Le gain net est **52 parents** sur ce canal. Cependant, la recette gelée
exigeait zéro perte d'un succès géographique antérieur parmi les contrôles.
Les cinq pertes rejettent donc la recette telle quelle, même si le solde
est positif. Pas de deuxième combinaison, quota protecteur ou seuil ajusté
après observation. Ces cinq pertes sont celles du canal géographique : elles
ne prouvent pas cinq pertes du retrieval multicanal complet, non rejoué ici.

Les gains nets par fold sont respectivement **+6, +12 et +34** :

| Fold | Requêtes | Ancien | OOF | Gains | Pertes | Plafond |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 353 | 159 | 165 | 14 | 8 | 177 |
| 3 | 346 | 152 | 164 | 23 | 11 | 180 |
| 4 | 335 | 146 | 180 | 37 | 3 | 196 |

Ce sont des résultats sur un panel réutilisé, pas une estimation nationale,
ni une démonstration de généralisation inter-CRM.

## Ce qui est réellement nouveau

- Parmi les **132 parents absents des preuves multicanales précédentes**,
  17 existent dans le pool spatial avant coupe. Le modèle en retient **7**.
- Parmi les **12 absents proches** étudiés en v27, 10 sont accessibles dans
  ce pool et **3** sont retenus par la nouvelle sélection.
- Les deux proches exclus avant les4096 sites restent inaccessibles : le
  modèle ne les voit pas et aucun positif n'est ajouté.

Les indices combinés apportent donc du signal pour la sélection des parents,
mais ne corrigent pas toute la génération et ne suffisent pas à supprimer
les régressions. Aucune ablation n'a été réalisée : on ne peut pas attribuer
causalement le gain à une feature particulière.

## SIRET exact et opérationnel : plafonds d'expansion, pas sorties produit

On vérifie ici si le SIRET humain, ou un équivalent officiel, figure déjà
dans les sites naturels derrière les parents retenus. **Ces sites ne sont
pas une sortie plafonnée à512 SIRET, ni un Master2000.**

| Présence derrière les parents retenus | Ancienne sélection | Sélection OOF |
|---|---:|---:|
| SIRET humain exact | 389 | 424 |
| SIRET exact ou équivalent opérationnel | 410 | 445 |

Sur les132 parents manquants, les sept nouveaux parents donnent **cinq**
cas où le SIRET exact est déjà disponible dans ce pool (cinq aussi en vue
opérationnelle). Cela ne signifie pas cinq gains Master ou top100 : expansion,
sélection et admission end-to-end restent à mesurer dans une autre expérience.

Les métriques Master v24 sont recopiées, inchangées : **823 exacts /825
opérationnels**, dont311/313 difficultés et512/512 contrôles. V24 reste
non promue; v23 reste la référence de recherche. Aucun nouveau Recall100,
PreRank500, Wilson, SLO ou national, aucun dégel aval.

## Méthode et cloisonnement

Génération sans labels sur les1034 requêtes des folds2/3/4 : **3160357 paires
requête/site**, au plus4096 chacune, et **2949964 paires requête/parent**.
Même BAN, commune, coordonnées et ordre de génération que v16. L'ancienne
sélection512 est reproduite; aucun nouveau site géographique introduit.

Vingt et une features légères : similarités des noms SIRENE courants autorisés,
voie/numéro/CP, état, qualité BAN, distance/rang spatial/nombre de sites. Pas
de noms de personnes, d'identifiants comme features, d'alias CRM ou de noms
RNE au rôle ambigu. L'accord conjoint nom×voie est calculé sur le même site,
sans numéro obligatoire et sans déplacer un nom de site vers ses frères.
Les indices BAN sont expérimentaux et restent inéligibles à l'activation
produit; aucune hydratation riche n'a été exécutée.

Un seul XGBRanker parent, avec les hyperparamètres v23 inchangés :240 arbres,
profondeur6, taux0,05, objectif pairwise; trois modèles outer OOF. L'échantillon
d'entraînement contient uniquement le positif déjà naturellement présent et
jusqu'à512 négatifs (256 tête +256 queue stratifiée). Aucun SIREN humain
partagé entre folds d'entraînement et de test. Aucun composant appris aval
n'utilise ces scores, donc pas de calibration interne supplémentaire ici.

| Fold tenu à l'écart | Groupes positifs naturels entraînés | Lignes d'entraînement | Requêtes train non entraînables | Parents scorés hors entraînement |
|---|---:|---:|---:|---:|
| 2 | 376 | 189442 | 305 | 1009755 |
| 3 | 373 | 188031 | 315 | 951107 |
| 4 | 357 | 179751 | 342 | 989102 |

Les groupes non entraînables ne disparaissent **jamais** du dénominateur
d'évaluation. Les rangs retenus dépendent du score OOF, puis de l'ancien
rang spatial et du SIREN comme ultime départage. Aucun label tenu à l'écart
n'entre dans la sélection, les features ou les seuils.

## Incident de bilan et reprise sans réentraînement

Après les trois fits/scorings terminés, une assertion exigeant que chaque
SIRET exact figure dans la table opérationnelle a arrêté la publication.
L'unique absence concerne précisément le cas déjà publié
`OFFICIAL_CATALOG_ABSENT`. Le sidecar officiel est présent, scellé et indique
l'absence; le SIRET est aussi absent du pool géographique. Ce n'est ni un
label à exclure, ni un positif à inventer.

Avant tout résultat global, les neuf fichiers modèles/scores/splits ont été
hashés dans `config/retrieval_v28_completed_fits.json`. La finalisation
distincte vérifie cette absence connue, conserve les1034 requêtes et marque
le cas faux en exact comme en opérationnel. **Aucun modèle réentraîné, aucun
score modifié.** Les artefacts originaux restent intacts; le répertoire
initial sans manifeste final n'est pas un résultat admissible autonome.

## Vérification et ressources

**82 tests passent.** Le vérificateur `scripts/audit_retrieval_v28_results.py`
confirme les hashes, l'identité des copies de fits, le ledger folds2/3/4,
l'égalité préflight/complet, les populations naturelles, l'échantillonnage
des groupes train et les caps. Il **rejoue les trois modèles et reproduit
exactement tous les scores**, reconstruit les512 retenus puis recalcule les
vues parent/exact/opérationnel. Verdict `PASS_V28_OUTPUT_RECONCILIATION`.

- Génération :54,472 s hors vérification préalable des hashes, RSS9332113408
  octets (~8,69 Gio), zéro swapout.
- Finalisation :1,254 s hors hashes, RSS2614657024 octets (~2,44 Gio), zéro
  swapout. Ce n'est pas la durée des trois entraînements.
- Entraînements : modèles terminés sous garde-fous RSS14 Gio/30 minutes/zéro
  swapout, mais **rapport quantitatif de ressources non scellé** à cause de
  l'arrêt du bilan. Aucun temps global ni pic chiffré n'est inventé.
- Mac/CATNAT exclusivement, aucun coût externe. Aucun X/Y brut exporté.

Les manifestes publient ensemble historique/cohorte, V2/V3, folds, origines,
types de label, états et étapes de perte. Les plafonds produit2000/500/100
et les modèles aval restent inchangés. Les4096 voisins appartiennent au
pilote expérimental existant, jamais à une sortie produit ou à un nouveau
plafond Master.

## Décision

**Ne pas intégrer cette recette.** Conserver ses scores et les résultats
comme preuve qu'une sélection avant coupe peut aider, avec un coût réel en
pertes. Le critère préenregistré est appliqué sans retuning. Aucune expansion
end-to-end ni autre recette n'est lancée automatiquement dans ce cycle.
Verdict scientifique **`PIVOT_GENERATION`**, pas `READY_FOR_FOLD1`, pas de
garantie99,2 %. Folds0/1 fermés.

## Artefacts scellés

Configuration SHA256
`1f604bd8a92e9fd923d4792e228d414d44d8e224ec559cb7eb22bdd9a5446394`.
Sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/` :

- `retrieval_v28_geo_features/1f604bd8a92e9fd9/manifest.json`, SHA256
  `a960f8697bfcc7d18944445504a4634c36bcc1c333c487bc95d3183188c69762`.
- `retrieval_v28_geo_oof/1f604bd8a92e9fd9-sealed-fits/manifest.json`, SHA256
  `38bf7a7b09ce4b1b1c53812dc8017c03db3d2b1b37a99560f217a662ab21d17b`.

Reprise des fits SHA256
`884d10da18ef90a61cdb8382f7b81bed82606eb3c0bcf5f1521e9a4e1c0afb44`.
