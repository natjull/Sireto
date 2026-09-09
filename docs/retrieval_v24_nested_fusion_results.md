# V24 — la fusion calibrée n'apporte pas de progrès convaincant

9 septembre 2026. Verdict **`PIVOT_GENERATION`**. Pas de promotion de v24.
Le protocole et le code ont été figés dans `af3ba922` avant le run complet.
Le [contrat v24](retrieval_v24_nested_fusion_contract.md) définit l'unique recette.

## Résultat principal

| Mesure sur le panel de développement | V23 | V24 |
|---|---:|---:|
| SIRET exact conservé, 522 échecs initiaux | 310 | **311** |
| SIRET exact conservé, 512 contrôles | 512 | **512** |
| Exact, tous les 1034 labels | 822 | **823** |
| Opérationnel, tous les 1034 labels | 824 | **825** |
| Gains / pertes exacts contre v23 | — | **6 / 5** |

Les six gains récupèrent six des 29 régressions précédentes contre v22.
Mais cinq réussites nouvelles de v23 disparaissent. Contre v22, v24 présente
73 gains et 23 pertes (v23 : 78 et 29). La combinaison ne règle donc pas
le désaccord entre les deux approches. Le +1 net ne constitue pas une preuve
d'amélioration robuste et ne justifie pas de remplacer v23 comme référence
de recherche. Aucune de ces versions n'est promue en produit.

Ce résultat concerne **Master@2000**, pas Recall@100. Le panel contient
522 échecs initiaux et 512 succès présélectionnés : ses pourcentages exact
79,59 % et opérationnel 79,79 % ne sont **pas des estimations nationales**.
Aucune projection avec les 23 049 anciens succès n'est effectuée.

## Ce qui a été effectivement testé

Réservoir v21 inchangé : 3 900 144 références exploratoires. Réutilisation
des noms/site et des six modèles parent v23; aucun nouveau LambdaMART.
La calibration apprend un scalaire par fold externe, uniquement sur les
prédictions inner-OOF des deux autres folds. Elle minimise une vraisemblance
**parent**, avec la barrière symétrique préenregistrée. Aucun poids individuel,
aucune liste de dossiers à sauver, aucune mesure externe utilisée pour choisir
le poids. Les trois calibrations appliquent la même procédure, pas trois
recettes choisies après coup.

| Fold cible | Folds de calibration | Requêtes | Calibrables | Parent absent | Poids appris / retrieval |
|---|---|---:|---:|---:|---:|
| 2 | 3, 4 | 681 | 581 | 100 | 76,323 % / 23,677 % |
| 3 | 2, 4 | 688 | 600 | 88 | 78,252 % / 21,748 % |
| 4 | 2, 3 | 699 | 619 | 80 | 79,457 % / 20,543 % |

Les non-calibrables ne fournissent pas de positif artificiel et restent au
dénominateur de l'évaluation. Les modèles internes n'ont jamais appris sur
le fold externe, ni sur le fold qu'ils prédisent. Les identifiants servent
aux jointures/groupes et au départage terminal, pas aux preuves de classement.

Le poids mélange les contributions parent/site v22 et v23, en comptant une
seule fois la preuve directe. Sélection globale de 1 973 172 références,
**maximum 2000 par requête**. Pas de préfixe protégé, pas de modèle SIRET sur
le grand réservoir, pas d'hydratation riche. Le transfert de la calibration
parent vers les scores site est une hypothèse de substitution; ce test ne
démontre pas que toutes les formes de fusion seraient inutiles.

## Segments, sans exclusions

| Segment | Requêtes | Exact | Opérationnel | Gains / pertes vs v23 |
|---|---:|---:|---:|---:|
| Fold 2 | 353 | 296 | 296 | 3 / 3 |
| Fold 3 | 346 | 269 | 269 | 3 / 2 |
| Fold 4 | 335 | 258 | 260 | 0 / 0 |
| CRM récent | 672 | 521 | 522 | 3 / 2 |
| CRM historique | 362 | 302 | 303 | 3 / 3 |
| MATCH_EXACT | 874 | 720 | 721 | 3 / 2 |
| AMBIGUOUS | 7 | 7 | 7 | 0 / 0 |
| UNRESOLVED | 153 | 96 | 97 | 3 / 3 |
| État GT A | 628 | 547 | 547 | 3 / 2 |
| État GT F | 246 | 173 | 174 | 0 / 0 |
| État GT vide | 160 | 103 | 104 | 3 / 3 |

Les vues historique all-human, V2 et V3 sont publiées dans le manifeste.
Les qualifications V2/V3 sont vides pour les 1034 lignes : chacune de ces
vues conserve donc le total 823 exacts/825 opérationnels; aucun sous-segment
n'est inventé. Aucun statut ne retire une ligne du dénominateur.

La vue opérationnelle utilise le sidecar same-site déjà scellé : 2 succès
opérationnels seuls, 25 requêtes avec successeur actif présent et 22 avec
plusieurs équivalents actifs présents. Il s'agit d'une **présence dans le
Master**, pas d'une bonne décision top1 ni d'une nouvelle validation de BAN.
Les limites antérieures du sidecar restent valables, notamment les compléments
d'adresse non comparés. Exact et opérationnel ne sont pas confondus.

## Ce qui bloque encore

| Cause officielle de perte exacte | V23 | V24 |
|---|---:|---:|
| Parent absent des preuves | 132 | **132** |
| Site non développé | 3 | **3** |
| SIRET présent mais évincé vers Master2000 | 76 | **75** |
| Absent du catalogue officiel | 1 | **1** |
| Total | 212 | **211** |

Le recalcul depuis le ledger trouve seulement **898/1034 vérités exactes
dans le réservoir naturel**, avant la limite de 2000. Un classement parfait
de ce réservoir ne pourrait pas récupérer les 136 absentes. Ce plafond
86,85 % décrit uniquement le panel biaisé, jamais le rappel national.

Décision de recherche : ne pas relancer une série de mélanges ou de quotas
pour gagner quelques dossiers connus. Le prochain chantier prioritaire est
la **couverture de découverte parent**, sans abandonner les 75 évictions.
La v18 avait déjà essayé noms de sites/historiques, adresses historiques,
relations et successions : les réintroduire sous un nouveau nom ne serait
pas une approche nouvelle.

Prochaine étape proposée, **pas encore exécutée** : audit label-side séparant
pour les parents absents (a) preuve officielle inexistante/inexploitable,
(b) preuve disponible mais absente de la représentation indexée, et (c)
preuve indexée non atteinte par les requêtes/contraintes. Inclure des contrôles
et publier folds/origines, sans créer d'alias CRM. Choisir ensuite une seule
correction structurelle préenregistrée et générée sans labels. Ne pas lancer
une expansion nationale sans preuve locale de couverture. Les erreurs de
vérité éventuelles restent des erreurs exactes, jamais corrigées discrètement.

## Vérifications et coût

- **53 tests passent** sur les suites v19 fusion, v20, v21, v22, v23 et v24.
  Notamment : perturbation des labels cibles sans effet sur calibration,
  folds interdits, splits, optimum convexe, positifs absents, rangs ex-aequo,
  invariance à l'ordre des lignes, unicité/cap, preuve directe non doublée.
- Préflight technique : 16 requêtes, 54 473 références, 30 276 sorties,
  poids technique fixe 0,5; **aucun label ni résultat de qualité consulté**.
  Ce poids n'est pas une variante évaluée ni un paramètre choisi pour le run.
- Audit reproductible indépendant : recalcul des probabilités internes depuis
  les scores scellés, optimum de chaque poids, six splits, replay de tout le
  Master, exact/opérationnel depuis ledger+sidecars, naturalité et attribution
  exhaustive. Résultat : 823 exacts, 825 opérationnels, 6 gains, 5 pertes,
  898 vérités dans le réservoir, cap2000. Hashes vérifiés.

| Phase | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Calibration et Master | 10,797 s | 4,66 Gio | 0 |
| Évaluation | 0,209 s | 0,79 Gio | 0 |
| Audit indépendant | 2,743 s | 1,90 Gio | 0 |

Validation préalable des hashes exclue. Ce ne sont pas des latences produit
end-to-end. Mac/CATNAT uniquement, temporaires externes, aucun coût externe.
Le faible espace interne n'a nécessité aucune suppression de fichier utilisateur.

## Scellement et limites

Configuration SHA-256 :
`78d37b7cdebea46be1576114fa62b7481d691fc0e8a2220d392d1f370a742704`.
Racine des artefacts : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

- Préflight `retrieval_v24_preflight/78d37b7cdebea46b/manifest.json` :
  `196f7e9af00ad90e11aa31d63aa701d581f23977d43fa77a7f39b4c054b47b48`.
- Génération `retrieval_v24_generation/78d37b7cdebea46b/manifest.json` :
  `fdbe26c8f24394f6b0099a42f0735eb9ed4b21d9b2c4d4364732fb68c0a37961`.
- Évaluation `retrieval_v24_evaluation/78d37b7cdebea46b/manifest.json` :
  `cac499c5b326cff564c2d9a30305308272a1d5f34de5d8ff86e89754dd191bdb`.
- Audit `retrieval_v24_independent_audit/78d37b7cdebea46b/manifest.json` :
  `4bf61d778b5733fec87aad262acba9e1c08423ddbeef076825943b36ab5ce781`.

Les autorités v21/v23 et les sidecars officiels sont référencés avec leurs
hashes dans les manifestes; le script d'audit est
`scripts/audit_retrieval_v24_nested_fusion.py`. Aucun paramètre de recette
n'a changé après résultat. Aucun fichier des anciennes versions n'est modifié.

Pas de national, pas de PreRank500/Recall100/Wilson/SLO validés, fold0/fold1
fermés, aval gelé. Le panel demeure adapté par plusieurs cycles de stratégie;
la séparation OOF ne le rend pas vierge. Une validation cross-CRM nécessitera
un export réellement indépendant, scellé avant décision. `PIVOT_GENERATION`
ne vaut ni promesse d'atteindre 99,2 %, ni autorisation d'ouvrir fold1.
