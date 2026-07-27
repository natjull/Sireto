# V4.1 — Résultats du shadow actif-courant

Date : 27 juillet 2026  
Verdict de phase : **`PIVOT_CERTIFICATION`**

## Conclusion directe

La chaîne V4.1 est implémentée, reproductible et exécutable entièrement sur le
Mac M4 Pro et le SSD externe, sans GPU ni service payant.

Les résultats de développement autorisent la chaîne technique :

- le retrieval A conserve 305/305 bons SIRET sur le dev à Top-100 ;
- le ranker R1 atteint 1 216/1 217 = 99,918 % Hit@1 sur les cas exacts du dev ;
- l'accepteur atteint 1 186/1 188 = 99,832 % de précision observée et
  81,593 % de couverture sur le dev.

Le shadow complet produit cependant seulement 10 292 `AUTO_MATCH` sur
19 025 lignes, soit **54,097 % de couverture opérationnelle observée**. Les
8 733 autres lignes sont envoyées en revue. Comme ce shadow n'a aucune vérité
terrain indépendante, il est interdit d'en déduire une précision réelle ou de
baisser le seuil pour augmenter artificiellement la couverture.

La V4.1 n'est donc ni un échec d'architecture ni un GO production. Le blocage
restant est la certification sur de nouvelles données indépendantes, avec un
écart de distribution visible entre le dev et le CRM shadow.

## Retrieval gelé

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/retrieval_v41_dev_feede27`

Les trois variantes A, B et C obtiennent 305/305 = 100 % de Recall@100 SIRET
exact sur le dev autorisé, sans candidat fermé et sans dépassement de 100.
La variante A est retenue par le tie-break préenregistré : elle est la plus
simple et sa latence p95 est la meilleure.

| Variante | Recall@100 | Max candidats | Fermés | Latence p95 |
|---|---:|---:|---:|---:|
| A — sparse actif | 305/305 | 100 | 0 | 872,6 ms |
| B — A + preuve SIRET/SIREN | 305/305 | 100 | 0 | 1 103,1 ms |
| C — B + alias fermé | 305/305 | 100 | 0 | 1 019,2 ms |

Le correctif décisif a consisté à fusionner séparément les classements nom,
adresse et secours, au lieu de comparer directement leurs scores TF-IDF non
commensurables. Le seul miss antérieur passe ainsi du rang 131 au Top-100.

Ce résultat est un gate de développement sur 305 cas, pas une nouvelle
certification statistique.

## Dataset et modèles

Dataset :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`

- 7 003 requêtes autorisées ;
- 5 883 `MATCH_EXACT` et 1 120 `AMBIGUOUS` ;
- 698 428 paires requête-candidat ;
- 6 vérités exactes absentes du Top-100, conservées comme erreurs ;
- zéro identifiant des deux tests consommés ;
- zéro positif injecté ;
- uniquement des candidats actifs, avec 100 candidats maximum.

Bundle :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155`

Release : `f1058826e20ecad4`

Le split train/dev et les cinq folds OOF sont construits par composantes
connexes entre SIREN d'entrée et SIREN cible. Le ranker R1 ajoute uniquement
des relations et preuves de retrieval, jamais les identifiants bruts.

| Mesure dev | R0 | R1 retenu |
|---|---:|---:|
| Hit@1 SIRET exact | 99,836 % | 99,918 % |
| Cas exacts | 1 217 | 1 217 |
| Régression segmentaire maximale | — | 0 point |

L'accepteur est une régression logistique standardisée, sans isotonic. Son
seuil `0,4631331627` a été sélectionné une fois sur le dev :

- 1 188 AUTO sur 1 456 scènes ;
- 2 erreurs observées ;
- précision observée 99,832 % ;
- couverture 81,593 % ;
- aucun test ou holdout consulté.

Le champ `confidence` est explicitement de type
`ROUTING_SCORE_UNCALIBRATED` et ne doit pas être présenté comme une
probabilité de justesse.

## Shadow complet

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/shadow/v4_1/runs/v41_shadow_f1058826_20260727_v3`

| Mesure | Résultat |
|---|---:|
| CRM brut | 23 609 |
| Lignes autorisées et scorées | 19 025 |
| Lignes exclues et non scorées | 4 584 |
| `AUTO_MATCH` | 10 292 (54,097 %) |
| `REVIEW` | 8 733 (45,903 %) |
| `REVIEW_LOW_CONFIDENCE` | 8 648 |
| `REVIEW_NO_ACTIVE_CANDIDATE` | 85 |
| Latence p50 | 140,8 ms |
| Latence p95 | 982,0 ms |
| Latence maximale | 11,49 s |
| Candidats Top-10 exportés | 188 975 |
| Précision mesurée | aucune |

Répartition de la couverture AUTO selon l'état du SIRET CRM :

| État du SIRET CRM | Lignes | AUTO | Couverture AUTO |
|---|---:|---:|---:|
| Actif | 13 723 | 8 060 | 58,73 % |
| Fermé | 4 473 | 1 901 | 42,50 % |
| Introuvable | 819 | 326 | 39,80 % |
| Invalide | 10 | 5 | 50,00 % |

Invariants vérifiés :

- 19 025 décisions et 19 025 `SERVICE ID` uniques ;
- zéro ligne de test consommée scorée ;
- zéro écriture dans le CRM ;
- tous les candidats exportés sont actifs ;
- aucun pool ne dépasse 100 candidats ;
- les hashes des sorties et des neuf artefacts modèles correspondent ;
- inventaire, dataset, retrieval, release, code et checkpoint sont liés ;
- 206 tests passent.

Le shadow a duré environ 1 h 34. Paris provoque une pointe mémoire proche de
8 Go et une latence nettement supérieure aux petites communes. Cela ne remet
pas en cause la faisabilité sur le Mac, mais doit rester un axe d'optimisation
avant exploitation récurrente.

## Incidents détectés et corrigés

Le fail-closed du runner a révélé deux défauts avant publication :

1. le hash des composants modèles était transmis au validateur du retrieval
   au lieu du validateur de release ;
2. le magasin rapide de candidats omet certains SIRET connus du stock SIRENE
   complet. Le stock complet figé est désormais autoritaire pour l'état
   `ACTIVE`/`CLOSED`, sans inventer de candidat manquant. Toute contradiction
   réel `ACTIVE` contre `CLOSED` reste bloquante.

Les runs interrompus avant correction n'ont publié aucun dossier final. Le
run V3 est le seul artefact shadow à retenir.

## Pourquoi `PIVOT_CERTIFICATION`

Le dev et le shadow ne répondent pas à la même question :

- le dev possède des labels et permet de choisir puis geler les modèles ;
- le shadow montre le comportement sur le CRM disponible, mais ses lignes
  sont historiquement contaminées et non labellisées indépendamment.

L'écart de couverture entre 81,593 % sur le dev et 54,097 % en shadow signale
un changement de difficulté ou de distribution. Il ne permet pas de savoir si
les REVIEW sont excessivement prudents ou réellement nécessaires.

Le prochain pas scientifique n'est donc ni un nouveau modèle, ni un réglage du
seuil sur le shadow. Il faut :

1. geler un prochain snapshot CRM réellement nouveau ;
2. exécuter V4.1 une seule fois sans adaptation ;
3. auditer humainement les décisions AUTO et un échantillon structuré des
   REVIEW ;
4. publier précision SIRET exacte, couverture et intervalles de confiance ;
5. décider ensuite `GO`, `PIVOT` ou `STOP`.

Pour revendiquer 99,8 % avec zéro erreur et une borne unilatérale à 99 %, il
faudra environ 2 300 décisions AUTO indépendantes auditées. Avant cela, V4.1
reste une architecture candidate techniquement validée, non certifiée.

## Provenance Git

- contrat : `eea75f2` ;
- retrieval actif : `a599e4a` ;
- modèles et décision : `f158da2`, `942a443`, `c4ffb2a` ;
- inventaire et exports : `af18779` ;
- évaluation retrieval : `85f7674` ;
- dataset sans fuite : `ab13fb4`, `9fd30d8`, `feede27` ;
- fusion sparse séparée : `993e088` ;
- intégrité release/modèles : `d86f6f6`, `41cbc0e`, `8e96961` ;
- autorité du stock SIRENE au shadow : `cc5dec1`, `9a322bc`.
