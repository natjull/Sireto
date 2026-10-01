# V36–V37 : progrès mesuré, défaut de sélection amont encore ouvert

Le 2 octobre 2026, le recall SIRET humain exact à 100 passe de
**95,01953 % à 96,97266 % sur les mêmes 1 024 requêtes**. Ce progrès ne
ferme pas le gap vers 99,2 %. Le verdict reste **PIVOT_GENERATION**.
Ce sont des pilotes nested OOF de développement, pas le résultat complet
des 23 571 requêtes, ni une preuve cross-CRM ou un benchmark SOTA.

## Expériences et résultats appariés

| Recette | Exact@100 | Exact@500 | Master exact@2000 |
|---|---:|---:|---:|
| Ordre heuristique v30 | 910 / 1024 = 88,86719 % | 996 = 97,26563 % | 1018 = 99,41406 % |
| Fusion v35 | 973 = 95,01953 % | 1000 = 97,65625 % | 1018 = 99,41406 % |
| V36 primaire | 990 = 96,67969 % | 1010 = 98,63281 % | 1018 = 99,41406 % |
| V36 ablation, anciennes features / nouveau fit | 990 = 96,67969 % | 1005 = 98,14453 % | 1018 = 99,41406 % |
| V37 primaire | **993 = 96,97266 %** | **1010 = 98,63281 %** | 1018 = 99,41406 % |

Les vues opérationnelles sont calculées séparément ; elles coïncident ici
avec l'exact à ces trois profondeurs, pour chaque recette. Il n'y a aucune
promotion exclusivement opérationnelle à 100 dans v37. Cinquante requêtes
ont un équivalent actif dans les 100 alors que la vérité exacte est fermée ;
ce n'est pas une mesure de précision top1. Vingt requêtes ont plusieurs
équivalents actifs selon le sidecar officiel.

- V36 primaire contre v35 à 100 : 22 gains / 5 pertes, +17 net.
- Ablation v36 contre v35 à 100 : 21 gains / 4 pertes, +17 net.
- V37 contre v36 primaire à 100 : 3 gains / 0 perte.
- V37 contre v35 à 100 : 23 gains / 3 pertes, +20 net.
- V37 contre l'ordre heuristique v30 à 100 : 84 gains / 1 perte, +83 net.
- V37 contre v36 à 500 : 1 gain / 1 perte ; contre v35 : 10 gains / 0 perte.

V37 a une borne basse Wilson 95 % de **95,73502 %** à 100.
Les folds externes ont respectivement 314/328, 341/348 et 338/348 hits
à 100, soit 95,73171 %, 97,98851 % et 97,12644 %. À 500 : 324, 343 et 343.
Aucun gate produit n'est franchi. Il faudrait conserver au moins 1016/1024
pour atteindre ponctuellement 99,2 %, soit encore 23 récupérations nettes.
Même un classement parfait de ce Master pilote aurait une borne Wilson
basse de 98,72757 %, insuffisante pour le gate de confiance à 99,0 %.

## Ce qui a changé, et ce que l'ablation permet de dire

V36, préenregistré dans `1c375be3`, ajoute des noms officiels séparés par
portée/provenance, des distances XY pour tous les sites coordonnés et les
rangs/scores des 14 canaux directs/structurés. Les 51 features v35 deviennent
131. Les preuves restent liées à leur entreprise ou à leur site réel.
Ni numéro discordant ni distance ne devient un filtre d'éligibilité.

L'apprentissage utilise tous les négatifs naturels au lieu de 128 par
requête, avec des paires LambdaMART `topk=100` plutôt que `mean=8`. Les
équivalents opérationnels sont neutralisés, les absents restent des erreurs
dans la mesure. Les autres hyperparamètres sont inchangés, sans recherche
de paramètres ni early stopping sur le fold externe.

L'ablation préenregistrée conserve les 51 features mais utilise le nouveau
fit. Elle atteint le même nombre de hits à 100 que v36 primaire. Donc :

- l'apport net des nouvelles features v36 à 100 **n'est pas établi** ;
- leur apport net à 500 par rapport à cette ablation est de 5 requêtes ;
- le changement conjoint de concurrence et d'objectif de paires explique
  le gain observé par l'ablation, sans isoler causalement ces deux composants.

V37, préenregistré dans `6648ec66`, restaure les 26 canaux de découverte
parent de la configuration v16 : scores relatifs et rangs par canal, soit
52 features supplémentaires, 183 au total. Ils existaient dans les sources
scellées mais étaient auparavant fortement agrégés avant ce modèle.
Tous les canaux sont inclus, pas seulement ceux utiles aux erreurs vues.
Les paramètres de fit v36 et les candidats sont inchangés.

Sur les 28 pertes de classement v36, 15 parents avaient une preuve dans
ces canaux, dont 9 dans un top10. Le gain réel est de 3, **pas de 15**.
Une preuve disponible n'est pas nécessairement discriminante ou exploitable
par le modèle actuel.

## Le défaut conceptuel encore présent

Le modèle enrichi intervient APRÈS une coupe qui n'utilise pas ces preuves
en détail. `make_master` dans `src/xgb_matcher/retrieval_v23_joint_parent.py`
combine encore `direct_rrf` et, pour un site local,
`1 / (60 + learned_parent_rank + joint_site_rank)`, puis coupe à 2 000.
Le parent appris upstream n'utilise que ses 16 features historiques ; son
membership n'est pas modifié par v36/v37. Le score conjoint historique
multiplie en outre nom et voie par l'accord de numéro lorsqu'il est fourni.
Ces heuristiques sont des signaux en aval mais restent décisives en amont.

Sur le corpus complet scellé, 111 vérités présentes au réservoir sont
évincées avant le nouveau modèle. Sur les 245 absences du Master :
130 parents non retrouvés, 3 sites non développés, 111 évictions et 1 absence
du catalogue. Master@2000 = **98,96059 %**. Aucun reclassement de ce même
Master ne peut produire 99,2 % à 100. Le réservoir naturel lui-même atteint
99,43150 %, donc offre très peu de marge sur le gate 99,4 % à 500.

V37 perd encore 31 requêtes du pilote : 3 parents absents, 3 évictions vers
Master, 8 évictions Master→500 et 17 évictions 500→100. Le bon SIREN apparaît
dans les 100 pour 995 requêtes contre 993 SIRET exacts : **29 des 31 échecs
n'ont donc pas le bon parent admis**. Ce n'est principalement ni un mauvais
choix entre frères du même SIREN ni un simple manque de voisins XY.

Le diagnostic exploratoire v36 indique aussi une faible similarité de noms
officiels sur beaucoup de pertes : 19/28 ne partagent aucun token de nom
avec les vues examinées. Enseigne, équipement, groupe, exploitant et ancien
établissement peuvent désigner des réalités liées mais non identiques.
Cela motive des preuves relationnelles/temporaires avec provenance ; cela
ne permet pas de déclarer les labels humains faux ou de les supprimer.
L'audit v27 avait déjà montré que les correspondances historiques exactes
qu'il avait trouvées existaient dans le canal temporel : ne pas présenter
une simple réintroduction de ces historiques comme une nouvelle couverture.

## Orientation suivante, non encore validée

Le prochain cycle doit déplacer l'apprentissage utile AVANT les pertes
irréversibles, et non multiplier les petits réglages du même top100 :

1. Construire une représentation du dossier entreprise à partir des canaux
   officiels distincts, des meilleures preuves conjointes réellement portées
   par un site, de leur disponibilité, temporalité et incertitude BAN.
   Conserver la distinction entre proximité lexicale et relation structurée.
2. Apprendre la sélection des parents en compétition avec le réservoir
   naturel. Produire de nouveaux scores ET memberships nested OOF ; aucun
   score entraîné sur le fold cible, même indirectement, ne peut définir son
   pool de fit. Ne pas appliquer un modèle SIRET à un réservoir dépassant le
   plafond de 2 000 pour contourner le contrat.
3. Définir une seule allocation parent→sites avant le cap 2 000, apprise ou
   calibrée dans l'inner OOF, puis le LambdaMART SIRET unique jusqu'à 500/100.
   Tester explicitement l'hypothèse hiérarchique : soutien à l'entreprise et
   choix conditionnel de son site. Pas de quota rigide, ni de multiplicateur
   arbitraire choisi sur les erreurs externes.
4. Évaluer d'abord, avec les caches, si cette nouvelle sélection conserve
   effectivement plus de vérités sous les caps, puis passer le gate500.
   Une correction d'admission seule ne résout pas les parents absents ; les
   relations officielles manquantes ou un canal complémentaire doivent faire
   l'objet d'un contrat séparé si la génération reste insuffisante.
5. Ne lancer le replay complet coûteux qu'après une recette figée et un
   préflight de coût. Publier aussi les régressions, les segments, l'exact et
   l'opérationnel ; garder le fold1 fermé jusqu'aux gates et à l'autorisation.

Ce plan n'est pas une affirmation que le nouvel apprentissage atteindra la
cible. L'usage répété du panel1024 en fait un outil de développement, pas
une validation indépendante. Une famille de paramètres ne doit pas être
balayée sur ses résultats OOF. Les cas examinés ne deviennent jamais des
exceptions, alias CRM ou lookups de labels.

La [recherche NAACL2024 sur la généralisation des bloqueurs](https://aclanthology.org/2024.naacl-long.483.pdf)
montre pourquoi un modèle neural performant sur un benchmark ne suffit pas
à garantir la couverture dans un autre régime de candidats. La
[documentation XGBoost](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/learning_to_rank.html)
décrit les paires de ranking ; optimiser NDCG avec des paires top100 ne
constitue pas une optimisation exacte ni une garantie de Recall@100.
Ces références guident les hypothèses, pas une revendication SOTA.

## Intégrité, segments et coût

Même panel label-indépendant préchoisi que v35 ; candidats outer inchangés,
fit sur les Masters inner appropriés. Caps2000/500/100, sidecars officiels
et hashes contrôlés. Aucun positif injecté, aucune feature identifiante,
folds0/1 non ouverts, modèles aval inchangés. Pas d'hydratation riche ou de
SLO end-to-end de production mesuré. Aucun déploiement.

Le complément de reporting `b6e840f0` rejoint les métadonnées du ledger
scellé et publie, pour les quatre recettes, folds, historique/label_kind,
V2/V3, origine, état, unseen_siren, new_site_known_siren et mega_base_pool.
**Les V2/V3 sont vides dans les 1 024 lignes du ledger source**, et
mega_base_pool est absent partout : ils sont explicitement publiés UNKNOWN,
pas présentés comme des qualifications disponibles. Les booléens historiques
de statut SIREN ne prouvent pas l'exposition aux fits actuels.
Les 16 labels sources absents/inexploitables restent publiés séparément
(8/3/5) ; aucune des 1 024 requêtes n'est retirée du dénominateur.

Quelques segments v37 (exact et opérationnel identiques à100 ici) :

| Segment source | Dénominateur | Hits à100 | Hits à500 |
|---|---:|---:|---:|
| CRM20260817 | 605 | 583 | 594 |
| CRM historique | 419 | 410 | 416 |
| État humain actif | 757 | 743 | 749 |
| État humain fermé | 174 | 165 | 171 |
| État humain inconnu | 93 | 85 | 90 |
| MATCH_EXACT | 931 | 908 | 920 |
| AMBIGUOUS | 16 | 16 | 16 |
| UNRESOLVED | 77 | 69 | 74 |

Les trois familles de segmentation se recouvrent ; ne pas sommer leurs
dénominateurs. Ces catégories n'interviennent ni dans les features ni dans
le choix des candidats. Les détails par recette sont dans l'audit JSON.

Le premier processus v36 a terminé le primaire puis a été stoppé par la
garde de ressources pendant le dernier fit d'ablation. Son STOP est conservé.
Le déclencheur exact RSS/swap n'avait pas été enregistré : ne pas conclure
à un OOM démontré et ne pas annoncer un PASS pour ce premier processus.

La reprise physique `6509551b` isole chaque fold dans un processus neuf,
sans modifier recette, données ni paramètres. Les prédictions reproduisent
exactement les 64 shards déjà terminés des folds2/3. Reprise en84,17s,
pic1,48Gio, zéro nouveau swapout. V37 complet en136,97s, pic3,53Gio,
zéro nouveau swapout, deux threads ; chaque fold dure environ43s.
Aucune génération nationale refaite. **23 tests ciblés passent.**

## Autorités reproductibles

Sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/` :

| Artefact | SHA256 |
|---|---|
| `retrieval_v36_evidence_ranker/pilot_20261002_a/primary/manifest.json` | `422ea7cc24524f9e71b01122bc07a33d5a40af6d58caead7ec81078c6ea8fb5f` |
| `retrieval_v36_evidence_ranker/recovery_20261002_a/manifest.json` | `97abf5e886068ccd878304f8f44ddec4efe470a5ddd7798949c9a9e602e471db` |
| `retrieval_v37_parent_evidence/pilot_20261002_a/manifest.json` | `f6386d054f7b68186f6e7f8550efe7052548b8f942efbda949e6427655e7e018` |
| `retrieval_v37_parent_evidence/pilot_20261002_a/reporting_audit.json` | `7e5f98542549355b5b517deb44bf9d307fb2d17b5fb0aabd500aa4a1a8a63db9` |

Le fichier de résultats primaire v37 s'appelle `ablation_outcomes.parquet`
par réutilisation du helper d'agrégation ; le manifeste le référence bien
comme primaire. Il n'existe pas d'ablation concurrente v37 sélectionnée
après coup. Aucun job suivant lancé automatiquement, aucun push.

Conclusion : **PIVOT_GENERATION**, amélioration de classement sur le pilote,
mais génération complète insuffisante et gates500/100/confiance/SLO non
validés. **Pas de READY_FOR_FOLD1.**
