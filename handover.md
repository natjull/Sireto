# SIRETO Handover - 26 Juillet 2026

## Etat des lieux
Le chantier **Retrieval sélectif SIRET Recall@100** est terminé avec une
décision contractuelle **`PIVOT`**. Sur le test final gelé, la qualification
V3 conserve 2 128/2 652 dossiers exacts, soit 80,241 % de couverture, et
l'admission gelée place le bon SIRET dans 100 candidats pour 2 116/2 128,
soit **99,436 % de Recall@100**. Tous les gates globaux passent. Le `PIVOT`
vient uniquement de deux gates de stabilité de couverture : établissements
fermés et mégapoles. Le test est désormais définitivement fermé à tout tuning.
Contrat : `docs/retrieval_selective_recall100_contract.md`. Rapport :
`reports/recall100/selective_test_certification.md`.

L'experience V9 sans GPU precedente est terminee avec une decision `PIVOT`.
Ses pools denses multicanaux ne sont pas promus. Le rapport reste
`reports/v9/v9_go_pivot_stop.md`.

V7/V8b et Route B restent physiquement disponibles comme baselines legacy.
Le gate retrieval n'a modifié aucun modèle aval. La phase aval est désormais
ouverte sous le contrat séparé
`docs/downstream_selective_matching_contract.md`, sans réutiliser le test
sélectif consommé.

La qualification V4 « SIRET actif au snapshot » est maintenant exécutée sans
score modèle et sans test. Elle corrige les cinq conflits bloquant E2b et
publie 4 060 exacts train / 872 exacts dev, tous actifs et soutenus par une
correspondance directe unique. Leur compatibilité avec le top-100 actuel est
excellente : 4 058/4 060 = 99,951 % train et 872/872 = 100 % dev. Le ranker E1
atteint 96,034 % / 94,954 % Hit@1 sur ce noyau. Mais V4 échoue à son gate
pré-enregistré : couverture ~34 % au lieu de 50 %, moins de 5 000 exacts train
et 14 SIREN actuels partagés entre les anciens splits. Verdict :
`STOP_V4` sur ce corpus seul, sans assouplissement post-hoc.

Ce blocage est désormais levé par **V4-Fresh** : 6 330 lignes CRM absentes du
benchmark ont été qualifiées avec la même règle, puis séparées par hash SIREN.
Elles ajoutent 819 exacts au fit, 305 au nouveau dev et 302 au holdout scellé.
Le fit combiné atteint **5 751 exacts**, avec zéro SIREN exact partagé entre
fit/dev/holdout. Verdict : **`PASS_V4_FRESH`**. Le holdout n'a reçu aucune
prédiction modèle. Rapport : `reports/v9/v4_fresh_expansion_results.md`.

Le gate retrieval V4 est maintenant franchi : l'admission déterministe gelée
conserve 5 749/5 751 = **99,965 %** des vérités du fit combiné et 305/305 =
**100 % observé** sur le nouveau dev indépendant, toujours avec 100 candidats
maximum. Les 1 124 cas frais ont tous leur vérité visible et conservée. Verdict
de phase : **`GO_RANKER_V4`**. Ce résultat autorise l'entraînement aval ; avec
305 cas dev, il ne constitue pas une garantie statistique de 99 % en
production. Rapport : `reports/v9/v4_retrieval_gate_results.md`.

Le ranker V4 est également validé sur ce nouveau dev : **299/305 = 98,033 %**
Hit@1 SIRET, contre 290/305 = 95,082 % pour l'ancien ranker compatible.
Il corrige dix erreurs et dégrade un ancien succès. Verdict de phase :
**`GO_ACCEPTEUR_V4`**. Le holdout reste fermé. Rapport :
`reports/v9/v4_ranker_e1_results.md`.

## Actions terminees (fenetre recente)
- **Ranker V4 validé sur le nouveau dev** : dataset de 604 938 paires,
  5 749 requêtes fit et 305 dev, 55 features déterministes, exactement un
  positif réel par requête, aucune injection et aucun SIREN exact partagé.
  Le nouveau XGBRanker atteint 299/305 = 98,033 % Hit@1 SIRET contre
  290/305 = 95,082 % pour l'ancien ranker épinglé ; comparaison appariée :
  dix erreurs corrigées, une créée. Verdict `GO_ACCEPTEUR_V4`. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/ranker_v4/1aebeada820d92a7`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/ranker_v4/ranker_1aebeada820d92a7_6236365`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/ranker_v4_e1_250a05f`.
  Suite complète à 140 tests passants. *(commits GitHub : contrat `0c90c25`,
  builder `6236365`, évaluateur `250a05f`, rapport `13e3547`)*
- **Gate retrieval V4 franchi sans GPU** : contrat gelé avant calcul, reprise
  des 4 932 anciennes listes et reconstruction des seuls 1 124 cas frais.
  Recall@100 : noyau historique 4 930/4 932 = 99,959 %, ajout fit
  819/819 = 100 %, fit combiné 5 749/5 751 = 99,965 %, nouveau dev
  305/305 = 100 %. Zéro dépassement de 100, zéro positif injecté, zéro SIREN
  exact partagé fit/dev, holdout et ancien test non lus. Les deux misses sont
  des scènes fit historiques visibles dans les canaux mais éliminées par
  l'ancienne admission. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/retrieval_v4/ddefe3daaacdf5ef`.
  Suite complète à 136 tests passants. *(commits GitHub : contrat `510868b`,
  builders/tests `e566c25`, rapport `6948aa1`)*
- **Expansion V4-Fresh passée sans réutiliser le benchmark** : les 6 330
  `SERVICE ID` absents du benchmark ont fourni 1 426 SIRET actifs uniques,
  247 ambigus et 4 657 non résolus. Séparation gelée : `fit_addition`
  819 exacts, `dev_new` 305, `holdout_sealed` 302. Le fit combiné avec le noyau
  V4 contient 5 751 exacts. Zéro chevauchement de SIREN exact entre les trois
  rôles, zéro identifiant déjà présent dans le benchmark, zéro SIRET fermé.
  Le holdout est hashé mais non évalué. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f`.
  Suite complète à 132 tests passants. *(commits GitHub : contrat `1c2e84c`,
  builder/tests `613cf7d`, rapport `d8d36b9`)*
- **Qualification V4 actuelle pré-enregistrée puis exécutée** : examen de
  toutes les lignes actives de la partition géographique, sans top-k, rang,
  score ni décision modèle. V4 produit 4 060/11 837 = 34,299 % exacts train et
  872/2 565 = 33,996 % exacts dev ; 759 SIRET et 351 SIREN changent face à
  l'historique, et les cinq conflits E2b sont corrigés. Chaque exact a une
  preuve active unique. Gate `STOP_V4` : couverture <50 %, train <5 000 exacts
  et 14 SIREN V4 traversent l'ancien split. Diagnostic post-qualification :
  Recall@100 99,951 % train / 100 % dev et ranker E1 Hit@1 96,034 % /
  94,954 %. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759`.
  Suite complète à 129 tests passants. *(commits GitHub : contrat `ce82b01`,
  builder/tests `799c32d`, rapport `299bc8a`)*
- **E2b pré-enregistré puis exécuté sans test** : comparaison fermée de la
  régression logistique standardisée et de XGBoost avec score brut, sigmoid ou
  isotonic. Le brut logistique gagne : 85/1 280 = 6,641 % AUTO à 100 % observé
  contre 33/1 280 = 2,578 % pour XGBoost isotonic, mais le gate de 25 % échoue.
  Les 320 premiers scores ne comportent que cinq erreurs formelles ; les cinq
  prédisent un SIRET dont le nom/adresse SIRENE correspondent directement au
  CRM, tandis que le label désigne une autre entité, une ancienne entité ou
  `UNRESOLVED`. Exemples : VISSELECT actif contre AVENIS fermé, PGDIS contre
  OFFICE DEPOT, LMP SANTE actif contre LMP SANTE fermé. Verdict formel
  `STOP_E2B`, lecture architecturale `PIVOT_DATASET_AVAL`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_e2b_3171ef5020c0f068_070c123`.
  Suite complète à 123 tests passants. *(commits GitHub : contrat `cf91432`,
  code `070c123`, rapport `ebb4bf2`)*
- **Expérience aval E1/E2 exécutée sur train/dev** : dataset immuable de
  1 438 845 paires, 100 candidats maximum, zéro doublon, zéro détail manquant
  et Recall@100 V3 de 99,162 % train / 99,572 % dev. Le ranker final atteint
  1 754/2 104 = 83,365 % Hit@1, soit +2,804 points sur l'ancien ranker, avec
  gains actifs, fermés et multi-sites. L'accepteur XGBoost calibré ne couvre
  que 33/1 280 = 2,578 % à 100 % observé ; le palier suivant tombe déjà à
  98,507 %. Verdict `PIVOT_ACCEPTEUR`, sans retour au risk model historique,
  sans dense, sans GPU et sans ouverture du test. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_3171ef5020c0f068_fc9cb1b`.
  Suite complète à 121 tests passants. *(commits GitHub : correction builder
  `fc9cb1b`, rapport `9ab7f6a`)*
- **Socle de l'expérience aval E1/E2** : builder train/dev alimenté par les
  listes top-100 gelées, provenance alignée sur les sept canaux sparse
  réellement certifiés, ranker déterministe par défaut, folds OOF groupés par
  SIREN et courbes accepteur aux points 99,0/99,5/99,8 %. Le canal dense
  abandonné n'est plus utilisé comme faux signal de provenance. Le canal
  `current_sparse` audité peut servir directement de baseline train, sans
  recalcul redondant. Suite complète à 120 tests passants. *(commits GitHub :
  `0a75b73`, `dbd8906`, `2c24052`)*
- **Contrats aval exact-SIRET renforcés** : déduplication obligatoire des SIRET
  avant toute scène, plafond absolu 100, preuves top-1/top-2 et deltas
  nom/adresse transportés jusqu'à l'accepteur, folds OOF par SIREN et
  évaluation du holdout final désactivée par défaut. Une autorisation liée à
  un nouveau dataset est obligatoire pour l'ouvrir; le test sélectif consommé
  est explicitement refusé. Suite complète à 119 tests passants. *(commit
  GitHub : `aeeaf0f`)*
- **Contrat matching aval gelé** : trajectoire `top-100 gelé → ranker SIRET
  unique → accepteur exact-SIRET`, première expérience sans sémantique ni GPU,
  vraies scènes OOF, publication end-to-end et nouveau holdout indépendant
  obligatoire. *(commit GitHub : `c18bf28`)*
- **Audit reproductible de l'architecture aval** : la référence historique
  contient 1 428/2 512 scènes avec le même SIRET en top-1 et top-2. Toutes sont
  AUTO. Sur les scènes réellement distinctes, la couverture tombe à 40,959 %
  et la précision brute à 98,649 %. Le fichier versionné donne 1 866/1 872 =
  99,679 %, pas 99,84 %. Le decider garde toutefois un signal utile de +3,03 à
  +4,66 points Hit@1 sur les mêmes scènes. Les risk models ciblent le SIREN et
  1 539/16 621 lignes V7 ont le bon SIREN mais le mauvais SIRET. Verdict :
  `PIVOT AVAL`, conserver les preuves dans `ranker final + accepteur
  exact-SIRET`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/downstream_architecture_audit_a59fb0f`.
  *(commits GitHub : runner/tests `a59fb0f`, rapport `e186439`)*
- **Audit de stabilité V3 limité à train/dev** : décomposition reproductible
  des pertes entre contradictions structurelles V2 et absence de preuve V3,
  avec refus explicite du split test. La couverture V3 vaut 79,632 % sur train
  et 82,027 % sur dev. Pour les fermés, elle vaut déjà 65,055 % sur train et
  69,405 % sur dev : la difficulté n'est pas créée par le test. Parmi les
  V2 exacts écartés faute de preuve, le nom et l'adresse sont tous deux
  éloignés dans 774/1 569 cas train et 131/296 cas dev; un simple assouplissement
  des seuils ne traite donc pas la cause dominante. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/qualification_stability_train_dev_111b07c`.
  Suite complète à 109 tests passants. *(commit GitHub : `111b07c`)*
- **Certification finale selective sur test** : qualification V2/V3 produite
  avant tout retrieval, puis exécution unique de la configuration gelée.
  Couverture V3 2 128/2 652 = 80,241 %, Recall@100 V3
  2 116/2 128 = 99,436 %, oracle interne 2 128/2 128, maximum 100 candidats et
  zéro dépassement. Les gates globaux passent. Les segments fermés
  (62,633 % de couverture) et mégapoles (77,586 %) ratent uniquement leurs
  planchers de stabilité; leurs recalls atteignent 98,305 % et 99,259 %.
  Verdict pré-enregistré `PIVOT`; aucune nouvelle variante ne doit être testée
  sur ce test. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/certification/selective_test_c33b80855f560074_6fab035`.
  Suite complète à 105 tests passants. *(commits GitHub : `d1c0fc9`,
  `6fab035`, rapport `41ff2e1`)*
- **Contrat final pré-enregistré avant ouverture du test** : double gate
  couverture V3 ≥80 % et Recall@100 exact ≥99 %, oracle sans vérité invisible,
  plafond strict 100, publication historique/V2/V3 et stabilité segmentaire.
  L'admission, les seuils, les hashes de snapshots et le runner de qualification
  ont été gelés avant l'évaluation. *(commits GitHub : `4f6e317`,
  `eb0e6a3`)*
- **Qualification V3 par preuve directe** : politique indépendante du
  retrieval séparant `NAME_AND_ADDRESS`, `NAME_ONLY`, `ADDRESS_ONLY` et
  `NO_DIRECT_EVIDENCE`. Un label V2 exact sans preuve directe devient
  `UNRESOLVED`, sans promotion automatique d'un autre SIRET. Dev :
  2 104 exacts, 81 ambigus, 380 non résolus, couverture 82,027 % et
  Recall@100 gelé 99,572 %. Test qualifié avant retrieval : 2 128 exacts,
  105 ambigus, 419 non résolus et couverture 80,241 %. *(commits GitHub :
  `09b9d46`, `cf7133c`, `c6c8186`)*
- **Qualification V2 train/dev sans suppression ni relabel automatique**:
  politique indépendante des résultats du retrieval, builder immuable et
  double publication des métriques historique/V2. Sur dev, 2 400/2 565
  restent `MATCH_EXACT`, 81 deviennent `AMBIGUOUS` et 84 `UNRESOLVED`.
  L'admission passe seulement de 2 495/2 565 = 97,271 % à
  2 343/2 400 = 97,625 % sur le périmètre exact: il manque encore 33 succès au
  gate. L'oracle interne atteint 2 394/2 400 = 99,750 %, avec 6 vérités non
  vues et 51 vues puis éliminées. Sur train, 10 995 labels restent exacts,
  440 deviennent ambigus et 402 non résolus. Aucun SIRET alternatif n'est
  promu; le benchmark original et le test restent inchangés. Artefacts:
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v2/522351669d5313dc`
  (dev) et `.../f8af7e1da18fa94a` (train). Rapport:
  `reports/recall100/benchmark_v2_qualification.md`. Suite complète à 99 tests
  passants. *(commits GitHub: `16e657e`, `a68f679`)*
- **Audit global de non-unicité des labels exacts**: nouveau runner immuable
  comparant, pour chaque requête, le label aux autres SIRET du même SIREN via
  la clé d'adresse canonique. Sur les 2 565 requêtes dev, 231 ont un autre
  sibling à l'adresse exacte, 165 au moins un sibling actif, 87 un label fermé
  avec sibling actif exact et 29 plusieurs siblings actifs exacts. Ce volume
  dépasse très largement les 25 erreurs tolérées à 99 % et prouve que le SIRET
  exact n'est pas toujours identifiable avec les champs CRM. Artefact:
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/site_label_audit_dev_c33b80855f560074_ac971e0`.
  Suite complète à 94 tests passants. *(commits GitHub: `638d093`,
  `ac971e0`)*
- **Second passage autonome sur les 63 prunings**: comparaison de tous les
  établissements du SIREN historique, vérification de relations opaques par
  sources publiques et application diagnostique du ranker historique gelé.
  28/63 ont un sibling SIRET dont l'adresse correspond mieux au CRM, 23 ont un
  meilleur sibling actif, et 11 un sibling actif à adresse pratiquement exacte.
  Dix de ces alternatives cohérentes sont déjà dans le top-100 actuel. Le
  ranker historique ne récupère que 22/63; même sans aucune nouvelle perte, son
  plafond optimiste serait 98,13 %. Plusieurs labels sont confirmés comme alias
  métier, tandis que Mercure/Oceania et Globecast/Kinepolis sont contredits par
  les adresses et activités publiques. La recommandation devient la création
  d'un benchmark versionné avec politique `actif à l'adresse`,
  `AMBIGUOUS_SITE` et alias historiques avant tout nouveau modèle. Rapport:
  `reports/recall100/pruned_63_audit.md`. *(commit GitHub: `f3bd0b1`)*
- **Audit des 63 vérités trouvées puis éliminées**: 13 ne sont présentes que
  dans l'overlay fermé et ne reçoivent pas de score complet; parmi les 50
  présentes dans V7, une seule reste dans le top-100 de la fusion, 17 sont
  classées 101–200, 17 entre 201–500 et 15 après 500. L'examen métier sépare
  12 preuves d'adresse, 8 preuves de nom, 13 choix du mauvais établissement
  d'un bon SIREN, 12 équipements publics reliés à leur propriétaire
  administratif et 18 relations historiques faibles ou opaques à valider
  humainement. Les petites règles testées plafonnent à 97,35 %; scorer tout
  l'overlay dégrade à 96,41 %. Rapport:
  `reports/recall100/pruned_63_audit.md`. *(commit GitHub: `58d8b31`)*
- **Décision Recall@100 = PIVOT**: sur les 2 565 requêtes dev, le sparse gelé
  atteint 2 379/2 565 = 92,75 %, la meilleure admission déterministe observée
  2 495/2 565 = 97,27 %, et l'oracle des canaux internes à K=5 000
  2 558/2 565 = 99,73 %. Le plafond de sortie 100 est strictement respecté,
  mais il manque 45 succès au gate; 7 vérités ne sont vues par aucun canal et
  63 sont vues puis éliminées par l'admission. Aucune nouvelle variante n'a été
  exécutée sur le test. Le pivot proposé est une tête d'admission apprise,
  distincte du ranker aval, sous un nouveau contrat. Rapport:
  `reports/recall100/final_go_pivot_stop.md`. *(commit GitHub: `ccb3689`)*
- **Évaluateur d'admission reproductible**: validation des manifests et hashes,
  RRF pondéré déterministe, quotas overlay, plafond strict, oracle interne,
  attribution `unseen`/`pruned`, segments et latence de sélection. L'artefact
  immuable dev est
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f`.
  Suite complète à 90 tests passants. *(commit GitHub: `5a0e67f`)*
- **Audit profond des canaux à K=5 000**: le pool V7 et l'overlay fermé ont été
  audités séparément sans positif injecté. Leur oracle combiné voit
  2 558/2 565 SIRET exacts, contre 2 540 requis, établissant que le sourcing
  peut théoriquement dépasser 99 % mais ne constitue pas une sortie éligible à
  100. *(commit GitHub: `d4255de`)*
- **Canaux SIREN locaux audités**: ajout de `siren_head` et `siren_sites` pour
  regrouper les candidats par SIREN puis ordonner/étaler leurs établissements.
  À K=100, `siren_head` récupère 47 misses du sparse courant; l'oracle V7 passe
  à 96,18 %. *(commit GitHub: `d4255de`)*
- **Overlay fermé construit et audité**: le builder limité en mémoire a publié
  8 230 664 lignes physiques INSEE et 8 286 671 lignes CP sur 52 408 fichiers,
  pour environ 872 Mo. Les 62 vérités absentes du store V7 sont toutes présentes
  dans l'overlay; son sparse courant en récupère 52 à K=100 et son oracle 60 à
  K=5 000. *(commits GitHub: `e39fddd`, `d71d3cb`)*
- **Builder overlay des fermés legacy**: construction immuable et sans lecture
  des labels d'un canal contenant uniquement les SIRET fermés exclus par le
  filtre V7 `dateDebut >= 2016`. Le périmètre géographique est dérivé des seuls
  champs INSEE/CP du benchmark, les snapshots et le benchmark sont contrôlés
  par hash, le build est atomique, manifeste et compatible avec le store
  partitionné. *(commit GitHub: `601eee5`)*
- **Audit unitaire sparse publié**: sur dev, caractères et mots récupèrent
  respectivement 65 et 59 misses du sparse@100; adresse TF-IDF 6, nom exact 15,
  adresse exacte 2 et numérique 0. L'oracle des canaux à leur propre top-100
  atteint 95,59 %, encore sous le plafond du store à 97,58 % et sous la cible.
  Rapport et lecture architecturale dans
  `reports/recall100/channel_audit_dev.md`. *(commit GitHub: `d070db8`)*
- **Audit unitaire des canaux sparse instrumenté**: runner immuable séparant
  TF-IDF nom mots, TF-IDF nom caractères, TF-IDF adresse, nom normalisé exact,
  adresse exacte et rescue numérique. Il conserve les listes/rangs par requête,
  mesure la complémentarité appariée, le SIREN et la géographie, et refuse le
  run si `current_sparse` ne reproduit pas exactement l'artefact baseline gelé.
  Smoke réel sans divergence; suite complète à 83 tests passants. *(commit
  GitHub: `218c22c`)*
- **Baseline sparse Recall@K dev publiée**: préfixe @50 identique sur les
  2 565 requêtes à la baseline historique; Recall SIRET @50/@100/@200/@500 =
  90,33/92,75/94,15/95,79 %. Le store V7 plafonne à 97,58 %: 62 SIRET, tous
  fermés, sont absents des 14,3 M candidats mais présents dans le snapshot brut
  StockEtablissement. Zéro perte filtre ou déduplication; 124 autres vérités
  sont classées après 100. *(commit GitHub: `67f1a9c`)*
- **Préfixes Recall@K stabilisés et cache mutualisé**: un passage max-K ne
  classait pas les partitions de taille comprise entre 51 et K, rendant son
  préfixe @50 différent de la baseline. Ajout d'un seuil de déclenchement du
  ranking indépendant du budget final; le smoke reproduit désormais exactement
  les dix préfixes @50 historiques. Les matrices TF-IDF utilisent un hash
  d'artefact indépendant du cutoff avec fallback vers les 7,7 Go de cache
  legacy. Le premier run `..._963160b` est conservé mais déclaré diagnostic
  invalide pour les préfixes @50/@100/@200. Suite complète à 81 tests passants.
  *(commit GitHub: `bdc7ad4`)*
- **Instrumentation Recall@K et causes de perte**: séparation explicite des
  états avant filtre, après filtre et après déduplication dans le retrieval
  partagé; nouveau runner immuable calculant en un passage les préfixes
  @50/@100/@200/@500, intervalles, segments, latence, cardinalités et buckets
  de perte mutuellement exclusifs. Smoke réel et suite complète à 78 tests
  passants. *(commit GitHub: `5e3fd5f`)*
- **Contrat Recall@100 pre-enregistre**: cible SIRET exacte >=99,0 %, plafond
  absolu 100, courbes diagnostiques @50/@100/@200/@500, attribution obligatoire
  partition/filtre/deduplication/pruning, audit canal par canal, tuning
  train/dev et unique evaluation de la variante gelee sur test. `AGENTS.md`
  pointe desormais vers ce goal actif. *(commit GitHub: `8b77af3`)*
- **Decision finale V9 = PIVOT**: sparse + dense local perd 1,83 point de
  Recall@50 SIRET et sparse + dense global SIREN perd 2,61 points. Les deux
  regressions sont statistiquement nettes et violent les gates segmentaires.
  En revanche, leur Hit@1 SIRET brut gagne respectivement 7,33 et 11,31 points,
  avec des IC95 strictement positifs. Gate 3 ranker/accepteur, Gate 4 open-set
  et le cross-encoder ne sont pas ouverts sur le pool rejete. Le pivot propose
  une nouvelle ablation: pool sparse fixe + dense comme feature de scoring.
  Le Mac a execute l'ensemble sans GPU ni depense cloud. *(commit GitHub:
  `59ec78c`; lecture STOP preliminaire supersedee: `53ad3b3`)*
- **Hit@1 SIRET/SIREN publie par le runner**: les comparaisons appariées
  incluent désormais les Hit@1, leurs recuperations/deplacements, IC95
  bootstrap et tests de McNemar. Sur global, Hit@1 SIRET passe de 36,22 % a
  47,52 % et Hit@1 SIREN de 41,91 % a 53,92 %. Suite complete a 68 tests
  passants. *(commit GitHub: `de0079a`)*
- **Gate 2 dense global SIREN echouee sur dev**: sparse atteint 2 317/2 565 =
  90,33 % contre 2 250/2 565 = 87,72 % pour l'hybride global. Delta apparie
  −2,61 points, IC95 [−3,51; −1,72], McNemar p=1,47e-8; 37 misses recuperes
  mais 104 hits deplaces. Actifs −3,27 points et multi-sites −2,28 points.
  La latence p95 passe a 1,079x et le budget 50 est respecte. *(commit GitHub:
  `53ad3b3`)*
- **Audit de budget multicanal corrige**: deux sorties globales de 50 candidats
  etaient faussement marquees non conformes car leur seul pool local contenait
  41 ou 18 lignes. Le controle refuse maintenant les depassements de K et les
  underfills locaux, sans rejeter les candidats qui completent un pool court
  via un nouveau canal. Rapport global regenere avec zero violation; suite
  complete a 68 tests passants. *(commit GitHub: `bc49918`)*
- **Expansion globale SIREN bornee avant materialisation**: le store candidat
  v2 calcule la densite par zone et applique dans DuckDB un top-K par SIREN,
  ordonne par correspondance INSEE/CP puis densite locale, avant tout transfert
  vers Python. Le cap SQL est 40 et le cap metier final reste 20 SIRET par
  SIREN; le lecteur reste compatible avec les stores v1. Suite complete a
  66 tests passants. *(commit GitHub: `43f2c64`)*
- **Manifeste d'expérience dense fermé**: chaque run V9 référence désormais le
  hash du contrat de son store local, ANN global, géographie mmap et store
  candidat SIREN; les stores partitionnés sans manifeste racine sont liés par
  un hash agrégé déterministe de leurs  manifestes. Suite complète à 66 tests
  passants. *(commit GitHub: `fef3658`)*
- **Expansion globale SIREN rendue exécutable**: le smoke historique rechargeait
  jusqu'à des dizaines de partitions aléatoires par requête (p95 17,5 s sur
  cinq cas, contre 0,67 s sparse). Ajout d'un store DuckDB read-only indexé par
  SIREN qui récupère les 50 groupes de candidats en une requête, tout en
  conservant la priorité géographique et la limite de 20 SIRET par SIREN.
  La lecture Arrow conserve les `None`/listes des partitions, sans conversion
  pandas en `NaN`. Suite complète à 65 tests passants. *(commits GitHub:
  `13d66d2`, `654413f`)*
- **Lookup géographique SIREN compatible 24 Go**: remplacement optionnel du
  chargement legacy de 37,8 M lignes dans pandas/dict par un artefact trié,
  quatre tableaux NumPy mmap et une recherche binaire SIREN. Le builder DuckDB
  travaille sur SSD avec limite mémoire, publie hashes et cardinalité; le
  lecteur conserve la compatibilité avec le parquet historique. Suite complète
  à 64 tests passants. *(commit GitHub: `7781d31`)*
- **Index dense global SIREN construit sur Mac**: 28 982 797 unités légales
  encodées en CPU avec le modèle générique épinglé, puis indexées en IVFPQ
  4096/48; manifeste avec hash source, fingerprint modèle et hashes FAISS/IDs.
  Un contrôle reproductible échantillonne les row groups du parquet, vérifie
  l'intégrité des sorties et mesure self-recall@1/@50 et latence avant toute
  évaluation métier. Suite complète à 63 tests passants. *(commits GitHub:
  `2d74b2b`, `6718d8b`)*
- **Contrat dense global SIREN renforcé**: le builder publie désormais la
  progression d'encodage, le fingerprint intégral du modèle et les hashes des
  fichiers FAISS/IDs. Le benchmark et l'inférence V9 refusent un index construit
  avec un autre modèle avant même de charger FAISS. Suite complète à 62 tests
  passants. *(commit GitHub: `2d74b2b`)*
- **Gate 2 dense local échouée sur dev**: sparse atteint 90,33 % Recall@50
  SIRET contre 88,50 % pour sparse+dense local et 70,29 % pour dense seul.
  L'hybride récupère 45 misses mais déplace 92 hits: delta apparié −1,83 point,
  IC95 [−2,73; −0,94], p exact 0,000073. Budget et latence passent, mais actifs
  (−2,26), mégapoles (−3,03) et multi-sites (−2,28 points) violent le gate
  segmentaire. Les 168 misses sparse au niveau SIREN et 25 récupérations SIREN
  uniques par le dense justifient la dernière expérience globale SIREN, sans
  tuning opportuniste de RRF. *(commit GitHub: `71c68ef`)*
- **Store dense local dev complet**: les 871 partitions INSEE et 14 partitions
  CP du plan gelé ont été encodées sur CPU avec le MiniLM générique épinglé,
  soit 10 216 448 candidats dans 885 paires index/manifeste (3,0 Go sur SSD).
  La vérification exhaustive confirme un unique fingerprint modèle, le hash
  exact du plan, zéro fichier manquant/temporaire et 61 tests passants. Le
  builder cherchait initialement `cp_codes` au lieu du champ canonique
  `postcode_codes`; le défaut est corrigé et couvert par régression. *(commit
  GitHub: `8ec1881`)*
- **Comparateur apparié Gate 2**: validation des hashes de l'expérience et de
  l'alignement exact des requêtes, décompte des misses récupérés et hits
  déplacés, IC95 bootstrap apparié, test exact de McNemar, deltas par segment,
  ratio de latence p95 et refus explicite de toute violation du budget fixe.
  Le rapport JSON/Markdown produit est immuable et lié au manifeste de
  l'expérience; suite complète à 60 tests passants. *(commit GitHub:
  `86dea2c`)*
- **Dense local non contamine prepare**: fingerprint integral du modele
  semantique impose entre build et inference, revision generique MiniLM
  `86741b4e` copiee sans telechargement sur le SSD, reparation du tokenizer
  Unigram et plan de partitions immuable. Le plan dev couvre 871 partitions
  INSEE et 14 CP, environ 10,2 M de lignes physiques; aucune requete dev sans
  partition planifiable. *(commit GitHub: `10dd990`)*
- **Baseline sparse-50 V9 mesuree**: sur les 2 652 requetes test gelees,
  Recall@50 SIRET 88,54 % (2 348 hits, IC95 87,27–89,69), Recall@50 SIREN
  92,16 %, recall du pool geographique 98,00 %, zero violation de budget.
  Les 304 erreurs comprennent 53 absences de partition et 251 prunings; 96
  erreurs conservent le bon SIREN. Segments critiques: fermes 67,09 %,
  megapoles 77,01 %. Artefacts bruts hashes sur le SSD et rapport dans
  `reports/v9/retrieval_baseline_sparse50.md`. *(commit GitHub: `8adc5f3`)*
- **Runner retrieval V9 immuable**: execution sparse, hybride local, dense-only
  et hybride global SIREN avec budget final strict, preuves par requete,
  Recall SIRET/SIREN et Wilson 95/99 %, segments, latences p50/p95/p99, cache
  SSD borne en RAM et manifeste lie au commit. Le benchmark segmente v2
  `c33b80855f560074` remplace le build v1 pour les experiences; le v1 reste
  conserve. *(commit GitHub: `771beb6`)*
- **Benchmark ferme V9 gele**: reconstruction exacte du split V7 historique
  par SIREN (seed 42), validation contre les scenes positives V7, ajout des 692
  requetes historiquement absentes des scenes afin de compter les misses
  end-to-end, hash integral des 4 119 fichiers de partitions et des snapshots
  SIRENE. Build initial immuable `8967e72e07c9f4bf` puis revision segmentee
  `c33b80855f560074` sur le SSD externe: 11 837 train,
  2 565 dev, 2 652 test, zero SIREN partage. Les labels restent des verites CRM
  historiques non reaudites et le modele dense fine-tune local est declare
  contamine pour toute revendication finale sur ce corpus. *(commit GitHub:
  `b384509`)*
- **Gate 0 V9 sans GPU franchie**: cles d'index dense alignees sur les vraies
  partitions, refus des subsets mega-communes incompatibles, manifeste de
  cardinalite et d'ordre SIRET, isolation stricte de PyTorch et FAISS dans
  deux sous-processus persistants sans `KMP_DUPLICATE_LIB_OK`, builders local
  et global SIREN corriges, mode dense-only repare et entrypoints V9
  executables directement. Validation: 52 tests passants, smoke 512 lignes,
  index local reel de 17 462 candidats et index global SIREN de 1 000 entites
  construits/interroges avec succes sur CPU. *(commit GitHub: `88e97e0`)*
- **Contrat d'execution V9 sans GPU**: directive active `GO/PIVOT/STOP`
  placee en tete de `AGENTS.md`, ressources locales autorisees, ordre des
  experiences, gates et regles d'arret formalises dans
  `docs/v9_execution_contract.md`. Les descriptions V6/V7/V8 sont explicitement
  historiques et ne pilotent plus les travaux. *(commit GitHub: `72d2749`)*
- **Benchmark open-set, ablation cross-encoder et gates V9**: feuille
  d'adjudication stratifiee, validation humaine/evidence/snapshot obligatoire,
  gel adresse par hash, cross-encoder top-20 avec revision epinglee, gates
  retrieval/segments/latence/deploiement et guide d'execution. Les trois
  variantes cross-encoder produisent des predictions OOF compatibles avec le
  meme accepteur. *(commits GitHub: `c4cf99f`, `b82271e`)*
- **Ranker unique + accepteur selectif V9**: 54 features brutes partagees
  train/serve puis sous-ensemble manifeste, features retrieval/SIREN, ranker
  XGBoost avec predictions OOF, misses conserves, correction stricte SIRET,
  calibration et selection de seuil sur deux moities dev distinctes, comparaison
  logistique/XGBoost, moteur d'inference `AUTO_MATCH|REVIEW` compatible
  `routing_status`. L'injection de positifs est autorisee uniquement dans le fit
  ranker train et interdite dans les scenes/evaluations. *(commit GitHub:
  `db4ab27`)*
- **Retrieval hybride V9 a budget fixe**: RRF sparse/dense/rescue, vrais scores
  TF-IDF ordonnes, provenance/rangs par canal, configurations 50 et ablation 100,
  index dense global SIREN streaming avec manifeste/tokenizer, expansion limitee
  SIRET et benchmark p50/p95. *(commit GitHub: `36404ae`)*
- **Contrats et dataset canonique V9**: ajout du contrat public `AUTO_MATCH/REVIEW` avec mapping legacy, labels `MATCH_EXACT/NO_MATCH/AMBIGUOUS/UNRESOLVED`, split deterministe SIREN-disjoint, bundle parquet immuable adresse par hash, manifeste de provenance/config/tokenizer/features et registre explicite des artefacts legacy interdits aux entrypoints V9. *(commit GitHub: `afb0f3d`)*
- **Socle V9 semantique + prediction selective**: chargement lazy de SentenceTransformer, reparation runtime du tokenizer Unigram exporte comme BertTokenizer, healthcheck anti-`<unk>`, injection semantique partagee train/serve, remise en service du mining d'homonymes geographiques et primitives testees de courbe risque-couverture/certification binomiale. Suite de tests retablie a 20 tests passants. *(commit GitHub: `fcfc33f`)*
- **Spikes architecture neurale (cross-encoder + dual-encoder)**: benchmark reproductible sur un holdout SIREN-disjoint de 400 requetes. Le cross-encoder court ne remplace pas XGBoost (51,75% vs 85,25% Hit@1 sur les memes scenes). Le dual-encoder structure atteint 74,50% Recall@1 et 96,00% Recall@50; l'union TF-IDF top-50 + dense top-50 atteint 99,25% Recall@50 (8 des 11 misses TF-IDF recuperes). Le modele semantic exporte declare a tort `BertTokenizer`; le chargement actuel via SentenceTransformer produit excessivement des tokens `<unk>`, donc les anciens benchmarks semantiques doivent etre revalides apres correction. *(commit GitHub: `7640772`)*
- **V8 features + hard negatives + hyperparams decider**: ajout de 7 features d'interaction, extension des hard negatives colocataires/homonymes/siblings, tuning decider (`lr=0.05`, `max_depth=7`, `400 rounds`). *(commit GitHub: `35fb441`)*
- **Route B (SIREN-first) implementee**: nouvel index global SIREN, nouveau module de retrieval SIREN, branchement conditionnel dans l'inference profile/engine. *(commit GitHub: `3e090b7`)*
- **Correctifs bloquants Route B**: fix DuckDB `:memory:`, fix champ CRM nom, fix filtre closed/open, ajout CLI `--siren-index` dans le generateur de samples. *(commit GitHub: `c356923`)*
- **Branchement Route B dans le retrieval partage (training)**: `build_candidate_pool()` supporte Route B via indices SIREN, propagation sequentielle + multiprocess dans `generate_training_samples_v5fast.py`. *(commit GitHub: `1305012`)*
- **Implementation V8b SIREN expansion (V7 + local + cross-partition)**: ajout Step 5 d'expansion apres prefilter, feature flag, cap pool dedie et telemetrie d'expansion. *(commit GitHub: `9c0e806`)*
- **Correctifs critiques V8b**: exclusion explicite Route B quand expansion activee, filtres metier expansion, recalc GT coverage/loss reason post-expansion. *(commit GitHub: `f1fbbb8`)*
- **Fix expansion SIREN en mode geo-only**: chargement des index dissocie (global vs geo) dans le generateur de samples pour eviter la desactivation silencieuse de l'expansion quand seul `siren_to_geo.parquet` est present. *(commit GitHub: `c961371`)*

## Historique structurant (deja en place)
- **Retrieval hybride sparse+dense + cache TF-IDF persistant + timing**: integration du socle P0/P1. *(commit GitHub: `9ab297e`)*
- **Ablation dense-only corrigee + flag sparse explicite**: alignement des modes retrieval et signature de config. *(commit GitHub: `35fc3a3`)*
- **Defaults partitions V7 + manifest INSEE O(1)**: bascule des chemins/scripts vers `data/candidates_v7_all`. *(commit GitHub: `a309a7c`)*
- **Priorisation mega-communes embeddings**: orchestration dense amelioree pour runs longs. *(commit GitHub: `66b5b87`)*

## Fichiers modifies recemment
- `docs/benchmark_v3_evidence_policy.md` *(commit GitHub : `09b9d46`)*
- `docs/retrieval_selective_recall100_contract.md` *(commit GitHub :
  `4f6e317`)*
- `scripts/build_benchmark_v3_evidence.py` *(commits GitHub : `cf7133c`,
  `c6c8186`, `eb0e6a3`)*
- `scripts/certify_selective_retrieval_test.py` *(commits GitHub : `d1c0fc9`,
  `6fab035`)*
- `scripts/audit_v3_qualification_stability.py`,
  `tests/test_v3_qualification_stability.py` *(commit GitHub : `111b07c`)*
- `reports/recall100/selective_test_certification.md` *(commit GitHub :
  `41ff2e1`)*
- `reports/recall100/final_go_pivot_stop.md` *(rapport dev historique marqué
  supersédé, commit GitHub : `50e804b`)*
- `scripts/audit_downstream_architecture.py`,
  `tests/test_downstream_architecture_audit.py` *(commits GitHub :
  `a59fb0f`, `aeeaf0f`)*
- `reports/v9/downstream_architecture_audit.md` *(commit GitHub :
  `e186439`)*
- `docs/downstream_selective_matching_contract.md` *(commit GitHub :
  `c18bf28`)*
- `docs/downstream_acceptor_e2b_contract.md` *(commit GitHub : `cf91432`)*
- `docs/benchmark_v4_current_snapshot_policy.md` *(commit GitHub :
  `ce82b01`)*
- `docs/v4_fresh_expansion_contract.md` *(commit GitHub : `1c2e84c`)*
- `docs/v4_retrieval_reconstruction_contract.md` *(commit GitHub :
  `510868b`)*
- `docs/v4_ranker_e1_contract.md` *(commit GitHub : `0c90c25`)*
- `scripts/build_benchmark_v4_current_snapshot.py`,
  `tests/test_benchmark_v4_current_snapshot.py` *(commit GitHub :
  `799c32d`)*
- `scripts/build_v4_fresh_expansion.py`,
  `tests/test_v4_fresh_expansion.py` *(commit GitHub : `613cf7d`)*
- `scripts/prepare_v4_retrieval_inputs.py`,
  `scripts/finalize_v4_retrieval_gate.py`,
  `tests/test_v4_retrieval_gate.py` *(commit GitHub : `e566c25`)*
- `scripts/build_v4_ranker_dataset.py`,
  `tests/test_v4_ranker_dataset.py` *(commit GitHub : `6236365`)*
- `scripts/evaluate_v4_ranker_e1.py`,
  `tests/test_v4_ranker_e1.py` *(commit GitHub : `250a05f`)*
- `scripts/build_downstream_selective_dataset.py`,
  `tests/test_downstream_selective_dataset.py` *(commits GitHub :
  `0a75b73`, correction des partitions overlay `fc9cb1b`)*
- `scripts/train_v9_ranker.py`, `src/xgb_matcher/v9_scene.py`,
  `scripts/train_v9_acceptor.py`, `src/xgb_matcher/v9_acceptor.py`
  *(commits GitHub : `aeeaf0f`, `0a75b73`, `dbd8906`)*
- `reports/v9/downstream_e1_e2_results.md` *(commit GitHub : `9ab7f6a`)*
- `reports/v9/downstream_e2b_results.md` *(commit GitHub : `ebb4bf2`)*
- `reports/v9/benchmark_v4_current_snapshot_results.md` *(commit GitHub :
  `299bc8a`)*
- `reports/v9/v4_fresh_expansion_results.md` *(commit GitHub :
  `d8d36b9`)*
- `reports/v9/v4_retrieval_gate_results.md` *(commit GitHub : `6948aa1`)*
- `reports/v9/v4_ranker_e1_results.md` *(commit GitHub : `13e3547`)*
- `src/xgb_matcher/features.py` *(commits GitHub: `35fb441`, `fcfc33f`, `db4ab27`)*
- `scripts/generate_training_samples_v5fast.py` *(commits GitHub: `35fb441`, `c356923`, `1305012`, `c961371`, `fcfc33f`, `db4ab27`)*
- `scripts/train_xgb_decider.py` *(commit GitHub: `35fb441`)*
- `scripts/build_siren_global_index.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/siren_retrieval.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/infer.py` *(commits GitHub: `3e090b7`, `c356923`, `36404ae`)*
- `src/xgb_matcher/retrieval.py` *(commits GitHub: `1305012`, `9c0e806`, `f1fbbb8`, `36404ae`)*
- `src/xgb_matcher/retrieval_config.py` *(commits GitHub: `3e090b7`, `9c0e806`, `36404ae`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/v9_dataset.py` *(commits GitHub: `afb0f3d`, `db4ab27`)*
- `src/xgb_matcher/v9_scene.py`, `v9_acceptor.py`, `v9_infer.py`
  *(commit GitHub: `db4ab27`)*
- `src/xgb_matcher/fusion.py`, `v9_features.py` *(commit GitHub: `36404ae`)*
- `src/xgb_matcher/v9_adjudication.py`, `v9_cross_encoder.py`,
  `v9_evaluation.py` *(commit GitHub: `c4cf99f`)*

## Travail en cours
- Aucun run long n'est en cours. Les canaux train, le dataset aval, le ranker
  E1, les accepteurs E2/E2b, V4, V4-Fresh et le gate retrieval V4 sont publiés
  sur le SSD.
- Le test final a été lu une fois et est maintenant définitivement fermé à
  toute nouvelle variante, règle ou seuil.
- E1 historique est conservé comme baseline. Le nouveau ranker V4 est validé
  sur `dev_new`, mais aucun modèle produit n'est déployé.
- V4-Fresh a levé le manque d'exemples et la fuite SIREN ; le gate retrieval
  V4 et le ranker E1 sont franchis. Le prochain travail est de reconstruire les
  scènes `AMBIGUOUS` fit/dev, puis d'entraîner et calibrer l'accepteur
  exact-SIRET, sans ouvrir `holdout_sealed`.

## Points d'attention
- **Plafond absolu 100**: les mesures @200/@500 sont diagnostiques et ne
  constituent jamais une configuration eligible.
- **Test final consommé** : ne plus lancer de variante, analyser de miss pour
  choisir une règle, changer de seuil ou modifier la qualification sur ce
  split. Toute évolution nécessite un nouveau holdout indépendant.
- **Portée du 99,436 %** : Recall candidat sur les 80,241 % de dossiers V3
  exacts, pas précision `AUTO_MATCH` et pas taux d'automatisation global.
- **Modèles historiques gelés** : aucun modèle legacy n'est modifié. La phase
  aval E1/E2 est ouverte sous
  `docs/downstream_selective_matching_contract.md`.
- **Decision PIVOT scopee**: elle invalide l'admission/fusion des candidats
  denses V9 testes, pas leur signal de scoring ni le pipeline sparse/XGBoost.
- **Comparaison retrieval uniquement a budget constant**: un gain avec 100
  candidats ne justifie pas la promotion de la variante 50.
- **Precision strictement SIRET**: un bon SIREN mais mauvais etablissement est
  une erreur pour l'accepteur.
- **UNRESOLVED n'est pas un négatif prouvé** : le traiter comme faux match
  pendant l'apprentissage crée du bruit de cible. Il doit rester hors du fit
  tant qu'une validation indépendante ne lui attribue pas `NO_MATCH`,
  `AMBIGUOUS` ou `MATCH_EXACT`.
- **Date de vérité absente** : V4 fixe désormais explicitement la politique
  « actif au snapshot ». Elle ne peut pas servir à reconstruire un exploitant
  historique sans date CRM.
- **Split historique invalidé par les nouvelles vérités** : 14 SIREN V4 exacts
  étaient partagés entre train et dev. L'ancien dev est abandonné pour la
  nouvelle cible ; V4-Fresh fournit un nouveau dev et un holdout sans SIREN
  exact partagé avec le fit.
- **Holdout Fresh scellé** : aucune génération de candidats, prédiction ou
  mesure modèle ne doit lire `holdout_sealed` avant gel du bundle et du seuil.
- **NO_MATCH temporel**: toujours rattache au snapshot SIRENE et a la date de
  reference.
- **Cross-encoder conditionnel**: aucune promotion sans +1 point de couverture
  a precision cible et gates segments/latence. Il reste hors chemin critique et
  aucune location de GPU n'est autorisee.
- **Certification**: avant environ 2 300 AUTO independants audites sans erreur,
  publier une estimation observee, jamais une garantie a 99,8 %.
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V9)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Bundle canonique | `data/v9/<build_id>/{queries,labels,candidates}.parquet` |
| Manifeste dataset | `data/v9/<build_id>/manifest.json` |
| Mapping geo SIREN | `data/siren_index/siren_to_geo.parquet` |
| Index dense SIREN | `data/v9_indices/siren_dense_<snapshot>/` |
| Ranker + predictions OOF | `models/v9/ranker_<build_id>/` |
| Accepteur + calibration | `models/v9/acceptor_<build_id>/` |
| Benchmark open-set gele | `data/v9_open_set/<benchmark_id>/` |
| Dataset aval E1/E2 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068/` |
| Ranker E1 expérimental | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b/` |
| Accepteur E2 refusé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_3171ef5020c0f068_fc9cb1b/` |
| Accepteur E2b refusé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_e2b_3171ef5020c0f068_070c123/` |
| Qualification V4 refusée | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759/` |
| Expansion V4-Fresh passée | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f/` |
| Gate retrieval V4 passé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/retrieval_v4/ddefe3daaacdf5ef/` |
| Dataset ranker V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/ranker_v4/1aebeada820d92a7/` |
| Ranker V4 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/ranker_v4/ranker_1aebeada820d92a7_6236365/` |

## Prochaines etapes
1. Ne plus toucher au test final actuel.
2. Geler la politique V4 stricte et ne pas assouplir ses seuils sur le dev
   audité.
3. Reconstruire les scènes `AMBIGUOUS` fit/dev pour l'accepteur, sans transformer
   `UNRESOLVED` en négatifs.
4. Produire les scènes exactes de l'accepteur à partir des prédictions OOF du
   ranker V4 et des preuves top-1/top-2.
5. Entraîner puis calibrer l'accepteur sur des parties disjointes du dev ;
   publier la couverture à 99,0 %, 99,5 % et 99,8 % de précision observée.
6. Geler le bundle et le seuil,
   puis seulement alors autoriser une ouverture unique de `holdout_sealed`.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
