# SIRETO Handover - 31 Juillet 2026

## Etat des lieux

### Population prospective CRM GT v2 — provenance commerciale certifiée

Le nouvel apport CRM a été fusionné sans modifier `data/crm_ok_gt.csv` dans
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_gt_v2_population/6fb3e7ca96dbfa9a/`.
Le fichier publié `crm_ok_gt_v2.csv` contient 37 263 lignes ; la population
modèle contient 37 218 scènes, dont 20 121 nouvelles éligibles. Les composants
historiques sont conservés, les SIREN inédits sont répartis 70/15/15 par
composante, et 88 lignes liées à des conflits historiques multi-fold sont
quarantainées. La qualification ne consomme aucun retrieval. L'échantillon
indépendant de 400 labels demeure un diagnostic de suspicion, pas un oracle
d'étiquettes : le SIRET a été saisi par l'assistant commercial lors de la
création du site. Les 20 121 lignes géographiquement cohérentes sont donc
admises comme GT humain sous existence SIRENE + garde INSEE/CP. La population
certifiée est publiée sous
`crm_gt_v2_commercial_certified_population/4b07f3b3d245358e/`. Les 37 218
labels, historiques comme nouveaux, portent désormais explicitement leur
provenance CRM humaine. Le rebuild des
partitions conserve désormais aussi tous les établissements fermés, sans
cutoff arbitraire à dix ans.

*(builder, split prospectif et tests : commit `09f27f1`)*
*(certification de provenance commerciale : commit `43d1e14`; correction des
partitions fermées : commit `1a7cfea`)*
*(provenance humaine explicite sur les 37 218 labels : commit `852ae30`; vues
retrieval exacte et opérationnelle : commit `018c7d1`)*

Le contrat commun XGBoost/BGE/CamemBERT/fusion est gelé dans
`docs/crm_gt_model_usage_contract.md` : folds 2/3/4 pour l'apprentissage
(23 587 lignes), fold 0 pour le développement prospectif (7 009), fold 1 fermé
pour le test final (6 622). Le fusionset ne reçoit que des scores out-of-fold.
Après l'évaluation finale, le modèle de production peut être refitté sur
train+dev, puis éventuellement sur les 37 218 labels à condition de remplacer
le holdout consommé par de futures données CRM.

*(contrat d'utilisation commun aux familles de modèles : commit `803ee37`)*

Le retrieval a ensuite été rejoué sur les 3 510 lignes du développement
prospectif, avec partitions SIRENE courantes, 100 candidats maximum et aucune
injection. La couverture et la présence du SIRET dans le pool géographique sont
de 100 %, mais le Recall@100 exact/opérationnel n'est que de 3 279/3 510
(93,42 %). L'oracle lexical @5000 atteint 3 448/3 510 (98,23 %). Verdict :
`PIVOT_RETRIEVAL`; aucun modèle aval n'est entraîné avant le gate de 99,0 %.
Résultats et artefacts : `docs/crm_gt_v2_retrieval_results.md`.

*(benchmark exact + opérationnel et évaluateur : commits `33a9568`,
`018c7d1`)*

### Nouvel apport CRM — tri géographique SIRET du 17 août 2026

Le fichier VPS `plateforme_new_data_to_cure.csv` a été copié via Tailscale
SSH sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_new_data_20260817/source/`.
Ses 14 954 776 octets et son SHA-256
`829cee816839bd7c24962127e2d35784923bc0d0c5ecbe12ce34e45a1439f7da`
ont été vérifiés avant traitement. Le tri est indépendant des modèles et joint
les SIRET au snapshot SIRENE gelé
`c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845`.

Sur 131 749 lignes, 43 284 renseignent un SIRET. Après déduplication par
service, 20 725 services passent le garde géographique strict : INSEE exact,
ou CP exact uniquement si l'INSEE CRM manque. Les 312 CP exacts qui
contredisent explicitement l'INSEE sont isolés en revue. La déduplication
globale avec `crm_ok_gt` et le rejet des surfaces CRM portant plusieurs SIRET
laissent 20 209 couples nouveaux, uniques et non ambigus, couvrant 16 659
SIRET. Parmi eux, 1 472 appartiennent déjà à une composante train folds 2/3/4;
les SIREN nouveaux restent `UNSEEN_SIREN_NEEDS_ASSIGNMENT` et les composantes
dev/test ou folds 0/1 sont quarantainées. `data/crm_ok_gt.csv` reste inchangé.

Artefacts et manifeste reproductible :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_new_data_20260817/location_triage_v1/`.
Le replay donne les mêmes hashes pour les sept sorties.

*(tri géographique et garde anti-fuite : commit `ce6f47a`)*

### Politique métier — équivalence opérationnelle même SIREN / même site

La décision métier du 15 août 2026 distingue désormais la justesse SIRET
exacte de la justesse opérationnelle CRM. Un autre SIRET du même SIREN prouvé
à la même adresse physique que le CRM est
`OPERATIONAL_EQUIVALENT_SAME_SITE` et compte comme correct dans la vue
opérationnelle. Si la vérité exacte est fermée et ce candidat actif, il devient
le résultat préféré `ACTIVE_SUCCESSOR_SAME_SITE`. Les métriques et gates SIRET
exacts historiques restent publiés séparément et inchangés.

Pour tout futur apprentissage, un sibling même SIREN / même site est un positif
opérationnel et ne peut plus être utilisé comme hard negative. Les cycles déjà
gelés ne sont pas modifiés rétroactivement. Politique :
`docs/siret_operational_equivalence_policy.md`.

*(politique et directive centrale : commit `7889974`)*

### BGE fine-tuné + stack XGBoost — cycle clos par STOP

Le cycle préenregistré `docs/v412_bge_xgb_stack_contract.md` est terminé. Le
BGE groupwise fine-tuné atteint 2 400/2 797 SIRET exacts sur le fold 0
(85,81 %), contre 2 437/2 797 pour `BUSINESS_LEARNED`. Les trois scores BGE
d'apprentissage du stack sont strictement cross-fittés sur les folds 2/3/4.
Le stack XGBoost top 10 final atteint 2 436/2 797 : 41 erreurs corrigées mais
42 régressions. Il échoue les gates exact (seuil 2 452) et difficile
(32/38, seuil 33). Verdict : `STOP_RANKER_GATE`.

Le fold 1 et le test final n'ont pas été ouverts. La branche CamemBERT
conditionnelle n'était pas autorisée par le résultat. L'accepteur pré-Maps
n'est pas entraîné, puisque le gate ranker était sa précondition ; aucune
métrique AUTO ou Maps n'est revendiquée.

La vue opérationnelle secondaire, conforme à la nouvelle politique même-site,
reste elle aussi derrière la baseline : 2 453/2 797 pour le stack contre
2 454/2 797 pour `BUSINESS_LEARNED`. Elle promeut 17 équivalents même-site
dans chaque système, dont 5 successeurs actifs d'une vérité fermée, sans
modifier les métriques primaires gelées.

Artefacts :

- BGE fold 0 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_groupwise/01e1049c16af2600` ;
- BGE OOF folds 2/3/4 : `2b424777fbf2f02e`, `9c5091071d727cb6`,
  `a79c8c3adb3ca3bc` sous le même répertoire ;
- stack :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_xgb_stack/8c1bce0bbf9593b5` ;
- vue opérationnelle secondaire :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_operational_secondary/cccd9f1b99877848` ;
- rapport : `reports/v412_bge_xgb_stack_verdict.md`.

Le cycle a consommé 13,45 h de calcul BGE cumulé, avec un pic RSS de 5,39 Go,
zéro OOM, zéro GPU loué et zéro service payant. Tous les manifests ont été
rehashés sans écart.

*(verdict, rapport et vue opérationnelle secondaire : commit `c633ec2`)*

### Corpus GT synthétique équilibré — production P000

La trajectoire des canaries est close. Le corpus cible compte 20 000 variantes
promues, au plus trois par SIRET et au plus 8 000 cibles. Sa distribution
globale est gelée à 20 % `EASY`, 50 % `MEDIUM`, 30 % `HARD`; l'augmentation
est orthogonale : 20 % erreurs conjointes BGE/XGBoost, 15 % échecs XGBoost
seul, 15 % échecs BGE seul, 40 % distribution train et 10 % contrôles presque
propres. Le synthétique est réservé au ranker/decider avec poids de scène
`0.5/k` par identité; il est interdit au risk model, à la calibration, au
choix des seuils AUTO et au test final.

Le catalogue de ciblage est calculé uniquement sur les prédictions OOF des
folds train 2/3/4 : 8 252 scènes jointes, dont 839 échecs conjoints, 224 cas où
seul BGE est juste et 480 où seul XGBoost est juste. Il ne contient aucun
query id, texte CRM, SIRET ou SIREN. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/train_oof_bge_xgb_failure_catalog_v1.json`,
SHA-256 `9ab3eb713cde5122`.

Le batch compté `P000` est sélectionné et figé : 150 cibles / 450 contrats,
75 actives et 75 fermées; exactement 90 faciles, 225 moyens, 135 difficiles;
90 `FAIL_BOTH_MODELS`, 67 `FAIL_XGB_ONLY`, 68 `FAIL_BGE_ONLY`, 180
`TRAIN_DISTRIBUTION` et 45 `NEAR_CLEAN_CONTROL`. Il mobilise 334 références
train distinctes, maximum 28 usages par référence, 62 opérateurs exacts et
aucune paire de relations au-delà de 45/450. Les 59 SIRET/SIREN des canaries
et du pilot30 sont exclus. Seed :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/balanced_v1/P000_seed_input.jsonl`,
SHA-256 `6e7b08584c358857`.

Le runtime accepte désormais la promotion indépendante par variante lorsque
le run l'active explicitement; le comportement historique reste atomique 3/3.
Un slot épuisé ne détruit donc plus deux couples valides. Le critic reçoit
uniquement les variantes passées. Les efforts sont séparés par rôle : Luna
`low` pour GENERATOR, `high` pour CRITIC et `max` pour ADJUDICATOR. P000 doit
être exécuté à concurrence 32, extensible à 64 selon la stabilité transport,
puis audité full-SIRENE avant promotion par variante. Les 110 tests du sous-
système synthétique passent.

*(plan équilibré, catalogue OOF, sélection P000 et runtime : commit `5e738c3`)*

P000 a été exécuté le 16 août 2026 avec deux pools de 32 workers, soit jusqu'à
64 appels concurrents. Les 699 appels Luna ont tous abouti au premier essai de
transport. En 633 secondes, les 450 slots ont produit 401 preflights passés et
49 slots épuisés ; le critic a rendu 394 `ACCEPT`, 1 `REJECT` et 6 `SILVER`.
Le re-scan full-SIRENE exhaustif qualifie 392/394 `ACCEPT` comme
`EXACT_IDENTIFIABLE` avec témoin singleton `G_N_A`; deux ambiguïtés officielles
sont exclues. Le promoteur par variante publie donc 392 couples sur 141 SIRET,
tous distincts, sous SHA-256 `0e0d19c530c62353`. Distribution promue : 73
`EASY` (18,6 %), 201 `MEDIUM` (51,3 %), 118 `HARD` (30,1 %); 79 échecs
conjoints, 57 échecs XGBoost seul, 60 échecs BGE seul, 157 train-distribution
et 39 contrôles near-clean. Actifs/fermés : 199/193. L'audit indépendant
stratifié de 20 lignes trouve zéro faux réalisme certain et rend `PASS`.

Le promoteur a été corrigé avant publication pour filtrer par clé
`(seed_id, variant_id)` et ne jamais entraîner une variante sœur ambiguë ou
rejetée. Les caps références/opérateurs/paires sont maintenant réservés et
vérifiés cumulativement entre batches, au lieu d'être remis à zéro. La
sélection P001 s'arrête fail-closed avant Luna : après exclusion des cibles
P000, les seules transformations nominales strictes restantes ne permettent
plus de maintenir 20 % d'`EASY` sur 250 ou 500 nouvelles cibles. Aucun P001
n'a été écrit. La voie d'extension sûre est d'ajouter les noms alternatifs et
enseignes officiels SIRENE comme relation autoritative validée, plutôt que de
réactiver les subsets nominaux libres ou de relâcher les quotas en silence.

*(promotion exacte par variante : commit `cb61310`; caps cumulatifs : commit
`a8ce07d`; artefacts P000 :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/balanced_v1`)*

L'audit final post-production est préparé sans modifier le runner. Il ne peut
démarrer qu'après 20 000 promotions enregistrées, reconstruit 100 % des
contrats, provenances et audits full-SIRENE scellés, puis publie séparément
SIRET exact et équivalence opérationnelle. Sa vue distributionnelle inclut le
CRM réel complet, les seuls folds train 2/3/4 et l'union disponible. Une seule
revue de réalisme déterministe, stratifiée et bornée à 200 surfaces produit le
verdict downstream ; elle ne peut interrompre la génération en cours. Runbook :
`docs/synthetic_gt_balanced_final_audit.md`.

*(audit final synthétique : commit `98c25fe`)*

L'audit des batches P015–P018 explique l'essentiel de l'exhaustion croissante
sans baisse générale de Luna. Le snapshot terminal compte respectivement
91, 115, 107 et 129 slots épuisés sur 600. Parmi ces 442 slots, 397 portent
une suppression de ponctuation sans espace à une frontière lexicale située
avant les deux derniers tokens. La surface exacte retire alors une frontière
de token avant la fin du nom ; le runtime gelé réindexe la sortie et ne peut
plus retrouver les paramètres sources. Ces 397 contrats ont tous échoué ;
hors de cette classe, 1 958 slots passent et 45 seulement s'épuisent.

Le sélecteur refuse désormais cette seule classe non validable avant Luna. Il
conserve les suppressions avec espace, les marques de bord et la jonction de
la dernière paire lexicale ; quotas, critic, runtime et qualité ne sont pas
relâchés. Le MILP doit remplacer ces capacités en gardant états et difficultés,
ou échouer avant génération. Les 45 autres exhaustions restent 32 erreurs Luna
ou répétitions et 13 gardes saines d'ambiguïté/contrat. Les 66 tests ciblés du
sélecteur, des fragments et du runtime passent.

*(filtre fail-closed des jonctions non validables : commit `9922992`)*

### Évaluation nocturne réel seul vs réel + synthétique — protocole prêt

Le protocole apparié est gelé avant le corpus complet. Il entraîne uniquement
sur les folds 2/3/4 et évalue sur le fold 0 réel ; le fold 1 et le test restent
fermés. Le mixeur sélectionne une scène synthétique pour deux scènes réelles,
stratifiée par difficulté et famille d'augmentation, avec un poids `0.5/k` par
SIRET cible. Il refuse tout SIREN chevauchant le corpus réel, toute vérité
injectée, tout pool au-delà de 100 et toute autorisation du synthétique pour le
risk model, la calibration ou les seuils AUTO.

Deux comparaisons reproductibles sont prêtes : XGBoost reconstruit les bras
`REAL_ONLY` et `REAL_PLUS_SYNTHETIC` avec les mêmes 129 features et les mêmes
hyperparamètres ; BGE réutilise le contrôle réel publié et entraîne un seul
bras augmenté. La loss BGE accepte maintenant le poids au niveau de la scène
groupwise. Les métriques exactes et opérationnelles même-site sont publiées
séparément. Runbook : `docs/synthetic_augmented_model_eval_runbook.md`.

Le chemin de matérialisation du bundle non injecté
`sireto-synthetic-gt-model-features-1` est maintenant exécutable. Le premier
builder agrège les manifests `P*_promoted`, refuse moins de 20 000 variantes,
vérifie la disjonction de tous les SIREN réels et assigne les folds 2/3/4 avant
retrieval. Deux replays `audit_retrieval_channels.py` k=5000 alimentent ensuite
le finaliseur : admission gelée à 100, features BUSINESS, textes et groupes
BGE. Le ranker réel publié sert uniquement au minage des négatifs ; les
siblings même SIREN/même identifiant d'adresse sont exclus des hard negatives.
Les commandes exactes et les budgets temps/disque sont dans le runbook. Aucun
fit long n'a été lancé pendant cette préparation. Huit tests ciblés passent,
ainsi que deux smokes réels de projection source/SIRENE.

*(préparation mix, XGBoost, BGE pondéré et comparateurs : commit `541b560`)*
*(raccord retrieval top100 et bundle BUSINESS/BGE : commit `79b3a3b`)*

L'expérience préenregistrée est terminée sur le fold 0 réel, sans ouvrir le
fold 1 ni le test. Le corpus final scellé contient exactement 20 000 surfaces
et 9 737 SIRET (`final_corpus_v1/promoted_20000.jsonl`, SHA-256
`9e871f0f3c5a19d28a59619c4fd09c87be5d1e75e54296ff41bf34a4dd5cbcc1`).
La revue humaine bornée de réalisme reste `PENDING_BOUNDED_REVIEW`; les gates
déterministes et full-SIRENE sont complets. Le snapshot retrieval
`6281c9d1470f3913` et le bundle `9f99de01516dde9a` restent sans injection :
8 430/20 000 vérités sont naturellement présentes dans l'admission top100.
Le mix `71ceda354734fb7a` retient 8 192 scènes réelles et 4 096 synthétiques,
pondérées `0.5/k` et limitées aux folds train 2/3/4.

XGBoost rend `STOP_SYNTHETIC_AUGMENTATION_XGB` : 2 435/2 797 exacts pour le
bras réel seul contre 2 430/2 797 pour le bras augmenté ; la vue opérationnelle
passe de 2 451 à 2 446. Les corrections/régressions appariées sont 14/19.
L'actif perd 5 exacts, le fermé et le difficile restent inchangés. Artefact :
`synthetic_augmented_xgb_v1/bf439dfbad1584c5`, 55 s murales.

BGE rend `STOP_SYNTHETIC_AUGMENTATION_BGE` : 2 400/2 797 exacts pour le
contrôle réel publié contre 2 403/2 797 pour le bras augmenté, sous le gate de
+10 ; la vue opérationnelle reste à 2 418/2 797. L'actif gagne 7 exacts, le
fermé en perd 4 et le difficile reste à 32/38. Les corrections/régressions
appariées sont 56/53. Artefacts : `synthetic_augmented_bge_v1/47ac65d7f3f4fbf0`
et `synthetic_augmented_bge_comparison_v1/b01dc7d33958f72f`. Fit : 4 h 15 min
38 s ; scoring : 2 h 55 min 54 s ; pic RSS : environ 3,33 Gio. Aucun risk
model, calibration ou seuil AUTO n'a été entraîné. Rapport complet :
`docs/synthetic_augmented_model_eval_results.md`. Le correctif qui garde la
sortie CLI du builder lisible par l'orchestrateur est `07b288a`.

### Corpus GT synthétique corrigé — v2 final scellé

Le corpus `final_corpus_v1` est définitivement en quarantaine : sa première
revue bornée avait trouvé 2 faux réalistes certains sur 200. Le runtime et le
scanner partagé rejettent désormais sans réparer le texte les mutations de
type de voie hors composant officiel, les élisions `DE L'` cassées, les doubles
espaces et les acronymes asymétriques, y compris la soudure de la dernière
initiale au mot suivant. Le scan cumulatif a mis en quarantaine 453 lignes :
48 verrous de composant de voie, 326 élisions, 58 acronymes, 9 doubles espaces
d'adresse et 13 de nom, avec un recouvrement entre raisons.

Les remplacements ont été écrits uniquement par Luna dans les batches
`P039`–`P043`, critiqués puis qualifiés full-SIRENE exacts avant promotion.
Quatre remplacements intermédiaires de `P039` ont eux-mêmes été repris par la
quarantaine cumulative ; le registre terminal compense exactement toutes les
lignes exclues. Le rescan final examine 20 000 lignes sûres et ne trouve aucune
nouvelle violation déterministe (`new_quarantined_rows=0`, statut `PASS`).

Le corpus publié atomiquement contient exactement 20 000 surfaces distinctes
et 9 869 SIRET, zéro doublon, zéro injection positive et uniquement des
promotions `G_N_A` full-SIRENE exactes. Les quotas finaux sont exacts : 4 000
`EASY`, 11 110 `MEDIUM`, 4 890 `HARD`; 3 000 `FAIL_BGE_ONLY`, 4 000
`FAIL_BOTH_MODELS`, 3 000 `FAIL_XGB_ONLY`, 2 000 `NEAR_CLEAN_CONTROL` et
8 000 `TRAIN_DISTRIBUTION`. Les états sont 10 000 actifs / 10 000 fermés, avec
5 369/4 500 identités et 54,403 % d'identités actives. Tous les caps globaux
sont respectés.

Artefacts scellés sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/balanced_v1` :

- registre `production_registry_v3_final.json`, SHA-256
  `267e4043f67a1b37db8ff5176532cdd0f8ed754a719ec56bd1cbb3a152f138b4` ;
- quarantaine cumulative
  `v2_realism_quarantine_cumulative_with_acronyms.json`, SHA-256
  `060b49ec0cfdfa4e95c09662848e2a207fb1016c9327e06443a7a129f865666d` ;
- rescan `v2_realism_final_rescan.json`, SHA-256
  `b9ce068bf607fda6c670f1af2b65294cc399646ad80451df963508869d069935` ;
- corpus `final_corpus_v2/promoted_20000.jsonl`, SHA-256
  `1d370e51512bbd5d574072c046e49486eb40df753c20c6243ab3095d4d3f45ce` ;
- manifeste final `final_corpus_v2/manifest.json`, SHA-256
  `bee805770f4dd822a2e066f8d041dce0a5e7b4f7cef60feeba881c1ec92a17a8`.

La seconde revue indépendante porte sur 200 seeds nouvelles, une surface par
seed, sans aucun recouvrement avec les 400 lignes des deux échantillons
précédents. Elle rend 199 `PASS`, 0 `BORDERLINE` et 1
`CERTAIN_FALSE_REALISM` (`DE L ARBRE` devenu `DE ARBRE`). Le verdict final
préenregistré reste `PASS`, car le seuil de pause est 2 faux certains sur 200 ;
ce cas reste explicitement publié dans l'audit, pas masqué. Rapport
`final_audit_v2/report.json`, SHA-256
`e12cf9256643d2b20107492562a771c67dcc6403f3467fc5a039f4c094f5d5e0` ;
revue SHA-256
`012e51c9f168fbb3b4c858473d678a1682dc0ca7b41b5108c700c1e12de5fd37`.

*(gates et quarantaine initiale : `e64764f`; comptage post-quarantaine :
`c5cc118`; réutilisation terminale sûre : `d2c3189`; échantillon frais :
`46fe47e`; soudure d'acronyme : `5dadcc7`; quarantaines cumulatives :
`b88520f`, `4dfc2e6`; exclusions multi-échantillons : `fd1be49`)*

### Rerun XGBoost sur le corpus corrigé v2

L'audit delta retrouve 86 scènes retirées parmi les 4 096 scènes synthétiques
de l'ancien mix ; les deux faux réalistes certains de la première revue
humaine n'étaient pas sélectionnés. Le préparateur est désormais lié
directement au manifeste du corpus final audité au lieu de redécouvrir des
promotions de batch supersédées. La source v2 est `799bf5b289a0e943`, les
replays complets sans injection sont
`synthetic_gt_v7_channels_799bf5b289a0e943` et
`synthetic_gt_overlay_channels_799bf5b289a0e943`, tous deux sans mismatch.

Le bundle `aa30dbeecaadd8d0` retrouve naturellement 8 472/20 000 vérités dans
le top100, contient 861 739 lignes candidat et conserve
`positive_injection=false`. Le mix `34decc91a18ad5f7` contient toujours 8 192
scènes réelles et 4 096 synthétiques pondérées `0.5/k`; 3 995 scènes
synthétiques sont communes au mix v1 et 101 changent.

Le rerun fold0 `synthetic_augmented_xgb_v1/5f9a4228ff4ab939` confirme
`STOP_SYNTHETIC_AUGMENTATION_XGB` : le contrôle reste à 2 435/2 797 exacts,
le bras v2 tombe à 2 424, soit -11. La vue opérationnelle passe de 2 451 à
2 440 ; l'actif perd dix cas, le fermé un, le difficile reste à 32/38. Les
corrections/régressions appariées sont 11/22. Fold1, test, risk model,
calibration et seuils AUTO restent fermés. BGE n'est pas relancé : XGBoost ne
révèle aucune surprise favorable et son ancien `STOP` manquait simultanément
quatre gates pour une contamination d'environ 0,21 % du poids total.

Rapport détaillé : `docs/synthetic_augmented_model_eval_results.md`.

*(raccord explicite au corpus final audité : commit `302bb25`)*

### Corpus GT synthétique — boucle agentique Luna v2

La génération mécanique Python est retirée du chemin autorisé. Le runtime
unique `scripts/run_synthetic_gt_agentic_loop.py` orchestre désormais deux
workers Luna `low`, un critique indépendant et un adjudicateur conditionnel
au moyen d'une file SQLite WAL, de leases expirables et d'un journal
append-only. Le code ne rédige et ne répare aucun champ CRM : il assigne les
seeds, valide les JSON, rejette les fuites SIRET/SIREN et les doublons globaux,
route les cas risqués et conserve les réponses brutes avec SHA-256.

Le pilote réel v4 est exécuté sur quatre seeds sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/5b6362f76951fa84`.
Deux shards GENERATOR Luna LOW ont été loués, puis deux tâches CRITIC et une
tâche ADJUDICATOR. L'export final est `export_v4_final/` : 5 variantes
`ACCEPT`, 1 `SILVER`, 6 `REJECT` au preflight et 2 seeds épuisant leur retry
sans promotion. Le gate de trois positifs acceptés par seed n'est donc pas
franchi : verdict `PIVOT_PILOT`, aucune montée en volume. Le manifeste export
contient les hashes des trois JSONL de sortie. Quinze tests ciblés passent,
dont reprise de lease, critique aveugle aux résumés du générateur,
adjudication fail-closed et absence de générateur mécanique dans le runtime.

*(pilote v4 et export : artefacts lourds sous `/Volumes/CATNAT_DATA`; exécution
validée après le runtime, commit de référence `c786ae2`; contrat de réponse
GENERATOR renforcé par `e5ab523`)*

Un assembleur de hard negatives SIRENE-only a été ajouté (`554afcb`). Il ne
transforme aucun texte CRM : il sélectionne au plus dix SIRET candidats par
positif, avec familles `SAME_SIREN_OTHER_SIRET`, `SHARED_ADDRESS`,
`ACTIVE_CLOSED`, `LOCAL_HOMONYM` et `GEOGRAPHIC_NEIGHBOR`, provenance des
champs sources et manifeste hashé. Sur l'export pilote, 50 paires sont
produites (16 même SIREN, 34 actif/fermé); les familles absentes du petit pool
restent explicitement absentes, sans remplissage artificiel.

Une itération v6 sur 8 seeds train-only a ensuite été préparée depuis les
sources hashées et exécutée avec 2 workers GENERATOR Luna LOW (4 seeds par
worker), puis 1 CRITIC et 1 ADJUDICATOR. Export :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/67653eb32d9998cc/export_v6/`.
Résultat : 3 `ACCEPT`, 18 `SILVER`, 3 `REJECT`; le gate de 3 positifs
acceptés par seed n'est pas atteint, donc verdict `PIVOT_PILOT` maintenu.
Les hard negatives correspondants comptent 30 paires (9 même SIREN,
21 actif/fermé). Aucun artefact v6 n'est promu en corpus d'entraînement.

Le prompt GENERATOR a ensuite été renforcé (`8704a4d`) : toute variante
destinée à être acceptée doit conserver au moins deux ancres indépendantes
et ne peut pas omettre simultanément nom/enseigne et adresse-numéro. Une
nouvelle itération pilote est requise avant toute extension.

Le pilote v7 a été exécuté avec le prompt renforcé et les deux shards GENERATOR
Luna LOW déjà loués. Les 8 seeds ont produit 22 variantes `ACCEPT` et 2
`REJECT`, sans `SILVER`; six seeds seulement disposent de trois variantes
acceptées. L'export est
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/67653eb32d9998cc/export_v7/`.
Le filtre de complétude a retenu 6 SIRET et 18 positifs pour les hard
negatives, soit 180 paires : 132 `ACTIVE_CLOSED` et 48
`SAME_SIREN_OTHER_SIRET`. Les autres familles sont absentes du pool local et
n'ont pas été fabriquées. Le gate de volume/diversité n'est pas franchi :
verdict `PIVOT_PILOT`, aucune extension ni promotion. Commit du filtre et de
ses tests : `730816d`.

Un intake séparé `SIRENE_ONLY_TRAIN` a ensuite été matérialisé par
`scripts/select_synthetic_gt_sirene_seeds.py` (commit `1d4ffd6`). Le sélecteur
ne transforme aucun texte : il exclut les 14 198 SIREN de `crm_ok_gt`, conserve
un seul SIRET par SIREN selon un ordre hashé stable et écrit les champs
officiels SIRENE avec provenance. L'artefact pilote contient 20 000 SIRET et
20 000 SIREN distincts, zéro recouvrement CRM, et le manifeste
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/sirene_only_train_seeds_v1.jsonl.manifest.json`
porte les hashes source et de sortie. Cet intake rend possible l'atteinte du
nombre de seeds préenregistré, mais ne constitue pas encore des positifs :
aucune génération Luna n'a été lancée sur ces seeds.

Un pilote agentique SIRENE-only v1 a ensuite été exécuté sur 4 seeds avec deux
leases GENERATOR Luna LOW, une reprise bornée, un CRITIC et un ADJUDICATOR.
Trois seeds ont atteint la critique ; une seed a été abandonnée fail-closed
car sa fiche officielle ne contenait ni nom/enseigne ni numéro/voie. Résultat
exporté dans
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/sirene_pilot_export_v1/` :
0 `ACCEPT`, 9 `SILVER`, 0 `REJECT`. Le pilote démontre la provenance et la
reprise, mais échoue le gate de positivité et de diversité utile ; aucune
extension n'est autorisée. Le runtime accepte désormais explicitement
`SIRENE_ONLY_TRAIN` avec `oof_fold=-1` et fournit une clôture auditable des
seeds non identifiables, sans inventer de texte (commit `91e775d`).

Le smoke suivant a renforcé l'intake avec une garde stricte contre les valeurs
SIRENE `[ND]` (`34bf4b1`). Sur 4 seeds désormais identifiables (nom/enseigne,
numéro/voie, CP et commune), le run
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/sirene_identifiable_export_v3/`
a produit 3 `ACCEPT` et 9 `REJECT` après critique ciblée et une reprise
GENERATOR bornée. Le taux de variantes acceptées reste 25 % et le gate de
trois variantes acceptées par seed n'est franchi que pour 1 seed ; verdict
`PIVOT_PILOT`, sans extension ni hard negatives promus.

Un micro-pilote v4 a testé une consigne renforcée d'empreintes structurelles
distinctes sur une seule seed. Luna LOW a encore produit une variante au
`name` vide ; le schéma a rejeté la réponse avant tout preflight et la tâche a
été clôturée `ABANDONED` avec raison
`GENERATOR_SCHEMA_INVALID_EMPTY_NAME_AFTER_DIVERSITY_RETRY`. Ce résultat
confirme le `PIVOT_PILOT` : aucune génération massive ni multiplication
artificielle de variantes n'est autorisée tant que le débit de sorties valides
et distinctes n'est pas démontré.

Une vérification du schéma a précisé que les champs CRM vides sont autorisés
pour `FIELD_MISSING`; l'échec v4 provenait de la liste de familles vide sur
une variante inchangée, exigée non vide par le contrat. Un micro-pilote v5,
avec ce point explicitement rappelé à Luna, a produit 3 `ACCEPT` sur une seed
et une critique indépendante sans `SILVER` ni `REJECT`. Export :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/sirene_micro_export_v5/`.
Ce résultat valide le chemin agentique corrigé sur un cas, mais ne constitue
pas un gate d'extension : le rendement v3 reste insuffisant et les hard
negatives ne sont pas encore promus.

La chaîne hard-negative a ensuite été exercée sur l'export v5. Le producteur
de cartes SIRENE officielles (`02d05d9`) a projeté 3 617 candidats autour de
la seed, sans modifier de texte CRM. Le builder a produit 30 paires uniques
pour les 3 positifs : 21 `ACTIVE_CLOSED`, 3
`SAME_SIREN_OTHER_SIRET` et 6 `SHARED_ADDRESS`, avec provenance complète,
zéro collision positif/négatif et hashes des sources. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/sirene_micro_candidate_cards_v5.jsonl`
et `sirene_micro_hard_negatives_v5.jsonl` avec leurs manifests. Cette
milestone valide la chaîne de négatifs sur un micro-lot, sans constituer une
promotion du corpus ni une extension quantitative.

L'audit distributionnel reproductible a été ajouté (`02a47f2`) et exécuté sur
les exports v3 et v5. Pour v5, les 3 positifs ont trois empreintes uniques,
100 % de présence nom/adresse/CP/commune, 66,7 % de présence INSEE et les
familles `ADDRESS_ABBREVIATION`, `ACCENT_PUNCTUATION` et `FIELD_MISSING`.
Le rapport conserve le profil observé de 7 095 lignes train, ses hashes, et
confirme `retrieval_inputs_used=false`. Ces statistiques restent un audit de
micro-lot, pas une preuve de fidélité distributionnelle à l'échelle du corpus.

Une calibration v6 bornée à 8 seeds identifiables a ensuite passé tout le
chemin agentique : 24/24 variantes `ACCEPT`, 24 empreintes uniques et aucun
`SILVER`/`REJECT`. Le rapport est
`sirene_calibration_distribution_v6.json`. Les cartes candidates optimisées
(`890a287`) contiennent 163 009 projections SIRENE officielles ; le builder a
produit 240 hard negatives uniques (156 `ACTIVE_CLOSED`, 15
`SAME_SIREN_OTHER_SIRET`, 69 `SHARED_ADDRESS`), avec zéro collision positif /
négatif et zéro provenance manquante. Cette calibration mesure 3 variantes
acceptées par seed sur 8/8 seeds, mais ne prouve pas encore le volume minimal
20 000 / 60 000 ni la fidélité distributionnelle globale.

L'extension bornée v7 a ensuite traité 32 seeds avec deux shards GENERATOR
Luna LOW, CRITIC exhaustif et adjudication des deux cas non unanimement
acceptés. Résultat : 93 `ACCEPT`, 3 `SILVER`, 0 `REJECT`; 30 seeds ont leurs
3 variantes acceptées. La comparaison à v6 trouve 73 signatures structurelles
nouvelles sur 93 (78,5 %). Les cartes SIRENE ont fourni 900 hard negatives
uniques sur les 90 positifs éligibles : 516 `ACTIVE_CLOSED`, 105
`SAME_SIREN_OTHER_SIRET`, 279 `SHARED_ADDRESS`, avec zéro collision
positif/négatif. Le palier est classé `CONTINUE`, mais aucun corpus final n'est
encore promu ; les artefacts et manifests sont sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/`.

L'extension v8 a traité 64 seeds. Après CRITIC exhaustif et adjudication des
4 cas ambigus, 187 variantes sont `ACCEPT`, 5 `SILVER`, 0 `REJECT`; 60 seeds
ont trois variantes acceptées. L'audit distributionnel v8 compte 187
empreintes uniques, et 97 sont nouvelles par rapport à v7 (51,9 %). Les
cartes SIRENE ont produit 1 800 hard negatives uniques sur 180 positifs
éligibles : 858 `ACTIVE_CLOSED`, 264 `SAME_SIREN_OTHER_SIRET`, 678
`SHARED_ADDRESS`, sans collision positif/négatif. Le palier reste
`CONTINUE` conditionnel ; les objectifs 20 000 seeds / 60 000 positifs ne sont
pas encore atteints.

L'extension v9 a traité 128 seeds disjointes (offset 64, lot 128) avec les
deux shards GENERATOR Luna LOW, puis CRITIC et adjudication ciblée. Elle
produit 362 `ACCEPT`, 22 `SILVER`, 0 `REJECT` ; 106 seeds ont trois variantes
acceptées et 22 en ont deux. Les 362 empreintes sont nouvelles par rapport à
v8. L'audit distributionnel signale une concentration sur
`ADDRESS_ABBREVIATION` et `FIELD_MISSING`, à surveiller avant un nouveau
palier. Les cartes officielles SIRENE comptent 8 596 108 projections pour
128 seeds ; les 318 positifs issus de 106 seeds complets ont produit 3 180
hard negatives : 1 317 `ACTIVE_CLOSED`, 594 `SAME_SIREN_OTHER_SIRET` et 1 269
`SHARED_ADDRESS`, sans collision positif/négatif. Ce palier reste
`CONTINUE` conditionnel et ne constitue pas le corpus final ; les artefacts
v9 et leurs manifests sont sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_pilot/`.

*(runtime, schéma v2, prompts, runbook et tests : commit `c786ae2`)*

### Corpus GT synthétique SIRETO — préenregistrement

Le cycle train-only est préenregistré dans `docs/synthetic_gt_corpus_contract.md`
et `config/synthetic_gt_corpus_plan.json`, avec une branche séparée
`MAPS_ASSISTED` désactivée par défaut et son avenant
`docs/synthetic_gt_corpus_maps_addendum.md`. La population autorisée joint
`crm_ok_gt.csv` aux assignments V4.12-L et conserve uniquement
`legacy_split=train` et les folds OOF 2/3/4 : 7 095 seeds, 5 961 composants,
zéro recouvrement avec les folds 0/1 ou le test. Aucun appel Maps n'a été
effectué ; le secret éventuel ne pourra être lu que depuis
`SIRETO_GOOGLE_MAPS_API_KEY`, sans journalisation.

*(préenregistrement et avenant : commit `9a79fbb`; sources, pins, quotas et
gates sont dans le plan)*

### V4.12-BGE — cycle BGE + XGBoost préenregistré

Le nouveau cycle teste BGE fine-tuné seul puis, en priorité, ses scores
cross-fittés comme features d'un méta-ranker XGBoost sur le top 10
`BUSINESS_LEARNED`. Les folds 2/3/4 servent exclusivement à apprendre, le
fold 0 à sélectionner et le fold 1 reste fermé jusqu'au gate. Le test final
historique reste fermé.

Une seule configuration BGE est autorisée : loss groupwise, un positif
réellement présent + quinze négatifs maximum, quatre couches supérieures,
`1e-5`, un epoch et seed 42. Le stack utilise les 129 features métier et des
scores/rangs/marges/accords BGE strictement OOF. Il n'existe ni règle par
dossier, ni injection positive, ni tuning libre.

Gate fold 0 : 2 452/2 797 exacts, 33/38 difficiles, 2 164/2 391 actifs,
246/406 fermés, couverture complète et zéro fuite. Le fold 1 n'est ouvert
qu'une fois après franchissement ; `GO` y exige au moins +10 réponses nettes
face à `BUSINESS_LEARNED`.

Le contrat inclut désormais le vrai gate produit avant Google Maps. Après
confirmation du ranker, un accepteur strictement nested OOF compare une
baseline `BUSINESS_LEARNED` et le nouveau stack avec leurs accords, rangs,
marges, stabilité top 1 et concurrence SIREN. Il ne change jamais le SIRET.
Un `GO_PRODUCT_PRE_MAPS` exige une précision AUTO observée d'au moins 99,8 %,
zéro cas ouvert audité automatisé et au moins +1 point de couverture AUTO
totale face à la baseline reconstruite. Les AUTO/17 097, AUTO/13 704,
REVIEW/appels Maps théoriques et Wilson 95/99 seront publiés séparément.
Aucune API Maps ne sera appelée dans ce cycle.

Contrat : `docs/v412_bge_xgb_stack_contract.md`.

*(préenregistrement : commit `b6b6006`; seuils de minage : `b1fa89c` ; gate
pré-Maps : `0081c82`; quotas négatifs équilibrés : `4a033a3`; présent
handover : commit du présent milestone)*

Les groupes BGE définitifs sont publiés sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_bge_training_groups/114b407f2ccf7b40` :
8 192 scènes, 131 072 paires, exactement un positif réel et quinze négatifs,
zéro injection et zéro SIREN vérité traversant les folds 2/3/4. Les négatifs
comprennent 40 960 top XGBoost OOF, 1 659 autres sites du même SIREN, 12 028
homonymes/adresses fortes, 16 384 concurrents actif/fermé et 51 849 compléments
par rang. Le premier build trop large `12b9127e397bbc65` est supersédé avant
entraînement mais conservé physiquement pour traçabilité.

*(builder, tests et rapport : commit `fcd41d0`; présent handover : commit du
présent milestone)*

### V4.12-N — reranker neuronal seul arrêté, pivot BGE + XGBoost

Le benchmark fold 0 est clos sans ouvrir le fold 1 ni le test final. La
baseline `BUSINESS_LEARNED` reste à **2 437/2 797 (87,129 %)**. BGE zéro-shot
obtient 2 171/2 797, CamemBERT zéro-shot 1 846/2 797 et Qwen 1 782/2 797.
Le fine-tuning groupwise de CamemBERT progresse à **2 353/2 797 (84,126 %)**
mais reste 84 réponses derrière XGBoost et sous tous les gates de promotion.

Le fine-tuning Qwen a été interrompu à 3 000/8 192 scènes sur instruction
explicite ; aucun modèle final ni score de sélection n'en est publié et le
pilote Qwen 1,7B est annulé. La seconde variante CamemBERT n'a produit aucun
artefact final et n'est pas retenue.

Verdict : **`STOP_PURE_NEURAL_REPLACEMENT`**. La complémentarité des erreurs
autorise un nouveau cycle distinct, préenregistré avant fit, consacré à BGE
fine-tuné puis à ses scores cross-fittés dans un stack déterministe avec
XGBoost. Les résultats du cycle clos sont immuables.

Rapport : `reports/v412_neural_ranker_benchmark.md`.

*(rapport de clôture : commit `b6b674c`; présent handover : commit du présent
milestone)*

### V4.12-N — expérience de reranker neuronal préenregistrée

Le nouveau cycle ne modifie ni le CRM, ni les labels, ni le retrieval : il
teste si un reranker de texte peut mieux choisir le SIRET parmi les 100
candidats V4.12-L. Les folds SIREN-disjoint sont figés avant mesure : 2/3/4
pour apprendre, 0 pour sélectionner et 1 pour une confirmation unique. Le test
final historique reste fermé.

Le tournoi compare Qwen3-Reranker-0.6B, GTE multilingual reranker et un
cross-encoder CamemBERT français, avec mMiniLM et BGE comme références. Les
deux meilleurs modèles spécialisés seront entraînés avec des groupes de vrais
candidats concurrents. Qwen3-1.7B listwise n'est autorisé qu'en pilote si les
rerankers spécialisés plafonnent. Tous les calculs restent sur le Mac et le
SSD externe, sans dépense.

Contrat : `docs/v412_neural_ranker_contract.md`.

*(préenregistrement : commit `704678c`; présent handover : commit du présent
milestone)*

### V4.12-N — corpus texte top-100 matérialisé

Le corpus commun contient 17 097 CRM et 1 708 184 candidats, sans dépasser
100 candidats par CRM. Aucun positif n'est injecté et le SIRET candidat n'est
pas écrit dans le texte donné aux modèles. La baseline XGBoost vaut
11 939/13 704 au total ; sur le fold 0 de sélection elle vaut 2 437/2 797,
dont 33/38 cas difficiles. Les sorties des nouveaux modèles sur le fold 1
restent fermées.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_neural_text_corpus/02b8668f8050c5e9`.

*(builder, test et rapport : commit `2a26446`; présent handover : commit du
présent milestone)*

### V4.12-N — scènes groupwise matérialisées

Les folds d'entraînement 2/3/4 fournissent 8 192 scènes de 16 candidats : une
vérité réellement présente et 15 concurrents. Les négatifs prioritaires sont
les autres établissements du même SIREN, puis le haut du retrieval gelé. Il
n'y a ni sélection XGBoost, ni règle de promotion, ni injection du positif.
Les folds 0 et 1 sont absents de cet artefact.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_neural_training_groups/55b5fa545d29fd26`.

*(builder, test et rapport : commit `895d88d`; entraînement encoder : commit
`e18b31e`; entraînement Qwen MLX : commit `a914dee`; présent handover : commit
du présent milestone)*

### V4.12-L — gate retrieval unifié franchi

La politique sélective gelée a été matérialisée à 100 candidats maximum pour
les 17 097 requêtes de la population apprise. Les 43 ajouts frais ont été
rejoués sur les mêmes canaux actif et overlay : leurs anciens pools V4.11
divergeaient et n'ont pas été mélangés à la mesure. Les 33 ajouts frais exacts
sont tous retrouvés.

La vue corrigée contient 13 704 `MATCH_EXACT` et atteint **13 604/13 704 =
99,270 % de Recall@100**, pour **80,154 % de couverture identifiable**. Aucun
pool ne dépasse 100 candidats. Les références publiées ensemble valent
96,106 % sur l'historique, 96,423 % sur V2 exact et 99,268 % sur V3 exact. Les
cinq plis OOF sont tous au-dessus de 99,15 %.

Verdict : **`GO_RANKER_TRAINING`** en développement consommé, sans prétention
de certification indépendante. Le segment actif vaut 99,552 %, contre 97,698
% pour les fermés ; les fermés restent donc des exemples auxiliaires pondérés
à 0,5 et non des cibles opérationnelles préférées.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f`.
Il contient les 17 097 pools, les outcomes, les métriques et les hashes. La
suite autorisée est la matérialisation des lignes candidat/features puis le
ranker XGBoost en cinq plis OOF, sans règle de promotion déterministe.

*(préparation du replay, évaluateur, tests et rapport : commit `bf41918` ;
présent handover : commit du présent milestone)*

### V4.12-L — population unifiée 17 097 pour apprentissage OOF

Le dataset appris est matérialisé sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_learned_unified_population/2d29be3ccd8fcc3e`.
Il lie les 17 054 lignes de `crm_ok_gt.csv` aux qualifications V3, remplace 236
labels par leurs adjudications locales, ajoute 43 dossiers frais audités et
applique deux corrections de contrôle déjà présentes. Deux contrôles frais
restent volontairement hors entraînement.

La population contient 13 704 `MATCH_EXACT`, 625 `AMBIGUOUS` et 2 768
`UNRESOLVED`. Parmi les exacts, 11 619 SIRET sont actifs et 2 085 fermés. Les
cinq plis OOF groupés par composante SIREN contiennent 3 499, 3 321, 3 539,
3 419 et 3 319 requêtes, avec zéro composante traversant deux plis. Aucun
candidat, hit, rang ou score retrieval n'entre dans le build. Les anciens
splits et les audits étant consommés, cette population sert au développement
OOF et ne recrée pas un test indépendant.

Le prochain milestone est la matérialisation uniforme des pools retrieval
gelés à 100 candidats maximum sur ces 17 097 requêtes, puis la publication des
métriques globale, active, fermée, mégapole et multi-site avant tout
réentraînement du ranker.

*(builder, test et rapport : commit `b0e71ba`; présent handover : commit du
présent milestone)*

### V4.12 — politique accepteur North Star : GO dev à zéro erreur observée

Sur les scènes role-aware, les sept `UNRESOLVED` ont été réintégrés comme
négatifs évaluables : la population difficile contient 227 top 1 corrects et
52 erreurs, ambiguïtés ou non-résolus. Aucun modèle CPU seul ne franchit le
gate prudent ; XGBoost sans contraintes est le meilleur candidat sûr avec
105/227 AUTO et 0/52 erreur en nested component-OOF.

L'union de ce modèle et de huit preuves métier directionnelles atteint
**149/227 (65,64 %) AUTO difficiles avec 0/52 erreur**, ainsi que 1 124/1 127
AUTO sur les contrôles positifs. La projection combinée vaut **1 273/1 406
(90,54 %) AUTO, 1 273/1 273 correct observé**. Les décisions individuelles
conservent le nom CRM et les raisons d'acceptation.

Verdict `GO_NORTH_STAR_DEV_ZERO_ERROR`, sans déploiement : les seuils des
règles ont été choisis sur le dev consommé, les contrôles ne contiennent que
des top 1 corrects, et cette projection ne certifie pas 99,8 %. Le test final
reste fermé. Prochaine preuve utile : figer exactement la politique et
l'évaluer sur un nouveau lot indépendant.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_cpu_families/7270609bd3d59376`.
Implémentation et rapport : commit
`a5ffbcd0c6d5ab161fdb77487c8e8756beb06e79`.

*(présent handover : commit du présent milestone)*

### V4.12 — règles métier génériques : 227/241 strict, zéro régression contrôle

Quatre règles sans condition sur `query_id` ont été ablatées au-dessus de
l'ensemble conservateur : garde du siège intra-SIREN, résolution des conflits
de rôle, préférence pour une activité opérationnelle face aux holdings et
choix de site intra-SIREN. La correction de périmètre est appliquée uniquement
à la mesure : aucun label n'entre dans la sélection des candidats.

La baseline ensemble vaut 219/241 sur le périmètre local strict. Les règles
séparées atteignent respectivement 220, 222, 220 et 225/241, toutes avec zéro
régression observée sur les 241 labels stricts et sur les 1 127 contrôles. Leur
composition atteint **227/241 (94,19 %) et 1 127/1 127 contrôles**, soit huit
corrections strictes, aucune régression stricte et un changement faux-vers-faux.
Le seuil demandé de 225/241 est franchi sans exception par dossier.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_generic_business_rules/6f428a66c8f273d4`.
Il contient les métriques d'ablation, les décisions et 137 736 candidats classés
(1 381 requêtes, rang maximal 100), avec les signaux métier et la trace des
règles. Le test final reste fermé.

*(implémentation : commit `66d0e9a`; présent handover : commit du présent
milestone)*

### V4.12 — parité train/serve du bundle validée

Le bundle `c2a01c6bca43a468` a été rechargé et exécuté sur 1 127 requêtes et
112 389 candidats de contrôle. Les top 1 ranker, les 80 features de scène, les
scores accepteur et les 1 077 AUTO / 50 REVIEW sont identiques bit à bit aux
artefacts de développement. Aucun pool ne dépasse 100 et aucun SIRET n'est
dupliqué dans un pool.

Verdict `PASS_BUNDLE_TRAIN_SERVE_PARITY`. Le dernier gate interne vérifiable
est fermé ; il reste uniquement le nouvel export CRM indépendant.

*(script, JSON, rapport et présent handover : commit du présent milestone)*

### V4.12 — bundle trusted-label figé

Le bundle `c2a01c6bca43a468` lie le ranker, l'accepteur, le seuil
`0.9886879324913025`, les ordres de features, la taxonomie, le retrieval V4.2
max 100 et tous les hashes d'entrée. Les artefacts copiés ont été revérifiés.

Chemin : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/bundles/v4_12_trusted/c2a01c6bca43a468`.
Le bundle n'autorise ni production ni ouverture d'un ancien test. Prochaine
preuve obligatoire : nouvel export CRM indépendant.

*(script, rapport et présent handover : commit du présent milestone)*

### V4.12 — gate dev conservateur franchi à 82,93 %

Après le pivot calibration/comparaison, le seuil conservateur est fixé à
`0.9886879324913025` sur tout l'OOF trusted consommé. Il automatise 89/279 cas
difficiles sans erreur observée. Sur 1 127 contrôles positifs dev non utilisés,
il automatise 1 077 dossiers sans erreur.

La projection combinée est 1 166/1 406 AUTO, soit 82,93 %, avec 0 erreur et 0
ambiguïté AUTO. La borne Wilson basse 95 % est 99,67 % : 99,8 % n'est pas
certifié. Verdict `GO_FREEZE_FOR_NEW_HOLDOUT`.

Les anciens holdouts sont consommés et les racines du nouvel intake CRM sont
absentes du SSD. Aucun test final n'est ouvert ni recyclé.

*(artefact `v4_12_acceptor_conservative/88e50a879d7fcc2b`; code, rapport et
présent handover : commit du présent milestone)*

### V4.12 — accepteur trusted-label : PIVOT

Avec le ranker corrigé et l'accepteur XGBoost monotone poids `10` figés, le
seuil appris sur 147 scènes sans erreur produit deux ambiguïtés AUTO sur les
132 scènes de comparaison. Le cumul OOF est 121/123 AUTO corrects, insuffisant
pour 99,8 %.

Les erreurs Promotrans Mondeville et Ligue AURA Handball sont des entités
co-localisées. Les gardes manuelles et surpondérations négatives bornées ont été
rejetées : elles restent fautives ou détruisent trop de couverture. Verdict
`PIVOT_ACCEPTOR`; prochaine étape : calibration conservatrice sur tout l'OOF
consommé, puis gate end-to-end dev. Test final fermé.

*(artefact `v4_12_trusted_acceptor/7bde8fd021ec1915`; code, rapport et présent
handover : commit du présent milestone)*

### V4.12 — ranker trusted-label : 216/254 OOF

Le ranker a été réentraîné sans changement de retrieval ni de features sur les
254 labels exacts de l'audit complet. Le poids `0,5` reste optimal : 216/254
bons top 1 OOF contre 168/254 pour la baseline, 52 erreurs corrigées et quatre
régressions. Les trois vérités absentes du pool restent des erreurs end-to-end.

Le contrôle dev non concerné reste à 1 127/1 127. Verdict
`GO_BUILD_TRUSTED_OOF_SCENES`; prochaine étape : accepteur sur les prédictions
OOF corrigées, test final toujours fermé.

*(artefact `v4_12_trusted_label_ranker/2f57628196fefce0`; code, rapport et
présent handover : commit du présent milestone)*

### V4.12 — audit métier des 279 REVIEW terminé

Les 279 REVIEW de développement sont tous adjudiqués : 254 SIRET exacts
identifiables, 25 ambiguïtés structurelles et aucun non-résolu. L'audit montre
que 218 des 254 dossiers identifiables avaient été transformés à tort en
`AMBIGUOUS` par la construction mécanique V4.

Sur les 99 dossiers réellement aveugles, le ranker corrigé trouve 75/89 SIRET
et l'accepteur clean-target automatise 32/99 dossiers, avec 32/32 corrects. Les
14 erreurs du ranker et les 10 ambiguïtés restent toutes en REVIEW. Ce volume
ne certifie pas 99,8 %.

Verdict `GO_RETRAIN_ON_TRUSTED_LABELS` : les 254 labels exacts et 25 abstentions
fiables permettent maintenant un apprentissage OOF corrigé. Retrieval inchangé,
test final fermé.

*(commits Git : dernier docket `b61664f`; labels `5c77a36`; résultats, bilan
métier et présent handover : commit du présent milestone)*

Les huit sources d'adjudication sont aussi normalisées dans
`reports/v412_review_trusted_labels_279.csv` : 279 identifiants uniques, 254
`MATCH_EXACT`, 25 `AMBIGUOUS`, tous dans le split dev. Ce fichier est l'entrée
canonique du prochain apprentissage OOF. *(commit Git : milestone de
normalisation qui suit l'audit `9f60f36`)*

### V4.12 — B2 aveugle : 6 AUTO sur 30, tous corrects

Le deuxième lot indépendant de 30 REVIEW a été gelé avant scoring puis
adjudiqué avec preuves : 28 `MATCH_EXACT`, deux `AMBIGUOUS`, aucun non résolu.
Le ranker corrigé trouve 23/28 SIRET exacts. L'accepteur clean-target figé
automatise 6/30 dossiers, tous corrects ; les cinq erreurs du ranker et les deux
ambiguïtés restent en REVIEW.

Le cumul des deux lots aveugles atteint 19 AUTO sur 60, avec 19/19 corrects,
mais ce volume ne certifie pas 99,8 %. Verdict `GO_COMPLETE_REMAINING_39` :
modèle et seuil inchangés, audit des 39 REVIEW encore vierges, test final fermé.

*(commits Git : docket `2a62d69`; labels `a653027`; résultats et présent
handover : commit du présent milestone)*

### V4.12 — clean-target validé sur 13 AUTO aveugles sans erreur

Un nouveau docket de 30 REVIEW parmi les 99 encore vierges a été gelé avant
scoring. L'audit métier produit 28 `MATCH_EXACT` actifs courants et deux
`AMBIGUOUS`. Le ranker corrigé trouve 24/28 vérités exactes.

Le candidat clean-target figé automatise **13/30 dossiers (43,33 %)** : les
13 sont corrects, aucune des deux ambiguïtés et aucune des quatre erreurs du
ranker ne sont automatisées. Ce résultat réfute l'abstention totale du modèle
précédent et confirme que le défaut principal venait des faux négatifs de la
cible d'apprentissage.

Verdict `GO_EXTEND_BLIND_MEASUREMENT`, sans déploiement : 13 AUTO sans erreur
ne suffisent pas à revendiquer 99,8 %. Le candidat reste figé pendant la mesure
des 69 REVIEW historiques non adjudiqués restants. Le test final reste fermé.

*(commit Git : docket `54f6815`; labels, résultats et rapport : commit du
présent milestone)*

### V4.12 — nettoyage de la cible accepteur : 47/143 AUTO difficiles OOF

La cause aval est confirmée : 881 `AMBIGUOUS` mécaniques non adjudiquées
étaient encore utilisées comme négatifs dans le fit de l'accepteur. Le candidat
nettoyé conserve 4 666 `MATCH_EXACT` historiques et les 143 dossiers difficiles
adjudiqués, mais retire ces 881 faux négatifs potentiels. Le modèle et les
features restent inchangés.

Le XGBoost monotone, poids difficile `10`, seuil figé
`0.9940522313117981`, automatise **47/143** scènes difficiles OOF sans erreur
ni ambiguïté, contre 3/143 auparavant. Après sélection, il automatise 8/30
dossiers du lot désormais consommé, également sans erreur. Verdict
`GO_NEXT_BLIND_DOCKET`, sans déploiement : les volumes sont insuffisants pour
certifier 99,8 %.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_clean_target/ac50c1c8c00344b5`.

*(commit Git : implémentation et rapport du présent milestone)*

### V4.12 — validation indépendante : accepteur à 0 % de couverture

Un docket de 30 REVIEW jamais adjudiqués a été gelé par hash avant tout score
du modèle corrigé. L'audit métier aveugle produit 26 `MATCH_EXACT`, quatre
`AMBIGUOUS` et zéro `UNRESOLVED`. Le ranker corrigé retrouve 24/26 vérités
exactes (92,31 %).

L'accepteur XGBoost monotone, poids difficile `10` et seuil figé
`0.8974587321281433`, envoie **30/30 dossiers en REVIEW**. Ses scores restent
entre `0,0308` et `0,8328`. Zéro erreur AUTO est donc obtenu au prix d'une
couverture AUTO nulle. Le `GO_NEW_INDEPENDENT_ACCEPTOR_DOCKET` fondé sur trois
AUTO difficiles ne se généralise pas.

Verdict : **`PIVOT_ACCEPTOR_REDESIGN`**. Le seuil ne doit pas être abaissé
après coup et la pondération ne doit plus être poursuivie. Le prochain travail
utile est de revoir la cible et les preuves de l'accepteur, puis de réserver un
autre lot parmi les 99 REVIEW historiques encore non adjudiqués pour la future
validation. Le test final reste fermé.

*(commit Git : docket `311603e`; labels, résultats et rapport : commit du
présent milestone)*

### V4.12 — ranker corrigé en progrès, accepteur toujours bloquant

L'overlay des 60 REVIEW supplémentaires est scellé : 56 `MATCH_EXACT`, quatre
`AMBIGUOUS`, 53 corrections de label, avec preuves et anciennes cibles
conservées. Il porte à 133 exacts et dix ambiguïtés la population difficile
adjudiquée utilisée en développement.

Le ranker pondéré `0,5`, évalué hors échantillon sur les 133 exacts, passe de
69 à **110 bons top 1** (82,71 %), corrige 43 dossiers et en régresse deux. Il
reste à 1 175/1 175 sur le contrôle classique disjoint. Les deux vérités
absentes du retrieval restent comptées comme erreurs.

L'accepteur XGBoost monotone pondéré `10` est le seul candidat sûr sur les deux
contrôles : 602/669 AUTO classiques sans erreur et 3/143 AUTO difficiles sans
erreur. Cette couverture difficile de **2,10 %** est insuffisante et trois AUTO
ne certifient rien. La régression logistique atteint 615/669 AUTO classiques
sans erreur mais commet une erreur parmi trois AUTO difficiles hors
échantillon. Verdict : `GO_NEW_INDEPENDENT_ACCEPTOR_DOCKET`, sans déploiement ni
ouverture du test final.

Artefacts :

- ranker :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_label_ranker/9fea31939cff7fea` ;
- scènes :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_label_stack/aae2ad5814ecfb5b` ;
- accepteur :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_acceptor_family_weight/c88e443950d188cf`.

*(commit Git : overlay `c140f55`; implémentation et rapport : commit du présent
milestone)*

### V4.12 — 60 REVIEW supplémentaires adjudiqués et cause des faux labels établie

Deux lots supplémentaires de 30 REVIEW historiques ont été traités sans
validation utilisateur : **56 `MATCH_EXACT` fiables, quatre `AMBIGUOUS`, zéro
`UNRESOLVED`**. Le ranker avait déjà le bon SIRET en top 1 dans **51/56** cas
exacts ; cinq erreurs relèvent réellement du classement. Au total, **53/60**
qualifications V4.1 doivent être corrigées.

L'audit de provenance écarte un bug de jointure dans V4.1 : le CRM y est relié
par `crm_record_id`/`SERVICE ID` unique et les labels restent identiques dans
V4.6 puis V4.11. La cause dominante est la politique V4
`MULTIPLE_ACTIVE_DIRECT_MATCHES → AMBIGUOUS`, trop prudente : parmi les 50
vérités exactes marquées ambiguës, 46 conservaient le bon SIRET historique et
49 avaient la vérité dans les correspondances directes. L'ancien benchmark V6A
avait séparément un vrai défaut d'identité positionnelle (`query_id` recréé
après filtrage), ce qui explique les anciens SIRET totalement étrangers sans
expliquer V4.1.

Le prochain essai autorisé est la reconstruction locale des scènes hors
échantillon avec l'overlay de ces 60 décisions, puis le réentraînement du
ranker et de l'accepteur. Ces dossiers doivent être exclus du choix de seuil et
de la comparaison classique ; le test final reste fermé.

*(commits Git : premier lot `7866e34`/`0215151`, second lot
`7edcc97`/`d3e14cd`; correction du diagnostic : commit du présent milestone)*

### V4.12-R30 — contrat de collecte gelé, réseau encore fermé

Le goal North Star est réactivé après le gel du service V4.12. L'inventaire
confirme que les labels déterministes V2/V3 et les datasets ranker/accepteur
historiques ne sont pas des vérités humaines indépendantes. Le meilleur noyau
ré-auditable reste V4.4/V4.7, déjà consommé, tandis que les 279 `REVIEW` du
dev V4.12-G n'ont jamais été adjudiqués. Il est donc injustifié de
réentraîner immédiatement le ranker ou l'accepteur.

Un pilote autonome de 30 REVIEW est préenregistré : dix multi-sites du même
SIREN, dix collisions entre SIREN et dix autres REVIEW. La sélection est
aveugle aux labels, déterministe et liée au SHA-256
`ec481d8db07165185fecc61bf437d868bfcbe4db6f4938a62b6c344e7000c2ee`.
La collecte est séparée en découverte d'identité sans top-1 puis comparaison
au pool gelé, bornée à trois requêtes et six pages par dossier, avec faits
atomiques, deux groupes indépendants et table de décision exhaustive. Le
launcher futur devra fermer les accès locaux par allowlist/denylist et
journal d'ouvertures avant tout réseau.

Le gate exige au moins 18 décisions fiables, quatre par strate, six négatifs
fiables et au moins un négatif dans les strates multi-sites et collision.
`SCALE_ADJUDICATION` autorisera seulement un lot historique plus large, jamais
un entraînement ou une revendication produit. Deux audits indépendants ont
reproduit population, exclusions, docket et hash, puis rendu
**`GO_CONTRACT`**. L'amendement ultérieur précise que quatre identifiants
préfixés `fresh:` appartiennent en réalité au dev historique déjà consommé et
ne constituent donc jamais un holdout frais.
*(commits Git : contrat `2418e9e`, handover initial `960efb6`, amendement
`694bd10`)*

Le builder label-free est maintenant implémenté et gelé. Les entrées sont
copiées, lues et revalidées par descripteurs ; les sorties sont écrites,
scellées, promues et rouvertes par descripteurs ; toute substitution de la
racine provoque un arrêt. Le chemin d'échec post-promotion utilise une
quarantaine atomique `NOREPLACE` et non destructive. Les attaques de
substitution, collisions de quarantaine, mutations d'ancêtres, hardlinks et
publications concurrentes sont couvertes. Deux audits indépendants rendent
**`GO_DOCKET_BUILDER_PHASE`** sur les hashes exacts :

- builder :
  `ed08c907a34f19389e436c1270ba7d07a35f8c1ce3472b26e501fde19625352d` ;
- tests :
  `0efec26a6e89858a34e8794e87ab33779abd9d6c49fe791b6a21a0d2fc0b7e7b` ;
- contrat :
  `f594800f4011ebf243987c36f31dd03d425f59d01226b03a1ddc2c11806592cc`.

La validation locale donne 21 tests ciblés, 28 tests dans la commande combinée
avec les suites V4.7 présentes, et 20/20 répétitions de publication
concurrente.
*(commit Git : builder et tests `9f942cf`)*

Le dossier canonique immuable est publié sous :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a/`

La revalidation indépendante confirme 30 requêtes, dix par strate, 90
requêtes de recherche préenregistrées, 3 000 lignes candidates, un plafond
strict de 100 par cas, le hash de sélection attendu et le sceau d'arbre
`8a9aade72e741e393bdd5647ae440f38793da879462b185640dbf8ac6cf02df0`.
Aucun label, accès réseau, modèle ou test final n'a été ouvert.

Le contrat d'exécution de la collecte est maintenant fermé et reproductible.
Il épingle la politique réseau, 14 autorités locales, les quotas, les schémas,
les archives et la séparation irréversible entre worker identité et worker
comparaison. La protection SSRF utilise 31 CIDR interdits et des replays
incluant IPv4/IPv6, NAT64, 6to4, Teredo et réponses mixtes. Les goldens
couvrent aussi quatre scènes de reconstruction de faits, sept scènes
`facts → evidence` et six décisions finales. Deux contre-audits indépendants
ont rendu **`GO_COLLECTION_CONTRACT`** sur :

- contrat :
  `af0550c86b5f06b62240289bfbb349bb7429883ae9bd8216e6fa6938be0f5cea` ;
- politique :
  `1238eb957f84c811ac64375c66a0d62e1bef977a139c0a685e669a5d18c63b88`.

*(commit Git : contrat, politique et vectors `0fbdd80`)*

Prochaine étape autorisée : implémenter les deux workers métier, les
validateurs et un run synthétique strictement hors réseau. La collecte réelle
reste interdite avant deux
**`GO_IDENTITY_BROKER_WORKER_PHASE`**. L'adjudication, le gate R30 complet,
l'entraînement et le test final restent également fermés.

Les primitives hors réseau sont désormais gelées. Le replay recalcule la
politique, les 14 pins, les versions de bibliothèques, le parseur DDG, les
identifiants, les faits avec spans UTF-8, les preuves et les adjudications.
Un audit indépendant a rendu **`GO_OFFLINE_REPLAY_PRIMITIVES`** après 25
tests adversariaux.
*(commit Git : replay et tests `d8e4722`)*

La frontière de processus est également prouvée par un worker natif arm64
déterministe et signé, ancré avec toute sa chaîne de build. Les deux workers
sont des processus `sandbox-exec` distincts. Le binaire temporaire est
authentifié, signale `READY`, puis est supprimé avant traitement. Le profil
refuse réseau, fork, exec/ré-exec, écritures et lectures hors capacités,
y compris les alias `/System/Volumes`. Le journal O_EXCL est chaîné et écrit
par descripteurs retenus ; la révocation est monotone ; les timeouts et les
quatre positions d'échec de pipe sont sans fuite. La suite combinée donne
64 tests réussis. Deux audits indépendants rendent
**`GO_OFFLINE_NATIVE_RUNTIME_PRIMITIVES`** et
**`GO_OFFLINE_NATIVE_BUILD_PRIMITIVES`**.
*(commit Git : runtime natif, build, binaire ancré et tests `467f096`)*

Le broker M2 hors ligne et son harness M1 sont maintenant gelés. Le broker
authentifie sans paramètres les deux Parquets identité et leur plan 30/90 ;
il n'ouvre jamais le docket de comparaison. Son transport injecté est scellé,
one-shot et sans socket. Les corps sont consommés par chunks d'au plus 64 Kio,
avec arrêt au premier dépassement, archivage O_EXCL avant parsing, recalcul des
faits depuis les octets bruts, quotas et ordre stricts, lookup SIRENE dérivé
et cache global par SIRET. Toute anomalie de streaming ferme l'intention,
empoisonne définitivement le broker et interdit la suite.
*(commit Git : broker M2 hors ligne et tests `3b6e567`)*

Le harness intégré exerce ce vrai broker sur les 30 dossiers et les 90
requêtes canoniques avec une fixture externe volontairement vide, sans SIRET,
SIREN ni preuve. Il archive les 90 réponses, scelle et révoque M2 avant toute
ouverture du docket et des 3 000 candidats. Exactement deux workers natifs
séquentiels sont lancés ; leur rôle reste explicitement limité au protocole et
aux digests, avec `native_business_logic_executed=false`. Aucun fait ni lookup
n'est fabriqué depuis un candidat. Deux builds sont byte-identical.
*(commit Git : fixture et harness M1-M2 `775787d`)*

Après deux premiers `NO_GO` qui ont détecté une preuve circulaire puis deux
erreurs de streaming, le contre-audit final rend
**`GO_M1_M2_INTEGRATION_HARNESS`** et **`GO_BROKER_M2_OFFLINE`**. La commande
combinée donne 110 tests réussis. Ces verdicts excluent explicitement M3, le
transport live et les tables de sortie de la section 5.

Ces GO ne permettent toujours aucun réseau. Étape suivante : implémenter les
vraies transformations des deux workers métier sur fixtures locales, puis les
writers/validateurs de la section 5 et obtenir deux
**`GO_IDENTITY_BROKER_WORKER_PHASE`** sur le run synthétique complet avant la
première collecte.

La fixture M3 indépendante est désormais gelée avec 17 scénarios réellement
exercés : recherches vide/top-5, déduplication, quotas, HTML, texte, PDF,
triple direct valide, identifiant distant rejeté, dates, SIRENE actif/fermé/
non unique, cache global, timeouts et dépassement. Ses quatre SIRET sont
générés localement avec Luhn, absents des 3 000 candidats et de leur univers
SIREN, et la fixture est explicitement `NOT_EVIDENCE`. Aucun fichier de
comparaison n'est ouvert lors du build. Le contre-audit rend
**`GO_M3_FIXTURE`**.
*(commit Git : fixture M3 et tests `7c036bf`)*

Le spike de workers Python sandboxés est également gelé. Un runtime arm64
relogeable minimal de 35 Mo est copié dans une capacité privée, inventorié et
haché avant/après ; exactement deux processus séquentiels communiquent par un
unique socket AF_UNIX hérité. Réseau, nouvelle socket, fork, exec/ré-exec,
spawn, subprocess et lectures hors capacité sont refusés. Le binaire est
supprimé après `READY` et avant `GATE`; environnement et CWD sont fermés, les
deadlines sont absolues et le cleanup couvre tous les échecs pré/post-spawn.
Après plusieurs `NO_GO` sur ré-exec, late-path, slow-drip et fuite de stage,
le contre-audit final rend **`GO_M3_PYTHON_SANDBOX_FEASIBILITY`** avec 27 tests
séquentiels. Ce verdict prouve uniquement la faisabilité : aucun broker,
Parquet, traitement métier, live ou sortie section 5 n'est exécuté.
*(commit Git : spike sandbox Python, pin et tests `870cafe`)*

Prochaine étape active : construire les deux workers Python réellement métier
sur cette frontière validée. Le premier orchestre M2 et produit les artefacts
d'identité depuis les archives ; le second reconstruit et scelle d'abord les
faits/preuves sans candidat, puis reçoit le docket après cette barrière pour
produire les comparaisons. Le C natif reste une sonde de capacités, jamais un
substitut digest-only au métier.

Les primitives de protocole M3 sont maintenant gelées. Le framing binaire
borne chaque frame à 64 Kio et les objets à 32 Mio, impose rôles, directions,
séquences et machine d'état, utilise JSON canonique et des deadlines absolues,
et transporte sans pickle les lots de 3 000 candidats. Les bombes de
profondeur/nœuds sont rejetées par pré-scan avant `json.loads`; l'encodage est
préflighté et une plage de séquences complète est réservée avant tout octet.
Le contre-audit rend **`GO_M3_PROTOCOL_PRIMITIVES`** après 53 tests.
*(commit Git : protocole M3 et tests `4c532f0`)*

Le cœur métier M3b est également gelé, mais avec une portée volontairement
conditionnelle. Il vérifie le plan 30×3, les 90 tentatives, les relations
SEARCH/DNS/PAGE, les décisions d'ouverture, domaines, quotas et slots, décode
uniquement `text/plain` depuis les octets raw, reconstruit spans/SIRET Luhn,
lookups SIRENE globaux et comparaisons 30×100. La validation URL rejoue IDNA
3.11/UTS46/STD3 et les suffixes épinglés. Aucun objet ne peut porter
`reliable=true` ni `provenance_verified=true`; même une source auto-attestée
reste `M3B_TEXT_PLAIN_IDNA311_CONDITIONAL_SUPPORT_NOT_LABEL`. Après sept
contre-audits successifs, le verdict final est
**`GO_M3B_CONDITIONAL_SUPPORT_PRIMITIVES`** avec 59 tests.
*(commit Git : cœur conditionnel M3b et tests `dc868c6`)*

Le prochain jalon n'est donc pas un label : il faut intégrer protocole,
sandbox, broker M2 et cœur M3b dans deux vrais workers, puis authentifier la
provenance par des reçus broker/store avant toute promotion vers une
adjudication fiable. HTML/PDF restent hors du cœur M3b tant que leurs
dépendances ne sont pas packagées dans le runtime sandboxé.

### V4.12-S — service persistant gelé après parité exacte et gate Mac

Le worker persistant complet est maintenant validé sur les 1 456 requêtes
`dev` déjà consommées, sans ouvrir de label ni de test final et sans
réentraîner de modèle. Il exécute réellement, requête par requête :

```text
retrieval sparse top 100
  → 45 features Ranker C
  → Ranker C gelé
  → 80 features de scène
  → accepteur COMPACT_LOGIT gelé
  → preuve directe sur l'univers géographique actif
  → veto V4.12-G
```

L'orchestrateur charge modèles et stores une seule fois par processus,
interdit réseau, fork, sous-processus enfant, reconstruction et écriture des
caches, lie les sources, modèles, closures et constantes de politique, puis
publie les sorties et leur rapport atomiquement. Les contre-audits ont fermé
successivement l'attestation terminale, les closures de méthodes, les helpers
globaux et les constantes non-callables ; seuls les deux états IDF
transitoires, recalculés sous verrou avant chaque génération de features,
restent explicitement mutables.
*(commits Git : orchestrateur/gate `97d74d7`, `d848a89`, `4fecf7e` ;
attestation `b06cfe8`, `459f360`, `fbfd431`, `4805574`, `c261408`)*

Quatre exécutions fail-closed ont été conservées avant le GO final :

1. divergence d'identité runtime causée par `platform.platform()` sous le
   hook interdisant `sw_vers` ;
2. même dépendance cachée dans la validation runtime des manifests legacy ;
3. équivalence physique Arrow `string`, mais metadata Pandas
   `StringDtype/object`, plus un champ descriptif `evaluation_partition`
   absent des requêtes assainies ;
4. tolérance `1e-15` appliquée au score accepteur, mais pas encore à sa copie
   strictement identique dans la trace de garde.

Les corrections remplacent l'identité runtime par `sys.version_info` et
`os.uname`, continuent d'exiger les deux plateformes legacy scellées
exactement, ne normalisent que le stockage de vraies chaînes/nulls et
appliquent la tolérance uniquement aux deux copies `float64` du même score.
Les valeurs texte, ordre, nullité, dtypes numériques, non-finis et écarts
supérieurs à `1e-15` restent refusés. La suite ciblée finale donne
`116 passed`.
*(commits Git : runtime lock `465cf13`, runtime legacy `443b2fb`, parité
texte logique `6789df2`, score dupliqué `bcec840` ; locks successifs isolés
`a0b3c9e`, `c4fef1f`, `f3c0511`, `fa2161f`, lock final `404f7da`)*

Le run officiel immuable est :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/runs/v4_12_service_parity_latency/4c50821748becdfc/`

- verrou SHA-256 :
  `471165d626483f53c767a7c68d8a2c24d3f8825dee5361fa41398ee4c83e7322` ;
- commit source lié :
  `bcec840a9d00b73488b7bed40d722df21d5e4c78` ;
- rapport SHA-256 :
  `9289aad1223f00ee4fed2908d07d34bfe76cfa0f9eaeb00376bcf9f1562131a9` ;
- verdict scellé : **`GO_V412_SERVICE_FREEZE`**.

| Mesure | V4.11 | V4.12-G |
|---|---:|---:|
| Parité | `EXACT_5_STAGES` | `EXACT_7_STAGES` |
| Requêtes | 1 456 | 1 456 |
| Candidats | 145 236 | 145 236 |
| Maximum par requête | 100 | 100 |
| p95 complet | 2,435 s | 2,770 s |
| Pic RSS | 3,65 Go | 5,53 Go |
| Chargements modèle/store | 1 / 1 | 1 / 1 |

Le ratio p95 est `1,1377`, sous le gate `2×`. Les misses lookup/cache scellé,
reconstructions et écritures sont tous à zéro. Deux audits indépendants ont
revalidé les Parquets, manifests, hashes, permissions, deux PID distincts,
rapport, sceau et calcul du gate ; tous deux rendent
**`GO_V412_SERVICE_FREEZE`**.

Ce résultat gèle une baseline de service reproductible sur le Mac. Il ne
mesure pas la précision produit, ne certifie donc pas encore la North Star
`99,8 %`, et ne justifie aucune ouverture du test final. Le prochain jalon
autorisé est de constituer par preuves traçables un jeu fiable de cas
difficiles déjà consommés (`REVIEW`, erreurs de ranker, ambiguïtés
multi-sites), puis de décider si un nouvel entraînement OOF du ranker ou de
l'accepteur est justifié. La certification produit restera impossible tant
qu'une collection CRM indépendante réellement fraîche n'est pas disponible.

### V4.12-S — référence de parité service assainie et doublement auditée

Le chantier de certification produit V4.13 reste fermé faute de nouvelle
collection CRM indépendante. Le travail autorisé a donc repris sur le chemin
produit déjà gelé V4.12-G, sans réentraînement, sans nouveau seuil et sans
ouvrir de test final.

Le contrat service/parité/latence fixe la chaîne sparse top 100, Ranker C,
scène 80 features, accepteur `COMPACT_LOGIT` au seuil
`0.8720916706888049`, puis veto de preuve directe V4.12-G. Le cœur aval pur
valide le plafond 100, les 45 features, les identités SIRET/SIREN et
l'irréversibilité du veto.
*(commits GitHub : contrat `ab2bbe7`, cœur aval `087ee2b`)*

Le builder de référence projette désormais neuf sources historiques gelées
sans désérialiser leurs colonnes de vérité. La sortie guard est reconstruite
depuis l'accepteur, les scènes et les preuves, puis comparée à la décision
historique indépendante. Après deux audits `STOP`, la frontière a été fermée :
capture immuable des octets sources avant projection, politique
score/seuil/décision recalculée, rangs et valeurs numériques stricts, API
production non injectable, provenance code/contrat/runtime dans le build ID,
staging et promotion ancrés par FD, revalidation terminale des huit Parquets
et liaison d'identité du répertoire nommé. Deux audits indépendants rendent
**GO** sur le commit exact
`fab0e52ba563e539e1ff3d1ab1869e51a360aa90`; 70 tests ciblés passent.
*(commits GitHub : builder initial `5a87baa`, durcissement `05b9254`,
liaison staging finale `fab0e52`)*

La référence immuable production est publiée sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/v4_12_service_parity/b4b7fef24c5e7036/`.
Son manifeste SHA-256 est
`cbcb3303107cd00f895561b49b8ad3a26e5c8e3df8a07777817e7a6ed97f2340`.
La revalidation indépendante confirme 1 456 requêtes, 145 236 candidats,
plafond 100, 1 177 décisions `AUTO_MATCH`, 279 `REVIEW`, schémas exacts,
hashes exacts et zéro colonne interdite. Cela prouve une parité d'ingénierie,
pas encore la précision produit de 99,8 %.

Cette étape historique est désormais close par le
`GO_V412_SERVICE_FREEZE` documenté ci-dessus. Le futur export CRM indépendant
reste indispensable pour toute certification produit.

### V4.12-S — retrieval, features et preuve directe persistants

Le premier étage du worker persistant est maintenant implémenté sans modifier
les modèles gelés. Le retrieval unitaire expose le pool INSEE complet, la
sélection avant lookup, les rangs de canaux et le snapshot nécessaires pour
recalculer exactement les 45 features Ranker C. Le service de features calcule
l'IDF sur le pool aligné complet, la densité sur le top 100 avant filtrage des
établissements fermés, puis hydrate l'ordre actif final. Le service de preuve
directe cherche séparément dans tout l'univers géographique actif via les
partitions strictes certifiées, jamais dans le top 100.
*(commit GitHub : étages retrieval et preuve initiaux `e902ab6`)*

Les contre-audits initiaux ont correctement arrêté l'intégration sur cinq
défauts : ancien store non scellé pour la preuve, ordre des features trop
permissif, contexte IDF concurrent, schéma vide absent et types divergents.
Le correctif utilise uniquement `StrictPartitionStore`, épingle les 39 codes
INSEE mégapoles du manifeste historique SHA-256 `a07bf9cd...`, filtre leur
partition INSEE certifiée par code postal, impose le tuple Ranker C et son
hash `760db4db...`, conserve un schéma vide exploitable et sérialise la
section globale IDF/features. La première correction avait modifié
`features.py`, pourtant scellé historiquement ; la correction suivante le
restaure octet pour octet et place le verrou uniquement dans le service.
*(commits GitHub : durcissement `13e2418`, restauration de la source scellée
et verrou service `bc33483`)*

La parité exhaustive donne 1 456/1 456 requêtes, 145 236 candidats, plafond
maximal 100, mêmes identités/rangs et 45 features `float32` bit à bit. Les
100/100 requêtes `insee_cp` reproduisent exactement les preuves directes et
leurs agrégats ; le cas vide aboutit à `REVIEW / NO_CANDIDATE` sans appeler
le ranker. Deux requêtes réelles concurrentes restent bit-exactes. Deux
contre-audits techniques du commit exact `bc33483` rendent GO, avec
151/151 et 137/137 tests ciblés. La suite globale donne 1 554 tests passés,
62 ignorés et 7 échecs préexistants exclusivement liés aux artefacts réels
Fresh-S1 déjà présents sur le Mac ; aucun ne touche ce service.

Cette étape historique est désormais close par le worker persistant et le
gate apparié documentés ci-dessus. Le test final produit reste fermé ; tout
réentraînement exige d'abord des labels difficiles fiables et un protocole
OOF séparé.

### Direction active V4.13 — labels réellement frais, sans nouvelle PKI

Le provisioner S1/V1 a échoué proprement sur le Mac réel avec
`errSecMissingEntitlement`; ses artefacts incomplets restent gelés et cette
piste n'est plus le chemin critique de la North Star. L'analyse du dépôt et du
SSD confirme que les 23 609 lignes locales sont toutes consommées et
qu'aucune nouvelle collection CRM avec mapping SIRET indépendant n'est
présente.

Le préenregistrement minimal V4.13 a été réécrit après deux audits `NO_GO`.
Il ferme désormais la sélection déterministe manifest-only de la première
collection, sépare Gate 0A et ouverture payload 0B, autorise seulement la
lecture HMAC historique nécessaire aux quatre keysets anti-chevauchement,
définit les schémas CRM/preuves/queries/oracle, la fonction de split exacte,
les folds OOF, le volume AUTO minimal, le protocole test one-shot et le verdict
terminal `GO|PIVOT|STOP`. Onze tests dédiés passent. Aucun payload frais,
retrieval ou modèle n'a été ouvert ou exécuté. Le commit doit maintenant
obtenir deux audits indépendants `GO_V413_PREREGISTRATION` avant toute
implémentation.
*(commit GitHub : préenregistrement exécutable corrigé `574746f`)*

Le ré-audit de `574746f` a validé les gates, splits, retrieval, OOF et
one-shot, mais a encore refusé l'auto-attestation de provenance, un claim
abusif sur le keyset historique `input_siret_lineage` et un schéma ne
contrôlant que les noms de clés. Le correctif `bf4ed26` remplace ce schéma par
un contrôle du contenu canonique exact, teste six familles de mutations,
préenregistre une allowlist d'autorités réelles volontairement vide, réduit
le claim anti-overlap aux trois projections réellement applicables avec STOP
sur toute ligne non comparable, et rend obligatoire un préenregistrement
modèle séparé avant la première lecture retrieval dev. Quinze tests dédiés
passent. La conséquence est explicite : l'implémentation synthétique peut être
préparée après double GO, mais aucune collection réelle ni aucun
`MATCH_EXACT` n'est autorisé tant qu'un issuer/système de référence vérifiable
n'est pas pinné avant dépôt.
*(commit GitHub : fermeture provenance, schéma exact et gate modèle
`bf4ed26`)*

Deux ré-audits indépendants du commit exact `bf4ed26` rendent maintenant
**`GO_V413_PREREGISTRATION`**. Le verrou de préenregistrement pinne le commit,
le plan, son schéma exact-content, le contrat, le catalogue d'autorités vide,
les tests, l'UID et le volume SSD. Dix-sept tests plan+lock passent. Le seul
travail désormais autorisé est l'implémentation et l'audit sur fixtures
synthétiques ; Gate 0A, Gate 0B, retrieval dev et modèles réels restent
interdits.
*(commit GitHub : lock double-GO V4.13 `19986f0`)*

La première tranche d'implémentation V4.13 est terminée sur fixtures
synthétiques uniquement. L'auditeur Gate 0A valide le lock, observe seulement
les manifests, prouve par tests instrumentés qu'il n'ouvre aucun payload,
sélectionne déterministement et écrit ledger/claim `O_EXCL`. Le builder
qualifie toutes les lignes, sépare queries/oracle, conserve un pont privé des
SIREN pour le split et refuse les autorités non test. Le validateur détecte
les fuites Unicode de 9/14 chiffres et les incohérences oracle. Le sealer
union-find applique exactement la fonction de split gelée et écrit trois
manifests séparés. Un test intégré relie qualification, validation et split.
La suite V4.13 donne 73 tests verts ; aucune inbox réelle, SSD de production,
Keychain, retrieval ou modèle n'a été ouvert.
*(commits GitHub : split `3f862f1`, validation `2b27df2`, disponibilité
`cb6d7ff`, qualification intégrée `6936174`)*

Les deux premiers audits de cette tranche ont rendu `NO_GO` : Gate 0B
n'était pas relié à la claim 0A, les API synthétiques n'étaient pas toutes
confinées à `/tmp`, l'inventaire de preuves pouvait devenir incohérent et le
writer ne rescannait pas une mutation post-qualification. Le correctif ajoute
un opener 0B one-shot : marqueur durable avant le premier FD payload, lecture
same-FD avec identité/taille/SHA, parsing CSV exact, liaison
claim/ledger/collection, receipt terminal et interdiction de reprise après
crash sans receipt. Les claims sont maintenant entièrement liés, les writers
rescannent, les trois manifests de split sont promus depuis un staging, et
toutes les racines synthétiques sont confinées à l'OS tmp. La suite V4.13
donne 86 tests verts, dont availability → Gate 0B → qualification et matrice
de crash. Aucun run réel n'est autorisé.
*(commit GitHub : fermeture Gate 0B synthétique `8cb5d51`)*

Le ré-audit de `8cb5d51` a encore rendu `NO_GO` : les trois racines pouvaient
se chevaucher, une receipt masquait une claim/ledger corrompue après succès et
le test Gate 0B ne relisait pas les artefacts écrits jusqu'au split. Le
correctif `9353294` impose des racines deux à deux disjointes, revalide toute
la chaîne claim/ledger/marker/receipt à chaque reprise, relit et valide les
CSV réellement écrits, exécute les trois projections anti-chevauchement
applicables plus le registre SIREN, puis scelle fit/dev/test avant la receipt.
Il refuse aussi une sortie imbriquée dans la collection et tout overlap
synthétique injecté. La suite V4.13 donne désormais 96 tests verts.
*(commit GitHub : chaîne Gate 0B jusqu'aux splits scellés `9353294`)*

Un audit sur deux a rendu GO sur `9353294`; le second a trouvé que les hooks
de crash tardifs permettaient encore une mutation entre validation et
receipt. Ces hooks sont supprimés, les mêmes FDs payload restent ouverts
jusqu'au contrôle final d'identité et queries/oracle, counts, contamination
et trois manifests split sont reparsés puis comparés sémantiquement juste
avant la receipt. Des mutations tardives de chacune de ces familles sont
désormais testées. La suite V4.13 donne 99 tests verts.
*(commit GitHub : fermeture fenêtre pré-receipt `7b6f587`)*

Les audits de `7b6f587` ont ensuite reproduit une dernière fenêtre entre la
validation sémantique et les hashes de receipt, ainsi qu'une validation trop
faible de l'audit et du manifeste de collection. Le correctif lit, valide et
hashe désormais les huit artefacts depuis les mêmes FDs retenus, compare
l'objet d'audit exact, relit le manifeste source par hash et contenu, puis
recontrôle l'identité de tous les FDs juste avant la receipt. Un premier
ré-audit a rendu GO ; le second a encore refusé une dérive de permissions
post-receipt. La reprise idempotente revalide donc maintenant l'arborescence
exacte, les propriétaires, modes `0700/0600`, types, liens et hashes. Les
mutations de permissions, entrées supplémentaires et anciennes attaques
TOCTOU sont refusées. Deux audits indépendants rendent finalement
**`GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION`** sur le commit exact `280043d`;
la suite auditée donne 105 tests verts.
*(commits GitHub : fermeture validation/hash same-FD `c98fa4f`, fermeture
arborescence post-receipt et double GO `280043d`)*

Le verrou de milestone pinne le commit audité, les hashes du contrat, du
préenregistrement, des six modules et des neuf tests, les deux verdicts et le
runtime Mac. Son statut est volontairement limité à
`GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION_ONLY`. Il interdit explicitement
Gate 0A/0B réels, Keychain/registre historique réel, retrieval, entraînement
et test final. Le test du verrou porte la suite V4.13 à 106 tests verts.
L'allowlist d'autorités réelles reste vide : aucune ouverture de données
réelles n'est autorisée par ce milestone. Un premier contre-audit a confirmé
le contenu du verrou mais refusé un test susceptible d'auto-attestation. Le
test impose maintenant littéralement le commit `280043d`, l'ensemble exact
des 17 blobs, les deux verdicts, le scope fermé, le runtime et désactive les
objets de remplacement Git. Deux contre-audits indépendants rendent
**`GO_V413_SYNTHETIC_INTAKE_LOCK`** sur `c09e6c8`.
*(commits GitHub : verrou intake synthétique V4.13 `ca0a754`, fermeture
anti-auto-attestation et double GO du verrou `c09e6c8`)*

Le pivot V4.12 vers un holdout CRM réellement frais est désormais
préenregistré et contre-audité **`GO_CONTRACTS_FINAL`** sans ouverture d'un
nouveau CRM. Trois frontières séparées sont gelées :

Le successeur exécutable S1 est maintenant préenregistré et deux audits
indépendants rendent **`GO_S1_IMPLEMENTATION`**. L’admission est manifest-only,
Worker Q voit le CRM sans vérité, Worker E voit les preuves sans nom/adresse
CRM, et le scorer ne voit que les requêtes scellées. Les registres matériels,
schémas exacts, catalogues payload/seal, signatures Ed25519, ledger producteur
séquentiel, anti-relance durable, gates distincts et évaluation retrieval
one-shot sont fermés. La suite complète donne `1152 passed`. Ce GO autorise
uniquement la construction des catalogues et l’implémentation synthétique ; il
n’autorise aucune ouverture CRM réelle.
*(commits GitHub : réparation tests R3 `421ec40`, préenregistrement
`6cbd80a`, fermeture autorités `f7079ed`, ordre producteur `288a1de` ;
rapport : `reports/v9/v4_12_fresh_s1_preregistration_results.md`)*.

L'autorité locale Ed25519 du futur producteur S1 est préenregistrée, sans
accès au CRM et sans création de clé réelle. Le correctif ferme
l'appartenance de l'item Keychain par le SHA-256 exact du claim dans
`kSecAttrGeneric`, lie le claim au lock, à l'autorisation et à un nonce
aléatoire, interdit la synchronisation Keychain et impose
`AfterFirstUnlockThisDeviceOnly`. Un audit hostile a ensuite relevé que cette
accessibilité exige sur macOS le Data Protection Keychain : les dictionnaires
`SecItemAdd` et `SecItemCopyMatching` imposent maintenant
`kSecUseDataProtectionKeychain=true` et sont fermés séparément de la
projection des attributs retournés par l'OS. Les schémas imbriqués sont
désormais fermés, le commit de certification S1 est corrigé et l'autorisation précède
les audits de provisionnement. Les 7 tests ciblés et la suite complète
(`1159 passed`) sont verts. Deux audits indépendants rendent désormais
**`GO_S1_LOCAL_PRODUCER_IMPLEMENTATION`** sur le commit exact `c64c0c9`.
Ce verdict autorise uniquement le code et les tests synthétiques du
provisioner ; aucun item Keychain, root S1 ou CRM réel n'a été ouvert.
*(commits GitHub : préenregistrement initial `aacc76a`, fermeture ownership
et gates `28ef796`, Data Protection Keychain `c64c0c9`)*.

Le cœur synthétique du provisioner local S1 est désormais implémenté. Il
produit le claim durable, la clé/signature Ed25519 via un backend injecté, le
genesis, le payload, le seal et le receipt ; il reprend le même attempt après
crash, refuse tout item étranger ou artefact divergent et retourne un receipt
valide sans relire le secret. Vingt tests dédiés couvrent notamment le vecteur
RFC 8032, les contrats Data Protection Keychain, cinq frontières de crash,
l'idempotence et la corruption ; la suite complète donne `1179 passed`. Le
backend macOS réel reste volontairement fermé par
`NATIVE_KEYCHAIN_NOT_PINNED` jusqu'à son implémentation et son audit : aucun
accès Keychain réel, CRM ou `/Volumes` n'a eu lieu.
*(commit GitHub : cœur et tests synthétiques `97f7d0d`)*.

Le premier contre-audit du cœur synthétique rend **`NO_GO`** avant backend
natif. Il valide le one-shot, la cryptographie, la reprise et la gestion des
secrets, mais exige trois fermetures : validation récursive de tous les types
et pins du lock, reconstruction publique exacte du payload depuis plan+lock,
et store entièrement FD-ancré appliquant les pins root/device/volume. La
couche native et tout run restent fermés jusqu'au correctif et à deux nouveaux
audits. Le correctif est maintenant figé : validateur récursif des 13 schémas,
contrôles exhaustifs plan/contrat/implémentation/runtime/Keychain/device/UUID,
reconstruction publique du payload et du genesis, et store exclusivement
`openat`/`O_NOFOLLOW` avec identité avant/après. Les tests dédiés passent à 31,
dont mutations de chaque famille de pins, payload auto-cohérent falsifié,
symlink, hardlink et permissions ; la suite complète donne `1190 passed`.
Il reste soumis aux deux ré-audits avant toute couche native.
*(commit GitHub : fermeture des frontières de confiance `83efe3b`)*.

Le second ré-audit confirme ces trois corrections mais maintient `NO_GO` sur
deux derniers écarts : absence de preuve de concurrence et temps logique du
claim seulement typé, pas égal au plan. Le claim compare maintenant cette
valeur exacte ; un test concurrent synchronise deux launchers et prouve un
seul claim, item, arbre d'autorité et receipt. Les frontières de crash
`CLAIM_DURABLE`, `KEYCHAIN_QUERIED` et `SEED_GENERATED`, ainsi que les états
terminaux du receipt, sont aussi couvertes. Les tests dédiés passent à 38 et
la suite locale donne `1197 passed`; la reproduction d'audit collecte les
mêmes 1197 tests avec `1135 passed, 62 skipped` selon ses capacités
d'environnement. Deux audits indépendants rendent désormais
**`GO_SYNTHETIC_CORE_NEXT_NATIVE_BACKEND`**. Ce GO autorise uniquement
l'implémentation et les tests mockés du backend Data Protection ; un test
multiprocessus reste obligatoire avant tout run réel.
*(commit GitHub : concurrence, temps logique et crashs `3b38fe0`)*.

Le backend macOS Data Protection est maintenant implémenté en processus via
Security.framework et CoreFoundation, sans commande `security`, UI, argument,
environnement ou fichier temporaire. Il construit séparément les
dictionnaires exacts `SecItemCopyMatching` et `SecItemAdd`, vérifie uniquement
la projection persistée autorisée, copie la graine dans un buffer mutable et
libère tous les objets CF. Les 43 tests du provisioner utilisent des APIs
factices ; `main` prouve qu'il s'arrête sur l'absence du lock avant même de
construire le backend natif. La suite locale donne `1202 passed`. Aucun appel
Keychain réel ni création sous `/Volumes` n'a eu lieu ; le commit reste soumis
à deux audits et à un pré-vol lecture seule avant toute autorisation.
*(commit GitHub : backend Data Protection natif `7d40c85`)*.

Le premier audit du backend rend `NO_GO` uniquement parce que les tests
simulaient l'API après le pont ctypes. Deux faux frameworks CF/Security
exercent maintenant le pont complet : ABI, résolution fermée des constantes,
neuf clés exactes de chaque dictionnaire, conversions booléen/chaîne/data/
symbole, projection du CFDictionary et libérations sur succès, not-found,
erreur, champ manquant et add en erreur. La graine intermédiaire est mutable
et remise à zéro sur les sorties d'erreur et après add. Les 50 tests du
provisioner et la suite locale `1209 passed` sont verts, sans appel Keychain
réel. Nouveau double audit requis avant lock.
*(commit GitHub : tests complets du pont CoreFoundation `2ed1ba8`)*.

Deux audits indépendants rendent désormais **`GO_NATIVE_BACKEND_NEXT_LOCK`**
sur `2ed1ba8`. Ils confirment l'ABI, les dictionnaires exacts, la projection,
les statuts, les libérations CF et la remise à zéro des buffers, sans appel
Keychain réel. Ce GO autorise seulement la construction et l'audit du lock,
du pré-vol lecture seule et du test multiprocessus ; il n'autorise pas encore
la lecture du locator, l'autorisation one-shot ou le provisionnement.

La preuve multiprocessus requise avant lock utilise maintenant deux processus
Python `spawn`, synchronisés avant `O_EXCL`, et un backend atomique partagé.
Elle converge vers exactement un succès, un claim, un add/item simulé, un
arbre d'autorité et un receipt ; trois répétitions indépendantes passent. La
suite locale complète donne `1210 passed` (le warning `fork` observé lors du
premier essai a été supprimé par le passage à `spawn`). Aucun Keychain réel.
*(commit GitHub : convergence multiprocessus `ad74b4e`)*.

Le sealer du lock d'exécution est implémenté mais n'a pas encore été lancé.
Il épingle le commit `ad74b4e` contenant à la fois le provisioner certifié et
la preuve multiprocessus, vérifie les blobs de ce commit, les hashes
plan/contrat/code/tests, le runtime exact, la politique Keychain, l'UID, le
device et l'UUID du volume. L'écriture future est privée, exclusive, durable
et non-clobbering. Ses 4 tests et la suite locale `1214 passed` sont verts.
Le sealer doit maintenant être audité avant toute matérialisation du lock ;
lock, autorisation, claim et root réel restent absents.
*(commit GitHub : sealer du lock producteur `8ca9c39`)*.

Le premier audit du sealer rend `NO_GO` car la preuve commit→blobs vivait
seulement dans pytest. Le sealer exécute maintenant lui-même, avant écriture,
`/usr/bin/git` dans un environnement fermé : commit existant, ancêtre de
HEAD, puis lecture `cat-file blob commit:path` et comparaison byte-for-byte
aux fichiers vivants. Les tests ferment blob modifié, commit absent et commit
non ancêtre. Ses 6 tests et la suite locale `1216 passed` sont verts ; le lock
réel reste absent en attente du nouveau double audit.
*(commit GitHub : provenance Git dans le sealer `7249239`)*.

Le ré-audit détecte encore la possibilité Git locale `refs/replace`. Tous les
appels `cat-file` et `merge-base` imposent désormais
`GIT_NO_REPLACE_OBJECTS=1`, en plus des configurations globale et système
neutralisées. Un test capture chaque environnement subprocess et prouve cette
valeur ; les 7 tests du sealer passent. Nouveau verdict requis, lock toujours
absent.
*(commit GitHub : replacement objects interdits `8c3ac72`)*.

Après deux `GO_SEAL_LOCAL_PRODUCER_LOCK`, le sealer a été exécuté une fois.
Le lock canonique de 15 champs a pour SHA-256
`78665f07bdcee12cfdd3989c7e7c55dd3ac625571181b1b2b6a52ea98f54954b`,
mode matériel `0600`, UID 501. Il épingle le commit d'implémentation
`ad74b4e`, le runtime Python 3.14.3/cryptography 46.0.3 et le volume externe
UUID `76ff6087-fe11-4be1-8bb0-c89638a64de8`. L'autorisation, le root, le claim
et toute clé réelle restent absents. Le lock doit maintenant être audité
matériellement avant pré-vol et autorisation.
*(commit GitHub : lock d'exécution scellé `d21d028`)*.

Deux audits matériels indépendants rendent
**`GO_LOCK_MATERIAL_NEXT_PREFLIGHT`** : lock 2 375 octets, hash exact,
canonicalité, schémas imbriqués, blobs Git, runtime, politique
Data Protection, UID, device et UUID concordent. L'autorisation, le root et le
claim restent absents. Le prochain geste autorisé est uniquement le pré-vol
du locator Keychain en lecture seule, avec receipt non secret ; aucun add ni
provisionnement.

Le pré-vol Keychain est désormais préenregistré, mais **n'a pas été
exécuté**. Sa requête fermée contient exactement les sept clés de localisation
du generic password, sans `kSecReturnData`, `kSecReturnAttributes` ni buffer
de sortie. Son seul succès possible est `errSecItemNotFound` (`-25300`) et son
receipt canonique ne contient que des champs non secrets. Le plan est
cross-pinné au lock matériel et au contrat. La
suite d'état post-lock donne `1217 passed, 3 deselected` : les trois exclusions
sont les assertions historiques de phase pré-scellage qui exigent
explicitement l'absence du lock désormais committé. Les deux premiers audits
ont rendu `NO_GO` sur `a833e33` : hash de requête ambigu, résultat insuffisamment
lié au code, gardes d'absence et cycle de crash ouverts. Le correctif
`f07c84e` ferme la canonicalisation et les types/constantes/égalités, impose un
lock de pré-vol liant commit et blobs, vérifie l'absence de l'autorisation, du
root et du claim producteur, et réserve l'appel par un claim durable : tout
état indéterminé interdit un second appel. Ses 6 tests ciblés passent. Deux
nouveaux audits ont donné un premier `GO` et un second `NO_GO` : la notion
`ABSENT` acceptait encore implicitement les liens pendants et le runtime du
futur lock n'était pas fermé champ par champ. Le correctif `2049251` définit
désormais l'absence par un parcours `lstat` sans symlink conclu uniquement
par `ENOENT`, ferme les 15 champs du lock, les 9 champs d'implémentation et
les 9 champs runtime, et préenregistre la matrice fichiers/répertoires/liens/
erreurs/crashs avec zéro appel natif sur chaque état invalide. Ses 8 tests
ciblés passent. Le ré-audit suivant a encore rendu un `GO` et un `NO_GO`,
cette fois sur l'exhaustivité réelle de la matrice : chaque champ n'exerçait
pas séparément type, valeur et absence, et le replay claim+reçu valide n'avait
pas sa propre attente zéro appel. Le correctif `7c33a9b` exécute désormais
chaque mutation sur chaque champ racine/implémentation/runtime, ferme chaque
cas de garde et de cycle de vie par un compteur natif nul, et remplace le
parcours `lstat` par un parcours FD-ancré `openat`/`fstatat` sans symlink pour
fermer le remplacement concurrent des parents. Les 8 tests ciblés passent.
Deux audits indépendants rendent désormais
**`GO_PREFLIGHT_PREREG_NEXT_IMPLEMENTATION`** sur `7c33a9b`. Ce GO autorise
uniquement l'implémentation du pré-vol et de son sealer avec frameworks
fictifs, puis leur audit. Il n'autorise ni lock de pré-vol matériel, ni appel
Keychain réel.

Le pré-vol et son sealer sont maintenant implémentés sur synthétique, sans
exécution réelle. Le pont CoreFoundation construit les sept paires exactes et
appelle `SecItemCopyMatching` avec un pointeur résultat nul ; aucune API
d'ajout, suppression, mise à jour ou retour de secret n'existe dans le
programme. Le runtime est comparé au lock producteur, Git est invoqué par
`/usr/bin/git` dans un environnement fermé avec `GIT_NO_REPLACE_OBJECTS=1`,
les gardes et écritures sont FD-ancrées, et les mutations exhaustives
top/implémentation/runtime prouvent zéro appel natif avant les gates. Les 156
tests ciblés et la suite d'état `1370 passed, 3 deselected` sont verts. Le
commit doit recevoir deux audits
`GO_PREFLIGHT_IMPLEMENTATION_NEXT_LOCK` avant toute matérialisation du lock ;
lock, claim, résultat, autorisation, root et item Keychain restent absents.
*(commit GitHub : implémentation status-only `b71747e`)*.

Les deux audits d'implémentation ont rendu `NO_GO` sur `b71747e`. Ils ont
reproduit une égalité runtime tautologique dans le validateur du sealer et une
vraie course de namespace : renommer puis remplacer le parent pendant
`fstatat` permettait de conclure absent sur l'ancien FD alors que le chemin
canonique contenait l'autorisation. Le correctif `a8e7aac` injecte le lock
autoritatif séparément dans chaque égalité runtime, conserve toute la chaîne
FD depuis `/`, la rouvre et compare chaque `(dev, ino)` avant et après l'appel,
et garde aussi le parent claim/reçu ancré pendant tout le run. Les tests
effectuent maintenant de vrais renommages concurrents, toutes les mutations
claim/guards/lifecycle et les 102 mutations du lock ; Git sélectionne le
dernier commit ayant effectivement modifié les quatre blobs plutôt que le
simple `HEAD`. Les 228 tests ciblés et la suite `1442 passed, 3 deselected`
sont verts. Deux nouveaux audits sont requis avant lock réel.
*(commit GitHub : runtime autoritatif et courses namespace `a8e7aac`)*.

Les ré-audits de `a8e7aac` ont encore rendu `NO_GO` sur une fenêtre plus
étroite après le dernier `stat`, et sur les erreurs `fstat` qui pouvaient
s'échapper sans normalisation. Le correctif `a49d28b` revalide toute la chaîne
de namespace après l'observation `ENOENT`, double la réouverture du parent
d'état, convertit chaque erreur `open/stat/fstat` en arrêt fermé et ajoute une
matrice réelle `EACCES/EIO`, au dernier contrôle pré-appel, après claim et
après appel simulé. Les 239 tests ciblés et la suite
`1453 passed, 3 deselected` sont verts. Nouveau double audit requis ; aucun
lock ni appel réel.
*(commit GitHub : revalidation post-statut et erreurs syscall `a49d28b`)*.

Les audits de `a49d28b` ont divergé (`GO` / `NO_GO`) sur une propriété
impossible à garantir en espace utilisateur : un processus du même UID peut
toujours renommer un parent juste après la dernière vérification, quel que
soit le nombre de relectures. L'amendement `442416e` remplace cette prétention
par une borne explicite : 26 cas restent à zéro appel ; dans l'unique fenêtre
finale, au plus une requête status-only à pointeur nul peut avoir lieu, mais
la revalidation post-appel impose `STOP`, interdit le reçu et laisse le claim
seul. Le déplacement volontaire des autorités persistantes par un processus
non coopératif est désormais explicitement hors modèle. L'implémentation
`efe76fe` normalise aussi `fsync` et teste cette borne exacte ainsi que
`open/stat/fstat × EACCES/EIO`. Les 242 tests ciblés passent. L'amendement doit
recevoir deux `GO_PREFLIGHT_RACE_AMENDMENT_NEXT_IMPLEMENTATION_AUDIT`, puis le
code deux `GO_PREFLIGHT_IMPLEMENTATION_NEXT_LOCK`, avant tout lock réel.
*(commits GitHub : amendement de concurrence `442416e`, implémentation de la
borne `efe76fe`)*.

Le premier audit de l'amendement a rendu `GO`, le second `NO_GO` sur deux
ambiguïtés : l'ordre du plan ne plaçait pas explicitement la revalidation
post-appel avant le reçu, et le nom du cas semblait limiter la fenêtre au
parent d'état. Le correctif `c170865` généralise l'unique exception à tous les
namespaces protégés et inscrit dans `entry_order` :
appel → revalidation de toutes les gardes et de l'état → `STOP` éventuel →
seulement alors création possible du reçu. `25aa424` aligne les constantes et
tests du code. Les 242 tests ciblés passent. Nouveau double audit de
l'amendement requis avant le gate d'implémentation.
*(commits GitHub : amendement généralisé `c170865`, alignement code
`25aa424`)*.

Deux audits indépendants rendent désormais
**`GO_PREFLIGHT_RACE_AMENDMENT_NEXT_IMPLEMENTATION_AUDIT`** sur `c170865`.
Ils confirment les 27 cas, la borne unique, l'interdiction du reçu et l'ordre
post-appel fermé. Le prochain geste autorisé est uniquement l'audit technique
de `25aa424`; le lock réel reste interdit.
Autorisation, root, claim, item Keychain et receipt réel restent absents.
*(commits GitHub : préenregistrement initial `a833e33`, fermeture des
autorités et crashs `f07c84e`, fermeture absence/runtime `2049251`,
matrice exhaustive et parcours FD `7c33a9b`)*.

L'audit technique de `25aa424` a ensuite réfuté la garantie générique : un
processus non coopératif du même UID peut modifier une garde après sa dernière
revalidation et avant l'écriture du reçu. L'amendement `db68537` sépare donc
les deux fenêtres irréductibles. Un remplacement du parent d'état reste borné
à un appel et interdit tout reçu ; un remplacement d'une garde après son
dernier contrôle reste borné à un appel mais peut laisser un reçu purement
observationnel, qui n'autorise jamais le provisionnement. `cf99d95` ferme le
validateur sur 28 cas exacts — 26 à zéro appel, deux à un — et reproduit les
deux courses avec un backend synthétique. Deux audits indépendants rendent
**`GO_PREFLIGHT_RACE_AMENDMENT_NEXT_IMPLEMENTATION_AUDIT`** sur `db68537` et
**`GO_PREFLIGHT_IMPLEMENTATION_NEXT_LOCK`** sur `cf99d95`. Les 243 tests
ciblés et la suite `1457 passed, 3 deselected` sont verts ; les trois
exclusions sont les assertions historiques exigeant l'absence du lock
d'autorité déjà committé. Aucun appel Keychain réel, lock de pré-vol, claim,
reçu, autorisation ou root producteur n'a été créé. Le seul geste désormais
autorisé est de fabriquer puis faire auditer le lock de pré-vol ; le run
matériel reste interdit.
*(commits GitHub : séparation contractuelle `db68537`, alignement code/tests
`cf99d95`)*.

Le lock de pré-vol a été scellé une seule fois par `71920a7`. Son SHA-256 est
`af6d6685b5379ab0bddb7ef1cc30feb52bfe4982ffbd86977ba1b3ca5447a1ac` ;
il ferme 15 champs racine, 9 champs d'implémentation et 9 champs runtime,
épingle `cf99d95` et les quatre blobs code/tests, le plan `b580157b…`, le
contrat `cf48e739…`, l'autorité `78665f07…`, la requête et l'UID 501. Deux
audits indépendants, dont 12 mutations adversariales en mémoire, rendent
**`GO_PREFLIGHT_LOCK_MATERIAL_NEXT_RUN`**. Ils confirment le commit
mono-fichier, les hashes Git/worktree, le runtime local exact et l'absence du
claim, du reçu, de l'autorisation et de la racine producteur. Le prochain
geste autorisé est désormais l'unique requête Keychain status-only fermée par
ce lock ; elle ne constitue toujours pas une autorisation de provisionnement.
*(commit GitHub : lock matériel de pré-vol `71920a7`)*.

L'unique pré-vol réel a ensuite retourné `-25300 /
KEYCHAIN_LOCATOR_ABSENT`. `58798a4` fige le claim one-shot
`c6446c0a…` et le reçu non secret `a4e5dc25…`, tous deux canoniques et
matériellement `0600`. Deux audits indépendants rendent
**`GO_PREFLIGHT_RESULT_MATERIAL_NEXT_AUTHORIZATION`** après reconstruction
exacte, dix mutations adversariales et replay synthétique à zéro appel natif.
L'autorisation canonique à six champs a alors été committée seule par
`651ef43` ; son SHA-256 est `b8796626…`. Deux contre-audits complets rendent
**`GO_S1_LOCAL_PRODUCER_PROVISION`** : implémentation `ad74b4e`, lock
producteur `78665f07…`, plan/contrat, runtime, UID, volume, pré-vol et
autorisation sont cohérents. La suite active adaptée donne `1453 passed,
7 deselected` ; les sept exclusions sont uniquement des assertions
historiques exigeant l'absence des artefacts de phases désormais achevées,
sans retrait des tests de secret, crash, concurrence ou idempotence. Le root
et le claim producteur restent absents. Le prochain geste autorisé est
l'unique run réel du provisioner, toujours sans lecture CRM.
*(commits GitHub : résultat one-shot `58798a4`, autorisation producteur
`651ef43`)*.

Le run producteur V1 autorisé s'est arrêté proprement sur
`KEYCHAIN_ADD_STATUS_-34018` (`errSecMissingEntitlement`). Le root et le claim
V1 existent en `0700/0600`, mais `authorities/` est vide et aucun receipt,
payload, seal ou genesis n'a été créé. Deux diagnostics indépendants
confirment que le Python pinné est signé ad hoc sans Team ID ni entitlement :
le Data Protection Keychain imposé par V1 exige un profil de provisioning
appliqué au processus hôte. V1 reste immuable et ne sera pas relancée.
Verdict : **`PIVOT_FILE_BASED_KEYCHAIN_V2`**. V2 aura un locator, un root et
toutes ses autorités distincts, utilisera `SecItem` sur le Keychain macOS
traditionnel avec ACL fermée, et n'autorisera aucun fallback de graine en
fichier. Ce pivot reste local, sans GPU, location ni dépense externe.
*(rapport : `reports/v9/v4_12_fresh_s1_local_producer_v1_failure.md`)*.

L'audit d'alignement North Star montre ensuite que V2 Keychain serait encore
un détour : aucun exporteur, catalogue ou worker S1 métier ne consomme
l'autorité, sept composants S1 restent `UNIMPLEMENTED`, et aucune collection
CRM fraîche n'existe dans l'inbox. Le contrat minimal V4.13 retire donc la
PKI locale du chemin scientifique sous le threat model « opérateur local
coopératif ». Il conserve frame exhaustive, preuve SIRET indépendante,
registres anti-chevauchement, séparation physique queries/oracle, splits
SIREN-disjoints, plafond 100 et test one-shot. Son gate zéro exige avant tout
nouveau code ML au moins 657 `MATCH_EXACT`, 80 % de couverture et zéro
chevauchement ; sinon `WAITING_FOR_NEW_SOURCE` ou
`PIVOT_SOURCE_EVIDENCE`. Les six tests du plan passent. Le ranker, le
decider, le risk model et l'accepteur restent gelés. Deux audits de
préenregistrement sont requis avant toute implémentation ou ouverture source.
*(contrat : `docs/v4_13_fresh_labels_minimal_contract.md`; plan :
`config/v4_13_fresh_labels_minimal_plan.json`)*.

- le registre de compatibilité ferme les 23 609 anciennes lignes avec des
  empreintes SIRET-masked/fuzzy et des clés de lignée HMAC privées
  *(commit GitHub : `96be59e`)* ;
- le registre `consumed_sirens` ferme uniquement les identités autoritatives
  déjà consommées, en excluant candidats, prédictions et sondes techniques
  *(commit GitHub : `0b47b4c`)* ;
- l'intake impose une frame exhaustive sans arrêt opportuniste, une couverture
  `MATCH_EXACT / toutes lignes source` >= 80 %, au moins 657 exacts, des
  preuves oracle-side séparées, un scoring retrieval one-shot à 100 candidats
  maximum et une certification AUTO 99,8 % distincte
  *(commit GitHub : `9c7eccd`)*.

Les deux registres préalables sont désormais réellement construits, scellés
et contre-audités **`GO_V412_CONTAMINATION_REGISTRIES`** :

- `consumed_sirens` ferme 64 618 observations et 19 754 SIREN uniques, sans
  candidat, prédiction ou sonde technique et avec zéro rejet. Build
  `fbc0b84d9c81b01a`, manifest
  `b220efd7c4dc89a980b9d0b5501e16fd286edcafdff61573ae6c5e8d8423c6ff`
  *(commits GitHub : code/tests `3b66fd7`, contrat/plan `9f74c00`,
  cross-pin intake `a20c704`)* ;
- `consumed_compatibility` ferme les 23 609 anciennes lignes, dont les 225
  cas du challenge, par des keysets privés HMAC/masked/fuzzy. Build
  `48851668dd2f173686f3240ecc62e30fcbfdb96d8abf0ced498eb29891d8a490`,
  seal `2068a5d18aac189b7bffc0515054fa31166cb5cd9e4d066f143d3c2d5bc3e976`,
  zéro rejet. La clé reste dans le Keychain et est lue en processus sans UI,
  argument, environnement, fichier temporaire ou log
  *(commits GitHub : identité volume `6de4585`, Keychain `4a5ac60`,
  contrat/plan `38b18d8`, cross-pin `4b8bd2a`)*.

Le premier lancement du registre de compatibilité s'est arrêté sans payload :
le CSV réel porte un BOM UTF-8 non déclaré. L'attempt
`v412-compat-8c4f31ce-attempt-01` reste immuable avec son seul receipt et
`ATTEMPT_RECEIPTED`. Le correctif épingle exactement le BOM initial, conserve
un éventuel `U+FEFF` dans une valeur CRM sale et prouve la parité réelle des
23 609 lignes. Le second attempt a été publié après reproduction
byte-for-byte et deux audits indépendants
  *(commits GitHub : ancien lock révoqué `213a3b0`, code BOM `47e9772`,
  contrat/plan `6f9ad7e`, cross-pin `63e45f1`, lock final `5516ba6`)*.

Le scanner/sealer d'arrivée S0 est maintenant préenregistré et deux audits
indépendants rendent **`GO pour commencer le code S0`**. Sa fixture de six
lignes est entièrement déterministe, l'identifiant de run n'est plus
circulaire, les trois types d'arbres scellés et le journal de reprise sont
fermés, et les tests négatifs couvrent stabilité, structure CSV, quarantaine,
conflits et crashs. Ce GO autorise uniquement l'implémentation sur synthétique :
aucun CRM réel ni run autoritaire ne peut être ouvert avant le sandbox, le
launcher, le verrou et le control manifest pinnés.
*(commit GitHub : `50333d3`)*.

Le cœur S0 est désormais implémenté et contre-audité
**`GO_CORE_PRELOCK`**. Le producteur déterministe et le scanner test-only
ferment la stabilité sur FDs, les arbres et receipts scellés, les
quarantaines, le journal et sa reprise, les bindings de provenance et les
sorties Parquet. La matrice défensive compte 62/62 tests verts sur le SSD ;
elle couvre aussi les métadonnées applicatives, liens, ancêtres de chemins,
reçus partiels ou concurrents et dates impossibles. Toute invocation hors du
répertoire pytest dédié reste refusée : le prochain geste est exclusivement
la matérialisation puis l'audit du sandbox, du launcher et du verrou.
*(commit GitHub : `38287c1`)*.

Le contrat autoritatif du lancement S0 franchit désormais
**`GO_IMPLEMENTATION`** après trois cycles de contre-audit. Il ferme un worker
FD-only distinct du core, le runtime Python/PyArrow privé, la sandbox
`deny default`, les autorités parent/worker disjointes, le protocole de
contrôle, l'automate lease puis claim anti-rejeu, les canaris synthétiques et
la cohérence complète résultat/exit/receipt. Le plan canonique amendé a pour
hash
`f73d855b9d6c76f6175cae5e04f2bd2bc61a19a5d78d356ebe99d3d6289f8596`
et épingle le contrat
`b969a8d552ba060e5b7e24bd1e295abbaf025f1dfbfb7e5683bd5853b689b5df`.
L'amendement `GO_AMENDMENT` interdit de fabriquer des preuves canaris lors
d'un STOP précoce : succès = liste complète, STOP avant preuve = liste vide.
Ce GO autorise seulement l'implémentation du launcher, du worker, du sealer et
du profil ; aucun run, fixture nouvelle ou CRM réel n'a été ouvert.
*(commits GitHub : contrat initial `46b1958`, amendement `7a3353f`)*.

Le bundle autoritatif S0 est implémenté et deux audits indépendants rendent
**`GO_CODE_BUNDLE`**, sans fixture autoritative ni run. Le sealer construit un
runtime privé de 1 528 fichiers et un lock sans suivre les liens ; le launcher
sans argument ferme lease, claim, reprise, receipts, TOCTOU, canaris et arbres
de sortie ; le worker n'accepte que les FDs, attend réellement 60 secondes et
réutilise le core immuable. Les 109 tests S0 passent. La suite complète donne
1 071 succès et le seul échec historique connu, causé par un test qui interdit
tout `/Volumes/CATNAT_DATA` alors que le `TMPDIR` obligatoire y réside. Ce GO
autorise uniquement la construction de la fixture puis du lock, suivie de leur
audit avant autorisation et lancement.
*(commit GitHub : `42d9027`)*.

Le premier lock autoritatif S0
`feeef92c7df4c24473d3850f0b074aa5e5f904ac79c507f674606d4b6057a598`
a été révoqué **avant autorisation et avant lancement** : son audit matériel
était intégralement vert (1 722 contrôles), mais un contre-audit de cohérence a
détecté que le launcher exigeait à tort une identité de fichier non nullable
pour le canari `EXISTING_DIRECTORY`, alors que le sealer et le schéma
autoritatif imposent trois valeurs nulles. Le launcher valide maintenant ce
répertoire par ouverture ancrée, sans lien symbolique, puis contrôle son
propriétaire, son volume et ses permissions. Deux audits indépendants rendent
`GO_PATCH`; les 110 tests S0 passent et le vrai manifeste de canaris est
accepté. L'ancien lock ne doit jamais être autorisé : il doit être archivé de
façon récupérable, puis la fixture et un nouveau lock doivent être reconstruits
sur le commit corrigé.
*(commit GitHub : `61a52c5`)*.

Le deuxième lock S0
`f918b8af6c9dc47bc61bcb6ab36d0808704206a28865f8fa32b629b1a32d59e2`
avait franchi deux audits statiques, puis le pré-vol exécutable a détecté un
second défaut avant tout worker : le helper de lecture imposait l'UID
utilisateur aux deux autorités macOS légitimement détenues par `root`
(`SystemVersion.plist` et `/usr/bin/sandbox-exec`). L'autorisation initiale
`0bcdb7a`, non canonique, puis sa correction `10b907e` sont toutes deux
révoquées avec ce lock et ne doivent jamais servir à un lancement. Le helper
accepte désormais un propriétaire explicite, limité à `uid=0` pour ces deux
fichiers système ; toutes les autorités privées restent obligatoirement
détenues par l'utilisateur. Deux audits rendent `GO_PATCH` et `GO_PATCH_2`,
les 111 tests S0 passent et le pré-vol runtime réel est vert. Le deuxième
environnement doit être archivé sans suppression, puis lock et autorisation
doivent être reconstruits sur le commit corrigé.
*(commit GitHub : `2bb2bc2`)*.

Le troisième lock S0
`d608a0e13334270a16a554f3ca676135b4cec671af3a52d468ec8a8a28a40e50`
a lui aussi été arrêté au pré-vol, sans autorisation mise à jour ni worker. Le
launcher mettait en cache l'UUID de volume par `st_dev`; or, sur ce Mac, le
dépôt (volume Data) et `/` (volume System) partagent le même `st_dev` tout en
ayant des UUID APFS distincts. Selon l'ordre de lecture, la frontière de
confiance système était donc remplacée par celle du dépôt. Le resolver relit
désormais l'UUID directement sur chaque FD avec contrôle d'identité
avant/après, sans cache ambigu. Deux audits indépendants rendent
`GO_APFS_PATCH` et `GO_APFS_PATCH_2`; les 112 tests S0 et les validations
runtime/volumes du lock réel passent. Ce troisième lock reste révoqué car il
n'épingle pas le blob corrigé ; reconstruire encore lock et autorisation avant
tout lancement.
*(commit GitHub : `75edb12`)*.

Le run autoritatif S0-R1 a été exécuté une seule fois et conclut **`PIVOT`**.
Son receipt immuable
`68d1267351447d6dd755cfca62cccec700715191b45a906e28ecc59b40bc6746`
rapporte `WORKER_CONTROL_INVALID`, enfant `exit=65`, aucune frame
`READY`, aucune sortie, aucun canari et aucune stabilité fabriqués ; les
13 autorités parent sont identiques avant/après. La cause est prouvée
byte-for-byte : le stderr de 62 octets, SHA
`1d24b61273dbf35a7162215eaa0aa2668c83773f003884a01e326b8065132cf7`,
est exactement
`sandbox-exec: /dev/fd/effective.sb: No such file or directory\n`.
Le transport verrouillé `sandbox-exec -f /dev/fd/<fd>` échoue donc avant
Python et avant le worker. Le claim et le receipt R1 restent immuables ; aucun
rerun, déplacement ou rebuild sous les mêmes identifiants n'est autorisé.
La suite exige une autorité S0-R2 préenregistrée avec nouvelle racine,
nouveaux `synthetic_run_id` et `attempt_id`, et transmission du profil par
`-p` depuis les octets relus et rehashés du FD retenu.
*(commit GitHub d'autorisation R1 : `37b453f`)*.

Le successeur S0-R2 franchit **`GO_R2_IMPLEMENTATION`** et
**`GO_R2_IMPLEMENTATION_2`**. Il conserve R1 immuable, utilise la racine
distincte `fresh_holdout_intake_synthetic_r2`, dérive un nouveau run
`bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo`,
et impose un nouvel attempt. Le transport du profil devient `sandbox-exec -p`
depuis les octets relus et rehashés du FD parent, sans FD profil transmis. Un
smoke final sans payload doit réussir après construction du runtime et être
attesté dans un lock R2 schema-3 avant toute autorisation. Plan canonique :
`e05102a36b9aaf37ed3aa1052814a9e2bb8ff77a62d26cf135f9ff1f240abd27`;
contrat :
`2933d217f169b67d3eff399c5b270a91590a2ccec82430469a3ab8489a17a937`.
Ce GO autorise seulement l'implémentation R2.
*(commit GitHub : `4cf640e`)*.

Le Gate A R2 a ensuite découvert, avant création de la racine R2, que
`sandbox-exec` supprime les variables `DYLD_*` et rend inexécutable la copie
du stub Homebrew `bin/python3.14`. L'amendement R2-B franchit désormais deux
audits indépendants **`GO_R2B_IMPLEMENTATION`**. Il copie le vrai helper
`Python.app`, conserve stdlib et PyArrow privés, supprime tout `DYLD_*`, et
épingle comme unique exception hôte la bibliothèque framework exacte,
retenue et rehashée avant/après. Le smoke pré-lock doit importer
`encodings` et PyArrow 23.0.1 depuis le runtime privé, sans stdout/stderr.
`otool` reste limité au sealer pré-lock ; le launcher revalide l'install name
Mach-O en processus afin de conserver un seul child autoritatif. Plan
canonique :
`2ab9a1d5954588c01de22c54e21c721aa0e9da9a9e7f140d9f93950cb8b1abf4`;
contrat :
`66418a23ae6b166f253f7ef4bc220e3a47ce0655c2ee96c7e8a9db51e0519a42`.
La sonde homologue hors racine R2 réussit avec `exit=0`, stdout/stderr vides
et aucun accès général à `/opt`. La racine R2 reste absente ; ce GO autorise
uniquement l'implémentation et son Gate B avant toute création R2.
*(commit GitHub : `5fc7116`)*.

Le bundle R2-B franchit maintenant le Gate B **`GO_R2B_CODE_FINAL`**, confirmé
par deux contre-audits. Le runtime homologue construit hors racine R2 contient
1 525 records fermés ; son smoke réel charge `encodings` et PyArrow 23.0.1
depuis le privé avec `exit=0`, stdout/stderr vides, puis le launcher reconstruit
exactement la même attestation. Le Gate B a détecté et corrigé avant commit un
alias Homebrew de stdlib qui dupliquait le framework hôte, puis une capture
stdout/stderr initialement plafonnée après coup : la lecture est maintenant
bornée en continu à 65 536 octets par flux, avec kill/close/wait au
dépassement ou timeout et stdin sur `/dev/null`. La suite complète est verte :
1 090 tests passent sur le `TMPDIR` SSD. La racine R2 reste absente ; le
prochain geste autorisé est l'audit du commit, puis seulement la construction
de la fixture et du lock R2.
*(commit GitHub : `0afb010`)*.

L'unique exécution autoritative S0-R2 a ensuite conclu
**`PIVOT_R2_WORKER_IDENTITY`**. Son receipt canonique immuable, SHA-256
`6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e`,
atteste onze canaris refusés, le même processus et les mêmes cinq FD pendant
`60.005023459` secondes, puis `WORKER_CONTROLLED_STOP`, sans stdout, stderr
ni sortie `sealed`, `scan`, `quarantine` ou `tmp`. Le builder R2 dérive
correctement `bjpoib...` avec le domaine successeur et le receipt R1, mais le
worker recalcule encore l'ancienne identité cœur `komapn...`; il échoue donc
avant toute écriture sur l'invariant d'identité. Le catch global masque cette
cause sous un code générique. R2 est consommé et ne doit jamais être relancé.
Avant tout R3, corriger la dérivation, produire un STOP à phase/code fermés et
faire atteindre `INGESTED` au vrai `_process` dans un gate sandbox jetable.
Autorisation R2 : commit GitHub `5dbb2ff`. Rapport de pivot :
`reports/v9/v4_12_fresh_s0_r2_pivot.md`
*(commit GitHub : `648cd4f`)*.

Le gate jetable du successeur conclut désormais **`GO_PREREG_R3`**, sans
autoriser encore build, lock ou run R3. L'artefact probant
`diag-r3-successor-gate.yww2qf5m` possède un résultat persistant canonique,
SHA-256
`c86ad8bf1a4b8af0525c6870e05ddabb2f27c4208f9f07c8be07edebb52e212b`.
Sous le vrai Python privé et Seatbelt, le worker atteint `INGESTED`, conserve
les mêmes FD pendant `60.003147917` secondes, refuse les onze canaris avec
`EPERM`, garde stdout/stderr vides et publie trois générations de journal.
Une identité R1 complète est rejetée
`IDENTITY/EXECUTION_IDENTITY_SCHEMA_INVALID` sans mutation de sortie. Le
worker consomme désormais l'identité successeur du spec, et
`control-result-2` transporte un diagnostic fermé validé par le launcher.
Les 1 095 tests passent. Les gates antérieurs `zi2oynzm` et `9fy69i1l`
restent non promotables. Le prochain geste est exclusivement la
préinscription du contrat/plan R3 fermant la chaîne lock → spec → worker.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_gate_results.md`
*(commit GitHub : `5d1820d`)*.

Le contrat et le plan S0-R3 franchissent maintenant deux audits indépendants
post-commit **`GO_R3_IMPLEMENTATION`**. Le plan canonique SHA-256
`ce7f8ed4a9d6236e61cffca72b92a1043d414afc69571ae79c94f191e6def1e2`
est lié au contrat
`247b41f60a39211f85431d141625bf0d8321ae88c701d17ffd380a04ef7a9353`.
L'overlay fermé applique 30 overrides et quatre suppressions au plan R2 :
les schémas R3 matérialisent intégralement champs, nullabilité et types,
`SANDBOX_EXEC` reste une autorité système hors des blobs Git, et aucune
identité R2 interdite ne pilote R3. Les 11 tests R3 et les 1 106 tests du
dépôt passent sur le SSD. La racine R3 reste absente. Ce GO autorise seulement
l'implémentation R3 ; ni build, ni autorisation, ni run, ni CRM réel ne sont
encore ouverts.
Préenregistrement : commits GitHub `7bf1ea4`, correctif `a483107`.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_preregistration_audits.md`.

Le bundle S0-R3 franchit désormais deux contre-audits indépendants
**`GO_R3_CODE_BUNDLE`**. Le commit d'implémentation `8d8e0a3` relie le builder
R3, le sealer, le launcher receipt-3 et le worker à l'identité littérale
préenregistrée. Le gate Seatbelt jetable
`diag-r3-successor-gate.sfzj9buk` atteint `INGESTED`, refuse 11/11 canaris,
conserve les mêmes FD pendant `60.010024083` secondes, publie trois
générations de journal et rejette une identité R1 sans mutation. Son résultat
canonique a pour SHA-256
`556558d4372b003d23190b86ff8163e021e0a937b83539b0fcc1e4828b53185b`.
Les deux audits ont rehashé les 1 525 fichiers du runtime, les seals et le
journal. La suite donne 1 078 succès et 62 skips, zéro échec, sur le SSD. La
racine autoritative R3 reste absente ; le prochain geste autorisé est sa
construction unique, puis le lock et son audit, jamais le CRM réel.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_code_gate_results.md`.

L'unique exécution autoritative S0-R3 est désormais doublement certifiée
**`INGESTED_R3_CERTIFIED`**. Le receipt-3 canonique SHA-256
`8061247794f403f52a692e41f19549dcf2803a6db744c74e9719cb824ad96a08`
lie l'autorisation `f686ffd9…`, le lock `de545687…`, le claim `c6a1c5…`,
le spec worker et les sorties. Le worker termine `exit=0`, stdout/stderr
vides, après `60.002720750` secondes avec le même processus et les mêmes cinq
FDs ; 11/11 canaris sont refusés, les 14 observations parent sont identiques,
les arbres sont scellés et le journal compte trois générations. Il existe
exactement un claim, un lease, un spec et un receipt ; le chemin idempotent
interdit désormais tout nouveau spawn. Aucun processus R3 n'est actif.
Autorisation initiale non canonique jamais lancée : commit `ffccb7e` ;
autorisation canonique utilisée : commit GitHub `b64133f`.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_authoritative_results.md`.
Ce succès ouvre uniquement la construction puis la qualification aveugle du
CRM frais ; labels, retrieval, modèles et test final restent fermés.

Rapport complet :
`reports/v9/v4_12_contamination_registries_results.md`
*(commit GitHub : `a0b510a`)*. Le prochain geste autorisé est le
scanner/sealer d'arrivée sur paquets synthétiques uniquement. Aucun futur CRM
ne peut encore être ouvert et le ranker, le decider, le risk model et
l'accepteur restent gelés. Tout test ou build suivant doit utiliser le SSD
externe ; aucun nettoyage destructif n'a été effectué.

Le builder des entrées sûres du moteur unitaire V4.12 est implémenté et
contre-audité, sans build réel. Une première revue a rendu `STOP_CODE` malgré
18 tests verts et a exposé une mauvaise empreinte TF-IDF, un contournement du
plan interne, des sorties Parquet non revérifiées, une signature seulement
recopiée et une fenêtre TOCTOU CRM. Les défauts ont été reproduits puis
corrigés. Deux contre-audits indépendants rendent désormais `GO_CODE` et
`GO_CODE_2`; 27 tests ciblés et les 645 tests du dépôt passent. Les
inventaires réels partitions/cache et la signature historique ont été
recalculés exactement. Ce GO autorise seulement le verrou puis le build des
entrées physiquement aveugles ; il n'autorise ni oracle, ni worker, ni
benchmark. *(commits GitHub : builder `18eb76e`, audit `46868db`)*
Le séquencement du verrou a ensuite été fermé : le commit du verrou peut
suivre le commit audité sans rendre l'exécution impossible, tandis que chaque
source reste identique au worktree, au verrou et au blob du commit audité.
Le contre-audit rend `GO_LOCK_SEQUENCING`; 29 tests ciblés et 647 tests
complets passent. *(commit GitHub : `c97c737`)*
Le verrou d'exécution des entrées sûres est contre-audité `GO_LOCK`. Il
épingle le commit audité, les cinq sources, queries/split, les deux
inventaires complets, le runtime et les trois racines SSD. Son hash avant
commit est `e794c60f...e9def315`; aucun build n'avait encore été lancé lors
du gel. *(commit GitHub : `7c31051`)*
Le build réel des entrées sûres franchit **`GO_V412_UNIT_INPUTS`**, confirmé
indépendamment par `GO_V412_UNIT_INPUTS_AUDIT` et 21 161 assertions sans
import du builder. Les 7 003 requêtes, dont 1 456 dev, ne contiennent que les
six champs CRM autorisés ; les inventaires scellent 4 119 partitions et
1 454 paires TF-IDF. Le ledger séparé couvre exactement 7 029 fichiers.
Aucun label, oracle, modèle, résultat candidat ou chemin sensible n'est
présent dans le paquet runtime. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e`.
Rapport : `reports/v9/v4_12_unit_input_results.md`. Ce GO autorise seulement
le préenregistrement de l'oracle séparé. *(commit GitHub : `e5d01a9`)*
Le contrat de l'oracle dev V4.12 est préenregistré `GO_CONTRACT_ORACLE`.
L'oracle sera truth-only : 1 456 IDs, 1 217 `MATCH_EXACT` et 239
`AMBIGUOUS`, sans candidat, rang, score, preuve ou décision historique. Une
première revue a refusé la simple séparation de dossiers sur le même SSD ;
le contrat corrigé exige que le futur worker tourne sous `sandbox-exec`, avec
les racines oracle/audit interdites et une sentinelle d'ouverture réellement
refusée. L'oracle reste historique, non indépendant et non certifiant.
*(commit GitHub : `1dd7428`)*
Le contrat précise désormais le ledger exhaustif des huit fichiers réellement
ouverts par le builder oracle : six fichiers du paquet runtime sûr plus
labels/split. Les inventaires sont ouverts uniquement pour contrôler
l'intégrité du paquet, jamais comme résultats de retrieval ni pour former la
vérité. *(commit GitHub : `bbf31b9`)*
Le builder d'oracle et ses tests franchissent `GO_CODE_ORACLE` après quatre
refus d'audit : rescellation complète, sibling modifié à taille/mtime
restaurées, ledger incomplet puis ledger réordonné. Les trois PoC finaux sont
désormais bloqués ; 23 tests ciblés et 670 tests complets passent. Aucun
build réel n'avait encore été lancé. *(commits GitHub : builder `7eafad8`,
audit `02e954b`)*
Le verrou d'exécution oracle est contre-audité `GO_LOCK_ORACLE` avec 4 434
assertions : cinq sources Git, quatre inputs, six fichiers runtime sûrs,
populations, ordre, payloads, runtime et racines sont exacts. Hash du verrou :
`4d598cf1...f6d4c8b1`. *(commit GitHub : `04a22db`)*
L'oracle séparé franchit **`GO_V412_UNIT_ORACLE`**, confirmé par
`GO_V412_UNIT_ORACLE_AUDIT` et 4 430 contrôles sans import du builder. Il
contient 1 456 lignes : 1 217 `MATCH_EXACT` et 239 `AMBIGUOUS`, uniquement
issues des labels/split historiques. Le ledger ordonné couvre les huit
fichiers réellement ouverts. Aucun résultat retrieval, score, décision,
modèle, challenge ou test final n'a participé à la vérité. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/oracles/v4_12_unit_engine/c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70`.
Rapport : `reports/v9/v4_12_unit_oracle_results.md`. *(commit GitHub :
`9d2c68e`)*

Le contrat Gate A des stores stricts et de la sandbox V4.12 est
préenregistré `GO_CODE_V412_STRICT_STORES`. Trois audits indépendants ont
fermé le routage, les 648 partitions, les 648 caches et le lookup snapshot,
ainsi que la liste blanche réelle de `sandbox-exec`. Une erreur de conception
a été interceptée avant code : les caches portent sur 4 764 472 rows
filtrées/dédupliquées et non sur les 8 030 285 rows physiques. La sandbox
épingle `sandbox-exec`, Git, `Python.app` et la bibliothèque framework,
refuse oracle/audit, réseau, fork et écritures hors espaces privés, et expose
exactement 1 945 fichiers à l'enfant. Le
ledger parent attendu en couvre 1 954. Ce GO autorise seulement
l'implémentation puis son audit, pas encore le build réel ni les modèles.
Rapport : `reports/v9/v4_12_strict_stores_contract_audit.md`. *(commit
GitHub : `0173d6b`)*

Le contrat Gate A a été durci après dix-sept refus successifs : contrôles
consommés par descripteurs ancrés, profil transmis en mémoire, Git absolu,
runtime Python privé, publication atomique et reprise rattachée aux entrées
courantes. La frontière de confiance est maintenant explicite : le runtime
local `/System`, `/usr` et `/opt/homebrew` est enregistré mais n'est pas
présenté comme un système intégralement scellé. *(commit GitHub : `d23c287`)*
Les trois stores stricts et leur certificateur sandbox sont implémentés et
contre-audités **`GO_CODE_V412_STRICT_STORES`**. Les contrôles couvrent
partitions, caches TF-IDF, lookup DuckDB via FD, refus sandbox, rescellation,
publication/recovery et nettoyage des espaces privés. Le smoke macOS réel,
les 41 tests ciblés et les 711 tests du dépôt passent. Aucun build Gate A,
verrou, oracle ou modèle n'a été ouvert par cette implémentation. Le prochain
geste autorisé est la création puis le contre-audit du verrou d'exécution.
*(commit GitHub : `e059148`)*
Le premier verrou candidat a été révoqué avant exécution : le contrôle
indépendant a détecté que le hash de la bibliothèque Python, cohérent entre
plan/code/lock, était tronqué à 63 caractères face au fichier réel. Le hash
64 caractères a été corrigé dans le contrat, le plan et le certificateur ;
`GO_CODE_PATCH`, 41 tests ciblés, le smoke réel et 711 tests complets
confirment le correctif. Aucun build n'a été lancé avec le verrou fautif.
*(commit GitHub : `c22d05a`)*
Le verrou corrigé, hash
`31aab729f33db26350da37e8d1fbf427d19a8153112d353973088df83e620b9f`,
franchit `GO_LOCK_V412_STRICT_STORES` et `GO_LOCK_2`. Les deux
contre-audits ont validé respectivement 7 901 et 342 contrôles, dont les
1 945 fichiers physiques (7 224 974 001 octets), les blobs Git, le runtime
réel, le routage, les subsets et l'absence d'inputs interdits. Le prochain
geste autorisé est désormais l'unique build Gate A sous sandbox, toujours
sans accès à l'oracle. *(commit GitHub : `775c3bb`)*
Le premier lancement Gate A s'est arrêté sans publication : `Path.cwd()`
recevait `EPERM` sous la politique metadata-only de `RUN_ROOT`, un chemin que
le smoke initial n'exerçait pas. Deux PoC ont isolé cette cause ; les refus
joblib `SemLock` et `mach-lookup` étaient du bruit non fatal. Le worker
compare désormais l'identité `st_dev/st_ino` de `.` et `RUN_ROOT`, sans
élargir aucun droit, et force joblib en série. `GO_CODE_CWD_PATCH`, 44 tests
ciblés, le smoke réel et 714 tests complets sont verts. Le lock `775c3bb`
est révoqué ; un nouveau verrou est requis avant relance. *(commit GitHub :
`158014e`)*
Le verrou post-correctif, hash
`f9e5738eef35c9a4b9c636cf810a87ed8eb412077f7ac6bcb48c90ae02f8d189`,
franchit `GO_LOCK_CWD_PATCH` et `GO_LOCK_CWD_2` avec 7 900 et 325
contrôles indépendants. Il autorise la seconde tentative complète du même
Gate A, toujours sous sandbox et sans oracle. *(commit GitHub : `e759492`)*
La seconde tentative a terminé le worker complet puis s'est arrêtée avant
publication : APFS `noowners` refuse le renommage d'une racine déjà en
`0555`. Le PoC SSD reproduit l'écart. La promotion conserve maintenant
uniquement la racine en `0700` pendant `rename`, via un FD ancré, puis la
repasse en `0555` dans un `finally`, vérifie l'inode et synchronise les deux
parents. La recovery gèle les états transitoires avant validation.
`GO_CODE_APFS_PATCH`, 58 tests ciblés, 728 tests complets et le smoke réel
sont verts. Le lock `e759492` est révoqué ; aucun artefact incomplet n'a été
publié. *(commit GitHub : `809bb7e`)*
Le verrou APFS corrigé, hash
`265bc418d95a1de1902773b7f5548b5607a2b1360722192658dbddb544a0630d`,
franchit `GO_LOCK_APFS_PATCH` et `GO_LOCK_APFS_2` avec 7 900 et 325
contrôles indépendants. Il autorise la troisième tentative complète du Gate
A, sans changement du worker de matching ni accès à l'oracle. *(commit
GitHub : `7443ef4`)*
La troisième tentative franchit **`GO_V412_STRICT_STORES_SANDBOX`**. Les
648 partitions et 648 caches nécessaires aux 1 456 requêtes dev sont
accessibles sous sandbox, sans cache manquant ou reconstruit ; le lookup
retourne exactement les 10 000 SIRET contrôlés. Le worker n'a ouvert ni
oracle, label, modèle ou résultat historique, et les refus oracle/audit,
écriture, réseau et fork sont effectifs. Son pic RSS est de 1,9568 Go.
Deux audits indépendants rendent également `GO`, avec 20 848 et 11 898
contrôles réussis et le rehash complet des 1 954 entrées du ledger. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/certifications/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4`.
Rapport : `reports/v9/v4_12_strict_stores_results.md`. Ce GO certifie les
stores et l'isolement, pas encore le Recall@100 ni la latence par requête.
Il autorise uniquement le contrat du moteur unitaire et de sa parité ; les
modèles et le test final restent fermés. *(commit GitHub : `614efc2`)*
Le contrat du moteur unitaire retrieval et de sa parité est maintenant
préenregistré et reçu **`GO_CONTRACT_FINAL`**, `GO_CONTRACT_SECURITY` et
`GO_CONTRACT_INDEPENDENT`. Le worker aveugle reproduira localement le sparse
V4.11 exact — nom, adresse, rescues simples, RRF et padding — puis publiera
uniquement `query_id`, rang et SIRET, avec un plafond strict de 100. Il ne
peut ouvrir ni oracle, historique, modèle ou réseau. Après sa terminaison, un
contrôleur séparé, lui-même sans accès au Parquet historique contenant la
vérité, comparera les deux payloads canoniques aux hashes préenregistrés :
145 236 candidats, pools 46–100, hash candidat `1689a2...ab00` et statut
`65e662...5518`. Les blobs du commit reproduisent exactement le contrat
`007ada2f...fe33` et le plan `7eff59a9...180d`. Ce GO autorise uniquement
l'implémentation et son audit de code, pas encore le run, le Recall, les
modèles ou l'oracle. *(commit GitHub : `370a3aa`)*
Le moteur unitaire, son orchestrateur sandbox, le contrôleur de parité et
leur contre-audit indépendant sont implémentés et reçus
**`GO_V412_UNIT_RETRIEVAL_INDEPENDENT_AUDIT`**. Le cœur reproduit le sparse
historique gelé sans importer son pipeline ; le worker ne publie que les
statuts et les listes ordonnées de 100 SIRET au plus. Les deux profils
Seatbelt ont été exécutés réellement sur le Mac avec un Python privé. Les
lectures sensibles descendent par `openat` et `O_NOFOLLOW` sur chaque
composant, les mêmes FDs sont recontrôlés avant/après, et les promotions
utilisent un renommage atomique sans remplacement. Les tests adversariaux
rejettent faux exécutables, rapport de recovery mensonger, substitution du
pending et collision de destination. L'audit final vérifie 13 sources, 12
contrôles statiques et 13 contrôles synthétiques ; 133 tests ciblés et les
803 tests du dépôt passent. Aucun dev réel, store réel, oracle, historique
ou modèle n'a été ouvert par ce jalon. Le prochain geste autorisé est la
création puis les deux contre-audits du verrou d'exécution ; le run worker,
la parité, le Recall et les modèles restent interdits jusque-là. *(commit
GitHub : `6726a95`)*
Un audit d'exécutabilité post-commit a ensuite refusé le passage au verrou :
le worker publié n'avait aucun producteur de run-spec de parité en production.
Le runner enchaîne désormais obligatoirement worker, revalidation, construction
canonique du run-spec, contrôleur Seatbelt et validation complète du `GO`.
La source contrôleur et le run-spec sont consommés depuis des FDs ancrés ; le
Python parent est copié depuis les octets verrouillés dans un runtime privé
scellé ; plan, lock, sources et inputs sont recontrôlés après la parité. Les
PoC faux `GO`, spec mutée, publication substituée, binaire remplacé et cleanup
TOCTOU sont bloqués. `GO_EXECUTION_PATH`, 151 tests ciblés et 821 tests
complets valident le raccord, sans dev réel. Le lock candidat antérieur est
révoqué et doit être régénéré sur ce commit avant toute exécution. *(commit
GitHub : `f4a5309`)*
Le verrou d'exécution final, hash
`0852ee260af4dd66976adaea9831204c1ad968dbdbca6241f07e5f2964b27caf`,
franchit **`GO_LOCK_1`** et **`GO_LOCK_2`**. Il épingle le commit code
`f4a53096338ec5bf2cb3237b5361c6e0e513eedf`, 13 sources, 16 entrées,
les exécutables, le runtime, les quatre racines et les deux projections
worker. Le premier audit a exécuté 15 104 contrôles et rehashé physiquement
7 225 618 142 octets ; le second a recalculé sans importer le runner les
1 945 entrées Gate A contre le ledger de 1 954 lignes. Aucun chemin oracle,
historique ou modèle n'entre dans le worker. Ce GO autorise désormais
l'unique exécution end-to-end worker puis parité ; il n'autorise toujours ni
l'ouverture de l'oracle, ni le Recall, ni les modèles. *(commit GitHub :
`bf82e74`)*
La première tentative sous ce verrou s'est arrêtée en environ cinq secondes,
avant le démarrage du module, toute requête et toute publication :
`ModuleNotFoundError: No module named 'xgb_matcher'`. Le paquet privé était
présent, mais Seatbelt empêchait Python d'en découvrir le répertoire. Le
verrou `bf82e74` est donc révoqué. Le correctif limite la nouvelle lecture au
seul paquet privé scellé, fixe `PYTHONPATH` sur le staging, désactive les
chemins Python implicites et nettoie uniquement le staging courant sur échec
du worker. Deux audits rendent `GO_IMPORT_PATCH` et `GO_IMPORT_PATCH_2`; 153
tests ciblés, deux intégrations macOS natives et les 823 tests du dépôt
passent. Aucun résultat dev, worker ou parité n'a été produit ; oracle,
historique et modèles sont restés fermés. Rapport :
`reports/v9/v4_12_unit_retrieval_launch_failure.md`. Une relance exige un
nouveau verrou doublement audité. *(commit GitHub : correctif `58fabf3`)*
Le verrou de remplacement, hash
`097b65ea73578f2993ffedb133878a7708138a1ab7fa3acc16dfc7b102861359`,
épingle le commit corrigé `58fabf3b42540d1862d1ef3d12cf7cd2f22a2fd4`
et franchit **`GO_LOCK_IMPORT_1`** et **`GO_LOCK_IMPORT_2`**. Les audits
confirment 13 sources, 16 entrées, les exécutables et le runtime exacts. Le
second audit, sans importer le runner, a rehashé les 1 945 fichiers Gate A
(6,73 Gio) et les a rapprochés des 1 954 lignes du ledger. Les projections
worker restent sans oracle, historique, dataset ou modèle. Ce verrou autorise
une nouvelle tentative end-to-end worker puis parité ; il n'autorise toujours
pas l'ouverture de l'oracle, le calcul du Recall ou le dégel des modèles.
*(commit GitHub : `a1c1db8`)*
La tentative autorisée par `a1c1db8` a franchi l'import puis s'est arrêtée
avant toute requête : sous Seatbelt, `platform.platform()` omettait le
processeur et `Mach-O`, ce qui faisait diverger le nom de plateforme du plan
alors que Python et les huit bibliothèques étaient identiques. La valeur
Darwin est désormais reconstruite sans sous-processus à partir de la version
macOS, de la machine et de la taille de pointeur. Le test natif compare le
dictionnaire runtime complet sous la sandbox réelle. Deux audits rendent
`GO_RUNTIME_PATCH_1` et `GO_RUNTIME_PATCH_2`; 153 tests ciblés et les 823
tests du dépôt passent. Le verrou `a1c1db8` est révoqué. Aucun candidat,
manifeste worker ou résultat de parité n'a été produit. Rapport d'incident
mis à jour : `reports/v9/v4_12_unit_retrieval_launch_failure.md`. Une
nouvelle relance exige encore un nouveau verrou doublement audité. *(commit
GitHub : correctif `a0a0e37`)*
Le verrou runtime corrigé, hash
`778946fae29fb427318c29eee4fba71dea60f1b1d6ea67906caab872441d1def`,
épingle `a0a0e3795948d92c5c41e65cfd3998d8e21781ab` et franchit
**`GO_LOCK_RUNTIME_1`** et **`GO_LOCK_RUNTIME_2`**. Les 13 sources, 16
entrées, quatre exécutables et le runtime sont exacts. L'audit indépendant
sans import du runner a de nouveau rehashé les 1 945 fichiers Gate A contre
les 1 954 lignes du ledger. Aucun chemin oracle, historique, dataset ou
modèle n'est exposé au worker. Une nouvelle tentative worker puis parité est
autorisée ; l'oracle et les modèles restent fermés. *(commit GitHub :
`662d555`)*
La troisième tentative termine en 1 030,16 secondes et franchit
**`GO_V412_UNIT_RETRIEVAL_PARITY`**, confirmé indépendamment par
`GO_ARTIFACTS_1` et `GO_ARTIFACTS_2`. Les 1 456 requêtes produisent 145 236
candidats, avec des pools de 46 à 100, 13 pools sous le plafond, aucun pool
vide et aucun lookup manquant. Les payloads candidats
`1689a2f3...ab00` et statuts `65e662c0...5518` égalent exactement les
valeurs préenregistrées. Le ledger couvre 1 980 entrées inchangées ; oracle,
labels, historique, modèles et réseau sont restés fermés. Pic mémoire :
3,39 Gio. Artefacts worker `d2915fe7...dd1a` et parité
`d587937b...05f5`. Rapport :
`reports/v9/v4_12_unit_retrieval_parity_results.md`. Ce GO autorise seulement
le contrat puis l'audit d'un évaluateur oracle séparé ; il ne republie pas
encore le Recall et ne dégèle aucun modèle.
Le contrat de l'évaluateur oracle séparé est préenregistré et doublement
audité **`GO_EVALUATOR_CONTRACT_1`** et
**`GO_EVALUATOR_CONTRACT_2`**. Il gèle la jointure worker
`d2915fe7...dd1a` / oracle `c4045da8...fa70`, les 1 456 requêtes dont
1 217 `MATCH_EXACT`, Recall@1/10/50/100, les Wilson 95/99 et les gates
observés couverture ≥ 80 % / Recall@100 ≥ 99 %. Les références historique,
V2 et V3 seront republiées ensemble mais distinguées de la mesure V4.12.
Un reçu et un journal parent-only sont synchronisés avant toute ouverture
oracle ; la publication audit puis évaluation est non-clobber et sa reprise
est limitée à la promotion d'arbres déjà complets. Tous les payloads,
keysets, schémas et états sont déterministes. Aucun oracle, historique,
modèle ou test final n'a été ouvert pendant ce jalon. Ce GO autorise
uniquement l'implémentation et l'audit de l'évaluateur, pas encore la mesure.
*(commit GitHub : `fe266bd`)*
L'audit de code a ensuite montré qu'une vraie reprise après ouverture oracle
était impossible avec une preuve conservée seulement en mémoire. Le contrat
et le plan sont amendés et doublement reçus
`GO_EVALUATOR_CONTRACT_AMEND_1` / `GO_EVALUATOR_CONTRACT_AMEND_2` :
`computed_attestation.json` scelle désormais les 16 entrées et les deux
arbres validés, puis son hash devient monotone dans le journal v2. Une
reprise post-oracle peut ainsi valider et promouvoir les octets déjà calculés
sans rouvrir la vérité. Les arbres, manifests, rôles 12 data + 4 runtime,
temporaires d'état hors slot et verrou parent durable sont définis
exactement. Aucun oracle ni résultat réel n'a été ouvert pendant
l'amendement. *(commit GitHub : `9e25ebf`)*
L'évaluateur scellé, son parent, son profil Seatbelt, son audit indépendant
et leurs tests sont implémentés et doublement reçus
**`GO_EVALUATOR_CODE_1`** / **`GO_EVALUATOR_CODE_2`**. Le worker est chargé
et attesté avant le commit oracle, puis reçoit les quatre FDs oracle dans
l'ordre contractuel unique via `SCM_RIGHTS`. L'attestation calculée, le
journal v2, le verrou de slot, la reprise sans réouverture oracle, les
manifests, le ledger, la provenance, le plafond RSS et la publication
exclusive sont testés, y compris via le vrai orchestrateur. Les falsifications
coordonnées, symlinks, IDs/rangs invalides et fenêtres de crash sont rejetés.
67 tests evaluator et les 890 tests du dépôt passent ; smokes et audit
statique sont `GO`. Aucun input réel n'a été ouvert. Le prochain geste
autorisé est la création puis le double contre-audit du verrou evaluator, pas
encore l'ouverture oracle. *(commit GitHub : `3ebddc9`)*
Le verrou evaluator, hash
`bcda9024258031ca10e00313443e75ddb5f5650d599e310c4e7eafd27b1e6b4f`,
épingle le commit code `3ebddc9e977151c91d827a783d9996c642e04a58`
et franchit **`GO_EVALUATOR_LOCK_1`** /
**`GO_EVALUATOR_LOCK_2`**. Les 7 sources correspondent au worktree et aux
blobs Git ; les 12 entrées non-oracle ont été rehashées physiquement. Les
quatre engagements oracle ont uniquement été comparés entre plan et verrou,
sans accès filesystem. Runtime, sandbox, racines, RSS et identités build
`50cbc46e...32e7c`, slot `9cf7f6d3...21b7` et attempt
`01260473...c2ed` sont exacts ; aucune destination n'existait au gel. Ce
verrou autorise désormais l'unique évaluation oracle officielle. *(commit
GitHub : `d886ee9`)*
L'évaluation officielle termine `FINAL` et franchit
**`GO_V412_UNIT_RETRIEVAL_EVALUATION`**, confirmé par
`GO_EVALUATOR_ARTIFACTS_1` / `GO_EVALUATOR_ARTIFACTS_2`. Sur 1 456
requêtes, 1 217 sont `MATCH_EXACT` et 239 `AMBIGUOUS` : couverture
identifiable **83,585 %**. Le Recall exact vaut 1 075/1 217 à @1
(88,332 %), 1 211/1 217 à @10 (99,507 %) et 1 217/1 217 à @50/@100
(100 %), avec zéro vérité absente. La borne Wilson bilatérale 99 % de
Recall@100 est 99,458–100 %. Les 145 236 candidats respectent tous le
plafond 100. La chaîne officielle possède sept événements, 16 entrées
conformes et termine `FINAL`; aucun modèle ni test final n'a été ouvert.
Rapport : `reports/v9/v4_12_unit_retrieval_evaluation_results.md`. Ce GO est
un gate développement historique ; il autorise le contrat de l'unique test
retrieval final, pas encore le dégel du ranker/accepteur.
L'audit de transférabilité conclut ensuite
**`PIVOT_NEW_HOLDOUT_REQUIRED`**. Le test final sélectif consommé mesurait
une admission multicanal différente : elle obtenait 2 116/2 128 = 99,436 %,
alors que son sparse seul — correspondant à la famille V4.12 — obtenait
2 059/2 128 = 96,758 %. Le résultat final ne peut donc pas être hérité.
L'inventaire confirme que les 23 609 lignes CRM locales sont toutes
consommées : 23 384 par historique/V4-Fresh puis les 225 restantes par le
challenge V4.11. Aucun nouvel export CRM local n'existe depuis le registre du
28 juillet. Rapport :
`reports/v9/v4_12_retrieval_final_evidence_decision.md`. V4.12 reste
candidat grâce à son GO dev, mais toute certification finale exige un nouvel
export CRM indépendant ; ranker et accepteur restent gelés.

Le contrat V4.11-B est préenregistré avant tout nouveau dataset ou fit.
Il corrige la frontière produit : le SIRET/SIREN historique du CRM devient
une étiquette cachée et ne peut plus alimenter le retrieval, le ranker ou
l'accepteur. Un diagnostic de sensibilité montre que 5 882/5 883 exacts
restent dans le sparse et au top-1 après masquage des signaux directs. V4.11
reconstruira donc un vrai top-100 sparse sans branche identifiant, entraînera
un ranker C de 45 features sur ces pools, puis comparera exactement deux
accepteurs sur une scène compacte de 80 features. Le dev historique reste
développement uniquement ; les 225 lignes inédites restent fermées jusqu'au
gel du candidat. *(commit GitHub : `ca83603`)*
L'audit indépendant pré-fit a ensuite fermé les ambiguïtés restantes :
`UNRESOLVED` est exclu des cibles, les 80 formules/types/contraintes sont
définis sans score absolu inter-fold, les deux moitiés dev ont des volumes
attendus, les baselines sont épinglées et les seuils utilisent une règle
entière déterministe. *(commit GitHub : `399252f`)*
Le calcul de scène V4.11 partagé train/serve est implémenté et testé : ordre
exact de 80 features, 34 binaires et 46 standardisées, vecteur monotone
`49/+`, `6/-`, `25/0`, normalisation des scores par requête, tie-break et cas
0/1 candidat. La suite complète passe 395 tests. Aucun modèle n'a été
entraîné. *(commit GitHub : `c7075ac`)*
L'audit d'intégration des scènes a fermé un échec silencieux avant tout fit :
les cinq champs SIRENE nécessaires aux rôles/NAF sont maintenant obligatoires,
un NAF inconnu n'est plus un faux accord, et plafond/rangs/SIRET sont validés
strictement. La suite passe 402 tests avec le builder retrieval en cours.
*(commit GitHub : `58c70f4`)*
Le contrat transporte désormais explicitement les cinq champs SIRENE bruts
nécessaires aux rôles/NAF *(commit GitHub : `f1bdcdd`)*. Le builder V4.11
input-blind est implémenté et audité `GO build` : vraie requête sparse sans
argument SIRET/SIREN/vérité, top-100 actif, vérité jointe seulement après
fermeture de tous les pools, 45 features ranker et cinq champs de rôle issus
du snapshot. Un smoke réel produit 100 candidats et un NAF réel sans colonne
interdite ; 402 tests passent. Aucun fit n'a encore eu lieu.
*(commit GitHub : `3149d04`)*
Le premier lancement complet s'est arrêté en préflight avant toute requête :
les splits V4.6 attribuent un numéro de pli aux lignes dev aussi. Le
garde-fou a été corrigé pour valider les cinq plis gelés et l'unicité du pli
par composante sur les 7 003 lignes ; 403 tests passent. Aucun pool, label ou
résultat n'a été produit par cette tentative. *(commit GitHub : `734dc24`)*
Le lancement suivant a été interrompu proprement avant toute ouverture des
labels lorsque le RSS a dépassé 7,6 Go : le builder rescannait le snapshot
SIRENE à chaque requête et gardait un cache TF-IDF non borné. Le chemin
corrigé écrit d'abord les pools bruts aveugles, borne le cache RAM à 20,
hydrate état et rôles par une jointure bulk unique, puis recalcule les 45
features avant de fermer le top-100 final. Les identifiants CRM sont exclus
dès la projection Parquet ; les caches disque sont liés aux partitions et
vérifiés par SHA-256 avant désérialisation ; labels et baseline ne sont
hashés/lus qu'après fermeture du pool. Un oracle indépendant confirme la
parité exacte avec l'algorithme précédent, y compris IDF par défaut, fermés,
tie-break et enseignes divergentes. Deux audits ont conclu `GO`, un smoke
réel produit 100 candidats/59 colonnes avec un seul scan, et 437 tests
passent. Aucun résultat retrieval n'a encore été produit. *(commit GitHub :
`fc8c848`)*
Le build complet V4.11 franchit désormais
**`GO_TRAIN_INPUT_BLIND_RANKER`**. Sur le dev historique, le bon SIRET est
présent dans 1 217/1 217 pools à 100 ; sur le fit, dans 4 665/4 666
(99,9786 %), avec l'unique miss `6818`. Les 7 003 requêtes ont toutes un
pool, le plafond maximal est exactement 100 et les 698 892 candidats sont
actifs, uniques et sans injection. Une recomputation indépendante confirme
les hashes, les rangs, les cibles, l'absence des identifiants CRM et le blob
Git du builder. Le dev reste un jeu de développement consommé ; ce `GO`
autorise le ranker C, pas la preuve produit finale. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d`.
Rapport : `reports/v9/v4_11_input_blind_retrieval_results.md`.
*(commit GitHub : `db5d233`)*
Le runner du ranker C est maintenant implémenté, sans l'avoir encore
exécuté : cinq modèles OOF scorent les scènes fit, un modèle complet score le
dev, les misses retrieval restent des erreurs end-to-end et chaque fit est
rejoué deux fois avec égalité exacte exigée. Le diagnostic ranker B masqué
reconstruit correctement le canal sparse unique (`channel_count=1`) et son
score RRF à partir du rang. Le runner ne sera lancé que si le build
input-blind franchit d'abord le gate Recall@100. *(commit GitHub :
`b6a2332`)*
Le ranker C franchit maintenant **`GO_RANKER_C`** : 4 661/4 666
(99,8928 %) au top-1 SIRET OOF fit et 1 216/1 217 (99,9178 %) sur le dev,
avec modèles, scores et rangs reproduits bit à bit. L'unique vérité absente
du pool, `6818`, reste une erreur end-to-end ; l'unique erreur dev est
`13958`. Un premier artefact correct sur les scores a été superseded avant
promotion parce que son compteur `retrieval_miss` désignait les pools vides
et que la description du canal sparse ne correspondait pas à la matrice. Le
runner corrigé distingue pool vide et vérité absente, puis documente
`channel_count=1`; l'audit indépendant conclut `GO` vers l'accepteur.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/e13eb3ac7498256e`.
Rapport : `reports/v9/v4_11_ranker_c_results.md`.
*(commits GitHub : correctif `d2a6f5b`, résultats `c9f16c4`)*
Le dataset et le runner de l'accepteur compact sont aussi implémentés, mais
restent inactifs jusqu'aux gates précédents. Ils imposent le rattachement à
un artefact ranker C `GO` réellement OOF, l'étanchéité des composantes
train/dev, les 5 547 scènes fit et les volumes dev préenregistrés. Les
`UNRESOLVED` sont exclus du fit et matérialisés en `REVIEW`; un éventuel
bundle `GO` épinglera par hash le retrieval, le ranker C, la taxonomie, le
contrat, le calcul de scène et l'accepteur. Le contrôle indépendant a trouvé
puis fait fermer cinq manques de gouvernance avant tout fit ; la suite
complète passe 428 tests. *(commit GitHub : `2a9f51f`)*

Le dataset de scènes de l'accepteur V4.11 franchit maintenant
**`GO_FREEZE_PLAN`**. Il contient 5 547 scènes fit produites par prédictions
OOF et 1 456 scènes dev hors échantillon, soit 7 003 requêtes, 80 features et
5 877 cibles positives. Les cinq erreurs fit, l'erreur dev et les 1 120 cas
`AMBIGUOUS` restent explicitement négatifs ; aucun cas n'a été retiré. Un
premier manifeste a été supersédé avant tout fit car il n'épinglait pas le
code transitive de fonction de site. Le build corrigé verrouille retrieval,
ranker, prédictions, contrat, taxonomie, calcul de scène et fonction de site.
Son parquet est bit à bit identique au premier et le contre-audit conclut
`GO_FREEZE_PLAN`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/52ea3faba9a56aff`.
Rapport : `reports/v9/v4_11_acceptor_scene_dataset_results.md`.
*(commits GitHub : correctif `c462a21`, résultats `19f1169`)*

Le plan d'entraînement de l'accepteur V4.11 est gelé avant tout fit. Il
autorise exactement une logistique compacte et un XGBoost monotone peu
profond, avec leurs hyperparamètres fixes, les 80 features dans leur ordre,
les trois populations étanches et la sélection à 99,8 % de précision,
80 % de couverture et zéro `AMBIGUOUS` automatisé. Le verrou d'exécution lie
ce plan au runner et aux sources commités ; préflight et contre-audit
concluent `GO`. Aucun challenge, holdout, unseen ou test final n'a été ouvert.
*(commits GitHub : plan `8033934`, verrou `fd70a64`)*

Le développement de l'accepteur V4.11 conclut
**`GO_FREEZE_V411_CANDIDATE`**. La logistique compacte gagne au seuil
`0.8720916706888049` : 614/746 AUTO, 614 corrects, zéro ambigu automatisé,
soit 82,306 % de couverture et 100 % de précision observée sur
`comparison_dev`. Le XGBoost monotone obtient 612/746 sans erreur. La
baseline obtient 618/746 avec une erreur. Deux audits recomputent à
l'identique modèles, seuils, métriques, familles et bundle. Ce GO autorise
uniquement le gel et le challenge descriptif des 225 cas : le ranker était
déjà exact sur les 634 labels exacts de comparaison, la borne basse Wilson
95 % du 614/614 n'est que 99,378 %, et le dev est historique. Il ne s'agit
donc ni d'une certification 99,8 % ni d'une promotion produit. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/9d23bf3deb6b63de`.
Rapport : `reports/v9/v4_11_acceptor_development_results.md`.
*(commit GitHub : `f99c1d1`)*

Le challenge descriptif V4.11 est préenregistré avant qualification et
inférence. Une inspection du registre a exposé à l'orchestrateur le SIRET CRM
de trois lignes ; elles restent dans l'unique run mais sont exclues de la
métrique aveugle principale, qui portera sur 222 lignes, et publiées dans une
cohorte `EXPOSED_3`. Le CRM sera projeté physiquement sans identifiant, les
labels seront produits mécaniquement par la politique V4 gelée et hashés
avant toute inférence, puis les prédictions seront scellées avant ouverture
des labels. Le challenge reste descriptif et ne constitue aucun gate produit.
*(commit GitHub : `30fa0b8`)*

Les builders du challenge V4.11 sont implémentés et audités avant ouverture :
projection CRM physique, mapping scellé, qualification mécanique V4, preuves
et labels immuables avec validateurs fail-closed. La suite passe 451 tests.
Le docket assaini est maintenant construit avec 225 lignes et exactement
sept colonnes CRM ; aucun SIRET/SIREN, fingerprint, label, candidat ou score
n'est présent. Les cohortes contiennent 222 lignes aveugles et trois lignes
exposées. Le contre-audit conclut `GO_QUALIFY`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_sanitized/1c994c852c10acaf`.
*(commit GitHub : `1fc058f`)*

La qualification mécanique du challenge est gelée avant toute prédiction :
74 `MATCH_EXACT`, 17 `AMBIGUOUS` et 134 `UNRESOLVED`. La cohorte aveugle
principale contient 73 exacts sur 222 lignes ; la cohorte exposée un exact
sur trois. Les 138 preuves actives, leurs cardinalités et les identifiants
exacts sont cohérents ; aucun `NO_MATCH`, secours web, modèle ou score n'a
été utilisé. Le contre-audit conclut `GO_FREEZE_LABELS`. Cette population
n'est identifiable qu'à 32,889 % et reste descriptive. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_qualification/4f9ef46516b89ab8`.
Rapport : `reports/v9/v4_11_unseen_qualification_results.md`.
*(commit GitHub : `6b84597`)*

Le runner du challenge descriptif unique est maintenant commité et verrouillé,
sans avoir ouvert ni scoré les 225 cas. Il impose un ledger global indépendant
du répertoire de sortie, scelle et re-hashe les 225 prédictions avant toute
désérialisation des labels, contrôle exactement les populations 222/3 et
sépare erreurs confirmées, AUTO invérifiables et couverture des seuls
`MATCH_EXACT`. Les cinq scripts d'orchestration, tous les modules
`src/xgb_matcher`, les modèles, données et versions runtime sont épinglés par
hash et commit. Deux audits concluent `GO_COMMIT_RUNNER`; 462 tests passent et
la parité historique est bit-exacte sur 1 456 requêtes et 145 236 candidats,
avec cinq contrôles exacts. Le prochain acte autorisé est l'unique exécution
descriptive sous ce verrou. *(commits GitHub : runner `cd1cab5`, verrou
`da6924a`)*

L'unique challenge descriptif V4.11 est terminé avec une intégrité
contre-auditée, et conclut **`PIVOT_ACCEPTOR_EVIDENCE_GATE`**. Sur les 222
lignes aveugles, le retrieval et le ranker réussissent les 73/73 cas exacts,
mais l'accepteur automatise un cas `AMBIGUOUS` : 73/74 décisions AUTO
évaluables sont correctes, soit 98,649 %. Les 72 autres AUTO aveugles sont
`UNRESOLVED` et restent invérifiables. L'erreur contient deux candidats forts
de deux SIREN différents, tous deux dans le top 2 ; la scène ne compte pas
explicitement les identités fortes inter-SIREN. La prochaine variante doit
donc tester une garde déterministe « plusieurs SIREN forts → REVIEW » et les
features correspondantes sur les anciennes populations, puis être gelée
avant un nouvel export. Il est interdit de régler le seuil sur ce challenge
consommé. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_execution/ddb7336e8c2e042d`.
Rapport : `reports/v9/v4_11_unseen_execution_results.md`.
*(commit GitHub : `62e9741`)*

Le contrat V4.12-G est préenregistré avant tout nouveau build. Retrieval,
Ranker C, accepteur V4.11 et seuil restent gelés ; une garde déterministe
n'autorise AUTO que si l'univers géographique actif contient exactement un
candidat direct fort, égal au top-1 déjà accepté. La garde est un veto pur et
ne peut ni choisir ni injecter un candidat. Une allowlist par chemin, hash,
phase et projection limite le développement aux artefacts historiques ; les
trois racines du challenge consommé et tous leurs outputs sont interdits par
hash. Avant le seal des preuves, seules les queries et partitions sont
ouvrables. Le contrat distingue les gates retrieval (couverture exacte et
Recall@100) des gates de décision (couverture AUTO et précision), documente
la circularité des labels mécaniques et exige un nouvel export indépendant.
Deux audits concluent `GO_CONTRACT`. Aucun build V4.12 n'a encore été lancé.
Le gate de performance hors-ligne est clarifié : le temps moyen par requête
sur un batch n'est pas assimilé à une latence de service. Le constructeur
label-free est désormais implémenté et contre-audité : il parcourt l'univers
géographique actif complet, produit une preuve par requête et par candidat,
refuse tout artefact non autorisé et scelle ses sorties de façon atomique.
Les 50 tests V4.12 ciblés et les 512 tests complets passent. Son exécution est
gelée par un verrou audité `GO_COMMIT_LOCK`, qui fixe le commit, les 53
sources, les entrées et le runtime. Le calcul sur les 7 003 requêtes n'a
été autorisé qu'après ce gel. *(commits GitHub : contrat `66e7b9c`,
clarification `31f2721`, constructeur `e822136`, verrou `11c5de9`)*

Le build label-free V4.12 est désormais scellé et contre-audité
**`GO_SEALED_EVIDENCE`**. Sur 7 003 requêtes, 5 883 (84,007 %) ont exactement
un candidat direct actif et 1 120 (15,993 %) en ont plusieurs : 977 collisions
inter-SIREN et 143 cas multisites intra-SIREN. Il n'existe aucun cas sans
preuve directe. Les 10 275 preuves candidates sont actives, uniques par
requête/SIRET et reliées bijectivement aux agrégats. Le pic RSS est de
2,99 Go. Aucun label, challenge, pool ranker, scène ou modèle n'a été ouvert
avant le seal. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_direct_evidence/10f16403795ccee6`.
Rapport : `reports/v9/v4_12_direct_evidence_build_results.md`. La prochaine
étape est le runner post-seal audité, puis l'unique gate historique
`comparison_dev`; aucune promotion n'est encore autorisée.
*(commit GitHub : `3aff8d9`)*

Le runner post-seal V4.12-G est implémenté et contre-audité
**`GO_COMMIT_EVALUATOR`**. Il recalcule les trois populations, reproduit la
baseline V4.11 `614 AUTO / 0 erreur / 0 ambiguïté`, applique un veto pur et
publie les gates entiers et segmentaires. Sa publication est fermée par Git,
hashes, TOCTOU, RSS, fsync et validation sémantique ligne à ligne. Les 23
tests ciblés et les 535 tests complets passent. Le verrou externe audité fixe
11 sources, 13 entrées, le seal V4.12, le modèle, le seuil et le runtime.
*(commits GitHub : évaluateur `37f1476`, fermeture source `0182248`, verrou
`a99dd31`)*

Le gate historique V4.12-G est exécuté, validé et contre-audité
**`GO_V412_HISTORICAL_GATE`**. Sur `comparison_dev`, V4.11 et V4.12-G
conservent 614/746 AUTO, tous exacts, sans ambiguïté automatisée et sans perte
sur aucun des onze segments publiés. Hors gate, la garde retire trois erreurs
`AMBIGUOUS` du fit et une du threshold dev, toutes dues à deux preuves fortes
inter-SIREN, sans retirer d'AUTO exact. Deux préflights se sont arrêtés avant
scoring et sans artefact sur des conventions valides du fit/dev ; les
correctifs et reverrouillages ont été audités avant l'exécution publiée.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/v4_12_guard_historical/fedcd1d512bfd269`.
Rapport : `reports/v9/v4_12_guard_historical_results.md`. Ce GO reste un
contrôle de cohérence sur des labels circulaires :
`latency_gate_evaluated=false` et `production_certified=false`. La suite est
le gel du bundle, la parité batch/inférence et la latence appariée, puis un
nouvel export indépendant. *(commits GitHub : correctifs `7d70249`,
`e8b052f`, reverrouillages `9452d0f`, `3e3dabc`, résultat `94414af`)*

Le préalable d'inférence V4.12 est préenregistré et contre-audité
**`GO_CONTRACT_LOOKUP`**. V4.11 hydrate aujourd'hui le snapshot SIRENE en
bulk ; aucune p95 par requête honnête n'est donc encore mesurable. La brique
suivante est un DuckDB local en lecture seule contenant exactement les sept
colonnes utilisées par l'hydratation historique, indexées par SIRET. Le
contrat fixe le snapshot de 42 322 035 SIRET uniques, les 698 892 candidats
historiques (508 081 SIRET uniques), un contrôle indépendant de 10 000 SIRET,
les limites Mac/SSD et l'API fail-closed. Aucun lookup n'a encore été
construit. Après parité exacte seulement, un moteur persistant pourra mesurer
la p95 appariée sur les 1 456 requêtes dev.
*(commit GitHub : `00ce1c3`)*

Le builder et le store lookup V4.12 sont implémentés et contre-audités
**`GO_COMMIT_LOOKUP_BUILDER_FINAL`**. Une mini-publication exerce réellement
DuckDB, l'index, la référence/parité, RSS/disque, WAL, fsync, renommage
atomique et revalidation ; les tests de falsification ferment verrou Git,
provenance, TOCTOU et schémas imbriqués. Les 60 tests ciblés et 595 tests
complets passent. Le verrou réel audité fixe sept sources, quatre entrées,
DuckDB 1.4.3 et la racine SSD ; environ 1 049 Gio sont libres et la cible est
absente. Le build réel de 42 322 035 lignes n'a pas encore été lancé.
*(commits GitHub : builder/store `a06cf00`, verrou `591f339`)*

Le build réel du lookup franchit **`GO_V412_SNAPSHOT_LOOKUP`** :
42 322 035 SIRET uniques, zéro invalide, index et lecture seule conformes,
zéro écart sur les 508 081 candidats V4.11, sample indépendant conforme et
pic RSS de 7,8004 Gio. Un premier contre-audit avait publié à tort
`STOP_V412_LOOKUP_PARITY` : ses trois commandes hashaient les deux caractères
littéraux antislash et `n` (`160 000` octets), produisant `72f43460...`.
Avec le véritable octet LF demandé par le contrat, le payload fait
`150 000` octets et reproduit bien `58c970...`. L'incident reste documenté
et un contre-validateur distinguant explicitement les deux encodages sera
ajouté avant le benchmark. L'artefact
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup/ff0f33ad10803cfb`
autorise désormais la construction du moteur d'inférence, pas la production.
Rapport :
`reports/v9/v4_12_snapshot_lookup_results.md`.
*(commits GitHub : faux STOP conservé `880e57c`, correction `00d71c4`)*

Le contre-validateur indépendant du lookup est préenregistré
**`GO_CONTRACT_INDEPENDENT_AUDIT`** avant son implémentation. Il refait la
sélection SIRET depuis le snapshot sans importer le builder, construit le
payload avec `byte 0x0A`, vérifie explicitement le contre-exemple `5C 6E`,
rejoint séparément les six valeurs métier et compare le store par lots de
100. Schéma, cardinalité, index, quatre fichiers, ressources, sources
transitives et publication sont gelés. Un test devra modifier puis resceller
une valeur DuckDB : le validateur historique peut l'accepter, le nouveau doit
la refuser. Aucun audit formel n'a encore été exécuté.
*(commit GitHub : `5234084`)*

Le contre-audit formel franchit
**`GO_V412_LOOKUP_INDEPENDENT_AUDIT`**. Le runner, ses 23 tests ciblés et la
suite complète de 618 tests ont été contre-audités avant verrouillage. Le
run `4055be6e7a11b003` recalcule 10 000 SIRET depuis le snapshot :
vrai LF `150 000` octets/hash `58c970...`, contre-exemple `5C 6E`
`160 000` octets/hash `72f434...`. La comparaison fraîche donne zéro absent
et zéro écart de valeur/nullité ; le pic RSS est de 4 365 549 568 octets.
Le test hostile confirme qu'une valeur métier modifiée puis rescellée,
acceptée par le validateur historique, est refusée par le nouveau. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_snapshot_lookup/4055be6e7a11b003`.
Rapport :
`reports/v9/v4_12_snapshot_lookup_independent_audit_results.md`. Le lookup
autorise désormais le contrat du moteur requête par requête, pas la
production. *(commits GitHub : runner `3de02dc`, verrou `dd696de`, résultat
`1653175`)*

Le premier jalon du moteur unitaire est préenregistré
**`GO_CONTRACT_INPUTS`** après refus d'un contrat monolithique insuffisamment
séparé. Ce jalon ne calcule aucun match : il doit publier, dans une racine
runtime sans chemin sensible, les six champs CRM sûrs pour 7 003 requêtes et
les 1 456 dev, plus les inventaires cryptographiques des 4 119 partitions et
1 454 paires cache TF-IDF. Les hashes de contenu attendus sont gelés à
`680f1884...5463` et `589360b1...83ce`. Une racine d'audit distincte scellera
les 7 029 fichiers ouverts ; elle ne sera jamais transmise au worker. Ce GO
autorise seulement le builder/tests des inputs, pas l'oracle, le store, le
worker ou le benchmark. *(commit GitHub : `5ebd9de`)*

Le registre V4.11-A des populations consommées est construit et franchit
**`PASS_REGISTRY`**. Le benchmark fermé historique couvre 17 054 lignes
source et le pool V4-Fresh 6 330, sans recouvrement : leur union a déjà
consommé 23 384 des 23 609 lignes de `data/entrainements.csv`. Les 225 lignes
restantes ont toutes un `SERVICE ID` absent et ne peuvent pas constituer une
validation représentative. Elles sont réservées à un challenge descriptif
après gel de V4.11 ; une preuve finale exige un nouvel export CRM indépendant.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585`.
Rapport : `reports/v9/v4_11_consumed_population_registry_results.md`.
*(commits GitHub : contrat `d0eb5f3`, builder `0aa8ad2`)*

Le développement V4.10b se termine par
**`PIVOT_STRUCTURED_FEATURES`**. Aucune des six variantes structurées ne
franchit le gate. Les logits refusent seulement 16 à 18 des 25 mauvais cas et
automatisent l'unique ambigu ; les XGBoost en refusent 20 à 21, mais perdent
des bons cas ou la non-infériorité historique. `CURRENT80_W1` reste le plus
proche avec 23/25 mauvais refusés, sous le minimum de 24. Les 54 fits logiques
ont été répétés sans aucun écart, les deux audits indépendants confirment
hashes, seuils, gates et populations. Aucun bundle ni fresh dev n'est
autorisé. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_10b_structured_acceptor_development/71e067f75536180b`.
Rapport : `reports/v9/v4_10b_acceptor_development_results.md`.
*(commits GitHub : plan `5ed1ba3`, runner `6ae4cf7`, verrou `fb33c76`)*

Le dataset corrigé V4.10b est construit et franchit
**`GO_FREEZE_TRAINING_PLAN_V410B`**. Le nouveau catalogue autorise 641
features structurées : 157 continues/comptages à standardiser et 484
binaires non standardisées. Les 58 alias sont vérifiés ligne par ligne, les
16 signaux d'instrumentation retrieval et les 75 signaux de provenance sont
hors modèle. Les trois parquets restent identiques au build V4.10 et
`CURRENT80` est bit à bit inchangé. Aucun fit ni seuil n'a été produit.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10b_structured_acceptor/3ad8e97ce0118e8c`.
Rapport : `reports/v9/v4_10b_structured_dataset_results.md`.
*(commits GitHub : politique `eb85597`, clarification `d500fe2`, builder
`f78a9ba`)*

Le plan d'entraînement V4.10b est désormais gelé avant tout fit. Il conserve
trois variantes `CURRENT80` comme contrôles non promouvables et compare six
variantes structurées. Les facteurs de classe, seuils par pli, gates entiers,
54 fits logiques rejoués deux fois, lectures filtrées et bundles multiples
sont préenregistrés. Un verrou externe devra encore épingler le runner
commité avant le premier entraînement. *(commit GitHub : `5ed1ba3`)*

L'audit statistique pré-fit a invalidé l'ordre structuré V4.10 avant tout
entraînement. Il contenait 58 copies sémantiques et 16 signaux résiduels
capables de distinguer l'instrumentation retrieval V4.1 de V4.2-B. La
politique V4.10b, préenregistrée sans utiliser les labels ni les splits,
conserve `CURRENT80` bit à bit, ramène l'ordre structuré de 715 à 641
features, standardise aussi les compteurs pour la logistique et précise les
gates en arithmétique entière. Le build `0d6b87fd50fb550c` et son ancien plan
sont `superseded`; aucun fit ne les a consommés. *(commit GitHub : `eb85597`)*

Le plan d'entraînement V4.10 est gelé avant le premier fit dans
`config/v4_10_training_plan.json`. Il autorise exactement `BASE_FROZEN` et
neuf variantes appariées (`CURRENT80`, `STRUCTURED_LOGIT`,
`STRUCTURED_XGB`, poids difficiles 1/2/4), cinq plis difficiles group-OOF et
une sélection de seuil uniquement sur les 1 452 scènes du dev historique
effectif. Les cas random, frais, descriptifs verrouillés et le test final
restent interdits au fit, au seuil et au gate. Un audit indépendant pré-fit a
ensuite fermé les ambiguïtés de sélection et de reproductibilité sans ouvrir
ni scorer aucune donnée : `BASE_FROZEN` est seulement comparateur, le parquet
verrouillé est hashé mais jamais chargé sémantiquement, et chaque modèle
complet ou de pli doit être reproduit. *(commits GitHub : `47ff289`,
amendement pré-fit `dd0e3c8`)*

Le dataset V4.10 de l'accepteur structuré est construit et franchit
**`GO_TRAIN_V410`**. Il contient 7 003 scènes historiques, 94 cas difficiles
`hard_oof` et quatre cas descriptifs verrouillés. Les 80 features baseline
sont identiques bit à bit aux sources ; 715 features structurées sont
autorisées au modèle et 75 features de provenance restent audit-only. Les
698 428 paires prédiction/candidat V4.1 se joignent exactement, les jointures
CRM et SIRENE utiles sont à 100 %, les 20 supports de composantes sont
préservés et aucun ID random V4.8 n'entre dans les sorties. Aucun modèle ni
seuil n'a encore été entraîné. Rapport :
`reports/v9/v4_10_structured_dataset_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10_structured_acceptor/0d6b87fd50fb550c`.
*(commits GitHub : contrat `bc2384c`, `8ab9b01`, `1401269`, `99b2438`,
`2d86b5c`, `b19abed`; builder `2966d2b`; correctif `e10e9af`)*

L'audit V4.10 distingue désormais les 31 labels négatifs non détectés par le
garde lexical des véritables faux AUTO `HARD_W1` : sur 26 cas hors pli,
`HARD_W1` n'en automatise que deux ; les trois cas random de ce sous-ensemble
sont tous refusés. Les 31 se répartissent en 14 mauvais sites au sein du même
SIREN, 14 autres personnes morales à la même adresse, deux acteurs affiliés
ou support et un CRM composite. Le flux explique ces erreurs : l'accepteur
perd 47 des 64 features candidat du ranker, ne voit pas l'activité SIRENE et
réduit les frères d'un même SIREN à quelques agrégats. La prochaine
architecture sera donc un accepteur query-level unique enrichi, sans nouveau
veto ni modification retrieval/ranker. Rapport :
`reports/v9/v4_10_error_and_feature_flow_audit.md`.

La V4.9 se termine avec **`STOP_SITE_FUNCTION_GUARD`**. La taxonomie
déterministe, gelée avant mesure, refuse les trois erreurs random V4.8
mairie/école, maternelle/primaire et FAM/MAS, sans refuser aucun des 116
top-1 corrects. Elle ne couvre cependant que 3 des 34 top-1 faux ou ambigus
fiables, sous le minimum préenregistré de cinq. Aucune cohorte fraîche n'est
donc ouverte et la taxonomie ne sera pas enrichie après observation. Aucun
modèle ni seuil n'a changé et le test final reste fermé. Rapport :
`reports/v9/v4_9_site_function_guard_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_9_site_function_retrospective/30e22eae11620538`.
*(commits GitHub : contrat `169d9cf`; taxonomie `a311306`; entrées épinglées
`49832e4`; évaluateur `67b2cb5`)*

La V4.8 se termine avec le verdict final **`STOP_RETRAIN`**. L'ouverture
random unique invalide `HARD_W1` : 47/52 AUTO, mais seulement 44 corrects,
soit trois erreurs et 93,617 % de précision observée. Le baseline gelé fait
43/45 = 95,556 % avec deux erreurs. Le winner automatise deux
`TOP1_WRONG` et l'unique `AMBIGUOUS`; les deux gates de sécurité échouent.
Les trois faux AUTO ont des scores de 0,980 à 0,999 et confondent la fonction
exacte de sites très proches : mairie/école, maternelle/primaire et FAM/MAS.
Ce n'est donc pas un simple problème de seuil. Le registre global empêche
toute réouverture du random ; le test final reste fermé et aucun modèle n'est
promu. Rapport : `reports/v9/v4_8_random_holdout_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout/f1ac35f4f7450b6a`.
*(commits GitHub : contrat `b738ec5`; ouvreur `685ebae`; préflight
`ba4377f`)*

Le développement V4.8 retient **`HARD_W1`** et autorise l'ouverture unique
du random avec le statut **`GO_RANDOM_OPEN_V48`**. Sur 94 cas difficiles hors
pli, il rejette 23/25 mauvais top-1 contre 13/25 pour `BASE_REFIT`, soit dix
erreurs supplémentaires, tout en gardant 58/68 bons AUTO contre 61/68
(-4,412 points, limite -5). Sur le dev historique effectif, il produit
1 184/1 186 AUTO corrects = 99,831 % observés et 81,680 % de couverture,
sans erreur supplémentaire. Le modèle complet et son seuil
`0.3617231974526733` sont gelés. Aucun random n'a été lu ou scoré et le test
final est resté fermé. Rapport :
`reports/v9/v4_8_acceptor_development_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_acceptor_development/f2ea5be7c1a40647`.
*(commits GitHub : contrat/partitions `a15dd07`; runner `3f4671b`;
correctif de lecture `dab961d`)*

La V4.8 a préenregistré puis gelé ses partitions avant tout score accepteur.
Sur 98 labels ciblés fiables, 94 restent évaluables hors pli : 68 top-1
corrects, 25 mauvais et un ambigu. Quatre autres cas fiables sont
`hard_dev_locked` et seront seulement descriptifs. Les 57 cas random sont
tous scellés, leurs cibles sont absentes de l'artefact de partition et 48
scènes historiques reliées ont été exclues. Le fit V4.1 réellement éligible
est bien de 5 545 scènes, pas 5 547 ; le dev effectif futur en contient
1 452 après isolement random. Aucun modèle n'a été chargé, scoré ou entraîné
et le test final est resté fermé. Rapport :
`reports/v9/v4_8_acceptor_partition_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_8_acceptor_partitions/1c78764d5263afca`.
*(commits GitHub : contrats `f56472b`, `1ca9648`, `b63f383`; constructeur
`6bb8518`; correctifs de préflight `08018f9`, `eedac96`)*

La V4.7 a réadjudiqué les 37 top-1 ayant dérivé entre V4.4 et le stack
courant V4.2-B + ranker A, sans transporter l'ancien verdict. Chaque preuve
publique a été téléchargée, archivée et contrôlée par des faits
préenregistrés ; une décision fiable exige toujours au moins deux groupes
indépendants, dont le registre officiel. Vingt-trois nouveaux verdicts sont
fiables (huit `TOP1_CORRECT`, quatorze `TOP1_WRONG`, une `AMBIGUOUS`) et
quatorze restent `UNRESOLVED`. Le corpus courant atteint exactement 150/172
labels fiables, dont 52/57 aléatoires, 28 négatifs ciblés et six négatifs
aléatoires. Tous les gates préenregistrés passent. Verdict :
**`GO_ACCEPTOR_FEASIBILITY`**. Il autorise une expérience V4.8 hors test, pas
un déploiement ni une revendication à 99,8 %. Aucun modèle n'a été entraîné et
le test final est resté fermé. Rapport :
`reports/v9/v4_7_current_top1_adjudication_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_7_current_adjudications/4cc5420fb5da0683`.
*(commits GitHub : contrat `25b881b`, docket `6af0e45`, registre `b85daf7`,
adjudication `bdfbadc`; 324 tests passants)*

La V4.6 a reconstruit deux fois, avec caches séparés, les pools V4.2-B des
7 003 requêtes historiques puis comparé le ranker A gelé à un ranker B
réentraîné sur ces pools. Les deux datasets contiennent exactement 698 991
candidats et partagent le même hash de contenu ; Recall@100 vaut 100 % sur fit
(4 666/4 666) et dev (1 217/1 217), sans doublon, candidat fermé, pool >100 ou
positif injecté. B atteint 1 216/1 217 = 99,918 % Hit@1 SIRET, contre
1 213/1 217 = 99,671 % pour A : trois corrections, zéro régression. Le gain
n'atteint toutefois ni les quatre corrections minimales, ni une borne
bootstrap strictement positive, ni McNemar `p<0,05` (`p=0,25`). Verdict
contractuel : **`KEEP_RANKER_A`**. B n'est pas promu. Aucun accepteur, seuil,
label V4.4/random ou test final n'a été utilisé. Rapport :
`reports/v9/v4_6_aligned_ranker_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_a/301b24f47820f992`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_b/301b24f47820f992`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_6_aligned_ranker/421f2cd0cc436af7`.
*(commits GitHub : contrat `acfd4d2` et `70df9c9`, builder `a9439ed`,
correctif instrumentation `458dd97`, évaluateur `f2b6b9c`, `c94d100` et
`8b835aa`, rapport `67e9e76`; 316 tests passants)*

La V4.5 a vérifié si les labels V4.4 pouvaient être transportés vers les
scènes réellement produites par le retrieval V4.2-B et le ranker V4.1 gelé.
Verdict : **`PIVOT_SCENE_DRIFT`** et `training_authorized=false`. Sur 172
dossiers, 135 seulement conservent le même top-1 et 37 dérivent. Les gates
échouent avec 46/53 labels aléatoires fiables compatibles, 2/6 négatifs
aléatoires, 16/37 `TOP1_WRONG` ciblés et 1/5 `AMBIGUOUS` ciblé. Seul le
minimum des `TOP1_CORRECT` ciblés passe avec 64/67. Aucun accepteur n'a été
chargé, aucun seuil calculé, aucun modèle entraîné et le test final est resté
fermé. Rapport : `reports/v9/v4_5_scene_compatibility_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_5_hard_scenes/21f8c0b0b172b907`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/gates/v4_5_scene_compatibility/5c8b87fd8e226157`.
*(commit GitHub : `5c1343e`; 296 tests passants)*

La V4.4 d'adjudication autonome est terminée. Les lots A–R et les
contradictions connues couvrent exactement les 172 `AUTO_MATCH` V4.3 : 162
décisions sont validées par au moins deux groupes de preuves indépendants,
dont 114 `TOP1_CORRECT`, 42 `TOP1_WRONG` et six `AMBIGUOUS`; dix restent
`UNRESOLVED`. Cinquante-trois décisions validées appartiennent au tirage
aléatoire. Verdict contractuel : **`STOP_AUTONOMOUS_LABELING`**. Les seuils
correct et random sont franchis, mais la population entière ne contient que 42
erreurs prouvées pour un minimum préenregistré de 50. Il est impossible de
combler les huit manquantes sans fabriquer des erreurs, abaisser le seuil
après observation ou ouvrir prématurément les `REVIEW`. Aucun modèle n'est
modifié. L'adaptateur recalcule les pools top-10
figés, refuse les hashes de preuve inexacts, interdit l'injection d'un positif
et reproduit les tables canoniques à partir des JSON revus. Les lots ont été
corrigés contre les archives réelles, notamment leurs hashes et les groupes
d'indépendance. La première passe est maintenant bloquée en code aux seuls
`AUTO_MATCH` V4.3 ; une tentative de sélectionner prématurément des `REVIEW`
est refusée. Rapport :
`reports/v9/v4_4_adjudication_gate_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_official_evidence/87983e83c11f5284`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_evidence_facts/7ec4f63e1a22b082`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_sector_evidence/3149124f69dd7b1f`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_sector_facts/6a08bff403154884`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/320fe62322e14d25`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/70c65679dfb2c82d`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/1e2c68337408c453`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/925ef3f8ef3f3a4a`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/2bfdc46480e52784`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_gate/9fb43b4f7bb0919a`.
*(commits GitHub : lots vérifiés `f2aeec0`, adaptateur canonique `c6cb686`,
rapport `a66499e`, garde AUTO `4930521`, invariance `12a088b`, lots F/H
`d644e54`, G `1509841`, I `0e04eb3`, L `617c73c`, J/K `62669ac`, M
`d72c7d8`, N `30617b5`, O/P `65feba0`, Q/R `22f8ba2`; 283 tests passants)*

La V4.3 a transformé les 542 cas non résolus en une file d'adjudication
complète : 172 AUTO et 370 REVIEW, dont 144 cas du tirage aléatoire. La
première population AUTO est maintenant entièrement consommée par V4.4. Les
370 `REVIEW` restent fermés : ils ne peuvent pas être utilisés pour contourner
le gate AUTO. Aucun signal seul n'est converti artificiellement en vérité.

Le « gold standard » historique ne résout pas le problème. Il a été construit
en gardant le SIRET CRM si sa commune ou son CP correspondait à SIRENE, sans
validation humaine. Il recouvre 313/542 cas difficiles. Sur les 172 AUTO,
116 y figurent et 40 top-1 contredisent ce SIRET historique. Verdict
**`PIVOT_VALIDATION`**, statuts **`NO_RETRAIN`** et **`STOP_DEPLOYMENT`**.
Un premier lot de 250 dossiers avec preuves complètes est prêt dans l'artefact
V4.3. Rapport : `reports/v9/v4_3_hard_labels_results.md`.

Le correctif retrieval V4.2 franchit le gate représentatif :
**242/242 = 100 % Recall@100** sur les `MATCH_EXACT` provisoires figés, dont
91/91 dans le tirage aléatoire. Aucun pool ne dépasse 100, aucun candidat fermé
n'est produit, aucune vérité n'est injectée et aucun des 237 anciens succès ne
régrèsse. Les cinq pertes V4.1 sont récupérées aux rangs 1, 2, 1, 3 et 1.

Le correctif ne change ni TF-IDF, ni RRF, ni ranker, ni accepteur. La variante
B exploite le SIRET/SIREN d'entrée comme indice et la barrière finale prend
désormais l'état administratif dans le snapshot SIRENE complet de 42 322 035
établissements, au lieu du magasin rapide incomplet de 14 378 332 candidats.
Verdict **`GO_HARD_LABELS`** : le prochain travail est la constitution de
vrais cas difficiles représentatifs avant tout réentraînement. Le statut
production reste **`STOP_DEPLOYMENT`**. Rapport :
`reports/v9/v4_2_retrieval_integrity_results.md`.

L'audit représentatif V4.1 invalide désormais l'extrapolation des scores dev
au CRM réel. Sur 250 lignes tirées aléatoirement, la preuve déterministe ne
conclut que 106 cas = 42,4 % (91 exacts, 15 ambigus, 144 non résolus), alors
que V4.1 automatise 147 cas. Cinq contradictions AUTO manifestes ont été
documentées ; même en supposant tous les autres AUTO corrects, elles bornent
provisoirement la précision à 142/147 = 96,60 %. Ce n'est pas une estimation
certifiée : labels et contradictions restent `AI_PROVISIONAL`.

Le retrieval A ne conserve que 237/242 = 97,934 % des vérités exactes
provisoires de l'audit. B et C remontent à 240/242 = 99,174 % sans dépasser
100 candidats : elles récupèrent trois lignes grâce au SIRET/SIREN d'entrée.
Les deux pertes restantes sont trouvées par le sparse puis supprimées par le
magasin global d'état incomplet. Décision contractuelle **`PIVOT_LABELS`**,
décision opérationnelle **`STOP_DEPLOYMENT`**. Prochaine action : index
SIRET→état complet depuis le snapshot autoritaire, retrieval B, puis vrais
labels sur les cas difficiles avant tout réentraînement. Rapport :
`reports/v9/v4_1_representative_audit_results.md`.

La V4.1 actif-courant est maintenant implémentée et exécutée entièrement en
local. Le gate retrieval dev retient la variante A avec 305/305 bons SIRET à
Top-100, zéro candidat fermé et 872,6 ms de latence p95. Le dataset canonique
contient 7 003 requêtes et 698 428 paires, sans positif injecté et sans aucun
ID des tests consommés. Le ranker R1 atteint 99,918 % Hit@1 exact sur le dev ;
l'accepteur brut atteint 99,832 % de précision observée et 81,593 % de
couverture sur ce même dev.

Le shadow complet a ensuite scoré exactement 19 025 lignes autorisées et
produit 10 292 `AUTO_MATCH` = 54,097 % et 8 733 `REVIEW`, sans modifier le
CRM. Sa latence p95 est de 982,0 ms. Comme ce corpus n'est pas un test
indépendant et ne porte pas de nouveaux labels, aucune précision shadow n'est
publiée. Verdict de phase : **`PIVOT_CERTIFICATION`**. L'architecture est
techniquement validée, mais un `GO` production exige un prochain snapshot CRM
réellement nouveau et audité sans retuning. Rapport :
`reports/v9/v4_1_shadow_results.md`.

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

L'accepteur V4 franchit ensuite son gate : sur les 189 scènes de la moitié
`threshold`, il automatise **149/189 = 78,836 %** avec **149/149 correctes**.
Il rejette les 31 ambiguës, les quatre erreurs du ranker et cinq bons cas
incertains. Verdict : **`GO_HOLDOUT_V4`**. Les six variantes testées sont à
égalité ; le winner logistique + isotonic vient du tie-break déterministe, pas
d'une supériorité démontrée. Le holdout n'a toujours pas été lu. Rapport :
`reports/v9/v4_acceptor_e2_results.md`.

Le holdout V4-Fresh a ensuite été ouvert une seule fois après gel complet.
Le retrieval atteint **302/302 = 100 % Recall@100 exact**, et le ranker
**296/302 = 98,013 % Hit@1 exact**. L'accepteur automatise 282/354 scènes,
mais commet deux erreurs : précision **280/282 = 99,291 %**, sous le gate de
99,8 %. La qualification stricte ne couvre par ailleurs que 302/1 345 =
22,454 % de la source. Verdict final : **`PIVOT`**, sous-verdict
**`TECHNICAL_PIVOT`**. Le retrieval et le ranker sont validés ; la correction
porte sur le routage des scènes ambiguës, l'état actif/fermé du top1 et la
calibration saturante. Le holdout est désormais consommé. Rapport :
`reports/v9/v4_final_holdout_results.md`.

## Actions terminees (fenetre recente)
- **V4.8 arrêtée par le random unique** : `HARD_W1` commet trois erreurs sur
  47 AUTO, contre deux sur 45 pour le baseline. Verdict `STOP_RETRAIN`; aucun
  shadow ni déploiement. Les erreurs sont des confusions de fonction de site
  malgré des scores 0,98–0,999. Le random est définitivement consommé et le
  test final est resté fermé. Rapport :
  `reports/v9/v4_8_random_holdout_results.md`. *(commits GitHub :
  `685ebae`, `ba4377f`)*
- **Winner accepteur V4.8 gelé avant random** : `HARD_W1` rejette 23/25
  erreurs difficiles hors pli contre 13/25 pour le refit de base, en perdant
  trois bons AUTO. Le dev historique reste à deux erreurs et gagne deux bons
  AUTO. Statut `GO_RANDOM_OPEN_V48`; seuil gelé
  `0.3617231974526733`. Le random et le test final restent fermés. Rapport :
  `reports/v9/v4_8_acceptor_development_results.md`. *(commits GitHub :
  `3f4671b`, `dab961d`)*
- **Partitions V4.8 gelées avant modélisation** : 94 ciblés fiables sont
  disponibles en cinq folds groupés, avec exactement 25 erreurs et une
  ambiguïté. Les 57 random sont scellés sans cible exposée ; 48 scènes
  historiques reliées sont exclues. Le prochain gate compare uniquement des
  accepteurs logistiques à 80 features, avec seuil propre à chaque modèle
  OOF. Aucun score random ni test final n'a été consulté. Rapport :
  `reports/v9/v4_8_acceptor_partition_results.md`. *(commits GitHub :
  `b63f383`, `6bb8518`, `08018f9`, `eedac96`)*
- **Gate V4.7 franchi sur les scènes courantes** : 37/37 top-1 dérivés ont
  été traités ; 23 portent désormais un label fiable et 14 restent
  `UNRESOLVED`. Le corpus agrégé atteint 150/172 labels fiables, 52/57
  aléatoires, 28 négatifs ciblés et six négatifs aléatoires. Zéro ancien
  verdict a été transporté vers un autre SIRET. Verdict
  `GO_ACCEPTOR_FEASIBILITY`; V4.8 doit être préenregistrée avant tout
  entraînement et le test final reste fermé. Rapport :
  `reports/v9/v4_7_current_top1_adjudication_results.md`. *(commit GitHub :
  `bdfbadc`; 324 tests passants)*
- **Population AUTO V4.4 épuisée sans quota fabriqué** : les 172/172
  `AUTO_MATCH` ont été audités. Bilan : 114 `TOP1_CORRECT`, 42
  `TOP1_WRONG`, six `AMBIGUOUS`, dix `UNRESOLVED`, 162 labels acceptor et 53
  cas random validés. Les minima correct/random passent, mais le minimum de 50
  erreurs est impossible puisque la population entière n'en contient que 42
  prouvées. Verdict `STOP_AUTONOMOUS_LABELING`; aucun réentraînement sous le
  contrat V4.4 et aucun `REVIEW` ouvert. Le contrat expérimental V4.5 a été
  préenregistré avant entraînement et reste bloqué tant qu'un pivot explicite
  n'est pas adopté. Rapport :
  `reports/v9/v4_4_adjudication_gate_results.md`. *(commits GitHub : I
  `0e04eb3`, L `617c73c`, J/K `62669ac`, M `d72c7d8`, N `30617b5`, O/P
  `65feba0`, Q/R `22f8ba2`, contrat V4.5 `70cf70f`; 283 tests passants)*
- **Sous-gate random V4.4 franchi avec les lots F–H** : les 36 nouveaux cas
  sont tous des `AUTO_MATCH` V4.3 et ne chevauchent aucun dossier antérieur.
  Ils ajoutent 20 `TOP1_CORRECT`, 14 `TOP1_WRONG`, 13 random validés et deux
  `UNRESOLVED`. Le corpus atteint 89 cas, dont 81 validés : 55 corrects, 25
  incorrects et 32 random. Le gate reste `PIVOT_MORE_EVIDENCE`, avec déficits
  réduits à 20 corrects et 25 incorrects. Aucun SIRET alternatif non doublement
  prouvé n'a été créé. *(commits GitHub : F/H `d644e54`, G `1509841`,
  invariance des lots `12a088b`; 282 tests passants)*
- **Lots V4.4 A–E rendus canoniques et gate recalculé** : 48 dossiers
  sectoriels ont été reliés à la queue V4.3 et aux vrais pools top-10 du shadow,
  puis combinés aux cinq contradictions déjà canoniques. Les erreurs de hashes
  et de taxonomie de sources ont été détectées par recomputation puis corrigées
  contre les archives. Corpus final à ce stade : 53 cas, 47 validés, 35
  `TOP1_CORRECT`, 11 `TOP1_WRONG`, une `AMBIGUOUS`, six `UNRESOLVED` et 19
  random validés. Verdict `PIVOT_MORE_EVIDENCE`; aucun réentraînement.
  La passe initiale refuse désormais en code toute ligne V4.3 `REVIEW`.
  Rapport : `reports/v9/v4_4_adjudication_gate_results.md`.
  *(commits GitHub : données `f2aeec0`, adaptateur `c6cb686`, rapport
  `a66499e`, garde AUTO `4930521`; 282 tests passants)*
- **Premier gate V4.4 canonique publié** : les cinq contradictions ont été
  reliées aux pools top-10 réellement archivés dans le shadow, au bundle et à
  la signature retrieval gelés, puis validées par recomputation. Résultat :
  quatre `TOP1_WRONG` éligibles accepteur, zéro cible ranker et un
  `UNRESOLVED`. Les cinq appartiennent au tirage aléatoire V4.3 ; une première
  perte de cette provenance a été détectée, corrigée et les artefacts ont été
  reconstruits. Verdict partiel `PIVOT_MORE_EVIDENCE`, avec déficits 75
  corrects, 46 incorrects et 26 random. *(commits GitHub : dossiers
  `7c1a6fd`, validateur `edbfbfe`, adaptateur `ef2df25`, correction provenance
  `3a74b8a`, gate `17bf904` ; 271 tests passants)*
- **Preuves sectorielles V4.4 collectées sans dépense** : 117 observations sur
  52 dossiers, auprès des producteurs UAI, FINESS, Agence Bio et ADEME.
  UAI retrouve 27/29 identifiants, FINESS 33/33, Bio 10/10 et RGE 45/45
  couples qualification/SIRET. Les 115 réponses positives portent toutes le
  même SIRET explicite que l'observation ; les deux UAI absents restent sans
  interprétation. La file de priorité contient 14 dossiers avec signal
  sectoriel non attaché au top-1, 37 attachés au top-1, un identifiant non
  résolu et 120 sans signal sectoriel. Zéro label créé automatiquement.
  *(commits GitHub : collecte `332094d`, faits `428942b`)*
- **Faits V4.4 dérivés sans faux consensus** : les 440 vues de l'API
  officielle ont été ramenées à une seule famille de source. Les 172 dossiers
  AUTO portent désormais 53 faits auditables, mais zéro conclusion de
  correction et zéro label entraînable. Un accord SIRET direct, une recherche
  nom + géographie, le score ou l'adresse ne peuvent pas créer une vérité.
  L'ordre de réentraînement est également figé : accepteur logistique d'abord,
  avec retrieval et ranker gelés ; ranker ensuite uniquement pour les erreurs
  ayant un SIRET alternatif exact prouvé et naturellement présent dans le
  pool. Rapport :
  `reports/v9/v4_4_evidence_validated_retraining_design.md`. *(commits GitHub :
  faits `9274399`, design `1d4d0f1` ; 6 tests ciblés passants)*
- **Preuves officielles V4.4 collectées pour les 172 AUTO difficiles** :
  politique autonome gelée, sans validation demandée à l'utilisateur. API
  Recherche d'entreprises interrogée à débit limité : 440/440 réponses HTTP
  200, 325 requêtes avec résultat. Les réponses brutes, URLs et dates sont
  conservées sur le SSD ; elles ne valent pas encore adjudication et aucun
  entraînement n'est ouvert. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_official_evidence/87983e83c11f5284`.
  *(commits GitHub : contrat `ede441b`, collecteur `341acf2` ; 223 tests
  passants)*
- **File de labels difficiles V4.3 construite** : population figée de 542
  `UNRESOLVED`, sans suppression ; 172 AUTO, 370 REVIEW, 144 random. Priorités :
  cinq contradictions connues, 35 AUTO adresse-seule, 28 AUTO en désaccord
  avec un input actif, 104 autres AUTO, puis les REVIEW. Un lot opérationnel
  de 250 dossiers réunit CRM, top-1 et preuves SIRENE. Zéro nouveau label
  entraînable : verdict `PIVOT_VALIDATION`, aucun réentraînement autorisé.
  Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_3_hard_labels/0f832305ab199267`.
  Le disque interne étant saturé, seuls les caches Python/pytest régénérables
  ont été supprimés ; aucun artefact métier n'a été touché. *(commits GitHub :
  contrat `a2232bf`, builder `c3c5944`, correction normalisation `3388649`,
  lot 250 `b16ce8b`, rapport `c420c26` ; 221 tests passants)*
- **Intégrité retrieval V4.2 validée sans GPU** : contrat figé avant le
  correctif ; variante B et état autoritaire lu dans le snapshot complet.
  Résultat 242/242 = 100 % Recall@100, random exact 91/91, zéro fermé, zéro
  injection, zéro vérité absente du snapshot et zéro régression sur les 237
  anciens succès. Les cinq misses sont tous récupérés. Latence p50 455,1 ms,
  p95 2 878,6 ms sur cet échantillon difficile. Verdict `GO_HARD_LABELS`,
  sans autorisation de réentraînement ou de déploiement. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_2_retrieval_integrity_7c4b957`.
  *(commits GitHub : contrat `c33d3e0`, source d'état `48ed90b`, évaluateur
  `7c4b957`, rapport `2d7070e` ; 218 tests passants)*
- **Audit représentatif V4.1 exécuté en aveugle** : échantillon figé de 800
  lignes dont 250 aléatoires ; preuves construites sans décision, score,
  prédiction ni rang modèle ; 242 `MATCH_EXACT`, 16 `AMBIGUOUS` et 542
  `UNRESOLVED` provisoires. Le random n'est mécaniquement conclusif qu'à
  42,4 %. Retrieval A : 237/242 ; B/C : 240/242. Deux pertes restantes
  viennent du magasin d'état incomplet. Cinq contradictions AUTO nettes
  réfutent la sécurité extrapolée depuis le dev. Verdict `PIVOT_LABELS` /
  `STOP_DEPLOYMENT`; aucun modèle ni seuil modifié. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative/e06cf0d79849aad4`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_evidence/e696f22d68c0210f`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_summary/2d18ef172f32aefc`.
  *(commits GitHub : contrat/échantillon `015d718`/`5f8ea00`, preuves
  `361c138`/`edf0858`, synthèse `771be6b`, rapport `17465cd` ; 211 tests
  passants)*
- **V4.1 actif-courant exécutée en shadow local** : gate retrieval A à 305/305
  sur dev avec 100 candidats maximum et zéro fermé ; dataset de 7 003 requêtes
  et 698 428 paires ; ranker R1 à 99,918 % Hit@1 dev ; accepteur brut à
  99,832 % de précision observée et 81,593 % de couverture dev. Le shadow
  atomique contient 19 025 décisions, 10 292 AUTO et 8 733 REVIEW, zéro exclu
  scoré, zéro écriture CRM et aucune revendication de précision. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/retrieval_v41_dev_feede27`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/shadow/v4_1/runs/v41_shadow_f1058826_20260727_v3`.
  Verdict `PIVOT_CERTIFICATION`; 206 tests passants. *(commits GitHub :
  contrat `eea75f2`, retrieval `a599e4a`/`85f7674`/`993e088`, modèles
  `f158da2`/`942a443`/`c4ffb2a`/`d86f6f6`, inventaire et dataset
  `af18779`/`ab13fb4`/`9fd30d8`/`feede27`, runner
  `41cbc0e`/`8e96961`/`cc5dec1`/`9a322bc`)*
- **Évaluation finale V4 exécutée une seule fois** : autorisation gelée
  `7dbd5527374ca0d4`, zéro chevauchement SIREN, zéro injection et 100 candidats
  maximum. Résultats : Recall@100 302/302 = 100 %, Hit@1 exact 296/302 =
  98,013 %, AUTO 282/354 = 79,661 %, précision AUTO 280/282 = 99,291 %.
  Deux causes simples : un ancien SIRET PALAFIS fermé mais textuellement
  parfait est automatisé à la place du SIRET actif ; une scène ELGEA déjà
  qualifiée `AMBIGUOUS` avec 80 SIRET actifs est automatisée. La calibration
  isotonic attribue exactement 1,0 aux 282 AUTO. Verdict `PIVOT` /
  `TECHNICAL_PIVOT`. Le premier rapport `STOP` est conservé : il inversait
  deux booléens d'intégrité. Sa correction n'a relu ni le holdout ni les
  modèles. Suite complète à 148 tests passants. *(commits GitHub : contrat
  `fb6a20c`, runner `8cc9bfa`, correction instrumentale `aead6f5`, rapport
  `4aade83`)*
- **Accepteur V4 validé avant holdout** : dataset de 7 215 scènes
  (6 054 exactes, 1 161 ambiguës), 721 007 paires et zéro `UNRESOLVED`.
  Les exactes train utilisent les prédictions OOF ; les ambiguës étaient
  entièrement absentes du fit ranker. Sur le demi-dev `threshold`, le bundle
  retenu produit 149/189 AUTO = 78,836 %, zéro erreur observée, contre un gate
  de 25 % à 99,8 %. Les six variantes arrivent au même point. Verdict
  `GO_HOLDOUT_V4`, sans lecture du holdout. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/acceptor_v4/2b8a9c994e0944be`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8`.
  Suite complète à 145 tests passants. *(commits GitHub : contrat `9a22fd8`,
  préparation `af5ce0b`, builder/train `9ec88c8`, rapport `ff1eea4`)*
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
- `docs/v4_acceptor_e2_contract.md` *(commit GitHub : `9a22fd8`)*
- `docs/v4_final_holdout_contract.md` *(commit GitHub : `fb6a20c`)*
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
- `scripts/prepare_v4_ambiguous_retrieval.py`,
  `tests/test_v4_ambiguous_retrieval.py` *(commit GitHub : `af5ce0b`)*
- `scripts/build_v4_acceptor_dataset.py`,
  `tests/test_v4_acceptor_dataset.py`, `scripts/train_v9_acceptor.py`,
  `src/xgb_matcher/v9_scene.py` *(commit GitHub : `9ec88c8`)*
- `scripts/freeze_v4_final_holdout.py`,
  `scripts/prepare_v4_final_holdout.py`,
  `scripts/evaluate_v4_final_holdout.py`,
  `tests/test_v4_final_holdout.py` *(commit GitHub : `8cc9bfa`)*
- `scripts/repair_v4_final_verdict.py` et correction des booléens du runner
  *(commit GitHub : `aead6f5`)*
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
- `reports/v9/v4_acceptor_e2_results.md` *(commit GitHub : `ff1eea4`)*
- `reports/v9/v4_final_holdout_results.md` *(commit GitHub : `4aade83`)*
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
- V4.10b est close sans modèle promu. Le prochain travail autorisé est un
  nouveau contrat d'architecture alignant de façon homogène retrieval,
  prédictions ranker hors échantillon et scènes accepteur. Les 94 cas
  difficiles V4.10b sont consommés et ne peuvent plus valider cette
  architecture. Ni random V4.8, ni locked, ni test final ne doivent être
  rouverts.
- Le registre V4.11-A est gelé. Les 225 lignes `UNSEEN` ne doivent pas être
  ouvertes avant gel du candidat V4.11 et ne peuvent servir qu'à un challenge
  descriptif, pas à une preuve représentative.
- Le contrat V4.11-B est gelé. Toute implémentation doit rester aveugle au
  SIRET/SIREN CRM, reconstruire le retrieval avant les labels et respecter
  les ordres ranker 45 / accepteur 80.
- Le garde-fou V4.9 de fonction de site est clos par
  `STOP_SITE_FUNCTION_GUARD`. Il ne doit pas être retouché sur les 172 cas
  consommés.
- Le test final historique et le holdout V4-Fresh ont chacun été lus une fois
  et sont maintenant définitivement fermés à toute nouvelle variante, règle
  ou seuil.
- E1 historique est conservé comme baseline. Le nouveau ranker V4 est validé
  sur `dev_new`, mais aucun modèle produit n'est déployé.
- V4-Fresh a validé définitivement le retrieval V4 et le ranker. V4.8 a
  invalidé le nouvel accepteur sur le random et V4.9 a invalidé le garde-fou
  lexical comme piste assez large. La prochaine étape est un diagnostic
  structuré des 31 erreurs/ambiguïtés non interceptées, sans tuning.

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
  mesure nouvelle ne doit désormais réutiliser `holdout_sealed`, consommé par
  l'évaluation finale `7dbd5527374ca0d4`.
- **Deux erreurs finales explicables** : `AMBIGUOUS` doit être routé `REVIEW`
  avant l'accepteur ; un top1 fermé ne doit pas être automatisé dans la cible
  V4 actif-courant. Ces règles sont des hypothèses post-holdout et exigent une
  nouvelle validation indépendante.
- **Calibration saturante** : l'isotonic retenu par tie-break donne 1,0 aux
  282 AUTO finaux. Ne pas présenter ce score comme une probabilité fiable.
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
| Dataset accepteur V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/acceptor_v4/2b8a9c994e0944be/` |
| Accepteur V4 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8/` |
| Autorisation finale V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/releases/v4_final/7dbd5527374ca0d4/authorization.json` |
| Première évaluation finale V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4/` |
| Verdict final V4 corrigé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4_verdict_repair/` |
| Dataset V4.11 input-blind | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d/` |
| Ranker C V4.11 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/e13eb3ac7498256e/` |
| Scènes accepteur V4.11 validées | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/52ea3faba9a56aff/` |
| Candidat accepteur V4.11 gelé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/9d23bf3deb6b63de/` |
| CRM challenge V4.11 assaini | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_sanitized/1c994c852c10acaf/` |
| Labels challenge V4.11 gelés | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_qualification/4f9ef46516b89ab8/` |
| Audit métier 30 REVIEW V4.12 | `reports/v412_review_adjudication_30.md` |

## Milestone métier V4.12 — 30 REVIEW adjudiqués

- Commit : `fd39303` (`audit: adjudicate 30 V4.12 review dossiers`).
- Les 30 dossiers préenregistrés ont été traités sans réentraînement, GPU,
  service payant ni validation utilisateur.
- Résultat : 27 SIRET exacts exploitables, 3 labels `AMBIGUOUS`, aucun
  `UNRESOLVED`.
- Parmi les 27 cas résolus, le top 1 V4.12 est correct dans 15 cas et faux
  dans 12 cas. Les 12 erreurs ont toutes leur bon candidat dans le pool : 7
  erreurs intra-SIREN, 4 collisions de sociétés ou de groupe à la même
  adresse, et 1 adresse CRM historique.
- Le prochain travail autorisé est une expérience corrective bornée sur
  train/dev réutilisant ces familles d'erreurs. Aucun nouveau gate
  d'infrastructure et aucune ouverture du test final.

## Expérience V4.12-R30 — correction bornée du ranker REVIEW

- Commit : `d0a2c9a` (`experiment: test bounded REVIEW reranking`).
- Sans entraînement, une correction conditionnelle réutilisant deux features
  existantes (`name_sim_max_ul` et `is_siege`) passe de 15 à 25 bons top 1
  sur les 27 labels exacts R30 : 10 corrections, aucune régression et aucune
  ambiguïté convertie en positif.
- La contre-expérience globale est rejetée : elle atteindrait 26/27 sur R30
  mais dégraderait 20 des 116 anciens top 1 fiables.
- La variante bornée, appliquée seulement si le score top 1 initial est au
  moins 2,5, ne dégrade aucun de ces 116 cas. Ce contrôle est un écran de
  développement consommé, pas une validation indépendante ni une
  autorisation de production.
- Verdict : `GO_EXPAND_LABELS_BEFORE_TRAINING`. Les deux erreurs restantes
  sont GHNE Longjumeau (fonction du site non portée par le nom légal) et INLOG
  (correction bloquée par le seuil prudent).
- Rapport : `reports/v412_review_rerank_spike.md`. Labels machine :
  `reports/v412_review_adjudication_labels.csv`.

## Prochaines etapes
1. Ne plus réutiliser le test historique, le holdout V4-Fresh, le random V4.8
   ni les 172 cas V4.9 pour sélectionner ou valider une variante.
2. Produire une taxonomie descriptive, sans règle de décision, des 31 erreurs
   ou ambiguïtés fiables que V4.9 ne refuse pas. **Terminé en V4.10.**
3. Préenregistrer puis construire une matrice accepteur unique qui conserve
   les relations entrée/candidat, l'état, la provenance, la forme juridique,
   l'activité/fonction, les interactions nom/adresse et la concurrence
   intra-SIREN complète. **Terminé : `GO_TRAIN_V410`.**
4. Comparer une régression logistique et un XGBoost peu profond sans modifier
   le retrieval V4.2-B ni le ranker A, avec OOF par composante et reproduction
   exacte du baseline. **Terminé : `PIVOT_STRUCTURED_FEATURES`.**
5. Toute décision de promotion exigera une population fraîche,
   indépendante et disjointe ; le test final reste fermé.
6. Conserver séparément le chantier qualification/réparation CRM : la
   couverture source 22,454 % reste très loin du gate de 80 % et ne se corrige
   pas par le retrieval.
7. Le challenge descriptif V4.11 est consommé et conclut
   `PIVOT_ACCEPTOR_EVIDENCE_GATE`. Préenregistrer V4.12 sur les anciennes
   populations uniquement : garde multi-SIREN forte, nouvelles features
   d'unicité, comparaison garde seule puis accepteur. Geler le candidat avant
   tout nouvel export CRM indépendant, indispensable à une décision produit.
8. La correction minimale du classement est terminée avec le verdict
   `GO_EXPAND_LABELS_BEFORE_TRAINING` (`d0a2c9a`). **Priorité active** :
   sélectionner parmi les 249 REVIEW historiques restants ceux dont cette
   correction change le top 1, puis les adjudiquer avec preuves traçables.
   Aucun réentraînement avant ce contre-échantillon ; aucune ambiguïté ne doit
   être transformée en positif.

## Contre-audit V4.12-R53 — changements de top 1

- Commit : `393c52f` (`audit: adjudicate 53 rerank counter-cases`).
- Les 53 dossiers `REVIEW` restants dont la correction exploratoire change le
  top 1 ont été adjudiqués sur le snapshot SIRENE local, les pools réellement
  servis au ranker et des preuves externes traçables pour les collisions,
  transferts et changements d'exploitant.
- Résultat : 50 labels `MATCH_EXACT`, trois `AMBIGUOUS`, aucun `UNRESOLVED`.
  La correction choisit le SIRET exact dans 43/50 cas identifiables, mais
  régresse sur trois top 1 initiaux fiables (`IDEF 86`, `CCI EMERAINVILLE`,
  `AVELIS GROUP`).
- Quatre vérités ne sont ni l'ancien ni le nouveau choix : deux sont absentes
  des 100 candidats (`GROUPE DELAMBRE`, `SIX ARES`) et deux sont présentes
  mais classées 23e et 30e (`CLINIQUE DE TOURNAN`, `ALCYACONSEIL`).
- Verdict : `PIVOT_FROM_RULE_TO_TRAINABLE_SIGNAL`. Le bonus fixe n'est pas
  déployable. Les 50 nouveaux labels exacts peuvent alimenter un entraînement
  borné de développement avec groupes SIREN et prédictions hors échantillon ;
  les trois ambiguïtés restent réservées à l'abstention. Toute promotion exige
  une population indépendante nouvelle et le test final reste fermé.
- Rapports : `reports/v412_review_rerank_counteraudit_first10.md`,
  `reports/v412_review_rerank_counteraudit_53.md` et table machine
  `reports/v412_review_rerank_counteraudit_53.csv`.

## Expérience V4.12-Ranker — apprentissage des labels difficiles

- Commit : `4f57c2f` (`experiment: train hard-label ranker candidate`).
- Les 27 labels exacts R30 et les 50 labels exacts R53 sont compatibles
  bit-à-bit avec les 45 features et les pools V4.11. Les 77 SIREN sont
  disjoints du `fit` historique. Deux positifs absents des 100 candidats
  restent des erreurs end-to-end ; les six ambiguïtés sont exclues du fit.
- Cinq modèles OOF utilisent les folds SIREN gelés. Le baseline obtient 18/77
  bons top 1 sur cette population difficile. Le candidat non pondéré atteint
  60/77 mais régresse sur trois des 1 197 contrôles non adjudiqués : rejet.
- L'ablation figée retient un poids de groupe `0,5` pour les nouveaux cas :
  59/77 bons top 1, 43 corrections, deux régressions difficiles connues et
  zéro régression sur les 1 197 contrôles. Les poids `0,25` et `0,1` sont
  dominés sous cette contrainte.
- L'expérience a été reproduite indépendamment avec prédictions, écran de
  régression et modèle bit-à-bit identiques. Modèle candidat :
  `45f8735382111ee3dc308926bd4883f2c71601cb9e30be72ebb76eba36fd62cd`.
- Verdict : `GO_NEW_INDEPENDENT_VALIDATION`. Il s'agit d'un candidat de
  développement, sans autorisation produit. Prochaine priorité : geler un
  nouveau lot parmi les REVIEW non adjudiqués, avant lecture des vérités, en
  ciblant les désaccords entre baseline et candidat ; le test final reste
  fermé.
- Rapport : `reports/v412_hard_label_ranker_experiment.md`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_hard_label_ranker/bba02575366ebe80`.

## Validation indépendante V4.12-Ranker

- Commit de gel antérieur aux vérités : `c39dfb1` (`audit: freeze independent
  ranker validation docket`). Parmi les 196 REVIEW non adjudiqués, les sept
  désaccords baseline/candidat ont été retenus exhaustivement.
- Commit d'adjudication : `21e87e9` (`audit: adjudicate independent ranker
  validation`). Résultat : six `MATCH_EXACT`, un `AMBIGUOUS`, aucun
  `UNRESOLVED` et aucune vérité absente du pool.
- Le candidat pondéré `0,5` gagne les six dossiers exacts ; le baseline en
  gagne zéro. `PROMOTRANS LYON` reste ambigu car deux établissements actifs
  du même SIREN partagent exactement adresse, date et activité.
- Verdict : `GO_RANKER_CANDIDATE_FOR_ACCEPTOR_DEVELOPMENT`. Ce résultat
  autorise la production de scènes OOF pour un accepteur de développement,
  pas le déploiement du ranker. Six décisions exactes ne prouvent ni la
  précision AUTO à 99,8 %, ni une certification statistique.
- Rapports : `reports/v412_ranker_independent_validation_docket.md`,
  `reports/v412_ranker_independent_validation.md` et labels machine
  `reports/v412_ranker_independent_validation_labels.csv`.

## Expérience V4.12-Stack — ranker candidat + accepteur

- Commit : `931ec52` (`experiment: evaluate V4.12 selective stack`).
- Cinq rankers conjoints produisent des prédictions OOF pour les 5 547 scènes
  fit historiques et les 83 adjudications difficiles. Le fit accepteur compte
  5 630 scènes ; 665 scènes non adjudiquées servent au seuil, 701 à la
  comparaison et les sept décisions indépendantes restent séparées jusqu'à
  la sélection de la famille.
- La logistique compacte est refusée : 619/701 AUTO, une erreur et une
  ambiguïté automatisée. Le XGBoost monotone est sélectionné avec 592/701
  AUTO (84,45 %), zéro erreur observée et zéro ambiguïté AUTO au seuil
  `0,9892662764`.
- Sur le docket difficile indépendant, le ranker a six top 1 exacts et une
  ambiguïté, mais l'accepteur route les sept cas en REVIEW : couverture 0 %.
  La sécurité sans aucune décision n'est pas un gain North Star.
- Verdict : `PIVOT_ACCEPTOR_COVERAGE`. Les sept cas indépendants sont
  consommés. La prochaine expérience peut pondérer les scènes difficiles et
  doit être sélectionnée uniquement sur OOF/développement consommé. La
  qualification ultérieure a établi que les 189 REVIEW restants ont déjà
  contribué aux lots de seuil/comparaison via leurs labels historiques : ils
  peuvent être ré-adjudiqués, mais ne constituent pas une preuve indépendante
  de l'accepteur.
- Rapport : `reports/v412_ranker_acceptor_stack.md`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_ranker_acceptor_stack/f6d3c21bd8a8359e`.

## Expérience V4.12-Accepteur — poids des cas difficiles

- Commit : `0cf3f09` (`experiment: pivot V4.12 acceptor weighting`).
- Les poids `1`, `5`, `10`, `20` et `50` ont été comparés sur les 83 scènes
  difficiles avec décisions accepteur hors apprentissage en cinq plis SIREN.
  Les sept adjudications indépendantes précédentes sont exclues et le test
  final reste fermé.
- Les cinq variantes n'automatisent chacune qu'un seul dossier difficile sur
  83, sans erreur ni ambiguïté AUTO. La pondération n'apporte donc aucun gain
  par rapport au poids `1`. Sur les 701 contrôles consommés, les variantes
  produisent 593 à 597 AUTO sans erreur observée, mais ce petit écart ne résout
  pas le goulot métier étudié.
- Une ablation locale retirant les contraintes monotones reste également à
  zéro ou un AUTO difficile. Le problème n'est pas un simple réglage de poids
  ou une contrainte isolée.
- Verdict : `PIVOT_ACCEPTOR_FEATURES`. Le ranker candidat classe correctement
  60/77 dossiers exacts difficiles, tandis que l'accepteur n'en laisse passer
  qu'un. La prochaine expérience autorisée doit tester, sur développement déjà
  consommé, des preuves relationnelles explicites : avantage du nom légal sur
  le nom d'établissement, écart avec le second candidat et concurrence entre
  sites/SIREN. Aucun nouveau docket indépendant ne doit être ouvert avant un
  gain hors apprentissage réel.
- Rapport : `reports/v412_acceptor_hard_weight.md`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_hard_weight/a9bdb09ea504194e`.

## Expérience V4.12-Accepteur — preuves relationnelles

- Commit : `adb1507` (`experiment: test V4.12 acceptor relational evidence`).
- Deux variantes bornées ajoutent respectivement deux puis quatre relations
  explicites entre ressemblance au nom légal, ressemblance au nom
  d'établissement, écart ranker et concurrence entre sites. Ranker, pools,
  poids `10`, splits et seuil restent inchangés.
- La référence produit 597/701 AUTO sur les contrôles et 1/83 AUTO difficile
  OOF. Les variantes relationnelles produisent 594 et 595 AUTO contrôles, mais
  restent toutes deux à 1/83 AUTO difficile ; aucune erreur ni ambiguïté n'est
  automatisée dans ces comparaisons.
- Verdict : `PIVOT_ACCEPTOR_FEATURES`. Les deux explications simples issues de
  l'audit métier — pondération et relations nom légal/établissement — sont
  falsifiées comme leviers de couverture. Ajouter d'autres combinaisons sur
  les mêmes 83 dossiers risquerait le sur-ajustement.
- Le candidat historique franchit le gate global de développement, mais il
  n'existe plus de preuve produit locale vierge : test final et challenge sont
  consommés, et les 189 REVIEW restants ont déjà contribué au seuil ou à la
  comparaison par leurs labels historiques. Une conclusion produit exige une
  nouvelle cohorte CRM avec preuve SIRET indépendante du matching.
- Rapport : `reports/v412_acceptor_relational_features.md`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_relational_features/81a976729f2140de`.

## Audit métier V4.12 — 30 REVIEW complémentaires

- Docket des cinq premiers gelé avant vérité : commit `f477828` (`audit:
  freeze next five V4.12 REVIEW cases`). Adjudication : commit `7866e34`
  (`audit: adjudicate five remaining V4.12 REVIEW cases`).
- Docket des 25 suivants gelé avant vérité : commit `ff54e30` (`audit: freeze
  next 25 V4.12 REVIEW cases`). Adjudication : commit `0215151` (`audit:
  adjudicate 25 additional V4.12 REVIEW cases`).
- Bilan métier : 27 `MATCH_EXACT` fiables, trois `AMBIGUOUS`, zéro
  `UNRESOLVED`. Les 27 labels exacts sont utilisables pour le développement ;
  26/30 anciennes qualifications doivent être corrigées.
- Le ranker avait déjà le bon SIRET en top 1 dans 24/27 cas exacts. Les trois
  erreurs réelles sont un site hospitalier après restructuration, CESI
  Association confondu avec CESI SAS à la même adresse, et Institut Lemonnier
  confondu avec sa filiale CFC co-localisée. Les trois ambiguïtés concernent
  LG Alès après cession, Constructys sans date CRM et Promotrans avec deux
  entités actives sous la même marque au même lieu.
- Ce lot révèle la cause dominante du blocage accepteur : 24 anciens labels
  `AMBIGUOUS` sont en réalité exacts. L'accepteur et les deux ablations
  précédentes apprenaient donc sur de nombreux faux négatifs. Aucun nouveau
  réentraînement n'est autorisé avant d'avoir mesuré et corrigé cette
  contamination sur un volume plus large des REVIEW restants.
- Rapports : `reports/v412_remaining_review_audit_first5.md`,
  `reports/v412_remaining_review_audit_30.md` et tables machine associées.

## Ensemble ranker conservateur V4.12 — labels de contrôle contre-audités

- Commit : `35391c7` (`experiment: build conservative ranker ensemble`).
- Quatre prétendues régressions sur les 1 127 contrôles sont des labels
  historiques erronés : LEVAC, NETWORK HOLDING, STOCK J BOUTIQUE JENNYFER et
  l'office Xavier Maitre / Guillaume Laguë. Les corrections, toutes de
  fiabilité haute, sont consignées sans modifier le dataset canonique dans
  `reports/v412_control_label_counteraudit_4.csv`.
- Le gate retenu conserve le ranker trusted par défaut. Il ne promeut que son
  rang 2, soit avec le petit CE mélangé à `alpha=0,75`, soit lorsque BGE
  (`alpha=10`) et le business ranker ciblé s'accordent avec une marge BGE brute
  d'au moins `0,004`.
- Sur la vue identifiable corrigée : 225/254 top-1 exacts difficiles
  (88,58 %), 1 127/1 127 contrôles exacts et 1 352/1 381 combinés (97,90 %),
  soit 13 corrections et zéro régression métier observée par rapport au ranker
  trusted. Sur les labels historiques non corrigés : 1 348/1 381 (97,61 %).
- Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f`.
  Il contient `decisions.parquet`, `evaluation.json` et les 27 620 candidats
  top-20 reclassés des 254 requêtes fiables et 1 127 contrôles dans
  `ranked_candidates.parquet`, prêts pour la reconstruction des scènes de
  l'accepteur.
- Le périmètre est entièrement consommé en développement : aucun résultat
  n'est présenté comme validation indépendante et le test final reste fermé.

## Audit qualité des labels et ablation ranker locale V4.12

- Commit de code, données dérivées et résultats : `93f637f`
  (`experiment: audit and rerank local-identifiable labels`). Le canonique
  `reports/v412_review_trusted_labels_279.csv` n'est pas modifié.
- L'overlay qualité contient 14 dossiers : une correction SIRET
  (`10613`), deux corrections vers `AMBIGUOUS`, huit exclusions du périmètre
  local et trois quarantaines dépendant d'une preuve externe. Le builder
  rejouable produit 279 lignes : 241 `MATCH_EXACT`, 31 `AMBIGUOUS` et sept
  `UNRESOLVED`.
- Huit rankers ont été entraînés en cinq plis OOF : variantes `targeted` et
  `source_relational`, poids difficiles `0,25`, `0,50`, `0,75` et `1,00`,
  avec leurs sorties candidats complètes. Les sorties `non_trusted_dev`
  natives à 1 135 cas sont ignorées pour le gate : les modèles finaux ont été
  rescored sur les 1 127 identifiants figés `scope=CONTROL`, puis les quatre
  vérités de `reports/v412_control_label_counteraudit_4.csv` ont été
  appliquées en mémoire.
- La meilleure variante standalone est `targeted`, poids `0,50`, artefact
  `a8e21e7eb9e1c0cf` : 219/241 OOF (90,87 %) contre 212/241 pour le ranker
  courant, soit douze corrections, cinq régressions et un gain net de sept.
  Sur les contrôles corrigés, elle obtient 1 125/1 127 contre 1 123/1 127,
  avec deux corrections et aucune perte parmi les 1 123 dossiers déjà
  corrects. Elle ne corrige toutefois que deux des quatre contre-audits et son
  écran base-fit baisse de 4 655/4 666 à 4 647/4 666.
- L'ensemble conservateur existant, réévalué sur la vue stricte, obtient lui
  aussi 219/241, mais conserve 1 127/1 127 contrôles corrigés. Les deux
  systèmes ne commettent pas les mêmes erreurs : le standalone corrige sept
  erreurs de l'ensemble et en réintroduit sept autres.
- Verdict : **`PIVOT_STANDALONE_KEEP_CONSERVATIVE_ENSEMBLE`**. Le nouveau
  ranker ne remplace pas l'ensemble conservateur : son gain contre le ranker
  courant ne dépasse pas l'ensemble sur les 241 cas et sa protection des
  contrôles est inférieure. Conserver l'ensemble `9ba1012722cc4b3f` comme
  candidat de développement ; le standalone `targeted/0,50` reste une
  ablation utile ou un signal candidat, sans promotion autonome.
- Hashes SHA-256 des vues suivies : overlay qualité
  `0744952bb2eb76352aa7fb0b5985c1e48f38d957bc4b17387bfe7785fcff9f87`,
  labels locaux 279
  `11e499168d0df4268c8cb62abeb6f49725a5675c83749e29d60591d982a3d2c4`,
  synthèse CSV
  `a6a2e374a9dbee8638da3dbe59ada2d972e134de0af9e309e88d009ba2b1aef4`.
- Artefacts locaux sous
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_ranker_business_features_local241`.
  Pour chaque run, les trois hashes ci-dessous correspondent à
  `evaluation.json`, au modèle ranker et aux candidats rescored sur les 1 127
  contrôles :

| Run | Variante / poids | Evaluation | Modèle | Contrôles 1 127 |
|---|---|---|---|---|
| `71d7b303ea41560a` | targeted / 0,25 | `b41dafc71bf803e8bccefcd23740f555fc5af0d2ceb7cf9849a806d713ea008d` | `37c218da135b3645cef1282700b94be6d800fa560ef9ade182e3497c5f14f0b7` | `c99edc7dd48b7226b92a9a78ba9c4287f270b9c9544a0f63f35322a68447e08f` |
| `a8e21e7eb9e1c0cf` | targeted / 0,50 | `0238120bd7bfc454dcc59e487cd3b0aa061d932de972698521c8a7dc3f835887` | `73d5fde9435c10c45c17bae7e9c26a702c4ed3132a0a80707222f56bf8648acd` | `54af007b8e9040687d2924e888bb2d8aa5cbf4cfaea24d1dfeed82aac6399d4d` |
| `b94578d415ab5310` | targeted / 0,75 | `1b2f406a9ee4b65e18b15497a84ba8d0656b062f146ebdc02ef04bbd47f8a4c6` | `08897a65c02fdceb017fb47ef6fa427e62bc726f7e86d26c19c5f1e5831fce90` | `e66de6f2fd47d2ef1940a8777348041124ff442983b5941cc7feda3773aaf38d` |
| `d8d2d84c2d646213` | targeted / 1,00 | `e0aee9a46a0ba5f899f2195926b42521264caf591bee8a2d7825c5878936d53a` | `bc1c161e103924fe2edf876c4024ab443e8e1b38e57d0398841b055ec7cb4908` | `b826928a8d441b9c4208ba0537b95aa899b3ac8f28b1a1c1af37ca5e3d8516c6` |
| `7bddbecea1ebe30c` | source_relational / 0,25 | `a4f0da884257f9f664c586b37b0e92a76d4b5101505eb0d0e171ccf8e2f49836` | `2275e48e439917df3d89a32df22303a58f2e1c7da542cbabdc4332c035f6a81e` | `0ed0b1e3fedb50212ec69dfb4d3cb64d3b9321efc3cc989140873d0c6b21acfb` |
| `f6ef119fea7885ce` | source_relational / 0,50 | `8aeb29f1733d85b74d6d70f6f7f94dd760f5366db094838b2de9bdbf8ac8f49f` | `d4ad7c94efcd70666892e5595247bac20c8b2b8aa10033b86f070f044af5d0e3` | `b125342c107005503163915279af394bc8c0a6b8c30ed85a8483d0388bef02c0` |
| `d52284e24ccce43a` | source_relational / 0,75 | `14fbcda8ebe5c1756f00681fd81316cad7b52e8fd307356f3691a2e4ec0bc968` | `ddc596b77d275981d023ed4495e0fd9e183e621855fbabf935b80fac583497be` | `5678ab9d537f467b76153d5ab4c664377b01c881f7944851e9dd048509f7575c` |
| `f685d7def7e926cd` | source_relational / 1,00 | `a7e36debd8ccca214d45ff28f80c116ce6c4207e7789ab47369bf48e1fe698de` | `02c51b633c7a384a9e6f65796c95644ea6fcff7595c0ecb68bcdb33f578a50ec` | `a6a8640377f3923a0c9be1ec25a4e54a13e41dc8e95687deba4334062000ed8f` |
- Hashes de l'ensemble conservé `9ba1012722cc4b3f` : `evaluation.json`
  `572902d1301ed1a737ab9c985fe8c7e232b5eb67fa2c76912bfe4dd5ba170307`,
  `decisions.parquet`
  `991b8c0ce6c869637f49d42c664e6872d5000990650b2046f91f104d6e701191`
  et `ranked_candidates.parquet`
  `4e1a522bd451bad9dfd73510529a1c4c8b1434e9a839329c9d5368fe999c27dd`.
- Le périmètre reste du développement entièrement consommé. Aucun test final
  n'a été ouvert et aucun résultat n'est présenté comme validation
  indépendante.

## Audit ranker et accepteur V4.12 — ablations locales

- Commit : `f76e016` (`experiment: audit ranker and acceptance alternatives`).
- Les 35 erreurs récupérables du ranker ont été matérialisées candidat par
  candidat avec raisons sociales, enseignes, activités, statuts de siège,
  adresses et rangs de retrieval. L'audit sépare les erreurs de modèle des
  ambiguïtés et des labels historiquement faux.
- Les ablations réfutent trois solutions simples prises isolément : ajout de
  features métier dans le même XGBoost (`219/241` au mieux), second XGBoost
  top-20 (`215/254`) et cross-encoder seul (`222/254` au mieux). Les deux
  cross-encoders gratuits ont été exécutés localement sur le Mac, sans GPU
  loué ni service payant.
- L'accepteur enrichi de 24 features de concurrence progresse de `87/216` à
  `113/216` bons top-1 acceptés sans erreur sur les labels historiques, mais
  reste à `87/219` (`39,73 %`) sur l'ensemble ranker et les labels strictement
  locaux. Verdict : `PIVOT_FEATURES`; le seuil n'est pas abaissé.
- Rapport : `reports/v412_ranker_error_audit_and_ablations.md`. Artefacts et
  limites de développement consommé y sont référencés ; le test final reste
  fermé.

## V4.12-L — ranker métier appris sur population unifiée

- Commit : `ca73b03` (`ranker: evaluate learned V4.12-L business signals`).
- Construction immuable de 1 708 184 lignes candidat, 17 097 scènes et 129
  features apprises. Les comparaisons requête, même SIREN et même adresse sont
  des entrées XGBoost ; aucune règle ne promeut directement un candidat.
- Le run initial tronqué aux 40 premiers négatifs a été rejeté comme défaut
  d'exécution. Tous les scores retenus utilisent jusqu'à 100 candidats au train
  comme à l'inférence, sans injection du positif.
- Meilleur résultat propre : `BUSINESS_LEARNED`, 11 939/13 704 exacts en OOF
  groupé SIREN, dont 220/241 cas difficiles. La baseline 45 features obtenait
  11 501/13 704 et 211/241.
- Les ablations poids humains x2/x4, labels ouverts faibles, objectif NDCG,
  spécialiste humain et scores de deux cross-encoders locaux ne franchissent
  pas 220/241 après apprentissage propre. Verdict : **`PIVOT_RANKER`** ; le
  gate 225/241 reste fermé et le test final n'est pas ouvert.
- Artefacts : dataset métier `8800ef53f6927215`, comparaison principale
  `839ef55308d5077e`, pondérations `ed06ca38cb669291`, NDCG
  `46803026b12aae59`, sous `/Volumes/CATNAT_DATA/SIRETO_RECALL100`.
- Rapport : `reports/v412_learned_oof_ranker.md`. Tests ciblés : cinq passants
  avec les tests dataset/retrieval déjà gelés.

## V4.12-L — scènes, accepteur et conclusion du goal

- Commit : `4dd252d` (`acceptor: evaluate V4.12-L learned scenes`).
- Dataset `2f2bb2b0208241e0` : 17 097 scènes OOF, 259 features query-level
  décrivant top1, top2, meilleur autre SIREN, marges, densité et concurrence.
  Aucune vérité ni règle de promotion n'entre dans les features.
- Accepteur nested OOF `13997088931181ba` : chaque fold externe est exclu du
  train et de la calibration. XGBoost obtient 743 AUTO/17 097 (4,35 %) à
  99,596 % observés, trois erreurs et 0/38 AUTO sur les ambigus/non résolus
  audités. Sur les seules exactes : 741 AUTO/13 704, 99,865 % et une erreur.
- La borne oracle du ranker est 11 939/17 097 = 69,831 % sur toute la
  population et 11 939/13 704 = 87,121 % sur les exactes : l'objectif
  88–92 % AUTO ne peut pas être atteint par un accepteur qui ne corrige pas le
  top1.
- Le pilote pairwise cross-encoder `d517650eb9951cf9`, entraîné localement sur
  10 825 requêtes et évalué sur le fold 0 exclu, gagne au mieux 1/2 797 au
  global et 0/38 sur les cas difficiles. Les quatre autres folds n'ont pas été
  exécutés.
- Verdict final : **`PIVOT`**. Retrieval V4.12-L conservé ; ranker/accepteur
  non promus ; baseline historique inchangée ; test final non ouvert.
- Rapport consolidé : `reports/v412_learned_goal_conclusion.md`. Sept tests
  ciblés passent et les trois artefacts sont reproductibles à froid par leur
  identité immuable.

## V4.12 — BGE groupwise, sélection fold 0

- Commit : `3a9acad` (`report: record BGE groupwise fold0 result`).
- BGE groupwise fine-tuné sur les folds 2/3/4 obtient 2 400/2 797 au fold 0
  (85,806 %), contre 2 171/2 797 en zero-shot, 2 353/2 797 pour CamemBERT
  fine-tuné et 2 437/2 797 pour `BUSINESS_LEARNED`.
- Le modèle BGE seul n'est pas promu. Il corrige toutefois 79 erreurs propres
  de `BUSINESS_LEARNED`; leur union oracle vaut 2 516/2 797. Le stack
  préenregistré reste donc expérimentalement justifié sans sélection oracle.
- Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_groupwise/01e1049c16af2600`.
  Les huit sorties du manifeste ont été revérifiées sans mismatch.
- Rapport : `reports/v412_bge_groupwise_fold0.md`. Le fold 1 et le test final
  restent fermés. Les trois runs BGE cross-fittés folds 2/3/4 sont en cours
  séquentiellement avant le méta-ranker XGBoost.

## Boucle GT synthétique — durcissement après v15

- Commit : `ae5f9cb` (`fix: harden synthetic GT agent loop`).
- Le run v15 `pilot-agentic-v15-calibration-retry1` est terminé : 128 seeds,
  263 `ACCEPT`, 29 `SILVER` et 92 `REJECT`. Son export est sous
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_loop_v2/pilot/v15_retry1_export_final/`.
- Audit : les 128 contrats v15 plaçaient au moins une famille dans une
  dimension incompatible (`ADDRESS_TOKEN_ORDER` sur le nom, `LEGAL_FORM` sur
  l'adresse, etc.). Les 292 lignes publiées n'ont ni doublon, ni fuite
  d'identifiant, ni champ vide, mais v15 reste **en quarantaine** jusqu'à audit
  de la réalité des altérations déclarées et ne doit pas être ajouté au corpus
  d'entraînement consolidé.
- Le runtime refuse maintenant ces contrats avant lease, transmet à Luna un
  contrat explicite `v1=name`, `v2=address`, `v3=orthographic`, et fournit au
  retry les empreintes et erreurs précédentes. `abandon` réinscrit la seed dans
  sa file `PENDING_*`; le superviseur refuse toute seed sans exactement trois
  variantes. Le code ne génère ni ne répare aucun texte CRM.
- Validation : `PYTHONPATH=. pytest -q tests/test_run_synthetic_gt_agentic_loop.py`
  — 12 tests passants ; compilation Python et `git diff --check` passants.

## BGE final et boucle GT synthétique v17–v21

- Verdict BGE final documenté par `c633ec2` et `77edb72`. Le BGE groupwise
  fold 0 atteint 2 400/2 797 (85,806 %) contre 2 437/2 797 pour
  `BUSINESS_LEARNED`. Le stack XGBoost+BGE obtient 2 436/2 797 : 41
  corrections et 42 régressions face au meilleur XGBoost. Verdict
  `STOP_RANKER_GATE`; fold 1 et test restent fermés. Rapport :
  `reports/v412_bge_xgb_stack_verdict.md`.
- Boucle Luna durable et contrats officiels introduits par `0dbaf6c`, puis
  compatibilité Structured Outputs, faisabilité des familles et schéma strict
  corrigés par `4743f89`. Le driver conserve les réponses brutes, exécute Luna
  LOW en sessions éphémères indépendantes et ne génère/répare aucun champ CRM.
- Les opérations exactes et le nombre total d'essais ont été corrigés par
  `f545a8d`; le schéma validateur est figé par run par `875acb2`; les ajouts
  gratuits d'accents/ponctuation et l'empreinte de surface ont été corrigés par
  `cff96ba`; le CRITIC a été aligné sur le contrat de corruption par `c275b9e`.
  La suite ciblée compte 23 tests passants.
- v17 (`pilot-agentic-v17-strict`) est terminé et quarantainé : 32 seeds,
  25 `ACCEPT`, 71 `REJECT` (26,0 %). Il a exposé l'incompatibilité entre
  variantes alphanumériquement conservatrices et ancienne empreinte, ainsi que
  des faux `ACCEPT` par ajout d'accents. v18 et v19 sont des diagnostics
  interrompus avant publication après identification de ces biais.
- v20 (`pilot-agentic-v20-critic-v3`) passe le mini-gate : 8 seeds, 24/24
  `ACCEPT`, aucune fuite, aucun doublon de surface, aucun mauvais préflight et
  correspondance complète des hashes entre réponses brutes et ledger.
- v21 (`pilot-agentic-v21-full-gate`) passe le gate complet : 32 seeds,
  81 `ACCEPT`, 15 `REJECT`, 0 `SILVER`, soit 84,375 % de rendement strict.
  Les 81 surfaces acceptées sont uniques, sans fuite d'identifiant ni
  préflight invalide. Export :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_loop_v2/pilot/v21_export_final/`.
- Prochaine étape : enrichir l'intake SIRENE-only avec les dénominations de
  l'unité légale, étendre uniquement les familles prouvées dans le train
  (`TOKEN_ORDER`, OCR borné, ordre d'adresse), puis produire par shards avec
  les versions de prompts et gardes v21 gelées jusqu'à 20 000 positifs
  acceptés. Maps reste désactivé dans le plan gelé (`max_requests=0`).

## GT synthétique — extension prouvée et préproduction v22

- Commit : `cb32f9e` (`feat: scale evidence-backed Luna GT contracts`).
- Le profil train-only v2 compare les 7 095 lignes strictement train à leurs
  fiches SIRENE : 23 `TOKEN_ORDER`, 14 OCR nom, 22 OCR adresse avec numéro
  intact, 1 `ADDRESS_TOKEN_ORDER` et 15 `ENSEIGNE_VS_DENOMINATION`. Les paires
  de substitutions autorisées sont stockées avec leur compte et leur hash ;
  aucune famille n'est déclarée sur la seule base d'une intuition.
- L'intake SIRENE-only a été enrichi avec les unités légales officielles :
  13 316 SIRET/SIREN distincts possèdent désormais nom, numéro, type/nom de
  voie, CP, commune et INSEE complets. Aucun texte CRM n'est produit par cet
  enrichissement.
- Le scheduler déterministe a construit 8 000 contrats disjoints représentant
  24 000 variantes brutes. Il ne transforme aucun champ ; il assigne seulement
  des familles faisables et prouvées. À partir du profil et des sources hashés,
  le seed JSONL de production contient 8 000 SIRET et 8 000 SIREN distincts.
- Le gate v22 étendu (`pilot-agentic-v22-extended`) est terminé : 32 seeds,
  91 `ACCEPT`, 5 `REJECT`, 0 `SILVER`, soit 94,792 %. Par famille :
  `TOKEN_ORDER` 24/24, OCR nom 19/19, OCR adresse 2/2, commune 6/6,
  forme juridique 4/4. Les 91 surfaces sont uniques, sans fuite ni mauvais
  préflight. Export :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/agentic_loop_v2/pilot/v22_export_final/`.
- Production autorisée sous prompts GENERATOR/CRITIC v4 et runtime `cb32f9e`.
  L'estimation au rendement v22 est ~22 750 positifs acceptés pour 24 000
  variantes brutes ; l'exécution doit rester reprenable et s'arrêter à la
  consolidation lorsque 20 000 couples uniques ont franchi tous les gates.

## GT synthétique — consolidateur final auditable

- Commit : `a3f5911` (`feat: audit and consolidate agentic GT corpus`).
- `scripts/consolidate_synthetic_gt_agentic_corpus.py` ne génère ni ne répare
  aucun texte. Il sélectionne de façon déterministe les seuls `ACCEPT`, garde
  intégralement les familles rares, et publie exactement le volume demandé.
- Avant publication, il vérifie chaque CRM contre les champs exacts de la
  réponse brute Luna, recalcule les hashes GENERATOR/CRITIC/ADJUDICATOR,
  contrôle l'indépendance du critique et la présence d'une adjudication sur
  tout désaccord, refuse les doublons exacts ou de surface et les fuites
  SIREN/SIRET.
- La chaîne de provenance est vérifiée jusqu'aux manifests SIRENE, au profil
  observé train-only, au plan gelé et aux affectations de folds. Les SIREN de
  production doivent rester disjoints de tout `crm_ok_gt`; folds 0/1, dev et
  test restent fermés. Le plan prouve également Maps désactivé,
  `max_requests=0` et coût maximal nul.
- Un smoke en lecture seule sur le ledger de production vivant a réaudité
  1 091 variantes acceptées et consolidé 200/200 exemples avec tous les gates
  à `true`. La suite ciblée compte 33 tests passants ; compilation Python et
  `git diff --check` passent.

## GT synthétique — arrêt v1 et pilote composite v2

- Commit : `0578522` (`feat: launch evidence-backed composite GT pilot`).
- La production unitaire v1 est arrêtée et quarantainée : son audit indépendant
  a montré des dégradations mono-champ faciles et répétitives, une amplification
  excessive de preuves OCR rares et des familles qui se recouvrent. Aucun de
  ses `ACCEPT` ne compte vers la cible explicite de 20 000 couples.
- Le nouveau profil composé retrouve 452 lignes CRM train avec au moins deux
  relations classifiées. Une banque opaque distincte conserve 299 vrais couples
  officiel→CRM issus uniquement des folds 2/3/4, soit 299 SIREN et 261 signatures
  structurelles ; elle ne publie aucun identifiant source et ne génère aucun texte.
- Le contexte officiel complet couvre 8 000 cibles SIRENE-only et leurs siblings,
  collisions de site et collisions nom+géographie dans le snapshot local. 7 897
  sont exactes à la baseline et 103 seulement opérationnelles selon la politique
  même-SIREN/même-site ; les deux vues restent séparées.
- Le pilote préenregistré contient 30 cibles / 90 couples : 15 actives, 15 fermées,
  12 SIREN multisites, 4 multi-actifs et 90 inspirations train distinctes. Chaque
  variante doit modifier le nom et l'adresse ou la commune ; le code ne choisit
  que les sources et contrats, Luna LOW écrit directement tout nouveau CRM.
- Le préflight v2 refuse les changements de casse seuls, tokens/numéros inventés,
  ponctuation ou diacritiques ajoutés, numéro de voie altéré, champ hors contrat,
  doublon et ambiguïté connue. Les identifiants bruts des concurrents internes ne
  sont jamais transmis au modèle. Ledger :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/composite_v2/pilot30.sqlite`.
- Validation ciblée : 26 tests passants pour runtime/driver/sélecteur, plus 15
  tests passants pour les builders de preuves, inspirations et contexte.
- Correction de lancement : `2424a32` (`fix: constrain composite Luna generation`).
  Le premier micro-cycle diagnostique a été conservé séparément après avoir
  révélé deux défauts avant montée en volume : famille de sortie non contrainte
  et rejet erroné des jonctions de tokens réelles. Le schéma v6 force désormais
  uniquement `OBSERVED_COMPOSITE_ANALOGY`; le garde autorise jonction/séparation
  ou réordonnancement seulement sans nouveau matériau alphanumérique. Le ledger
  propre a été réinitialisé sous `synthetic-gt-composite-v2-pilot30-v1` et la
  suite runtime/driver compte 26 tests passants.
- Audit/pivot full-SIRENE : `3716817` (`fix: gate composite GT against full
  SIRENE`). Le pilote v1 a terminé 30/30 seeds avec 65 `ACCEPT`, mais seulement
  21 seeds ont 3/3. Le re-scan sans retrieval ni injection sur 590 206
  établissements locaux reclasse 2 `ACCEPT` `NAF NAF` en ambiguïtés réelles ;
  seules 20 seeds conservent 3/3 exacts.
- L'audit a découvert une casse SQL erronée dans le contexte v2b, qui masquait
  presque toutes les collisions. Le contexte v3 corrigé publie 962 734 relations
  même-adresse, 3 988 même-nom+géographie et 33 419 siblings. Sur 8 000 cibles,
  5 987 restent éligibles, 1 234 sont trop denses, 676 heurtent un SIREN protégé
  et 103 sont seulement opérationnelles. Hash v3 : `b323ca196d3900e2`.
- Le langage composite est désormais fini et identique dans le selector, le
  prompt et le préflight. Chaque champ généré doit reproduire la classe de sa
  vraie inspiration train (`TOKEN_SUBSET`, abréviation/type de voie, delta de
  ponctuation/diacritique, etc.). Les permutations internes de rue, suppressions
  de commune et analogies génériques sont rejetées. Le pilote v2 resélectionné
  contient 30 seeds/90 inspirations distinctes sous hash `1048781832882431`.
- Gate de reprise : aucun passage à 150 avant terminaison de
  `synthetic-gt-composite-v2-pilot30-v2`, audit indépendant, 3/3 strict et
  `EXACT_IDENTIFIABLE` sur le snapshot SIRENE complet pour chaque ligne admise.

## GT synthétique — PIVOT opérateurs exacts et canary fragments v3

- Commit : `5018bf1` (`fix: lock synthetic GT field operators`).
- Le pilote composite v2 est terminé mais reste en quarantaine : 75/90
  variantes ont été acceptées par l'agentique, 69 sont exactes au re-scan
  full-SIRENE et seulement 19/30 seeds conservent réellement 3/3, soit 57
  variantes. Les six ambiguïtés sont cross-SIREN et ne sont pas des
  équivalences opérationnelles.
- Les audits indépendants concluent unanimement `PIVOT` avant 150. Le verrou de
  classe a laissé passer des accents absurdes (`VITAL→VÎTAL`,
  `VERDUN→VERDÚN`) et une distribution dominée par `TOKEN_SUBSET`. Le critic a
  aussi accepté les six ambiguïtés ; il n'est donc jamais le gate d'identité.
- Toute addition de ponctuation ou de diacritique est maintenant interdite
  fail-closed. Une banque de 2 026 fragments réels par champ, issue seulement
  des folds 2/3/4, transporte l'opérateur exact : positions conservées,
  permutation, frontière de marque retirée ou paire de type de voie. Aucun
  texte n'est créé par le builder et aucun identifiant source n'est publié.
- Le contexte fusionne désormais tous les canaux de noms SIRENE quel que soit
  l'ordre d'ingestion, y compris les dénominations légales et noms de personnes
  physiques. Le preflight peut donc voir les collisions que le pilote v2 avait
  ratées.
- L'audit full-SIRENE n'accorde plus `EXACT_IDENTIFIABLE` à un singleton nom ou
  adresse isolé : seul `G_N_A={target}` est exact. Le statut ledger 3/3 et
  l'allowlist finale 3/3+exact sont publiés sous deux champs distincts ;
  l'équivalence opérationnelle exige aussi l'égalité du site officiel.
- Un canary v3 de 10 cibles/30 variantes est scellé sous hash
  `97a211558b3abce0` : 5 actives/5 fermées, 6 multisites, 1 multi-actif, 29
  opérateurs exacts distincts, top opérateur 4, zéro contrat d'ajout de marque.
  Le sélecteur ne génère aucun CRM ; Luna LOW reste l'unique auteur des sorties.
- Validation avant lancement : 46 tests ciblés passent. Aucun stage 30, 150 ou
  epoch 500 n'est autorisé avant 3/3, full-SIRENE et audit indépendant du
  canary.
- Correctif canary : `f70ea30` (`fix: preserve one distinctive subset anchor`).
  Le premier run diagnostique a révélé que le sélecteur prouvait la conservation
  d'au moins une ancre distinctive alors que le preflight exigeait par erreur
  toutes les ancres listées. Il est conservé sous
  `canary10_v3_diagnostic_anchor_bug.sqlite` et ne compte pas. Le contrat neuf
  protège exactement une ancre réellement retenue par l'opérateur de subset ;
  25 tests ciblés passent avant le run `synthetic-gt-composite-v3-canary10-v1`.
- Deuxième correction : `1bf13f6` (`fix: lock exact join and legal-form edits`).
  Le run v1, également diagnostique et non compté, a montré que `JOIN_SPLIT`
  ne transportait que le nombre final de tokens et que `LEGAL_FORM_REMOVE`
  n'imposait pas la forme exacte. Le contrat v2 transporte désormais les groupes
  contigus exacts à joindre et refuse une forme juridique cible différente ;
  l'ancre distinctive n'est exigée que pour `TOKEN_SUBSET`, les autres relations
  conservant déjà tout le matériau alphanumérique. Banque régénérée sous hash
  `b43a7b6b2675cf16`, canary v2 sous hash `c5aa6cb7431f31ef`.
- Verdict du canary v2 : `PIVOT`. Le run
  `synthetic-gt-composite-v3-canary10-v2` termine à 6 `ACCEPT` / 24 `REJECT` ;
  les 6 variantes admises sont toutes `EXACT_IDENTIFIABLE` au re-scan
  full-SIRENE, mais une seule seed conserve 3/3, soit seulement trois couples
  admissibles. Vingt-et-un rejets sont des rejets preflight seed-level : une
  variante fautive a entraîné le rejet collatéral de ses deux sœurs. Le critic
  a également rejeté à tort une permutation canonique de `GOPA-HAIDER` et deux
  suppressions attestées de `DES` dans une adresse.
- Pivot d'orchestration : `f89c823` (`fix: retry synthetic GT variants
  independently`). Le nouveau mode opt-in `per-variant` crée une tâche Luna
  distincte pour v1/v2/v3, ne retente que le slot fautif et conserve de façon
  immuable toute sortie déjà validée. Un critic ne peut être loué qu'après
  trois slots `PASSED`; un slot épuisé reste fail-closed. Les réponses brutes,
  SHA et task IDs Luna restent rattachés à chaque variante, sans génération ni
  réparation Python.
- Le critic reçoit désormais une preuve déterministe hashée par variante :
  tokenizer canonique, tokens source/cible, relation et paramètres exacts. Sa
  mission est recentrée sur le réalisme et l'ambiguïté sémantique ; il ne doit
  plus substituer sa propre tokenisation au preflight.
- `JOIN_SPLIT` est suspendu. `TOKEN_ORDER` est limité à un échange adjacent ou
  au déplacement d'une forme juridique explicite. `ADDRESS_TOKEN_SUBSET` ne
  retire dans ce canary qu'un mot fonctionnel, avec numéro, type de voie et
  ancre de rue conservés. Le sélecteur conjoint relation+fragment applique les
  caps avant génération et utilise un flot déterministe reproductible.
- Le canary v4 préparé contient 10 cibles / 30 contrats, 5 actives et 5
  fermées, 5 multisites et 1 multi-actif, 45 références train, 29 opérateurs
  exacts, zéro ajout de marque et zéro combinaison d'opérateurs dupliquée dans
  une seed. Correctif final : `7875f92` (`fix: constrain synthetic GT canary
  contracts`), qui impose aussi l'unicité des signatures relationnelles exigée
  par le runtime. Seed hash : `3cd4337436979ea8`. Les deux générations indépendantes
  du sélecteur sont byte-identiques. Suite ciblée : 44 tests passants. Seul ce
  canary peut être lancé ; aucun passage à 30/150/20 000 avant son audit
  full-SIRENE et un nouveau verdict indépendant.
- Résultat v4 : 22 `ACCEPT`, 7 `REJECT`, 1 `SILVER`; 28/30 slots passent le
  preflight et 22/22 `ACCEPT` sont exacts full-SIRENE. La mécanique par
  variante et toute la provenance passent, mais seulement 7/10 seeds gardent
  3/3, soit 21 couples admissibles. Les audits indépendants concluent `PIVOT` :
  six `ACCEPT` coupent un composé ou finissent sur un mot fonctionnel, et le
  fingerprint alphanumérique a créé un faux doublon entre deux suppressions de
  ponctuation. Aucun de ces 21 couples n'est promu.
- Correctif v5 : `d4e1d17` (`fix: reject linguistically invalid GT
  contracts`). La déduplication utilise la surface canonique, jamais
  l'empreinte alphanumérique. `TOKEN_ORDER` est réservé au déplacement d'une
  forme juridique; `TOKEN_SUBSET` conserve atomiquement les composés
  apostrophés/tiretés, au moins trois tokens sur quatre et ne peut finir par un
  mot fonctionnel. Sa part planifiée baisse de 14/30 à 10/30. Le critic ne peut
  qualifier `OPERATIONAL_ONLY` qu'avec même SIREN et même site officiel.
- Le canary v5 neuf contient 10 cibles / 30 contrats, 5 actives/5 fermées,
  4 multisites, 1 multi-actif, 42 références train et 24 opérateurs exacts ;
  zéro ajout de marque et zéro relation suspendue. Seed hash :
  `17b896016dd30199`. Deux sélections indépendantes sont byte-identiques et les
  30 contrats passent l'initialisation fail-closed. Nouveau canary10 obligatoire
  avant toute extension.
- Résultat v5 : 27 `ACCEPT` / 3 `REJECT`, 29/30 slots passent le preflight,
  9/10 seeds conservent 3/3 et les 27 `ACCEPT` sont toutes
  `EXACT_IDENTIFIABLE` au re-scan full-SIRENE. La mécanique, le transport et la
  provenance sont intègres, mais l'unique slot épuisé recevait un contrat
  `TOKEN_SUBSET` qui se recanonisait nécessairement en retrait de forme
  juridique. L'audit de réalisme a aussi isolé des sous-ensembles qui
  conservaient une particule ou un terme générique au lieu du patronyme ; le
  lot reste donc en quarantaine malgré son exactitude.
- Correctif v6 : `44c8303` (`fix: prevalidate synthetic GT name anchors`). Le
  selector refuse avant Luna tout subset qui ne retire que des formes
  juridiques, exclut toutes les formes juridiques des ancres distinctives,
  préserve la dernière vraie ancre du nom et interdit les particules terminales
  telles que `DA`, `DOS`, `VAN` ou `VON`. La suite ciblée compte 48 tests
  passants.
- Le canary v6 de confirmation contient 10 cibles / 30 contrats, 5 actives/5
  fermées, 7 multisites, 1 multi-actif, 44 références train et 28 opérateurs
  exacts ; zéro ajout de marque. Deux générations indépendantes sont
  byte-identiques sous hash `8f1c78af6d53f2aa`. Les 30 contrats passent
  l'initialisation per-variant fail-closed. Seul ce canary de confirmation est
  autorisé avant de statuer sur le pilote 30 ; aucun passage direct à 150 ou
  20 000 n'est autorisé.
- Résultat v6 : 27 `ACCEPT` / 3 `REJECT`, 29/30 slots passés, 9/10
  seeds strictes et 27/27 `EXACT_IDENTIFIABLE` au full-SIRENE. Les 47 appels
  Luna ont réussi au premier essai de transport et toute la provenance est
  cohérente. L'unique slot épuisé était un faux rejet du validateur : avec
  deux apostrophes identiques, il attribuait la suppression à la première
  occurrence au lieu de comparer leurs frontières lexicales. Le micro-rerun
  isolé après correction passe 3/3 critic et 3/3 full-SIRENE exact.
- L'audit indépendant strict maintient toutefois `PIVOT` avant le pilote 30 :
  deux `TOKEN_SUBSET` perdaient aussi l'apostrophe de `PRUD'HOMME` sans la
  déclarer, et des ancres faibles permettaient `THREE III SARL→III SARL` ou
  `ASSOCIATION ... GYMNASTIQUE→ASSOCIATION DE GYMNASTIQUE`. Le critic avait
  accepté les 27 surfaces et ne remplace donc pas ces gardes déterministes.
- Correctif v7 : `4e0f69b` (`fix: isolate synthetic GT field operations`). Les
  marques sont comparées par frontière de token et toute opération autre que
  `PUNCTUATION_REMOVED` doit conserver celles portées par les tokens retenus.
  L'ancre de nom est maintenant le token officiel le plus rare dans la
  population SIRENE gelée ; chiffres romains, formes juridiques et concurrents
  locaux sont exclus. Le quota `TOKEN_SUBSET` baisse de 10 à 6 sur 30.
- La banque d'inspirations train-only v2 retire 90 fragments multi-opérateurs :
  1 936 preuves propres sous hash `707928c9aeb2a9f1`, folds 2/3/4 uniquement,
  sans texte synthétique. Le canary v7 resélectionné contient 30 opérateurs
  exacts distincts et 45 références, sous hash `c4cc16643917537b`. Deux runs
  du selector sont byte-identiques, 58 tests passent et les 30 contrats
  s'initialisent fail-closed. Nouveau canary10 obligatoire avant le pilot30.
- Résultat v7 : 24 `ACCEPT` / 6 `REJECT`, 28/30 slots passés et 24/24
  `EXACT_IDENTIFIABLE`, mais seulement 8/10 seeds strictes. Les deux slots
  épuisés demandaient à Luna de retirer une seule occurrence parmi plusieurs
  ponctuations identiques ; ces contrats non transférables sont désormais
  exclus avant lease. Le lot reste en quarantaine.
- Le canary v8 améliore le rendement ledger à 27 `ACCEPT` / 3 `REJECT` et
  29/30 slots passés. Le full-SIRENE trouve toutefois une vraie ambiguïté
  cross-SIREN sur `L'HOSPITAL` : 26 variantes exactes seulement et 8/10 seeds
  strictes. Le critic avait accepté 27/27, confirmant qu'il juge le réalisme et
  jamais l'identité.
- Correctif de contexte : `1a794ef` (`fix: close synthetic GT context gaps`).
  Tous les candidats siblings/même-adresse/name-geo reçoivent maintenant les
  noms complets d'unité légale, indépendamment de leur canal de découverte.
  Le snapshot v5 compte 8 000 cibles sous hash `e63681b0ea637560` et le cas
  `L'HOSPITAL` déclenche désormais `KNOWN_CONTEXT_AMBIGUOUS` avant critique.
  Les `TOKEN_SUBSET` sur groupes apostrophés ou tiretés sont aussi refusés :
  Luna et le tokenizer canonique ne découpent pas `J'ENTENDS` de façon assez
  fiable pour transporter des positions.
- Préparation pilot30 : le sélecteur joint désormais dans un même MILP les
  cibles, relations, opérateurs exacts et capacités des preuves sources. Les
  29 SIRET/SIREN de tous les canaries v3-v8 sont exclus par registres hashés.
  Le dry-run frais produit 30 cibles/90 contrats, 15 actives/15 fermées,
  15 multi-sites dont 4 multi-actives, 142 références distinctes (maximum 3),
  45 opérateurs exacts (maximum 10), zéro ajout de marque. Hash seed :
  `33065afa11d58c39`. Deux matérialisations indépendantes sont byte-identiques.
  Les quotas sont gelés avant toute génération Luna : nom
  30 subset / 12 ordre / 24 forme légale / 24 ponctuation ; localisation
  24 abréviation / 24 subset adresse / 15 ponctuation adresse / 27 ponctuation
  ville. Le cap 9 était mathématiquement infaisable ; 10 est le premier cap
  opérateur faisable, soit 5,56 % des 180 opérations de champ. Les 90 contrats
  couvrent 75 signatures composites ; une seule atteint le plafond 3.
  Sélecteur intégré, caps et promoteur : commit `fa22174`.
- Le promoteur `scripts/promote_synthetic_gt_full_exact.py` impose désormais
  une promotion atomique 3-ou-0 par seed, vérifie le raw Luna et tous ses SHA,
  le preflight, le critic, l'absence d'injection et le témoin full-SIRENE
  `G_N_A`, puis publie par renommage exclusif. Exercé sur le canary v8, il
  promeut exactement les 8 seeds/24 variantes réellement exactes et exclut
  l'ambiguïté `L'HOSPITAL` ainsi que la seed incomplète.
- Cache d'audit full-SIRENE : commit `de18d52` (`perf: cache official SIRENE
  audit projection`). Le premier audit sans cache vérifie les SHA-256 des deux
  parquets gelés, matérialise une unique projection officielle préjointe et
  indexée par INSEE, puis la publie atomiquement en lecture seule sous une clé
  dérivée des SHA sources et du schéma SQL. Les lots suivants valident le
  sceau, le compte de 42 322 035 lignes, les tables et colonnes exactes, puis
  ouvrent uniquement ce cache en `read_only`; aucune donnée CRM, cible,
  décision, score ou rang n'y entre. `qualify_variant` et ses gates sont
  inchangés. Tests ciblés : 10 passants. P009 était déjà audité/promu et ses
  artefacts n'ont pas été touchés; la construction lourde est laissée au
  premier audit ultérieur afin de ne pas concurrencer un lot actif.
- Optimisation de l'audit cache : commit `0c04e2d` (`perf: prune full SIRENE
  audit qualification`). Le profiling P011 attribue 37,5 s seulement à
  `query_candidates` sur 3 553 387 lignes et le reste des 363 s aux
  normalisations et couvertures de spans Python. La requête utilise désormais
  les couples INSEE+code postal du CRM, avec wildcard uniquement quand le CP
  manque, et réduit P011 à 2 141 491 lignes (-39,7 %). La matérialisation est
  streamée, les doubles normalisations sont supprimées et `_span_cover` reçoit
  un rejet préalable mathématiquement nécessaire. Aucune table ni qualification
  n'est modifiée. P010 passe d'environ 446 s hors construction à 169,58 s;
  P011 de 363 s à 144,65 s. Les JSON avant/après sont byte-identiques, SHA-256
  `b268e01b…` pour P010 et `22eb662e…` pour P011. Les 11 tests ciblés passent,
  dont une équivalence exhaustive du nouveau rejet sur les petites séquences.
- Optimisation du sélecteur de production : commit `dba7487` (`perf: cache
  balanced selector context and solve mix once`). Un index content-addressed
  scelle les offsets, l'éligibilité et les fréquences documentaires du contexte
  officiel, puis ne charge que le pool retenu. La recherche exacte puis relaxée
  en cascade est remplacée par un MILP unique qui minimise lexicographiquement
  l'écart de difficulté et le nombre de cibles, sans changer les caps, gardes ni
  contraintes de Hall. Sur le registre P012 gelé, le replay P013 passe de plus
  de 321 s à 21,41 s cache chaud et 469 Mio de pic RSS ; deux replays sont
  byte-identiques et les 20 tests ciblés passent.
- Planification sur capacité résiduelle : commit `a420976` (`fix: plan balanced
  batches from residual pair capacity`). Le sélecteur retire désormais avant
  stratification et thinning toute paire composite ayant atteint son cap final,
  et borne aussi les répétitions d'un bundle par sa capacité résiduelle. Le pool
  MILP utilise le pool de capacités demandé au lieu d'une seconde troncature
  fixe à 3 000. Après le filtre fail-closed `9922992`, P019 est faisable avec un
  pool 12 000 : 600 contrats sur 386 cibles, A/F 300/300, difficulté exacte
  128/293/179, zéro nouvelle occurrence des deux paires saturées et SHA-256
  `501987ed53204d7e8`. Les 22 tests ciblés passent.
- Certificat de complétion du corpus équilibré : commit `ada9ae1` (`feat:
  certify balanced synthetic corpus completion`). Le plan final conserve
  exactement 20 % EASY / 55 % MEDIUM / 25 % HARD, 10 000 variantes actives et
  10 000 fermées, 55–59 % d'identités actives, au plus trois variantes par
  SIRET, 60 % d'alias et 20 % par paire; le plafond d'identités passe à 11 000
  afin de réduire la corrélation. L'audit Hall nomme `RUE→R` comme opérateur
  dominant : 1 778 usages étaient déjà promus sous l'ancien cap 2 000. Le
  certificat entier + flow sur les 9 960 variantes résiduelles exige au plus
  4 143 usages cumulés par opérateur et 4 079 par référence; l'avenant scellé
  garde une marge bornée à 4 200 (10,5 % des 40 000 opérations de champ) et
  4 100. La paire reste plafonnée à 4 000 et l'alias à 12 000. Le certificat
  final passe au premier flow sur 5 843 nouvelles identités, SHA seed
  `2cfeda71319e4f1e`, et les 35 tests ciblés passent. Le contexte officiel v10
  est scellé sous `f0b0a5645efc62b9`; la banque train-only v4, folds 2/3/4 et
  sans modèle/retrieval, sous `f4a1cefc96514e5b`.
- Avenant de capacité résiduelle après P032 : commits `b4b61d9` (`fix:
  preserve reachable final identity balance`) et `8fa2e0d` (`fix: seal
  residual synthetic capacity avenant`). Le préfixe réel de 17 616 promotions
  compte A/F=8 802/8 814 variantes et 4 216/4 034 identités. Les rejets
  fail-closed ont rendu le plan initial couplé 55 % / alias 60 % / paire 20 %
  infaisable : le diagnostic LP donne seulement 91,56 HARD disponibles sous
  le mix état-identité alors que 543 restent requis. L'avenant borné conserve
  exactement E/M/H=20/55/25, A/F=50/50, le plafond identitaire 59 %, max trois
  variantes/SIRET, tous les caps opérateur/référence/subset et toutes les
  validations ; seuls le plancher d'identités actives (54,5 %), l'alias (65 %)
  et la paire (25 %) changent. Le certificat final sans override sélectionne
  exactement 2 384 variantes résiduelles sur 1 616 identités, A/F=1 198/1 186,
  E/M/H=440/1 401/543, passe le flow exact au premier essai et borne les maxima
  cumulés opérateur/référence à 3 818/3 754 sous les caps 4 200/4 100. SHA-256
  seed `8dc90749c645a8a5`, manifest `55842f2f96c42880`; les 35 tests ciblés
  passent.
- Arrondi identitaire terminal après P035 : commit `b4ae409` (`fix: preserve
  reachable terminal identity balance`). Le registre réel compte 19 329
  promotions, A/F=9 656/9 673 variantes et 4 965/4 327 identités. Les 671
  variantes finales imposent exactement 344 actives et 327 fermées. Même avec
  une identité par variante active et trois variantes par identité fermée, le
  maximum arithmétique est 5 309/(5 309+4 436)=54,47922 % d'identités actives,
  juste sous l'ancien plancher 54,5 %. Le plancher est donc borné à 54,45 %;
  le plafond 59 %, les 10 000/10 000 variantes A/F, le mix E/M/H, tous les
  caps et toutes les validations restent inchangés. Les 35 tests ciblés
  passent; un certificat entier et flow exact des 671 variantes terminales
  demeure obligatoire avant génération.
- Mix de difficulté terminal après P035 : commit `cc94ad1` (`fix: preserve
  reachable terminal difficulty mix`). Sous les états finaux exacts et la
  borne identitaire corrigée, le LP sans aucun cap global borne les 671 lignes
  restantes à 146,377 `HARD`, alors que le reliquat de l'ancien objectif en
  exigeait 211. Les caps globaux offrent séparément au moins 2 210 lignes : le
  déficit est donc structurel au croisement état/difficulté, pas un manque de
  références ou un cap saturé. Le mix final devient exactement 20 % `EASY`,
  55,5 % `MEDIUM`, 24,5 % `HARD`; les états 10 000/10 000, les bornes
  identitaires, les caps et toutes les validations restent inchangés. Les 35
  tests ciblés passent; le reliquat exact exige désormais E/M/H=124/436/111.
- Cap d'alias officiel terminal : commit `576bbff` (`fix: reserve terminal
  official alias capacity`). P035 avait déjà consommé 12 985 des 13 000 alias
  du cap 65 %. Le diagnostic LP terminal rend `OPTIMAL` en retirant uniquement
  ce cap, mais reste `INFEASIBLE` si l'on retire seul le cap paire, subset,
  relation, référence ou opérateur. Un flow exact des 671 lignes utilise 648
  alias supplémentaires, soit 13 633 cumulés. Le cap alias est donc porté à
  68,2 % (13 640, marge 7); paire 25 %, opérateur 4 200, référence 4 100,
  subset et toutes les validations restent inchangés.
- Marge d'arrondi après promotion P036 : commit `40f99c7` (`fix: reserve
  terminal identity rounding margin`). P036 promeut 645/671 variantes et porte
  le registre à 19 974. Le reliquat exact A/F=19/7 ne peut dépasser
  5 304/(5 304+4 439)=54,43908 % d'identités actives, même avec une identité
  par variante active et trois variantes par identité fermée. Le plancher est
  borné à 54,40 % afin d'absorber les rejets fail-closed terminaux; plafond,
  états, difficultés, caps et validations restent inchangés.
- Marge de difficulté du reliquat P037 : commit `435bd25` (`fix: reserve
  terminal hard-case margin`). Après les 645 promotions P036, le reliquat de
  26 lignes exigeait encore 11 `HARD`, mais le LP sous états et identités
  finaux en borne au plus 7 même sans caps globaux. Le déplacement terminal de
  dix lignes fixe donc le corpus à 20 % `EASY`, 55,55 % `MEDIUM` et 24,45 %
  `HARD`, soit un reliquat E/M/H=4/21/1 avec marge; tous les autres invariants
  restent inchangés.
- Corpus équilibré final : les batches P037 et P038 promeuvent respectivement
  24 et 2 lignes et portent le registre à exactement 20 000 promotions sur 39
  lots. Le finaliseur valide 20 000 clés et surfaces CRM distinctes, 9 737
  SIRET cibles, trois variantes maximum par cible, A/F=10 000/10 000 et le mix
  exact E/M/H=4 000/11 110/4 890. Les cinq strates atteignent exactement leurs
  quotas 4 000/3 000/3 000/8 000/2 000. Tous les caps restent respectés;
  l'alias officiel termine à 13 628/13 640 et la paire la plus utilisée à
  4 915/5 000. Le corpus consolidé est
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/balanced_v1/final_corpus_v1/promoted_20000.jsonl`,
  SHA-256 `9e871f0f3c5a19d28a59619c4fd09c87be5d1e75e54296ff41bf34a4dd5cbcc1`;
  le registre scellé a le SHA-256
  `c98655b5780118e5c4e1aa3bb6486bd23e2d4cedc871b3422955a785419bb0a0`.
  L'auditeur final a été corrigé pour reconstruire la population réelle train
  strictement disjointe : il retire aussi les quatre lignes fold 2 dont le
  SIREN ou la composante apparaît en dev/test, au lieu de compter naïvement
  les 7 099 étiquettes train. Les 7 095 lignes gelées, leurs comptes par fold,
  état, composante et SIREN sont désormais vérifiés fail-closed. Les 15 tests
  finaliseur/auditeur passent. L'audit déterministe des 20 000 lignes valide
  toutes les provenances agentiques, les 39 audits full-SIRENE, la disjonction
  des SIREN et des surfaces avec le réel, et publie l'échantillon stratifié de
  200 lignes sous
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/synthetic_gt_corpus/balanced_v1/final_audit_sample_v1`.
  Son statut reste explicitement `PENDING_BOUNDED_REALISM_REVIEW` jusqu'à la
  revue humaine bornée; le rapport a le SHA-256
  `d7ac367005300be6369921683155b6d1dbbc372b2ef59b965a37f9034c7e79d0`.

  *(audit de population train strict et publication finale : commit
  `2b04970`)*

## Ground truth CRM prospectif assaini — 17 août 2026

- Le tri géographique brut des 20 209 nouvelles lignes CRM est conservé comme
  diagnostic, mais ne constitue plus un label exact : sa revue indépendante
  initiale a trouvé 233 `PASS`, 82 `BORDERLINE` et 85 faux labels certains sur
  400. Aucun retrieval, score ou rang n'a été utilisé pour cette qualification.
- Le commit `6e3c242` ajoute une admission fail-closed identité + site : preuve
  nominale officielle stricte, INSEE et code postal exacts, voie fortement
  concordante, numéro et suffixe BIS/B/TER préservés, puis unicité du SIRET
  soutenu parmi tous les établissements SIRENE au même site officiel. Le scan
  exhaustif admet 4 129/20 209 lignes, met 503 collisions de site en quarantaine
  et rejette 15 577 lignes sans preuve directe suffisante. Une ligne admise est
  ensuite exclue par conflit de composante historique.
- La population candidate content-addressée
  `crm_gt_v2_sanitized_population/388725422e1e561e` conserve strictement les
  folds/composantes du build prospectif initial et maintient toutes les nouvelles
  lignes à poids/éligibilité zéro jusqu'au gate. Son nouvel échantillon est
  disjoint des 400 diagnostics, SIREN-unique, 200 TRAIN/100 DEV/100 TEST et
  stratifié par état : 337 actifs et 63 fermés.
- La revue indépendante fraîche passe : 395 `PASS`, 5 `BORDERLINE`, zéro faux
  certain, SHA-256 `d29735327c0908fdc6327ddadd761944a610315e91f999dfd313e6caf515e57d`.
  Les cinq borderline restent `UNRESOLVED`, poids zéro. La population certifiée
  est `crm_gt_v2_certified_population/70624cbc18ec84bd` : 4 123 nouvelles scènes
  exactes (2 591 TRAIN, 751 prospective DEV, 781 prospective TEST), aucun statut
  pending et aucune fuite de composante. Les 16 080 lignes non admises restent
  disponibles uniquement en quarantaine d'évidence ou pour une future vue
  opérationnelle/weak supervision, jamais comme ground truth exact.

## Retrieval hiérarchique sparse open-set — 17 août 2026

- Le commit `47e1e34` ajoute, sans modifier le moteur historique par défaut, un
  retrieval Tantivy explicitement opt-in : filtre INSEE puis fallback CP,
  BM25/exact/q-grams, documents agrégés `(INSEE, SIREN)`, expansion top-5
  limitée à 32 sites, succession officielle à un saut et sortie déterministe
  plafonnée à 100 candidats. Il ne consomme aucun alias ni label CRM.
- Le builder DuckDB parcourt les snapshots nationaux en flux et produit un
  index adressé par le contenu. Les historiques établissement/unité légale et
  les successions sont absents localement : un build courant reste possible
  avec `temporal_complete=false`, mais le runtime de production le refuse.
  Tantivy 0.25.1 est déclaré; l'évaluation bornée publie séparément Recall@100
  exact/opérationnel, maximum observé et latences p50/p95/p99 dans un JSON
  scellé. Ne pas ouvrir le fold test avant gel et gate dev.
- Validation livrée : 9 tests ciblés en moins d'une seconde, dont deux vrais builds/requêtes
  Tantivy sur fixtures; `py_compile` et les deux entrées CLI passent. Aucun
  build national, entraînement, téléchargement massif ou test final lancé.
- Le commit `f046817` lie l'identité content-addressée de l'index au SHA-256
  du builder lui-même, afin qu'une modification du code ne puisse pas réutiliser
  silencieusement un index construit par une version antérieure.
- Le commit `cc0c25a` désactive la préservation inutile de l'ordre d'insertion
  DuckDB afin que les agrégations des historiques officiels puissent déverser
  sur le SSD plutôt que saturer la mémoire du Mac.
- Le commit `cdf3dd2` matérialise séquentiellement et atomiquement les trois
  lookups officiels historiques sur le SSD avant leur jointure. Il évite ainsi
  de maintenir simultanément plus de 160 millions de lignes d'agrégation en
  mémoire et rend ces projections réutilisables après une interruption.
- Le commit `45f5557` supprime les tris globaux inutiles des 43,9 millions de
  documents SIRET et des documents SIREN agrégés : les résultats sont désormais
  transmis à Tantivy en flux au lieu d'être matérialisés avant la première ligne.
- Le commit `6a9af1a` branche explicitement les colonnes certifiées
  `crm_address` et `crm_postcode` dans la requête hiérarchique, avec un test de
  régression sur le schéma exact de la population CRM v2.
- Le commit `c1ec1f6` extrait le numéro de voie placé en tête de l'adresse CRM
  lorsque la source ne fournit pas de colonne numéro séparée. Le tri des sites
  d'un SIREN applique donc réellement le contrat numéro/adresse avant le statut
  de siège sur les benchmarks CRM commerciaux; les 9 tests ciblés passent.
- Le premier build national a terminé les 43,9 millions de documents SIRET,
  puis l'agrégat global SIREN a dépassé le plafond DuckDB de 3 Gio avant sa
  première sortie. Le commit `9d5aa87` matérialise désormais cet agrégat
  content-addressé avant toute écriture Tantivy : un échec précoce reste
  réutilisable et ne peut plus supprimer une passe SIRET achevée. Les 9 tests
  ciblés passent.
- Le build national final `096dd81d4102bdcd` est publié : 83 064 260 documents,
  historiques et successions complets, aucun label CRM. Le commit `0638c3b`
  rend les requêtes CRM robustes aux mots réservés Tantivy, supprime le fan-out
  pathologique des pseudo-termes `g3/g4/g5`, parallélise les canaux et réserve
  les q-grams au rescue. Les 9 tests ciblés passent.
- Évaluation dev, sans positif injecté et avec un maximum observé de 100 :
  commercial humain 3 301/3 510 = 94,046 % (baseline 93,419 %), p50/p95/p99
  166/635/1 020 ms; prospectif certifié identité+site 748/751 = 99,601 %.
  Le diagnostic montre 121 anciens échecs récupérés mais 99 anciens succès
  évincés. L'oracle combiné sparse historique + hiérarchique atteint seulement
  3 473/3 510 = 98,946 %, deux cas sous le gate avant admission. Verdict
  `PIVOT`; fold test maintenu fermé. Rapport :
  `docs/hierarchical_retrieval_v1_results.md`.
- Le dense global V9 a été rejoué uniquement sur les 37 scènes dev absentes
  des deux oracles sparse. Le résultat hybride contient cinq bons SIREN, mais
  leur provenance est le sparse local : interrogé seul en top-50, le dense en
  retrouve zéro. Avec p95 9,35 s, il n'apporte aucun nouveau candidat prouvé et
  n'est pas retenu comme rescue. L'oracle réel reste 3 473/3 510 = 98,946 %;
  il faut un nouveau signal temporel/caractère avant de travailler l'admission.
- Une profondeur sparse 5 000 sur ces mêmes 37 scènes retrouve finalement quatre
  vérités nouvelles (rangs 480/690, 2 113, 4 418 et 952 selon les canaux). L'oracle
  combiné franchit 99 % à 3 477/3 510 = 99,060 %, mais au coût de 136 s pour 37
  scènes et avec des rangs très au-delà de 100. Le prochain problème est donc
  strictement borné : trigger d'incertitude + fusion score-level train/dev,
  sans dense, sans alias et sans ouverture du test.

## Retrieval officiel RNE/BODACC + admission LambdaMART — 18 août 2026

- Le commit `cc4d81c` ajoute les acquisitions officielles immuables. RNE est
  accepté uniquement par SFTP avec `known_hosts`, FTPS après `AUTH TLS`, ou
  HTTPS authentifié; le secret reste dans le Trousseau macOS et n'apparaît ni
  dans les arguments, ni dans les logs, ni dans les manifests. BODACC supporte
  le backfill officiel et l'incrément Opendatasoft v2.1 par curseur date+ID.
  Les publications sont content-addressées, hashées et renommées atomiquement.
  Le serveur `www.inpi.net:21` fourni pour le RNE n'annonce pas TLS : il a été
  sondé sans authentification puis refusé fail-closed; aucun identifiant n'a été
  envoyé en FTP clair. Un accès INPI SFTP/FTPS/HTTPS est requis pour le sync RNE
  réel. L'API BODACC v2.1 a été vérifiée sur une réponse réelle sans texte libre.
- Le commit `1207384` construit une couche canonique SIRENE/RNE/BODACC avec
  valeurs officielles brutes et normalisées, précédence SIRENE, conflits en
  quarantaine et seules relations structurées SIRET/SIREN. Dirigeants,
  bénéficiaires effectifs et texte intégral BODACC restent exclus. L'overlay
  Tantivy est séparé de l'index national de 98 Go, content-addressé et jetable;
  il ne contient ni label CRM ni valeur brute. L'union hiérarchique exporte au
  plus 2 000 candidats avec rang/score/provenance pour 13 canaux, métadonnées
  de qualification, latence et garde géographique sur les successions.
- Le commit `b66c5bb` ajoute l'admission unique LambdaMART `rank:ndcg` : folds
  2/3/4 pour l'entraînement, fold 0 pour le développement, protections exactes
  et consensus, plafond absolu de 100 et exclusion des équivalents même
  SIREN/même site des négatifs. Les identifiants ne sont pas des features; le
  dense et le synthétique sont refusés. Les rapports publient historique/V2/V3,
  exact/opérationnel, couverture, oracle et latences. Les gates gelés sont
  couverture >=80 %, oracle >=99,3 %, Recall@100 exact >=99 %, p95 <=1 s et
  p99 <=2 s. Le fold 1 nécessite une autorisation hashée consommée une seule
  fois avant lecture.
- Validation bornée : 40 tests ciblés passent, dont un vrai build Tantivy et un
  E2E `builder -> overlay -> union Parquet -> build_internal_union`; le parseur
  BODACC imbriqué est aussi confronté à une réponse API réelle. Aucun build
  national enrichi, entraînement réel, dev réel ou fold 1 n'a été lancé. La
  prochaine exécution est : obtenir le transport RNE sécurisé, synchroniser les
  snapshots, construire l'overlay, entraîner sur 2/3/4 puis ouvrir une seule
  évaluation dev 0. Le test reste fermé tant que tous les gates dev ne passent
  pas.
- La documentation INPI Formalités v4 fournie le 18 août lève le blocage
  transport sans réutiliser le FTP. Le commit `f21198d` implémente le login
  HTTPS officiel `/api/sso/login`, le jeton Bearer, le flux différentiel
  `/api/companies/diff`, le curseur de réponse `pagination-search-after` et les
  bornes documentées `from` exclusive / `to` inclusive. Le jeton et le secret
  sont effacés après exécution; aucun credential n'entre dans un manifeste.
  Le parseur RNE couvre désormais les chemins JSON `content.personneMorale` et
  `content.personnePhysique` de la documentation. Il met fail-closed en
  quarantaine `diffusionCommerciale=false` et `diffusionINSEE=N`. Les 43 tests
  retrieval officiels passent. L'endpoint TLS de production répondait, mais le
  Trousseau `com.sireto.rne-official` n'était pas encore initialisé à ce jalon :
  aucune authentification ni synchronisation RNE réelle n'avait alors été lancée.
- L'accès API a ensuite été initialisé dans le Trousseau et la première
  synchronisation réelle a abouti : intervalle du 17 août exclus au 18 août
  inclus, 16 771 formalités, 171 549 863 octets, payload SHA-256
  `72ba7661fcaf84e4fc96623c2bc390ff0c4c20ee3c5a1023b934e762932ab794`,
  manifest `6fad4c6d7136e5b2`. Le commit `299bf3f` couvre l'enveloppe live
  supplémentaire `company.formality` observée dans ce flux. La canonisation
  ciblée SIRENE+RNE produit 5 885 preuves RNE autorisées, met en quarantaine
  10 860 oppositions de réutilisation, 96 conflits géographiques de précédence
  et 26 enregistrements vides. L'overlay
  `official_evidence_overlay_v1/8c2178602fca3a3b` contient 21 048 documents,
  pèse 46 MiB, n'a pas modifié l'index national et ne contient ni label CRM ni
  valeur brute. Son manifest a le SHA-256
  `2f40ed20ddeb8a861c5492c291724e0bcb0a4223c35e91b4fa18c57aa9cc75ab`.
- Le commit `369e1ad` ferme le dernier défaut observé sur l'incrément BODACC :
  les livres A/B/C peuvent être partitionnés et le curseur
  `dateparution + numeroannonce` est comparé numériquement, avec littéral date
  Opendatasoft explicite. Le sync réel de la parution `20260156` contient
  exactement 20 291 annonces (A 3 829, B 4 749, C 11 713), publiées dans trois
  artefacts content-addressés sans secret. La couche combinée quotidienne
  `official_2026-08-18_v2` compte 43 873 preuves et 151 relations structurées;
  elle exclut toujours dirigeants, bénéficiaires effectifs, texte BODACC et
  labels CRM. L'overlay séparé
  `official_evidence_overlay_v2/338cd52006a9c30f` contient 39 959 documents
  pour 69 MiB, n'a pas modifié l'index national et son manifest a le SHA-256
  `0b17954785d7442623b6dc25909280fcc327367cc393a7c0fe88b383c2d632fb`.
  Les 44 tests ciblés retrieval officiel/LTR passent; aucun fold test n'a été
  ouvert. Le prochain jalon n'est plus l'acquisition mais le backfill officiel
  historique, puis l'union train 2/3/4 et l'unique évaluation dev 0 prévue par
  le contrat.

## Dossier SIREN Parquet/DuckDB + backfill RNE — 18 août 2026

- Le commit `38d62c2` remplace le téléchargement RNE mono-intervalle par un
  backfill historique partitionné, reprenable et scellé par manifeste. Le
  receipt vérifie le SHA-256 de chaque partition déjà publiée avant de la
  sauter. Le commit `da2fde7` écrit les partitions API en JSONL gzip
  déterministe et `36b02f3` impose un plancher de 64 Gio libres avant toute
  nouvelle partition. Le run réel 2000-01-01 -> 2026-08-18 est lancé en 1 390
  fenêtres de sept jours; aucune donnée déjà scellée ne sera rejouée après une
  interruption. Après un refus de connexion INPI sur la septième fenêtre, le
  commit `e3e1443` ajoute six tentatives réseau par partition avec attente
  exponentielle; la reprise repart du dernier manifeste valide.
- Le commit `148a0ce` introduit le store content-addressé « dossier SIREN ».
  Son data plane est constitué de Parquet séparés pour unités légales, sites
  SIRET, preuves de noms, preuves d'adresses, relations structurées, résolution
  prudente adresse-SIREN vers site unique et synthèse du dossier. Un catalogue
  DuckDB portable lie ces fichiers sans les dupliquer. BODACC/RNE restent des
  preuves sourcées et datées; ils ne créent jamais automatiquement une identité
  SIRET, un label ou un score modèle.
- Le commit `29c773c` expose des projections opt-in communes : documents
  retrieval directs SIRET et hiérarchiques SIREN, features officielles
  `(query_id,candidate_siret)` pour ranker/decider/risk, et textes longs séparés
  par champ/source pour BGE, CamemBERT et fusion. Les identifiants restent des
  clés d'audit, pas des features. Les anciens modèles demeurent gelés jusqu'au
  gate retrieval actif (couverture >=80 %, Recall@100 exact >=99 %, max100).
  Le build national initial SIRENE courant + BODACC 2008-2026 + RNE quotidien
  est publié sous `siren_dossier_v1/e4196b1eed7199b6` : 29 922 486 SIREN,
  43 896 818 SIRET, 107 996 779 preuves de noms, 88 230 482 preuves
  d'adresses, 24 114 279 résolutions de sites et 692 010 relations. Un smoke
  réel de 500 candidats/5 scènes a projeté exactement 500 lignes, dont 366 avec
  plusieurs sources de noms, 258 avec site externe résolu et 77 avec relation.
  Il sera reconstruit sous un nouvel ID lorsque le backfill RNE complet sera
  disponible. Le backfill conserve six fenêtres vides 2000 déjà scellées; la
  septième attend la remise en service du port HTTPS INPI, actuellement refusé
  avant authentification.
- Les commits antérieurs `aa3bccd`, `0a9b49b`, `58c2f6a` et `ff00ad4`
  documentent respectivement le spill disque de l'overlay officiel, la borne
  des workers DuckDB, l'indexation incrémentale des couches temporelles et la
  correction des folds du retrieval commercial. Le fold 1 reste fermé.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub correspondant dans ce document.*
