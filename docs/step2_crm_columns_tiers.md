# Étape2-13 — colonnes CRM et trois statuts : avenant AVANT FIT

## Autorités, périmètre, absence de déploiement implicite

Audit CRM `390fd5e1`, référence top1 S1+S2-BASE et risque49 `b90c4bc9`/attestation `0b086f29`, nesting risque `2cf01765`, G0 `9a8ecdd6`.17427 dossiers/11483 entreprises, mêmes100 candidats, cinq CV entreprise0–4 (distincts des folds population2/3/4). Aucun changement V142/labels/pools, positif injecté, FORT retiré, alias/identifiant feature, accès folds0/1/test4323/control962/web/installation/push.526 déjà consommés : descriptions seules, jamais fit/encodage/calibration/sélection. La politique à trois niveaux ci-dessous est l'avenant métier autorisé par ce brief ; aucun déploiement ni certification produit n'est implicite.

## Provenance et contrôle anti-fuite AVANT apprentissage

Lire seulement les métadonnées query_id/crm_record_id/data_origin/oof_fold des17427+526 autorisés, projection filtrée des queries de la population `73328b7611f37a06`. Jointure par service pour la seule origine récente, jamais par SIRET/nom/label ; vérification unicité du couple catégorie/état par service et identité de la cohorte. Les historiques n'ont pas ces colonnes : MISSING explicite, sans imputation ni lookup de vérité. L'export brut n'est consulté côté label que par agrégation du booléen `order_siret` présent, jamais sa valeur ni les labels des autres dossiers. Ne pas lire gt_siret historique. Aucun identifiant, source, fold ou présence de label dans les matrices.

Publier les effectifs/rates de présence brute du label par catégorie et état, leurs associations Cramer V avec ce booléen, et dans G0 les tables par CV entreprise et fold population avec Cramer V, pureté modale et taux de présence exact/UNION parmi100. G0 est entièrement labellisé par construction, ce qui est dit explicitement. Analyser également le sous-ensemble récent pour séparer le mécanisme MISSING historique. STOP avant fit si jointure ambiguë, label/source modifié, modalité commune (>=1% des G0) prédit parfaitement un fold, ou V>=0,5 avec fold/CV ; les petites cellules sont signalées, pas une preuve de fuite. Une association métier à la présence brute n'est pas une preuve de fuite d'identité SIRET ; l'absence d'horodatage empêche d'exclure une fuite temporelle. Disponibilité à l'entrée acceptée sur déclaration du brief, pas vérifiée rétrospectivement. Pas de transformation postérieure fondée sur les résultats de cet audit.

## Encodage et usages A figés

Encoder distinct pour CHAQUE fit, uniquement sur ses requêtes d'apprentissage. Catégorie trim/casse conservée, modalités >=1% des catégories non manquantes du train, OTHER pour rares/inconnues, MISSING séparé ; états one-hot des valeurs observées dans le train + OTHER/MISSING. Pas d'encodage ordinal de la modalité, pas de fréquence/target encoding comme feature. Les attributs sont répétés sur les100 candidats ; ajouter exactement état×`v2_is_closed` et catégorie×`v2_is_siege`. Les autres signaux actif/secondaire/effectifs éventuellement disponibles restent les colonnes publiques content85 historiques, pas de nouvelle hydratation. La constante d'une requête seule ne départage pas ses candidats ; les croisements et arbres permettent des interactions.

Trois références appariées, aucun HPO/poids0 :
- `base` :94features content85+S1sept+S2deux,25R3 et5risques49 déjà scellés/rejoués réutilisés. Scores/métriques attendus12275exacts/13382UNION. Risque49 distinct de l'ancien46, non confondu.
- `rank_crm` : ajouter les one-hot et deux croisements au seul reclasseur ; risque49 historique, recalculé sur ses propres top1/marges/signaux, SANS ajout direct des colonnes CRM au risque.25nouveaux R3 +5risques49.
- `risk_crm` : reclasseur94/OOF inchangé, one-hot et croisements top1 ajoutés au risque49 ;5risques supplémentaires. Encoder risque ajusté au proper seulement, figé pour calibration/test/526.

Mêmes paramètres LTR200arbres/profondeur3/eta0,05/child10/L2=10/L1=1/hist128/mean8/seed20261017/threads1. Risques mêmes paramètres200arbres, monotonies+1 margins12/13 seulement,0 nouvelles colonnes. Aucun nouveau fit NB/Gemma/adaptateur : S1 proper/inner/cal/outer exactement rejoués depuis les modèles originaux et leurs SHA features94/cibles, S2-BASE/cosines immuables. Pour chacun des cinq CV :3R3 proper OOF,1proper→cal25%entreprises,1outertrain→test+526, rôles originaux. Risque entraîné sur proper OOF, jamais sur calibration/test. Transport proper→refit outertrain déclaré. Smoke réel100entreprises entières du properCV0, tirage SHA `20261020|crm-columns-smoke|`, modèle LTR sérialisé et scores avant/après vérifiés ; risque smoke100entreprises proper, scoring calibration originale, aucun choix de recette sur smoke.

Mesures A : top1 exact/UNION, IC bootstrap10000entreprises query-weighted seed20261017, appariés vsbase (y compris simultanés), IC absolus ; totaux et par CV/populationfold. Courbes SUR aux cibles Wilson calibration96/97/97,5/98% : plus petit score observé admissible/maxcouverture, ex aequo indivisibles, borne bilatérale95 z1,959963984540054 ; None=>0SUR/précision indéfinie, jamais seuil choisi sur test. Modèles sérialisés, vrais inputs/scores rejoués, vérification indépendante des seuils par helper contractuel. Référence risque49 existante strictement reproduite.

## B — politique complète sélectionnée uniquement dans l'apprentissage

SUR : même règle Wilson calibration>=97,5%. Puis PROBABLE : chercher sur la BANDE restante sous le seuil SUR le préfixe score maximal dont précision empirique>=90% ET Wilson bas>=88%. Cette condition porte sur les PROBABLE exclusifs, pas sur un mélange avec les SUR qui pourrait cacher une bande médiocre. None=>bande vide. A_VERIFIER = reste. Les seuils score sont inclusifs, aucune coupure d'ex aequo. Les dossiers des trois statuts et toutes les absences restent au bilan.

« Meilleure variante de A, sinon référence » interprétée sans fuite de sélection : choix SÉPARÉ dans chaque CV, sur sa calibration d'apprentissage uniquement, jamais via des OOF d'autres CV contenant l'entreprise test. `rank_crm` admissible seulement si son taux top1 UNION calibration>=référence ; `risk_crm` conserve exactement le top1. Maximiser lexicographiquement nombre SUR puis nombre SUR+PROBABLE en calibration ; égalité=>base puisrisk_crm puisrank_crm. Une variante ne remplace base que si elle améliore ce couple. C'est la politique adaptative entraînée qui est évaluée, pas un gagnant global choisi sur les tests. Publier en plus les trois politiques fixes pour transparence, et une indication exploratoire de gain top1 A via IC, jamais une autorisation de promotion. Sélection et recherche de seuils partagent calibration : Wilson adaptatif et corrélations entreprise ne donnent pas de garantie produit, même si le test paraît favorable.

Pour chaque statut :n/N, couverture, précision UNION et exacte/Wilson, pooled/parCV/parfold ; vue par catégorie produit BRUTE (rares incluses, MISSING historique séparé) sans suppression des petites cellules. Courbe précision–couverture10seuils aux déciles des scores CALIBRATION (quantile higher, pas des scores test), ex aequo regroupés ; publier aussi les bandes de décile disjointes, cellules vides non estimables. Les quatre meilleures propositions des100 sont toujours exportées, tri décroissant stable (ordre G0 en égalité), identifiants en sortie/jointure seulement. JSONL privé G0 et526 avec query_id/score/statut/top4SIRET, aucun label de vérité exporté. Pas de fiche sans proposition.

526 : moyenne des cinq vecteurs ranking ; pour chacun des trois usages et pour la politique sélectionnée par CV, statut SUR si les cinq top1 concordent avec le top1 moyen ET cinq SUR, PROBABLE si accord et tous au moinsPROBABLE, sinonA_VERIFIER. Aucun seuil/choix/fit sur526 ; top4 issus de la moyenne. Leurs métadonnées ne peuvent affecter aucun vocabulaire.

## Ressources, scellés et publication

Namespace neuf `H/step2_crm_columns_tiers_20261020`, CPU un worker séquentiel, threads1/offline, DuckDB256/512Mio sans spill ; aucun cache Gemma ni gros cache neuf. Garde8Gio RSS sans VM détectée/5,75 sinon, métadonnées process uniquement, ne jamais agir sur hydraulic. Attente si grosV142>4Gio/VM+freeinactive<8Gio ; tout temps d'attente compte dans les3h. CATNAT arrêt propre<12Gio, aucune suppression. Budget strict10800s depuis lancement, réserve publication120s, projection conservatrice après1erfit complet, STOP si hors budget sans réduire le banc. Code/avenant commis avant fits ; sources/labels/pools/vocabulaires/rôles/features/cibles/models SHA avant/après, modèles rejoués, smokes réels, aucun test unitaire.

Publication automatique `docs/step2_crm_columns_tiers.{json,md}` et ligne append-only `handover.md`, commit isolé explicite, aucun push. Refus chemins sales/concurrents/JSONpréexistant, fichiers étrangers etpending intouchés. STOP explicite, jamais un partiel promu. Rapport final<=25lignes :anti-fuite/tableauA/tableau3statuts/durée/commits/une recommandation ; tableaux complets A SUR/déciles/catégories dans les rapports détaillés. Une réussite ici reste développement sur jeux consommés ; validation indépendante et vérification temporelle restent nécessaires avant production.

## Résultats

**Fait** —17427/11483, mêmes5CV/OOF ; `scripts/step2_crm_columns_tiers.py`, rapports `docs/step2_crm_columns_tiers.{json,md}`, handover.
**Anti-fuite** —Jointure non ambiguë ; V maximal folds/CV=0.073. Présence label raw selon état : terminated 26.04%, active 41.93% ; temporalité non vérifiée.

| A | Exact % [IC95 entreprise] | UNION % [IC95 entreprise] | ΔUNION points [IC95 apparié] |
|---|---|---|---|
| base | 70.44 [69.15, 71.73] | 76.79 [75.51, 78.04] | — |
| rank_crm | 70.51 [69.21, 71.81] | 76.84 [75.55, 78.11] | +0.05 [-0.05, 0.16] |
| risk_crm | 70.44 [69.15, 71.73] | 76.79 [75.51, 78.04] | +0.00 [0.0, 0.0] |

| B statut | n / part % | Précision UNION % [Wilson95] | Précision exacte % [Wilson95] |
|---|---|---|---|
| SUR | 2658 / 15.25 | 98.23 [97.66, 98.67] | 95.22 [94.34, 95.97] |
| PROBABLE | 9774 / 56.09 | 89.12 [88.49, 89.73] | 80.07 [79.27, 80.85] |
| A_VERIFIER | 4995 / 28.66 | 41.20 [39.84, 42.57] | 38.38 [37.04, 39.74] |

**Écarts / choix** —B choisie par CV sur calibration seulement, jamais gagnant global OOF ; PROBABLE calibré en bande exclusive. Historiques7530=MISSING.
**Preuve** —`run` puis vérification réelle25LTR/10risques+2smokes/90seuils, top4 des17427 ; 61.85min, pic4.63Gio.
526 descriptifs seuls ; courbes A96/97/97,5/98, déciles et catégories détaillés dans les rapports ; labels/pools/jeuxfermés/VM/tiers intacts.
**Reste / risques** —Jeux consommés, sélection/calibration adaptatives, temporalité et validation indépendante manquent ; aucun déploiement.
Recommandation unique : conserver S1+S2-BASE et réserver PROBABLE à la revue ; ne pas automatiser cette politique avant validation temporelle et indépendante.

## Attestation, anti-fuite et limites

Avenant/code AVANT FIT `0936dc48`, auto-publication `eaa66765`. Run+publication3711,470s=61,86min ; contrôle complémentaire achevé3808,354s=63,47min depuis lancement, sous10800s, pic4970921984octets=4,63Gio. La projection prudente initiale8508,927s n'a pas imposé de réduction. Aucun nouveau cache Gemma, label/pool/jeu fermé/source modifié, VM contrôlée, fichier tiers/pending touché ou push. Les deux smokes réels et25LTR+10risques nouveaux ont abouti du premier lancement ; la référence25R3/5risques49 reste scellée/réutilisée. Toutes les entreprises apprentissage/calibration/test sont disjointes selon les groupes officiels.

Contrôle principal :25LTR/89765vecteurs ranking et leurs49features risque/cibles rejoués,15risques (10nouveaux/5réutilisés),90contrôles de seuils par Wilson contractuel (60préfixes A et30statuts, SUR97,5 étant aussi revérifié),1450référencesSHA. `H/step2_crm_columns_tiers_20261020/postverify.py` passé sans fit :25encoders recalculés indépendamment,500requêtes réelles/1500croisements candidat contrôlés,15politiques de bandes calibration vérifiées, tous déciles et comptes produit recomposés,19531sorties top4/statuts vérifiées (17427G0+4×526descriptifs), code/protocole et sourceGit-externe identiques. Preuve `postverification.json` SHA `ed9d480105cc58eac45cd29129bb2e084eb49773dec4a04cd21c1905af45325d`.

### Présence brute `order_siret` par modalité — audit seulement

| Colonne / modalité | Lignes raw | Label non vide | Présence % |
|---|---:|---:|---:|
| Catégorie : Fibre Essentiel | 69865 | 27473 | 39,32 |
| Catégorie : ADSL | 24106 | 3926 | 16,29 |
| Catégorie : Revente ADSL | 21412 | 4945 | 23,09 |
| Catégorie : Revente Fibre Essentiel AI | 13467 | 5984 | 44,43 |
| Catégorie : SDSL | 2027 | 433 | 21,36 |
| Catégorie : Fibre Confort | 568 | 262 | 46,13 |
| Catégorie : Fibre Premium | 283 | 244 | 86,22 |
| Catégorie : Revente Fibre Premium AI | 21 | 17 | 80,95 |
| État : terminated | 75239 | 19590 | 26,04 |
| État : active | 56510 | 23694 | 41,93 |

Aucune valeur SIRET raw n'a été extraite ; ce booléen est seulement non-vide, pas une validation de qualité du label. Association de présence : Vcatégorie0,22419/Vétat0,16746, **donc les colonnes ne sont pas indépendantes de la disponibilité du label**. Ceci peut refléter le processus métier/sélection, et ne prouve ni l'absence ni la présence d'une fuite temporelle. Les catégories Premium rares àforte présence ne doivent pas être présentées comme une preuve de qualité du modèle. Dans G0,100% des dossiers ont un label humain par construction ; les taux exact/UNION parmi100 parmodalité, plus instructifs ici, sont dans le JSON.7530historiques sontMISSING, sans renseignement fictif depuis une autre source.

| V de Cramer | CV G0 | Fold population G0 | CV récents seuls | Fold récents seuls |
|---|---:|---:|---:|---:|
| Catégorie produit | 0,05991 | 0,05088 | 0,07346 | 0,06464 |
| État service | 0,04327 | 0,02520 | 0,05737 | 0,03848 |

Pas de proxy parfait commun de fold ni deV>=0,5 observé, jointures sansambiguïté. L'absence de dates reste un **blocage de certification temporelle**, pas résolu par cesV faibles. L'encodage n'a utilisé que les requêtes du train de chaque fit ; ni le test ni526 n'ont servi à déterminer les modalités ou les seuils.

### Ce que les résultats permettent de conclure

A :12275/13382exacts/UNION référence ;12287/13391avec colonnes au reclasseur (+12/+9), soitUNION47gains/38pertes et+0,05164point[-0,05115;0,15723]. IC ordinaire et simultané incluent0 ; **aucun gain top1 confirmé**. Usage risque seul :mêmes propositions/classement qu'avant, n/N inchangés. Àciblecal97,5 :risque seul2027/2058UNION=98,4937%,C11,8093%,contre1899/1928=98,4959%,C11,0633%pourréférence ; effet descriptif, pas certification de gain produit.

B :choix uniquement sur calibration de chaqueCV :0=rank_crm,1=base,2/3/4=risk_crm. Ce n'est PAS un gagnant global sélectionné sur les tests. SUR2611/2658UNION=98,23%[97,66;98,67],2531/2658exacts95,22%,C15,25% ; la borne Wilson brute test dépasse97,5 mais reste descriptive sur ces jeux consommés et corrélés. PROBABLE8711/9774UNION=89,12%[88,49;89,73],7826exacts80,07%,C56,09% : **les90% de calibration ne se transportent pas au test**, même si la borne test dépasse88%. A_VERIFIER2058/4995UNION41,20%,1917exacts38,38%,C28,66%. La somme couvre17427, tous avec4propositions ; SUR+PROBABLE=12432dossiers/71,34%, ce qui n'autorise pas à appeler71,34%desrattachements fiables/automatiques.

526, politique sélectionnée parCV puis appliquée descriptivement :387exacts/410UNION ;SUR63/63UNION100%[94,25;100],61/63exacts ;PROBABLE267/289UNION92,39%[88,74;94,92] ;A_VERIFIER80/174UNION45,98%. Le63/63 n'est pas une précision produit garantie, et n'a modifié aucun paramètre. Les scores dans les sorties sont des scores de risque, pas une probabilité certifiée d'occupation fibre. Aucun module/contrat de production existant n'a été changé : les propositions/statuts de ce jalon sont des sorties de développement.

## Courbes A — précision UNION / couverture %

| Cible Wilson calibration | base | rank_crm | risk_crm |
|---|---|---|---|
| 96 | 96.49 / 42.17 | 96.63 / 42.96 | 96.56 / 40.90 |
| 97 | 97.81 / 25.45 | 97.49 / 25.84 | 97.71 / 24.05 |
| 97.5 | 98.50 / 11.06 | 97.88 / 7.85 | 98.49 / 11.81 |
| 98 | 99.26 / 1.55 | 98.06 / 2.96 | 98.68 / 3.04 |

## Courbe B — seuils déciles de calibration, cumuls et bandes disjointes

| Préfixe calibration % | Couverture test % | UNION cumul % | Exact cumul % | n bande | UNION bande % |
|---|---|---|---|---|---|
| 10 | 10.78 | 98.46 | 96.49 | 1879 | 98.46 |
| 20 | 20.86 | 97.99 | 94.53 | 1756 | 97.49 |
| 30 | 31.33 | 97.38 | 92.49 | 1825 | 96.16 |
| 40 | 41.33 | 96.58 | 90.41 | 1742 | 94.09 |
| 50 | 51.07 | 95.76 | 88.69 | 1698 | 92.29 |
| 60 | 61.11 | 93.91 | 86.29 | 1749 | 84.45 |
| 70 | 70.65 | 91.41 | 83.59 | 1664 | 75.42 |
| 80 | 80.40 | 88.10 | 80.67 | 1698 | 64.13 |
| 90 | 90.43 | 83.15 | 76.25 | 1748 | 43.48 |
| 100 | 99.98 | 76.80 | 70.45 | 1664 | 16.59 |
| below_calibration_min | — | — | — | 4 | 0.00 |

## B par catégorie produit brute — toutes cellules, historiques séparés

| Catégorie | Statut | n | Part interne % | UNION % [Wilson95] | Exact % [Wilson95] |
|---|---|---|---|---|---|
| ADSL | SUR | 24 | 3.09 | 95.83 [79.76, 99.26] | 95.83 [79.76, 99.26] |
| ADSL | PROBABLE | 385 | 49.61 | 89.09 [85.58, 91.83] | 82.34 [78.21, 85.82] |
| ADSL | A_VERIFIER | 367 | 47.29 | 44.96 [39.95, 50.07] | 43.87 [38.88, 48.98] |
| Fibre Confort | SUR | 14 | 19.44 | 100.00 [78.47, 100.0] | 100.00 [78.47, 100.0] |
| Fibre Confort | PROBABLE | 42 | 58.33 | 85.71 [72.16, 93.28] | 76.19 [61.47, 86.52] |
| Fibre Confort | A_VERIFIER | 16 | 22.22 | 62.50 [38.64, 81.52] | 56.25 [33.18, 76.9] |
| Fibre Essentiel | SUR | 902 | 14.30 | 97.56 [96.33, 98.38] | 93.46 [91.65, 94.9] |
| Fibre Essentiel | PROBABLE | 3684 | 58.41 | 89.01 [87.96, 89.98] | 78.50 [77.15, 79.8] |
| Fibre Essentiel | A_VERIFIER | 1721 | 27.29 | 43.87 [41.54, 46.23] | 40.73 [38.43, 43.07] |
| Fibre Premium | SUR | 3 | 9.38 | 66.67 [20.77, 93.85] | 66.67 [20.77, 93.85] |
| Fibre Premium | PROBABLE | 16 | 50.00 | 93.75 [71.67, 98.89] | 93.75 [71.67, 98.89] |
| Fibre Premium | A_VERIFIER | 13 | 40.62 | 30.77 [12.68, 57.63] | 30.77 [12.68, 57.63] |
| Revente ADSL | SUR | 62 | 5.76 | 100.00 [94.17, 100.0] | 98.39 [91.41, 99.71] |
| Revente ADSL | PROBABLE | 561 | 52.14 | 82.35 [78.98, 85.28] | 77.18 [73.53, 80.46] |
| Revente ADSL | A_VERIFIER | 453 | 42.10 | 41.72 [37.27, 46.31] | 39.96 [35.55, 44.53] |
| Revente Fibre Essentiel AI | SUR | 283 | 18.08 | 98.59 [96.42, 99.45] | 96.47 [93.62, 98.07] |
| Revente Fibre Essentiel AI | PROBABLE | 864 | 55.21 | 89.70 [87.49, 91.55] | 82.18 [79.48, 84.58] |
| Revente Fibre Essentiel AI | A_VERIFIER | 418 | 26.71 | 34.69 [30.28, 39.37] | 31.34 [27.08, 35.94] |
| Revente Fibre Premium AI | SUR | 0 | 0.00 | indéf. | indéf. |
| Revente Fibre Premium AI | PROBABLE | 0 | 0.00 | indéf. | indéf. |
| Revente Fibre Premium AI | A_VERIFIER | 4 | 100.00 | 0.00 [0.0, 48.99] | 0.00 [0.0, 48.99] |
| SDSL | SUR | 4 | 6.15 | 100.00 [51.01, 100.0] | 100.00 [51.01, 100.0] |
| SDSL | PROBABLE | 40 | 61.54 | 87.50 [73.89, 94.54] | 82.50 [68.05, 91.25] |
| SDSL | A_VERIFIER | 21 | 32.31 | 52.38 [32.37, 71.66] | 42.86 [24.47, 63.45] |
| __MISSING__ | SUR | 1366 | 18.14 | 98.61 [97.84, 99.11] | 95.97 [94.8, 96.89] |
| __MISSING__ | PROBABLE | 4182 | 55.54 | 90.05 [89.11, 90.92] | 81.16 [79.94, 82.31] |
| __MISSING__ | A_VERIFIER | 1982 | 26.32 | 39.30 [37.18, 41.47] | 36.38 [34.29, 38.52] |

Les déciles ne coupent pas les ex aequo et ne garantissent pas10% de dossiers test par bande ; la queue sous le minimum calibration est publiée séparément. Les IC de couverture sont descriptifs. Les JSON incluent tous n/N, Wilson, associations anti-fuite, résultats parCV/fold et politiques fixes. Propositions privées et modèles/reçus : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/step2_crm_columns_tiers_20261020`. Choix calibration parCV : `{'0': 'rank_crm', '1': 'base', '2': 'risk_crm', '3': 'risk_crm', '4': 'risk_crm'}`.
