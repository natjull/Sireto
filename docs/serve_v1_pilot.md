# Serve V1 — avenant AVANT FIT, pilote sans SIRET

## Recette unique autorisée
Références : coût `bbd211d8`, trois statuts `eaa66765`/`4e40f231`, ranker S1+S2-BASE94 et risque49 de `b90c4bc9`. Aucun nouveau signal, hyperparamètre, choix de modèle ou apprentissage Gemma. G0 initial100 inchangé ; V142 reste gelé. Production ici signifie un artefact de serving reproductible autorisé par ce brief, PAS une certification de précision sur la cible sans SIRET.

Les15,25/56,09/28,66% historiques concernent la politique adaptative choisie séparément dans chaque CV (rank_crm/base/risk_crm), pas un modèle unique. Le brief demande explicitement le reclasseur S1+S2-BASE de référence : figer **base94 + risque49**, sans colonnes CRM ni sélection des variantes. Publier séparément la comparaison aux trois parts historiques demandées et aux parts recalculées pour cette référence avec seuils OOF poolés ; ne pas annoncer leur identité par construction.

## Full-fit et seuils
Conserver tous les17427 dossiers/11483 entreprises,100 candidats naturels et absences, cibles exactes/UNION originales, folds population2/3/4 seulement. Aucun label corrigé ni poids0. Reclasseur : paramètres200arbres/profondeur3/eta.05/child10/L2=10/L1=1/hist128/mean8/seed20261017/jobs1 de la référence. Les sept features S1 de ses entrées d'apprentissage restent **crossfittées en trois plis entreprise**, même GroupKFold3/shuffle/seed ; ces trois auxiliaires NB sont nécessaires à la recette, ne constituent pas trois réglages. Un pipeline S1 final appris sur les17427 est ensuite figé pour servir (CountVectorizer et deux NB NAF paramètres/code de référence). S2-BASE deux features inchangées, réemploi des vecteurs vérifiés par texte/checkpoint/prompt/BF16/eager/mean masqué/L2/768/max512/batch4, jamais vecteurs adaptés.

Un seul R3 final appris sur toutes les lignes avec ces entrées. Un seul risque final49, paramètres/monotonies et200rounds référence, appris sur les **49features des top1 R3 OOF externes originaux** de chaque dossier, cible UNION correspondante, et non sur les scores in-sample du nouveau full-fit. Ainsi tous les17427 contribuent une fois au fit du risque sans cible fabriquée. Les OOF restent les anciens prédicteurs hors entreprise ; le transport OOF→full-fit (NB/R3/risque) est explicite et n'est pas garanti.

Seuils fixés AVANT lecture des sorties pilote : concaténer exactement les **scores risque base49 OOF externes** des17427, chacun lié à ses hits UNION originaux. SUR = plus petit score observé dont préfixe inclusif a borne Wilson bilatérale95 (z=1,959963984540054) >=97,5%, ex aequo indivisibles. Puis, dans la bande strictement sous SUR (ou tout si aucun SUR), PROBABLE = plus petit score observé dont préfixe a précision empirique>=90% ET Wilson bas>=88%. Aucun admissible=>bande vide. A_VERIFIER = reste. Même helper que `step2_crm_columns_tiers.py`, contrôle indépendant Wilson contractuel. Aucun réglage manuel/pilote. Ces seuils utilisent les OOF de développement comme calibration poolée : métriques obtenues sur ces mêmes17427 sont **de calibration**, pas une nouvelle précision test97,5%/90%.

## Serving, fidélité et sources
Script unique `scripts/sireto_serve_v1.py`, CSV colonnes export ; sortie une ligne par fiche : SIRET, SIREN, statut, risque, trois alternatives ordonnées. Modèles/paramètres/sources/hashes sous `H/serve_v1`, manifeste sans noms/IDs CRM dans le dépôt. Sources officielles readonly ; géocodage BAN local avec normalisation historique, aucune requête web. Les nouvelles fiches n'ont pas de fold humain : marqueur technique-1 dans le seul join géométrique, jamais faux fold2 ni feature. Manquants conservés ; G0 fournit100 candidats naturels même sans géométrie. Vecteurs nouveaux par texte, sans modifier les caches existants, encodés uniquement par `H/runtime/bin/python` ; un worker Metal à la fois, fichiers CPU/Metal liés par SHA. Cache persistant namespace propre, partage par texte compatible.

Fidélité :200 dossiers uniformes du banc seed20261021. Première preuve, entrées de serving reconstruites vs entrées85/S1/S2/49 originales (modèles OOF originaux si nécessaire). Deuxième preuve, nouveau full-fit : comparaison des scores du serving et du pipeline d'évaluation recalculé indépendamment sur les mêmes entrées/poids, rapporter écarts max R3/risque et top1. **Les nouveaux poids full-fit ne peuvent pas être identiques aux anciens poids OOF** ; publier l'écart full-fit vs OOF séparément, pas le faire passer pour défaut de sérialisation ni pour amélioration de précision. Smoke réel avant gros pilote, pas tests unitaires.

## Pilote et exclusions sans labels fermés
Export source immuable131749 lignes, cible booléenne `order_siret` vide ; aucune valeur SIRET raw utilisée comme label. Tirage uniforme sans remise de2000 lignes éligibles, RNG seed20261022, avant scores ; pas de sélection géographique, sur statut ou caches. Répétitions de services conservées et publiées. Pour être conservateur, exclure **toute la population historique37218**, pas seulement les jeux fermés : anti-jointure identifiant service/record et empreinte CRM canonique nom+adresse+CP+INSEE ; seules colonnes query-only de la population, jamais labels/pools fermés. Lecture des métadonnées d'appartenance query_id de test4323 et validation2212 pour vérifier leur inclusion dans ce sur-ensemble ; control962 est un sous-ensemble documenté de validation2212, donc tout validation2212 exclu. Folds0/1 et G0 vérifiés par ces mêmes métadonnées. Les champs CRM fermés ne servent qu'à calculer une empreinte d'exclusion non réversible, aucune sortie ou feature de développement ne les utilise. Si une appartenance ne peut être reliée au sur-ensemble, STOP avant pilote. Aucun label/SIREN humain fermé consulté ; exclusion par entreprise inconnue non revendiquée, impossibilité de certifier des doublons sémantiques non identiques explicitée.

Sorties nominatives exclusivement `H/serve_v1/pilot_2000/`, permissions0700/0600 ; aucune ligne/ID/service/nom/adresse pilote dans Git. Publier parts des3statuts, distribution risque et features clés nom/adresse/distance/candidats vs bancOOF, KS et PSI (déciles du banc, manquants séparés, epsilon1e-6), temps par étage, picRSS+Metal, nouveaux vecteurs/disque. Aucune précision sur pilote sans labels nouveaux. Préparer sans lancer une revue HTML autonome60 fiches (20/statut, seed20261023), CRM vs proposition seule, aucun label/score/statut visible, trois réponses Oui/Non/Je ne sais pas à « Cette société occupe-t-elle ce site ? », sauvegarde locale/export. Si un statut a<20 lignes, garder toutes, signaler le déficit sans duplication ni changement de seuil. Ne pas remplir/lancer les jugements.

## Budget, arrêt et publication
Budget10800s depuis lancement, attentes incluses, réserve publication120s. Worker CPU/Metal séquentiel, threads1, DuckDB512Mio sans spill, gardeRSS+driver5,75Gio (<6), pause si un autre Gemma est actif, ne jamais contrôler hydraulic niPID8288. Arrêt propre CATNAT<12Gio ou projection hors budget ; aucun nettoyage/suppression/installation/web/push, fichiers étrangers/pending intacts. Namespace neuf, reprise seulement après reçus complets et SHA, pas de fit incomplet réutilisé. STOP publié explicitement si partiel, aucun faux succès. Auto-publication agrégats/manifeste SHA `docs/serve_v1_pilot.{json,md}` + ligne handover, commit isolé après contrôle de non-concurrence du journal. Aucun déploiement CRM ni écrasement de SIRET existant.

## Résultats
Statut **COMPLETE_PILOT_NOT_PRECISION_CERTIFIED**, mur2680.4s, pic3.889Gio.

| Statut | Pilote sans SIRET n/% | Banc référence, calibration OOF poolée % | Politique adaptative historique % |
|---|---:|---:|---:|
| SUR | 279/13.95 | 20.80 | 15.25 |
| PROBABLE | 719/35.95 | 46.43 | 56.09 |
| A_VERIFIER | 1002/50.10 | 32.77 | 28.66 |

Fidélité full-fit200 : R3 max0.0, risque max0.0, top1200/200. L’écart aux anciens poids OOF est distinct : ranking2.359623432159424, risque0.531751275062561.
Sélection :{'raw_without_siret': 88465, 'eligible': 86420, 'exclusions': {'fingerprint_excluded': 322, 'service_excluded': 1723}, 'selected': 2000, 'seed': 20261022, 'selected_distinct_services': 1989, 'population_rows': 37218, 'closed_fold0': 7009, 'closed_fold1': 6622, 'test4323_membership': 4323, 'validation2212_superset_control962': 2212, 'overlap_fold0': 0, 'overlap_fold1': 0, 'overlap_test4323': 0, 'overlap_control962': 0, 'overlap_g0': 0, 'method': 'Exclude all37218 population records by service OR canonical CRM fingerprint; test+validation query_id projections verified subsets. control962 documented subset of validation2212, all excluded.', 'no_company_disjointness_claim': True}. Aucun label fermé utilisé ; exclusion par entreprise non revendiquée. Les noms/empreintes d’exclusion ne servent pas aux features.
Coûts :{'preparation': {'queries': 2000, 'seconds': 552.4641490001231, 'stages_seconds': {'ban_geocoding': 14.47747949976474, 'generation_initialization': 0.32443662360310555, 'generation': 455.38100434746593, 'content85_hydration_features': 44.66875604074448, 'candidate_metadata_neighbours_density': 16.88610575068742}, 'generation_p95_seconds': 0.4664861921221018, 'new_labels_read': False}, 'cache': {'texts': 189614, 'new_texts': 147627, 'reused': 41987, 'base_shards_verified': 1233, 'seconds': 9.12648204062134}, 'encoding': {'new_vectors': 147627, 'seconds': 1848.9359632916749, 'load_seconds': 2.323321125470102, 'truncated': 0, 'peak_bytes': 2982821888, 'adapter': None}, 'semantic': {'seconds': 3.148016417399049}}. Inférence :{'s1_load_and_inference': 2.03758158441633, 'ranker_load_and_inference': 0.4324531676247716, 'risk_features_load_inference': 0.6395126255229115}. Disque namespace1.196Gio, libre16.041Gio.
Revue préparée NON lancée : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/serve_v1/pilot_2000/review_60/jugement.html`, 60 fiches, {'SUR': 20, 'PROBABLE': 20, 'A_VERIFIER': 20}. Statut/score cachés, aucune réponse préremplie. Aucun navigateur ni jugement lancé.

| Variable | PSI | KS | Médiane banc / pilote |
|---|---:|---:|---|
| risk_score | 0.2629 | 0.20296557066620763 | 0.8930219411849976 / 0.7548013925552368 |
| name_similarity | 0.3231 | 0.17539602341194696 | 0.9090909361839294 / 0.8615384697914124 |
| street_similarity | 0.0590 | 0.10506745280312157 | 1.0 / 1.0 |
| log_distance | 0.0714 | 0.13259681917363858 | 3.7131290435791016 / 3.188814640045166 |
| distance_missing | 0.0053 | 0.027845900040167515 | 0.0 / 0.0 |
| number_missing | 0.0565 | 0.09792597693234628 | 0.0 / 0.0 |
| candidate_count | 0.0000 | 0.0 | 100.0 / 100.0 |
| crm_name_length | 0.3433 | 0.2275525621162564 | 19.0 / 13.0 |

Parts de statuts ≠ précision. OOF poolé calibré sur lui-même, transport vers full-fit et décalage de population : aucune garantie98,23% sur les sans-SIRET. Recommandation : faire juger les60 fiches avant automatisation ; ne pas écraser les SIRET existants.

## Attestation finale et entrée CSV réelle

Avenant avant fit `b47b1390`, implémentation `1cfc2ecb`, auto-publication `052e51b9`. Vérification réelle supplémentaire sans fit :2000 vecteurs R3 et2000 risques rejoués depuis les poids sérialisés,2000 top4/SIREN/statuts/CSV et100 candidats uniques par dossier vérifiés ; poids200arbres et SHA des modèles/sources référencées inchangés. `H/serve_v1/postverification.json`, référence SHA et preuve intégrale dans le JSON public. Contrôle jusqu’à2962,920s=49,38min depuis lancement, sous3h ; run principal2680,411s=44,67min. Aucun calcul de précision sur le pilote non labellisé.

**Smoke de l’entrée CSV réelle** : dix premières lignes du pilote, export11colonnes, commande `serve` avec nouveau dossier privé. Identité stricte des dix classements100, scores de risque, top4 et statuts avec le run2000 ; écarts max R3/risque0,1003 textes réutilisés/0 nouveau vecteur. Références/agrégats dans `postverification.csv_cli_smoke`, nominatif uniquement sous `pilot_2000/cli_csv_smoke10`. Les anciens poids OOF sont différents des poids full-fit : l’écart non nul annoncé plus haut n’est pas une erreur de fidélité ni un gain de précision.

Seuils score figés : SUR **0,9629830718040466**, PROBABLE **0,7579634189605713** (bande sous SUR). Calibration poolée : SUR3553/3625=98,0138%, Wilson[97,5062;98,4198] ; PROBABLE7282/8091=90,0012%, Wilson[89,3285;90,6360]. Ces chiffres sont calculés sur les dossiers ayant servi à choisir les seuils, jamais une validation indépendante ni un transfert des98,23% historiques.

Décalage complémentaire : numéro inexploitable côté CRM OU candidat27,00% banc/17,20% pilote ; distance manquante16,37%/19,15% ; distance typée médiane40,0m/23,3m parmi les mesurables. Les adresses ne sont donc pas uniformément moins bonnes : noms plus courts/moins proches et transport du modèle contribuent aux observations, sans attribution causale prouvée.

Confidentialité : toutes les données nominatives produites (pilote, fixtures fidélité, cache des nouveaux textes, revue) résident **physiquement sous `H/serve_v1/pilot_2000`**,0700/0600. Deux liens de compatibilité sans contenu nominatif à la racine (`fidelity_200`, `shared_new_texts.json`) conservent les références scellées et pointent dans ce répertoire ; aucun poids/cache étranger déplacé ni aucune donnée supprimée. Modèles et métadonnées agrégées restent à la racine autorisée. Espace après smoke/contrôles :**1,200Gio** alloués,**15,989Gio libres**, au-dessus de12Gio.

Revue non lancée :60cartes/180boutons,aucune réponse précochée,20cas distincts/statut ; clé privée et poids de sondage sous `review_60/private_key.json`. Poids d’expansion vers le pilote :SUR13,95/PROBABLE35,95/A_VERIFIER50,10 ; aucune réponse remplie. Ne partager que le HTML, pas la clé ;20cas par statut ne suffiront pas à certifier98% même si tous sont jugés corrects.

Usage (un lot à la fois, dossier neuf ; sorties `proposals.csv` et `proposals.jsonl`, ordre d’entrée conservé, `SERVE:n`=indice de ligne) :
```bash
H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/sireto_serve_v1.py serve \
  --input /chemin/export.csv --output-dir "$H/serve_v1/pilot_2000/lot_neuf"
```
Le manifeste figé `docs/serve_v1_manifest.json` est identique au manifeste privé : trois poids SHA, paramètres,49features risque/94features ranking, seuils, reçus de fit et sources. Aucun réentraînement implicite par `serve`, aucun écrasement CRM.
