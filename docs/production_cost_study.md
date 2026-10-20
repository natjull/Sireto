# Resiretisation de la base CRM — étude de coût

Étude du 2026-10-20T08:44:48.691862+00:00. **Mesures locales, estimations et inconnues séparées ; aucune promotion production.**

## 1. Périmètre et inventaire mesuré
Export : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_new_data_20260817/source/plateforme_new_data_to_cure.csv` (SHA dans le JSON). **131749 lignes / 105351 services distincts / 102495 entrées CRM brutes distinctes**, 9922 communes renseignées.
SIRET existant présent/absent : **43284 / 88465** ; présents tous de 14 chiffres (ni validité Luhn ni rattachement vérifiés). INSEE présent/absent : **131586 / 163** ; format cinq caractères, pas preuve de validité : 16 codes absents du snapshot officiel.
Les 26 398 répétitions de service ne sont pas supprimées : plusieurs lignes peuvent porter des entrées différentes. Les exports historiques d’apprentissage ne sont pas additionnés à cet export, sous peine de doubles comptes. Une déduplication des seules entrées strictement identiques serait possible après validation métier ; projections conservatrices sur toutes les lignes.
Le champ source `order_siret` est compté par présence uniquement, jamais utilisé comme vérité. Aucun label fermé ouvert. Comptage par projection `oof_fold` seulement : fold0 **7009**, fold1 **6622** ; métadonnées existantes seulement : test4323 **4323**, control962 **962**, non recomptés individuellement. Ce ne sont pas quatre populations disjointes à additionner. Ledger développement : 7767/7967/7837 lignes folds2/3/4.

| Taille de commune : établissements officiels tous états | Communes | Lignes CRM |
|---|---:|---:|
| 0–<1000 | 5675 | 13197 |
| 1000–<10000 | 3670 | 35597 |
| 10000–<50000 | 482 | 38811 |
| 50000–<100000 | 53 | 15486 |
| 100000–<1000000 | 42 | 28495 |
Fiches par commune : médiane2, p95=45, maximum1616. Sites officiels par commune CRM : médiane777, p95=11 879.65, maximum503476. Union : **34839788 sites** ; **8671847968 couples dossier×site** (hors163 lignes sans commune).

## 2. Rejeu G0 réel sur 200 dossiers
Tirage uniforme sans remise seed20261021 dans les17427 requêtes des folds2/3/4, avant scores ; identifiants/indices et plis dans le JSON. **20000 candidats régénérés** ; 19893 SIRET uniques ; candidats ordonnés,85 features,200 vecteurs ranking et200 risques identiques aux artefacts scellés. Aucun fit ni évaluation de précision nouvelle.
| Étage mesuré | Secondes / 200 | Secondes / fiche | Projection131749 (h) |
|---|---:|---:|---:|
| generation_init | 0.41 | 0.002038 | 0.07 |
| generation | 115.17 | 0.575851 | 21.07 |
| content85_hydration_features | 45.10 | 0.225512 | 8.25 |
| candidate_metadata_neighbours_density | 2.98 | 0.014883 | 0.54 |
| s2_cache_verify_read_cosines | 11.37 | 0.056866 | 2.08 |
| s1_load | 0.12 | 0.000625 | 0.02 |
| s1_inference | 0.46 | 0.002315 | 0.08 |
| ranker_load | 0.06 | 0.000304 | 0.01 |
| ranker_inference | 0.10 | 0.000478 | 0.02 |
| risk_load | 0.07 | 0.000348 | 0.01 |
| risk_scene_inference | 0.01 | 0.000058 | 0.00 |
Somme des étages : **0.8793s/fiche**, soit **32.18h** extrapolées. Mur du rejeu hors imports : 176.79s ; pic RSS **2.957Gio**, garde3,85Gio. Génération p50/p95/p99 : {'p50': 0.47821595799177885, 'p95': 1.1395711059682068, 'p99': 1.509440507534878}. Démarrages/chargements des cinq modèles OOF compris ; pas cinq inférences par fiche.
**Chemin avec caches, pas temps froid complet** : géométrie CRM déjà disponible ; 19473 clés BASE exactes couvertes, 1207 shards SHA vérifiés, **0 nouveau texte Gemma**. La phase S2 chronométrée comprend lecture/vérification/cosines, pas un encodage fictif. Les textes candidats officiels nom+NAF ont été reconstruits ; correspondance par texte/encodeur, pas seul SIRET. S1 utilise le nom normalisé et ses modèles NB sérialisés, sans entraînement.
Les mêmes candidats SIRET répétés partagent un vecteur seulement si le texte, checkpoint, prompt, troncature et binding de batching sont identiques ; le vecteur requête reste propre au nom. Les cosines et classements restent par dossier. Vérifier les SHA des shards une fois par banque et utiliser des lookups nationaux persistants éviterait les scans répétés par lot ; gain non mesuré/non implémenté.

### G0 à froid : estimations conditionnelles
Débit BASE existant : 0.006420s/nouveau texte (958595 nouveaux en6154,344s, `step2_gemma_finetune_10h_20261019/full/base_encode/receipt.json`). Référence SITE plus lente :0,022821s/texte (`docs/step2_error_families_g0.md`). Pas le même mélange de longueurs ; intervalle de scénarios, pas une garantie. Requêtes CRM à encoder : environ 0.23–0.84h supplémentaires si aucune réutilisation.
| Réutilisation/dédup hypothétique des13 174 900 expositions | Nouveaux textes candidats | Encodage (h) | Vecteurs f32/f16 (Gio) |
|---|---:|---:|---:|
| 0% | 13174900 | 23.50–83.52 | 37.69/18.85 |
| 50% | 6587450 | 11.75–41.76 | 18.85/9.42 |
| 80% | 2634980 | 4.70–16.70 | 7.54/3.77 |
Ajouter le chemin chaud (32.18h), l’encodage des noms CRM et les nouveaux textes candidats : **pas de total unique honnête sans mesurer l’union des candidats sur toute la base**. Le scénario sans partage reste une capacité maximale de100 textes/fiche, pas une mesure de l’union. Banque BASE existante1266170 textes≈3,62Gio, à réutiliser si compatible ; aucun gain national de recouvrement supposé. Features94 persistées sur toutes les lignes≈4.61Gio ; préférer lots bornés et export final seulement.

## 3. V142 : estimation sans nouvel encodage
Snapshot officiel : **43896818 sites = 16715258 actifs + 27181560 fermés**, dont 38301607 publics. Les82 772 691 documents Tantivy ne sont PAS autant de sites. Aucun filtre de fermeture validé pour retirer des sites : tous les fermés conservés, ni label ni règle de cas. Pertinence temporelle des fermés inconnue ; la restreindre changerait la recette gelée.
Banque SITE nationale768 : **125.59Gio float32**, **62.79Gio float16**, hors clés/textes/index/temp. Déduplication des textes peut réduire, réduction nationale non mesurée. PARENT : scénario d’un vecteur par chacune des29922486 unités légales (footer parquet), soit **85.61/42.80Gio** supplémentaires ; ce n’est PAS une borne : un parent peut avoir plusieurs vues et des textes identiques peuvent partager un vecteur. Le pilote officiel compte2126387 parents/2131929 vues/1862302 textes uniques ; ratio non extrapolé nationalement. Les38875873 documents parent Tantivy ne sont pas un décompte de vecteurs. SITE+PARENT à un vecteur/unité≈211.20Gio f32. Besoin réel PARENT inconnu. Les non-publics ne permettent aucune reconstruction de nom masqué.
À0,020–0,025s/site mesurées auparavant : **243.87–304.84h SITE**. Proxy PARENT0,009–0,012s/site : **109.74–146.32h**, soit **353.61–451.16h banques**. Ce sont des heures équivalentes de préparation incluantCPU/packing/premier score, pas des heures GPUbusy mesurées nationalement. Temps partageable une fois par binding ; ne pas le multiplier par les fiches.
Banques amorties : **9.66–12.33s/fiche** sur cet export. Génération native seule après préparation, proxy mesuré17,067–19,341s/fiche : **624.60–707.82h** de plus. Les banques+native seules totalisent≈978.21–1 158.98h ; autres preuves/scoring/préparation requêtes/reclassement/attentes en plus. Pas de coût end-to-end V142 national validé.
Ordre de grandeur isolé V19 : les8,67milliards de couples ×23,732s/319488 donnent≈178.93h ; ce proxy de boucle seule ne couvre pas les autres preuves. Si on ne prépare que les communes CRM :34839788 sites, SITE≈193.55–241.94h, encore hors PARENT. Point de comparaison historique :3000 dossiers160–250h avec partage, pas une règle de trois linéaire.
Float16 est une estimation de stockage uniquement : conversion/changement de précision non autorisé par le gel et aucune fidélité des sorties démontrée. V142 n’a pas de politique de serving sans vérité : routage OOF dépendant du SIREN humain, full-fit distinct à autoriser (45–75min estimées), sources/caches en ligne et validation indépendante manquants. Pic natif historique9,14Gio ; réserver12Gio à cette phase (non garanti nationalement), séquentiellement avec l’autre agent, pas une banque entière en RAM.

## 4. Disque : inventaire sans suppression
Libre au relevé : **19.23Gio**. `du -sk` mesure les octets alloués, sans suivre les liens. Parents ci-dessous et leurs enfants ne s’additionnent pas :
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100` : 483.78Gio ; mtime répertoire 2026-10-10.
- `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` : 335.80Gio ; mtime répertoire 2026-10-20.
- `/Volumes/CATNAT_DATA/SIRETO_V9` : 9.15Gio ; mtime répertoire 2026-08-24.
- `/Volumes/CATNAT_DATA/SIRETO_PPLX` : 2.26Gio ; mtime répertoire 2026-10-08.
- `/Volumes/CATNAT_DATA/SIRETO_RECOVERABLE_GIT_GARBAGE_20260824` : 0.34Gio ; mtime répertoire 2026-08-24.
- `/Volumes/CATNAT_DATA/SIRETO_STORAGE_QUARANTINE` : 0.15Gio ; mtime répertoire 2026-10-03.
- `/Volumes/CATNAT_DATA/SIRETO_M3_PYTHON_TCB` : 0.00Gio ; mtime répertoire 2026-07-31.
- `/Volumes/CATNAT_DATA/SIRETO_M3_TCB_TEST` : 0.00Gio ; mtime répertoire 2026-07-31.

Dix plus gros sous-dossiers non chevauchants des deux racines principales :
| Dossier | Gio | Date mtime répertoire UTC | Doc récent100commits | Chemin dans sceauV142 |
|---|---:|---|---|---|
| `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits` | 160.42 | 2026-10-07 | oui | oui |
| `/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes` | 108.66 | 2026-10-02 | oui | oui |
| `/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources` | 100.43 | 2026-08-21 | oui | oui |
| `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments` | 53.17 | 2026-08-24 | oui | non trouvé |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v174` | 48.42 | 2026-10-17 | oui | non trouvé |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v42` | 31.02 | 2026-10-12 | non trouvé | non trouvé |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v73` | 29.23 | 2026-10-13 | non trouvé | non trouvé |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v89` | 26.81 | 2026-10-13 | non trouvé | non trouvé |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v71` | 25.37 | 2026-10-12 | oui | oui |
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v28` | 23.55 | 2026-10-12 | non trouvé | non trouvé |
Dates=mtime du répertoire, PAS dernière écriture récursive. Références docs au nom du dossier et ensemble des100 derniers commits : heuristique, pas preuve de non-utilisation. Les détails et exemples de docs sont dans le JSON ; un ancêtre peut rester nécessaire même sans référence récente. Ne pas supprimer index/sources/caches scellés ; demander une analyse des dépendances par fichier avant tout ménage. Aucun fichier pending ouvert ; aucun fichier étranger ni cache modifié, aucune action sur la VM.

## 5. Recommandation et conditions
**Choisir G0 + S1/S2-BASE pour une chaîne de propositions à relire (`A_VERIFIER`), pas pour écraser automatiquement les SIRET.** Moindre coût de recherche, encodage limité aux candidats réellement nouveaux, banque partagée et traitement par lots≤200 sous4Gio ; aucun nouveau produit déployé ici. Projection du chemin chaud :32.18h ; sans aucun partage des candidats, **55.91–116.53h** avec encodage des candidats et noms CRM ; avec80% de partage,37.11–49.72h. Ce sont des scénarios, hors nouveau géocodage, pas une promesse.
Avant exécution globale : confirmer que cet export est bien toute la base à traiter, valider la sémantique des doublons et la politique de conservation des SIRET, autoriser une recette de serving/full-fit et une validation indépendante ; mesurer la génération de tous les candidats sans labels pour connaître les vrais textes nouveaux, puis réserver leur disque+temporaires+10Gio de sécurité. Exécution séquentielle CPU/Metal avec l’autre agent ; un seul Gemma, jamais de contrôle dehydraulic/PID8288.
Réserve conseillée : pour le scénario G0 sans partage,≈38Gio de vecteurs f32 + marge/exports/10Gio de sécurité, donc **au moins50–60Gio libres** ; c’est une capacité, pas le besoin réellement mesuré. Pour une banque V142 SITE+PARENT f32,≈211Gio hors textes/index/temp, donc **≥250Gio libres** de capacité initiale à confirmer. Les19Gio libres sont insuffisants pour promettre le full : même f16 candidats sans partage≈18,85Gio avant marge. RAM G0 d’inférence mesurée ci-dessus<4Gio par lots ; GemmaBASE historique≈2,2–4,3Gio selon phase, donc worker strictement borné et pas chargement complet de banque. Entraînement/full-fit non budgété ici.
Qualité : G0 top100478/526 vsV142507/526 (pilote consommé). G0+S1S2BASE top1 UNION13382/17427=76,79%, exact12275/17427=70,44%. Risque calibration97% :4360/4459=97,78% UNION, couverture25,59%, mais4151/4459=93,09% exact. Calibration97,5% :98,02% UNION, borneWilson97,37%, couverture13,59%. Calibration98% :0SUR. **Aucun statut SUR98% certifié ; conserver A_VERIFIER.** Ces chiffres de développement ne prédisent ni la couverture ni les erreurs nationales.

## Preuve et reproduction
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py inventory
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/production_cost_study.py bench > /tmp/sireto-production-cost-bench.log 2>&1 &
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py disk
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/production_cost_study.py finalize
```
La première ligne de suivi a été intégrée par la publication concurrente `be26eab2` ; son attestation finale est mise à jour dans le commit isolé des quatre livrables de cette étude. Aucun commit tiers réécrit. Exécuter les phases séquentiellement pour éviter une mise à jour concurrente du JSON. Suivi du run livré : `/tmp/sireto-production-cost-bench-corrected.log`, `/tmp/sireto-production-cost-disk.log`. Petit correctif de projection NAF et normalisationNB avant le rejeu final ; première tentative conservée dans le journal, sans résultat retenu. Un diagnostic DISTINCT SIREN arrêté par DuckDB512Mio ; remplacé par le compte de capacité du footer, pas par un faux nombre mesuré. Aucun test unitaire, fit, nouvel encodage, web, suppression, actionVM/PID8288 ou ouverture de label fermé.
