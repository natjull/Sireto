# Bruit des labels G0 — avenant v1 préenregistré avant exécution

## Périmètre et ressources

17427 dossiers G0, folds population2/3/4, mêmes100 candidats, vérités et cinq CV entreprise. Objectif métier de Nathan : occupant actif à l'adresse fibre CRM ; cette vue secondaire ne change aucune cible officielle. Aucun jeu fermé, web, fit, VM ni fichier tiers/pending touché. Deux heures depuis lancement, garde RSS3,75Gio (<4), CPU/DuckDB1thread/512Mio, pas de spill ; worker séquentiel détaché et journal. Écritures autorisées : ces deux docs, une ligne handover, namespace neuf `$H/label_noise_review_20261020/` pour code/reçus/log/fiches et clé privée. Si namespace préexistant : STOP, ne pas écraser.

## 1. Grille sans modèle figée

Sources locales officielles août2026, géométrie juillet2026/BAN G0 ; données publiques seulement. Pour chaque dossier, examiner les100 SIRET G0 et les établissements actifs publics du stock à l'adresse CRM : jointure de présélection INSEE/numéro, puis confirmation stricte numéro/suffixe/voie normalisée identique à la grille `step2_sur_label_audit`. Ni score ni rang appris ne servent à cette sélection.

N = maximum Jaro-Winkler entre nom CRM et noms publics site/UL courants, mêmes normalisations historiques ; N≥0,92, N label connu. Concurrent actif obligatoire. Site admissible = adresse stricte OU distance typée≤tolérance UNION (200m si deux points au numéro,500m sinon voie ; aucune coordonnée communale/inconnue). Distance typée concurrent>500m bloque même une adresse textuelle exacte. Un concurrent déjà correct UNION ne crée aucun conflit : exact ou même SIREN ET (adresse stricte OU distance admissible). Les masques UNION G0 scellés restent la référence pour les100 ; même définition appliquée aux ajouts diagnostic, jamais injectés dans le pool.

Dominance héritée : (adresse concurrente strictement meilleure ET N concurrent≥N label) OU (deux adresses strictes ET N concurrent−N label≥0,08). Adresse meilleure = candidat strict et contradiction label connue sur les champs complets numéro/suffixe/voie/INSEE, OU concurrent≤50m et label typé>500m. Extension BAN expressément demandée : un candidat seulement proche, sans adresse stricte, peut dominer uniquement si le label typé est à>500m et le candidat dans la tolérance UNION ; il faut toujours N concurrent≥N label. Absence de nom ou d'adresse/point n'est jamais une contradiction connue.

Niveaux exclusifs :
- **FORT** : concurrent dominant ET (adresse meilleure ET gain nom≥0,08, OU label fermé, OU label typé>500m).
- **FAIBLE** : concurrent dominant restant (gain sur un axe, autre axe non inférieur).
- **AUCUN** : aucun concurrent satisfaisant ; cela ne prouve pas un label correct.

Plusieurs concurrents : FORT avant FAIBLE, puis gain N décroissant, adresse stricte d'abord, distance typée croissante (inconnue après connue), SIRET lexical pour égalité uniquement. Aucun score du modèle, NAF, statut siège ou identifiant n'est une feature du criblage. NAF et état sont montrés au juge ; dates CRM/occupation/filiation non attestées. Comptes distincts G0 seul et G0+co-adresse, absences/non-publics et UNION exclus publiés.

## 2. Criblage modèle et accord

Limite préconstatée dans le code risque v1 : une probabilité OOF de correction UNION du top1, pas de probabilités par SIRET. Aucun nouveau fit pour en fabriquer. Publier séparément `risk_probability_top1_union` et poids relatifs du reclasseur `w_j=softmax(score_j)` à température1 fixe. `w_label` uniquement si label exact dans100 ; absent = non disponible, jamais zéro. Ces poids ne sont PAS des probabilités calibrées ni une implémentation validée de confident learning.

Signal modèle prédéclaré : top1 non UNION, label exact présent, rapport `w_top1/w_label≥4` (écart score≥ln4), risque OOF acceptant le top1 au seuil calibration97 déjà gelé. Sinon NON ; label absent = INDISPONIBLE. Publier aussi le rapport continu et la probabilité risque, sans sélection sur erreurs. Rejouer les masques97 et l'ordre des scores depuis les artefacts ; aucun réglage sur G0. Accord entre criblage sans modèle (FORT ou FAIBLE) et ce signal : matrice2×2, accord brut, Cohen kappa, Jaccard, recouvrement par niveau ; dénominateur disponible explicite et absents séparés. C'est un contraste avec un signal de rejet très sélectif, pas deux estimateurs indépendants du bruit réel.

Taux FORT/FAIBLE/AUCUN et Wilson bilatéral95 descriptif : global, folds2/3/4, SUR97/non-SUR, dix familles historiques scellées ; groupes complémentaires `pool_union_absent_base` et `hors_erreurs_base` pour couvrir tout le banc sans rebaptiser les familles. Publier n/N et ne jamais retirer une ligne.

## 3. Tirage humain en aveugle

`random.Random(20261020)`, listes par strate en indices G0 croissants, sans remise :100 FORT,50 FAIBLE,50 AUCUN. Si strate trop petite : prendre toute la strate et signaler le déficit, sans remplacement ni transfert. Ordre des fiches et options aléatoire indépendant, graine20261021 ; identifiants de revue opaques graine20261022. Deux options : label humain et concurrent choisi ; AUCUN : label et top1 OOF. Deux SIRET identiques restent identiques, jamais remplacés pour embellir le témoin.

CSV juge : CRM nom/adresse puis options1/2 (dénomination, enseigne, adresse, actif/fermé, distance typée/qualité, NAF), réponse vide et commentaire. Valeurs permises : `Option 1 / Option 2 / Les deux / Aucune / Je ne sais pas`. Anonymisation des rôles, identifiants SIREN/SIRET/query, modèle/strate/scores ; les textes métier officiels nécessaires au jugement sont conservés, pas pseudonymisés arbitrairement. Aucun ordre label/concurrent déductible des colonnes. Aucune formule CSV active (préfixer les champs commençant par =,+,-,@).

Clé de décodage et résultats individuels dans `private/` mode0700, clé mode0600 ; SHA256 publié, aucune correspondance option→label dans les docs ni la fiche. Ne transmettre au juge que `review_sheet.csv`, jamais le namespace ni les docs privés. Ces permissions protègent un autre compte ; un juge utilisant le même compte propriétaire peut techniquement lire la clé : séparation organisationnelle requise, non garantie par un chmod. Temps de remplissage à estimer comme hypothèse1–2min/dossier, pas un temps mesuré.

## 4. Estimation après jugement, préenregistrée

Aucune réponse fabriquée et aucun taux de bruit humain annoncé avant retour des fiches. Strates h, poids W_h=N_h/17427. Indicateur primaire Y=1 si concurrent différent du label choisi seul,0 si label/Les deux ; `Aucune` indique les deux options non occupantes mais pas « concurrent choisi » : taux secondaire séparé. `Je ne sais pas`/manquant = inconnu, jamais supprimé du dénominateur. Publier somme W_h×moyenne(Y) si toutes les réponses sont déterminées ; sinon bornes avec inconnus0/1, pas un point prétendument mesuré. Le taux « concurrent choisi » est un proxy du label non-occupant, non exhaustif ; signaler séparément les AUCUN sans alternative et les options identiques. Ne pas considérer un concurrent identique au label comme un désaccord de vérité.

IC : bootstrap stratifié10000 rééchantillonnages de dossiers dans chaque strate, graine20261020, poids population fixes, percentiles2,5/97,5. Strate entièrement recensée : composante fixée, pas de variance d'échantillonnage. Pas de correction artificielle de labels ; IC descriptif sans erreur de juge modélisée, sensibilité cluster entreprise en complément si jugements disponibles.

Vue précision SUR occupant : estimation Horvitz–Thompson `somme_h (N_h/n_h) somme_revue_h 1_SUR × Z /4459`, à côté des précisions officielles exacte et UNION inchangées. Z=1 si top1 correspond à une option reconnue occupante ; Z=0 si top1 est montré mais rejeté au profit de l'autre seule ou de Aucune ; Z inconnu si top1 non montré, réponse inconnue, ou cas temporel non résolu. Publier des bornes en affectant inconnus0/1 plutôt que supposer que tout concurrent correct vaut top1 correct. Même bootstrap stratifié ; pas de précision « corrigée » effective tant que la fiche est vide.


## Résultats mesurés — criblage, pas correction des labels

**17427/17427 dossiers traités**, 1802592 paires examinées, plis population2/3/4 seulement. Aucune réponse humaine renseignée : **taux réel de bruit et précision SUR occupant non estimés**.

| Population | N | FORT n/N, %, Wilson95 | FAIBLE n/N, %, Wilson95 | AUCUN n/N, %, Wilson95 |
|---|---:|---|---|---|
| global | 17427 | 95/17427 = 0.545% [0.446;0.666] | 50/17427 = 0.287% [0.218;0.378] | 17282/17427 = 99.168% [99.022;99.292] |
| SUR97 | 4459 | 31/4459 = 0.695% [0.490;0.985] | 10/4459 = 0.224% [0.122;0.412] | 4418/4459 = 99.081% [98.755;99.321] |
| non_SUR | 12968 | 64/12968 = 0.494% [0.387;0.630] | 40/12968 = 0.308% [0.227;0.420] | 12864/12968 = 99.198% [99.029;99.338] |
| fold_2 | 5763 | 34/5763 = 0.590% [0.423;0.823] | 20/5763 = 0.347% [0.225;0.535] | 5709/5763 = 99.063% [98.780;99.281] |
| fold_3 | 6028 | 31/6028 = 0.514% [0.363;0.729] | 16/6028 = 0.265% [0.163;0.431] | 5981/6028 = 99.220% [98.965;99.413] |
| fold_4 | 5636 | 30/5636 = 0.532% [0.373;0.759] | 14/5636 = 0.248% [0.148;0.417] | 5592/5636 = 99.219% [98.954;99.418] |

### Familles historiques : dénominateurs complets et inchangés

| Famille | N | FORT n/N, %, Wilson95 | FAIBLE n/N, %, Wilson95 | AUCUN n/N, %, Wilson95 |
|---|---:|---|---|---|
| colocation | 1046 | 25/1046 = 2.390% [1.624;3.504] | 47/1046 = 4.493% [3.396;5.924] | 974/1046 = 93.117% [91.419;94.498] |
| homonym_elsewhere | 23 | 8/23 = 34.783% [18.811;55.110] | 0/23 = 0.000% [0.000;14.312] | 15/23 = 65.217% [44.890;81.189] |
| hors_erreurs_base | 12889 | 5/12889 = 0.039% [0.017;0.091] | 1/12889 = 0.008% [0.001;0.044] | 12883/12889 = 99.953% [99.898;99.979] |
| label_far | 463 | 22/463 = 4.752% [3.159;7.089] | 0/463 = 0.000% [0.000;0.823] | 441/463 = 95.248% [92.911;96.841] |
| no_number | 443 | 0/443 = 0.000% [0.000;0.860] | 0/443 = 0.000% [0.000;0.860] | 443/443 = 100.000% [99.140;100.000] |
| other | 1074 | 17/1074 = 1.583% [0.991;2.520] | 2/1074 = 0.186% [0.051;0.676] | 1055/1074 = 98.231% [97.253;98.865] |
| pool_union_absent_base | 1102 | 18/1102 = 1.633% [1.036;2.567] | 0/1102 = 0.000% [0.000;0.347] | 1084/1102 = 98.367% [97.433;98.964] |
| same_siren_near | 37 | 0/37 = 0.000% [0.000;9.406] | 0/37 = 0.000% [0.000;9.406] | 37/37 = 100.000% [90.594;100.000] |
| same_siren_other | 42 | 0/42 = 0.000% [0.000;8.380] | 0/42 = 0.000% [0.000;8.380] | 42/42 = 100.000% [91.620;100.000] |
| same_siren_siege | 267 | 0/267 = 0.000% [0.000;1.418] | 0/267 = 0.000% [0.000;1.418] | 267/267 = 100.000% [98.582;100.000] |
| same_siren_state | 41 | 0/41 = 0.000% [0.000;8.567] | 0/41 = 0.000% [0.000;8.567] | 41/41 = 100.000% [91.433;100.000] |
| generic_name | 0 | 0/0 — non défini | 0/0 — non défini | 0/0 — non défini |

Les familles décrivent les anciennes erreurs base85, pas toutes les scènes de cette nature. Leurs deux compléments couvrent aussi absences UNION et dossiers hors erreurs historiques. Wilson descriptif par dossier, sans garantie produit.

### Couverture du criblage indépendant

G0 seul : {"AUCUN": 17283, "FORT": 94, "FAIBLE": 50}. Après co-adresse locale : {"AUCUN": 17282, "FORT": 95, "FAIBLE": 50}. 59892 paires strictes supplémentaires hors100 ; 1 dossiers changent de niveau ; 1 concurrents retenus hors100, uniquement pour diagnostic, jamais injectés dans les candidats.
Concurrents retenus uniquement par la voie BAN sans adresse stricte :17. Paires déjà UNION correctes exclues :6462. Paires stock manquantes :0 ; non publiques :5127 (pas des dossiers uniques). AUCUN ne certifie ni un label correct ni une recherche exhaustive d'occupants absents des sources.

### Accord avec le signal modèle

**Écart explicite au brief** : les probabilités OOF par candidat/du label n'existent pas dans le risque v1. Aucun entraînement ni prétendue probabilité ajoutée. Nous comparons des poids relatifs softmax des scores R3 et la vraie probabilité risque de correction du top1 ; ces poids ne sont pas calibrés. Le signal très sélectif exige en plus l'acceptation97 gelée et un top1 non UNION.
Dénominateur disponible :15956 ; label exact absent, comparaison indisponible :1471. Matrice :{"both_signal": 30, "facts_only": 88, "model_only": 37, "neither": 15801}. Accord brut :99.217% ; Cohen kappa :0.32069 ; Jaccard :0.19355. L'accord brut est dominé par les négatifs et ne prouve pas une validation mutuelle.
Par niveau :{"FORT": {"NON": 48, "INDISPONIBLE": 27, "OUI": 20}, "FAIBLE": {"NON": 40, "OUI": 10}, "AUCUN": {"NON": 15801, "INDISPONIBLE": 1444, "OUI": 37}}. Les probabilités/poids/rapports individuels restent dans le fichier privé, jamais dans la fiche juge.

### Fiche humaine aveugle et clé séparée

- Fiche : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/label_noise_review_20261020/review_sheet.csv` ; **195 lignes**, réponses/commentaires vides, tirage {"FORT": 95, "FAIBLE": 50, "AUCUN": 50} ; graine20261020, options et ordre indépendamment randomisés.
- **Écart au volume demandé200 : seulement95 FORT existent sur tout le banc**, donc recensement des95,50 FAIBLE et50 AUCUN =195 fiches. La règle préenregistrée interdit de dupliquer ou transférer les cinq manquants ; les strates FORT et FAIBLE sont entièrement recensées.
- SHA256 fiche : `62cd4229b37388bcc26311e82fe4f019006b0ace06b0cbcc866e125b26bacf76`.
- Clé : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/label_noise_review_20261020/private/review_key.json` ; SHA256 **`9dd1bdcfdae62652ef8f4553e4b56df230be8abd5617bc8c8f3f5e6c23d10188`**, dossier0700/fichier0600. Ni la clé ni les correspondances option/label ne sont publiées dans Git.
- 44 fiches présentent deux SIRET identiques : aucune alternative inventée ; limite explicite pour l'estimation des conflits dans AUCUN.
- Le juge reçoit uniquement le CSV et `JUDGE_INSTRUCTIONS.txt`, jamais private/ ni les artefacts du criblage. Les noms/adresses officiels sont conservés pour permettre la comparaison, seuls rôles et identifiants sont anonymisés. Un juge sous le compte propriétaire peut techniquement accéder à la clé : utiliser un compte séparé ou transmettre uniquement le CSV.
- Remplissage estimé **195–390min**, hypothèse1–2min/dossier, pas un temps observé. Réponses : Option 1 / Option 2 / Les deux / Aucune / Je ne sais pas.

### Deux exemples anonymisés, hors fiche juge

Pseudonymes sans identifiants ; valeurs calculées sur les noms originaux, pas les pseudonymes. Premier dossier par niveau, sans sélection sur le résultat du modèle.
- **FORT** — CRM «Site Alpha» ; concurrent «Enseigne Alpha» actif, label «Entité Bêta» étatA. Noms 1.000/0.859, adresses strictes True/False, distances 13.93m/156.33m ; adresse meilleure=True, label fermé=False.
- **FAIBLE** — CRM «Site Gamma» ; concurrent «Enseigne Gamma» actif, label «Entité Delta» étatA. Noms 1.000/0.888, adresses strictes True/True, distances 38.79m/38.79m ; adresse meilleure=False, label fermé=False.

### Estimation : attente obligatoire des jugements

Les poids de strate population/échantillon et le décodage sont scellés dans la clé. La méthode préenregistrée ci-dessus reste inchangée : moyenne pondérée des concurrents choisis, bootstrap stratifié10000 ; inconnus et Aucune séparés, pas de dénominateur réduit. Le taux obtenu restera une mesure du conflit occupant observé dans ces options, pas de tous les labels faux possibles.
**Aucune précision SUR « corrigée » calculée**. Références officielles inchangées :4360/4459 UNION=97,7798%,4151/4459 exacts=93,0926%. Après jugement, publier la vue occupant et ses inconnus à côté de ces deux valeurs, jamais à leur place.

### Preuves, ressources et reproduction

Avenant commité AVANT exécution :`2b222a38a63d47c446bfccdc554118d400ca4aad`, SHA protocole `186cf96156a9a56291e408e177ea761de622c6a131467c1ec6c7822e53246400`. Première tentative arrêtée avant smoke pour une option de connexion DuckDB ; correctif SET seulement, reçu conservé. Smoke20 dossiers puis complet :20 identiques. Vérification réelle indépendante :17427 dossiers recomposés, 20 dossiers aléatoires rejoués depuis les sources, 36 références SHA recontrôlées, tirage/options et195 lignes CSV vérifiés, permissions vérifiées.
Worker complet :542.92s ; du namespace au résultat :780.15s, pic RSS 0.871Gio, garde3,75Gio. DuckDB512Mio/1thread ;0 fit,0web,0jeu fermé, aucun label/pool/statut/VM modifié. La vérification du préfiltre adresse constate986 numéros publics actifs non entiers et291 acceptés par le parseur, mais **zéro adresse CRM stricte omise** sur les17427.

```bash
H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u "$H/label_noise_review_20261020/run_screen.py"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u "$H/label_noise_review_20261020/verify_screen.py"
```

Le namespace livré est immuable : ne pas relancer le worker dessus (création exclusive du fichier de faits). Les commandes décrivent le run initial ; pour reproduire, copier le code dans un namespace neuf autorisé. Journaux/reçus :`worker.log`,`progress.json`,`screen_result.json`,`verification.json`, faits dans`private/full_screen.jsonl`. Aucun script du dépôt ni fichier étranger/pending touché. Commit résultats auto-référencé par `git log -1 --format=%H -- docs/label_noise_screen_v1.json`.

**Conclusion : le criblage fournit des strates de suspicion et une revue aveugle ; seul le retour humain permettra d'estimer un taux de labels non occupants et une vue de précision occupant, sans correction de vérité.**
