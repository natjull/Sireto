# Serve V1 — familles du pilote sans SIRET

## Protocole figé avant croisement des statuts (prod-3)

Cette section est écrite avant toute nouvelle lecture des statuts individuels ou des hits pour cette étude. Les résultats globaux du précédent pilote sont déjà connus (50,10% A_VERIFIER contre32,77% sur la référence BASE OOF) : il ne s'agit pas d'une étude aveugle à ces agrégats. Aucune règle ne sera révisée après croisement. Aucun nouvel apprentissage, score, seuil de serving, Gemma, appel web ni lecture d'un jeu fermé. Seuls le pilote2000, les requêtes du banc G0 explicitement autorisé17427/folds2/3/4 et ses OOF existants sont utilisés. Modèles, pools, labels et fichiers étrangers restent en lecture seule.

Normalisation commune aux CRM et enseignes : SQL `trim(regexp_replace(upper(strip_accents(texte)), '[^A-Z0-9]+', ' ', 'g'))`, espaces condensés. Les tokens, phrases, listes et paramètres exacts sont figés dans le JSON compagnon avant exécution. Aucune liste tirée des erreurs ou ajustée aux statuts.

Les **huit familles sont des indicateurs non exclusifs**, sauf «autre» qui signifie aucun des cinq types de nom. Une adresse incomplète ou une commune dense peut donc appartenir aussi à n'importe quel type de nom ; ne pas additionner les huit lignes. Un second tableau forme une partition exclusive pour l'attribution descriptive : technique/vide > générique/administratif > enseigne fréquente > sigle > personne probable > autre. Cette priorité est fixée ici, non choisie selon les résultats.

1. **Générique/administratif** : présence d'au moins un token du lexique public figé (institutions, équipements et gestion immobilière : notamment MAIRIE, ECOLE, SYNDIC, RESIDENCE, CENTRE, SERVICE). Indicateur lexical, pas une classification juridique ni une impossibilité de SIRET.
2. **Personne physique probable** : civilité initiale (M, MR, MME, MLLE, MONSIEUR, MADAME, MADEMOISELLE, DR, DOCTEUR, ME, MAITRE), ou2–4tokens alphabétiques dont au moins un prénom courant du lexique figé, sans token d'organisation/générique/technique ni enseigne fréquente reconnue. Un patronyme seul reste «autre» ; une civilité est un indice, pas une preuve d'entreprise individuelle ou d'occupant.
3. **Sigle/abréviation** : nom non technique contenant1–5lettres au total, ou forme brute entièrement en capitales contenant au moins2lettres et aucune voyelle AEIOUY. Les chiffres ne sont pas des lettres. Aucun nom n'est qualifié de sigle au seul motif d'être en majuscules.
4. **Enseigne fréquente (proxy de marque nationale demandé)** : le nom contient, avec frontières de tokens, une enseigne canonique entière présente sur au moins50SIRET distincts du stock officiel public (actifs ET fermés), parmi ses trois colonnes enseigne. Tous les sous-groupes de tokens des noms CRM sont recherchés : filtrage préalable par ces candidats seulement, sans perte d'une correspondance contenue. Pas de liste de marques CRM, aucun label. Le seuil50 ne prouve ni réseau national actuel, ni unicité du propriétaire : PHARMACIE/enseignes génériques peuvent aussi satisfaire ce critère. Une sous-vue débruitée exclut les expressions composées uniquement de mots génériques/juridiques/de liaison ; elle ne remplace pas cette famille principale.
5. **Adresse sans numéro** : `crm_number` ne commence pas par un entier strictement positif (zéros initiaux admis, suffixe/plage après l'entier admis). Ne pas utiliser le numéro du candidat ni la feature «number_missing», qui agrège CRM et candidat.
6. **Commune très dense** : nombre de sites publics actifs ET fermés par code commune dans le même stock officiel ; seuil95e centile (`higher`) parmi toutes ses communes non vides. Inclusion `>=seuil`, ex aequo inclus ; le pourcentage exact de communes est publié. Commune CRM manquante/non reconnue : non dense, absence publiée séparément. Ni percentile calculé sur le pilote ni densité dépendant du statut.
7. **Nom vide/technique** : vide, expression du lexique technique figé, expression technique suivie de codes alphanumériques contenant un chiffre, ou code en un token d'au moins4caractères dont chiffres>=lettres. Les expressions sont des libellés génériques (SITE, LOCAL TECHNIQUE…), jamais des corrections spécifiques à une fiche.
8. **Autre** : aucun des cinq indicateurs de nom1/2/3/4/7 ; les indicateurs d'adresse/densité ne retirent pas cette famille.

Mesures : parts et effectifs pilote/banc ; taux A_VERIFIER dans chaque famille ET contribution de la famille aux A_VERIFIER du pilote ; UNION top1 du banc OOF, aussi conditionné par ses trois statuts (même politique poolée figée). Exact et UNION publiés séparément. Top1 correct mais A_VERIFIER sur le banc = piste d'analyse, pas volume de remplissage garanti sur le pilote sans vérité. Pour les phrases d'interprétation, «bon»=UNION>=85%, «mauvais»=<70%, et piste «risque trop bas»=UNION>=80% parmi les A_VERIFIER avec au moins30cas ; seuils descriptifs seulement, aucun nouveau réglage.

Décomposition additive uniquement sur la partition exclusive : taux A_VERIFIER attendu si les parts des familles du pilote avaient les taux A_VERIFIER du banc ; effet composition=attendu−banc, résidu intra-famille/transport=pilote−attendu. Ce n'est pas une attribution causale : OOF du banc et full-fit du pilote sont des modèles distincts, et les familles sont grossières.

Même SIREN top1/top2 : ordre des scores existants (tri stable), comparaison des neuf premiers chiffres des deux SIRET ; dénominateur de chaque statut publié. Ce signal n'établit pas que l'entreprise est correcte ; contre-vérifier sa justesse UNION sur le banc OOF, y compris A_VERIFIER. Exemples : formes masquées de trois noms au maximum par famille, aucune lettre/chiffre original, aucun identifiant, SIRET, adresse ou commune en clair.

Budget : threads1, DuckDB512Mio sans spill, gardeRSS3,75Gio et3500s, aucune interaction avec VM/PID8288 ; seuls ces deux rapports et une ligne handover sont publiés. Programme reproductible et hashes du protocole/entrées/affectations avant statuts dans le JSON ; aucune affectation nominative persistée.

## Résultats

Run réel readonly : 9.96s, pic RSS 0.294Gio ; sans Gemma/web/fit, hashes des 9 sources inchangés. Protocole avant croisement SHA `4ce6c4550e19530fd29419936132dbad126e67effaf20183848b9f08a4ac96c0` ; règles SHA `bfbe7482bd0c35f8d00fbba435fad4c5aa07ee5cdbf4cf262394d9d6a95fa713`. Affectations figées à 2026-10-20T13:38:22.244724+00:00, avant lecture des statuts individuels, hashes dans le JSON.

| Famille (recouvrements) | Pilote n / part | Banc n / part | UNION top1 banc | Exact banc | AV dans famille pilote | Part des1002 AV pilote | UNION banc parmi AV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Générique / administratif | 78 / 3.90% | 3274 / 18.79% | 78.10% | 71.41% | 28 / 35.90% | 2.79% | 43.98% |
| Personne probable | 18 / 0.90% | 348 / 2.00% | 79.89% | 73.56% | 6 / 33.33% | 0.60% | 43.33% |
| Sigle / abréviation | 176 / 8.80% | 974 / 5.59% | 77.21% | 70.84% | 86 / 48.86% | 8.58% | 42.21% |
| Enseigne ≥50 sites | 283 / 14.15% | 4464 / 25.62% | 75.09% | 69.04% | 141 / 49.82% | 14.07% | 47.32% |
| Adresse sans numéro | 196 / 9.80% | 2775 / 15.92% | 64.90% | 64.83% | 133 / 67.86% | 13.27% | 40.44% |
| Commune dense | 1567 / 78.35% | 13613 / 78.11% | 76.69% | 69.65% | 754 / 48.12% | 75.25% | 45.00% |
| Vide / technique | 43 / 2.15% | 7 / 0.04% | 57.14% | 57.14% | 41 / 95.35% | 4.09% | 40.00% |
| Autre nom | 1453 / 72.65% | 9965 / 57.18% | 76.67% | 70.11% | 717 / 49.35% | 71.56% | 43.10% |

### Partition exclusive et coût descriptif

| Type primaire | Pilote n | Banc n | AV pilote | AV banc | UNION banc |
|---|---:|---:|---:|---:|---:|
| Vide / technique | 43 | 7 | 95.35% | 71.43% | 57.14% |
| Générique / administratif | 78 | 3274 | 35.90% | 28.68% | 78.10% |
| Enseigne ≥50 sites | 249 | 2936 | 52.61% | 39.99% | 75.10% |
| Sigle / abréviation | 162 | 908 | 50.00% | 28.85% | 77.97% |
| Personne probable | 15 | 337 | 26.67% | 25.22% | 79.53% |
| Autre nom | 1453 | 9965 | 49.35% | 32.57% | 76.67% |

Écart AV 17.33points ; avec uniquement le mélange de types de nom du pilote et les taux AV du banc, attendu 33.82% : composition +1.05points, résidu intra-famille/transport +16.28points. Non causal : modèles OOF/full-fit et contenus intra-famille différents. Ces contributions utilisent la partition exclusive, pas les8indicateurs chevauchants.

Répartition **comptable additive** du supplément d'AV (différences de nombres AV divisés par les tailles respectives des cohortes, pas un effet causal) : autre **+17,22points**, sigle+2,55, technique+2,02, personne−0,29, enseigne−0,19, administratif−3,99 ; somme+17,33points. Le détail exact est `pilot_minus_bank_averifier_contribution_pp` dans la partition JSON ; ne pas additionner la même colonne des indicateurs non exclusifs.

Enseignes : 596 expressions ≥50 sites correspondent à au moins un nom des cohortes ; 551 après débruitage lexical secondaire. Ce n'est pas le nombre total d'enseignes nationales. L'indicateur principal conserve les génériques conformément au critère brut demandé ; sous-vue non générique : pilote 229/2000, banc 3083/17427.
Densité officielle : 36914 communes, 37937120 sites publics avec code commune, seuil 2918sites, 1846communes (5.001%) avec ex aequo. Commune CRM non reconnue : pilote1, banc37 ; conservées au dénominateur.

### Que pourrait-on récupérer ?

- **Générique / administratif** : banc intermédiaire, UNION 78.10% ; parmi ses AV, 413/939 corrects (43.98%). Pas de justification pour un relâchement automatique selon règle préfixée. Ajouter le gestionnaire/exploitant et un libellé distinctif : un équipement ou une copropriété ne désigne pas forcément son entité juridique.
- **Personne probable** : banc intermédiaire, UNION 79.89% ; parmi ses AV, 39/90 corrects (43.33%). Pas de justification pour un relâchement automatique selon règle préfixée. Préciser raison sociale/activité et occupant plutôt que propriétaire ; un particulier sans établissement professionnel identifiable ne devient pas siretisable avec son seul nom.
- **Sigle / abréviation** : banc intermédiaire, UNION 77.21% ; parmi ses AV, 122/289 corrects (42.21%). Pas de justification pour un relâchement automatique selon règle préfixée. Développer le sigle et fournir nom commercial/raison sociale ; la courte chaîne seule ne justifie pas de relâcher le seuil.
- **Enseigne ≥50 sites** : banc intermédiaire, UNION 75.09% ; parmi ses AV, 796/1682 corrects (47.32%). Pas de justification pour un relâchement automatique selon règle préfixée. Identifier le franchisé/exploitant local et le site exact ; enseigne fréquente ne signifie pas propriétaire ou SIREN unique, surtout pour les libellés génériques.
- **Adresse sans numéro** : banc mauvais, UNION 64.90% ; parmi ses AV, 552/1365 corrects (40.44%). Pas de justification pour un relâchement automatique selon règle préfixée. Ajouter numéro/suffixe ou position du site certifiée ; un numéro absent peut laisser hésiter les sites sans invalider nécessairement l’entreprise.
- **Commune dense** : banc intermédiaire, UNION 76.69% ; parmi ses AV, 2018/4484 corrects (45.00%). Pas de justification pour un relâchement automatique selon règle préfixée. Compléter bâtiment/voie/numéro et exploitant ; densité et colocalisation réclament une preuve plus distinctive, pas un seuil plus permissif.
- **Vide / technique** : banc mauvais, UNION 57.14% ; parmi ses AV, 2/5 corrects (40.00%). Pas de justification pour un relâchement automatique selon règle préfixée. Remplacer code/équipement par occupant/exploitant ; une fiche limitée à un local technique sans entité juridique n’est pas siretisable avec cet export seul.
- **Autre nom** : banc intermédiaire, UNION 76.67% ; parmi ses AV, 1399/3246 corrects (43.10%). Pas de justification pour un relâchement automatique selon règle préfixée. Pas de signal unique : contrôler nom officiel/nom d’usage actuel et adresse, puis auditer erreurs de génération ou de classement sans inventer de vérité.

### Même SIREN top1/top2

| Statut pilote | Même SIREN / n | Taux | Banc UNION quand même SIREN | Banc UNION quand SIREN différents |
|---|---:|---:|---:|---:|
| SUR | 66/279 | 23.66% | 992/1006 (98.61%) | 2561/2619 (97.79%) |
| PROBABLE | 159/719 | 22.11% | 2736/2977 (91.90%) | 4546/5114 (88.89%) |
| A_VERIFIER | 84/1002 | 8.38% | 667/1165 (57.25%) | 1880/4546 (41.36%) |

Total pilote 309/2000=15.45%. Deux sites du même SIREN ne prouvent ni entreprise correcte ni équivalence physique ; ils ne suffisent donc pas à remplir automatiquement le SIRET. Les hits UNION existants du banc sont la contre-vérification, pas une nouvelle vérité pour le pilote.

### Exemples masqués de formes, pas des labels

- Générique / administratif : `***** ******… ***** ** **** ******…`, `***** ******`, `******… ****** ##`.
- Personne probable : `** ******… ** ******…`, `****** ****** ******…`, `**`.
- Sigle / abréviation : `*****`, `****##`, `*****#…`.
- Enseigne ≥50 sites : `****** **** *****`, `*****`, `* ****** *** ** ****`.
- Adresse sans numéro : `** ****`, `******…`, `*****`.
- Commune dense : `******…`, `*****`, `****** **** *****`.
- Vide / technique : `[vide]`, `**##`, `**###`.
- Autre nom : `******…`, `** ****`, `******`.

### Trois conclusions opérationnelles

1. Les catégories grossières de nom ne rendent compte descriptivement que de **+1,05point sur les +17,33points** d'A_VERIFIER : attendu33,82% plutôt que50,10%. Génériques/administratifs sous-représentés (3,90% contre18,79%) et densité quasi identique (78,35% contre78,11%) ne valident pas l'explication «plus de mairies ou de grandes communes». Le résidu16,28points mêle différences intra-famille et transport OOF→full-fit ; cette étude ne les sépare pas causalement.
2. **717/1002 AV sont dans «autre nom»** : cette grande famille doit être la cible principale d'une revue/enrichissement du nom d'occupant/exploitant, pas d'un assouplissement automatique. Les sans-numéro ont64,90% de justesse banc et67,86% AV pilote : ajouter numéro/suffixe est une piste concrète ; les noms techniques ont41/43 AV mais seulement4,09% des AV, donc ne peuvent expliquer à eux seuls la moitié des refus. Aucun de ces huit segments n'atteint80% UNION parmi ses AV du banc (minimum30cas) : aucun segment récupérable automatiquement n'est démontré.
3. **84/1002 AV seulement ont top1/top2 du même SIREN** (8,38%) ; même dans ce sous-groupe, la justesse UNION du banc AV est57,25% (667/1165). Ce signal ne caractérise donc qu'une minorité des AV ; son absence ne prouve pas que l'entreprise top1 est fausse, et la justesse UNION mesure le SIRET/site, pas la seule justesse du SIREN. Sa présence seule n'autorise pas le remplissage. Demander exploitant/site exact et valider humainement, sans relâcher la confiance.

### Limites et reproduction

La famille technique n'a que **7 dossiers sur le banc** :57,14% est une estimation très fragile, pas une preuve transférable aux43du pilote. La règle personne est conservatrice et peut retenir une civilité seule ; elle manque notamment les patronymes sans prénom reconnu. Personne probable et enseigne fréquente sont des heuristiques textuelles, pas des identités juridiques. Le lexique de prénoms manque les prénoms non listés et les patronymes seuls. Une enseigne≥50 peut être générique ou appartenir à plusieurs franchisés ; sa fréquence ne prouve pas une marque nationale. La densité inclut les fermés. Un nom technique n’est pas une preuve d’absence d’entreprise à cette adresse. Aucun chiffre de justesse du pilote n’est mesuré sans revue humaine. Les Wilson du JSON sont descriptifs sur le banc de développement, pas des gates ni une validation indépendante de nombreuses sous-familles.
Le JSON contient tous les croisements SUR/PROBABLE/A_VERIFIER, hits exact/UNION, effectifs, Wilson, scores médians, surreprésentations, coûts descriptifs et programme autonome. Commande de reproduction (mêmes règles et sources, sans fichier nominatif généré) :
```bash
.venv/bin/python -c "import json; p=json.load(open('docs/serve_v1_pilot_families.json')); exec(p['reproduction_program'], {'PROGRAM':p['reproduction_program']})"
```
Smoke réel : programme embarqué réexécuté sur les19427 fiches ; les10sections scientifiques (dont huit familles/deux cohortes, partition, hits et mêmes SIREN) sont strictement identiques, règles/affectations/sources scellées inchangées ; sommes de statuts, hits et décomposition contrôlées. Trois passes réelles mesurées :29,72s de calcul au total, pic maximal0,314Gio. Pas de tests unitaires. Vérification complète dans `verification` du JSON ; les identifiants du banc peuvent être de courts nombres, donc le contrôle de confidentialité cherche des valeurs sérialisées exactes et non leurs sous-chaînes dans des comptes ou SHA.

Log technique agrégé du run initial : `/tmp/sireto-prod3-families.log`. Seuls les deux rapports et la ligne handover sont des livrables ; aucun fichier source ou étranger modifié. Commit isolé auto-référencé par `git log -1 --format=%H -- docs/serve_v1_pilot_families.json`.
