# Audit de la structure CRM — lecture seule

**Mesuré localement ; aucun entraînement, label corrigé, pool modifié ou promotion.**
## Sources et périmètre
Export coût : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/crm_new_data_20260817/source/plateforme_new_data_to_cure.csv` : **131749 lignes, 102495 entrées brutes distinctes, 88465 sans SIRET** (trois comptes recomptés ici).
G0 : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/step2_option_d_20261017/full/training_queries.json` et `label_side.json`, **17427 dossiers /11483 entreprises**, issus de `queries.parquet` de la population certifiée `73328b7611f37a06`.
Chaîne : export récent → `filter_new_crm_siret_by_location.py` (triage) → `build_crm_gt_v2_population.py` (avec `data/crm_ok_gt.csv` historique) → certification commerciale → entrées retrieval → cohorte autorisée → optionD/G0. Le manifeste de triage cite les deux sources ; les colonnes de validation historiques ne sont pas des informations CRM brutes supplémentaires.
Répartition G0 : {'REAL_CRM_20260817': 9897, 'REAL_CRM_HISTORICAL': 7530}. Les taux du fichier historique portent sur son inventaire, jamais sur ses valeurs de labels ; il n'est pas concaténé à l'export récent.
Tous les labels utilisés viennent du seul artefact G0 scellé, exact même ordre que les OOF. Les autres labels folds2/3/4 ne sont pas ouverts pour élargir les sœurs : test4323/control962 et les entreprises du pilote restent exclus.
Comptage population par projection `oof_fold` seule : {0: 7009, 1: 6622, 2: 7775, 3: 7970, 4: 7842}. test4323=4323/control962=962 : comptes du manifeste de cohorte, pas recomptage de leurs fichiers ; populations non additionnables.

## Schéma exhaustif
Remplissage = chaîne non vide après trim ; exemples réels transformés caractère par caractère (A=lettre,9=chiffre), jamais valeur originale. « Utilisée » concerne les entrées G0/content85/S1/S2 et leur préparation, pas les enrichissements officiels candidats.

### Export récent — 131749 lignes
| Colonne | Remplissage | Exemple masqué | Utilisée ? | Intérêt / risque |
|---|---:|---|---|---|
| `stc_service_id` | 131749/131749 (100.00%) | `99999` | non (jointure seulement) | Dédupliquer les services avant décision collective ; identifiant technique, pas un compte et jamais feature. |
| `stc_reference` | 131749/131749 (100.00%) | `A999999_99999` | non | Jointure/doublons seulement tant que la sémantique client/contrat est inconnue ; identifiant instable, pas feature. |
| `stc_categorie_produit` | 131749/131749 (100.00%) | `AAAAA AAAAAAAAA` | non | Stratifier erreurs et routage par offre ; risque de proxy métier/temps et de faible transférabilité. |
| `stc_state` | 131749/131749 (100.00%) | `AAAAAAAAAA` | non | Stratifier le cycle de vie du service ; risque de fuite si statut postérieur au rattachement. |
| `client_final` | 129962/131749 (98.64%) | `AAAAA AAAAAAAA` | oui (nom) | Clé de groupe exacte normalisée, sans alias ; homonymes, franchises et noms génériques mélangent des SIREN. |
| `stc_zipcode` | 131749/131749 (100.00%) | `99999` | oui | Adresse et filtrage géographique déjà utilisés. |
| `stc_city` | 131749/131749 (100.00%) | `AAAAAA` | amont seulement, pas entrée directe G0 | Contexte commune conservé en population ; G0 reçoit INSEE, CP, numéro et voie, pas la ville. |
| `order_siret` | 43284/131749 (32.85%) | `[label/validation masqué ; présence seule]` | non (vérité en amont pour les lignes admises) | Ancre vérifiée possible en production, pas feature ; fuite directe si vérité de la cible ou sœur fermée. |
| `stc_street_number` | 131453/131749 (99.78%) | `99` | oui | Numéro et suffixe déjà utilisés. |
| `stc_street_name` | 131749/131749 (100.00%) | `AAAAAA AA AAAA AA AAAA` | oui | Voie déjà utilisée. |
| `stc_insee_commune` | 131586/131749 (99.88%) | `99999` | oui | Commune déjà utilisée. |

### Source historique dérivée du banc — 17054 lignes
| Colonne | Remplissage | Exemple masqué | Utilisée ? | Intérêt / risque |
|---|---:|---|---|---|
| `crm_name` | 17054/17054 (100.00%) | `AAAAAAAA` | oui | Nom historique. |
| `crm_cp` | 17054/17054 (100.00%) | `99999` | oui | CP historique. |
| `crm_insee` | 17054/17054 (100.00%) | `99999` | oui | Commune historique. |
| `crm_id` | 16620/17054 (97.46%) | `AA999999` | non (jointure seulement) | Identifiant historique, pas compte client démontré. |
| `crm_commune` | 17053/17054 (99.99%) | `AAAAAAAA AAAAAAA` | amont seulement | Ville historique, pas entrée directe G0. |
| `gt_siret` | 17054/17054 (100.00%) | `[label/validation masqué ; présence seule]` | non (label) | Vérité historique ; présence agrégée uniquement dans cet inventaire. |
| `crm_adresse` | 17030/17054 (99.86%) | `99 AAA AAAAAA` | oui | Adresse historique parsée en numéro/suffixe/voie. |
| `SITE_CLI_COMMUNE` | 17053/17054 (99.99%) | `AAAAAAAA AAAAAAA` | non directement | Champ commune auxiliaire ; redondance probable, sémantique à confirmer, jamais vérité officielle. |
| `sirene_insee` | 17054/17054 (100.00%) | `[label/validation masqué ; présence seule]` | non côté CRM | Enrichissement de validation label-side, interdit comme nouveau signal de la cible. |
| `sirene_cp` | 17054/17054 (100.00%) | `[label/validation masqué ; présence seule]` | non côté CRM | Enrichissement de validation label-side, risque de fuite. |
| `sirene_etat` | 17054/17054 (100.00%) | `[label/validation masqué ; présence seule]` | non côté CRM | État du label, pas du service ; état candidat officiel déjà utilisé, risque de fuite directe ici. |
| `loc_match_type` | 17054/17054 (100.00%) | `[label/validation masqué ; présence seule]` | non (qualification) | Qualification issue de la comparaison au label ; audit uniquement, fuite si feature. |

**Absents des sources inspectées** : compte/client/contrat explicite, raison sociale de facturation distincte, contact, domaine e-mail, téléphone, dates de création/mise en service, commentaire libre, ancien SIREN/SIRET distinct. `client_final` est le seul nom client ; `order_siret` est un SIRET existant, pas un ancien SIRET daté. Disponibilité dans le CRM vivant : **inconnue**.

## Groupes et couverture des lignes sans SIRET
Clés exactes : identifiant service, référence entière, ou nom normalisé NFKD/ASCII/majuscules/ponctuation en espaces. Aucun fuzzy matching/alias ou préfixe de référence supposé compte client. Les noms vides ne forment pas un groupe.
| Clé | Groupes | Groupes ≥2 / lignes concernées | p50 / p95 / max | Sans SIRET avec présence sœur (inventaire seul) | Avec donneur G0 autorisé |
|---|---:|---:|---|---:|---:|
| service | 105351 | 24329 / 50727 | 1 / 2 / 6 | 2594/88465 (2.93%) | 785/88465 (0.89%) |
| reference | 105351 | 24329 / 50727 | 1 / 2 / 6 | 2594/88465 (2.93%) | 785/88465 (0.89%) |
| name_norm | 70760 | 24521 / 83715 | 1 / 3 / 2609 | 4303/88465 (4.86%) | 2368/88465 (2.68%) |
Relation service↔référence : {'services_with_multiple_references': 0, 'references_with_multiple_services': 0}. Une bijection observée ne démontre pas une clé compte ; la référence n'apporte alors pas un groupe client plus large. Catégories produit : 8 ; états service : 2.
La colonne « inventaire » compte uniquement un booléen présence `order_siret`, sans lire/utiliser sa valeur ni sa justesse : ce n’est pas un matching par labels, et ce chiffre ne permet pas d’ouvrir des sœurs des jeux fermés. La dernière colonne est plus conservatrice : existence de donneurs du seul G0, identifiés par nom ou service d’origine récent. Aucun label raw n’est propagé. Ne pas extrapoler sa justesse à 88465 cibles non labellisées.

## Cohérence des seuls labels autorisés folds2/3/4
Statistique pondérée par paires non ordonnées de dossiers distincts, pas moyenne des clients. Même adresse = commune+CP+numéro+suffixe+voie normalisés exacts ; commune, CP, numéro, voie obligatoires, aucun rapprochement géographique. Les services/références sont joignables seulement pour la partie récente du banc.
| Clé | Dossiers / groupes | Paires même SIREN | Même SIRET à même adresse | Groupes multi-SIREN | Groupes traversant folds population / CV |
|---|---:|---:|---:|---:|---:|
| name_norm | 17427 / 14389 | 22831/60896 (37.49%) | 29/31 (93.55%) | 173 | 122 / 150 |
| service | 9897 / 9897 | 0/0 (indéfini) | 0/0 (indéfini) | 0 | 0 / 0 |
| reference | 9897 / 9897 | 0/0 (indéfini) | 0/0 (indéfini) | 0 | 0 / 0 |

Groupes SIREN humains : **0 traversée de CV et de fold population**. Un groupe de nom traversant les CV relie donc plusieurs entreprises humaines. Le JSON donne aussi le nombre de dossiers concernés. Les labels du reste des folds2/3/4 hors cohorte G0 ne sont pas inspectés (restriction volontaire de garde).

## Potentiel sur les OOF S1+S2-BASE existants
Référence actuelle : exact **12275/17427 (70.44%)**, UNION **13382/17427 (76.79%)** ; les 17427 dossiers et les absences restent au dénominateur.
Un vote par dossier sœur, cible exclue ; majorité unique (pluralité maximale), égalité = abstention, jamais départage par identifiant. Filtre sur les 100 candidats inchangés, meilleur score OOF dans le SIREN majoritaire ; si aucun candidat du SIREN, retour top1 original (pas d’injection). Ex æquo score : ordre G0. Labels/UNION scellés inchangés.
- `loo_all` : toutes les autres lignes autorisées du groupe ; **oracle transductif label-side, pas OOF valide pour une entreprise inconnue**.
- `other_record` : exclut aussi toutes les lignes du même identifiant de service/enregistrement et origine ; contrôle des duplications, reste transductif.
- `cross_cv_only` : sœurs hors CV externe cible ; respecte la séparation entreprise pour les donneurs. Ni calibration ni recette de production nouvelle. La vérité du SIREN cible étant entièrement exclue du train entreprise, ce mode ne peut révéler correctement son SIREN : contrôle négatif, pas promesse de gain.
| Groupe / mode | SIREN connu | Justesse SIREN connu | Top1 UNION | Δ succès UNION | Exact | Sans candidat / égalités majorité |
|---|---:|---:|---:|---:|---:|---:|
| name_norm/loo_all | 3653/17427 (20.96%) | 2839/3653 (77.72%) | 13605/17427 (78.07%) | +255/−32=+223 | 12468/17427 (71.54%) | 813 / 173 |
| name_norm/other_record | 3653/17427 (20.96%) | 2839/3653 (77.72%) | 13605/17427 (78.07%) | +255/−32=+223 | 12468/17427 (71.54%) | 813 / 173 |
| name_norm/cross_cv_only | 1144/17427 (6.56%) | 0/1144 (0.00%) | 13343/17427 (76.57%) | +0/−39=-39 | 12246/17427 (70.27%) | 1015 / 167 |
| service/loo_all | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |
| service/other_record | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |
| service/cross_cv_only | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |
| reference/loo_all | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |
| reference/other_record | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |
| reference/cross_cv_only | 0/17427 (0.00%) | 0/0 (indéfini) | 13382/17427 (76.79%) | +0/−0=+0 | 12275/17427 (70.44%) | 0 / 0 |

Le gain transductif éventuel est une simulation diagnostique conditionnelle à des labels de sœurs fiables, pas une borne optimale ni un gain certifié du reclasseur. Même sans auto-vote, les sœurs du même SIREN sont dans le même CV externe : leur vérité n’est pas disponible au ranker OOF. En outre une prédiction de sœur produite par un autre modèle OOF pourrait avoir été entraînée sur l’entreprise cible ; elle ne remplace pas naïvement ces vérités. Un futur collectif nécessiterait des composantes client entièrement disjointes et des ancrages antérieurs temporellement vérifiés.

## Mesuré / estimé / inconnu et recommandation
**Mesuré** : remplissages, groupes, présence brute agrégée, cohérence et simulations OOF ci-dessus ; sources SHA avant/après et liaison clés/vues/top1 revérifiées. **Estimé** : aucun gain national ni précision de déploiement n’est extrapolé. **Inconnu** : sémantique compte/contrat des références, dates des ancrages, fiabilité des SIRET raw et inventaire du CRM vivant.
**Recommandation** : demander un export métier comportant un vrai compte client stable et les dates de disponibilité des ancrages ; ne pas adopter le filtre SIREN oracle ni changer G0. Tester ensuite un collectif query-only ou des ancrages historiques audités avec composantes client disjointes. À court terme, dédupliquer les services après validation métier et stratifier par catégorie/état ; garder `A_VERIFIER`. Le gain des labels de sœurs n’autorise aucune feature identifiant, ni emploi de labels fermés.

## Preuve et reproduction
```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/crm_structure_audit.py --smoke
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/crm_structure_audit.py > /tmp/sireto-crm-structure-full.log 2>&1 &
```
Run complet : 9.39s ; pic RSS 0.415Gio, garde 3,85Gio/3500s, DuckDB256Mio sans spill. Aucune lecture de valeur label fermé, entraînement, web, nouveau paquet, action VM/PID8288, modification source/tiers/pending, suppression ou push. Les seuls fichiers écrits par le script sont les deux rapports ; le smoke n’écrit rien.

Contrôle après le run : **18 références SHA revérifiées**, les résultats des 100 dossiers smoke sont identiques au rejeu incorporé au complet. Une commande Python indépendante a énuméré explicitement les **60896 paires** (22831 mêmes SIREN, 31 mêmes adresses, 29 mêmes SIRET à même adresse), puis toutes les sœurs et les **17427 décisions dans chacun des 3 modes nom** : comptes et SHA des décisions identiques. Aucun test unitaire ni nouvelle suite de tests. La première tentative smoke a rencontré une parenthèse manquante avant tout calcul ; correction suivie des deux runs réels réussis, sans résultat partiel retenu.
