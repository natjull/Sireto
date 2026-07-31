# Audit métier des 30 dossiers REVIEW V4.12

Date : 31 juillet 2026  
Lot figé : `c7a9feecaf2d3c2a`  
Ordre : `selection_ordinal` 1 à 30, sans sélection a posteriori.  
Politique : identité courante dans le snapshot SIRENE disponible ; `AMBIGUOUS` dès que le CRM ne contient pas l'information permettant de choisir entre plusieurs établissements plausibles.

## Tableau métier

| # | CRM | Décision / SIRET | Fiabilité | Preuves consultées | Diagnostic du pipeline | Correction potentielle |
|---:|---|---|---|---|---|---|
| 1 | ADAGES, Grabels | `33977442400594` | Haute | CRM ; SIRENE local ; site ADAGES ; Annuaire des Entreprises | **Top 1 faux** : service « Coordination Inclusion » au lieu du siège | Nom générique + statut siège ; exploiter l'enseigne du candidat |
| 2 | CC Plaines et Monts de France | `20003309000016` | Haute | CRM ; SIRENE local ; BANATIC ; Service-Public | **Top 1 faux** : service BA REOMI co-localisé au lieu du siège | Nom générique + statut siège ; pénaliser une enseigne de service absente du CRM |
| 3 | DDFIP Hérault - Sète | **AMBIGUOUS** | Haute | CRM ; SIRENE local ; Service-Public | Choix arbitraire du SIE parmi SIP, SGC et SIE à la même adresse | Abstention obligatoire sans nom de service |
| 4 | SIDEM Électricité, Amiens | `40006189100042` | Haute | CRM ; SIRENE local ; document d'entreprise ; Qualifelec | **Top 1 faux** : établissement au 16 B au lieu du siège au 16 | Comparer et conserver les suffixes de numéro de voie |
| 5 | GH Nord-Essonne - Longjumeau | `26910214100018` | Haute | CRM ; SIRENE local ; site GHNE ; Annuaire des Entreprises | **Top 1 faux** : entité administrative GHT au lieu du site hospitalier | Faire correspondre « site/hôpital » avec enseigne et activité |
| 6 | Centre de gestion FPT, Labège | `28310002200021` | Haute | CRM ; SIRENE local ; convention publique donnant adresse et SIRET | **Top 1 faux** : coordination régionale au lieu du centre départemental/siège | Nom générique + statut siège ; pénaliser l'enseigne non demandée |
| 7 | ADAGES REGAIN, Montpellier | **AMBIGUOUS** | Haute | CRM ; SIRENE local ; site et rapport ADAGES | Le top 1 est l'EMSP, mais CHRS, ACT, LAM et autres services « Regain » partagent le site | Abstention : « ADAGES REGAIN » désigne le complexe, pas un service unique |
| 8 | Lycée Sainte-Thècle | `77918786300013` | Haute | CRM ; SIRENE local ; mentions légales officielles Sainte-Thècle | **Top 1 faux** : école primaire au lieu du lycée/siège | Correspondance explicite du niveau scolaire et de l'activité |
| 9 | SINEQUAE, Calais | `48932980500134` | Haute | CRM ; SIRENE local ; site officiel SINEQUAE ; document légal | **Top 1 faux** : second office « côté gauche » au lieu du siège générique | Nom générique + siège ; réparer l'adresse CRM `16` ↔ `6 C` sans inventer une sous-unité |
| 10 | Commune de Thue et Mue | `20006502700019` | Haute | CRM ; SIRENE local ; Annuaire des Entreprises ; Service-Public | Top 1 correct, REVIEW inutile | Accepter l'identité et l'adresse exactes malgré une divergence de CP interne au snapshot |
| 11 | DIXICOM, Vendargues | `39773214000041` | Haute | CRM ; SIRENE local ; comptes publiés | Top 1 correct, REVIEW inutile malgré plusieurs sociétés domiciliées à l'adresse | Renforcer l'identité légale exacte face aux simples co-occupants |
| 12 | CA Expertises, Écully | `52097289400015` | Haute | CRM ; SIRENE local ; fiche légale publique | Top 1 correct, REVIEW inutile | Accepter nom légal + adresse exacts |
| 13 | ECF Roissy Formation | `45387794600054` | Haute | CRM ; SIRENE local ; catalogue ECF | Top 1 correct ; voisin ECF Paris Sud également présent | Faire primer le nom complet « Roissy Formation » sur la marque partagée ECF |
| 14 | Trivium Packaging, Neuilly | **AMBIGUOUS** | Haute | CRM ; SIRENE local ; site Trivium ; documents légaux du groupe | Le top 1 « West France » n'est pas justifié : plusieurs sociétés Trivium partagent exactement l'adresse | Abstention sans dénomination légale ou rôle (France, West, Group, Metal, holding) |
| 15 | NEXT MEDIA, Clermont-Ferrand | `35063664300040` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Accepter nom légal + adresse exacts ; ne pas confondre avec Next Media Training |
| 16 | Objectif Bâtiment, Ozoir | `98779962400018` | Haute | CRM ; SIRENE local ; site officiel ; Annuaire des Entreprises | **Top 1 faux** : ACROBAT, filiale du groupe, devant la société portant exactement le nom CRM | Donner priorité au nom exact ; distinguer groupe et filiales co-localisées |
| 17 | AT Patrimoine Lyon | `43860010800072` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Accepter nom légal + ville + adresse exacts |
| 18 | Nanoe, Ballainvilliers | `50772210600048` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Faire primer NANOE sur NANOE GROUP et les co-occupants |
| 19 | EPRO, Villiers-sur-Marne | `75017758600013` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Accepter nom légal + adresse exacts |
| 20 | Groupe R & D, Rillieux | `42274306200040` | Haute | CRM ; SIRENE local ; fiche légale publique | **Top 1 faux** : Opticom ; le candidat au nom exact est rang 3 | Donner priorité au nom légal exact face à plusieurs sociétés du groupe à la même adresse |
| 21 | Hôtel Le Tremplin, Morzine | `52272686800022` | Haute | CRM ; SIRENE local | Top 1 correct ; le second est l'activité de bar/restauration « Le Tremplin » | Exploiter le mot métier « Hôtel » et l'activité 55.10Z |
| 22 | France Alliance 44, Couëron | `44869695500088` | Haute | CRM ; SIRENE local | Top 1 correct ; autre établissement du même SIREN au n° 14 | Accepter nom exact + numéro de voie exact |
| 23 | INLOG, Limonest | `38888534500032` | Haute | CRM ; SIRENE local ; comptes IN LOG publiés | **Top 1 faux** : holding INLOG devant la société opérationnelle IN LOG portant le nom CRM | Distinguer holding et société opérationnelle ; privilégier l'identité exacte sans suffixe « Holdings » |
| 24 | NATURE, Combs-la-Ville | `32178451400051` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Accepter nom légal + adresse exacts malgré un nom court |
| 25 | HAYS, Nancy | `33249506800386` | Haute | CRM ; SIRENE local | Top 1 correct ; Hays Services est co-localisé | Distinguer la dénomination exacte de ses entités de service |
| 26 | SCM Gastro République | `34469182900040` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Forte priorité à la dénomination légale exacte |
| 27 | ISTRANS Exploitation | `48007438400011` | Haute | CRM ; SIRENE local ; site officiel ISTRANS ; avis INSEE | **Top 1 faux** : GIE Convoi Fos Cadarache à la même adresse ; vrai candidat rang 2 | Faire primer la dénomination exacte sur le co-occupant et tolérer rue/avenue |
| 28 | X-FAB France SAS | `82294763600020` | Haute | CRM ; SIRENE local | Top 1 correct, REVIEW inutile | Accepter nom légal + adresse exacts ; ignorer le CSE et les sociétés voisines |
| 29 | VAPOSTORE, Collégien | `75188876900104` | Moyenne | CRM ; SIRENE local incluant l'ancien SIRET `75188876900062` au n° 47 ; dossier Vapostore 2016 ; fiche courante au n° 6 | **Top 1 faux** : propriétaire/holding à l'ancienne adresse. Le CRM est ancien ; le SIRET courant de Vapostore est rang 2 | Gérer explicitement les déménagements et la continuité SIREN ; ne pas laisser l'adresse obsolète battre le nom exact |
| 30 | Getinge Life Science France, Tournefeuille | `33370766900128` | Haute | CRM ; SIRENE local | Top 1 correct ; plusieurs sociétés Getinge partagent le site | Faire primer la dénomination complète « Life Science France » |

## Bilan chiffré

| Résultat | Nombre |
|---|---:|
| Résolus avec un SIRET exploitable | **27 / 30** |
| dont fiabilité haute | **26** |
| dont fiabilité moyenne | **1** |
| AMBIGUOUS | **3 / 30** |
| UNRESOLVED | **0 / 30** |
| Labels SIRET exacts utilisables | **27** |
| Labels d'abstention utilisables | **3** |
| Top 1 actuel correct parmi les 27 résolus | **15 / 27 (55,6 %)** |
| Top 1 actuel faux parmi les 27 résolus | **12 / 27 (44,4 %)** |

Les 30 dossiers sont donc exploitables comme labels : 27 positifs exacts et 3 scènes d'abstention. Le cas VAPOSTORE doit conserver un indicateur de fiabilité moyenne et ne doit pas être utilisé sans la politique « identité courante ».

## Causes d'erreur observées

| Famille | Cas | Nombre | Effet |
|---|---|---:|---|
| Mauvais établissement du bon SIREN, souvent un service co-localisé | 1, 2, 4, 5, 6, 8, 9 | **7** | Le retrieval trouve la bonne organisation, mais le rang 1 choisit le mauvais NIC |
| Mauvaise société parmi des co-occupants ou sociétés d'un même groupe | 16, 20, 23, 27 | **4** | Le nom exact est rang 2 ou 3, derrière une société à la même adresse |
| Adresse CRM devenue obsolète | 29 | **1** | L'adresse bat à tort l'identité et masque le successeur courant |
| Ambiguïté réelle non résoluble avec le CRM | 3, 7, 14 | **3** | Un SIRET forcé serait une invention |
| Bon top 1 mais rejet par prudence | 10–13, 15, 17–19, 21–22, 24–26, 28, 30 | **15** | Perte directe de couverture AUTO |

## Corrections à tester, maintenant que les 30 sont audités

1. **Corriger le classement avant de toucher à l'accepteur.** Sur 12 erreurs résolues, le bon candidat est déjà dans le pool : il faut promouvoir l'identité légale ou l'enseigne exacte face à la simple adresse.
2. **Créer des signaux métier simples au niveau établissement** : candidat siège, candidat porteur d'une enseigne/service absent du CRM, correspondance du type de site (« lycée », « hôpital », « hôtel »), activité compatible, suffixe de voie exact.
3. **Apprendre explicitement les scènes de même SIREN** avec les 7 erreurs observées : nom générique → siège seulement comme indice, jamais comme règle absolue ; nom de service → enseigne correspondante ; information insuffisante → REVIEW.
4. **Traiter les groupes et co-occupants** : un nom légal exact doit battre un candidat qui ne partage que l'adresse. Les cas 16, 20, 23 et 27 sont des contre-exemples directement réutilisables.
5. **Séparer l'obsolescence d'adresse de l'ambiguïté.** VAPOSTORE montre qu'un ancien SIRET et un déménagement doivent être reliés par le SIREN et une politique temporelle explicite.
6. **Ensuite seulement recalibrer l'accepteur** sur des prédictions hors échantillon : les 15 top 1 corrects envoyés en REVIEW constituent les cas de couverture à récupérer, tandis que les 3 ambiguïtés doivent rester en REVIEW.

## Principales preuves publiques consultées

- [ADAGES, siège](https://www.adages.net/?directory_type=general) et [ADAGES Regain](https://www.adages.net/etablissements/general/regain/)
- [CC Plaines et Monts de France — BANATIC](https://www.banatic.interieur.gouv.fr/intercommunalite/200033090-cc-plaines-et-monts-de-france)
- [DDFIP Sète — Service-Public](https://lannuaire.service-public.gouv.fr/occitanie/herault/4e32b3e8-fc91-4d54-a3fe-253374b0b742)
- [GHNE Longjumeau](https://www.gh-nord-essonne.fr/site-hospitalier/longjumeau-2/)
- [Sainte-Thècle — mentions légales](https://www.sainte-thecle.com/mentions-legales/)
- [SINEQUAE Calais](https://sinequae.fr/calais/)
- [Commune de Thue et Mue — Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/etablissement/20006502700019)
- [Objectif Bâtiment](https://objectifbatiment.com/contact/)
- [ISTRANS Exploitation](https://www.istrans.com/)
- [Vapostore courant](https://www.pappers.fr/entreprise/vapostore-751888769) et [document historique 2016](https://www.vapostore.com/upload/presentation_vapostore_2016.pdf)

Le détail enrichi des cinq premiers cas est conservé dans `reports/v412_review_adjudication_first5.md`.
