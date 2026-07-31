# Audit métier des 5 premiers dossiers REVIEW V4.12

Date d'adjudication : 31 juillet 2026  
Périmètre : cinq premiers dossiers dans l'ordre figé par `selection_ordinal` du pilote `c7a9feecaf2d3c2a`.  
Règle : aucune prédiction du pipeline n'est utilisée comme preuve. Le libellé et l'adresse CRM sont confrontés au snapshot SIRENE local, puis à une source publique décrivant l'organisation ou le site.

## Résultat

| # | Dossier | Décision métier | SIRET proposé | Fiabilité | Preuves consultées | Erreur du pipeline actuel | Label exact utilisable |
|---:|---|---|---|---|---|---|---|
| 1 | `fresh:FR028730` — ADAGES, 125 rue Clément-François-Prunelle, Grabels | Résolu | `33977442400594` | Haute | CRM ; snapshot SIRENE local : trois établissements ADAGES actifs à la même adresse, dont `00594` est le siège et `00669` « ADAGES Coordination Inclusion » ; [site officiel ADAGES](https://www.adages.net/?directory_type=general) : l'adresse est celle du siège ; [Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/etablissement/33977442400594) : SIRET du siège | Le top 1 `33977442400669` est un service spécialisé co-localisé ; le classement ne donne pas assez de poids au caractère générique du nom CRM et au statut de siège. | Oui |
| 2 | `15204` — CC Plaines et Monts de France, 6 rue du Général-de-Gaulle, Dammartin-en-Goële | Résolu | `20003309000016` | Haute | CRM ; snapshot SIRENE local : `00016` siège, `00032` assainissement, `00081` BA REOMI, tous à la même adresse ; [BANATIC](https://www.banatic.interieur.gouv.fr/intercommunalite/200033090-cc-plaines-et-monts-de-france) : SIREN et adresse du siège ; [Service-Public](https://lannuaire.service-public.gouv.fr/ile-de-france/seine-et-marne/4fa013ed-00ec-464a-a25b-ac1e58e0afbc) : identité et adresse | Le top 1 `20003309000081` est le service BA REOMI ; le classement confond l'organisation générique avec un budget/service annexe à la même adresse. | Oui |
| 3 | `fresh:FR027738` — DDFIP Hérault - Sète, 274 avenue du Maréchal-Juin | AMBIGUOUS | — | Haute pour l'ambiguïté | CRM ; snapshot SIRENE local : au moins trois établissements DDFIP actifs à l'adresse (`00472` SIP Littoral, `00936` SGC Littoral, `01017` antenne SIE) ; [Service-Public](https://lannuaire.service-public.gouv.fr/occitanie/herault/4e32b3e8-fc91-4d54-a3fe-253374b0b742) confirme le SGC à cette adresse, sans élément CRM permettant de choisir ce service plutôt que les deux autres | Le top 1 `13000723001017` choisit arbitrairement l'antenne SIE ; les nom et adresse CRM ne contiennent pas la nature du service nécessaire à un SIRET exact. | Non |
| 4 | `3699` — SIDEM Électricité, 16 rue André-Durouchez, Amiens | Résolu | `40006189100042` | Haute | CRM ; snapshot SIRENE local : `00042` est le siège au n° 16 et `00091` un établissement secondaire au 16 B ; [document d'entreprise publié](https://www.maitredata.com/app/accords-entreprise/sidem-electricite/299776) : identité, adresse et SIRET `00042` ; [Qualifelec](https://www.qualifelec.fr/certifmoteur/20988/559857.pdf) : même adresse et même SIRET | Le top 1 `40006189100091` ignore la différence entre `16` et `16 B` et préfère le secondaire au siège explicitement documenté. | Oui |
| 5 | `8816` — Groupe Hospitalier Nord-Essonne - Longjumeau, 159 rue du Président-François-Mitterrand | Résolu | `26910214100018` | Haute | CRM ; snapshot SIRENE local : `00018` « Site de Longjumeau », activité hospitalière ; `00075` « Groupement hospitalier territoire », activité administrative ; [site officiel GHNE](https://www.gh-nord-essonne.fr/site-hospitalier/longjumeau-2/) : site hospitalier de Longjumeau à l'adresse CRM ; [Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/entreprise/269102141) : enseigne, adresse et SIRET `00018` | Le top 1 `26910214100075` choisit l'entité administrative co-localisée au lieu du site hospitalier explicitement nommé dans le CRM. | Oui |

## Bilan des cinq

- Résolus fiables : **4/5**.
- Ambigus : **1/5**.
- Non résolus : **0/5**.
- Labels SIRET exacts directement utilisables : **4**.
- Labels d'abstention utilisables : **1 AMBIGUOUS**.
- Top 1 actuel correct parmi les quatre cas résolus : **0/4**.

La cause dominante n'est pas un échec à trouver l'organisation : c'est la sélection du mauvais établissement lorsque plusieurs SIRET du même SIREN partagent une adresse. Les corrections potentielles à tester après les 30 cas sont : exploiter les mots désignant le service ou le site dans le CRM, distinguer précisément numéro et suffixe de voie, reconnaître qu'un nom d'organisation générique désigne plus souvent son siège, et abstention obligatoire lorsqu'aucun indice ne distingue les services co-localisés.
