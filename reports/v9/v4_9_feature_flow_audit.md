# V4.9 — audit du flux de features après l'échec random V4.8

Date : 28 juillet 2026  
Statut : audit statique et diagnostic post-mortem, aucun retuning.

## Verdict

Le problème révélé par V4.8 n'est pas que le système manquerait globalement
de similarité nom/adresse. Il perd des informations métier entre le ranker et
l'accepteur.

Le ranker possède 64 features. L'accepteur en reconstruit 80, mais ce ne sont
pas les 64 features du top-1 et du top-2 : il ne conserve qu'un sous-ensemble
de 20 mesures de similarité, dupliquées en top-1/top-2/delta, plus 20
statistiques de scène.

Des signaux importants disparaissent donc avant la décision AUTO :

- top-1 égal au SIRET d'entrée ou seulement au même SIREN ;
- provenance du candidat par l'overlay SIRET/SIREN d'entrée ;
- état actif/fermé du candidat et état du SIRET d'entrée ;
- établissement siège ;
- forme juridique et association ;
- indicateur CRM « école » ;
- correspondance exacte du nom, de l'adresse et du numéro de voie ;
- type de nom ayant produit le meilleur match ;
- surtout, activité et fonction précise de l'établissement.

## Ce que voient réellement les modèles

### Ranker candidat — 64 features

Le ranker voit :

- 35 environ de similarité nom/adresse ;
- forme juridique, siège, association et indicateur école ;
- rangs et scores des canaux de retrieval ;
- relation avec le SIRET/SIREN d'entrée ;
- état actif/fermé et provenance du candidat.

Il ne voit déjà pas directement :

- le code NAF/APE de l'établissement ;
- une catégorie métier détaillée du site ;
- une incompatibilité explicite « mairie contre école » ou « maternelle
  contre primaire ».

### Accepteur query-level — 80 features

L'accepteur voit :

- 20 statistiques globales de scores et de concurrence ;
- pour 20 similarités seulement : valeur top-1, valeur top-2 et différence.

Il ne reçoit aucun des neuf signaux V4.1 de relation, état et provenance. Il
ne reçoit pas non plus `legal_form_category`, `is_siege`, `is_association`,
`is_crm_school`, `geo_exact_match`, `name_norm_exact`,
`street_number_match` ni `type_of_max_name`.

L'accepteur peut donc savoir que « les noms et adresses se ressemblent
beaucoup », mais pas pourquoi le ranker a choisi ce candidat ni si sa fonction
précise contredit le CRM.

## Lecture des trois erreurs random

### Mairie contre école

Le CRM demande la mairie de Merville-Franceville-Plage. Le top-1 est l'école
primaire voisine :

- même SIREN communal ;
- adresse presque identique ;
- nom générique de la commune ;
- score `HARD_W1` 0,980186.

Les preuves officielles distinguent pourtant nettement la fonction scolaire
de la fonction mairie. Aucun feature accepteur ne code cette contradiction.

### Maternelle contre primaire

Le CRM demande l'école maternelle André-Philip au 46 rue Dunoir. Le top-1
courant est l'école primaire au 48 :

- même SIREN municipal ;
- top-1 exactement égal au SIRET d'entrée CRM pourtant erroné ;
- adresse quasi identique ;
- score `HARD_W1` 0,999190.

Le modèle possède toutes les raisons statistiques de faire confiance à ce
candidat. Il ne possède pas la distinction de niveau scolaire qui permettrait
de le refuser.

### FAM contre MAS/EAM

Le CRM mélange « FAM MAS » pour un complexe AFAPEI. Le top-1 courant
représente une autre fonction du complexe :

- même SIREN ;
- même zone et nom d'entité ;
- plusieurs établissements du même SIREN dans le pool ;
- score `HARD_W1` 0,992347.

Cette scène devrait déclencher une abstention pour ambiguïté de fonction, pas
une confiance presque maximale.

## Architecture corrigée à tester

La prochaine brique ne doit pas être un modèle plus complexe. Elle doit être
un **garde-fou de fonction de site**, déterministe et placé après
l'accepteur :

```text
top-1 proposé
  → accepteur statistique
  → contrôle fonction exacte du site
      compatible : conserver AUTO
      contradiction ou mélange : REVIEW
```

Ce garde-fou ne crée jamais un AUTO. Il peut seulement transformer un AUTO en
REVIEW. Il vise donc directement la précision, avec comme risque principal
une perte de couverture mesurable.

Les informations peuvent être calculées localement :

- libellé CRM ;
- dénomination, enseigne et activité principale SIRENE ;
- SIREN/SIRET et état administratif ;
- fonctions des autres candidats du même SIREN.

Un lexique versionné doit distinguer au minimum :

- administration : mairie/hôtel de ville ;
- enseignement : maternelle, primaire/élémentaire, collège, lycée ;
- médico-social : FAM, MAS, EAM, IME, EHPAD, foyer ;
- petite enfance : crèche ;
- santé : hôpital, clinique, pharmacie.

Les règles d'incompatibilité doivent être explicites et testables. Aucun LLM
n'entre dans l'inférence.

## Limite scientifique

Cette hypothèse vient directement des erreurs du random V4.8 désormais
consommé. Ce random ne peut donc pas valider V4.9. Il pourra seulement servir
au développement et au diagnostic.

Toute promotion exige :

1. règles gelées avant le nouveau score ;
2. mesure rétrospective publiée comme exploratoire seulement ;
3. nouvelle population CRM jamais utilisée par V4.3–V4.8 ;
4. nouvelle adjudication autonome avec preuves ;
5. évaluation sans modification des règles ni du seuil.

Le test final historique reste fermé.

