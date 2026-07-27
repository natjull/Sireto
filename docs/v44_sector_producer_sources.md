# V4.4 — Sources producteur des identifiants sectoriels

Date de vérification : 27 juillet 2026.

## Périmètre observé

L'inventaire est dérivé des champs sectoriels présents dans
`matching_etablissements` des réponses officielles gelées pour les 172 cas
`AUTO_MATCH`. Il ne qualifie pas le lien CRM–SIRET.

| Famille | Champ source | Identifiants uniques | Cas concernés |
|---|---|---:|---:|
| UAI | `liste_uai` | 29 | 21 |
| FINESS | `liste_finess` | 33 | 19 |
| BIO | `liste_id_bio` | 10 | 10 |
| RGE | `liste_rge` | 45 | 4 |

L'union concerne 52 cas. Les répétitions d'un même identifiant dans plusieurs
vues de l'API Recherche d'entreprises sont conservées comme provenance, mais ne
sont pas comptées comme des preuves indépendantes.

## UAI

- Producteur : ministère de l'Éducation nationale.
- Source : jeu `fr-en-annuaire-education` sur
  `https://data.education.gouv.fr`.
- Requête : égalité stricte sur `identifiant_de_l_etablissement`.
- Données récupérables : nom et type de l'établissement, statut public/privé,
  adresse, commune, état, dates d'ouverture et de mise à jour, et
  `siren_siret` lorsqu'il est renseigné.
- Résultat du passage : 27 identifiants trouvés sur 29. Les réponses vides pour
  `0694606G` et `0801774U` sont conservées telles quelles ; elles ne sont ni
  transformées en absence d'établissement, ni utilisées comme label.

## FINESS

- Producteur : Agence du Numérique en Santé (ANS).
- Source : snapshot quotidien `FINESS - Structures FINESS-STR`, référencé par
  `https://www.data.gouv.fr/api/1/datasets/finess-structures-1/`.
- Requête : le snapshot JSON.GZ complet est archivé, puis les objets portant un
  `numFinessEge` exact sont extraits localement.
- Données récupérables : numéro FINESS géographique, SIRET, dénominations,
  adresses, catégories et statut de la structure, dates et événements présents
  dans le snapshot.
- Traçabilité temporelle : `generatedAt` du snapshot, URL de la ressource,
  date HTTP de collecte et hash du fichier brut.
- Résultat du passage : 33 identifiants trouvés sur 33.

## BIO

- Producteur : Agence Bio.
- Source : `https://opendata.agencebio.org/api/gouv/operateurs/`.
- Requête : paramètre exact et sensible à la casse `numeroBio`.
- Données récupérables : numéro BIO, SIRET déclaré, raison sociale,
  dénomination courante, adresses, activités, productions, certificats,
  organisme certificateur et date de mise à jour.
- Résultat du passage : 10 identifiants trouvés sur 10.

## RGE

- Producteur : ADEME.
- Source : jeu quotidien `liste-des-entreprises-rge-2-new` sur
  `https://data.ademe.fr`.
- Point de schéma essentiel : `liste_rge` contient des
  `code_qualification`. Un tel code est partagé par de nombreuses entreprises ;
  ce n'est pas un identifiant d'établissement.
- Requête : égalité conjointe sur `code_qualification` et le SIRET auquel le
  code était attaché dans le payload observé.
- Données récupérables : SIRET, nom et adresse de l'entreprise, code et nom de
  qualification, domaine, organisme, dates de validité et URL du certificat.
- Résultat du passage : 45 couples code/SIRET trouvés sur 45.

## Contrat d'utilisation

Chaque réponse brute est archivée avec l'URL demandée et finale, la date de
collecte, le producteur, l'identifiant, les en-têtes HTTP utiles et un SHA-256.
Le snapshot FINESS complet est partagé entre les 33 extractions, sans simuler
33 appels indépendants.

Ces données peuvent servir à constituer des faits observables ou une file de
revue. Elles ne prouvent pas à elles seules que le SIRET observé est le bon
match du CRM. Le collecteur ne produit donc aucun champ de vérité terrain, de
correction ou d'adjudication.
