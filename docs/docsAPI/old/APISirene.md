# Connexion à l'API Sirene - Mode d'emploi

L'URL d'appel à l'API est `https://api.insee.fr/api-sirene/3.11`.

## Première connexion

Pour pouvoir interroger l'API la première fois, il faut procéder aux étapes suivantes :

### 1. Se connecter au portail

Se connecter au portail en mode "Connexion pour les externes".

### 2. Créer un compte

Se créer un compte ou utiliser son ancien compte du portail `https://api.insee.fr/catalogue/`. Les caractères accentués sont à éviter car mal gérés par le portail.

### 3. Créer une application

- Donner un nom et une description (obligatoire)
- Renseigner un domaine utilisé par l'application (conseillé)
- Choisir le mode de création "simple" (et non "backend to backend" qui ne fonctionnera en aucun cas)
- Renseigner le type d'applicatif
- Laisser vide le client ID
- Laisser vide le champ "souscription"
- Cliquer sur "créer application" sur le récapitulatif

### 4. Souscrire via l'application

- Aller sur "Catalogue", puis "Applicatif", puis "API Sirene"
- Cliquer sur "Souscrire"
- Choisir le plan "Public" (le seul disponible)
- Cliquer sur "Suivant"
- Sélectionner l'application créée plus tôt
- Cliquer sur "Suivant"
- La souscription est validée

Une clé (API KEY) est alors fournie pour se connecter à l'API en mode public.

### 5. Retrouver la clé d'API

- Se connecter à son application
- Aller à l'onglet "Souscriptions"
- Choisir la souscription à l'API Sirene
- La clé d'API apparaît à droite

On peut la renouveler ou la révoquer. Elle n'a pas de durée limite.

## Usage de la clé d'API

### 1. Écriture d'une requête

La clé d'API se transmet dans le header de la requête dans le champ `X-INSEE-Api-Key-Integration`.

Exemple de commande curl :

```bash
curl --location 'https://api.insee.fr/api-sirene/3.11/siren/309634954' \
  --header 'X-INSEE-Api-Key-Integration: xxxxxxx'
```

### 2. Utilisation du swagger

- Cliquer sur le bouton "Authorize"
- Renseigner la clé d'API dans le champ "Value"
- Cliquer sur "Authorize" puis "Close"
- Cliquer sur "Try it out" dans le service choisi
- Renseigner votre requête
- Cliquer sur "Execute"


# API Sirene - Documentation des services

## Recherche sur une variable non-historisée

Permet de sélectionner les Siren (resp. les Siret) pour lesquels une certaine variable a une valeur spécifique. Dans le cas des variables non-historisées, il s'agit toujours de la valeur courante.

### Syntaxe

```
nomVariable:valeur
```

**Règles :**
- `nomVariable` doit correspondre exactement (casse comprise) à la variable de sortie de l'interrogation unitaire
- Toutes les variables peuvent être utilisées, y compris les indicatrices
- Quelques subtilités pour les variables au format date

### Exemples

1. Recherche de tous les établissements du Siren 775672272 :
   ```
   https://api.insee.fr/api-sirene/3.11/siret?q=siren:775672272
   ```

2. Recherche de toutes les unités purgées :
   ```
   https://api.insee.fr/api-sirene/3.11/siren?q=unitePurgeeUniteLegale:true
   ```

3. Recherche de tous les établissements des unités purgées :
   ```
   https://api.insee.fr/api-sirene/3.11/siret?q=unitePurgeeUniteLegale:true
   ```

4. Recherche de tous les établissements de la commune de Malakoff (code commune=92046) :
   ```
   https://api.insee.fr/api-sirene/3.11/siret?q=codeCommuneEtablissement:92046
   ```

---

## Recherche sur une variable historisée

Permet de sélectionner les Siren (respectivement les Siret) pour lesquels une certaine variable a une valeur spécifique sur au moins une période. Dans le cas des variables historisées, on peut obtenir leur valeur courante ou la valeur qu'elles ont eue depuis la création de l'unité légale (respectivement l'établissement).

### Syntaxe

```
periode(nomVariable:valeur)
```

**Règles :**
- `nomVariable` doit correspondre exactement (casse comprise) à la variable de sortie de l'interrogation unitaire
- Toutes les variables peuvent être utilisées, y compris les indicatrices
- Quelques subtilités pour les variables au format date
- L'utilisation du paramètre `date` permet de limiter la recherche à une seule période (qui inclut la date saisie)

### Format des noms de variables

Les noms de variables sont au format camelcase : premier mot tout en minuscules, les suivants avec l'initiale en majuscule, sans espaces ni accents.

**Exemple :** `economieSocialeSolidaireUniteLegale`

### Notion de période

Une période au sens de l'API est un intervalle de temps durant lequel aucune variable historisée n'a été modifiée. Le nombre de périodes est toujours supérieur ou égal à 1, pour les unités légales comme pour les établissements.

### Exemples

1. Recherche de toutes les UL dont la dénomination contient ou a contenu le mot GAZ :
   ```
   https://api.insee.fr/api-sirene/3.11/siren?q=periode(denominationUniteLegale:GAZ)
   ```

2. Recherche de toutes les UL qui ont été cessées :
   ```
   https://api.insee.fr/api-sirene/3.11/siren?q=periode(etatAdministratifUniteLegale:C)
   ```

3. Recherche de tous les établissements dont le code de l'activité principale a été (ou est) 33.01 :
   ```
   https://api.insee.fr/api-sirene/3.11/siret?q=periode(activitePrincipaleEtablissement:33.01)
   ```

   **Note :** 33.01 appartenant à une ancienne nomenclature, une unité légale (resp. un établissement) ne peut pas avoir ce code en valeur courante si elle est active.

---

## Commentaires

- Pour un utilisateur n'ayant pas le droit d'accès aux données en diffusion partielle, un contrôle est fait sur chaque variable présente dans la requête multicritères (paramètre `q`)
- Les unités légales ou les établissements pour lesquels au moins l'une de ces variables est en diffusion partielle, n'apparaîtront pas dans les résultats de la recherche
- Les recherches sur variables historisées ou non-historisées sont accessibles par la console