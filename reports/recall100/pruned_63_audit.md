# Audit des 63 SIRET trouvés puis éliminés

## Réponse courte

Les 63 erreurs ne viennent pas d'un réglage unique manifestement mauvais.

- **28 SIRET** sont bien placés dans au moins une méthode de recherche
  (rang 1 à 100), mais cette preuve est noyée lors de la fusion.
- **19 SIRET** n'apparaissent qu'entre les rangs 101 et 500.
- **16 SIRET** n'apparaissent qu'après le rang 500.
- **35 cas sur 63** ont un nom et une adresse faiblement liés au SIRET
  historique. Une simple règle de ressemblance ne peut pas les sauver de façon
  sûre.

Il existe quelques corrections simples, mais elles ne suffisent pas à combler
les 45 succès manquants pour atteindre 99 %.

## Pourquoi ils sont éliminés mécaniquement

### Les 13 établissements fermés issus de l'overlay

Treize vérités ne sont présentes que dans l'archive des établissements fermés.
La règle actuelle ne donne un score complet qu'aux candidats du store V7. Dans
l'overlay, elle réserve seulement une place au canal « mots du nom » et dix
places au canal « caractères du nom ».

Conséquence : un établissement fermé peut avoir une adresse parfaite et ne
recevoir aucune place.

Trois cas sont particulièrement nets :

| Requête | CRM | Vérité historique | Observation |
|---:|---|---|---|
| 128 | Collège Françoise d'Amboise, 11 rue Mondésir | OGEC Françoise d'Amboise, 11 rue Mondésir | adresse exacte, rang overlay 4 |
| 12274 | Laboratoire Diagnovie, 253 rue Jules Guesde | Laboratoire Demouveaux, 253 rue Jules Guesde | adresse exacte, rang overlay 1 |
| 15132 | Feller Henri Notaire, 31B avenue de Fontainebleau | Alain Gressier, 31 avenue de Fontainebleau | adresse quasi exacte, rang overlay 2 |

Ajouter quatre places prioritaires pour les adresses exactes récupère ces trois
cas, mais déplace aussi une vérité précédemment conservée. Le résultat global
ne passe que de 97,27 % à **97,35 %**.

### Les 50 candidats présents dans le store V7

Une fois les scores actuels recalculés :

| Position du vrai SIRET dans la fusion V7 | Nombre |
|---|---:|
| Dans les 100 premiers | 1 |
| 101 à 200 | 17 |
| 201 à 500 | 17 |
| Après 500 | 15 |

Le seul candidat encore dans les 100 premiers de la fusion V7 est ensuite
déplacé par les places réservées à l'overlay. Pour les 49 autres, le problème
est plus profond : ils obtiennent un score trop faible par rapport aux autres
candidats de la commune.

## Lecture métier des 63 cas

Les groupes ci-dessous sont exclusifs pour pouvoir compter exactement 63 cas.
Ils décrivent la meilleure explication visible, pas une validation définitive
du label historique.

### 1. Adresse directement exploitable — 12 cas

Le nom légal peut être différent, mais l'adresse du SIRET historique ressemble
fortement à celle du CRM.

Requêtes :
`107`, `128`, `1736`, `3135`, `4801`, `6060`, `6327`, `11412`, `12274`,
`13585`, `15132`, `16089`.

Exemples :

- `4801` : NORLANDA, cours Caffarelli → SINAY, 117 cours Caffarelli ;
- `13585` : Digital Campus, 275 boulevard Marcel Paul → ESGCV Nantes,
  275 boulevard Marcel Paul ;
- `16089` : ADOPT au centre Parinor → ADOPT', centre commercial Parinor.

Ces cas justifient une meilleure protection des preuves d'adresse, mais
l'adresse seule est dangereuse dans les immeubles partagés.

### 2. Nom directement exploitable — 8 cas

Le nom commercial ou l'enseigne reste fortement reconnaissable.

Requêtes :
`157`, `7472`, `7747`, `8814`, `12293`, `12467`, `13185`, `16830`.

Exemples :

- `12293` : MAPP → MAPP ;
- `13185` : REALITES → REALITES ;
- `16830` : PRIMON → PRIMION TECHNOLOGY.

Ici aussi, la méthode spécialisée voit souvent la vérité, mais sa position est
écrasée par la fusion avec les autres méthodes.

### 3. Bonne entreprise déjà présente, mauvais établissement conservé — 13 cas

Au moins un autre SIRET du bon SIREN est déjà dans les 100. L'entreprise est
donc reconnue, mais le bon établissement n'est pas conservé.

Requêtes :
`76`, `542`, `3462`, `4596`, `5890`, `5976`, `7107`, `7135`, `9037`,
`11990`, `14262`, `14411`, `16090`.

Exemples :

- une école est reliée à la commune, mais un autre établissement de la commune
  est retenu ;
- un stade est relié à Angers SCO, mais le mauvais site du club est retenu ;
- un ancien établissement fermé est remplacé par un autre établissement du
  même SIREN.

C'est une piste simple en apparence : conserver plusieurs établissements des
entreprises déjà reconnues. Mais ajouter naïvement le canal « autres sites du
SIREN » récupère 2 erreurs et en crée 5. Il faut choisir les sites avec une
preuve d'adresse ou de date, pas les ajouter indistinctement.

### 4. Équipement public relié à son propriétaire administratif — 12 cas

Le CRM décrit un bâtiment ou un service, tandis que le label pointe vers la
mairie, la commune ou l'administration propriétaire. Les deux textes peuvent
être totalement différents.

Requêtes :
`327`, `492`, `540`, `719`, `885`, `1554`, `2341`, `4422`, `4507`, `4586`,
`9815`, `11630`.

Exemples :

- « Bibliothèque Condition des Soies » → COMMUNE DE LYON ;
- « Caméra avenue du Général-de-Gaulle » → COMMUNE D'ESSEY-LES-NANCY ;
- « Pôle Enfance » → COMMUNE DE GRIGNY-SUR-RHÔNE.

Ce n'est pas réellement un problème de ressemblance de chaînes de caractères.
Il faut soit une règle métier « équipement public → collectivité », soit une
information de propriété absente des champs CRM/SIRENE utilisés.

### 5. Relation faible, ancienne ou non visible — 18 cas

Ni le nom ni l'adresse courante du SIRET historique n'apportent une preuve
directe suffisante, et aucun autre établissement du même SIREN n'est déjà
conservé.

Requêtes :
`1290`, `1768`, `2227`, `2299`, `4514`, `5041`, `5628`, `5778`, `5859`,
`6626`, `6714`, `8107`, `11039`, `11173`, `13155`, `13167`, `14651`,
`15639`.

Exemples :

- « INTERMARCHÉ Louvigny » → LESCONI ;
- « Hôtel Mercure place de Jaude » → OCEANIA / Société hôtelière du pays
  clermontois ;
- « GROUPAMA » → une personne physique à une adresse voisine ;
- « GLOBECAST France » → KINEPOLIS PROSPECTION ;
- un enregistrement SIRENE entièrement masqué par `[ND]`.

Ces 18 labels sont tous issus du référentiel historique non réaudité, sans date
de référence. Ils doivent être vérifiés humainement avant de concevoir une
logique complexe destinée à les reproduire. Ils ne doivent toutefois pas être
modifiés ou exclus silencieusement du benchmark.

## Tests de corrections simples

Toutes les variantes restent limitées à 100 candidats et sont évaluées
uniquement sur dev.

| Correction exploratoire | Recall | Erreurs récupérées | Nouvelles erreurs | Effet net |
|---|---:|---:|---:|---:|
| Règle actuelle | 97,27 % | — | — | — |
| Réserver jusqu'à 4 adresses exactes de l'overlay | 97,35 % | 3 | 1 | +2 |
| Donner un poids aux autres sites du même SIREN | 97,15 % | 2 | 5 | −3 |
| Réserver 10 candidats sparse de l'overlay | 97,15 % | 4 | 7 | −3 |
| Donner un score complet à tout l'overlay | 96,41 % | 6 | 28 | −22 |

Le résultat important est négatif : **aucune petite règle évidente ne récupère
les 45 cas nécessaires**. Ajouter des candidats utiles en chasse presque
toujours d'autres qui étaient corrects.

## Orientation raisonnable

Avant tout nouveau modèle :

1. valider humainement les 18 relations faibles ou opaques ;
2. conserver la petite règle « adresse exacte de l'overlay » comme candidate,
   mais ne pas la promouvoir sur le seul résultat dev ;
3. séparer explicitement le cas métier « équipement public → collectivité » ;
4. traiter le choix d'établissement seulement lorsque le bon SIREN est déjà
   identifié.

Après ces vérifications, on saura si le déficit restant vient réellement de la
sélection technique ou d'une vérité historique que les données disponibles ne
permettent pas de retrouver.
