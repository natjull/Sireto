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

Le premier audit ne suffit pas à justifier un nouveau modèle. Un second passage
a donc comparé chaque vérité aux autres établissements du même SIREN, appliqué
le ranker historique gelé au grand pool interne et recherché des preuves
externes sur les relations les plus opaques. Les conclusions corrigées sont
présentées ci-dessous.

## Second passage : audit autonome approfondi

### De nombreux labels pointent vers le mauvais établissement

Pour chaque requête, tous les établissements du SIREN historique ont été
comparés à l'adresse CRM.

- **28 cas sur 63** possèdent un autre établissement du même SIREN dont
  l'adresse correspond sensiblement mieux que celle du label ;
- dans **23 cas**, ce meilleur établissement alternatif est actif ;
- **12 cas** possèdent un autre établissement à une adresse pratiquement
  exacte, dont **11 actifs** ;
- dans **10 de ces 12 cas**, cet établissement plus cohérent est déjà présent
  dans les 100 candidats actuels.

Exemples certains :

| Requête | Label historique | Établissement du même SIREN plus cohérent |
|---:|---|---|
| 76 | mairie d'Albert, place Émile-Leturcq | école Alphonse-Daudet, rue des Capucines |
| 3462 | établissement fermé, 30 avenue de la Gare | établissement actif, 3 allée Luchino-Visconti |
| 4596 | ODONTOPOLE, 42 boulevard Carnot | ODONTOPOLE actif, 70 boulevard Faidherbe |
| 5890 | office notarial, 19 place des Ramacles | même office actif, 62 avenue de la Margeride |
| 7107 | SCM MEDIPOL, 9 rue Frédéric-Mistral | SCM MEDIPOL actif, 11 avenue de Pologne |
| 7472 | ESGCV, 27 rue James-Watt | ESGCV Tours actif, 35 rue Jehan-Fouquet |
| 9037 | SCE, 7 rue Arago | SINOTEC actif, 555 rue Gustave-Eiffel |
| 11990 | OPPELIA, 97 rue Jules-Siegfried | OPPELIA actif, 6 place Jules-Ferry |
| 12467 | Pharmacie Bodart, 220 rue des Postes | Pharmacie Bodart active, 3 place Barthélemy-Dorez |
| 13185 | REALITES, 103 route de Vannes | REALITES actif, 1 impasse Claude-Nougaro |
| 16090 | 4422 HOLDING, 11 rue de Villeneuve | 4422 HOLDING actif, 20 rue Saarinen |

Ces requêtes n'ont aucune date de référence. Dans ce contexte, demander au
système de retrouver l'ancien SIRET fermé plutôt que l'établissement actif à
l'adresse CRM n'est pas une cible cohérente.

### Le défaut dépasse largement les 63 erreurs

Un runner reproductible a ensuite contrôlé les 2 565 requêtes dev, sans
modifier aucun label :

- **231** requêtes ont un autre SIRET du même SIREN à l'adresse exacte ;
- **165** ont au moins un autre SIRET actif à cette adresse ;
- **87** ont un label fermé et au moins un sibling actif à l'adresse exacte ;
- **29** ont plusieurs siblings actifs possibles à la même adresse ;
- seules 5 requêtes n'ont pas d'adresse CRM exploitable.

Les 165 cas ne sont pas automatiquement 165 labels faux : plusieurs
établissements peuvent partager une adresse. Ils démontrent en revanche que
le SIRET exact n'est pas toujours identifiable à partir des champs CRM. Ce
volume est supérieur de très loin aux 25 erreurs maximales autorisées par une
cible à 99 %.

Artefact immuable :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/site_label_audit_dev_c33b80855f560074_ac971e0`.

### Certaines relations opaques sont néanmoins réelles

Plusieurs noms apparemment différents correspondent bien à une enseigne, un
ancien nom ou une société d'exploitation :

- Intermarché Louvigny est bien exploité par LESCONI au SIRET historique
  indiqué, ce que confirment un
  [arrêté préfectoral](https://www.calvados.gouv.fr/contenu/telechargement/7627/87523/file/numero4-16fevrier20095ee8.pdf)
  et la fiche du magasin ;
- INFODESCA appartient à l'environnement Descours & Cabaud, comme l'indiquent
  les [mentions légales du groupe](https://www.descours-cabaud.com/mentions-legales/) ;
- Compagnie des Signaux a effectivement porté les noms Ansaldo STS puis
  Hitachi Rail, relation confirmée par la
  [présentation de l'entreprise](https://www.railopenlab.com/en/membres/compagnie-des-signaux) ;
- Eurex Statuo Conseil est bien implanté au 149 avenue du Golf à Baillargues,
  mais le SIREN actuel n'est plus le SIREN fermé utilisé par le label
  historique.

Ces cas demandent une information d'alias ou d'historique d'entreprise. Un
score de ressemblance supplémentaire ne peut pas inventer cette relation.

### Certains labels sont contredits par des sources directes

- Le CRM « Hôtel Mercure place de Jaude » est à 1 avenue Julien selon le
  [site officiel Accor](https://all.accor.com/hotel/9171/index.fr.shtml), alors
  que le label pointe vers l'hôtel Oceania au 82 boulevard François-Mitterrand,
  adresse confirmée par le
  [site officiel Oceania](https://www.oceaniahotels.com/oceania-clermont-ferrand/acces-contact).
- Le CRM GLOBECAST à Brétigny pointe vers KINEPOLIS PROSPECTION, alors que
  l'établissement Kinepolis est explicitement un cinéma au 5 rue
  Michèle-Morgan dans la
  [fiche publique Acceslibre](https://acceslibre.beta.gouv.fr/app/91-bretigny-sur-orge/a/cinema/erp/kinepolis-prospection/).
- Le SIRET DMS MIROITERIE est réellement une entreprise de vitrerie domiciliée
  au « Foyer numérique » d'Arras, comme le montre
  [l'Annuaire des Entreprises](https://annuaire-entreprises.data.gouv.fr/etablissement/49398278900031).
  Le nom du bâtiment dans le CRM ne suffit donc pas à désigner cette entreprise
  plutôt qu'un autre occupant.

Ces labels doivent être classés `UNRESOLVED` ou corrigés dans une nouvelle
version du benchmark. Ils ne doivent pas servir à apprendre des rapprochements
arbitraires.

### Le ranker historique ne résout pas le problème

Le ranker rapide historique `xgbranker_fast_20260124_210313.json` a été appliqué
sans modification ni entraînement à l'union interne complète des canaux.

- il place **22 des 63** vérités éliminées dans ses 100 premiers ;
- il en laisse **41** au-delà de 100, parfois au-delà du rang 5 000 ;
- même en supposant irréalistement qu'il ne perde aucun des 2 495 succès
  actuels, son plafond serait 2 517 / 2 565 = **98,13 %**.

Un nouveau sélecteur ressemblant au ranker historique ne peut donc pas, à lui
seul, franchir 99 % sur ces labels.

## Décision technique

La priorité n'est plus une nouvelle architecture de sélection. Il faut d'abord
réparer le contrat de vérité :

1. lorsqu'aucune date historique n'est fournie, la vérité doit désigner
   l'établissement actif correspondant à l'adresse CRM ;
2. si seule l'entreprise propriétaire est démontrable, mais pas un
   établissement précis, le cas doit être `AMBIGUOUS_SITE` et non
   `MATCH_EXACT` ;
3. les anciens noms, enseignes et sociétés d'exploitation doivent être
   conservés dans une table d'alias versionnée ;
4. les équipements publics doivent être reliés d'abord à la collectivité au
   niveau SIREN ; un SIRET exact ne doit être imposé que si le site est
   identifiable ;
5. le benchmark corrigé doit être versionné séparément. Le benchmark gelé
   actuel reste intact pour conserver la traçabilité.

Après cette correction, les seules modifications simples du retrieval à
réévaluer sont :

- protéger les correspondances d'adresse exactes ;
- préférer l'établissement actif à l'adresse CRM lorsqu'un SIREN fiable est
  déjà trouvé ;
- injecter les alias historiques vérifiés dans la recherche par nom.

Ce n'est qu'après cette nouvelle mesure qu'un modèle d'admission pourra être
justifié ou définitivement écarté.

En conséquence, **aucune modification du retrieval n'est promue à ce stade**.
Optimiser les scores sur le benchmark actuel apprendrait en partie à choisir
arbitrairement entre plusieurs SIRET indiscernables ou à reproduire un site
fermé sans date historique.
