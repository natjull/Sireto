# Utilité des textes BODACC pour les ratés du retrieval

Les textes BODACC contiennent des preuves exploitables de mandat, de groupe,
de changement d'exploitant et d'activité. Ils peuvent enrichir la recherche
initiale, mais leur encodage brut avec notre Gemma fine-tuné ne démontre pas
un rattrapage général des ratés. Le résultat le plus solide reste la
combinaison d'une preuve corporate et d'une localisation propre du site.
La nouvelle piste la plus intéressante concerne les liens de groupe décrits
dans les actes, notamment Savencia et Fromageries du Lévézou.

Cet audit du 12 octobre 2026 couvre les 31 ratés de la meilleure recette V32
sur le pilote 526 déjà consommé. Le score reste 495/526, soit 94,1065% brut et
93,6717% pondéré. Aucun nouveau recall, candidat ou modèle produit n'est créé.

## Périmètre des annonces et des recherches

Les copies locales anciennes omettent plusieurs champs libres. L'API publique
BODACC a permis de lire les champs complets des annonces parues au plus tard
le 1er août 2026. Les nouvelles lectures sont des sources actuelles portant sur
des publications anciennes, pas une reproduction certifiée du snapshot V32.
Elles restent dans les dossiers de diagnostic et ne deviennent pas des entrées
de production sélectionnées à partir des vérités humaines.

- 337 annonces distinctes vérifiées concernent les sociétés cibles publiques,
  comme sujet, précédent exploitant/propriétaire ou partie mentionnée dans un
  texte. Les références aux SIREN sont vérifiées après les résultats de
  recherche : la recherche API sur chiffres espacés renvoie aussi du bruit.
- 20/31 requêtes ont des annonces ainsi vérifiées ; 9 autres n'ont pas de
  référence vérifiée dans ce périmètre. Deux cas à diffusion partielle ne sont
  pas backfillés pour reconstituer une identité ou une adresse masquée.
- 13 requêtes ont au moins une annonce portée par un autre sujet : limiter
  l'inspection à la société courante aurait omis des preuves de prédécesseur,
  d'absorption ou de société mère.
- Une recherche nationale complémentaire dans `acte`,
  `modificationsgenerales` et `divers` porte sur 12 libellés distinctifs, avec
  686 résultats API intégralement lus. La vérification littérale conserve 267
  annonces ; aucune ne mentionne un SIREN cible public du pilote dans ses
  parties structurées ou ses passages correspondants. Cette recherche n'est
  ni une recherche sémantique exhaustive ni une preuve d'absence universelle.
- La sonde du modèle utilise 240 passages uniques, y compris des annonces des
  sociétés identifiées comme points d'entrée corporate. Les 18 passages trop
  longs sont découpés avec recouvrement, pas tronqués. Les identifiants sont
  retirés avant encodage. Gemma reste gelé au checkpoint 1000, batch 8, 512 tokens.

## Cas dont les preuves corporate sont discriminantes

### EUREDEN GROUP

La cible humaine est un établissement de BERNARD AGRISERVICE, pas un
établissement légalement dénommé EUREDEN. L'annonce A202300971476 du 21 mai 2023
décrit le projet d'absorption de BERNARD AGRISERVICE par BDV SERVICES et
énonce leur détention par une société mère commune, COOPERATIVE EUREDEN.
L'annonce A202300971229 contient l'autre publication du même projet.
Les annonces suivantes portent OUEST NEGOCE AGRICULTURE comme nouvel
exploitant et BERNARD AGRISERVICE comme précédent exploitant.

Le nom CRM possède donc un pont officiel vers l'entreprise cible. La clause
de détention est une preuve plus précise que le seul mandat de président
EUREDEN présent dans le RNE. Elle ne suffit pas à choisir le SIRET : les deux
sociétés ont des sites locaux et il faut conserver les versions fermées
pertinentes. Un projet et sa date prévue ne prouvent pas seuls sa réalisation.

Le contrôle V51 avait réduit 755 sites nationaux à deux dans la commune,
dont le SIRET humain exact. Cette discrimination existe avant apprentissage.
Elle n'est pas encore un gain mesuré du collecteur 100.

### CATZAM

L'annonce B20240111141 du 11 juin 2024 nomme CATZAM comme président d'AU45 ; le
RNE confirme le rôle de la personne morale. Les 12 sites nationaux des sociétés
concernées se réduisent à UNsite au 45 rue de Gand : le SIRET attendu. Un
concurrent actuellement dans le top 100 appartient aussi à une société
présidée par CATZAM, dans la même rue mais à un autre numéro.

Cette preuve sépare réellement la cible d'un concurrent. Le texte BODACC
corrobore toutefois un rôle déjà disponible dans le RNE : il ne faut pas
compter deux fois le même fait comme deux découvertes indépendantes.

### FOURNIER

Le rôle de président FOURNIER apparaît dans le RNE et l'annonce A202201341580.
Le nom d'acteur élargi à ses dénominations contenant FOURNIER donne 609 sites,
dont deux dans la commune : la cible SODIMOB fermée et le concurrent actif
actuellement émis. Les actes 2023 décrivent aussi un projet d'absorption de
SODIMOB par SODICOOC. Le texte fournit un contexte de cycle de vie, mais pas
un motif pour supprimer l'ancien SIRET humain du retrieval.

L'adresse du stock est 350 rue des Clauwiers, contre 1110 avenue de l'Épinette dans
le CRM. Les actes doivent donc être rattachés à leurs parties et à leurs
sites, pas utilisés comme une adresse propre interchangeable. Certains
anciens actes portés par ce dossier citent une autre société également
appelée SODIMOB et un autre SIREN : le nom seul ne résout pas cette ambiguïté.

## Nouvelles clauses de groupe et ponts restant nécessaires

### SB ALLIANCE INFORMATIQUE vers FROMAGERIES DU LEVEZOU

Les annonces A2022021184 et A2022021185 de 2022 décrivent un projet commun de
fusion entre FROMAGERIES PAPILLON et FROMAGERIES DU LEVEZOU. Elles nomment
SAVENCIA FROMAGE & DAIRY EUROPE comme société mère détenant leur capital.
Ce fait n'est pas la simple similarité de deux noms : il fournit un lien
corporate explicite vers la cible humaine.

L'autre extrémité exige une preuve indépendante. BODACC retrace, sous le même
SIREN780876405, les noms CL-INFORMATIQUE, SB ALLIANCE INFORMATIQUE puis
SAVENCIA IT & DIGITAL SERVICES. Le RNE donne COMPAGNIE LAITIERE EUROPEENNE
comme président de cette société, et SAVENCIA SA comme administrateur de
COMPAGNIE LAITIERE EUROPEENNE. Ces mandats sont du contexte, pas des actes de
propriété à déduire automatiquement.

Des publications historiques du groupe Savencia, repérées dans les rapports
et index publics, citent SB Alliance Informatique et Fromageries du Lévézou.
La publication financière 2021 indexée fournit un corroborant particulièrement
pertinent. Les anciennes URL du groupe sont aujourd'hui protégées ou
redirigées : leurs extraits indexés ne doivent pas être importés comme une
preuve nationale gelée sans conserver le document source complet. La chaîne
est une piste documentaire forte, pas une arête de production déjà validée.

[Publication financière du groupe2021](https://www.savencia-fromagedairy.com/wp-content/uploads/2022/04/Savencia-Rapport-annuel-Financier-2021.pdf),
[publication2022 du groupe mentionnant SB Alliance Informatique](https://www.savencia.com/sites/default/files/2023-02/index%20H%20F%202022%20Savencia.com%20FR.pdf).

La piste à vérifier est une navigation entre sociétés d'un groupe et leurs
sites locaux, sans renommer une fromagerie SB Alliance et sans créer un
dictionnaire CRM→SIRET. Elle ne se résume pas à rapprocher directement le mot
INFORMATIQUE d'une activité de fabrication de fromage.

### INFODESCA vers PROSJET IRRIGARONNE

Les actes A20200138594 et A20200138595 de 2020 nomment DECABER comme propriétaire
des sociétés PROSJET IRRIGARONNE et des autres parties de leurs projets de
fusion. D'autres annonces rattachées à DECABER décrivent des détentions et
opérations de sociétés du réseau. Cela enrichit le graphe de la cible.

Mais aucun de ces passages ne fournit le pont INFODESCA→DECABER. La recherche
nationale des trois champs libres sur INFODESCA ne renvoie aucun résultat.
Les mentions légales de PROLIANS attestent un rôle d'hébergeur d'INFODESCA
auprès d'un site du groupe DESCOURS & CABAUD ; ce rôle ne prouve pas à lui seul
une détention capitalistique. Le dossier reste prometteur comme relation
commerciale/corporate à étayer, pas comme une identité déduite du BODACC seul.
[Mentions légales PROLIANS](https://www.prolians.fr/mentions-legales).

### NAOS vers AUBERT FRANCE

Les actes décrivent bien des sociétés et projets d'absorption Aubert, notamment
A202402251256 et A202402271227 en 2024, avec AUBERT anciennement AUBERT
INTERNATIONAL et AUBERT FRANCE. Ils ne fournissent pas le pont depuis NAOS.
Les 38 annonces contenant littéralement NAOS dans les champs libres recherchés
ne donnent aucune référence aux SIREN cibles publics de ce pilote.

Ces textes pourraient aider APRES une preuve indépendante prestataire→client,
mais ne créent pas cette preuve. Un modèle ne devrait pas décider que NAOS
signifie Aubert sur la seule existence d'un acte de fusion Aubert.

## Ce que reconnaît le modèle gelé dans les passages

La sonde mesure des cosinus entre les requêtes CRM et des passages, sans
sortie SIRET, collecte top 100 ou apprentissage. La banque est choisie pour
un diagnostic après lecture des labels ; elle ne constitue PAS un corpus
d'évaluation du retrieval et n'est jamais utilisée comme pool de candidats.

| Requête | Meilleur passage de son dossier | Meilleur autre passage de diagnostic | Interprétation |
|---|---:|---:|---|
| CATZAM |0,283|0,204| La nomination explicite est reconnue ; il manque encore le rattachement au site. |
| EUREDEN GROUP |0,219 administration ;0,145 horsadministration|0,213| Le rôle est reconnu, mais la clause longue de groupe ne domine pas. |
| FOURNIER |0,165|0,207| Les textes bruts ne séparent pas bien ce cas. |
| SB ALLIANCE INFORMATIQUE |0,106|0,179| Le texte Savencia ne réalise pas spontanément le pont depuis SB Alliance. |
| INFODESCA |0,280|0,185| La meilleure proximité vient d'une adresse de vendeur, pas du lien de groupe manquant. |
| CUISINES SCHMIDT HAUCONCOURT |0,260|0,250| L'activité cuisines est reconnue, mais d'autres cuisinistes restent proches. |
| BLANCHISSERIE |0,103|0,203| La présence de l'activité ne suffit pas à faire dominer ce passage. |

Les valeurs ne sont pas des probabilités ni des seuils d'admission. Le meilleur
autre passage n'est pas un faux positif métier certifié. Elles montrent
cependant que les documents les plus proches ne sont pas toujours ceux qui
contiennent la relation utile. Pour INFODESCA, un avis de vente cite l'adresse
du vendeur à Boé alors que le fonds vendu est ailleurs : mapper ce cosinus au
seul acquéreur ou à un site arbitraire du parent serait incorrect.

La sonde encode des passages libres, sans réinjecter un titre/nom/adresse déjà
présent dans l'index. Elle mesure ainsi l'apport des passages, pas un nouveau
document riche complet. Elle ne compare pas Gemma brut au fine-tuné et ne
prouve pas qu'un encodeur adapté aux textes légaux échouerait. Notre modèle
actuel a été adapté aux vues nom/adresse ; les actes longs constituent un
autre type d'entrée.

## Couverture des 31 ratés par famille

Chaque ligne donne le libellé CRM, la dénomination publique de la cible humaine
et les annonces liées vérifiées. Les six NAOS représentent six requêtes
distinctes ; aucun label n'est corrigé ni retiré du dénominateur.

| Libellé CRM | Cible humaine dans SIRENE | Requêtes | Annonces liées | Apport constaté |
|---|---|---:|---:|---|
| CONVERGENCE MEDIAS | INOX |1|0| Aucun pont retrouvé dans ce périmètre. |
| BLANCHISSERIE | HOM'AGE SERVICES |1|19| Une activité annoncée inclut blanchisserie ; contexte générique, pas identité de site. |
| FOYER TY BREIZ | ENVOL LOIRE ATLANTIQUE |1|0| Source commerciale inadaptée à cet organisme public. |
| ARTEIS SPORT CNSD | CERCLE MIXTE DU CENTRE NATIONAL DES SPORTS DE LA DEFENSE |1|0| Pas de texte commercial pour construire ce lien fonctionnel. |
| SIP PLAINE COMMUNE MAISON DES PROJETS | ABAQUEPLAST |1|23| Modifications génériques, aucun lien vers le libellé public du CRM. |
| OPHLM | DIRECTION DEPARTEMENTALE FINANCES PUBLIQUES CHARENTE |1|0| Aucun pont BODACC ; rapprochement institutionnel à documenter ailleurs. |
| GEDIMAT | LEON BRACQ ET COMPAGNIE |1|22| Scission et autres sociétés Bracq, mais pas de mention GEDIMAT dans les champs recherchés. |
| EBSN | ASS ESPACE BERNADETTE SOUBIROUS |1|0| Association ; acronymie/localisation à étudier dans d'autres sources. |
| SB ALLIANCE INFORMATIQUE | FROMAGERIES DU LEVEZOU |1|7| Clause mère Savencia utile ; pont d'entrée SB Alliance à conserver et valider. |
| SAS5EME ETAGE | LE MUR DU SONGE |1|18| Activité spectacles, pas de nom ou relation expliquant 5EME ETAGE. |
| GROUPAMA SUPPORTS ET SERVICES | EI à site partiellement diffusible |1|nonbackfillé| Pas de reconstruction d'adresse masquée ni métier déduit du seul nom. |
| CEGEDIMCLOUD SMARTRX VILLEURBANNE | SMART GESTION |1|12| Pas de lien vers Cegedim/Smartrx dans les textes. |
| INFODESCA | PROSJET IRRIGARONNE |1|42| Groupe DECABER et opérations documentés ; liaison depuis INFODESCA absente du BODACC inspecté. |
| CATZAM | AU45 |1|15| Président explicitement nommé, séparation forte avec l'adresse propre. |
| ATOL | MODERNE66 |1|17| Activité d'optique et cession, sans mention ATOL reliée à la cible. |
| NAOS | AUBERT FRANCE |6|21| Genealogie Aubert, pas de lien depuis NAOS. |
| TRESCAL | EI publique |1|0| Aucun pont textuel retrouvé ; ne pas présumer l'identité du groupe. |
| FOURNIER | SODIMOB |1|40| Mandat corporate, groupe et cycle de vie ; ambiguïtés de noms/identifiants à préserver. |
| EUREDEN GROUP | BERNARD AGRISERVICE |1|33| Détention explicite et ancien/nouvel exploitant, preuve forte. |
| CAPITAL SANTE INSTITUT | EI à unité légale partiellement diffusible |1|nonbackfillé| Identité masquée non reconstituée. |
| EUROVIA MANAGEMENT | CARRIERES KLEBER MOREAU |1|39| Mandat EUROVIA STONE/administrateurs ; correspondance partielle, pas identité MANAGEMENT. |
| CERGY PONTOISE RUE DES VERGERS | COMMUNAUTE D'AGGLO DE CERGY-PONTOISE |1|0| Données institutionnelles/sites nécessaires, pas annonces commerciales. |
| CAMPHES | NORMANDIE DISTRIBUTION SERVICE |1|27| Projet NDS/DAD, pas de mention CAMPHES vérifiée. |
| ADOS LOISIRS | COMMUNE DE L ETANG LA VILLE |1|0| Fonction/équipement municipal à documenter ailleurs. |
| CUISINES SCHMIDT HAUCONCOURT | SALVY |1|2| Activité cuisines et quartier Hauconcourt ; marque SCHMIDT non explicitée. |
| SYNDICAT MIXTE AGENCE LANDAISE POUR | COMMUNAUTE DE COMMUNES DES COTEAUX ET VALLEES DES LUYS |1|0| Libellés institutionnels/historique administratif nécessaires. |

Les dénominations différentes ne prouvent pas que les labels sont faux. Il
faut des preuves indépendantes d'occupant, de rôle ou d'historique pour
conclure. Elles expliquent en revanche pourquoi un embedding de noms seuls
ou d'un acte dépourvu du lien d'entrée ne suffit pas.

SMART GESTION a une activité principale 61.90Z dans le stock retenu : la
supposer pharmacie parce que le CRM mentionne Smartrx aurait été une erreur.
De même, l'EI du cas Groupama porte 68.20A : aucune identité d'agence d'assurance
ne doit être déduite de son nom. Les codes d'activité ne corrigent pas à eux
seuls les labels et ne deviennent pas des filtres du dénominateur.

## Décision pour la recherche initiale

Poursuivre la compilation de faits corporate typés et datés, puis leur
combinaison avec les preuves propres du site, avant le collecteur unique 100.
Le texte BODACC doit pouvoir découvrir un parent à partir d'une mention dans
un autre acte, pas uniquement enrichir des candidats déjà trouvés. Ne pas
fusionner les sociétés connectées ni diffuser une enseigne ou une adresse
de siège à tous les établissements.

Pour le canal dense, privilégier des passages courts autonomes comprenant les
parties et leur rôle, ou des vues compactes fidèles aux faits, plutôt qu'un
avis légal entier dominé par le capital et les clauses d'opposition. Le texte
brut reste conservé comme preuve et comme canal expérimental possible ; sa
valeur nationale doit être mesurée, pas supposée d'après ces cosinus.

Commencer par le contexte corporate déjà discriminant et la chaîne Savencia
à consolider, puis les activités/sites pour les libellés fonctionnels. Les
enseignes franchisées, institutions publiques et relations prestataire/client
nécessitent des sources complémentaires. Les sources choisies pour cet audit
par les SIREN humains ne doivent jamais être réemployées comme corpus positif
injecté dans une évaluation du retrieval.

## Artefacts techniques et intégrité

- V52 `v52/postseal_bodacc_fulltext_20261012_a` :24requêtes API,2,93s,
  session 99031 exit 0 collecté. cases SHA
  `535f830020b4f6dc169076a5d82c6305e7ff99110bcde921a573939a0919e1c0`.
- V53 `v53/postseal_bodacc_reverse_20261012_a` : sujets/prédécesseurs/mentions
  vérifiés et points d'entrée corporate,21,02 s, session 90352 exit 0 collecté.
  casesSHA `0fe2c9969fa64e85b0d45dcecaec43953ae6f86fe62e9b03221f58b2ae6d426f`.
- V54 `v54/postseal_bodacc_embedding_probe_20261012_a` : Gemma gelé1000,
  240 passages,18 atomes trop longs découpés intégralement,10,21 s d'encodage,
  session 43539 exit 0 collecté. cases SHA
  `2976bec07e17f15504b3c26fa018e8016bd177903a2c443caf84291bb720e517`.
  BanqueSHA `849896c390b0fbed6d9bb40c2265cced06c84f005c1bb2872dbfb76e98c9abfd`.
- V55 `v55/postseal_bodacc_name_text_20261012_a` :12recherches complètes,
  2,44 s, session 83121 exit 0 collecté. Les homonymes comme SCHMIDT ne sont
  pas des relations et les réponses stemmées/fuzzy sont revérifiées.
- Tous ces roots sont sous `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`.
  Les scripts, requêtes API, textes/extraits, dates et hashes sont conservés.
  Pipeline, recettes historiques, checkpoints et aval restent inchangés.
  Aucun gradient, unité de test, dépense externe, appel de LLM externe ou
  ouverturefold 0/1/test 4323 ; aucune revendication 99% ou nouveau score national.
