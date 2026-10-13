# Retrieval initial à 100 — audit des cinq étapes et décision scientifique

Mandat humain renouvelé le 13 octobre 2026. Ce document complète le
[bilan de la nuit](retrieval_entity_matching_night_20261013.md), sans rouvrir
ses budgets, ses automatisations ou les jeux fermés. Il distingue preuve
documentaire, préparation de source et amélioration mesurée.

## Conclusion provisoire : la priorité est l'information attribuée au bon sujet

La référence V99 retrouve **503/526 SIRET exacts, soit 95,6274 % brut et
94,7101 % pondéré**. Ce pilote de développement est consommé ; il ne certifie
ni la performance nationale ni les SLO. Huit gains sans perte par rapport
à V32 sont réels sur ce panel, mais ne suffisent pas à atteindre 99,2 %.

La direction proposée n'est pas de remplacer Gemma par un modèle plus gros.
Elle est de représenter séparément les **noms propres du site**, les **noms
propres de l'entreprise**, et les **relations vers des acteurs nommés**.
Une enseigne, un exploitant, un membre, un groupe et un fournisseur ne sont
pas interchangeables. Les preuves doivent intervenir avant l'unique
collecteur national de 100 SIRET, sans créer un pool fixe à reclasser.

La première ablation complète de ce mandat ajoute seulement le contexte
d'activité historique BODACC à V99, avec modèles et coefficients figés.
Son résultat sera inscrit ci-dessous après les 526 sorties et l'exit possédé.
Les autres nouvelles sources ne sont pas encore des gains de recall.

## 1. Diagnostic exhaustif des 23 ratés

Le ledger V111 conserve les 23 labels : **19 erreurs d'entreprise**, dont
deux cibles protégées, et **quatre erreurs de site dans la bonne entreprise**.
Ses catégories de travail sont 17 ponts d'identité non démontrés, quatre
sélections de site et deux limites d'information publique. « Non démontré »
ne signifie pas que toutes les sources possibles sont dépourvues de preuve.
Aucun label n'a été corrigé ou retiré du dénominateur.

Dans l'inventaire des noms propres SIRENE/RNE/BODACC, 20 des 21 ratés publics
n'ont aucun mot CRM commun avec les noms propres de la cible. Cet inventaire
n'inclut pas automatiquement les noms de sites FINESS/UAI. C'est une preuve
d'insuffisance des vues actuelles, pas une preuve d'irrécupérabilité.

| Requête CRM | Entreprise cible publique | Soupçon et vérification prioritaire |
|---|---|---|
| CONVERGENCE MEDIAS | INOX | Un autre acteur officiel correspond au nom et à l'adresse CRM. Vérifier succession/contrat/site ; pas de correction automatique. |
| BLANCHISSERIE | HOM'AGE SERVICES | Le nom décrit une activité explicitement publiée dans BODACC. Tester un contexte fonctionnel, pas un faux alias. |
| FOYER TY BREIZ | ENVOL LOIRE ATLANTIQUE | FINESS nomme le site retrouvé ; le label exact est le siège à une autre adresse. Alternative administrative conditionnelle. |
| SIP PLAINE COMMUNE MAISON DES PROJETS | ABAQUEPLAST | Nom et adresse divergent. Identifier le rôle réel du local/acteur avant d'interpréter l'acronyme. |
| OPHLM | DDFIP CHARENTE | Identité et adresse divergentes ; relation administrative ou label à documenter. |
| GEDIMAT | LEON BRACQ ET COMPAGNIE | Besoin d'un pont enseigne–exploitant daté et attribué, pas simplement d'un meilleur embedding. |
| EBSN | ASS ESPACE BERNADETTE SOUBIROUS | Sigle plausible à vérifier ; autre adresse/entrée du site non démontrée. RNA/JOAFE et documents de l'association. |
| SAS 5EME ETAGE | LE MUR DU SONGE | Adresse concordante, noms différents. Chercher l'identité publiée et l'historique, sans apprendre cette paire comme alias CRM. |
| GROUPAMA SUPPORTS ET SERVICES | Informations protégées | Garder le raté ; ne pas reconstituer les champs masqués par un dossier historique. |
| CEGEDIMCLOUD SMARTRX VILLEURBANNE | SMART GESTION | Nom composite et divergence d'adresse. Le rapport de groupe vérifié ne prouve pas le lien à SMART GESTION. |
| INFODESCA | PROSJET IRRIGARONNE | Bonne entreprise déjà trouvée, autre établissement fermé. Historique et intention site/administratif à séparer. |
| ATOL | MODERNE 66 | Pont réseau–exploitant et historique d'adresse à documenter. |
| TRESCAL | Dénomination légale absente dans le ledger | Ne pas confondre champ vide avec statut protégé ; vérifier la catégorie et les noms légalement diffusibles avant toute conclusion. |
| NAOS — Marseille | AUBERT FRANCE | Quatre requêtes, un seul groupe cible : dépendance forte. Pont documentaire déjà suggéré mais preuve/score insuffisants. |
| NAOS — Arçonnay | AUBERT FRANCE | Même mécanisme, autre site ; ne pas compter quatre succès indépendants. |
| NAOS — Mondeville | AUBERT FRANCE | Idem ; adresses commerciales et cadastrales potentiellement différentes. |
| NAOS — Villeneuve-d'Ascq | AUBERT FRANCE | Idem ; conserver l'adresse propre de chaque site. |
| CAPITAL SANTE INSTITUT | Informations protégées | Même protection que GROUPAMA ; pas de restauration de données masquées. |
| CERGY PONTOISE RUE DES VERGERS | CA CERGY-PONTOISE | Bonne entreprise, établissement différent ; adresse CRM distincte de celle du label. |
| CAMPHES | NORMANDIE DISTRIBUTION SERVICE | Marque/enseigne contre exploitant ; preuve et temporalité manquantes. |
| ADOS LOISIRS | COMMUNE DE L'ETANG-LA-VILLE | Site de loisirs retrouvé mais label au siège communal ; cas administratif, pas bonus siège universel. |
| CUISINES SCHMIDT HAUCONCOURT | SALVY | L'activité « cuisines » est publiée, mais ne prouve pas l'enseigne Schmidt ; vérifier exploitant du magasin. |
| SYNDICAT MIXTE AGENCE LANDAISE POUR | CC COTEAUX ET VALLEES DES LUYS | BANATIC publie une relation d'adhésion ALPI–communauté. Cela rend un label relationnel plausible, sans établir l'identité ou la situation au cutoff. |

Les divergences d'adresse ne sont pas toutes des erreurs : entrée adjacente,
ancien site, siège de facturation ou gestionnaire peuvent expliquer le CRM.
Pour conclure « label erroné », il faut une preuve officielle datée du rôle
du bon acteur, pas le seul désaccord avec SIRENE courant.

## 2. Ponts officiels : ce qui a réellement été préparé ou vérifié

### BODACC : activité attribuée, pas texte entier indifférencié

V112 parcourt les fichiers locaux 2008–2026, avec publication au plus tard
le 1er août 2026. Le parseur accepte un sujet personne morale et un
établissement uniques ; les annonces multi-acteurs/structures complexes
ne sont donc pas couvertes par cette première extraction. Le contrôle de
diffusion publique et de catégorie s'applique avant de retourner l'activité.

Résultat : **4 177 176 entreprises** portent un contexte, représentant
**12 641 678 références cumulées** pour les 526 requêtes. Ce nombre n'est ni
un pool par requête ni un nombre de candidats émis. L'activité est une union
historique de mots par sujet : ce n'est pas nécessairement une même phrase,
une seule annonce ou une activité actuelle.

V115 réutilise exactement ces références et scores, projetés sans perte en
shards par requête. Il ajoute un signal faible, plafonné à 0,5, modulé par
la rareté des mots et corroboré géographiquement, avant le seul collecteur.
Le poids de contexte existant, Gemma V6 best1000 et les calibrateurs ne
changent pas. Pas d'injection du positif, d'alias CRM ni de nouveau training.

### Groupes : preuve positive partielle, mais pas le pont attendu

Le [rapport Cegedim 2025 publié en 2026](https://www.cegedim.fr/Docs_Communiques/CEGEDIM_URD_2025_ASSEMBLAGE_FR_202604171735.pdf)
a été intégralement archivé et sa page 123 inspectée visuellement. La liste
des sociétés consolidées au 31 décembre 2025 donne explicitement Cegedim
Cloud et Smart RX, avec leurs identifiants. **SMART GESTION n'y figure pas**,
et la recherche dans le texte extrait n'a pas trouvé ce nom.

Cela ne prouve pas que SMART GESTION n'a jamais appartenu au groupe. Cela
interdit de prétendre que ce rapport a résolu ce raté. Il faut des archives
antérieures, une succession/TUP ou un autre document explicite. Une adresse
d'établissement principal à Boulogne ne prouve pas un site à Villeurbanne.

Autre hypothèse générique : `CEGEDIMCLOUD` peut empêcher un déclencheur exact
sur le nom publié `CEGEDIM CLOUD`. Une normalisation compacte de noms
officiels multi-mots est envisageable, à tester globalement avec contrôle
des collisions. Aucune table d'alias apprise des 23 labels ne doit la remplacer.

### Intercommunalité : une adhésion n'est pas une identité

[BANATIC](https://www.banatic.interieur.gouv.fr/intercommunalite/254003304-sm-agence-landaise-pour-l-informatique--alpi)
publie la communauté Coteaux et Vallées des Luys parmi les groupements
adhérents à ALPI. La fiche observée est mise à jour le **21 septembre 2026** :
elle est utile au diagnostic, pas admissible dans le benchmark arrêté au
1er août sans archive conforme.

L'API publique distingue les groupements et leurs membres ; son endpoint
« membres » observé ne fournit que les communes, pas toute l'adhésion
inter-groupements publiée sur la fiche. Une tentative de pagination globale
a détecté des doublons et a échoué explicitement. Nous n'avons donc pas un
catalogue exhaustif validé. Préférer un export producteur daté à un scraping
partiel pris à tort pour un graphe complet.

## 3. Site physique et destinataire administratif

Un bonus global au siège a déjà été testé et rejeté : V95 produisait cinq
gains mais quatre pertes. Il augmente la présence des sièges sans résoudre
l'intention de la requête. La bonne stratégie est une **preuve conditionnelle**.

V118 a inspecté les 410 603 références sectorielles disponibles. Pour chaque
site, il exige le nom informatif, l'adresse propre de la source et l'adresse
propre SIRENE concordant avec le CRM. Un gestionnaire différent du préfixe
SIREN du SIRET ne devient pas automatiquement l'exploitant du site.

Cette recette stricte qualifie seulement **deux requêtes**, avec deux
alternatives de siège : TY BREIZ et PHARMACIE VALLOGNES. TY BREIZ est un raté
V99 ; l'autre n'en est pas un. Sur ce panel et avec cette recette, le gain
exact maximal sur les ratés V99 serait donc **un seul cas**, avant toute
perte éventuelle. Ce n'est pas encore une amélioration mesurée.

Le site garde son nom, son adresse et ses XY. Le siège peut recevoir une
preuve de relation administrative, jamais le nom/adresse/point du site comme
s'il s'agissait de ses propres caractéristiques. Pas de quota siège forcé.
Les quatre cas de sélection de site requièrent ensuite une couverture
publique plus large que FINESS : services municipaux et intercommunaux,
structures associatives et historique des établissements.

## 4. Fine-tuning : changer la supervision avant de dépenser un nouveau training

Le contrôle V107 ne trouve aucun chevauchement de positif ou de négatif
supervisé V1/V6 avec les 526 requêtes et leurs 405 SIREN. Cela protège cette
comparaison contre cette fuite précise, mais le panel reste consommé pour
choisir des recettes. Il n'est pas un nouveau test aveugle.

Le diagnostic révèle une autre faiblesse : **1 148/15 035 lignes TRAIN**
partagent un texte identité ambigu, avec 138 chaînes exactes et jusqu'à
140 entreprises cibles pour une chaîne. Ajouter CP+voie réduit fortement
cette ambiguïté. Les collisions de faux négatifs concernent surtout la
vue identité ; aucune collision SITE identique n'a été observée dans le
contrôle ciblé. Un modèle ne peut déduire une entreprise unique d'un nom
insuffisamment informatif.

Recette proposée pour le prochain entraînement, non encore lancée :

1. Construire des vues séparées SITE, ENTREPRISE et CONTEXTE TYPÉ. Conserver
   l'adresse dans SITE ; ne pas noyer les noms distinctifs dans des annonces
   administratives ou un dossier de tous les établissements.
2. Attribuer chaque segment officiel à son sujet, à son rôle et à sa date.
   Un membre/acheteur/groupe n'est pas un alias positif d'identité.
3. Masquer les faux négatifs effectivement indiscernables dans une vue,
   ou employer des positifs multiples uniquement quand l'équivalence
   sémantique de cette vue est justifiée. Ne pas convertir tous les SIRET
   d'un SIREN en positifs exacts.
4. Miner des négatifs difficiles réels : même adresse, même enseigne avec
   autre exploitant, autre site de la même entreprise, successeur/prédécesseur
   et voisins géographiques. Seulement sur les groupes TRAIN.
5. Mesurer couverture et troncature des champs avant de choisir une longueur.
   Préférer plusieurs vues courtes attribuées à un texte arbitrairement long.
6. Comparer brut et fine-tuné sur les mêmes vues, mêmes canaux et même corpus.
   Sans cette ablation, un gain de données ne peut être attribué au training.

**Décision : ne pas lancer aujourd'hui un nouveau fine-tuning avant de
connaître l'apport complet du contexte et de disposer des ponts manquants.**
Ce mandat n'a encore produit ni nouveaux gradients ni nouvelle preuve de
performance du bi-encodeur. Les dirigeants physiques ne sont pas utilisés :
un nom partagé ne démontre ni identité de société ni identité de site.

## 5. Généralisation : protocole de confirmation à préparer

Les 526 requêtes proviennent de 295 communes et 405 SIREN ; le poids des
communes denses est important. Publier systématiquement brut et pondéré,
les segments et les gains/pertes par entreprise, pas seulement par ligne.
Les quatre NAOS ne sont pas quatre confirmations indépendantes.

La confirmation doit séparer deux questions : trouver les acteurs dans
une zone nouvelle et comprendre un type d'acteur nouveau. Préparer des
splits groupés SIREN avec zones géographiques disjointes et une stratification
public/EI/multisite/enseignes/fermés/diffusion protégée. Le split ne doit pas
supprimer les dossiers difficiles ou les cibles absentes du catalogue.

Pour un OOF complet, repartir d'un modèle de base ou officiel-only : **V6
ne peut initialiser un fold contenant des entreprises de son ancien TRAIN**.
Chaque recette/poids est appris hors du fold mesuré ; les vues CRM et les
alias induits des labels ne traversent pas les folds. Le corpus de recherche
reste national complet, avec les distracteurs des grandes communes.

Gel nécessaire avant confirmation : sources et cutoff, transformations,
poids, modèle, calibrateurs, règle d'émission à 100, sidecar label-side,
hashes et protocole de coût. Les canaux se mesurent seuls et en combinaison,
avec pertes appariées et intervalles tenant compte des groupes/communes.
Wilson sur les lignes est publié, sans le confondre avec une garantie
tenant compte de toutes les dépendances.

Fold 0 humain contaminé, fold 1 humain, test 4323 et contrôle indépendant
962 restent fermés. Aucune ouverture one-shot n'est inférée de ce mandat.
Les modèles aval et serving restent gelés. Les critères finaux comprennent
recall exact ≥99,2 %, Wilson bas ≥99,0 %, chaque fold ≥98,8 %, ainsi que
p95≤1 s et p99≤2 s mesurés selon le contrat. Aujourd'hui aucun READY.

## Open data : priorités et contre-indications

L'inventaire V113 a archivé les résultats paginés de huit recherches du
catalogue public. Un résultat de catalogue ne prouve ni la couverture, ni
la présence de SIRET, ni l'admissibilité temporelle du fichier.

| Source | Valeur attendue | État vérifié / prochaine décision |
|---|---|---|
| SIRENE historique + successions | Noms/adresses anciens et continuité du site | Déjà partiellement valorisés ; vérifier la couverture brute→index et typer succession plutôt qu'identité intemporelle. |
| BODACC activité et événements | Fonction du site, enseigne, transfert/exploitant | Activité whole-source préparée et ablation V115 ; événements multi-acteurs demandent une attribution stricte, pas du texte entier. |
| FINESS/UAI/DINUM | Nom réel du lieu et exploitant/gestionnaire | Source locale réelle, SIRET de site sous-valorisés ; V118 strict ne couvre que deux requêtes. Vérifier dates et autorité des liens par source. |
| BANATIC et annuaire administratif | Groupements, adhérents, services/sites publics | Adhésion ALPI vérifiée post-cutoff ; export complet daté et relations inter-groupements encore nécessaires. |
| Rapports de groupes publiés | Liens nommés, filiales et identifiants | Source riche mais hétérogène ; Cegedim contrôlé, pont à SMART GESTION non établi. Extraire des cohortes entières, pas seulement une cible ratée. |
| RNA/JOAFE | Sigles, titres, objet et historique associatif | Priorité pour EBSN/loisirs ; archive temporelle et pont RNA→SIREN à vérifier. Ne pas admettre automatiquement le miroir d'octobre. |
| INPI marques | Enseigne/marque→titulaire, dates de transfert | Piste pour GEDIMAT/ATOL/SCHMIDT/CAMPHES ; licence, accès et pont exploitant local doivent être vérifiés. Titulaire de marque ≠ franchisé. |
| Données essentielles de commande publique | Relations acheteur–titulaire, contexte fonctionnel | Sept fichiers : 219 980 marchés, 151 136 paires publiques, **aucun nom d'acteur exploitable**. Déprioriser comme source d'alias. |
| DATA ES / équipements sportifs | Noms de lieux municipaux et gestionnaires | Fichier local août 19 post-cutoff ; intérêt sectoriel, lien d'exploitation et archive à établir. Aucun gain mesuré. |
| Géolocalisation INSEE + BAN | Proximité, voies adjacentes, adresses incertaines | Conserver la qualité BAN et les points de chaque site. Ne pas attribuer au siège les coordonnées d'un établissement ; XY ne résout pas une marque sans lien. |

Les [DECP consolidées](https://www.data.gouv.fr/datasets/donnees-essentielles-de-la-commande-publique-fichiers-consolides)
sont une vraie source officielle, mais notre constat de zéro nom concerne
uniquement les sept versions analysées, pas toute la commande publique.
L'objet de marché peut décrire le besoin de l'acheteur, pas l'identité du
titulaire ; il ne devient pas son nom propre.

## Ordre d'action et critères d'arrêt

1. Terminer et comparer V115 à V99 sur les 526 sorties complètes. Si le
   contexte apporte surtout des concurrents ou des pertes, ne pas élargir
   son poids pour sauver des exemples connus.
2. Prioriser **les ponts publiés qui ajoutent réellement une information** :
   enseigne–exploitant et relation publique/groupement. Chaque extraction
   doit publier couverture, subjectivité et temporalité, puis une ablation
   complète à modèle identique.
3. Tester séparément l'alternative administrative V118, avec son potentiel
   borné à un raté V99, ou élargir d'abord une source publique pertinente.
   Aucun bonus siège global ; pas de promesse que cette piste ferme le gap.
4. Refaire la supervision et l'ablation brut/fine-tuné lorsque les documents
   apportent une preuve suffisante. Le training ne remplace pas les sources.
5. Une fois une recette stabilisée, confirmation OOF propre et SLO ; pas
   d'ouverture prématurée d'un jeu fermé ni de sélection sur ce jeu.

### Résultat du run complet V115

En cours à la rédaction initiale : aucune métrique partielle publiée.

## Traçabilité

- Ledger et activité : `51d8cbaa` ; catalogue sources : `44dbb996`.
- Première ablation interrompue : `a8fbff70`, correction de création du
  répertoire `d7a879f2` ; reprise sans perte V115 : `e015f25a`.
- DECP : `f93772d4`, parseur réel `ece44078`, correction syntaxe `62c2a141`.
- Sites sectoriels : `5976c505` ; API BANATIC : `6975d167`.
- Archive du rapport Cegedim : `f7b71fa1`.

Artefacts immuables sous `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`, roots
V111 à V120 datés `20261013_a`. Chaque source possède son manifeste/hashes
quand la génération est complète. L'échec de pagination BANATIC n'est pas
une génération complète. Aucun unit test, cloud, push ou nettoyage utilisateur.
