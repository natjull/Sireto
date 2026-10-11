# Audit du retrieval et direction recommandée

La priorité est une recherche initiale qui combine des preuves cohérentes et
des liens de vie des établissements avant de choisir ses 100 SIRET. Changer
d'encodeur ou réentraîner Gemma immédiatement n'est pas la prochaine action.
Le dernier score heuristique est rejeté. La référence mesurée reste
486 vérités exactes sur 526, soit 90,9823 % pondérés.

Deux problèmes indépendants doivent être résolus : perdre moins de bonnes
réponses déjà retrouvées et rendre retrouvables les 26 réponses absentes de
toutes les sorties essayées. Une meilleure fusion seule ne suffit pas.
L'audit ne démontre pas encore que 99,2 % exacts sont atteignables avec les
seuls champs CRM et les sources actuellement exploitées.

## Population et portée des preuves

Le diagnostic couvre les 526 requêtes du pilote probabiliste, 405 SIREN
annotés et tous les 2 554 848 établissements de ses 295 partitions communales.
Il rejoue les rangs SITE, identité conjointe et score natif à partir des scores
complets scellés, sans encodage, entraînement ni nouvelle sortie retrieval.
Chaque appartenance au top100 reproduit la génération initiale scellée.

Les 64 requêtes ratées par la référence ou par le nouvel essai ont un dossier
individuel : entrée CRM, label, rangs, scores, concurrents, noms et adresses
officiels, dates de succession et graines naturellement retrouvées.
L'inspection des sources produit 7 454 observations de nom et 8 490 d'adresse,
issues des autorités SIRENE, RNE et BODACC épinglées. Elle vérifie aussi les
successions du SIRET annoté lui-même, au lieu de compter les relations du
SIREN entier comme une preuve du site.

Ces sources constituent un dossier officiel dérivé et borné, pas toutes les
archives originales de toutes les administrations ni l'ensemble du Web.
Une absence dans ce dossier n'est pas une preuve d'absence partout.

Ce pilote est déjà consommé pour le développement. Les grandes communes
sont sous-échantillonnées et les poids corrigent cette allocation. Le brut
et le pondéré ne sont pas interchangeables ; les intervalles par communes
restent approximatifs. Aucun résultat ici n'est un OOF complet, une mesure
SLO, un test indépendant ou une certification de 99,2 %.
Folds0/1 et test4323 restent fermés. Tous les labels restent au dénominateur.

## Résultats et plafond des canaux actuels

| Mesure | Vérités exactes | Recall brut | Estimation pondérée |
| --- | ---: | ---: | ---: |
| Référence identité SITE lexical géo | 486 sur 526 | 92,40 % | 90,98 % |
| Nouveau maximum heuristique | 470 sur 526 | 89,35 % | 87,61 % |
| Présence dans au moins un des sept canaux | 500 sur 526 | 95,06 % | 94,13 % |

La dernière ligne est un ORACLE diagnostique : elle utilise les labels pour
savoir si un canal quelconque contient la vérité. Ce n'est pas une sortie
top100, un nouveau pool produit ni une performance atteinte. Elle borne
uniquement ce qu'une sélection pourrait récupérer à partir de ces sorties
finies, pas ce que pourraient atteindre d'autres scores ou représentations.

Les sept canaux sont SITE, identité conjointe, lexical national, ancienne géo,
nouvelle géo, famille et dernier score natif. Le dernier essai ne récupère
aucun des 26 cas absents des six anciens canaux. Il gagne 8 cas sur la référence
mais en perd 24. Les 40 ratés de la référence se décomposent donc en 14 cas
présents dans un autre canal et 26 absents partout. Le gain pondéré maximal
par simple sélection des sorties existantes serait de 3,15 points, laissant
encore environ 5,07 points jusqu'à l'objectif de 99,2 %.

## Pourquoi le dernier score échoue

Le maximum additionne une petite preuve géographique à la meilleure de
plusieurs valeurs qui ne mesurent pas la même chose. Une similarité de chaîne
de 0,90 n'est pas comparable directement à un cosinus Gemma de 0,60, et ni
l'une ni l'autre n'est une probabilité de bon établissement.

Sur 52 091 candidats émis, la branche adresse gagne 30 246 fois et la branche
nom lexical 21 130 fois. SITE gagne seulement 715 fois ; la branche identité
conjointe ne gagne jamais. Ainsi 98,63 % des candidats sont dominés par une
heuristique lexicale, alors que l'intention était d'enrichir Gemma.
Ce prototype n'a donc pas testé une bonne combinaison apprise de Gemma avec
toutes les preuves. Son échec ne démontre pas l'inutilité de ces preuves.

SOGEMA OREXAD illustre le mécanisme : le bon établissement est premier avec
SITE et avec l'identité conjointe, mais devient 413e avec le maximum.
Des entreprises aux noms peu concordants passent devant grâce à des voies
ressemblantes. LEROY SOMER passe de 9e SITE et 4e conjoint à 10 734e natif.
Ce ne sont pas de simples pertes autour du rang100.

Les ressemblances partielles sont aussi trop permissives. Sur CASINO CHALON,
CHALON COMMERCES obtient une similarité de nom de 0,855, sans être une preuve
d'identité CASINO. Les mots de localisation et les termes génériques doivent
être distingués des mots discriminants, avec prise en compte des contradictions.

Quatre pertes supplémentaires viennent d'une restriction de portée : trois
requêtes sans INSEE et une avec une partition communale vide. Le nouveau
prototype n'intégrait pas le canal lexical national qui permettait de les
retrouver. C'est une régression de couverture, pas un problème d'embedding.
Il faut conserver un accès national quand la géographie CRM est manquante ou
incohérente, sans imposer un filtre départemental ou communal éliminatoire.

## Noms historiques et pertes dans les représentations

Pour les 60 cas avec un document dans leur commune, le seul cas présentant
des noms officiels exploitables absents de ses vues est CASINO : PETIT CASINO
et LEADER PRICE EXPLOITATION. Le dossier en contient les observations mais la
projection des noms et les vues n'en disposent pas. D'autres noms CASINO sont
déjà présents et la cible figure déjà dans deux anciens canaux : ce manque
réel n'explique pas les 26 cas absents de tous les canaux.

Les autres écarts apparents concernent des marqueurs ND, une valeur devenue
vide après retrait d'un identifiant, ou les quatre documents hors partition.
Ils ne doivent pas être comptés comme des enseignes perdues à réparer.
La présence de ND répétés dans certains textes d'adresse reste à traiter
comme une absence de preuve, pas comme une vraie voie similaire.

Aucun des 26 cas encore manqués n'a un nom propre ou légal fortement
ressemblant au CRM selon le diagnostic lexical à seuil0,85. Deux ont un
meilleur nom sur un autre établissement du même SIREN. Ces ressemblances
sont des indices, pas une preuve d'identité. Aucun des 26 n'a une égalité
exacte numéro suffixe voie avec une adresse propre inspectée ; cette égalité
est stricte et ne détecte pas à elle seule les voies adjacentes ou synonymes.

NAOS constitue sept de ces 26 cas et neuf des 40 ratés de la référence.
Le lookup officiel national antérieur trouve 73 observations NAOS mais aucune
appartenant à AUBERT399226653. Ajouter NAOS comme alias AUBERT à partir de ces
labels serait précisément l'overfit interdit. Il faut une preuve indépendante
de la relation, ou déclarer honnêtement qu'elle n'est pas établie.

## Ce que les filiations apportent réellement

Parmi les 26 cas encore manqués, 17 possèdent au moins une succession directe
du SIRET annoté. Cinq ont un établissement directement lié déjà présent dans
au moins un des canaux essayés. Cela rend une recherche par relations plausible,
mais ne mesure aucun gain top100 : il faut encore retrouver et conserver le
bon bout de la relation sans perdre d'autres réponses.

L'inspection des noms et adresses propres de ces établissements liés précise
la portée : aucun cas n'obtient une similarité de nom de0,85 via son voisin
direct ; trois ont un voisin dont la voie ressemble à0,90 ou plus ; trois
ont un voisin situé à300m ou moins de la localisation CRM résolue.
Seuls quatre cas ont un voisin et une cible eux-mêmes à300m ou moins.
Une succession économique peut correspondre à un déménagement important.

Deux exemples constituent une vraie orientation de génération :

- Pour NAOS avenue de l'Avenir, le successeur AUBERT INTERNATIONAL est déjà
  retrouvé à environ0m de la position CRM. Le SIRET exact ancien est à491m
  et tous les canaux le manquent. Le lien de succession est daté du
  1er janvier2025. Une recherche de proximité seule favorise le successeur,
  alors qu'une vue relationnelle pourrait rendre l'ancien SIRET retrouvable.
- Pour ATOL rue de la Marne, un établissement lié est retrouvé à6m. Le SIRET
  annoté est à1 670m. Une autre succession conduit à un établissement situé
  près de199km de l'entrée CRM. Copier indistinctement toutes les adresses
  liées comme adresses propres fabriquerait des preuves fausses.

Les cinq cas avec une graine de succession directe sont les deux requêtes
SB ALLIANCE, ATOL, NAOS rue de Gardes et NAOS avenue de l'Avenir. Pour
SB ALLIANCE, les noms propres de la cible et du voisin restent FROMAGERIES
DU LEVEZOU : le graphe apporte un chemin, pas une explication du nom CRM.

La direction est donc d'indexer des VUES RELATIONNELLES typées, avec source,
dates, sens du lien et adresse du voisin explicitement contextuelle.
Une cession BODACC entre SIREN ne vaut ni identité, ni TUP, ni même site.
Une expansion aveugle de tous les voisins d'un groupe national serait nocive.
Le graphe doit agir avant la coupe finale, pas réparer un pool fixe déjà coupé.

## Ce que le fine tuning apprend et ce qui reste non démontré

Le modèle conservé est le checkpoint1000 du dernier entraînement, choisi sur
une loss contrastive de256 contrôles. L'entraînement final a atteint1340pas
mais n'a pas terminé sa première époque ; sa couverture de vues97,32 %
concerne l'état final, pas automatiquement le checkpoint1000 servi au test.

Les données comprennent15 035 requêtes TRAIN utilisables. Les négatifs
explicites conservés comptent35 570 lexicaux,14 079 denses d'autres parents,
1 964 autres sites prouvés du parent et609 autres sites denses du parent.
Soixante-six requêtes n'ont pas de négatif explicite, mais peuvent encore
recevoir des négatifs dans le batch.

Le minage dense utilise117 538 sites : établissements des parents TRAIN et
anciens négatifs approuvés. Il sélectionne un nouveau négatif statique par
requête, dans la commune, à partir de la première vue documentaire uniquement.
Cette banque n'est pas celle des communes complètes rencontrées en production.
C'est une limite concrète du protocole, pas une preuve qu'elle explique à elle
seule les ratés.

Le modèle reçoit du nom et de l'adresse textuels. XY, fiabilité BAN et preuves
de filiation ne sont pas entraînés comme variables numériques ou relations.
La perte75 % SITE et25 % identité n'apprend pas automatiquement que les
cosinus des deux tâches doivent être combinés avec les mêmes poids.
L'entraînement traite alternativement les vues comme positives, alors que
l'inférence maximise sur les vues ; l'intérêt d'une loss multivue agrégée
reste une hypothèse à tester, pas une amélioration acquise.

Enfin, l'effet isolé du fine-tuning n'a pas été établi sur ce pilote avec des
documents et une recette identiques. Les anciennes comparaisons changent aussi
les textes ou la représentation. Il faut conserver Gemma comme composant
mesuré, sans attribuer tout le recall à son fine-tuning ni financer un nouvel
encodage complet avant d'avoir testé l'hypothèse ciblée.

## Fiabilité de la vérité et généralisation

Les labels récents sont des SIRET saisis par les assistants commerciaux.
La certification de provenance assure cette origine et des contrôles
d'existence et de commune ; elle ne constitue pas une réadjudication indépendante
du nom et de l'adresse de chaque établissement. MATCH_EXACT décrit le label
admis, pas une démonstration documentaire nouvelle de son exactitude.

Sur les 26 cas manqués, la correspondance CONVERGENCE MEDIAS vers INOX,
SIP MAISON DES PROJETS vers ABAQUEPLAST et SB ALLIANCE vers FROMAGERIES DU
LEVEZOU exige une explication de métier ou une confirmation humaine.
Des locaux occupés, un payeur différent ou une relation absente des sources
peuvent aussi expliquer un écart. Aucun de ces labels n'est déclaré faux.
Les données inspectées ne prouvent aucune correction et aucune ligne n'est
retirée. Toute future correction humaine doit créer une version documentée,
avec ancienne métrique inchangée publiée à côté, jamais alléger le dénominateur.

Vingt requêtes du panel appartiennent à AUBERT ; quinze à VILLEURBANNE.
Le panel contient405 SIREN pour526 requêtes : les observations ne sont pas
indépendantes. Les neuf ratés NAOS ne représentent pas neuf problèmes de
généralisation indépendants. Les prochains choix doivent tenir face à des
SIREN, zones et familles de cas non utilisés pour les apprendre.

## Décision et prochaine expérience

Décision : PIVOT_GENERATION, sans promotion du dernier prototype.
Le traitement aval et le serving restent gelés. Aucun nouvel entraînement
long ou run national n'est lancé par cet audit.

La prochaine expérience doit tester une seule architecture : un retrieval
multichamp et relationnel, avec Gemma conservé et un score de preuves appris
hors SIREN vus, qui émet directement au plus100 SIRET.

1. Conserver le lexical national et la recherche sans dépendance obligatoire
   à un INSEE correct. Séparer les noms discriminants des mots de lieu et des
   termes génériques. Les preuves inconnues doivent rester inconnues.
2. Ajouter avant la collecte des vues de succession directe officielles,
   datées et typées. Les noms et adresses du voisin restent contexte du lien,
   jamais alias CRM appris ni preuves propres du site cible. Commencer par ce
   canal vérifiable plutôt que toute la closure BODACC d'un groupe.
3. Apprendre la comparaison des preuves SITE, parent, lexical, adresse, XY,
   fiabilité BAN et relation. Utiliser des interactions et des contradictions,
   pas un maximum entre scores bruts ni des quotas fixes par canal. Les scores
   denses d'apprentissage doivent être hors échantillon de leur encodeur,
   par cross-fitting SIREN ou un ensemble de calibration indépendant.
4. Le score doit être celui du moteur de recherche : scan natif des corpus
   complets pour le test, puis parcours d'index avec arrêt justifié par des
   bornes pour le serving. Pas de pool fixe500/2000/300 reclassé sous un autre
   nom. Les parcours candidats peuvent être des flux, pas un budget éliminatoire.
5. Mesurer séparément ajout des vues et changement de score, puis leur
   combinaison, sur une recette figée et le même corpus. Les526 servent
   seulement au diagnostic déjà consommé ; prévoir un contrôle non utilisé
   pour ajuster la recette, puis l'OOF2/3/4 exigé. Folds0/1 restent fermés.
6. Réexaminer les26 dossiers inexpliqués avec une preuve indépendante de
   marque, payeur, siège, site ou saisie humaine. Sans explication, annoncer
   le risque sur l'atteignabilité, sans apprendre leurs associations CRM.

Un nouveau fine-tuning n'intervient qu'après une attribution démontrant une
perte dense avec une preuve officielle pertinente disponible. Sa préparation
devra miner les distracteurs des communes TRAIN complètes, traiter les faux
négatifs équivalents/incertains et sélectionner les checkpoints sur retrieval
à corpus complet, pas seulement sur une loss de batch. Si une ablation ne
rapporte pas de gain hors échantillon, la voie est abandonnée.

## Appuis scientifiques et limites de transfert

[Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
établit une baseline de blocking lexical top-k solide et étudie le choix des
attributs et des tokenizers. Cela motive de conserver et qualifier le lexical,
pas de reprendre ses résultats comme garantie sur SIRENE.

[An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)
observe, sur ses benchmarks, l'intérêt d'une combinaison de scores apprise
face à RRF et la sensibilité de RRF à ses paramètres. Notre conclusion est
plus restreinte : RRF reste une baseline, pas une preuve qu'une fusion fixe
conserve toutes les vérités spécifiques à un canal.

[DPR](https://aclanthology.org/2020.emnlp-main.550/)
étudie des encodeurs contrastifs et des négatifs difficiles ; il ne garantit
pas le matching SIRET. [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/)
montre une autre manière de préserver les interactions fines à la recherche.
Cela justifie d'examiner les preuves par champ et par vue, pas de lancer
immédiatement un index ColBERT national ou de le déclarer nécessaire.

## Artefacts reproductibles

- Code de génération du dernier prototype : commit2b9c931d, historique figé.
- Résultat natif complet :
  `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v19/proof_engine_20261011_b/report.json`,
  SHA3a555a9aa2cb5efa04a528c936c6e023020eecaa1f984484d1ee24fd80c8dfaf.
- Audit complet des526 et dossiers64 :
  `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v20/deep_audit_20261011_c/summary.json`,
  SHAcb3da5109bc9310db3584163c46a4376af28632c44cbe2eaa1e0eb4069786ecc.
  `rows.json` contient les dossiers individuels et les rangs de tous les cas.
- Audit des voisins directs des26 :
  `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v20/direct_peer_audit_20261011_a/summary.json`,
  avec `cases.json` et les observations officielles originales conservées.
- Les namespaces d'audit incomplets a/b sont conservés, pas utilisés comme
  résultats. Les calculs d'audit réussis durent respectivement62,47 et73,57s ;
  aucun run d'encodage ou entraînement n'a été lancé.
